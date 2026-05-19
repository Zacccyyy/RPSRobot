"""
voice_control.py — Voice input controller for the RPS gesture recogniser.

This file sits between the microphone and the rest of the game.  It runs a
background thread that continuously listens to the mic, passes the audio through
the Vosk offline speech recogniser, and converts recognised words into typed
event dicts that the main game loop can poll each frame.

Two event types are produced:
  - "beat"  : a countdown word was heard  (ready / one / two / three)
  - "throw" : a gesture word was heard    (Rock / Paper / Scissors / …)
  - "nav"   : a navigation word was heard (up / down / select / quit / …)

Why Vosk?  It runs 100 % offline (no internet needed), loads fast, and the
small model is only ~50 MB.  We lock it to a closed vocabulary (grammar mode)
so it only searches ~100 words instead of 50,000+, which gives lower latency
and higher accuracy for command-word use.

INSTALL
-------
    pip install vosk sounddevice

Download the small English model (~50 MB) from:
    https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip

Unzip it so the folder exists at one of:
    ./vosk-model-small-en-us-0.15/          (project root, recommended)
    ~/Desktop/CapStone/vosk-model-small-en-us-0.15/
    ~/Downloads/vosk-model-small-en-us-0.15/

Usage
-----
    vc = VoiceController()
    ok = vc.start()                   # returns True if successfully started
    if not ok:
        print(vc.get_error())

    # each frame:
    for event in vc.drain_events():
        # event = {"type": "beat",  "word":    "ready" | "one" | "two" | "three"}
        # event = {"type": "throw", "gesture": "Rock"  | "Paper" | "Scissors"}
        # event = {"type": "nav",   "action":  "up" | "down" | "select" | ...}
        ...

    vc.stop()
"""

import json
import numpy as _np
import os
import queue
import threading

# ---------------------------------------------------------------------------
# Optional dependencies — wrapped so the app still starts if not installed.
# ---------------------------------------------------------------------------
# sounddevice gives us access to the microphone.
try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False

# vosk is the offline speech-recognition engine.
try:
    import vosk
    _VOSK_AVAILABLE = True
except ImportError:
    _VOSK_AVAILABLE = False

# Public flag — True only when BOTH packages loaded successfully.
VOSK_AVAILABLE = _SD_AVAILABLE and _VOSK_AVAILABLE

# ---------------------------------------------------------------------------
# Recognition vocabulary
# ---------------------------------------------------------------------------
# We give Vosk a closed grammar (a finite list of words it's allowed to hear).
# This is dramatically faster and more accurate than open-vocab recognition for
# a command-word application like this one.
#
# For every canonical word we care about, we also include every plausible
# mishearing or accent variant.  The _BEAT_CANONICAL dict then maps each
# variant back to its canonical form so the game logic only sees the clean
# version (e.g. "tree" → "three").

# -- Countdown words --
# Each entry covers the canonical word plus its most common mishearings.
_BEAT_WORDS = frozenset({
    # Canonical countdown words
    "ready", "one", "two", "three",
    # "ready" variants — Vosk sometimes hears "steady", "freddy", etc.
    "steady", "freddy", "eddie", "reddish", "already", "betty", "reddy",
    # "one" variants — Vosk often mishears this as short words like "on", "in"
    "won", "on", "and", "wan", "run", "gun", "none", "juan", "in",
    # "two" variants — easily confused with short similar-sounding words
    "to", "too", "do", "the", "a", "who", "new", "tu", "tew",
    # "three" variants — hardest to recognise due to wide accent variation
    "tree", "free", "freed", "sri", "through", "re", "street",
    "throw", "threat", "thresh", "thrice", "drei",
})

# Maps every recognised variant to its canonical countdown word.
# When the recogniser hears "tree", we want to emit "three" to the game.
_BEAT_CANONICAL = {
    # ready
    "ready": "ready", "steady": "ready", "freddy": "ready",
    "eddie": "ready", "reddish": "ready", "already": "ready",
    "betty": "ready", "reddy": "ready",
    # one
    "one": "one", "won": "one", "on": "one", "and": "one",
    "wan": "one", "run": "one", "gun": "one", "none": "one",
    "juan": "one", "in": "one",
    # two
    "two": "two", "to": "two", "too": "two", "do": "two",
    "the": "two", "a": "two", "who": "two", "new": "two",
    "tu": "two", "tew": "two",
    # three
    "three": "three", "tree": "three", "free": "three", "freed": "three",
    "sri": "three", "through": "three", "re": "three", "street": "three",
    "throw": "three", "threat": "three", "thresh": "three",
    "thrice": "three", "drei": "three",
}

# Maps every heard word that could be a gesture to the canonical gesture name.
# Includes phonetically similar words that Vosk might return instead of the
# real gesture word (e.g. "lock" → "Rock", "sisters" → "Scissors").
_THROW_WORDS = {
    # -- Rock --
    "rock":       "Rock",
    "lock":       "Rock",
    "block":      "Rock",
    "knock":      "Rock",
    "walk":       "Rock",
    "talk":       "Rock",
    "dock":       "Rock",
    "roc":        "Rock",
    "rok":        "Rock",
    # -- Paper --
    "paper":      "Paper",
    "favor":      "Paper",
    "taper":      "Paper",
    "pacer":      "Paper",
    "vapor":      "Paper",
    "later":      "Paper",
    "labor":      "Paper",
    "piper":      "Paper",
    "proper":     "Paper",
    "pepper":     "Paper",
    # -- Scissors --
    "scissors":   "Scissors",
    "sisters":    "Scissors",
    "seizures":   "Scissors",
    "cesars":     "Scissors",
    "figures":    "Scissors",
    "sizzle":     "Scissors",
    "scissor":    "Scissors",
    "cissors":    "Scissors",
    "scissored":  "Scissors",
    # -- RPSLS extras (Rock Paper Scissors Lizard Spock) --
    "lizard":     "Lizard",
    "wizard":     "Lizard",   # common mishearing
    "blizzard":   "Lizard",
    "spock":      "Spock",
    "spot":       "Spock",    # common mishearing
    "stock":      "Spock",
    "spark":      "Spock",
}

# Maps navigation / menu words to the canonical action the game loop expects.
# Multiple words can map to the same action (e.g. "yes", "ok", "enter" → "select").
_NAV_WORDS = {
    # -- Directional navigation --
    "up":           "up",
    "higher":       "up",
    "above":        "up",
    "previous":     "up",
    "prev":         "up",
    "down":         "down",
    "lower":        "down",
    "below":        "down",
    "next":         "down",    # "next item" in a list = scroll down
    "town":         "down",    # common Vosk mishearing of "down"
    "left":         "left",
    "right":        "right",

    # -- Confirm / select --
    "select":       "select",
    "yes":          "select",
    "yep":          "select",
    "yeah":         "select",
    "yah":          "select",
    "ok":           "select",
    "okay":         "select",
    "enter":        "select",
    "confirm":      "select",
    "go":           "select",
    "choose":       "select",
    "open":         "select",

    # -- Cancel / back --
    "back":         "back",
    "no":           "back",
    "nope":         "back",
    "nah":          "back",
    "cancel":       "back",
    "escape":       "back",
    "menu":         "back",
    "return":       "back",

    # -- Quit --
    "quit":         "quit",
    "exit":         "quit",
    "close":        "quit",

    # -- Restart / play again --
    "restart":      "restart",
    "again":        "restart",
    "replay":       "restart",
    "retry":        "restart",
    "repeat":       "restart",
    "redo":         "restart",

    # -- Start / begin (modes with an explicit start prompt) --
    "start":        "start",
    "begin":        "start",
    "play":         "start",
    "launch":       "start",

    # -- Next / skip (tutorial and multi-step flows) --
    "skip":         "next",
    "forward":      "next",
    "continue":     "next",
    "advance":      "next",

    # -- Toggle commentary --
    "commentary":   "commentary",
    "comment":      "commentary",
    "commentate":   "commentary",
    "narrate":      "commentary",

    # -- Direct main-menu shortcuts (say the mode name to jump straight there) --
    "cheat":        "cheat",
    "cheats":       "cheat",
    "fair":         "fair",
    "fairplay":     "fair",
    "challenge":    "challenge",
    "clone":        "clone",
    "clones":       "clone",
    "stats":        "stats",
    "statistics":   "stats",
    "scores":       "stats",
    "tutorial":     "tutorial",
    "help":         "tutorial",
    "settings":     "settings",
    "options":      "settings",
    "config":       "settings",
    "features":     "features",
    "toggles":      "features",
    "simulations":  "simulations",
    "simulate":     "simulations",
    "lab":          "simulations",

    # -- Direct game shortcuts (from anywhere) --
    "snake":        "snake",
    "squid":        "squid",
    "simon":        "simon",
    "bluff":        "bluff",
    "reflex":       "reflex",
    "rehab":        "rehab",
    "trainer":      "rehab",
    "race":         "race",
    "prediction":   "race",
    "rpsls":        "rpsls",
    "spock":        "rpsls",   # saying "Spock" on the menu means go to RPSLS mode
    "games":        "gamemodes",
    "modes":        "gamemodes",
}

# ---------------------------------------------------------------------------
# Model search paths
# ---------------------------------------------------------------------------
# We support two Vosk models: the standard US-English one and a smaller
# Indian-English model that works better for Australian / non-American accents.
_DEFAULT_MODEL_NAME = "vosk-model-small-en-us-0.15"
_INDIAN_MODEL_NAME  = "vosk-model-small-en-in-0.4"

# Ordered list of directories to search when the user hasn't given a path.
_DEFAULT_MODEL_SEARCH_PATHS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), _DEFAULT_MODEL_NAME),
    os.path.expanduser(f"~/Desktop/CapStone/{_DEFAULT_MODEL_NAME}"),
    os.path.expanduser(f"~/Downloads/{_DEFAULT_MODEL_NAME}"),
    os.path.expanduser(f"~/{_DEFAULT_MODEL_NAME}"),
]

_INDIAN_MODEL_SEARCH_PATHS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), _INDIAN_MODEL_NAME),
    os.path.expanduser(f"~/Desktop/CapStone/{_INDIAN_MODEL_NAME}"),
    os.path.expanduser(f"~/Downloads/{_INDIAN_MODEL_NAME}"),
    os.path.expanduser(f"~/{_INDIAN_MODEL_NAME}"),
]


def _find_model_path(override=None, prefer_indian=False):
    """
    Locate the Vosk model directory and return its path, or None if not found.

    If `override` is given and points to an existing directory, that is used
    immediately without any searching.  Otherwise we walk the default search
    paths in order, optionally trying the Indian-English model first.
    """
    # If the caller already knows where the model is, use it directly.
    if override and os.path.isdir(override):
        return override

    # When the Indian model is preferred (better for non-US accents), check
    # those paths first before falling through to the US model.
    if prefer_indian:
        for p in _INDIAN_MODEL_SEARCH_PATHS:
            if os.path.isdir(p):
                return p

    # Try the standard US-English model in the usual locations.
    for p in _DEFAULT_MODEL_SEARCH_PATHS:
        if os.path.isdir(p):
            return p

    # If the US model wasn't found and we haven't tried Indian yet, try it now
    # as a fallback — better than nothing.
    if not prefer_indian:
        for p in _INDIAN_MODEL_SEARCH_PATHS:
            if os.path.isdir(p):
                return p

    # Nothing found — caller must display an install message.
    return None


# ---------------------------------------------------------------------------
# VoiceController
# ---------------------------------------------------------------------------
class VoiceController:
    """
    Background-thread voice listener.

    Recognises a small fixed vocabulary and posts typed event dicts to an
    internal queue.  The main game loop calls drain_events() each frame to
    consume those events without blocking.  All public methods are thread-safe.
    """

    # Vosk's small model requires 16 kHz mono input — do not change this.
    SAMPLE_RATE = 16000   # Hz
    # How many audio frames are delivered per callback (~50 ms at 16 kHz).
    # Smaller block = lower recognition latency.  Was 4000 (250 ms) originally.
    BLOCK_SIZE  = 800

    def __init__(self, model_path=None, verbose=False, prefer_indian=False):
        """
        Set up the controller in a stopped state.  Nothing is allocated until
        start() is called.

        Parameters
        ----------
        model_path : str or None
            Path to an unpacked Vosk model directory.  If None, the controller
            searches the default locations listed in _DEFAULT_MODEL_SEARCH_PATHS.
        verbose : bool
            If True, print each recognised word to stdout as it is heard.
        prefer_indian : bool
            If True, prefer vosk-model-small-en-in-0.4 over the US model.
            Better for Australian and non-American accents.
        """
        self._model_path    = model_path
        self._verbose       = verbose
        self._prefer_indian = prefer_indian
        self._event_queue   = queue.Queue()  # thread-safe event buffer
        self._thread        = None           # the background listening thread
        self._stop_event    = threading.Event()  # set this to ask the thread to stop
        self._running       = False
        self._error         = None           # last error string (for UI display)
        self._last_word     = ""             # most recently recognised word
        self._mic_level     = 0.0            # RMS amplitude of last audio block (0–1)
        self._lock          = threading.Lock()  # guards _last_word and _mic_level

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """
        Load the Vosk model and start the background listening thread.

        Returns True on success.  Returns False (and stores a human-readable
        message in self._error) if a required package is missing or the model
        directory cannot be found.
        """
        # Already running — nothing to do.
        if self._running:
            return True

        # Check that vosk and sounddevice are both installed before proceeding.
        if not VOSK_AVAILABLE:
            missing = []
            if not _VOSK_AVAILABLE:
                missing.append("vosk")
            if not _SD_AVAILABLE:
                missing.append("sounddevice")
            self._error = (
                f"Missing package(s): {', '.join(missing)}.\n"
                "Install with:  pip install vosk sounddevice\n"
                "Then download the model from:\n"
                "  https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip\n"
                f"Unzip to: {_DEFAULT_MODEL_SEARCH_PATHS[0]}"
            )
            return False

        # Find the model directory on disk.
        path = _find_model_path(self._model_path, prefer_indian=self._prefer_indian)
        if path is None:
            self._error = (
                "Vosk model not found.  Download the small English model:\n"
                "  https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip\n"
                f"Unzip to:  {_DEFAULT_MODEL_SEARCH_PATHS[0]}"
            )
            return False

        # Load the model on the calling thread — it only takes ~0.3 s and it's
        # simpler to catch errors here than inside the background thread.
        try:
            vosk.SetLogLevel(-1)          # silence Vosk's own verbose stdout
            model = vosk.Model(path)
        except Exception as exc:
            self._error = f"Failed to load Vosk model at '{path}': {exc}"
            return False

        # Clear any leftover stop signal from a previous run, mark as running,
        # then launch the background thread as a daemon so it dies automatically
        # when the main process exits.
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._listen_loop,
            args=(model,),
            name="VoiceControlThread",
            daemon=True,
        )
        self._thread.start()
        print(f"[Voice] Started — model: {path}")
        return True

    def stop(self):
        """
        Signal the background thread to stop and wait for it to finish.

        After this returns, no more events will be added to the queue.
        """
        if not self._running:
            return
        # Set the stop event so the listen loop exits on its next iteration.
        self._stop_event.set()
        self._running = False
        # Give the thread up to 2 seconds to clean up the audio stream.
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        print("[Voice] Stopped")

    def drain_events(self):
        """
        Return all queued voice events and clear the queue.

        Call this once per frame.  It never blocks — if nothing has been heard
        it just returns an empty list.

        Each event is a dict:
          {"type": "beat",  "word":    <str>}    — a countdown word was heard
          {"type": "throw", "gesture": <str>}    — a throw gesture was heard
          {"type": "nav",   "action":  <str>}    — a navigation word was heard
        """
        events = []
        # Pull everything off the queue without waiting.
        while True:
            try:
                events.append(self._event_queue.get_nowait())
            except queue.Empty:
                break  # nothing left to drain
        return events

    def is_running(self):
        """Return True if the background listening thread is currently active."""
        return self._running

    def get_error(self):
        """Return the last error string, or None if no error has occurred."""
        return self._error

    def get_last_word(self):
        """Return the most recently recognised canonical word (used for UI display)."""
        with self._lock:
            return self._last_word

    def get_mic_level(self):
        """
        Return the normalised RMS microphone level (0.0 – 1.0) from the last
        audio block.  Useful for drawing a live waveform indicator in the UI.
        """
        with self._lock:
            return self._mic_level

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dispatch_word(self, word):
        """
        Classify a single recognised word and push the right event onto the queue.

        This is called from the audio callback (a background thread), so we use
        self._lock when touching shared state.
        """
        word = word.strip().lower()

        # Ignore empty strings and Vosk's unknown-word token.
        if not word or word == "[unk]":
            return

        # Map variant spellings/mishearings to the canonical beat word.
        # e.g. "tree" → "three", "steady" → "ready"
        canonical = _BEAT_CANONICAL.get(word, word)

        # Store for UI display (thread-safe write).
        with self._lock:
            self._last_word = canonical

        # Decide which event type to fire based on which vocabulary the word
        # belongs to.  Beat words take priority, then throw, then nav.
        if canonical in ("ready", "one", "two", "three"):
            # Only dispatch canonical beat words — variants were already mapped above.
            self._event_queue.put({"type": "beat", "word": canonical})
            if self._verbose:
                print(f"[Voice] Beat: {canonical} (heard: {word})")

        elif word in _THROW_WORDS:
            # The raw word (not the canonical beat version) is checked here
            # because throw words don't go through _BEAT_CANONICAL.
            gesture = _THROW_WORDS[word]
            self._event_queue.put({"type": "throw", "gesture": gesture})
            if self._verbose:
                print(f"[Voice] Throw: {gesture} (heard: {word})")

        elif word in _NAV_WORDS:
            self._event_queue.put({"type": "nav", "action": _NAV_WORDS[word]})
            if self._verbose:
                print(f"[Voice] Nav: {_NAV_WORDS[word]}")

    def _listen_loop(self, model):
        """
        Background thread body: open the microphone, feed audio to Vosk, and
        call _dispatch_word whenever a word is recognised.

        We use grammar-constrained recognition — we tell Vosk exactly which
        words it's allowed to hear.  This means the decoder searches ~100 words
        instead of 50,000+, giving lower latency and higher accuracy.

        We also act on partial results (mid-utterance) so there's no need to
        wait for the user to stop speaking before the game reacts.
        """
        # Build the closed vocabulary: every variant word from all three tables
        # plus "[unk]" which is Vosk's required catch-all for out-of-vocabulary sound.
        all_vocab = (
            list(_BEAT_WORDS)
            + list(_THROW_WORDS.keys())
            + list(_NAV_WORDS.keys())
            + ["[unk]"]
        )
        grammar_json = json.dumps(all_vocab)

        # Create the recogniser with the grammar.  Fall back to open-vocab if
        # this version of Vosk doesn't support the grammar parameter.
        try:
            rec = vosk.KaldiRecognizer(model, self.SAMPLE_RATE, grammar_json)
        except Exception:
            rec = vosk.KaldiRecognizer(model, self.SAMPLE_RATE)

        # Track the last partial result text so we don't fire the same word
        # twice from two consecutive identical partial frames.
        last_partial = ""

        def _audio_callback(indata, frames, time_info, status):
            """
            Called by sounddevice ~every 50 ms with a fresh block of audio.
            Runs on a real-time audio thread — must return quickly, no blocking.
            """
            nonlocal last_partial

            # If someone called stop(), abort the audio stream cleanly.
            if self._stop_event.is_set():
                raise sd.CallbackAbort()

            # Log any audio driver warnings (overruns, underruns, etc.).
            if status:
                print(f"[Voice] Audio status: {status}")

            data = bytes(indata)

            # Compute RMS amplitude and scale it to 0–1 for the UI level meter.
            # We multiply by 6 to make quiet speech visible; clamp at 1.0.
            samples = _np.frombuffer(data, dtype=_np.int16).astype(_np.float32)
            rms = float(_np.sqrt(_np.mean(samples ** 2))) / 32768.0
            with self._lock:
                self._mic_level = min(1.0, rms * 6.0)

            # Feed the audio block to the Vosk decoder.
            if rec.AcceptWaveform(data):
                # -- Final result: the utterance is complete --
                # Vosk has decided the person stopped talking.  Parse the JSON
                # result and dispatch the first recognised vocabulary word.
                try:
                    result = json.loads(rec.Result())
                    text = result.get("text", "").strip().lower()
                    last_partial = ""  # reset so the same word can fire next time
                    if text:
                        # Walk the words left-to-right and fire on the first hit.
                        for w in text.split():
                            if w in _BEAT_WORDS or w in _THROW_WORDS or w in _NAV_WORDS:
                                with self._lock:
                                    already_sent = (self._last_word == w)
                                if not already_sent:
                                    self._dispatch_word(w)
                                break  # only fire once per utterance
                    # Clear the dedup guard so the same word can fire in the next utterance.
                    with self._lock:
                        self._last_word = None
                except Exception:
                    pass  # malformed JSON from Vosk — just ignore

            else:
                # -- Partial result: utterance is still in progress --
                # Acting on partials gives sub-50 ms response time, which is
                # important for the countdown beat words.
                try:
                    partial_json = json.loads(rec.PartialResult())
                    partial_text = partial_json.get("partial", "").strip().lower()
                    # Only act if the partial text changed since last callback.
                    if partial_text and partial_text != last_partial:
                        last_partial = partial_text
                        for w in partial_text.split():
                            if w in _BEAT_WORDS or w in _THROW_WORDS or w in _NAV_WORDS:
                                with self._lock:
                                    already_sent = (self._last_word == w)
                                if not already_sent:
                                    self._dispatch_word(w)
                                break  # only fire once per partial update
                except Exception:
                    pass  # ignore JSON parse errors from Vosk

        # Open the microphone stream and keep it alive until stop() is called.
        try:
            with sd.RawInputStream(
                samplerate=self.SAMPLE_RATE,
                blocksize=self.BLOCK_SIZE,
                dtype="int16",
                channels=1,
                callback=_audio_callback,
            ):
                print("[Voice] Microphone open — listening")
                # Sleep in short increments so stop() is noticed quickly.
                while not self._stop_event.is_set():
                    self._stop_event.wait(timeout=0.05)

        except sd.CallbackAbort:
            # This is the normal clean-shutdown path — CallbackAbort is raised
            # inside the callback when we detect the stop event.
            pass

        except Exception as exc:
            # Any other exception (permissions, device error, etc.) — store the
            # message so the UI can show it to the user.
            self._error = f"Voice listener error: {exc}"
            self._running = False
            print(f"[Voice] Error: {exc}")
            # Give a specific hint for the most common failure: macOS mic permission.
            if any(k in str(exc) for k in ("Permission", "Invalid", "-9986", "denied")):
                print("[Voice] Microphone permission denied.")
                print("        System Settings → Privacy & Security → Microphone")
                print("        Enable Terminal or iTerm2, then restart the app.")

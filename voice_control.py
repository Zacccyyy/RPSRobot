"""
voice_control.py — Voice input controller for RPS gesture recogniser.

Provides an alternative to the physical pump countdown for accessibility.
Uses Vosk offline speech recognition with a grammar-constrained vocabulary.

PROTOCOL
--------
Countdown: say "ready" → "one" → "two" → "three" → "shoot"
Throw:     say "rock", "paper", or "scissors" during the SHOOT window

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
        # event = {"type": "beat",  "word":    "ready" | "one" | "two" | "three" | "shoot"}
        # event = {"type": "throw", "gesture": "Rock"  | "Paper" | "Scissors"}
        ...

    vc.stop()
"""

import json
import numpy as _np
import os
import queue
import threading

# --------------------------------------------------------------------------- #
# Optional dependencies — wrapped so the app still starts if not installed.   #
# --------------------------------------------------------------------------- #
try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False

try:
    import vosk
    _VOSK_AVAILABLE = True
except ImportError:
    _VOSK_AVAILABLE = False

VOSK_AVAILABLE = _SD_AVAILABLE and _VOSK_AVAILABLE

# --------------------------------------------------------------------------- #
# Recognition vocabulary                                                        #
# --------------------------------------------------------------------------- #
# Every word the grammar recogniser can hear. "[unk]" is the catch-all.
# Beat words: countdown words for the RPS pump protocol.
# Throw words: gesture names + phonetic variants.
# Nav words:  navigation + game shortcuts + game-specific actions.
#
# Design principle: for each canonical word, register every plausible
# mishearing or accent variant that maps to the same action.

_BEAT_WORDS = frozenset({
    # Canonical countdown
    "ready", "one", "two", "three",
    # "ready" variants
    "steady", "freddy", "eddie", "reddish", "already", "betty", "reddy",
    # "one" variants (Vosk frequently mishears)
    "won", "on", "and", "wan", "run", "gun", "none", "juan", "in",
    # "two" variants
    "to", "too", "do", "the", "a", "who", "new", "tu", "tew",
    # "three" variants (hardest — wide accent range)
    "tree", "free", "freed", "sri", "through", "re", "street",
    "throw", "threat", "thresh", "thrice", "drei",
})

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

_THROW_WORDS = {
    # ── Rock ─────────────────────────────────────────────────────────────
    "rock":       "Rock",
    "lock":       "Rock",
    "block":      "Rock",
    "knock":      "Rock",
    "walk":       "Rock",
    "talk":       "Rock",
    "dock":       "Rock",
    "roc":        "Rock",
    "rok":        "Rock",
    # ── Paper ────────────────────────────────────────────────────────────
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
    # ── Scissors ─────────────────────────────────────────────────────────
    "scissors":   "Scissors",
    "sisters":    "Scissors",
    "seizures":   "Scissors",
    "cesars":     "Scissors",
    "figures":    "Scissors",
    "sizzle":     "Scissors",
    "scissors":   "Scissors",
    "scissor":    "Scissors",
    "cissors":    "Scissors",
    "scissored":  "Scissors",
    # ── RPSLS extras ─────────────────────────────────────────────────────
    "lizard":     "Lizard",
    "wizard":     "Lizard",   # mishearing
    "blizzard":   "Lizard",
    "spock":      "Spock",
    "spot":       "Spock",    # mishearing
    "stock":      "Spock",
    "spark":      "Spock",
}

_NAV_WORDS = {
    # ── Directional navigation ────────────────────────────────────────────
    "up":           "up",
    "higher":       "up",
    "above":        "up",
    "previous":     "up",
    "prev":         "up",
    "down":         "down",
    "lower":        "down",
    "below":        "down",
    "next":         "down",    # "next item" in lists = scroll down
    "town":         "down",    # mishearing
    "left":         "left",
    "right":        "right",

    # ── Confirm / select ─────────────────────────────────────────────────
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

    # ── Cancel / back ─────────────────────────────────────────────────────
    "back":         "back",
    "no":           "back",
    "nope":         "back",
    "nah":          "back",
    "cancel":       "back",
    "escape":       "back",
    "menu":         "back",
    "return":       "back",

    # ── Quit ──────────────────────────────────────────────────────────────
    "quit":         "quit",
    "exit":         "quit",
    "close":        "quit",

    # ── Restart / play again ──────────────────────────────────────────────
    "restart":      "restart",
    "again":        "restart",
    "replay":       "restart",
    "retry":        "restart",
    "repeat":       "restart",
    "redo":         "restart",

    # ── Start / begin (for modes with explicit start) ──────────────────────
    "start":        "start",
    "begin":        "start",
    "play":         "start",
    "launch":       "start",
    "go":           "start",   # also mapped to select — both work contextually

    # ── Next / skip (tutorial and multi-step flows) ───────────────────────
    "skip":         "next",
    "forward":      "next",
    "continue":     "next",
    "advance":      "next",

    # ── Toggle commentary ─────────────────────────────────────────────────
    "commentary":   "commentary",
    "comment":      "commentary",
    "commentate":   "commentary",
    "narrate":      "commentary",

    # ── Direct main-menu shortcuts ────────────────────────────────────────
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

    # ── Direct game shortcuts (from anywhere) ─────────────────────────────
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
    "spock":        "rpsls",   # saying Spock on the menu = go to RPSLS
    "games":        "gamemodes",
    "modes":        "gamemodes",
}

# Default locations to search for the Vosk model directory.
_DEFAULT_MODEL_NAME = "vosk-model-small-en-us-0.15"
_INDIAN_MODEL_NAME  = "vosk-model-small-en-in-0.4"

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
    """Return the path to the Vosk model directory, or None if not found."""
    if override and os.path.isdir(override):
        return override

    # Try Indian English model first if preferred
    if prefer_indian:
        for p in _INDIAN_MODEL_SEARCH_PATHS:
            if os.path.isdir(p):
                return p

    for p in _DEFAULT_MODEL_SEARCH_PATHS:
        if os.path.isdir(p):
            return p

    # Fall back to Indian English if US not found
    if not prefer_indian:
        for p in _INDIAN_MODEL_SEARCH_PATHS:
            if os.path.isdir(p):
                return p

    return None


# --------------------------------------------------------------------------- #
# VoiceController                                                               #
# --------------------------------------------------------------------------- #
class VoiceController:
    """
    Background-thread voice listener.

    Recognises a small fixed vocabulary and posts typed events to a queue.
    All public methods are thread-safe.
    """

    SAMPLE_RATE = 16000   # Hz — required by Vosk small model
    BLOCK_SIZE  = 800     # Frames per audio callback (~50 ms at 16 kHz)
                          # Smaller = lower recognition latency (was 4000 = 250 ms)

    def __init__(self, model_path=None, verbose=False, prefer_indian=False):
        """
        Parameters
        ----------
        model_path : str or None
            Path to an unpacked Vosk model directory.  If None, the controller
            searches the default locations listed in _DEFAULT_MODEL_SEARCH_PATHS.
        verbose : bool
            If True, print each recognised word to stdout.
        prefer_indian : bool
            If True, prefer vosk-model-small-en-in-0.4 over the US model.
            Better for Australian and non-American accents.
        """
        self._model_path    = model_path
        self._verbose       = verbose
        self._prefer_indian = prefer_indian
        self._event_queue   = queue.Queue()
        self._thread        = None
        self._stop_event    = threading.Event()
        self._running       = False
        self._error         = None
        self._last_word     = ""
        self._mic_level     = 0.0   # RMS of last audio block (0.0–1.0)
        self._lock          = threading.Lock()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def start(self):
        """
        Start the background listening thread.

        Returns True on success, False if a dependency is missing or the model
        cannot be found.  Call get_error() for a human-readable explanation.
        """
        if self._running:
            return True

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

        path = _find_model_path(self._model_path, prefer_indian=self._prefer_indian)
        if path is None:
            self._error = (
                f"Vosk model not found.  Download the small English model:\n"
                "  https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip\n"
                f"Unzip to:  {_DEFAULT_MODEL_SEARCH_PATHS[0]}"
            )
            return False

        # Load the model on the calling thread — cheap enough (~0.3 s).
        try:
            vosk.SetLogLevel(-1)          # silence Vosk's verbose stdout
            model = vosk.Model(path)
        except Exception as exc:
            self._error = f"Failed to load Vosk model at '{path}': {exc}"
            return False

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
        """Signal the background thread to stop and wait for it to exit."""
        if not self._running:
            return
        self._stop_event.set()
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        print("[Voice] Stopped")

    def drain_events(self):
        """
        Return all queued voice events and clear the queue.

        Each event is a dict:
          {"type": "beat",  "word":    <str>}    — countdown word heard
          {"type": "throw", "gesture": <str>}    — throw gesture heard
        """
        events = []
        while True:
            try:
                events.append(self._event_queue.get_nowait())
            except queue.Empty:
                break
        return events

    def is_running(self):
        """Return True if the background thread is active."""
        return self._running

    def get_error(self):
        """Return the last error string, or None if no error."""
        return self._error

    def get_last_word(self):
        """Return the most recently recognised word (for UI display)."""
        with self._lock:
            return self._last_word

    def get_mic_level(self):
        """Return normalised RMS mic level (0.0–1.0) for the last audio block."""
        with self._lock:
            return self._mic_level

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _dispatch_word(self, word):
        """Classify a recognised word and post the appropriate event."""
        word = word.strip().lower()
        if not word or word == "[unk]":
            return

        # Normalise beat-word variants to their canonical form so controllers
        # only need to check "ready", "one", "two", "three" — not every alias.
        canonical = _BEAT_CANONICAL.get(word, word)

        with self._lock:
            self._last_word = canonical

        if canonical in _BEAT_WORDS and canonical in _BEAT_CANONICAL.values():
            # Only dispatch canonical forms so no variant slips through
            if canonical in ("ready", "one", "two", "three"):
                self._event_queue.put({"type": "beat", "word": canonical})
                if self._verbose: print(f"[Voice] Beat: {canonical} (heard: {word})")

        elif word in _THROW_WORDS:
            gesture = _THROW_WORDS[word]
            self._event_queue.put({"type": "throw", "gesture": gesture})
            if self._verbose: print(f"[Voice] Throw: {gesture} (heard: {word})")

        elif word in _NAV_WORDS:
            self._event_queue.put({"type": "nav", "action": _NAV_WORDS[word]})
            if self._verbose: print(f"[Voice] Nav: {_NAV_WORDS[word]}")

    def _listen_loop(self, model):
        """
        Background thread: reads microphone audio, feeds Vosk, dispatches words.

        Grammar-constrained recognition — the closed vocabulary is passed directly
        into KaldiRecognizer so Vosk's decoder only searches a ~30-word space
        instead of 50,000+ words. This is the approach used in the official Vosk
        example (test_words.py) and is the primary driver of low latency and
        high accuracy for command-word applications.

        The grammar includes phonetic variants (tree/free/lock etc.) so the
        constrained decoder will still map accented pronunciations correctly.
        """
        # Build grammar from every word we might dispatch — canonical + variants.
        # "[unk]" is required so Vosk has a catch-all for non-vocabulary sounds.
        all_vocab = (
            list(_BEAT_WORDS)
            + list(_THROW_WORDS.keys())
            + list(_NAV_WORDS.keys())
            + ["[unk]"]
        )
        grammar_json = json.dumps(all_vocab)

        # Pass grammar to constructor — this constrains the search at the decoder
        # level, which is far more effective than post-filtering in Python.
        try:
            rec = vosk.KaldiRecognizer(model, self.SAMPLE_RATE, grammar_json)
        except Exception:
            # Fall back to open-vocab if this Vosk version doesn't support it
            rec = vosk.KaldiRecognizer(model, self.SAMPLE_RATE)

        last_partial = ""

        def _audio_callback(indata, frames, time_info, status):
            nonlocal last_partial

            if self._stop_event.is_set():
                raise sd.CallbackAbort()

            if status:
                print(f"[Voice] Audio status: {status}")

            data = bytes(indata)

            # Update mic level for the waveform indicator
            samples = _np.frombuffer(data, dtype=_np.int16).astype(_np.float32)
            rms = float(_np.sqrt(_np.mean(samples ** 2))) / 32768.0
            with self._lock:
                self._mic_level = min(1.0, rms * 6.0)  # scale up for visibility

            # Feed audio to decoder first
            if rec.AcceptWaveform(data):
                # ── Final result (utterance complete) ────────────────────
                try:
                    result = json.loads(rec.Result())
                    text   = result.get("text", "").strip().lower()
                    last_partial = ""
                    if text:
                        for w in text.split():
                            if w in _BEAT_WORDS or w in _THROW_WORDS or w in _NAV_WORDS:
                                with self._lock:
                                    already_sent = (self._last_word == w)
                                if not already_sent:
                                    self._dispatch_word(w)
                                break
                except Exception:
                    pass
            else:
                # ── Partial result (mid-utterance, low latency) ──────────
                try:
                    partial_json = json.loads(rec.PartialResult())
                    partial_text = partial_json.get("partial", "").strip().lower()
                    if partial_text and partial_text != last_partial:
                        last_partial = partial_text
                        for w in partial_text.split():
                            if w in _BEAT_WORDS or w in _THROW_WORDS or w in _NAV_WORDS:
                                self._dispatch_word(w)
                                break
                except Exception:
                    pass

        try:
            with sd.RawInputStream(
                samplerate=self.SAMPLE_RATE,
                blocksize=self.BLOCK_SIZE,
                dtype="int16",
                channels=1,
                callback=_audio_callback,
            ):
                print("[Voice] Microphone open — listening")
                while not self._stop_event.is_set():
                    self._stop_event.wait(timeout=0.05)

        except sd.CallbackAbort:
            pass

        except Exception as exc:
            self._error = f"Voice listener error: {exc}"
            self._running = False
            print(f"[Voice] Error: {exc}")
            if any(k in str(exc) for k in ("Permission", "Invalid", "-9986", "denied")):
                print("[Voice] Microphone permission denied.")
                print("        System Settings → Privacy & Security → Microphone")
                print("        Enable Terminal or iTerm2, then restart the app.")

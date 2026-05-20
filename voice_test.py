"""
voice_test.py
=============
Standalone microphone + Vosk speech recognition diagnostic script.

Run this BEFORE the main app to confirm mic access and the Vosk model are
working correctly:
    cd ~/rps_hand_counter
    source .venv/bin/activate
    python voice_test.py

The script will:
  1. Check that sounddevice and vosk are installed.
  2. Locate the Vosk model folder.
  3. List all audio input devices and show which is the default.
  4. Record 2 seconds of audio to confirm the mic is actually receiving input.
  5. Start live speech recognition and print every word it hears.

Press Ctrl+C to exit the live recognition loop.
"""

import sys
import os
import json
import time

# ── Dependency checks ─────────────────────────────────────────────────────────
# Do these first so we get a clear error message before importing numpy etc.

print("Checking sounddevice...", end=" ")
try:
    import sounddevice as sd
    print("OK")
except ImportError:
    print("NOT INSTALLED — run:  pip install sounddevice")
    sys.exit(1)

print("Checking vosk...", end=" ")
try:
    import vosk
    print("OK")
except ImportError:
    print("NOT INSTALLED — run:  pip install vosk")
    sys.exit(1)

# ── Find Vosk model ───────────────────────────────────────────────────────────

MODEL_NAME = "vosk-model-small-en-us-0.15"

# Check a few common locations where the model might have been installed.
SEARCH_PATHS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_NAME),  # next to this file
    os.path.expanduser(f"~/Desktop/CapStone/{MODEL_NAME}"),                 # CapStone data folder
    os.path.expanduser(f"~/Downloads/{MODEL_NAME}"),                        # common download spot
    os.path.expanduser(f"~/{MODEL_NAME}"),                                  # home directory
]

print("Looking for Vosk model...", end=" ")
model_path = None
for p in SEARCH_PATHS:
    if os.path.isdir(p):
        model_path = p
        break  # found it — stop searching

if model_path is None:
    print("NOT FOUND")
    print("  Download: https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip")
    print(f"  Unzip to: {SEARCH_PATHS[1]}")
    sys.exit(1)

print(f"OK\n  {model_path}")

# ── List audio input devices ──────────────────────────────────────────────────

print("\nAudio input devices:")
devices       = sd.query_devices()
default_input = sd.default.device[0]  # index of the system default input device

for i, d in enumerate(devices):
    # Only show devices that can actually record audio (max_input_channels > 0).
    if d['max_input_channels'] > 0:
        marker = " ◀ DEFAULT" if i == default_input else ""
        print(f"  [{i}] {d['name']}{marker}")

# ── Test mic access ───────────────────────────────────────────────────────────

print(f"\nTesting microphone (2 seconds) ...")
print("  If this hangs: System Settings -> Privacy & Security -> Microphone")
print("  Enable Terminal / iTerm2, then re-run.\n")

import numpy as np

# Use a mutable container (list) so the callback can write to it.
# A plain bool can't be reassigned inside a nested function in Python,
# and using `global` here would be messy — the list trick is idiomatic.
audio_received = [False]

def _test_cb(indata, frames, t, status):
    """
    Audio callback during the 2-second mic test.
    Sets audio_received[0] = True if any non-trivial signal comes in.
    Threshold of 10 filters out near-silence / mic noise floor.
    """
    arr = np.frombuffer(bytes(indata), dtype=np.int16).astype(np.float32)
    if np.max(np.abs(arr)) > 10:
        audio_received[0] = True

try:
    # Record for 2 seconds at 16 kHz mono (same settings Vosk expects).
    with sd.RawInputStream(samplerate=16000, blocksize=4000,
                           dtype='int16', channels=1, callback=_test_cb):
        time.sleep(2)
except Exception as e:
    print(f"  ERROR: {e}")
    print("  -> Grant mic permission to Terminal in System Settings.")
    sys.exit(1)

if audio_received[0]:
    print("  Microphone receiving audio ✓")
else:
    print("  WARNING: No audio signal detected — check mic isn't muted.")

# ── Live speech recognition ───────────────────────────────────────────────────

print("\nLoading Vosk model...", end=" ", flush=True)
vosk.SetLogLevel(-1)  # suppress Vosk's verbose internal logging
model = vosk.Model(model_path)
print("OK")

# KaldiRecognizer converts raw audio bytes to text.
# 16000 must match the sample rate of the audio stream below.
rec = vosk.KaldiRecognizer(model, 16000)
# Note: SetGrammar isn't used here — it isn't available in all Vosk versions.
# Instead we filter recognised words against our known vocabulary in Python.

# Words the app actually uses for navigation and game control.
KNOWN = {
    "ready", "one", "two", "three",
    "rock", "paper", "scissors",
    "up", "down", "select", "yes", "back", "no", "quit",
    "challenge", "settings", "fair", "cheat",
}

print("\n" + "=" * 52)
print("LISTENING — speak any game word:")
print("  ready  one  two  three")
print("  rock   paper  scissors")
print("  up  down  select  back  quit")
print("Ctrl+C to stop.")
print("=" * 52 + "\n")

word_count   = 0    # total number of complete utterances Vosk has finalised
last_partial = ""   # tracks the last partial result so we only print on changes

def _rec_cb(indata, frames, t, status):
    """
    Audio callback for the live recognition loop.

    Called by sounddevice for every audio block (~250ms at blocksize=4000).
    We feed raw bytes into Vosk and print whatever it hears.
    """
    global word_count, last_partial
    data = bytes(indata)

    # --- Partial result: Vosk's running best guess (updates in real time) ---
    try:
        p = json.loads(rec.PartialResult()).get("partial", "").strip()
        if p and p != last_partial:
            last_partial = p
            matched = [w for w in p.split() if w in KNOWN]
            if matched:
                # \r overwrites the line so partials don't scroll the terminal.
                print(f"  partial -> {' '.join(matched)}", end="\r")
    except Exception:
        pass

    # --- Final result: Vosk is confident the utterance is complete ---
    if rec.AcceptWaveform(data):
        try:
            text = json.loads(rec.Result()).get("text", "").strip()
            last_partial = ""  # clear the partial display
            if text:
                matched    = [w for w in text.split() if w in KNOWN]
                word_count += 1
                if matched:
                    print(f"  HEARD [{word_count:03d}] ✓  {', '.join(matched)}  (raw: \"{text}\")          ")
                else:
                    print(f"  HEARD [{word_count:03d}] ✗  (not in vocabulary)  raw: \"{text}\"          ")
        except Exception:
            pass

try:
    # Open the mic stream and run until the user presses Ctrl+C.
    with sd.RawInputStream(samplerate=16000, blocksize=4000,
                           dtype='int16', channels=1, callback=_rec_cb):
        while True:
            time.sleep(0.1)  # just keep the main thread alive while callbacks fire

except KeyboardInterrupt:
    print(f"\n\nDone. Vosk processed {word_count} utterance(s).")
    if word_count == 0:
        print("No speech detected. Try:")
        print("  * Speak louder / closer to the mic")
        print("  * Check mic isn't muted in System Settings -> Sound")
    else:
        print("Voice recognition working correctly ✓")

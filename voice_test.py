"""
voice_test.py — Standalone microphone + Vosk diagnostic script.

Run this BEFORE the main app to confirm everything is working:
    cd ~/rps_hand_counter
    source .venv/bin/activate
    python voice_test.py

Press Ctrl+C to exit.
"""

import sys
import os
import json
import time

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

# ── Find model ──────────────────────────────────────────────────────────────
MODEL_NAME = "vosk-model-small-en-us-0.15"
SEARCH_PATHS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_NAME),
    os.path.expanduser(f"~/Desktop/CapStone/{MODEL_NAME}"),
    os.path.expanduser(f"~/Downloads/{MODEL_NAME}"),
    os.path.expanduser(f"~/{MODEL_NAME}"),
]

print("Looking for Vosk model...", end=" ")
model_path = None
for p in SEARCH_PATHS:
    if os.path.isdir(p):
        model_path = p
        break

if model_path is None:
    print("NOT FOUND")
    print("  Download: https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip")
    print(f"  Unzip to: {SEARCH_PATHS[1]}")
    sys.exit(1)
print(f"OK\n  {model_path}")

# ── List devices ────────────────────────────────────────────────────────────
print("\nAudio input devices:")
devices = sd.query_devices()
default_input = sd.default.device[0]
for i, d in enumerate(devices):
    if d['max_input_channels'] > 0:
        marker = " ◀ DEFAULT" if i == default_input else ""
        print(f"  [{i}] {d['name']}{marker}")

# ── Test mic access ─────────────────────────────────────────────────────────
print(f"\nTesting microphone (2 seconds) ...")
print("  If this hangs: System Settings → Privacy & Security → Microphone")
print("  Enable Terminal / iTerm2, then re-run.\n")

import numpy as np
audio_received = [False]

def _test_cb(indata, frames, t, status):
    arr = np.frombuffer(bytes(indata), dtype=np.int16).astype(np.float32)
    if np.max(np.abs(arr)) > 10:
        audio_received[0] = True

try:
    with sd.RawInputStream(samplerate=16000, blocksize=4000,
                           dtype='int16', channels=1, callback=_test_cb):
        time.sleep(2)
except Exception as e:
    print(f"  ERROR: {e}")
    print("  → Grant mic permission to Terminal in System Settings.")
    sys.exit(1)

if audio_received[0]:
    print("  Microphone receiving audio ✓")
else:
    print("  WARNING: No audio signal detected — check mic isn't muted.")

# ── Live recognition ────────────────────────────────────────────────────────
print("\nLoading Vosk model...", end=" ", flush=True)
vosk.SetLogLevel(-1)
model = vosk.Model(model_path)
print("OK")

rec = vosk.KaldiRecognizer(model, 16000)
# NOTE: SetGrammar not used — not available in all Vosk versions.
# We filter by known words in Python instead.

KNOWN = {
    "ready", "one", "two", "three",
    "rock", "paper", "scissors",
    "up", "down", "select", "yes", "back", "no", "quit",
    "challenge", "settings", "fair", "cheat",
}

print("\n" + "="*52)
print("LISTENING — speak any game word:")
print("  ready  one  two  three")
print("  rock   paper  scissors")
print("  up  down  select  back  quit")
print("Ctrl+C to stop.")
print("="*52 + "\n")

word_count = 0
last_partial = ""

def _rec_cb(indata, frames, t, status):
    global word_count, last_partial
    data = bytes(indata)

    # Partial
    try:
        p = json.loads(rec.PartialResult()).get("partial", "").strip()
        if p and p != last_partial:
            last_partial = p
            matched = [w for w in p.split() if w in KNOWN]
            if matched:
                print(f"  partial → {' '.join(matched)}", end="\r")
    except Exception:
        pass

    # Final
    if rec.AcceptWaveform(data):
        try:
            text = json.loads(rec.Result()).get("text", "").strip()
            last_partial = ""
            if text:
                matched = [w for w in text.split() if w in KNOWN]
                raw_all  = text
                word_count += 1
                if matched:
                    print(f"  HEARD [{word_count:03d}] ✓  {', '.join(matched)}  (raw: \"{raw_all}\")          ")
                else:
                    print(f"  HEARD [{word_count:03d}] ✗  (not in vocabulary)  raw: \"{raw_all}\"          ")
        except Exception:
            pass

try:
    with sd.RawInputStream(samplerate=16000, blocksize=4000,
                           dtype='int16', channels=1, callback=_rec_cb):
        while True:
            time.sleep(0.1)
except KeyboardInterrupt:
    print(f"\n\nDone. Vosk processed {word_count} utterance(s).")
    if word_count == 0:
        print("No speech detected. Try:")
        print("  • Speak louder / closer to the mic")
        print("  • Check mic isn't muted in System Settings → Sound")
    else:
        print("Voice recognition working correctly ✓")

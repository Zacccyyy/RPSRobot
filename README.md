# RPS Gesture Recogniser

A real-time Rock-Paper-Scissors gesture recognition system with an adaptive AI opponent,
multiple game modes, voice control, player profiling, and optional physical robot output.

Built for macOS (Apple M-series). Windows users see the Windows section below.

---

## Requirements

- Python 3.9 or later
- A standard webcam (built-in or USB)
- macOS 12+ (Monterey or later) — see Windows notes below

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/rps_hand_counter.git
cd rps_hand_counter
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Vosk speech recognition model (required for voice control)

Voice control requires a ~50MB model file. Download it manually:

```bash
mkdir -p models
cd models
curl -LO https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
cd ..
```

If you prefer the Indian/Australian accent model:

```bash
curl -LO https://alphacephei.com/vosk/models/vosk-model-small-en-in-0.4.zip
unzip vosk-model-small-en-in-0.4.zip
```

Select which model to use inside the app under Settings > Voice Model.

> Voice control is optional. The app runs fully without it — voice just won't respond.

### 5. (Optional) Set your Anthropic API key for live commentary

The commentary engine (press C in-game) uses the Claude API. Add your key to your shell profile:

```bash
echo 'export ANTHROPIC_API_KEY="your-key-here"' >> ~/.zshrc
source ~/.zshrc
```

Get a key at https://console.anthropic.com. Commentary is off by default and costs
only fractions of a cent per round when enabled.

### 6. Create the data directory

```bash
mkdir -p ~/Desktop/CapStone/fingerprints
```

---

## Running the app

```bash
source .venv/bin/activate
python main.py
```

On first launch you will be asked to enter your name. This creates your player profile.
After that the main menu loads automatically on every subsequent launch.

---

## Voice commands (overview)

Voice control is enabled under Features > Voice Mode.

| What you say | What it does |
|---|---|
| READY / ONE / TWO / THREE | RPS countdown beats |
| ROCK / PAPER / SCISSORS | Throw a gesture |
| LIZARD / SPOCK | RPSLS gestures |
| BACK / CANCEL | Return to previous screen |
| QUIT / EXIT | Quit the app |
| RESTART / AGAIN | Restart current game |
| START / BEGIN | Begin a session (Gesture Trainer etc.) |
| SNAKE / SQUID / SIMON / REFLEX | Jump directly to that game mode |
| STATS / SETTINGS / TUTORIAL | Open that menu section |
| COMMENTARY | Toggle live commentary |

Press `?` at any time for a full in-game voice command reference.

---

## Controls (keyboard)

| Key | Action |
|---|---|
| W / S or Arrow keys | Navigate menus |
| Enter | Select / confirm |
| ESC | Back / return to menu |
| Q | Quit |
| C | Toggle commentary |
| M | Toggle diagnostic mode |
| N | Toggle sound |
| ? | Show help overlay |

---

## Hardware robot arm (optional)

The app can send move commands to an ESP32-based servo arm over USB serial.
This is entirely optional — the app runs fully without it.

Connect the ESP32 via USB before launching the app. The serial bridge
auto-detects available ports. See `serial_bridge.py` for port configuration.

---

## Data and files

All persistent data is saved to `~/Desktop/CapStone/`:

| Path | Contents |
|---|---|
| `player_research_log.xlsx` | Full round-by-round research log |
| `simulation_results.xlsx` | Simulation Lab outputs |
| `snake_highscore.json` | Arcade Snake leaderboard |
| `fingerprints/` | Gesture fingerprint enrollment data |

Player profiles (gesture history, AI state, stats) are saved as JSON files
in the project directory under `profiles/`.

---

## Windows notes

The app is developed and tested on macOS. It will run on Windows with the
following adjustments:

### What works the same
- All gesture recognition, game modes, AI, stats, simulations, fingerprint system
- Voice control (Vosk is cross-platform)
- Serial bridge to ESP32

### What needs adjusting on Windows

**1. Sound is disabled**
Sound uses `afplay`, which is macOS-only. On Windows the sound player
silently does nothing — no crash, just no audio. A future update will
add Windows audio support via `winsound` or `playsound`.

**2. Terminal auto-close on quit does not work**
The app uses `osascript` (AppleScript) to close the terminal window when
you press Q. On Windows this silently fails — just close the terminal manually.

**3. Virtual environment activation is different**
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**4. Webcam index may differ**
If the camera doesn't open, edit `main.py` line ~60 and change
`cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` or `cv2.VideoCapture(2)`.

**5. Serial port names are different**
ESP32 shows as `COM3`, `COM4` etc. instead of `/dev/tty.usbserial-...`.
Update `serial_bridge.py` with your COM port if auto-detect fails.

**6. Run with**
```bat
python main.py
```
(not `python3` — on Windows the command is usually `python`)

### Tested Python versions
- macOS: Python 3.9, 3.11
- Windows: Python 3.11 (community tested — not officially supported)

---

## Troubleshooting

**Camera not opening**
Make sure no other app (FaceTime, Zoom, etc.) is using the camera.
Try changing the camera index in `main.py`.

**MediaPipe warnings in terminal**
The `landmark_projection_calculator` warning is harmless — MediaPipe
prints it internally and does not affect gesture recognition.

**Voice not responding**
Check that the Vosk model folder exists at `models/vosk-model-small-en-us-0.15/`.
Enable voice under Features > Voice Mode. Check your microphone permissions
in System Settings > Privacy > Microphone.

**Excel log corruption error**
If you see `Error -3 while decompressing data`, the Excel log file is
corrupted. The app will automatically back it up and create a fresh one.
No round data is lost from your JSON profile.

**Fingerprint enrollment not working**
Make sure you are enrolling via Settings > Enroll Fingerprint, NOT via
the Squid Game mode in the game menu. Only the Settings path activates
fingerprint collection. You need at least 25 dot captures before the
classifier trains.

---

## Project structure

```
rps_hand_counter/
├── main.py                    # Entry point and run loop
├── requirements.txt           # Python dependencies
├── config.json                # User configuration (auto-saved)
├── models/                    # Vosk speech model (download separately)
│   └── vosk-model-small-en-us-0.15/
├── profiles/                  # Player profile JSON files (auto-created)
├── ui_base.py                 # Colours, layout helpers, drawing primitives
├── ui_game.py                 # In-game screens
├── ui_modes.py                # Per-mode renderers
├── ui_menus.py                # Menu / settings / stats screens
├── ui_renderer.py             # Re-export shim (do not edit)
├── fair_play_state.py         # Main RPS game logic + Thompson Sampling AI
├── gesture_fingerprint.py     # Biometric fingerprint system
├── squid_fingerprint_state.py # Fingerprint enrollment via Squid Game
├── voice_control.py           # Vosk speech recognition
├── commentary_engine.py       # Claude API live commentary
├── player_profile_store.py    # Player profiles and round logging
├── serial_bridge.py           # ESP32 robot arm serial output
└── sound_player.py            # macOS audio (afplay)
```

---

## Academic context

This project is a robotics engineering capstone at an Australian university.
The gesture recognition system, AI opponent modelling, and biometric
fingerprint subsystem are original research contributions.

The companion business capstone covers TrickWing Toys' commercialisation
plan for the RPS Robot consumer product (AUD $39.95 RRP, target age 7-10).

---

*Last updated: April 2026*

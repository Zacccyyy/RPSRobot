# RPS Robot — Gesture Recognition System
**TrickWing Toys / RavensAgency**

Real-time Rock Paper Scissors gesture recognition with adaptive AI opponent,
hand geometry biometrics, voice control, and physical robot arm support.

---

## System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| macOS | 11 (Big Sur) | 13+ (Ventura) |
| Chip | Intel Core i5 | Apple M-series |
| RAM | 8GB | 16GB |
| Webcam | 720p built-in | 1080p USB |
| Storage | 2GB free | 5GB free |
| Internet | Required for install | — |

---

## Installation (First Time)

1. **Download** the RPS Robot folder and unzip it anywhere (Desktop recommended)

2. **Open Terminal**
   - Press `Cmd + Space`, type `Terminal`, press Enter

3. **Run the installer:**
   ```bash
   bash ~/Desktop/rps_hand_counter/install.sh
   ```
   - Enter your Mac password when prompted (for Homebrew)
   - The installer will download ~400MB of dependencies
   - Takes 5–10 minutes on a typical connection

4. **Launch:** Double-click `Launch RPS Robot.command` on your Desktop

---

## Launching the App

After installation, launch the app one of two ways:

**Option A — Desktop shortcut (recommended)**
Double-click `Launch RPS Robot.command` on your Desktop

**Option B — Terminal**
```bash
cd ~/rps_hand_counter
source .venv/bin/activate
python main.py
```

---

## First Run

On first launch you will be asked to:
1. Enter your name
2. Grant webcam access (macOS permission dialog — click Allow)

The app saves your data to `~/Desktop/CapStone/`

---

## Game Modes

| Mode | Description |
|---|---|
| **Fair Play** | Standard RPS vs adaptive AI with 7 personalities |
| **Challenge** | AI learns your patterns over 20+ rounds |
| **Prediction Race** | Beat the AI by NOT playing what it predicts |
| **Two Player** | PvP or PvP+AI on the same webcam |
| **Bluff Mode** | Show one gesture, lock in another |
| **Reflex** | Speed test — fastest response wins |
| **Simon Says** | Follow gesture sequences |
| **Red Light Green Light** | Squid Game-style hold challenge |
| **Arcade Snake** | Control Snake with hand gestures |
| **Gesture Trainer** | Rehabilitation-focused hold exercises |
| **RPSLS** | Rock Paper Scissors Lizard Spock |

---

## Key Controls (In Game)

| Key | Action |
|---|---|
| `ESC` | Back / Menu |
| `D` | Toggle Diagnostic overlay |
| `H` | Hardware Test (ESP32 serial) |
| `C` | Toggle AI commentary |
| `F` | Collect training data |
| `T` | Train ML model |
| `1 / 2 / 3` | Switch display mode |

---

## Voice Commands

The app supports offline voice control via Vosk.

**Menu navigation:** `"menu"`, `"settings"`, `"game modes"`, `"simulations"`

**Game shortcuts:** `"snake"`, `"squid"`, `"simon"`, `"reflex"`, `"rehab"`, `"race"`, `"rpsls"`

**In game:** `"rock"`, `"paper"`, `"scissors"`, `"restart"`, `"quit"`

**Voice model:** US English by default. Change to Indian English (better for 
Australian accents) in Settings → Voice Model.

---

## Hand Scan Biometrics

The app can identify players by their hand geometry.

1. Go to **Settings → Enroll Hand Scan**
2. Complete 20 scanning rounds (varied positions and distances)
3. After enrollment, use **Settings → Hand Scan Diagnostic** to test recognition
4. On the login screen, press **TAB** to log in by hand scan instead of typing

Data saved to: `~/Desktop/CapStone/fingerprints/`

---

## Physical Robot Arm (Optional)

If using the ESP32 microcontroller:

1. Install the [CP210x USB driver](https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers)
2. Upload the provided Arduino sketch to the ESP32
3. Connect via USB
4. In the app: `D` → `H` → select port → ENTER to connect

The app works fully without the robot arm.

---

## Data & Privacy

All data is stored locally on your machine at `~/Desktop/CapStone/`:

```
~/Desktop/CapStone/
├── fingerprints/          Hand geometry profiles
├── profiles/              Player game statistics  
├── simulations/           AI simulation results
├── challenge_research_log.xlsx
├── player_research_log.xlsx
└── crash_*.txt            Crash reports (if any)
```

No data is sent to any server except:
- **AI Commentary** (optional, off by default): sends round history to Anthropic API

---

## Troubleshooting

**"Camera not found"**
- Make sure no other app is using the webcam (Zoom, FaceTime, etc.)
- Grant camera permission: System Settings → Privacy & Security → Camera

**"Gesture not recognising"**
- Ensure good lighting — avoid backlit windows
- Hold hand flat, palm facing camera, fingers spread
- Press `D` to see the diagnostic overlay

**"Voice commands not working"**
- Check the Vosk model folder exists: `~/rps_hand_counter/vosk-model-small-en-us-0.15/`
- If missing, re-run the installer

**"ESP32 port busy"**
- Close Arduino IDE Serial Monitor before connecting in-app

**App crashed?**
- Crash reports saved to `~/Desktop/CapStone/crash_*.txt`

---

## Technical Stack

- Python 3.9+ 
- OpenCV 4.11 — camera capture and display
- MediaPipe 0.10.21 — hand landmark detection (21 landmarks)
- scikit-learn — SVM gesture classifier + hand biometrics
- Vosk — offline speech recognition
- Anthropic Claude API — AI commentary and Theory of Mind opponent
- openpyxl — research data logging to Excel
- pyserial — ESP32 robot arm communication

---

## Academic Attribution

This system was developed as part of an undergraduate robotics engineering
capstone project. The gesture recognition approach builds on:

- Ghanbari et al. (2022). Hand geometry biometrics using MediaPipe. ICEE 2022.
- Zhang et al. (2020). MediaPipe Hands: On-device Real-time Hand Tracking. arXiv.

---

*RPS Robot — TrickWing Toys | RavensAgency*

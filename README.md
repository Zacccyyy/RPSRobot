# RPS Robot — Gesture Recognition System
**TrickWing Toys / RavensAgency**

Real-time Rock Paper Scissors gesture recognition with adaptive AI opponent,
hand geometry biometrics, voice control, and physical robot arm support.

---

## System Requirements

| Requirement | macOS | Windows |
|---|---|---|
| OS Version | macOS 11 (Big Sur) or later | Windows 10 or later |
| Chip | Intel Core i5 or Apple M-series | Intel Core i5 or AMD Ryzen 5 |
| RAM | 8GB minimum, 16GB recommended | 8GB minimum, 16GB recommended |
| Webcam | 720p built-in or USB | 720p built-in or USB |
| Storage | 2GB free | 2GB free |
| Internet | Required for install | Required for install |
| Python | Installed automatically | 3.9+ required before running installer |

---

## Installation — macOS

### Step 1 — Download the installer

Download `install.py` from the GitHub repository, or run this one-liner in Terminal:

```bash
curl -fsSL https://raw.githubusercontent.com/Zacccyyy/RPSRobot/main/install.py -o ~/Downloads/install.py
```

### Step 2 — Open Terminal

Press `Cmd + Space`, type `Terminal`, press Enter.

### Step 3 — Run the installer

```bash
python3 ~/Downloads/install.py
```

- Enter your Mac password when prompted (required for Homebrew)
- The installer downloads ~400MB of dependencies
- Takes 5–10 minutes on a typical connection
- Everything is handled automatically — Python, Git, all packages, speech model

### Step 4 — Launch

Double-click **`Launch RPS Robot.command`** on your Desktop.

Or from Terminal:
```bash
cd ~/rps_hand_counter && source .venv/bin/activate && python main.py
```

---

## Installation — Windows

### Step 1 — Install Python

If you don't already have Python 3.9 or later installed:

1. Go to [python.org/downloads](https://www.python.org/downloads/)
2. Download the latest Python 3.x installer
3. Run it — **tick "Add Python to PATH"** before clicking Install
4. Verify by opening Command Prompt and typing: `python --version`

### Step 2 — Download the installer

Download `install.py` from the GitHub repository:

```
https://github.com/Zacccyyy/RPSRobot/blob/main/install.py
```

Or open Command Prompt and run:
```cmd
curl -fsSL https://raw.githubusercontent.com/Zacccyyy/RPSRobot/main/install.py -o %USERPROFILE%\Downloads\install.py
```

> `%USERPROFILE%` is Windows shorthand for your home folder, e.g. `C:\Users\YourName`

### Step 3 — Open Command Prompt

Press `Win + R`, type `cmd`, press Enter.

### Step 4 — Run the installer

```cmd
python %USERPROFILE%\Downloads\install.py
```

- Git will be installed automatically via winget if not already present
- The installer downloads ~400MB of dependencies
- Takes 5–10 minutes on a typical connection

> **Note:** Windows Defender or SmartScreen may show a warning when running
> the installer. Click **"More info" → "Run anyway"** — this is expected for
> unsigned Python scripts.

### Step 5 — Launch

Double-click **`Launch RPS Robot.bat`** on your Desktop.

Or from Command Prompt:
```cmd
cd %USERPROFILE%\rps_hand_counter
.venv\Scripts\activate
python main.py
```

---

## First Run (Both Platforms)

On first launch you will be asked to:
1. Enter your name
2. Grant webcam access when prompted — click **Allow**

Your data is saved to:
- **macOS:** `~/Desktop/CapStone/`
- **Windows:** `C:\Users\<YourName>\Desktop\CapStone\`

---

## Updating the App

The app checks GitHub for updates every time it launches.

When an update is available, a **yellow banner** appears in the main menu showing the version difference. Press **`U`** to download and apply the update — the app restarts automatically.

This works on both macOS and Windows as long as the app was installed using the installer above (which uses `git clone` under the hood).

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
| `U` | Apply available update |
| `C` | Toggle AI commentary |
| `F` | Collect training data |
| `T` | Train ML model |
| `1 / 2 / 3` | Switch display mode |

---

## Voice Commands

The app supports offline voice control via Vosk (no internet required after install).

**Menu navigation:** `"menu"`, `"settings"`, `"game modes"`, `"simulations"`

**Game shortcuts:** `"snake"`, `"squid"`, `"simon"`, `"reflex"`, `"rehab"`, `"race"`, `"rpsls"`

**In game:** `"rock"`, `"paper"`, `"scissors"`, `"restart"`, `"quit"`

**Voice model:** US English by default. Change to Indian English in Settings → Voice Model.

---

## Hand Scan Biometrics

The app can identify players by their hand geometry — no typing required at login.

1. Go to **Settings → Enroll Hand Scan**
2. Complete 20 scanning rounds (varied positions and distances, ~3 minutes)
3. Use **Settings → Hand Scan Diagnostic** to verify recognition is working
4. On the login screen, press **TAB** to log in by hand scan instead of typing

Data saved to:
- **macOS:** `~/Desktop/CapStone/fingerprints/`
- **Windows:** `C:\Users\<YourName>\Desktop\CapStone\fingerprints\`

---

## Player Feedback

Players can submit feature suggestions and feedback from inside the app.

Press **`N`** from the main menu to open the Notes screen. Type your suggestion
and press **Enter** to submit. Each submission is saved as a timestamped `.txt`
file to `Desktop/CapStone/feedback/` so the developer can review them.

---

## Physical Robot Arm (Optional)

If using the ESP32 microcontroller:

1. Install the [CP210x USB driver](https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers)
2. Upload the provided Arduino sketch to the ESP32
3. Connect via USB
4. In the app: `D` → `H` → select port with `[` `]` → ENTER to connect

The app works fully without the robot arm.

---

## Data & Privacy

RPS Robot asks for your consent on first launch before sending any data
outside your device. You can change your choice at any time in
**Settings → Privacy Settings**.

**If you accept:** crash reports and feedback you submit are sent to a
private developer Discord channel to help improve the app.

**If you decline:** everything stays on your device only. Nothing is sent.

All data is stored locally at `Desktop/CapStone/`:

```
Desktop/CapStone/
├── fingerprints/               Hand geometry profiles (local only)
├── profiles/                   Player game statistics (local only)
├── simulations/                AI simulation results (local only)
├── feedback/                   Your submitted suggestions (local copy)
├── crash_reports/              Crash reports (local copy)
├── challenge_research_log.xlsx
└── player_research_log.xlsx
```

**What is never collected:** camera data, gameplay video, location,
hand scan biometrics, or any background tracking.

For full details see [PRIVACY.md](PRIVACY.md).

---

## Troubleshooting

### Camera not found
- Make sure no other app is using the webcam (Zoom, Teams, FaceTime etc.)
- **macOS:** System Settings → Privacy & Security → Camera → allow RPS Robot
- **Windows:** Settings → Privacy → Camera → allow desktop apps

### Gesture not recognising
- Ensure good lighting — avoid sitting with a bright window behind you
- Hold hand flat, palm facing the camera, fingers spread
- Press `D` to see the diagnostic overlay and confidence scores

### Voice commands not working
- Check the Vosk model folder exists inside `rps_hand_counter/vosk-model-small-en-us-0.15/`
- If missing, re-run the installer — it will download the model again

### ESP32 port busy
- **macOS/Windows:** Close Arduino IDE Serial Monitor before connecting in-app

### Windows — Python not found
- Re-install Python from python.org and make sure **"Add Python to PATH"** is ticked

### Windows — app won't open or SmartScreen warning
- Right-click `Launch RPS Robot.bat` → Run as administrator
- Or open Command Prompt, navigate to the folder, and run `python main.py` directly

### App crashed
- Crash reports are saved to `Desktop/CapStone/crash_*.txt`
- Open the file and check the error message at the top

---

## Technical Stack

- Python 3.9+
- OpenCV 4.11 — camera capture and display
- MediaPipe 0.10.21 — hand landmark detection (21 landmarks)
- scikit-learn — MLP gesture classifier + SVM hand biometrics
- Vosk — offline speech recognition
- Anthropic Claude API — AI commentary
- openpyxl — research data logging to Excel
- pyserial — ESP32 robot arm communication

---

## Academic Attribution

This system was developed as part of an undergraduate robotics engineering
capstone project. The following open-source projects, libraries, and research
papers directly influenced the design and implementation.

### Core Libraries

| Library | Version | Use | Link |
|---|---|---|---|
| MediaPipe | 0.10.21 | 21-point hand landmark detection | [github.com/google/mediapipe](https://github.com/google/mediapipe) |
| OpenCV | 4.11.0 | Camera capture, frame rendering, UI | [github.com/opencv/opencv](https://github.com/opencv/opencv) |
| scikit-learn | Latest | MLP classifier, SVM biometrics | [github.com/scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn) |
| Vosk | 0.3.45+ | Offline speech recognition | [github.com/alphacep/vosk-api](https://github.com/alphacep/vosk-api) |
| Anthropic Python SDK | Latest | Claude API integration | [github.com/anthropics/anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python) |
| NumPy | 1.26.4 | Numerical computation | [github.com/numpy/numpy](https://github.com/numpy/numpy) |
| openpyxl | Latest | Excel research data logging | [github.com/theorchard/openpyxl](https://github.com/theorchard/openpyxl) |
| pyserial | Latest | ESP32 serial communication | [github.com/pyserial/pyserial](https://github.com/pyserial/pyserial) |

### Gesture Recognition — GitHub Projects

These open-source projects directly influenced the gesture classification pipeline.
No code was directly copied — each project informed architectural and methodological decisions.

- **Kazuhito00/hand-gesture-recognition-using-mediapipe** (2021)
  The data collection workflow and keypoint normalisation strategy (translate to wrist origin, scale by palm size) directly inspired our landmark collection and front-on training system.
  [github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe)

- **andypotato/fingerpose** (2022)
  The curl-state analysis approach (NoCurl / HalfCurl / FullCurl using PIP and DIP joint angles) was adopted from this library's published curl thresholds and integrated into our hybrid detection pipeline.
  [github.com/andypotato/fingerpose](https://github.com/andypotato/fingerpose)

- **AishTron7/Rock-Paper-Scissor** (2023)
  Validated that SGD classifiers on MediaPipe landmarks can achieve 99.7% accuracy, informing our confidence threshold tuning.
  [github.com/AishTron7/Rock-Paper-Scissor](https://github.com/AishTron7/Rock-Paper-Scissor)

- **stefanluncanu24/RockPaperScissors-using-MediaPipe-Cv2** (2023)
  XGBoost-based gesture recognition approach reviewed for comparison against our MLP architecture.
  [github.com/stefanluncanu24/RockPaperScissors-using-MediaPipe-Cv2](https://github.com/stefanluncanu24/RockPaperScissors-using-MediaPipe-Cv2)

- **ChetanNair/Rock-Paper-Scissors** (2023)
  OpenCV and MediaPipe finger-counting approach reviewed during early gesture detection development.
  [github.com/ChetanNair/Rock-Paper-Scissors](https://github.com/ChetanNair/Rock-Paper-Scissors)

- **hjpulkki/RPS** (2020)
  Deep learning approach using Keras RNN for gesture sequence modelling, reviewed for comparison.
  [github.com/hjpulkki/RPS](https://github.com/hjpulkki/RPS)

### AI Opponent — GitHub Projects

These projects were reviewed as part of the adversarial AI literature survey.
Our AI architecture differs from all of them by implementing intentional imperfection
and personality-based play rather than pure win-maximisation.

- **dmickelson/Rock-paper-scissors** (2023) — LSTM move prediction
  [github.com/dmickelson/Rock-paper-scissors](https://github.com/dmickelson/Rock-paper-scissors)

- **MattYu/Project---Predictive-Rock-Paper-Scissor-AI** (2020) — 15-scheme swarm mind with N-gram matching
  [github.com/MattYu/Project---Predictive-Rock-Paper-Scissor-AI-](https://github.com/MattYu/Project---Predictive-Rock-Paper-Scissor-AI-)

- **Asylumrunner/Rock-Paper-Scissors-AI** (2019) — Markov chain transition modelling
  [github.com/Asylumrunner/Rock-Paper-Scissors-AI](https://github.com/Asylumrunner/Rock-Paper-Scissors-AI)

- **wesleytian/roshambo-god** (2018) — Bayesian prediction
  [github.com/wesleytian/roshambo-god](https://github.com/wesleytian/roshambo-god)

- **goelp14/RockPaperScissors** (2021) — Markov Chain difficulty tiers
  [github.com/goelp14/RockPaperScissors](https://github.com/goelp14/RockPaperScissors)

- **wmodes/rock-paper-scissors** (2023) — Heuristic computer player strategies
  [github.com/wmodes/rock-paper-scissors](https://github.com/wmodes/rock-paper-scissors)

- **Raymond Hettinger — Pattern Recognition and Reinforcement Learning (PyCon 2019)**
  Digraph-based prediction with strategy weighting.
  [rhettinger.github.io/rock_paper.html](https://rhettinger.github.io/rock_paper.html)

### Academic Papers

- Zhang, H., et al. (2020). MediaPipe Hands: On-device Real-time Hand Tracking. *arXiv:2006.10214*

- Ghanbari, A., et al. (2022). Hand geometry biometrics using MediaPipe landmark ratios. *ICEE 2022*

- Zhang, Y., Moisan, E., & Gonzalez, C. (2021). Heterogeneous cycle-based behaviours in sequential RPS. *Cognitive Science*

- Brockbank, E., & Vul, E. (2021). Failure to randomise against adaptive opponents. *Psychological Science*

- Qi, J., et al. (2021). A review of vision-based hand gesture recognition for HRI. *Complex & Intelligent Systems*

- Ahmad, T., et al. (2021). MLP classifiers on MediaPipe landmarks for gesture recognition. *IEEE Access*

- Amprimo, G., et al. (2022). MediaPipe hand tracking accuracy for clinical applications. *IEEE CBMS*

- Zohaib, M., & Nakanishi, J. (2020). Diversifying DDA agents via player state modelling. *IEEE ToG*

- Dyson, B.J., et al. (2016). Negative outcomes evoke cyclic decisions in RPS. *Nature Scientific Reports*

---

*RPS Robot — TrickWing Toys | RavensAgency*

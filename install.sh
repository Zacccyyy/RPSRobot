#!/bin/bash
# =============================================================================
#  RPS Robot — Installer
#  TrickWing Toys / RavensAgency
#
#  This is the ONLY file you need to download.
#  Run it once:  bash install.sh
#
#  What it does:
#    1. Checks macOS + hardware requirements
#    2. Installs Homebrew if needed
#    3. Installs Git if needed
#    4. Installs Python 3.9 if needed
#    5. Clones the RPS Robot repo from GitHub (enables auto-updates)
#    6. Creates a Python virtual environment
#    7. Installs all Python packages
#    8. Downloads the Vosk speech recognition model
#    9. Creates Desktop shortcuts (launcher + data folder)
#   10. Verifies everything works
#   11. Launches the app
# =============================================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
GITHUB_REPO="https://github.com/Zacccyyy/RPSRobot.git"
APP_DIR="$HOME/rps_hand_counter"
VENV_DIR="$APP_DIR/.venv"
DATA_DIR="$HOME/Desktop/CapStone"
DESKTOP="$HOME/Desktop"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

ok()   { echo -e "${GREEN}  ✓  $1${NC}"; }
info() { echo -e "${CYAN}  →  $1${NC}"; }
warn() { echo -e "${YELLOW}  ⚠  $1${NC}"; }
fail() { echo -e "${RED}  ✗  $1${NC}"; }
step() { echo -e "\n${BOLD}${CYAN}━━━  $1  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }
line() { echo -e "${CYAN}─────────────────────────────────────────────────${NC}"; }

# ── Banner ────────────────────────────────────────────────────────────────────
clear
echo ""
echo -e "${BOLD}${CYAN}"
echo "  ██████╗ ██████╗ ███████╗    ██████╗  ██████╗ ██████╗  ██████╗ ████████╗"
echo "  ██╔══██╗██╔══██╗██╔════╝    ██╔══██╗██╔═══██╗██╔══██╗██╔═══██╗╚══██╔══╝"
echo "  ██████╔╝██████╔╝███████╗    ██████╔╝██║   ██║██████╔╝██║   ██║   ██║   "
echo "  ██╔══██╗██╔═══╝ ╚════██║    ██╔══██╗██║   ██║██╔══██╗██║   ██║   ██║   "
echo "  ██║  ██║██║     ███████║    ██║  ██║╚██████╔╝██████╔╝╚██████╔╝   ██║   "
echo "  ╚═╝  ╚═╝╚═╝     ╚══════╝    ╚═╝  ╚═╝ ╚═════╝ ╚═════╝  ╚═════╝    ╚═╝   "
echo -e "${NC}"
echo -e "  ${BOLD}RPS Robot — Installer${NC}   |   TrickWing Toys"
echo -e "  Real-time gesture recognition + adaptive AI"
echo ""
line
echo ""
info "App will be installed to: $APP_DIR"
info "Data will be saved to:    $DATA_DIR"
echo ""

# ── Step 1: macOS check ───────────────────────────────────────────────────────
step "Step 1 — Checking system requirements"

if [ "$(uname -s)" != "Darwin" ]; then
    fail "This installer is for macOS only."
    exit 1
fi

MACOS_VERSION=$(sw_vers -productVersion)
MACOS_MAJOR=$(echo "$MACOS_VERSION" | cut -d. -f1)
if [ "$MACOS_MAJOR" -lt 11 ]; then
    warn "macOS $MACOS_VERSION — macOS 11 (Big Sur) or later recommended."
else
    ok "macOS $MACOS_VERSION"
fi

ARCH=$(uname -m)
[ "$ARCH" = "arm64" ] && ok "Apple Silicon (M-series chip)" || ok "Intel Mac"
info "A webcam is required. Built-in or USB webcam both work."
echo ""

# ── Step 2: Homebrew ──────────────────────────────────────────────────────────
step "Step 2 — Package manager (Homebrew)"

if command -v brew &>/dev/null; then
    ok "Homebrew already installed ($(brew --version | head -1))"
else
    info "Installing Homebrew — this will ask for your Mac password..."
    echo ""
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    if [ "$ARCH" = "arm64" ]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> "$HOME/.zprofile"
    fi
    ok "Homebrew installed"
fi

# ── Step 3: Git ───────────────────────────────────────────────────────────────
step "Step 3 — Git version control"

if command -v git &>/dev/null; then
    ok "Git already installed ($(git --version))"
else
    info "Installing Git via Homebrew..."
    brew install git
    ok "Git installed"
fi

# ── Step 4: Python ────────────────────────────────────────────────────────────
step "Step 4 — Python 3.9+"

PYTHON_BIN=""
for candidate in python3.9 python3.10 python3.11 python3.12; do
    if command -v "$candidate" &>/dev/null; then
        VER=$("$candidate" --version 2>&1 | awk '{print $2}')
        MAJOR=$(echo "$VER" | cut -d. -f1)
        MINOR=$(echo "$VER" | cut -d. -f2)
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 9 ]; then
            PYTHON_BIN=$(command -v "$candidate")
            ok "Found Python $VER at $PYTHON_BIN"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    info "Installing Python 3.9 via Homebrew..."
    brew install python@3.9
    PYTHON_BIN=$(brew --prefix)/bin/python3.9
    ok "Python 3.9 installed"
fi
echo ""

# ── Step 5: Clone from GitHub ─────────────────────────────────────────────────
step "Step 5 — Downloading RPS Robot from GitHub"

if [ -d "$APP_DIR/.git" ]; then
    # Already a git repo — just pull latest
    info "RPS Robot already installed — pulling latest version..."
    cd "$APP_DIR"
    git pull origin main
    ok "Up to date with GitHub"
elif [ -d "$APP_DIR" ]; then
    # Folder exists but not a git repo — back it up and clone fresh
    BACKUP="${APP_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
    warn "Existing folder found but not a git repo — backing up to $BACKUP"
    mv "$APP_DIR" "$BACKUP"
    info "Cloning from GitHub..."
    git clone "$GITHUB_REPO" "$APP_DIR"
    ok "Cloned to $APP_DIR"
else
    info "Cloning from $GITHUB_REPO ..."
    git clone "$GITHUB_REPO" "$APP_DIR"
    ok "Cloned to $APP_DIR"
fi

cd "$APP_DIR"
echo ""

# ── Step 6: Virtual environment ───────────────────────────────────────────────
step "Step 6 — Python virtual environment"

if [ -d "$VENV_DIR" ]; then
    warn "Existing .venv found — removing and recreating for clean install"
    rm -rf "$VENV_DIR"
fi

info "Creating virtual environment..."
"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet
ok "Virtual environment ready at .venv/"
echo ""

# ── Step 7: Python packages ───────────────────────────────────────────────────
step "Step 7 — Installing Python packages"
echo ""
info "This takes 3–8 minutes depending on your internet speed."
info "Total download: ~400MB"
echo ""

install_pkg() {
    local name="$1"
    local pkg="$2"
    echo -ne "  Installing ${BOLD}$name${NC}..."
    if pip install "$pkg" --quiet 2>/dev/null; then
        echo -e "\r  ${GREEN}✓${NC} $name                          "
    else
        echo -e "\r  ${RED}✗${NC} $name — retrying with output..."
        pip install "$pkg"
    fi
}

install_pkg "NumPy"            "numpy==1.26.4"
install_pkg "OpenCV"           "opencv-python==4.11.0.86"
install_pkg "MediaPipe"        "mediapipe==0.10.21"
install_pkg "scikit-learn"     "scikit-learn"
install_pkg "openpyxl"         "openpyxl"
install_pkg "Pillow"           "Pillow"
install_pkg "Vosk (speech)"    "vosk"
install_pkg "pyserial (ESP32)" "pyserial"
install_pkg "Anthropic (AI)"   "anthropic"
install_pkg "urllib3"          "urllib3"

echo ""
ok "All packages installed"
echo ""

# ── Step 8: Vosk speech model ─────────────────────────────────────────────────
step "Step 8 — Speech recognition model (Vosk)"

VOSK_MODEL_NAME="vosk-model-small-en-us-0.15"
VOSK_MODEL_DIR="$APP_DIR/$VOSK_MODEL_NAME"
VOSK_MODEL_URL="https://alphacephei.com/vosk/models/$VOSK_MODEL_NAME.zip"
VOSK_ZIP="$APP_DIR/$VOSK_MODEL_NAME.zip"

if [ -d "$VOSK_MODEL_DIR" ]; then
    ok "Vosk model already present — skipping download"
else
    info "Downloading Vosk US English model (~40MB)..."
    echo ""
    curl -L --progress-bar "$VOSK_MODEL_URL" -o "$VOSK_ZIP"
    echo ""
    info "Extracting model..."
    unzip -q "$VOSK_ZIP" -d "$APP_DIR"
    rm -f "$VOSK_ZIP"
    ok "Vosk model installed"
fi
echo ""

# ── Step 9: Data directory + Desktop shortcuts ────────────────────────────────
step "Step 9 — Data folder + Desktop shortcuts"

mkdir -p "$DATA_DIR" "$DATA_DIR/fingerprints" "$DATA_DIR/profiles" "$DATA_DIR/simulations"
ok "Data folder ready: $DATA_DIR"

# Desktop launcher
LAUNCHER_PATH="$DESKTOP/Launch RPS Robot.command"
cat > "$LAUNCHER_PATH" << LAUNCHER_SCRIPT
#!/bin/bash
# RPS Robot Launcher — double-click to start

cd "$APP_DIR"
source "$VENV_DIR/bin/activate"

echo ""
echo "  Starting RPS Robot..."
echo "  Press Ctrl+C to quit"
echo ""

python main.py

if [ \$? -ne 0 ]; then
    echo ""
    echo "  App exited with an error. Check above for details."
    read -n 1 -p "  Press any key to close..."
fi
LAUNCHER_SCRIPT

chmod +x "$LAUNCHER_PATH"
ok "Launcher created: 'Launch RPS Robot.command' on Desktop"

# Data folder shortcut
if [ ! -L "$DESKTOP/RPS Robot Data" ]; then
    ln -s "$DATA_DIR" "$DESKTOP/RPS Robot Data" 2>/dev/null || true
    ok "Data folder shortcut on Desktop"
fi
echo ""

# ── Step 10: Verify ───────────────────────────────────────────────────────────
step "Step 10 — Verifying installation"
echo ""

FAILED=0
verify() {
    local name="$1"
    local test="$2"
    echo -ne "  Checking ${name}..."
    if "$VENV_DIR/bin/python" -c "$test" 2>/dev/null; then
        echo -e "\r  ${GREEN}✓${NC} $name                          "
    else
        echo -e "\r  ${RED}✗${NC} $name — FAILED"
        FAILED=$((FAILED + 1))
    fi
}

verify "NumPy"        "import numpy"
verify "OpenCV"       "import cv2"
verify "MediaPipe"    "import mediapipe"
verify "scikit-learn" "import sklearn"
verify "openpyxl"     "import openpyxl"
verify "Pillow"       "from PIL import Image"
verify "Vosk"         "import vosk"
verify "pyserial"     "import serial"
verify "Anthropic"    "import anthropic"
verify "Git repo"     "import subprocess; subprocess.check_call(['git','-C','$APP_DIR','rev-parse'],capture_output=True)"
verify "Vosk model"   "from pathlib import Path; assert Path('$VOSK_MODEL_DIR').exists()"

echo ""
if [ "$FAILED" -gt 0 ]; then
    fail "$FAILED verification(s) failed."
    warn "Try: source $VENV_DIR/bin/activate && pip install -r $APP_DIR/requirements.txt"
else
    ok "All verifications passed"
fi
echo ""

# ── Step 11: ESP32 notice ─────────────────────────────────────────────────────
step "Step 11 — Optional: ESP32 Robot Arm"
echo ""
info "If you are using the physical RPS Robot arm (ESP32 microcontroller):"
echo ""
echo "  1. Install the CP210x USB driver:"
echo "     https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers"
echo "  2. Connect the ESP32 via USB"
echo "  3. In the app: press D → H, select port, press ENTER"
echo ""
info "The app works fully without the ESP32 — this is optional."
echo ""

# ── Done ──────────────────────────────────────────────────────────────────────
line
echo ""
echo -e "${BOLD}${GREEN}  ✓  Installation complete!${NC}"
echo ""
echo -e "  ${BOLD}To launch:${NC}"
echo -e "  → Double-click ${BOLD}'Launch RPS Robot.command'${NC} on your Desktop"
echo ""
echo -e "  ${BOLD}Auto-updates:${NC}"
echo -e "  → The app checks GitHub on every launch"
echo -e "  → A yellow banner appears in the menu when an update is ready"
echo -e "  → Press ${BOLD}U${NC} to update and restart automatically"
echo ""
echo -e "  ${BOLD}Your data:${NC} $DATA_DIR"
echo ""
line
echo ""

read -p "  Launch RPS Robot now? [y/N] " -n 1 -r LAUNCH_NOW
echo ""
if [[ "$LAUNCH_NOW" =~ ^[Yy]$ ]]; then
    echo ""
    info "Starting RPS Robot..."
    cd "$APP_DIR"
    python main.py
fi

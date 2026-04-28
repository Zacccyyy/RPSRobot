#!/bin/bash
# =============================================================================
#  RPS Robot — Installer
#  TrickWing Toys / RavensAgency
#
#  Run this once:  bash install.sh
#
#  What it does:
#    1. Checks macOS + hardware requirements
#    2. Installs Homebrew if needed
#    3. Installs Python 3.9 if needed
#    4. Creates a Python virtual environment inside the app folder
#    5. Installs all Python packages
#    6. Downloads the Vosk speech recognition model
#    7. Creates Desktop shortcuts (launcher + data folder)
#    8. Verifies everything works
#    9. Launches the app
# =============================================================================

set -euo pipefail

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

# ── Locate the installer directory ───────────────────────────────────────────
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
APP_DIR="$SCRIPT_DIR"
VENV_DIR="$APP_DIR/.venv"
DATA_DIR="$HOME/Desktop/CapStone"
DESKTOP="$HOME/Desktop"

info "App folder: $APP_DIR"
info "Data folder: $DATA_DIR"
echo ""

# ── Step 1: macOS check ───────────────────────────────────────────────────────
step "Step 1 — Checking system requirements"

OS=$(uname -s)
if [ "$OS" != "Darwin" ]; then
    fail "This installer is for macOS only."
    fail "Detected: $OS"
    exit 1
fi

MACOS_VERSION=$(sw_vers -productVersion)
MACOS_MAJOR=$(echo "$MACOS_VERSION" | cut -d. -f1)
if [ "$MACOS_MAJOR" -lt 11 ]; then
    warn "macOS $MACOS_VERSION detected. macOS 11 (Big Sur) or later recommended."
    warn "Some features may not work on older versions."
else
    ok "macOS $MACOS_VERSION"
fi

# Check architecture
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    ok "Apple Silicon (M-series chip) — optimal performance"
    PYTHON_FORMULA="python@3.9"
else
    ok "Intel Mac"
    PYTHON_FORMULA="python@3.9"
fi

# Check webcam (can't easily verify from shell, just inform)
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

    # Add brew to PATH for Apple Silicon
    if [ "$ARCH" = "arm64" ]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> "$HOME/.zprofile"
    fi
    ok "Homebrew installed"
fi

# ── Step 3: Python 3.9 ───────────────────────────────────────────────────────
step "Step 3 — Python 3.9"

# Try existing pythons first
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

PYTHON_VERSION=$("$PYTHON_BIN" --version)
ok "Using $PYTHON_VERSION"
echo ""

# ── Step 4: Virtual environment ───────────────────────────────────────────────
step "Step 4 — Python virtual environment"

if [ -d "$VENV_DIR" ]; then
    warn "Existing .venv found — removing and recreating for clean install"
    rm -rf "$VENV_DIR"
fi

info "Creating virtual environment at .venv/ ..."
"$PYTHON_BIN" -m venv "$VENV_DIR"
ok "Virtual environment created"

# Activate it
source "$VENV_DIR/bin/activate"
ok "Virtual environment activated"

# Upgrade pip silently
info "Upgrading pip..."
pip install --upgrade pip --quiet
ok "pip up to date"
echo ""

# ── Step 5: Python packages ───────────────────────────────────────────────────
step "Step 5 — Installing Python packages"
echo ""
info "This will take 3–8 minutes depending on your internet speed."
info "Total download: ~400MB"
echo ""

install_pkg() {
    local name="$1"
    local pkg="$2"
    echo -ne "  Installing ${BOLD}$name${NC}..."
    if pip install "$pkg" --quiet 2>/dev/null; then
        echo -e "\r  ${GREEN}✓${NC} $name                          "
    else
        echo -e "\r  ${RED}✗${NC} $name — retrying with verbose..."
        pip install "$pkg"
    fi
}

install_pkg "NumPy"              "numpy==1.26.4"
install_pkg "OpenCV"             "opencv-python==4.11.0.86"
install_pkg "MediaPipe"          "mediapipe==0.10.21"
install_pkg "scikit-learn"       "scikit-learn"
install_pkg "openpyxl"           "openpyxl"
install_pkg "Pillow"             "Pillow"
install_pkg "Vosk (speech)"      "vosk"
install_pkg "pyserial (ESP32)"   "pyserial"
install_pkg "Anthropic (AI)"     "anthropic"
install_pkg "urllib3"            "urllib3"

echo ""
ok "All packages installed"
echo ""

# ── Step 6: Vosk speech model ─────────────────────────────────────────────────
step "Step 6 — Speech recognition model (Vosk)"

VOSK_MODEL_NAME="vosk-model-small-en-us-0.15"
VOSK_MODEL_DIR="$APP_DIR/$VOSK_MODEL_NAME"
VOSK_MODEL_URL="https://alphacephei.com/vosk/models/$VOSK_MODEL_NAME.zip"
VOSK_ZIP="$APP_DIR/$VOSK_MODEL_NAME.zip"

if [ -d "$VOSK_MODEL_DIR" ]; then
    ok "Vosk model already present — skipping download"
else
    info "Downloading Vosk US English model (~40MB)..."
    info "URL: $VOSK_MODEL_URL"
    echo ""

    # Download with progress
    if command -v curl &>/dev/null; then
        curl -L --progress-bar "$VOSK_MODEL_URL" -o "$VOSK_ZIP"
    else
        wget --show-progress -q "$VOSK_MODEL_URL" -O "$VOSK_ZIP"
    fi

    echo ""
    info "Extracting model..."
    unzip -q "$VOSK_ZIP" -d "$APP_DIR"
    rm -f "$VOSK_ZIP"
    ok "Vosk model installed at $VOSK_MODEL_DIR"
fi
echo ""

# ── Step 7: Data directory ────────────────────────────────────────────────────
step "Step 7 — Data folder setup"

mkdir -p "$DATA_DIR"
mkdir -p "$DATA_DIR/fingerprints"
mkdir -p "$DATA_DIR/profiles"
mkdir -p "$DATA_DIR/simulations"
ok "Data folder ready: $DATA_DIR"

# Copy default config if not present
if [ ! -f "$DATA_DIR/config.json" ]; then
    if [ -f "$APP_DIR/config.json" ]; then
        cp "$APP_DIR/config.json" "$DATA_DIR/config.json"
        ok "Default config copied"
    fi
fi
echo ""

# ── Step 8: Desktop launcher ──────────────────────────────────────────────────
step "Step 8 — Desktop launcher"

LAUNCHER_PATH="$DESKTOP/Launch RPS Robot.command"

cat > "$LAUNCHER_PATH" << LAUNCHER_SCRIPT
#!/bin/bash
# RPS Robot Launcher
# Double-click this file to start the app

cd "$APP_DIR"
source "$VENV_DIR/bin/activate"

echo ""
echo "  Starting RPS Robot..."
echo "  Press Ctrl+C to quit"
echo ""

python main.py

# Keep terminal open if the app crashes
if [ \$? -ne 0 ]; then
    echo ""
    echo "  App exited with an error. Check above for details."
    echo "  Press any key to close this window."
    read -n 1
fi
LAUNCHER_SCRIPT

chmod +x "$LAUNCHER_PATH"
ok "Desktop launcher created: 'Launch RPS Robot.command'"

# Also create a data folder shortcut
if [ ! -L "$DESKTOP/RPS Robot Data" ]; then
    ln -s "$DATA_DIR" "$DESKTOP/RPS Robot Data" 2>/dev/null || true
    ok "Desktop shortcut to data folder created"
fi
echo ""

# ── Step 9: Verification ──────────────────────────────────────────────────────
step "Step 9 — Verifying installation"

echo ""
FAILED=0

verify() {
    local name="$1"
    local import_test="$2"
    echo -ne "  Checking ${name}..."
    if "$VENV_DIR/bin/python" -c "$import_test" 2>/dev/null; then
        echo -e "\r  ${GREEN}✓${NC} $name                          "
    else
        echo -e "\r  ${RED}✗${NC} $name — FAILED"
        FAILED=$((FAILED + 1))
    fi
}

verify "NumPy"       "import numpy; assert numpy.__version__ >= '1.24'"
verify "OpenCV"      "import cv2"
verify "MediaPipe"   "import mediapipe"
verify "scikit-learn" "import sklearn"
verify "openpyxl"    "import openpyxl"
verify "Pillow"      "from PIL import Image"
verify "Vosk"        "import vosk"
verify "pyserial"    "import serial"
verify "Anthropic"   "import anthropic"
verify "Vosk model"  "from pathlib import Path; assert Path('$VOSK_MODEL_DIR').exists(), 'model not found'"

echo ""

if [ "$FAILED" -gt 0 ]; then
    fail "$FAILED verification(s) failed. See errors above."
    warn "Try running: source $VENV_DIR/bin/activate && pip install -r requirements.txt"
    echo ""
else
    ok "All verifications passed"
fi
echo ""

# ── Step 10: ESP32 driver notice ──────────────────────────────────────────────
step "Step 10 — Optional: ESP32 Hardware"

echo ""
info "If you are using the physical RPS Robot arm (ESP32 microcontroller):"
echo ""
echo "  1. Install the CP210x USB driver from Silicon Labs:"
echo "     https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers"
echo ""
echo "  2. Connect the ESP32 via USB"
echo ""
echo "  3. In the app: press D (Diagnostic) → H (Hardware Test)"
echo "     Use [ ] to select port, ENTER to connect"
echo ""
info "The app runs fully without the ESP32 — this is optional."
echo ""

# ── Done ──────────────────────────────────────────────────────────────────────
line
echo ""
echo -e "${BOLD}${GREEN}  ✓  Installation complete!${NC}"
echo ""
echo -e "  To launch the app:"
echo -e "  ${BOLD}→  Double-click 'Launch RPS Robot.command' on your Desktop${NC}"
echo ""
echo -e "  Or from Terminal:"
echo -e "  ${CYAN}  cd $APP_DIR && source .venv/bin/activate && python main.py${NC}"
echo ""
echo -e "  Your data is saved to:"
echo -e "  ${CYAN}  $DATA_DIR${NC}"
echo ""
line
echo ""

# Offer to launch now
read -p "  Launch RPS Robot now? [y/N] " -n 1 -r LAUNCH_NOW
echo ""
if [[ "$LAUNCH_NOW" =~ ^[Yy]$ ]]; then
    echo ""
    info "Starting RPS Robot..."
    cd "$APP_DIR"
    python main.py
fi

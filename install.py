#!/usr/bin/env python3
"""
install.py — RPS Robot Cross-Platform Installer
================================================
TrickWing Toys / RavensAgency

This is the ONLY file you need to download.
Run it once with:

    macOS / Linux:   python3 install.py
    Windows:         python install.py

What it does:
  1. Checks OS and Python version
  2. Installs Git if needed (macOS via Homebrew, Windows via winget)
  3. Clones the RPS Robot repo from GitHub (enables auto-updates)
  4. Creates a Python virtual environment
  5. Installs all Python packages
  6. Downloads the Vosk speech recognition model
  7. Creates a Desktop launcher (.command on macOS, .bat on Windows)
  8. Verifies everything works
  9. Optionally launches the app
"""

import os
import sys
import subprocess
import platform
import shutil
import urllib.request
import zipfile
import pathlib
import textwrap

# ── Config ────────────────────────────────────────────────────────────────────
GITHUB_REPO   = "https://github.com/Zacccyyy/RPSRobot.git"
APP_DIR       = pathlib.Path.home() / "rps_hand_counter"
VENV_DIR      = APP_DIR / ".venv"
DATA_DIR      = pathlib.Path.home() / "Desktop" / "CapStone"
DESKTOP       = pathlib.Path.home() / "Desktop"

VOSK_MODEL    = "vosk-model-small-en-us-0.15"
VOSK_URL      = f"https://alphacephei.com/vosk/models/{VOSK_MODEL}.zip"

PACKAGES = [
    ("NumPy",             "numpy>=1.26.4,<2.0"),
    ("OpenCV",            "opencv-python>=4.8.0"),
    ("MediaPipe",         "mediapipe>=0.10.9"),
    ("scikit-learn",      "scikit-learn>=1.3.0"),
    ("openpyxl",          "openpyxl>=3.1.0"),
    ("Pillow",            "Pillow>=10.0.0"),
    ("Vosk (speech)",     "vosk>=0.3.45"),
    ("pyserial (ESP32)",  "pyserial>=3.5"),
    ("Anthropic (AI)",    "anthropic>=0.25.0"),
    ("urllib3",           "urllib3>=2.0.0"),
    ("Sentry (reporting)","sentry-sdk>=2.0.0"),
]

# ── Platform ──────────────────────────────────────────────────────────────────
IS_MAC     = sys.platform == "darwin"
IS_WIN     = sys.platform == "win32"
IS_LINUX   = sys.platform.startswith("linux")
OS_NAME    = "macOS" if IS_MAC else ("Windows" if IS_WIN else "Linux")

# ── Colours (disabled on Windows cmd where ANSI isn't always supported) ───────
_USE_COLOR = IS_MAC or IS_LINUX or os.environ.get("TERM") == "xterm-256color"

def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

def ok(msg):    print(_c("32", f"  ✓  {msg}"))
def info(msg):  print(_c("36", f"  →  {msg}"))
def warn(msg):  print(_c("33", f"  ⚠  {msg}"))
def fail(msg):  print(_c("31", f"  ✗  {msg}"))
def step(msg):  print(_c("1;36", f"\n━━━  {msg}  {'━' * max(0, 44 - len(msg))}"))
def line():     print(_c("36",  "─" * 50))
def bold(msg):  return _c("1", msg)


# ── Helpers ───────────────────────────────────────────────────────────────────

def run(cmd, **kwargs):
    """Run a command, raise on failure."""
    return subprocess.run(cmd, check=True, **kwargs)

def run_quiet(cmd):
    """Run silently, return True on success."""
    try:
        subprocess.run(cmd, check=True,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def command_exists(cmd):
    return shutil.which(cmd) is not None

def venv_python():
    """Path to the venv Python executable."""
    if IS_WIN:
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"

def venv_pip():
    if IS_WIN:
        return VENV_DIR / "Scripts" / "pip.exe"
    return VENV_DIR / "bin" / "pip"


# ── Banner ────────────────────────────────────────────────────────────────────

def print_banner():
    print()
    print(_c("1;36", "  ██████╗ ██████╗ ███████╗    ██████╗  ██████╗ ██████╗  ██████╗ ████████╗"))
    print(_c("1;36", "  ██╔══██╗██╔══██╗██╔════╝    ██╔══██╗██╔═══██╗██╔══██╗██╔═══██╗╚══██╔══╝"))
    print(_c("1;36", "  ██████╔╝██████╔╝███████╗    ██████╔╝██║   ██║██████╔╝██║   ██║   ██║   "))
    print(_c("1;36", "  ██╔══██╗██╔═══╝ ╚════██║    ██╔══██╗██║   ██║██╔══██╗██║   ██║   ██║   "))
    print(_c("1;36", "  ██║  ██║██║     ███████║    ██║  ██║╚██████╔╝██████╔╝╚██████╔╝   ██║   "))
    print(_c("1;36", "  ╚═╝  ╚═╝╚═╝     ╚══════╝    ╚═╝  ╚═╝ ╚═════╝ ╚═════╝  ╚═════╝    ╚═╝   "))
    print()
    print(f"  {bold('RPS Robot — Installer')}   |   TrickWing Toys")
    print(f"  Real-time gesture recognition + adaptive AI")
    print()
    line()
    print()
    info(f"Platform:    {OS_NAME} ({platform.machine()})")
    info(f"Install to:  {APP_DIR}")
    info(f"Data folder: {DATA_DIR}")
    print()


# ── Step 1: System check ──────────────────────────────────────────────────────

def _install_python312_windows():
    """
    Install Python 3.12 on Windows via winget and re-launch this installer
    using the newly installed Python 3.12.
    """
    # Try to find Python 3.12 already installed first
    for candidate in ["py", "python3.12", "python"]:
        try:
            result = subprocess.run(
                [candidate, "-c",
                 "import sys; v=sys.version_info; print(v.major,v.minor)"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split()
                if len(parts) == 2 and int(parts[0]) == 3 and int(parts[1]) == 12:
                    py312 = shutil.which(candidate)
                    if py312:
                        info(f"Found Python 3.12 at {py312} - relaunching...")
                        os.execv(py312, [py312] + sys.argv)
        except Exception:
            continue

    # Not found - install via winget
    if command_exists("winget"):
        info("Installing Python 3.12 via winget...")
        try:
            run(["winget", "install", "--id", "Python.Python.3.12",
                 "-e", "--source", "winget",
                 "--accept-package-agreements",
                 "--accept-source-agreements"])
            ok("Python 3.12 installed")
        except Exception as e:
            fail(f"Could not install Python 3.12 automatically: {e}")
            fail("Please install Python 3.12 manually from:")
            fail("  https://www.python.org/downloads/release/python-3129/")
            fail("Make sure to tick 'Add Python to PATH' during install.")
            fail("Then re-run:  python install.py")
            sys.exit(1)
    else:
        fail("Cannot install Python 3.12 automatically (winget not available).")
        fail("Please install Python 3.12 from:")
        fail("  https://www.python.org/downloads/release/python-3129/")
        fail("Make sure to tick 'Add Python to PATH' during install.")
        fail("Then re-run:  python install.py")
        sys.exit(1)

    # Find and re-launch with Python 3.12
    py312_paths = [
        r"C:\Users\{}\AppData\Local\Programs\Python\Python312\python.exe".format(
            os.environ.get("USERNAME", "user")),
        r"C:\Program Files\Python312\python.exe",
    ]
    for p in py312_paths:
        if pathlib.Path(p).exists():
            info(f"Relaunching installer with Python 3.12...")
            print()
            os.execv(p, [p] + sys.argv)

    # If we can't find it, ask user to rerun manually
    warn("Python 3.12 installed. Please close this window and run:")
    warn("  python install.py")
    warn("(Python 3.12 will be used automatically)")
    sys.exit(0)


def check_system():
    step("Step 1 — Checking system requirements")

    if IS_WIN:
        win_ver = platform.version()
        win_rel = platform.release()
        if int(platform.release()) < 10:
            warn(f"Windows {win_rel} detected. Windows 10 or later recommended.")
        else:
            ok(f"Windows {win_rel} ({win_ver})")

    elif IS_MAC:
        mac_ver = platform.mac_ver()[0]
        major   = int(mac_ver.split(".")[0]) if mac_ver else 0
        if major < 11:
            warn(f"macOS {mac_ver} — macOS 11 (Big Sur) or later recommended.")
        else:
            ok(f"macOS {mac_ver}")

    else:
        ok(f"Linux ({platform.release()})")

    # Python version check
    # MediaPipe only supports Python 3.9 - 3.12.
    # Python 3.13+ breaks MediaPipe due to ABI changes in CPython.
    py = sys.version_info
    if py < (3, 9):
        fail(f"Python {py.major}.{py.minor} detected. Python 3.9+ is required.")
        fail("Download from https://www.python.org/downloads/")
        sys.exit(1)
    elif py >= (3, 13):
        warn(f"Python {py.major}.{py.minor} detected.")
        warn("MediaPipe does not support Python 3.13 or later yet.")
        warn("Python 3.12 is required. Installing it now...")
        print()
        if IS_WIN:
            _install_python312_windows()
        else:
            fail("Please install Python 3.12 from https://www.python.org/downloads/")
            fail("Then re-run this installer using:  python3.12 install.py")
            sys.exit(1)
    else:
        ok(f"Python {py.major}.{py.minor}.{py.micro} (compatible)")

    info("A webcam is required. Built-in or USB webcam both work.")
    print()


# ── Step 2: Git ───────────────────────────────────────────────────────────────

def _find_git_windows():
    """
    Find git.exe on Windows after a fresh install.
    winget installs Git to a predictable location that isn't on PATH yet
    until the terminal is restarted. We check common install paths directly.
    """
    common_paths = [
        r"C:\Program Files\Git\cmd\git.exe",
        r"C:\Program Files (x86)\Git\cmd\git.exe",
        pathlib.Path.home() / "AppData" / "Local" / "Programs" / "Git" / "cmd" / "git.exe",
    ]
    for p in common_paths:
        if pathlib.Path(p).exists():
            return str(p)
    # Also try refreshing PATH from registry
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment") as key:
            path_val, _ = winreg.QueryValueEx(key, "Path")
            for part in path_val.split(";"):
                candidate = pathlib.Path(part.strip()) / "git.exe"
                if candidate.exists():
                    return str(candidate)
    except Exception:
        pass
    return None


def ensure_git():
    step("Step 2 — Git version control")

    if command_exists("git"):
        ver = subprocess.check_output(["git", "--version"],
                                      text=True).strip()
        ok(f"{ver}")
        return

    info("Git not found — installing...")

    if IS_MAC:
        if not command_exists("brew"):
            info("Installing Homebrew first...")
            run(["/bin/bash", "-c",
                 "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"])
        run(["brew", "install", "git"])
        ok("Git installed via Homebrew")

    elif IS_WIN:
        if command_exists("winget"):
            info("Installing Git via winget...")
            run(["winget", "install", "--id", "Git.Git",
                 "-e", "--source", "winget",
                 "--accept-package-agreements",
                 "--accept-source-agreements"])

            # winget installs Git but the current session PATH isn't updated yet.
            # Find git.exe directly and add it to os.environ["PATH"] for this session.
            git_exe = _find_git_windows()
            if git_exe:
                git_dir = str(pathlib.Path(git_exe).parent)
                os.environ["PATH"] = git_dir + os.pathsep + os.environ.get("PATH", "")
                ok(f"Git installed and located at {git_exe}")
            else:
                warn("Git installed but could not locate git.exe automatically.")
                warn("Please close this window, open a new Command Prompt, and re-run install.py")
                sys.exit(0)
        else:
            fail("Git not found and winget is not available.")
            fail("Please install Git manually from: https://git-scm.com/download/win")
            fail("Then re-run this installer.")
            sys.exit(1)

    else:
        fail("Git not found. Install with: sudo apt install git")
        sys.exit(1)

    print()


# ── Step 3: Clone / update from GitHub ───────────────────────────────────────

def clone_or_update():
    step("Step 3 — Downloading RPS Robot from GitHub")

    git_dir = APP_DIR / ".git"

    if git_dir.exists():
        info("RPS Robot already installed — pulling latest...")
        run(["git", "-C", str(APP_DIR), "pull", "origin", "main"])
        ok("Up to date with GitHub")

    elif APP_DIR.exists():
        backup = APP_DIR.parent / f"rps_hand_counter_backup"
        warn(f"Folder exists but is not a git repo.")
        warn(f"Backing up to {backup} and cloning fresh...")
        APP_DIR.rename(backup)
        run(["git", "clone", GITHUB_REPO, str(APP_DIR)])
        ok(f"Cloned to {APP_DIR}")

    else:
        info(f"Cloning from {GITHUB_REPO} ...")
        run(["git", "clone", GITHUB_REPO, str(APP_DIR)])
        ok(f"Cloned to {APP_DIR}")

    print()


# ── Step 4: Virtual environment ───────────────────────────────────────────────

def create_venv():
    step("Step 4 — Python virtual environment")

    if VENV_DIR.exists():
        warn("Existing .venv found — recreating for clean install...")
        shutil.rmtree(VENV_DIR)

    info("Creating virtual environment...")
    run([sys.executable, "-m", "venv", str(VENV_DIR)])

    # Upgrade pip
    run_quiet([str(venv_python()), "-m", "pip", "install",
               "--upgrade", "pip", "--quiet"])
    ok(f"Virtual environment ready")
    print()


# ── Step 5: Install packages ──────────────────────────────────────────────────

def install_packages():
    step("Step 5 — Installing Python packages")
    print()
    info("This takes 3–8 minutes. Total download: ~400MB")
    print()

    failed = []
    for name, pkg in PACKAGES:
        print(f"  Installing {bold(name)}...", end="", flush=True)
        result = subprocess.run(
            [str(venv_pip()), "install", pkg, "--quiet"],
            capture_output=True
        )
        if result.returncode == 0:
            print(f"\r  {_c('32','✓')} {name}                          ")
        else:
            print(f"\r  {_c('31','✗')} {name} — retrying...")
            result2 = subprocess.run(
                [str(venv_pip()), "install", pkg])
            if result2.returncode != 0:
                failed.append(name)

    print()
    if failed:
        warn(f"Failed to install: {', '.join(failed)}")
    else:
        ok("All packages installed")
    print()


# ── Step 6: Vosk speech model ─────────────────────────────────────────────────

def install_vosk_model():
    step("Step 6 — Speech recognition model (Vosk)")

    model_dir = APP_DIR / VOSK_MODEL
    zip_path  = APP_DIR / f"{VOSK_MODEL}.zip"

    if model_dir.exists():
        ok("Vosk model already present — skipping download")
        print()
        return

    info(f"Downloading Vosk US English model (~40MB)...")
    info(f"URL: {VOSK_URL}")
    print()

    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(100, int(block_num * block_size * 100 / total_size))
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r  [{bar}] {pct}%", end="", flush=True)

    urllib.request.urlretrieve(VOSK_URL, zip_path, _progress)
    print()

    info("Extracting model...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(APP_DIR)
    zip_path.unlink()
    ok(f"Vosk model installed")
    print()


# ── Step 7: Data directory + launcher ─────────────────────────────────────────

def setup_data_and_launcher():
    step("Step 7 — Data folder + Desktop launcher")

    # Data folders
    for subdir in ["", "fingerprints", "profiles", "simulations"]:
        (DATA_DIR / subdir).mkdir(parents=True, exist_ok=True)
    ok(f"Data folder: {DATA_DIR}")

    # Desktop launcher
    if IS_WIN:
        _create_windows_launcher()
    elif IS_MAC:
        _create_mac_launcher()
    else:
        _create_linux_launcher()

    print()


def _create_mac_launcher():
    launcher = DESKTOP / "Launch RPS Robot.command"
    launcher.write_text(textwrap.dedent(f"""\
        #!/bin/bash
        cd "{APP_DIR}"
        source "{VENV_DIR}/bin/activate"
        echo ""
        echo "  Starting RPS Robot..."
        echo "  Press Ctrl+C to quit"
        echo ""
        python main.py
        if [ $? -ne 0 ]; then
            echo ""
            echo "  App exited with an error."
            read -n 1 -p "  Press any key to close..."
        fi
    """))
    launcher.chmod(0o755)
    ok(f"Launcher: 'Launch RPS Robot.command' on Desktop")

    # Data folder symlink
    symlink = DESKTOP / "RPS Robot Data"
    if not symlink.exists():
        try:
            symlink.symlink_to(DATA_DIR)
            ok("Data folder shortcut on Desktop")
        except Exception:
            pass


def _create_windows_launcher():
    launcher = DESKTOP / "Launch RPS Robot.bat"
    pip_path  = VENV_DIR / "Scripts" / "python.exe"
    launcher.write_text(textwrap.dedent(f"""\
        @echo off
        cd /d "{APP_DIR}"
        call "{VENV_DIR}\\Scripts\\activate.bat"
        echo.
        echo   Starting RPS Robot...
        echo   Press Ctrl+C to quit
        echo.
        python main.py
        if %ERRORLEVEL% neq 0 (
            echo.
            echo   App exited with an error. Check above for details.
            pause
        )
    """))
    ok(f"Launcher: 'Launch RPS Robot.bat' on Desktop")

    # Data folder shortcut
    shortcut = DESKTOP / "RPS Robot Data.lnk"
    if not shortcut.exists():
        try:
            # Use PowerShell to create a proper Windows shortcut
            ps_cmd = (
                f'$ws = New-Object -ComObject WScript.Shell; '
                f'$s = $ws.CreateShortcut("{shortcut}"); '
                f'$s.TargetPath = "{DATA_DIR}"; '
                f'$s.Save()'
            )
            subprocess.run(["powershell", "-Command", ps_cmd],
                           capture_output=True)
            ok("Data folder shortcut on Desktop")
        except Exception:
            pass


def _create_linux_launcher():
    launcher = DESKTOP / "Launch RPS Robot.sh"
    launcher.write_text(textwrap.dedent(f"""\
        #!/bin/bash
        cd "{APP_DIR}"
        source "{VENV_DIR}/bin/activate"
        python main.py
    """))
    launcher.chmod(0o755)
    ok(f"Launcher: 'Launch RPS Robot.sh' on Desktop")


# ── Step 8: Verify ────────────────────────────────────────────────────────────

def verify_installation():
    step("Step 8 — Verifying installation")
    print()

    checks = [
        ("NumPy",        "import numpy"),
        ("OpenCV",       "import cv2"),
        ("MediaPipe",    "import mediapipe"),
        ("scikit-learn", "import sklearn"),
        ("openpyxl",     "import openpyxl"),
        ("Pillow",       "from PIL import Image"),
        ("Vosk",         "import vosk"),
        ("pyserial",     "import serial"),
        ("Anthropic",    "import anthropic"),
        ("Git repo",     f"import subprocess; subprocess.check_call("
                         f"['git','-C',r'{APP_DIR}','rev-parse'],"
                         f"capture_output=True)"),
        ("Vosk model",   f"from pathlib import Path; "
                         f"assert Path(r'{APP_DIR / VOSK_MODEL}').exists()"),
    ]

    failed = 0
    for name, test in checks:
        print(f"  Checking {name}...", end="", flush=True)
        result = subprocess.run(
            [str(venv_python()), "-c", test],
            capture_output=True
        )
        if result.returncode == 0:
            print(f"\r  {_c('32','✓')} {name}                    ")
        else:
            print(f"\r  {_c('31','✗')} {name} — FAILED")
            failed += 1

    print()
    if failed:
        warn(f"{failed} verification(s) failed.")
        warn(f"Try: {venv_pip()} install -r {APP_DIR / 'requirements.txt'}")
    else:
        ok("All verifications passed")
    print()
    return failed == 0


# ── Step 9: ESP32 notice ──────────────────────────────────────────────────────

def print_esp32_notice():
    step("Step 9 — Optional: ESP32 Robot Arm")
    print()
    info("If you are using the physical RPS Robot arm (ESP32):")
    print()
    if IS_WIN:
        print("  1. Install the CP210x USB driver:")
        print("     https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers")
    else:
        print("  1. Install the CP210x USB driver:")
        print("     https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers")
    print("  2. Connect the ESP32 via USB")
    print("  3. In the app: press D -> H, select port, press ENTER")
    print()
    info("The app works fully without the ESP32 — this is optional.")
    print()


# ── Done ──────────────────────────────────────────────────────────────────────

def print_done():
    line()
    print()
    print(_c("1;32", "  ✓  Installation complete!"))
    print()

    if IS_WIN:
        launcher_name = "Launch RPS Robot.bat"
    elif IS_MAC:
        launcher_name = "Launch RPS Robot.command"
    else:
        launcher_name = "Launch RPS Robot.sh"

    print(f"  {bold('To launch:')}")
    print(f"  → Double-click '{bold(launcher_name)}' on your Desktop")
    print()
    print(f"  {bold('Auto-updates:')}")
    print(f"  → The app checks GitHub on every launch")
    print(f"  → A yellow banner appears in the menu when an update is ready")
    print(f"  → Press {bold('U')} to update and restart automatically")
    print()
    print(f"  {bold('Your data:')} {DATA_DIR}")
    print()
    line()
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print_banner()
    check_system()
    ensure_git()
    clone_or_update()
    create_venv()
    install_packages()
    install_vosk_model()
    setup_data_and_launcher()
    ok_all = verify_installation()
    print_esp32_notice()
    print_done()

    # Offer to launch
    try:
        answer = input("  Launch RPS Robot now? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"

    if answer == "y":
        print()
        info("Starting RPS Robot...")
        os.chdir(APP_DIR)
        os.execv(str(venv_python()),
                 [str(venv_python()), "main.py"])


if __name__ == "__main__":
    main()

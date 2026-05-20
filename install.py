#!/usr/bin/env python3
"""
install.py — RPS Robot Cross-Platform Installer
================================================
TrickWing Toys / RavensAgency

This is the ONLY file the user needs to download and run.
It sets up the entire app from scratch on macOS, Windows, or Linux.

    macOS / Linux:   python3 install.py
    Windows:         python install.py

What it does, in order:
  1. Checks the OS and Python version (requires 3.9–3.12; auto-installs 3.12 if needed)
  2. Installs Git if it's not already present
  3. Clones the RPS Robot repo from GitHub (enables auto-updates via git pull)
  4. Creates a Python virtual environment so packages don't pollute the system
  5. Installs all required Python packages into that venv
  6. Downloads the Vosk speech recognition model (~40 MB)
  7. Creates the CapStone data folder and a Desktop launcher shortcut
  8. Verifies every package can be imported successfully
  9. Prints optional ESP32 robot-arm setup instructions
 10. Asks whether to launch the app immediately
"""

import io
import os
import sys
import struct
import subprocess
import platform
import shutil
import urllib.request
import zipfile
import pathlib
import textwrap

# ── Configuration ─────────────────────────────────────────────────────────────
# Central place to change the repo URL or install locations.
GITHUB_REPO = "https://github.com/Zacccyyy/RPSRobot.git"
APP_DIR     = pathlib.Path.home() / "rps_hand_counter"  # where the repo lives
VENV_DIR    = APP_DIR / ".venv"                          # virtual environment inside the repo

# Vosk speech recognition model — small US English (~40 MB download)
VOSK_MODEL  = "vosk-model-small-en-us-0.15"
VOSK_URL    = f"https://alphacephei.com/vosk/models/{VOSK_MODEL}.zip"

# All Python packages the app needs.  Using (display_name, pip_spec) pairs so we
# can show a friendly name in progress output while still passing the exact version
# constraint to pip.
PACKAGES = [
    ("NumPy",             "numpy>=1.26.4,<2.0"),
    ("OpenCV",            "opencv-python>=4.8.0"),
    ("MediaPipe",         "mediapipe>=0.10.9,<=0.10.21"),
    ("scikit-learn",      "scikit-learn>=1.3.0"),
    ("openpyxl",          "openpyxl>=3.1.0"),
    ("Pillow",            "Pillow>=10.0.0"),
    ("Vosk (speech)",     "vosk>=0.3.45"),
    ("pyserial (ESP32)",  "pyserial>=3.5"),
    ("Anthropic (AI)",    "anthropic>=0.25.0"),
    ("urllib3",           "urllib3>=2.0.0"),
    ("Sentry",            "sentry-sdk>=2.0.0"),
    ("Bleak (BLE)",       "bleak>=0.21.0"),
]

# ── Platform detection ────────────────────────────────────────────────────────
# Checked once at import time so every function can just read these booleans.
IS_MAC   = sys.platform == "darwin"
IS_WIN   = sys.platform == "win32"
IS_LINUX = sys.platform.startswith("linux")
OS_NAME  = "macOS" if IS_MAC else ("Windows" if IS_WIN else "Linux")

# ── Terminal colours ──────────────────────────────────────────────────────────
# ANSI colour codes work on macOS/Linux and modern Windows Terminal.
# Older Windows CMD doesn't support them, so we disable colour there.
_USE_COLOR = IS_MAC or IS_LINUX or os.environ.get("TERM") == "xterm-256color"

def _c(code, text):
    """Wrap text in an ANSI colour escape, or return it plain if colour is off."""
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

# Status symbols — use ASCII on Windows CMD because Unicode can render wrong there
_OK   = "[OK]"  if IS_WIN else "  ok "
_FAIL = "[!!]"  if IS_WIN else "  !! "
_WARN = "[??]"  if IS_WIN else "  ?? "
_ARR  = "  ->  "  # arrow prefix, same on all platforms

# Shortcut print helpers — one call per message rather than repeating _c() everywhere
def ok(msg):   print(_c("32",   f"{_OK}  {msg}"))    # green:  success
def info(msg): print(_c("36",   f"{_ARR} {msg}"))    # cyan:   informational
def warn(msg): print(_c("33",   f"{_WARN} {msg}"))   # yellow: non-fatal warning
def fail(msg): print(_c("31",   f"{_FAIL} {msg}"))   # red:    error
def step(msg): print(_c("1;36", f"\n---  {msg}  {'-' * max(0, 44 - len(msg))}"))  # bold section header
def line():    print(_c("36",   "-" * 50))            # divider line
def bold(msg): return _c("1", msg)                    # bold text (returns string, doesn't print)


# ── General helpers ───────────────────────────────────────────────────────────

def run(cmd, **kwargs):
    """Run a shell command and raise CalledProcessError if it fails."""
    return subprocess.run(cmd, check=True, **kwargs)

def run_quiet(cmd):
    """
    Run a shell command, hiding all output.
    Returns True on success, False if the command fails or isn't found.
    """
    try:
        subprocess.run(cmd, check=True,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def command_exists(cmd):
    """Return True if the given command is available on the system PATH."""
    return shutil.which(cmd) is not None

def venv_python():
    """Return the path to the Python executable inside our virtual environment."""
    if IS_WIN:
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"

def venv_pip():
    """Return the path to pip inside our virtual environment."""
    if IS_WIN:
        return VENV_DIR / "Scripts" / "pip.exe"
    return VENV_DIR / "bin" / "pip"

def get_desktop():
    """
    Return the user's Desktop folder as a Path.

    On Windows, OneDrive can silently relocate the Desktop to
    C:/Users/<name>/OneDrive/Desktop.  Reading from the registry gives us
    the actual path regardless of where OneDrive moved it.
    """
    if IS_WIN:
        try:
            import winreg
            key_path = r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                desktop, _ = winreg.QueryValueEx(key, "Desktop")
                return pathlib.Path(desktop)
        except Exception:
            pass  # registry read failed — fall through to the default below
    return pathlib.Path.home() / "Desktop"

def get_data_dir():
    """
    Return the CapStone data directory for this platform.

    macOS puts it on the Desktop so it's easy to find in Finder.
    Windows and Linux put it in the home folder to keep the Desktop clean.
    """
    if IS_MAC:
        return pathlib.Path.home() / "Desktop" / "CapStone"
    return pathlib.Path.home() / "CapStone"


# ── Banner ────────────────────────────────────────────────────────────────────

def print_banner():
    """Print the RPS Robot ASCII-art logo and a summary of where things will be installed."""
    print()
    # Box-drawing characters work fine in modern Windows Terminal / PowerShell (10+).
    # Only progress symbols needed ASCII fallbacks — the banner itself is always fine.
    print(_c("1;36", "  ██████╗ ██████╗ ███████╗    ██████╗  █████╗ ██████╗  █████╗ ████████╗"))
    print(_c("1;36", "  ██╔══██╗██╔══██╗██╔════╝    ██╔══██╗██╔═══██╗██╔══██╗██╔═══██╗╚══██╔══╝"))
    print(_c("1;36", "  ██████╔╝██████╔╝███████╗    ██████╔╝██║   ██║██████╔╝██║   ██║   ██║   "))
    print(_c("1;36", "  ██╔══██╗██╔═══╝ ╚════██╗    ██╔══██╗██║   ██║██╔══██╗██║   ██║   ██║   "))
    print(_c("1;36", "  ██║  ██║██║     ███████║    ██║  ██║╚██████╔╝██████╔╝╚██████╔╝   ██║   "))
    print(_c("1;36", "  ╚═╝  ╚═╝╚═╝     ╚══════╝    ╚═╝  ╚═╝ ╚═════╝ ╚═════╝  ╚═════╝    ╚═╝   "))
    print()
    print(f"  {bold('RPS Robot - Installer')}   |   TrickWing Toys")
    print(f"  Real-time gesture recognition + adaptive AI")
    print()
    line()
    print()
    info(f"Platform:    {OS_NAME} ({platform.machine()})")
    info(f"Install to:  {APP_DIR}")
    info(f"Data folder: {get_data_dir()}")
    print()


# ── Step 1: System check ──────────────────────────────────────────────────────

def _find_python312():
    """
    Search the PATH and common Windows install directories for a Python 3.12 executable.

    Returns the executable path as a string if found, or None if not found.
    We need exactly 3.12 because MediaPipe doesn't support 3.13+ yet.
    """
    candidates = ["py", "python3.12", "python"]

    # On Windows, winget installs Python to user-local paths that aren't always on PATH
    if IS_WIN:
        username = os.environ.get("USERNAME", "user")
        candidates += [
            rf"C:\Users\{username}\AppData\Local\Programs\Python\Python312\python.exe",
            r"C:\Program Files\Python312\python.exe",
            r"C:\Program Files (x86)\Python312\python.exe",
        ]

    # Try each candidate and check whether it reports version 3.12
    for candidate in candidates:
        try:
            result = subprocess.run(
                [str(candidate), "-c",
                 "import sys; v=sys.version_info; print(v.major,v.minor)"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split()
                if len(parts) == 2 and int(parts[0]) == 3 and int(parts[1]) == 12:
                    return str(candidate)
        except Exception:
            continue  # this candidate didn't work, try the next one

    return None


def _install_python312_windows():
    """
    Install Python 3.12 on Windows via winget, then relaunch this installer under 3.12.

    If winget isn't available we can't automate it, so we print manual instructions
    and exit.  After a successful install we relaunch so the rest of the setup
    continues with the right Python version.
    """
    # Maybe 3.12 is already installed but just not on PATH — check first
    py312 = _find_python312()
    if py312:
        info(f"Found Python 3.12 at {py312} - relaunching...")
        subprocess.run([py312] + sys.argv)
        sys.exit(0)

    if not command_exists("winget"):
        fail("Cannot install Python 3.12 automatically (winget not available).")
        fail("Please install Python 3.12 from:")
        fail("  https://www.python.org/downloads/release/python-3129/")
        fail("Tick 'Add Python to PATH' during install, then re-run: python install.py")
        sys.exit(1)

    info("Installing Python 3.12 via winget...")
    try:
        run(["winget", "install", "--id", "Python.Python.3.12",
             "-e", "--source", "winget",
             "--accept-package-agreements",
             "--accept-source-agreements"])
        ok("Python 3.12 installed")
    except Exception as e:
        fail(f"Could not install Python 3.12: {e}")
        fail("Please install manually from:")
        fail("  https://www.python.org/downloads/release/python-3129/")
        fail("Tick 'Add Python to PATH', then re-run: python install.py")
        sys.exit(1)

    # Relaunch with the freshly installed 3.12 so the rest of setup uses it
    py312 = _find_python312()
    if py312:
        info("Relaunching installer with Python 3.12...")
        print()
        subprocess.run([py312] + sys.argv)
        sys.exit(0)

    # If we still can't find it, PATH won't pick it up until a new shell is opened
    warn("Python 3.12 installed. Please open a new Command Prompt and run:")
    warn("  python install.py")
    sys.exit(0)


def check_system():
    """
    Step 1 — confirm the OS and Python version are compatible.

    MediaPipe only supports Python 3.9–3.12.  If the user has 3.13+,
    we either auto-install 3.12 on Windows or print clear instructions elsewhere.
    """
    step("Step 1 -- Checking system requirements")

    # Report OS and warn if it's older than what we recommend
    if IS_WIN:
        win_rel = platform.release()
        try:
            rel_int = int(win_rel)
        except ValueError:
            rel_int = 10  # treat unrecognised release strings as "10" to avoid false warnings
        if rel_int < 10:
            warn(f"Windows {win_rel} detected. Windows 10 or later recommended.")
        else:
            ok(f"Windows {win_rel} ({platform.version()})")
    elif IS_MAC:
        mac_ver = platform.mac_ver()[0]
        major   = int(mac_ver.split(".")[0]) if mac_ver else 0
        if major < 11:
            warn(f"macOS {mac_ver} - macOS 11 (Big Sur) or later recommended.")
        else:
            ok(f"macOS {mac_ver}")
    else:
        ok(f"Linux ({platform.release()})")

    # Check Python version — must be in the 3.9–3.12 range for MediaPipe
    py = sys.version_info
    if py < (3, 9):
        fail(f"Python {py.major}.{py.minor} detected. Python 3.9-3.12 required.")
        fail("Download from https://www.python.org/downloads/")
        sys.exit(1)
    elif py >= (3, 13):
        warn(f"Python {py.major}.{py.minor} detected.")
        warn("MediaPipe requires Python 3.9-3.12. Installing Python 3.12 now...")
        print()
        if IS_WIN:
            _install_python312_windows()
        else:
            fail("Please install Python 3.12 from https://www.python.org/downloads/")
            fail("Then re-run:  python3.12 install.py")
            sys.exit(1)
    else:
        ok(f"Python {py.major}.{py.minor}.{py.micro} (compatible)")

    info("A webcam is required. Built-in or USB webcam both work.")
    print()


# ── Step 2: Git ───────────────────────────────────────────────────────────────

def _find_git_windows():
    """
    Find git.exe on Windows by checking common install paths and the registry.

    winget installs Git but doesn't update PATH in the current session, so we
    search manually.  Returns the full path to git.exe, or None if not found.
    """
    # These are the standard Git for Windows install locations
    common = [
        r"C:\Program Files\Git\cmd\git.exe",
        r"C:\Program Files (x86)\Git\cmd\git.exe",
        str(pathlib.Path.home() / "AppData" / "Local" / "Programs" / "Git" / "cmd" / "git.exe"),
    ]
    for p in common:
        if pathlib.Path(p).exists():
            return p

    # Also check whatever the system PATH says, via the registry
    try:
        import winreg
        key_path = r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
            path_val, _ = winreg.QueryValueEx(key, "Path")
            for part in path_val.split(";"):
                candidate = pathlib.Path(part.strip()) / "git.exe"
                if candidate.exists():
                    return str(candidate)
    except Exception:
        pass

    return None


def ensure_git():
    """
    Step 2 — make sure Git is installed and reachable from the command line.

    Git is needed to clone the repo and pull updates on every launch.
    On macOS we use Homebrew; on Windows we use winget.
    """
    step("Step 2 -- Git version control")

    if command_exists("git"):
        # Already installed — report the version and move on
        ver = subprocess.check_output(["git", "--version"], text=True).strip()
        ok(ver)
        return

    info("Git not found - installing...")

    if IS_MAC:
        if not command_exists("brew"):
            info("Installing Homebrew first...")
            run(["/bin/bash", "-c",
                 "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"])
        run(["brew", "install", "git"])
        ok("Git installed via Homebrew")

    elif IS_WIN:
        if not command_exists("winget"):
            fail("Git not found and winget is not available.")
            fail("Please install Git from: https://git-scm.com/download/win")
            fail("Then re-run this installer.")
            sys.exit(1)
        info("Installing Git via winget...")
        run(["winget", "install", "--id", "Git.Git",
             "-e", "--source", "winget",
             "--accept-package-agreements",
             "--accept-source-agreements"])
        # winget doesn't update PATH in the current session, so find the exe manually
        # and prepend its directory so subsequent git calls work immediately
        git_exe = _find_git_windows()
        if git_exe:
            git_dir = str(pathlib.Path(git_exe).parent)
            os.environ["PATH"] = git_dir + os.pathsep + os.environ.get("PATH", "")
            ok(f"Git installed: {git_exe}")
        else:
            warn("Git installed but could not locate git.exe.")
            warn("Please close this window, open a new Command Prompt, and re-run install.py")
            sys.exit(0)
    else:
        fail("Git not found. Install with: sudo apt install git")
        sys.exit(1)

    print()


# ── Step 3: Clone / update ────────────────────────────────────────────────────

def clone_or_update():
    """
    Step 3 — get the latest code from GitHub.

    Handles three situations:
      - Repo already cloned here: pull latest changes.
      - Folder exists but isn't a git repo: back it up, then clone fresh.
      - Folder doesn't exist at all: clone fresh.
    """
    step("Step 3 -- Downloading RPS Robot from GitHub")

    # On Windows, PATH may not include git yet from the current session,
    # so prefer the full absolute path if we can find it
    git_cmd = _find_git_windows() or "git" if IS_WIN else "git"

    git_dir = APP_DIR / ".git"

    if git_dir.exists():
        # The repo is already here — just pull down any new commits
        info("Already installed - pulling latest updates...")
        run([git_cmd, "-C", str(APP_DIR), "pull", "origin", "main"])
        ok("Up to date with GitHub")
    elif APP_DIR.exists():
        # Something is in our target folder but it's not our repo — back it up
        backup = APP_DIR.parent / "rps_hand_counter_backup"
        warn(f"Folder exists but is not a git repo - backing up to {backup}")
        APP_DIR.rename(backup)
        info(f"Cloning from {GITHUB_REPO} ...")
        run([git_cmd, "clone", GITHUB_REPO, str(APP_DIR)])
        ok(f"Cloned to {APP_DIR}")
    else:
        # Fresh install — nothing here yet
        info(f"Cloning from {GITHUB_REPO} ...")
        run([git_cmd, "clone", GITHUB_REPO, str(APP_DIR)])
        ok(f"Cloned to {APP_DIR}")

    print()


# ── Step 4: Virtual environment ───────────────────────────────────────────────

def create_venv():
    """
    Step 4 — create a Python virtual environment inside the app folder.

    We always delete and recreate it to guarantee a clean slate with no
    leftover packages from a previous install attempt.
    """
    step("Step 4 -- Python virtual environment")

    if VENV_DIR.exists():
        warn("Existing .venv found - recreating for clean install...")
        shutil.rmtree(VENV_DIR)

    info("Creating virtual environment...")
    run([sys.executable, "-m", "venv", str(VENV_DIR)])
    # Upgrade pip before installing anything — old pip sometimes can't handle newer wheel formats
    run_quiet([str(venv_python()), "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
    ok("Virtual environment ready")
    print()


# ── Step 5: Install packages ──────────────────────────────────────────────────

def install_packages():
    """
    Step 5 — install all required Python packages into the virtual environment.

    Each package is tried quietly first.  If quiet mode fails (e.g. a build
    error), we retry with full output so the user can see exactly what went wrong.
    All failures are collected and reported together at the end rather than
    aborting on the first error.
    """
    step("Step 5 -- Installing Python packages")
    print()
    info("This takes 3-8 minutes. Total download: ~400MB")
    print()

    failed = []
    for name, pkg in PACKAGES:
        print(f"  Installing {name}...", end="", flush=True)
        result = subprocess.run(
            [str(venv_pip()), "install", pkg, "--quiet"],
            capture_output=True
        )
        if result.returncode == 0:
            # \r overwrites the "Installing..." line with the success message
            print(f"\r  {_c('32', '[OK]')} {name}                          ")
        else:
            # Quiet install failed — retry visibly so the user sees the error
            print(f"\r  {_c('31', '[!!]')} {name} - retrying with output...")
            result2 = subprocess.run([str(venv_pip()), "install", pkg])
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
    """
    Step 6 — download and extract the Vosk US English speech recognition model.

    The download is ~40 MB compressed.  We skip it if the model folder already
    exists, which is the case when reinstalling without wiping the app folder.
    """
    step("Step 6 -- Speech recognition model (Vosk)")

    model_dir = APP_DIR / VOSK_MODEL
    zip_path  = APP_DIR / f"{VOSK_MODEL}.zip"

    if model_dir.exists():
        ok("Vosk model already present - skipping download")
        print()
        return

    info(f"Downloading Vosk US English model (~40MB)...")
    print()

    def _progress(block_num, block_size, total_size):
        """Print a simple ASCII progress bar while the file downloads."""
        if total_size > 0:
            pct = min(100, int(block_num * block_size * 100 / total_size))
            # Build the bar: '#' for done, '.' for remaining, always 20 chars wide
            bar = "#" * (pct // 5) + "." * (20 - pct // 5)
            print(f"\r  [{bar}] {pct}%", end="", flush=True)

    urllib.request.urlretrieve(VOSK_URL, zip_path, _progress)
    print()
    print()

    info("Extracting model...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(APP_DIR)
    zip_path.unlink()  # delete the zip now that we have the extracted folder
    ok("Vosk model installed")
    print()


# ── Step 7: Data directory + launcher ─────────────────────────────────────────

def setup_data_and_launcher():
    """
    Step 7 — create the CapStone data folder structure and a Desktop launcher.

    The app writes game data (fingerprints, profiles, crash reports, etc.) to the
    CapStone folder.  We create all the subdirectories here so the app never has
    to worry about them being missing.  Then we hand off to a platform-specific
    function to create the launcher.
    """
    step("Step 7 -- Data folder + Desktop launcher")

    data_dir = get_data_dir()
    desktop  = get_desktop()

    # Create all subdirectories the app expects (exist_ok=True means no error if they're already there)
    for subdir in ["", "fingerprints", "profiles", "simulations", "feedback", "crash_reports"]:
        (data_dir / subdir).mkdir(parents=True, exist_ok=True)
    ok(f"Data folder: {data_dir}")

    # Delegate launcher creation to the right platform-specific function
    if IS_WIN:
        _create_windows_launcher(desktop, data_dir)
    elif IS_MAC:
        _create_mac_launcher(desktop, data_dir)
    else:
        _create_linux_launcher(desktop)

    print()


def _create_mac_launcher(desktop, data_dir):
    """
    Create a .command launcher script in the app folder and a Finder alias on the Desktop.

    The .command file activates the venv and runs main.py when double-clicked in Finder.
    We also try to build a proper .icns icon from our PNG so the alias gets a nice icon.
    Icon creation is best-effort — if it fails we warn but don't abort.
    """
    # The launcher lives inside the app folder; the Desktop gets a Finder alias to it
    # (a proper alias, not a symlink, so Finder treats it like a real shortcut)
    launcher = APP_DIR / "Launch RPS Robot.command"
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
    launcher.chmod(0o755)  # mark it executable so Finder can run it on double-click

    icon_png  = APP_DIR / "TheRPSRobot.png"
    icon_icns = APP_DIR / "TheRPSRobot.icns"

    # Build a .icns file from our PNG.  macOS needs one file with multiple sizes baked in.
    if icon_png.exists():
        try:
            iconset = APP_DIR / "TheRPSRobot.iconset"
            iconset.mkdir(exist_ok=True)

            # macOS requires all these specific sizes to be present in the .iconset folder
            sizes = {
                "icon_16x16.png":      16,
                "icon_16x16@2x.png":   32,
                "icon_32x32.png":      32,
                "icon_32x32@2x.png":   64,
                "icon_128x128.png":    128,
                "icon_128x128@2x.png": 256,
                "icon_256x256.png":    256,
                "icon_256x256@2x.png": 512,
                "icon_512x512.png":    512,
                "icon_512x512@2x.png": 1024,
            }
            from PIL import Image
            src = Image.open(icon_png).convert("RGBA")  # open at full resolution for best quality
            for fname, sz in sizes.items():
                src.resize((sz, sz), Image.LANCZOS).save(iconset / fname, optimize=True)

            # iconutil is the macOS CLI tool that packs the .iconset folder into a single .icns
            result = subprocess.run(
                ["iconutil", "-c", "icns", str(iconset), "-o", str(icon_icns)],
                capture_output=True)
            shutil.rmtree(iconset, ignore_errors=True)  # clean up the temp folder
            if result.returncode == 0:
                ok("App icon (.icns) created at full quality")
            else:
                warn("iconutil failed - icon may not appear")
        except Exception as e:
            warn(f"Could not create .icns: {e}")

    # Apply the icon to the .command file using macOS's Objective-C/JS bridge
    if icon_icns.exists():
        try:
            subprocess.run(["osascript", "-l", "JavaScript", "-e", f'''
                ObjC.import("AppKit");
                var img = $.NSImage.alloc.initWithContentsOfFile("{icon_icns}");
                var ws  = $.NSWorkspace.sharedWorkspace;
                ws.setIconForFileOptions(img, "{launcher}", 0);
            '''], capture_output=True)
        except Exception:
            pass  # icon is cosmetic — don't abort install if this fails

    # Create a Finder alias on the Desktop pointing to the launcher script
    alias_name    = "RPS Robot"
    desktop_alias = desktop / alias_name
    try:
        subprocess.run(["osascript", "-e", f'''
            tell application "Finder"
                set src to POSIX file "{launcher}" as alias
                set dst to POSIX file "{desktop}" as alias
                make alias file to src at dst
                set name of result to "{alias_name}"
            end tell
        '''], capture_output=True, timeout=10)
        ok(f"Desktop icon created: '{alias_name}'")
    except Exception:
        # Finder scripting failed — fall back to a plain symlink
        try:
            if not desktop_alias.exists():
                desktop_alias.symlink_to(launcher)
            ok(f"Desktop shortcut created: '{alias_name}'")
        except Exception as e:
            warn(f"Could not create Desktop icon: {e}")

    # Add a shortcut to the data folder so users can find their save files easily
    symlink = desktop / "RPS Robot Data"
    if not symlink.exists():
        try:
            symlink.symlink_to(data_dir)
            ok("Data folder shortcut on Desktop")
        except Exception:
            pass


def _create_windows_launcher(desktop, data_dir):
    """
    Create a .bat launcher in the app folder and a .lnk shortcut on the Desktop.

    PIL's built-in ICO writer only produces a single-size file (~1 KB), which
    looks blurry in Explorer.  We write the ICO binary format manually so we
    can embed all standard sizes in one file (~140 KB), which Windows uses to
    pick the best resolution for each context.
    """
    # The .bat lives in the app folder — only the .lnk shortcut goes on the Desktop
    bat = APP_DIR / "Launch RPS Robot.bat"
    try:
        bat.write_text(textwrap.dedent(f"""\
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
    except Exception as e:
        warn(f"Could not create launcher script: {e}")
        return

    # Build a multi-size .ico from our PNG
    ico_path = APP_DIR / "TheRPSRobot.ico"
    png_path = APP_DIR / "TheRPSRobot.png"

    if png_path.exists():
        try:
            from PIL import Image

            src = Image.open(png_path).convert("RGBA")

            # Standard Windows icon sizes — Explorer picks the closest one for each context
            ico_px = [16, 24, 32, 48, 64, 128, 256]

            # Resize the image to each size and encode each as a PNG blob in memory
            frames = []
            for sz in ico_px:
                buf = io.BytesIO()
                src.resize((sz, sz), Image.LANCZOS).save(buf, format="PNG", optimize=True)
                frames.append((sz, buf.getvalue()))

            # Hand-write the ICO binary format:
            #   6-byte global header + 16-byte directory entry per frame + image data
            n        = len(frames)
            hdr_size = 6 + n * 16  # total size of header + directory
            offset   = hdr_size    # byte offset where the first image blob starts

            # Pre-compute the (width, data_size, data_offset) for each frame
            entries = []
            for sz, data in frames:
                w = sz if sz < 256 else 0  # ICO spec: 256px is encoded as 0
                entries.append((w, len(data), offset))
                offset += len(data)

            out = io.BytesIO()
            out.write(struct.pack("<HHH", 0, 1, n))  # ICO header: reserved=0, type=1 (icon), count
            for (w, size, off) in entries:
                # Each directory entry: width, height, color_count, reserved, planes, bit_count, size, offset
                out.write(struct.pack("<BBBBHHII", w, w, 0, 0, 1, 32, size, off))
            for _, data in frames:
                out.write(data)  # append the actual PNG image data for each frame

            with open(str(ico_path), "wb") as f:
                f.write(out.getvalue())
            ok(f"App icon (.ico) created ({len(out.getvalue())//1024}KB, {len(frames)} sizes)")
        except Exception as e:
            warn(f"Could not create .ico: {e}")

    # Create a .lnk shortcut on the Desktop using PowerShell's WScript.Shell COM object
    lnk_path = desktop / "RPS Robot.lnk"
    try:
        ps_cmd = (
            f'$ws = New-Object -ComObject WScript.Shell; '
            f'$s = $ws.CreateShortcut("{lnk_path}"); '
            f'$s.TargetPath = "{bat}"; '
            f'$s.WorkingDirectory = "{APP_DIR}"; '
            f'$s.IconLocation = "{ico_path},0"; '
            f'$s.Description = "RPS Robot - Gesture Recognition Game"; '
            f'$s.WindowStyle = 1; '
            f'$s.Save()'
        )
        subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True)
        ok("Desktop icon created: 'RPS Robot'")
    except Exception as e:
        warn(f"Could not create Desktop icon: {e}")

    # Also create a .lnk shortcut to the data folder for easy file access
    data_lnk = desktop / "RPS Robot Data.lnk"
    if not data_lnk.exists():
        try:
            ps_cmd2 = (
                f'$ws = New-Object -ComObject WScript.Shell; '
                f'$s = $ws.CreateShortcut("{data_lnk}"); '
                f'$s.TargetPath = "{data_dir}"; '
                f'$s.Save()'
            )
            subprocess.run(["powershell", "-Command", ps_cmd2], capture_output=True)
            ok("Data folder shortcut on Desktop")
        except Exception:
            pass

    # Refresh Explorer's icon cache so the new shortcut shows up immediately without a reboot
    try:
        subprocess.run([
            "powershell", "-Command",
            "ie4uinit.exe -show; "
            "Stop-Process -Name explorer -Force -ErrorAction SilentlyContinue; "
            "Start-Sleep 1; "
            "Start-Process explorer"
        ], capture_output=True, timeout=10)
    except Exception:
        pass


def _create_linux_launcher(desktop):
    """Create a minimal shell script launcher on the Desktop for Linux."""
    launcher = desktop / "Launch RPS Robot.sh"
    launcher.write_text(textwrap.dedent(f"""\
        #!/bin/bash
        cd "{APP_DIR}"
        source "{VENV_DIR}/bin/activate"
        python main.py
    """))
    launcher.chmod(0o755)  # mark it executable
    ok(f"Launcher created: {launcher}")


# ── Step 8: Verify ────────────────────────────────────────────────────────────

def verify_installation():
    """
    Step 8 — confirm that every installed package can actually be imported.

    Runs a short Python snippet for each dependency inside the venv and
    reports pass/fail.  Returns True if everything passed.
    """
    step("Step 8 -- Verifying installation")
    print()

    vosk_path = APP_DIR / VOSK_MODEL

    # Each tuple is (display_name, python_code_to_run_as_a_test)
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
        ("Sentry",       "import sentry_sdk"),
        # Verify git works inside the app folder (not just that the binary exists)
        ("Git repo",     f"import subprocess; subprocess.check_call("
                         f"['git','-C',r'{APP_DIR}','rev-parse'],"
                         f"capture_output=True)"),
        # Verify the Vosk model folder was actually extracted
        ("Vosk model",   f"from pathlib import Path; "
                         f"assert Path(r'{vosk_path}').exists()"),
    ]

    failed = 0
    for name, test in checks:
        print(f"  Checking {name}...", end="", flush=True)
        result = subprocess.run(
            [str(venv_python()), "-c", test],
            capture_output=True
        )
        if result.returncode == 0:
            print(f"\r  {_c('32', '[OK]')} {name}                    ")
        else:
            print(f"\r  {_c('31', '[!!]')} {name} -- FAILED")
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
    """Step 9 — print setup instructions for users who have the physical robot arm."""
    step("Step 9 -- Optional: ESP32 Robot Arm")
    print()
    info("If you are using the physical RPS Robot arm (ESP32):")
    print()
    print("  1. Install the CP210x USB driver:")
    print("     https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers")
    print("  2. Connect the ESP32 via USB")
    print("  3. In the app: press D -> H, select port, press ENTER")
    print()
    info("The app works fully without the ESP32 - this is optional.")
    print()


# ── Done ──────────────────────────────────────────────────────────────────────

def print_done():
    """Print the final success message with launch instructions."""
    line()
    print()
    ok("Installation complete!")
    print()

    # Pick the right launcher filename for the current platform
    if IS_WIN:
        launcher_name = "Launch RPS Robot.bat"
    elif IS_MAC:
        launcher_name = "Launch RPS Robot.command"
    else:
        launcher_name = "Launch RPS Robot.sh"

    print(f"  To launch:")
    print(f"  -> Double-click '{launcher_name}' on your Desktop")
    print()
    print(f"  Auto-updates:")
    print(f"  -> The app checks GitHub on every launch")
    print(f"  -> A yellow banner appears in the menu when an update is ready")
    print(f"  -> Press U to update and restart automatically")
    print()
    print(f"  Your data: {get_data_dir()}")
    print()
    line()
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    """
    Run all installation steps in order, then optionally launch the app.

    Each step is its own function so a failure in one step is easy to locate
    and the terminal output is clearly sectioned.
    """
    print_banner()
    check_system()
    ensure_git()
    clone_or_update()
    create_venv()
    install_packages()
    install_vosk_model()
    setup_data_and_launcher()
    verify_installation()
    print_esp32_notice()
    print_done()

    # Ask the user if they want to launch the app right now
    try:
        answer = input("  Launch RPS Robot now? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"

    if answer == "y":
        print()
        info("Starting RPS Robot...")
        os.chdir(APP_DIR)
        py = str(venv_python())
        if IS_WIN:
            # os.execv is unreliable on Windows, so we spawn a new process instead
            subprocess.run([py, "main.py"])
        else:
            # On Mac/Linux, replace the current process in-place (no zombie, cleaner exit)
            os.execv(py, [py, "main.py"])


if __name__ == "__main__":
    main()

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
  1. Checks OS and Python version (requires 3.9-3.12, installs 3.12 if needed)
  2. Installs Git if needed
  3. Clones the RPS Robot repo from GitHub (enables auto-updates)
  4. Creates a Python virtual environment
  5. Installs all Python packages
  6. Downloads the Vosk speech recognition model
  7. Creates a Desktop launcher
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
GITHUB_REPO = "https://github.com/Zacccyyy/RPSRobot.git"
APP_DIR     = pathlib.Path.home() / "rps_hand_counter"
VENV_DIR    = APP_DIR / ".venv"

# Vosk speech recognition model — small US English model (~40 MB download)
VOSK_MODEL  = "vosk-model-small-en-us-0.15"
VOSK_URL    = f"https://alphacephei.com/vosk/models/{VOSK_MODEL}.zip"

# All Python packages that need to be installed, as (display_name, pip_spec) pairs
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

# ── Platform ──────────────────────────────────────────────────────────────────
IS_MAC   = sys.platform == "darwin"
IS_WIN   = sys.platform == "win32"
IS_LINUX = sys.platform.startswith("linux")
OS_NAME  = "macOS" if IS_MAC else ("Windows" if IS_WIN else "Linux")

# ── Colours: disabled on Windows CMD (no ANSI support by default) ─────────────
_USE_COLOR = IS_MAC or IS_LINUX or os.environ.get("TERM") == "xterm-256color"

def _c(code, text):
    """Wrap text in an ANSI colour escape sequence, or return plain text on Windows CMD."""
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

# Use ASCII-safe symbols on Windows CMD, Unicode elsewhere
_OK   = "[OK]"   if IS_WIN else "  ok "
_FAIL = "[!!]"   if IS_WIN else "  !! "
_WARN = "[??]"   if IS_WIN else "  ?? "
_ARR  = "  ->  "   # same on all platforms

# Shortcut print helpers so each step only needs one function call
def ok(msg):   print(_c("32",   f"{_OK}  {msg}"))
def info(msg): print(_c("36",   f"{_ARR} {msg}"))
def warn(msg): print(_c("33",   f"{_WARN} {msg}"))
def fail(msg): print(_c("31",   f"{_FAIL} {msg}"))
def step(msg): print(_c("1;36", f"\n---  {msg}  {'-' * max(0, 44 - len(msg))}"))
def line():    print(_c("36",   "-" * 50))
def bold(msg): return _c("1", msg)


# ── Helpers ───────────────────────────────────────────────────────────────────

def run(cmd, **kwargs):
    """Run a shell command and raise an exception if it fails."""
    return subprocess.run(cmd, check=True, **kwargs)

def run_quiet(cmd):
    """Run a shell command, suppressing all output. Returns True on success."""
    try:
        subprocess.run(cmd, check=True,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def command_exists(cmd):
    """Return True if the given command is on the system PATH."""
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

    On Windows, OneDrive sometimes moves the Desktop folder to
    C:/Users/<name>/OneDrive/Desktop — we read the real path from the
    registry instead of assuming ~/Desktop.
    """
    if IS_WIN:
        try:
            import winreg
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"
            ) as key:
                desktop, _ = winreg.QueryValueEx(key, "Desktop")
                return pathlib.Path(desktop)
        except Exception:
            pass  # fall through to the default below
    return pathlib.Path.home() / "Desktop"

def get_data_dir():
    """
    Return the CapStone data directory for this platform.

    macOS: ~/Desktop/CapStone  (keeps existing user data in place)
    Windows/Linux: ~/CapStone  (keeps Desktop clean)
    """
    import sys as _sys
    if _sys.platform == "darwin":
        return pathlib.Path.home() / "Desktop" / "CapStone"
    return pathlib.Path.home() / "CapStone"


# ── Banner ────────────────────────────────────────────────────────────────────

def print_banner():
    """Print the RPS Robot ASCII-art logo and basic platform info."""
    print()
    # The box-drawing chars in the banner render correctly in modern Windows
    # Terminal and PowerShell (Windows 10+). Only progress symbols like
    # checkmarks caused issues on older Windows CMD — those use ASCII fallbacks.
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
    Search common locations for a Python 3.12 executable.

    Checks the PATH first (via the candidate names), then checks known
    Windows install directories.  Returns the executable path as a string,
    or None if not found.
    """
    candidates = ["py", "python3.12", "python"]

    # Add common Windows install paths to the search list
    if IS_WIN:
        username = os.environ.get("USERNAME", "user")
        candidates += [
            rf"C:\Users\{username}\AppData\Local\Programs\Python\Python312\python.exe",
            r"C:\Program Files\Python312\python.exe",
            r"C:\Program Files (x86)\Python312\python.exe",
        ]

    for candidate in candidates:
        try:
            result = subprocess.run(
                [str(candidate), "-c",
                 "import sys; v=sys.version_info; print(v.major,v.minor)"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split()
                # Only accept exactly Python 3.12
                if len(parts) == 2 and int(parts[0]) == 3 and int(parts[1]) == 12:
                    return str(candidate)
        except Exception:
            continue  # this candidate didn't work, try the next one

    return None


def _install_python312_windows():
    """
    Install Python 3.12 on Windows via winget, then relaunch this installer.

    If winget is not available, print instructions and exit.
    After a successful install, we relaunch automatically with the new Python.
    """
    # Check if Python 3.12 is already installed somewhere non-obvious
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

    # Relaunch the installer with the newly installed Python 3.12
    py312 = _find_python312()
    if py312:
        info("Relaunching installer with Python 3.12...")
        print()
        subprocess.run([py312] + sys.argv)
        sys.exit(0)

    warn("Python 3.12 installed. Please open a new Command Prompt and run:")
    warn("  python install.py")
    sys.exit(0)


def check_system():
    """
    Step 1 — confirm the OS and Python version are compatible.

    MediaPipe only supports Python 3.9–3.12, so we either bail or
    auto-install 3.12 if the user has something newer.
    """
    step("Step 1 -- Checking system requirements")

    # Report OS version and warn if it's older than recommended
    if IS_WIN:
        win_rel = platform.release()
        try:
            rel_int = int(win_rel)
        except ValueError:
            rel_int = 10  # treat unknown releases as "10" to avoid false warnings
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

    # MediaPipe requires Python 3.9–3.12 — check and handle out-of-range versions
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

    winget installs Git but doesn't always update PATH in the current session,
    so we need to look for it manually.  Returns the full path to git.exe as a
    string, or None if not found.
    """
    common = [
        r"C:\Program Files\Git\cmd\git.exe",
        r"C:\Program Files (x86)\Git\cmd\git.exe",
        str(pathlib.Path.home() / "AppData" / "Local" / "Programs" / "Git" / "cmd" / "git.exe"),
    ]
    for p in common:
        if pathlib.Path(p).exists():
            return p

    # Also check the system PATH via the registry
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
    """
    Step 2 — make sure Git is installed and on the PATH.

    Git is needed so we can clone the repo and pull updates later.
    On Mac we use Homebrew; on Windows we use winget.
    """
    step("Step 2 -- Git version control")

    if command_exists("git"):
        # Already installed — just report the version and move on
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
        # PATH isn't updated in the current session after winget installs,
        # so find git.exe directly and prepend its directory to PATH
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

    Three cases:
      - Repo already cloned: just pull the latest changes.
      - Folder exists but isn't a git repo: back it up, then clone fresh.
      - Folder doesn't exist: clone fresh.
    """
    step("Step 3 -- Downloading RPS Robot from GitHub")

    # On Windows, PATH may not include git yet, so use the full path if we can find it
    git_cmd = "git"
    if IS_WIN:
        found = _find_git_windows()
        if found:
            git_cmd = found

    git_dir = APP_DIR / ".git"

    if git_dir.exists():
        # Already a git repo — just pull latest changes
        info("Already installed - pulling latest updates...")
        run([git_cmd, "-C", str(APP_DIR), "pull", "origin", "main"])
        ok("Up to date with GitHub")
    elif APP_DIR.exists():
        # Folder exists but isn't ours — back it up so we don't destroy data
        backup = APP_DIR.parent / "rps_hand_counter_backup"
        warn(f"Folder exists but is not a git repo - backing up to {backup}")
        APP_DIR.rename(backup)
        info(f"Cloning from {GITHUB_REPO} ...")
        run([git_cmd, "clone", GITHUB_REPO, str(APP_DIR)])
        ok(f"Cloned to {APP_DIR}")
    else:
        # Fresh install
        info(f"Cloning from {GITHUB_REPO} ...")
        run([git_cmd, "clone", GITHUB_REPO, str(APP_DIR)])
        ok(f"Cloned to {APP_DIR}")

    print()


# ── Step 4: Virtual environment ───────────────────────────────────────────────

def create_venv():
    """
    Step 4 — create a Python virtual environment inside the app folder.

    We always delete and recreate the venv on a fresh install to ensure
    there are no stale or conflicting packages left over.
    """
    step("Step 4 -- Python virtual environment")

    if VENV_DIR.exists():
        warn("Existing .venv found - recreating for clean install...")
        shutil.rmtree(VENV_DIR)

    info("Creating virtual environment...")
    run([sys.executable, "-m", "venv", str(VENV_DIR)])
    # Upgrade pip quietly before installing anything else
    run_quiet([str(venv_python()), "-m", "pip", "install",
               "--upgrade", "pip", "--quiet"])
    ok("Virtual environment ready")
    print()


# ── Step 5: Install packages ──────────────────────────────────────────────────

def install_packages():
    """
    Step 5 — install all required Python packages into the virtual environment.

    Tries each package quietly first; if that fails, retries with full output
    so the user can see what went wrong.  Collects failures and reports them
    all at the end rather than aborting on the first error.
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
            # Use \r to overwrite the "Installing..." line with a success message
            print(f"\r  {_c('32', '[OK]')} {name}                          ")
        else:
            # Quiet install failed — retry with full output so user can see the error
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

    The model is ~40 MB compressed.  We skip the download if the model folder
    already exists (e.g. on a reinstall).
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
        """Print a simple ASCII progress bar — no Unicode block chars (works on Windows CMD)."""
        if total_size > 0:
            pct = min(100, int(block_num * block_size * 100 / total_size))
            bar = "#" * (pct // 5) + "." * (20 - pct // 5)
            print(f"\r  [{bar}] {pct}%", end="", flush=True)

    urllib.request.urlretrieve(VOSK_URL, zip_path, _progress)
    print()
    print()

    info("Extracting model...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(APP_DIR)
    zip_path.unlink()  # remove the zip now that we've extracted it
    ok("Vosk model installed")
    print()


# ── Step 7: Data directory + launcher ─────────────────────────────────────────

def setup_data_and_launcher():
    """
    Step 7 — create the CapStone data folder structure and a Desktop launcher.

    Creates subdirectories for all the data the app produces, then calls
    the platform-specific launcher creator.
    """
    step("Step 7 -- Data folder + Desktop launcher")

    data_dir = get_data_dir()
    desktop  = get_desktop()

    # Create all the subdirectories the app needs (exist_ok means no error if already there)
    for subdir in ["", "fingerprints", "profiles", "simulations", "feedback", "crash_reports"]:
        (data_dir / subdir).mkdir(parents=True, exist_ok=True)
    ok(f"Data folder: {data_dir}")

    # Create a platform-appropriate launcher
    if IS_WIN:
        _create_windows_launcher(desktop, data_dir)
    elif IS_MAC:
        _create_mac_launcher(desktop, data_dir)
    else:
        _create_linux_launcher(desktop)

    print()


def _create_mac_launcher(desktop, data_dir):
    """
    Create a .command script in the app folder and a Finder alias on the Desktop.

    Also builds a proper .icns icon from the PNG so the launcher gets a nice icon
    in Finder.  Everything is wrapped in try/except because icon creation is
    cosmetic — we don't want it to abort the install.
    """
    # The actual launcher lives inside the app folder, not on the Desktop.
    # The Desktop gets a Finder alias pointing to it (cleaner than a symlink).
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
    launcher.chmod(0o755)  # make it executable

    icon_png  = APP_DIR / "TheRPSRobot.png"
    icon_icns = APP_DIR / "TheRPSRobot.icns"

    # Build a proper .icns file from our PNG (macOS needs multiple sizes in one file)
    if icon_png.exists():
        try:
            iconset = APP_DIR / "TheRPSRobot.iconset"
            iconset.mkdir(exist_ok=True)
            # macOS requires all these sizes in an .iconset folder
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
            # Open the source PNG at full resolution for the best downscaling quality
            src = Image.open(icon_png).convert("RGBA")
            for fname, sz in sizes.items():
                src.resize((sz, sz), Image.LANCZOS).save(
                    iconset / fname, optimize=True)
            # iconutil is a macOS command-line tool that converts .iconset -> .icns
            result = subprocess.run(
                ["iconutil", "-c", "icns", str(iconset), "-o", str(icon_icns)],
                capture_output=True)
            import shutil as _sh
            _sh.rmtree(iconset, ignore_errors=True)  # clean up the temp iconset folder
            if result.returncode == 0:
                ok("App icon (.icns) created at full quality")
            else:
                warn("iconutil failed - icon may not appear")
        except Exception as e:
            warn(f"Could not create .icns: {e}")

    # Apply the icon to the .command file using macOS's JavaScript bridge
    if icon_icns.exists():
        try:
            subprocess.run(["osascript", "-l", "JavaScript", "-e", f'''
                ObjC.import("AppKit");
                var img = $.NSImage.alloc.initWithContentsOfFile("{icon_icns}");
                var ws  = $.NSWorkspace.sharedWorkspace;
                ws.setIconForFileOptions(img, "{launcher}", 0);
            '''], capture_output=True)
        except Exception:
            pass  # icon cosmetic only — don't abort install

    # Create a Finder alias on the Desktop pointing to the launcher
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
        # Fallback: create a regular symlink if Finder scripting fails
        try:
            if not desktop_alias.exists():
                desktop_alias.symlink_to(launcher)
            ok(f"Desktop shortcut created: '{alias_name}'")
        except Exception as e:
            warn(f"Could not create Desktop icon: {e}")

    # Also put a shortcut to the data folder on the Desktop for easy access
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

    Also builds a multi-size .ico from the PNG.  PIL's built-in ICO support
    only produces a single-size 1KB file, so we write the ICO binary format
    manually to get a proper multi-resolution icon (~140KB).
    """
    # Store the .bat in the app folder — only the shortcut goes on the Desktop
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

    # Build a multi-size .ico file
    ico_path = APP_DIR / "TheRPSRobot.ico"
    png_path = APP_DIR / "TheRPSRobot.png"

    if png_path.exists():
        try:
            from PIL import Image
            import struct, io as _io

            src = Image.open(png_path).convert("RGBA")

            # Generate PNG-encoded frames at each standard Windows icon size
            ico_px = [16, 24, 32, 48, 64, 128, 256]
            frames = []
            for sz in ico_px:
                buf = _io.BytesIO()
                src.resize((sz, sz), Image.LANCZOS).save(buf, format="PNG", optimize=True)
                frames.append((sz, buf.getvalue()))

            # Manually write the ICO binary format (6-byte header + 16-byte entry per frame)
            n        = len(frames)
            hdr_size = 6 + n * 16  # total header bytes before image data starts
            offset   = hdr_size    # byte offset of the first image data block

            entries = []
            for sz, data in frames:
                # ICO format encodes 256px as 0 in the width/height fields
                w = sz if sz < 256 else 0
                entries.append((w, len(data), offset))
                offset += len(data)

            out = _io.BytesIO()
            # ICO header: reserved=0, type=1 (icon), count=n
            out.write(struct.pack("<HHH", 0, 1, n))
            # One 16-byte directory entry per frame
            for (w, size, off) in entries:
                out.write(struct.pack("<BBBBHHII", w, w, 0, 0, 1, 32, size, off))
            # Append the raw PNG data for each frame
            for _, data in frames:
                out.write(data)

            with open(str(ico_path), "wb") as f:
                f.write(out.getvalue())
            ok(f"App icon (.ico) created ({len(out.getvalue())//1024}KB, {len(frames)} sizes)")
        except Exception as e:
            warn(f"Could not create .ico: {e}")

    # Create a .lnk shortcut on the Desktop using PowerShell's WScript.Shell COM object
    lnk_path = desktop / "RPS Robot.lnk"
    app_str  = str(APP_DIR)
    bat_str  = str(bat)
    lnk_str  = str(lnk_path)
    ico_str  = str(ico_path)

    try:
        ps_cmd = (
            f'$ws = New-Object -ComObject WScript.Shell; '
            f'$s = $ws.CreateShortcut("{lnk_str}"); '
            f'$s.TargetPath = "{bat_str}"; '
            f'$s.WorkingDirectory = "{app_str}"; '
            f'$s.IconLocation = "{ico_str},0"; '
            f'$s.Description = "RPS Robot - Gesture Recognition Game"; '
            f'$s.WindowStyle = 1; '
            f'$s.Save()'
        )
        subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True)
        ok(f"Desktop icon created: 'RPS Robot'")
    except Exception as e:
        warn(f"Could not create Desktop icon: {e}")

    # Also add a shortcut to the data folder on the Desktop
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

    # Restart Windows Explorer so the new icon appears immediately without a reboot
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
    launcher.chmod(0o755)  # make it executable
    ok(f"Launcher created: {launcher}")


# ── Step 8: Verify ────────────────────────────────────────────────────────────

def verify_installation():
    """
    Step 8 — confirm that every installed package can be imported correctly.

    Runs a small Python snippet for each package in the venv and reports
    pass/fail.  Returns True if everything passed.
    """
    step("Step 8 -- Verifying installation")
    print()

    vosk_path = APP_DIR / VOSK_MODEL

    # Each entry is (display_name, python_snippet_to_run)
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
        ("Git repo",     f"import subprocess; subprocess.check_call("
                         f"['git','-C',r'{APP_DIR}','rev-parse'],"
                         f"capture_output=True)"),
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
    """Step 9 — print optional instructions for users with the physical robot arm."""
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

    Each step is its own function so failures are easy to isolate and the
    output is clearly sectioned.
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

    # Ask whether to launch the app right now
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
            # os.execv is unreliable on Windows, so spawn a new process instead
            subprocess.run([py, "main.py"])
        else:
            # On Mac/Linux, replace the current process in-place (cleaner, no zombie)
            os.execv(py, [py, "main.py"])


if __name__ == "__main__":
    main()

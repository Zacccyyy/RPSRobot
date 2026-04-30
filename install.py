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

VOSK_MODEL  = "vosk-model-small-en-us-0.15"
VOSK_URL    = f"https://alphacephei.com/vosk/models/{VOSK_MODEL}.zip"

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
]

# ── Platform ──────────────────────────────────────────────────────────────────
IS_MAC   = sys.platform == "darwin"
IS_WIN   = sys.platform == "win32"
IS_LINUX = sys.platform.startswith("linux")
OS_NAME  = "macOS" if IS_MAC else ("Windows" if IS_WIN else "Linux")

# ── Colours: disabled on Windows CMD (no ANSI support by default) ─────────────
_USE_COLOR = IS_MAC or IS_LINUX or os.environ.get("TERM") == "xterm-256color"

def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

# Use ASCII-safe symbols on Windows, Unicode on Mac/Linux
_OK   = "[OK]"   if IS_WIN else "  ok "
_FAIL = "[!!]"   if IS_WIN else "  !! "
_WARN = "[??]"   if IS_WIN else "  ?? "
_ARR  = "  ->  " if IS_WIN else "  ->  "

def ok(msg):   print(_c("32",   f"{_OK}  {msg}"))
def info(msg): print(_c("36",   f"{_ARR} {msg}"))
def warn(msg): print(_c("33",   f"{_WARN} {msg}"))
def fail(msg): print(_c("31",   f"{_FAIL} {msg}"))
def step(msg): print(_c("1;36", f"\n---  {msg}  {'-' * max(0, 44 - len(msg))}"))
def line():    print(_c("36",   "-" * 50))
def bold(msg): return _c("1", msg)


# ── Helpers ───────────────────────────────────────────────────────────────────

def run(cmd, **kwargs):
    return subprocess.run(cmd, check=True, **kwargs)

def run_quiet(cmd):
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
    if IS_WIN:
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"

def venv_pip():
    if IS_WIN:
        return VENV_DIR / "Scripts" / "pip.exe"
    return VENV_DIR / "bin" / "pip"

def get_desktop():
    """Get the real Desktop path - handles OneDrive relocation on Windows."""
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
            pass
    return pathlib.Path.home() / "Desktop"

def get_data_dir():
    # macOS: ~/Desktop/CapStone (existing users keep their data location)
    # Windows/Linux: ~/CapStone  (keeps Desktop clean)
    import sys as _sys
    if _sys.platform == "darwin":
        return pathlib.Path.home() / "Desktop" / "CapStone"
    return pathlib.Path.home() / "CapStone"


# ── Banner ────────────────────────────────────────────────────────────────────

def print_banner():
    print()
    # The box-drawing chars in the banner are fine on Windows - they render
    # correctly in modern Windows Terminal and PowerShell (Windows 10+).
    # Only the progress/status symbols like checkmarks caused issues.
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
    """Find Python 3.12 executable if already installed."""
    candidates = ["py", "python3.12", "python"]
    # Also check common Windows install paths
    if IS_WIN:
        username = os.environ.get("USERNAME", "user")
        extra = [
            rf"C:\Users\{username}\AppData\Local\Programs\Python\Python312\python.exe",
            r"C:\Program Files\Python312\python.exe",
            r"C:\Program Files (x86)\Python312\python.exe",
        ]
        candidates += extra
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
            continue
    return None


def _install_python312_windows():
    """Install Python 3.12 on Windows and relaunch this installer with it."""
    # Check if already installed somewhere
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

    # Try to find and relaunch with it
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
    step("Step 1 -- Checking system requirements")

    if IS_WIN:
        win_rel = platform.release()
        try:
            rel_int = int(win_rel)
        except ValueError:
            rel_int = 10
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

    # MediaPipe only supports Python 3.9 - 3.12
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
    """Find git.exe on Windows - checks common paths and registry."""
    common = [
        r"C:\Program Files\Git\cmd\git.exe",
        r"C:\Program Files (x86)\Git\cmd\git.exe",
        str(pathlib.Path.home() / "AppData" / "Local" / "Programs" / "Git" / "cmd" / "git.exe"),
    ]
    for p in common:
        if pathlib.Path(p).exists():
            return p
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
    step("Step 2 -- Git version control")

    if command_exists("git"):
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
        # PATH not updated in current session - find git.exe directly
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
    step("Step 3 -- Downloading RPS Robot from GitHub")

    # Use full path to git on Windows in case PATH update didn't stick
    git_cmd = "git"
    if IS_WIN:
        found = _find_git_windows()
        if found:
            git_cmd = found

    git_dir = APP_DIR / ".git"

    if git_dir.exists():
        info("Already installed - pulling latest updates...")
        run([git_cmd, "-C", str(APP_DIR), "pull", "origin", "main"])
        ok("Up to date with GitHub")
    elif APP_DIR.exists():
        backup = APP_DIR.parent / "rps_hand_counter_backup"
        warn(f"Folder exists but is not a git repo - backing up to {backup}")
        APP_DIR.rename(backup)
        info(f"Cloning from {GITHUB_REPO} ...")
        run([git_cmd, "clone", GITHUB_REPO, str(APP_DIR)])
        ok(f"Cloned to {APP_DIR}")
    else:
        info(f"Cloning from {GITHUB_REPO} ...")
        run([git_cmd, "clone", GITHUB_REPO, str(APP_DIR)])
        ok(f"Cloned to {APP_DIR}")

    print()


# ── Step 4: Virtual environment ───────────────────────────────────────────────

def create_venv():
    step("Step 4 -- Python virtual environment")

    if VENV_DIR.exists():
        warn("Existing .venv found - recreating for clean install...")
        shutil.rmtree(VENV_DIR)

    info("Creating virtual environment...")
    run([sys.executable, "-m", "venv", str(VENV_DIR)])
    run_quiet([str(venv_python()), "-m", "pip", "install",
               "--upgrade", "pip", "--quiet"])
    ok("Virtual environment ready")
    print()


# ── Step 5: Install packages ──────────────────────────────────────────────────

def install_packages():
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
            # Use \r to overwrite the "Installing..." line
            print(f"\r  {_c('32', '[OK]')} {name}                          ")
        else:
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
    step("Step 6 -- Speech recognition model (Vosk)")

    model_dir = APP_DIR / VOSK_MODEL
    zip_path  = APP_DIR / f"{VOSK_MODEL}.zip"

    if model_dir.exists():
        ok("Vosk model already present - skipping download")
        print()
        return

    info(f"Downloading Vosk US English model (~40MB)...")
    print()

    # Simple progress without Unicode block chars (works on Windows CMD)
    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            pct  = min(100, int(block_num * block_size * 100 / total_size))
            bar  = "#" * (pct // 5) + "." * (20 - pct // 5)
            print(f"\r  [{bar}] {pct}%", end="", flush=True)

    urllib.request.urlretrieve(VOSK_URL, zip_path, _progress)
    print()
    print()

    info("Extracting model...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(APP_DIR)
    zip_path.unlink()
    ok("Vosk model installed")
    print()


# ── Step 7: Data directory + launcher ─────────────────────────────────────────

def setup_data_and_launcher():
    step("Step 7 -- Data folder + Desktop launcher")

    data_dir = get_data_dir()
    desktop  = get_desktop()

    for subdir in ["", "fingerprints", "profiles", "simulations", "feedback", "crash_reports"]:
        (data_dir / subdir).mkdir(parents=True, exist_ok=True)
    ok(f"Data folder: {data_dir}")

    if IS_WIN:
        _create_windows_launcher(desktop, data_dir)
    elif IS_MAC:
        _create_mac_launcher(desktop, data_dir)
    else:
        _create_linux_launcher(desktop)

    print()


def _create_mac_launcher(desktop, data_dir):
    # Store the actual .command in the app folder (not Desktop)
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
    launcher.chmod(0o755)

    # Build .icns from the PNG
    icon_png  = APP_DIR / "TheRPSRobot.png"
    icon_icns = APP_DIR / "TheRPSRobot.icns"

    if icon_png.exists():
        try:
            iconset = APP_DIR / "TheRPSRobot.iconset"
            iconset.mkdir(exist_ok=True)
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
            # Open at full resolution for best quality downscaling
            src = Image.open(icon_png).convert("RGBA")
            for fname, sz in sizes.items():
                src.resize((sz, sz), Image.LANCZOS).save(
                    iconset / fname, optimize=True)
            result = subprocess.run(
                ["iconutil", "-c", "icns", str(iconset), "-o", str(icon_icns)],
                capture_output=True)
            import shutil as _sh
            _sh.rmtree(iconset, ignore_errors=True)
            if result.returncode == 0:
                ok("App icon (.icns) created at full quality")
            else:
                warn("iconutil failed - icon may not appear")
        except Exception as e:
            warn(f"Could not create .icns: {e}")

    # Apply icon to launcher and create Desktop alias via osascript
    if icon_icns.exists():
        try:
            # Set icon on the .command file
            subprocess.run(["osascript", "-l", "JavaScript", "-e", f'''
                ObjC.import("AppKit");
                var img = $.NSImage.alloc.initWithContentsOfFile("{icon_icns}");
                var ws  = $.NSWorkspace.sharedWorkspace;
                ws.setIconForFileOptions(img, "{launcher}", 0);
            '''], capture_output=True)
        except Exception:
            pass

    # Create a Desktop alias pointing to the launcher (cleaner than symlink)
    alias_name = "RPS Robot"
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
        # Fallback: symlink
        try:
            if not desktop_alias.exists():
                desktop_alias.symlink_to(launcher)
            ok(f"Desktop shortcut created: '{alias_name}'")
        except Exception as e:
            warn(f"Could not create Desktop icon: {e}")

    # Data folder shortcut
    symlink = desktop / "RPS Robot Data"
    if not symlink.exists():
        try:
            symlink.symlink_to(data_dir)
            ok("Data folder shortcut on Desktop")
        except Exception:
            pass


def _create_windows_launcher(desktop, data_dir):
    # Store the .bat in the APP folder (not on Desktop - keeps Desktop clean)
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

    # Build high-quality .ico (up to 256px - Windows ICO format limit)
    # Also save a separate 512px PNG for use where supported
    ico_path = APP_DIR / "TheRPSRobot.ico"
    png_path = APP_DIR / "TheRPSRobot.png"

    if png_path.exists():
        try:
            from PIL import Image
            src = Image.open(png_path).convert("RGBA")
            # Windows ICO max is 256x256 for the format
            # but we include all sizes for crisp rendering at every size
            ico_sizes = [(16,16),(24,24),(32,32),(48,48),(64,64),(96,96),(128,128),(256,256)]
            ico_imgs  = [src.resize(s, Image.LANCZOS) for s in ico_sizes]
            ico_imgs[0].save(
                str(ico_path), format="ICO",
                sizes=ico_sizes,
                append_images=ico_imgs[1:]
            )
            ok("App icon (.ico) created at full quality")
        except Exception as e:
            warn(f"Could not create .ico: {e}")

    # Create ONE clean .lnk shortcut on Desktop with the high-quality icon
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

    # Data folder shortcut
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


def _create_linux_launcher(desktop):
    launcher = desktop / "Launch RPS Robot.sh"
    launcher.write_text(textwrap.dedent(f"""\
        #!/bin/bash
        cd "{APP_DIR}"
        source "{VENV_DIR}/bin/activate"
        python main.py
    """))
    launcher.chmod(0o755)
    ok(f"Launcher created: {launcher}")


# ── Step 8: Verify ────────────────────────────────────────────────────────────

def verify_installation():
    step("Step 8 -- Verifying installation")
    print()

    vosk_path = APP_DIR / VOSK_MODEL

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
    line()
    print()
    ok("Installation complete!")
    print()

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
            subprocess.run([py, "main.py"])
        else:
            os.execv(py, [py, "main.py"])


if __name__ == "__main__":
    main()

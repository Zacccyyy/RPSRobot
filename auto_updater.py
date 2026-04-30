"""
auto_updater.py
===============
Checks GitHub for a newer version of RPS Robot and applies it automatically.

How it works:
  1. On app launch, a background thread silently asks the GitHub API:
     "What is the latest commit on the main branch?"
  2. We compare that to the current local commit hash (git rev-parse HEAD).
  3. If they differ -> show an update banner in the app.
  4. User presses U -> run `git pull` -> restart the app automatically.

No extra dependencies needed  -  uses only Python stdlib.
Repo: https://github.com/Zacccyyy/RPSRobot
"""

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error

# ── Config ────────────────────────────────────────────────────────────────────
GITHUB_OWNER = "Zacccyyy"
GITHUB_REPO  = "RPSRobot"
BRANCH       = "main"

# GitHub API: latest commit on main branch
API_URL = (
    f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}"
    f"/commits/{BRANCH}"
)

# How long before giving up on the network request (seconds)
REQUEST_TIMEOUT = 6

# ── State (shared between background thread and main app) ─────────────────────
_state = {
    "status":          "idle",    # idle | checking | up_to_date | update_available | error
    "remote_sha":      None,
    "local_sha":       None,
    "error_msg":       "",
    "last_checked":    0.0,
    "update_applied":  False,
}
_lock = threading.Lock()


def get_state() -> dict:
    """Return a snapshot of the updater state (thread-safe)."""
    with _lock:
        return dict(_state)


def _set(**kwargs):
    with _lock:
        _state.update(kwargs)


# ── Git helpers ───────────────────────────────────────────────────────────────

def _project_dir() -> str:
    """The directory containing this file — i.e. the project root."""
    return os.path.dirname(os.path.abspath(__file__))


def _run_git(*args, timeout=30) -> tuple[int, str, str]:
    """
    Run a git command in the project directory.
    Returns (returncode, stdout, stderr).
    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=_project_dir(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except FileNotFoundError:
        return 1, "", "git not found"
    except subprocess.TimeoutExpired:
        return 1, "", "git command timed out"
    except Exception as exc:
        return 1, "", str(exc)


def get_local_sha():
    """Return the current local HEAD commit hash, or None if not a git repo."""
    code, out, _ = _run_git("rev-parse", "HEAD")
    return out if code == 0 and out else None


def is_git_repo():
    code, _, _ = _run_git("rev-parse", "--is-inside-work-tree")
    return code == 0


# ── Network helper ────────────────────────────────────────────────────────────

def _fetch_remote_sha():
    """
    Hit the GitHub API and return the latest commit SHA on the main branch.
    Returns None on any network or parse error.
    """
    req = urllib.request.Request(
        API_URL,
        headers={
            "Accept":     "application/vnd.github+json",
            "User-Agent": "RPSRobot-AutoUpdater/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("sha")
    except urllib.error.URLError:
        return None   # no internet  -  silent fail
    except Exception:
        return None


# ── Core check ────────────────────────────────────────────────────────────────

def check_for_updates():
    """
    Perform a single update check synchronously.
    Intended to be called from a background thread.
    """
    if not is_git_repo():
        _set(status="error",
             error_msg="Not a git repo  -  re-install via git clone to enable updates.")
        return

    _set(status="checking", last_checked=time.time())

    local_sha = get_local_sha()
    _set(local_sha=local_sha)

    remote_sha = _fetch_remote_sha()
    if remote_sha is None:
        # No internet or API error — don't bother user
        _set(status="idle")
        return

    _set(remote_sha=remote_sha)

    if local_sha and remote_sha.startswith(local_sha[:12]):
        _set(status="up_to_date")
    elif local_sha == remote_sha:
        _set(status="up_to_date")
    else:
        _set(status="update_available")


def check_in_background():
    """
    Spawn a daemon thread to check for updates without blocking the app.
    Safe to call at startup  -  will not slow down launch.
    """
    t = threading.Thread(target=check_for_updates, daemon=True, name="UpdateChecker")
    t.start()


# ── Apply update ──────────────────────────────────────────────────────────────

def apply_update():
    """
    Run `git pull` to apply the latest update.
    Returns (success: bool, message: str).

    After a successful pull the caller should restart the app.
    """
    _set(status="checking")

    # Make sure we're on the right branch
    _run_git("checkout", BRANCH)

    code, out, err = _run_git("pull", "origin", BRANCH, timeout=60)
    if code == 0:
        _set(status="up_to_date", update_applied=True)
        return True, out or "Up to date."
    else:
        msg = err or out or "git pull failed  -  check your internet connection."
        _set(status="error", error_msg=msg)
        return False, msg


def restart_app():
    """
    Restart the current Python process in-place.
    Uses os.execv on macOS/Linux (clean in-place replace).
    Uses subprocess + sys.exit on Windows (execv is unreliable there).
    """
    python = sys.executable
    args   = [python] + sys.argv
    if sys.platform == "win32":
        subprocess.Popen(args)
        sys.exit(0)
    else:
        os.execv(python, args)


# ── Convenience: apply + restart ──────────────────────────────────────────────

def apply_and_restart(on_error=None):
    """
    Pull latest changes and restart the app immediately.
    If the pull fails, call on_error(message) if provided, then return.
    """
    success, msg = apply_update()
    if success:
        time.sleep(0.3)   # brief pause so UI can show "Updating..."
        restart_app()
    else:
        if on_error:
            on_error(msg)


# ── Human-readable status ─────────────────────────────────────────────────────

def status_label() -> str:
    s = get_state()
    if s["status"] == "update_available":
        local  = (s["local_sha"]  or "?")[:7]
        remote = (s["remote_sha"] or "?")[:7]
        return f"Update available  ({local} -> {remote})  Press U to update"
    if s["status"] == "checking":
        return "Checking for updates..."
    if s["status"] == "error":
        return f"Updater: {s['error_msg'][:60]}"
    return ""

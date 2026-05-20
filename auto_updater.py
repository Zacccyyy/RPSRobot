"""
auto_updater.py
===============
Checks GitHub for a newer version of RPS Robot and applies it automatically.

How it works:
  1. On app launch a background thread silently asks the GitHub API:
     "What is the latest commit on the main branch?"
  2. We compare that to the current local commit hash (git rev-parse HEAD).
  3. If they differ → show an update banner in the app.
  4. User presses U → run `git pull` → restart the app automatically.

No extra dependencies — uses only Python stdlib.
Repo: https://github.com/Zacccyyy/RPSRobot
"""

from __future__ import annotations
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

# GitHub REST API endpoint that returns info about the latest commit on BRANCH.
API_URL = (
    f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}"
    f"/commits/{BRANCH}"
)

# How long to wait for the network request before giving up (seconds).
REQUEST_TIMEOUT = 6

# ── Shared state ──────────────────────────────────────────────────────────────
# This dict is read by the main thread and written by the background thread,
# so every access goes through _lock.

_state = {
    "status":         "idle",   # idle | checking | up_to_date | update_available | error
    "remote_sha":     None,     # latest commit SHA from GitHub
    "local_sha":      None,     # current local HEAD SHA
    "error_msg":      "",
    "last_checked":   0.0,      # monotonic timestamp of the last check
    "update_applied": False,    # True after a successful git pull
}
_lock = threading.Lock()


def get_state() -> dict:
    """Return a snapshot of the updater state dict (thread-safe copy)."""
    with _lock:
        return dict(_state)


def _set(**kwargs):
    """Update one or more fields in the shared state dict (thread-safe)."""
    with _lock:
        _state.update(kwargs)


# ── Git helpers ───────────────────────────────────────────────────────────────

def _project_dir() -> str:
    """Return the directory containing this file — that's the project root."""
    return os.path.dirname(os.path.abspath(__file__))


def _run_git(*args, timeout=30) -> tuple[int, str, str]:
    """
    Run a git command in the project directory.
    Returns (returncode, stdout, stderr) so callers can check for errors.
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


def get_local_sha() -> str | None:
    """Return the current local HEAD commit hash, or None if not a git repo."""
    code, out, _ = _run_git("rev-parse", "HEAD")
    return out if code == 0 and out else None


def is_git_repo() -> bool:
    """Return True if the project directory is inside a git repository."""
    code, _, _ = _run_git("rev-parse", "--is-inside-work-tree")
    return code == 0


# ── Network helper ────────────────────────────────────────────────────────────

def _fetch_remote_sha() -> str | None:
    """
    Hit the GitHub API and return the latest commit SHA on the main branch.
    Returns None on any network or parse error — always fails silently.
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
        return None  # no internet — silent fail
    except Exception:
        return None


# ── Core check ────────────────────────────────────────────────────────────────

def check_for_updates():
    """
    Perform a single update check synchronously.
    Compares the local HEAD SHA against the remote HEAD SHA and updates state.
    Intended to be called from a background thread via check_in_background().
    """
    if not is_git_repo():
        _set(status="error",
             error_msg="Not a git repo — re-install via git clone to enable updates.")
        return

    _set(status="checking", last_checked=time.time())

    # Get the local commit hash first.
    local_sha = get_local_sha()
    _set(local_sha=local_sha)

    # Ask GitHub for the latest commit.
    remote_sha = _fetch_remote_sha()
    if remote_sha is None:
        # No internet or API error — don't bother the user.
        _set(status="idle")
        return

    _set(remote_sha=remote_sha)

    # Compare the first 12 characters — enough to uniquely identify a commit.
    if local_sha and remote_sha.startswith(local_sha[:12]):
        _set(status="up_to_date")
    else:
        _set(status="update_available")


def check_in_background():
    """
    Spawn a daemon thread that checks for updates without blocking the app.
    Safe to call at startup — will not slow down launch.
    """
    t = threading.Thread(target=check_for_updates, daemon=True, name="UpdateChecker")
    t.start()


# ── Apply update ──────────────────────────────────────────────────────────────

def apply_update() -> tuple[bool, str]:
    """
    Run `git pull` to download and apply the latest update.
    Returns (success: bool, message: str).

    After a successful pull the caller should call restart_app().
    """
    _set(status="checking")

    # Make sure we're on the correct branch before pulling.
    _run_git("checkout", BRANCH)

    code, out, err = _run_git("pull", "origin", BRANCH, timeout=60)
    if code == 0:
        _set(status="up_to_date", update_applied=True)
        return True, out or "Up to date."
    else:
        msg = err or out or "git pull failed — check your internet connection."
        _set(status="error", error_msg=msg)
        return False, msg


def restart_app():
    """
    Restart the current Python process in-place.

    On macOS/Linux: uses os.execv — replaces the process cleanly.
    On Windows: uses subprocess + sys.exit — execv is unreliable there.
    """
    python = sys.executable
    args   = [python] + sys.argv

    if sys.platform == "win32":
        # Windows doesn't support os.execv reliably, so spawn a new process.
        subprocess.Popen(args)
        sys.exit(0)
    else:
        # Replace this process entirely — same PID, fresh Python.
        os.execv(python, args)


def apply_and_restart(on_error=None):
    """
    Pull the latest changes and immediately restart the app.

    on_error : optional callable that receives an error message string
               if the pull fails.  The app is NOT restarted on failure.
    """
    success, msg = apply_update()
    if success:
        time.sleep(0.3)  # brief pause so the UI can show "Updating..." before we exit
        restart_app()
    else:
        if on_error:
            on_error(msg)


# ── Human-readable status ─────────────────────────────────────────────────────

def status_label() -> str:
    """
    Return a short status string suitable for displaying in the app's UI.

    Shows abbreviated commit hashes when an update is available, a spinner
    message while checking, the error message on failure, or an empty string
    when everything is fine (no need to show anything).
    """
    s = get_state()

    if s["status"] == "update_available":
        local  = (s["local_sha"]  or "?")[:7]
        remote = (s["remote_sha"] or "?")[:7]
        return f"Update available  ({local} -> {remote})  Press U to update"

    if s["status"] == "checking":
        return "Checking for updates..."

    if s["status"] == "error":
        return f"Updater: {s['error_msg'][:60]}"

    return ""  # up_to_date or idle — nothing to show

"""
feedback_store.py
=================
Saves player feedback/suggestions to timestamped .txt files on disk.

Each submission creates one file:
    ~/Desktop/CapStone/feedback/YYYY-MM-DD_HH-MM-SS_<player>.txt

File format:
    Player:    Zac
    Submitted: 2026-04-29 22:15:30
    Version:   abc1234 (git short hash)

    <feedback text>

The developer can review these files at any time from the CapStone folder.
This module only handles local storage — sentry_reporter.py handles sending
feedback to the developer remotely (if the player consented).
"""

import time
from pathlib import Path


# Where feedback files are saved on disk
FEEDBACK_DIR = Path.home() / "Desktop" / "CapStone" / "feedback"


def save_feedback(player_name: str, text: str, git_sha: str = "") -> Path:
    """
    Save a feedback submission to a timestamped .txt file.

    Sanitises the player name so it is safe to use in a filename (replaces
    any non-alphanumeric characters with underscores).

    Returns the Path of the file that was written.
    """
    # Create the feedback directory if it doesn't exist yet
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

    # Build a filename like: 2026-04-29_22-15-30_Zac.txt
    ts   = time.strftime("%Y-%m-%d_%H-%M-%S")
    safe = "".join(
        c if (c.isascii() and c.isalnum()) or c in "-_" else "_"
        for c in (player_name or "unknown")
    )
    fpath = FEEDBACK_DIR / f"{ts}_{safe}.txt"

    # Truncate the git SHA to 7 characters (the standard "short hash" length)
    version = git_sha[:7] if git_sha else "unknown"

    # Build the text content and write it out
    content = (
        f"Player:    {player_name or 'unknown'}\n"
        f"Submitted: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Version:   {version}\n"
        f"\n"
        f"{text.strip()}\n"
    )

    fpath.write_text(content, encoding="utf-8")
    print(f"[Feedback] Saved to {fpath.name}")
    return fpath


def list_feedback() -> list:
    """
    Return a list of (filename, player, timestamp, preview) tuples for every
    feedback file, newest first.

    Any file that can't be parsed is silently skipped (e.g. corrupted files).
    """
    # If no feedback has ever been submitted, just return an empty list
    if not FEEDBACK_DIR.exists():
        return []

    results = []
    # Sort in reverse so the most recent file comes first
    for f in sorted(FEEDBACK_DIR.glob("*.txt"), reverse=True):
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
            # Line 0: "Player:    Zac"  →  strip the label to get just "Zac"
            player  = lines[0].replace("Player:", "").strip() if lines else "?"
            # Line 1: "Submitted: 2026-04-29 22:15:30"
            ts      = lines[1].replace("Submitted:", "").strip() if len(lines) > 1 else "?"
            # Line 4 is the first line of the actual feedback text
            preview = lines[4][:60] if len(lines) > 4 else ""
            results.append((f.name, player, ts, preview))
        except Exception:
            # Skip any file we can't parse cleanly
            pass

    return results

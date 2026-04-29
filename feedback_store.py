"""
feedback_store.py
=================
Saves player feedback/suggestions to timestamped .txt files.

Each submission creates one file:
    ~/Desktop/CapStone/feedback/YYYY-MM-DD_HH-MM-SS_<player>.txt

Format:
    Player:    Zac
    Submitted: 2026-04-29 22:15:30
    Version:   abc1234 (git short hash)

    <feedback text>

The developer can review these files at any time from the CapStone folder.
"""

import time
import os
from pathlib import Path


FEEDBACK_DIR = Path.home() / "Desktop" / "CapStone" / "feedback"


def save_feedback(player_name: str, text: str, git_sha: str = "") -> Path:
    """
    Save a feedback submission to a timestamped .txt file.
    Returns the path of the saved file.
    """
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

    ts      = time.strftime("%Y-%m-%d_%H-%M-%S")
    safe    = "".join(c if c.isalnum() or c in "-_" else "_"
                      for c in (player_name or "unknown"))
    fname   = f"{ts}_{safe}.txt"
    fpath   = FEEDBACK_DIR / fname

    version = git_sha[:7] if git_sha else "unknown"
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
    """Return list of (filename, player, timestamp, preview) tuples."""
    if not FEEDBACK_DIR.exists():
        return []
    results = []
    for f in sorted(FEEDBACK_DIR.glob("*.txt"), reverse=True):
        try:
            lines   = f.read_text(encoding="utf-8").splitlines()
            player  = lines[0].replace("Player:", "").strip() if lines else "?"
            ts      = lines[1].replace("Submitted:", "").strip() if len(lines) > 1 else "?"
            preview = lines[4][:60] if len(lines) > 4 else ""
            results.append((f.name, player, ts, preview))
        except Exception:
            pass
    return results

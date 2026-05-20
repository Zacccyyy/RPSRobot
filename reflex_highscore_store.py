"""
reflex_highscore_store.py
=========================
Persistent high-score storage for the Speed Reflex solo mode.

Keeps the top TOP_N all-time results on disk. Each entry records the
player name, score (number of correct throws), average reaction time,
and a timestamp.

Saved as a simple JSON list at:
    ~/Desktop/CapStone/reflex_highscores.json

Typical usage:
    store = ReflexHighscoreStore()
    is_new_best, rank = store.submit("Zac", score=18, avg_rt_ms=312)
    top  = store.get_top()   # list of dicts, best first
    best = store.get_best()  # single dict or None
"""

import json
import time
from pathlib import Path
from typing import List, Optional

# How many scores to keep on the leaderboard.
TOP_N = 10

# Where the scores file is stored.
SAVE_FILE = "reflex_highscores.json"
DATA_DIR  = Path.home() / "Desktop" / "CapStone"


class ReflexHighscoreStore:
    """
    Manages the Speed Reflex leaderboard.

    Scores are ranked primarily by score (higher is better).
    Ties are broken by average reaction time (lower is better).
    Only the top TOP_N entries are kept; the rest are dropped.
    """

    def __init__(self, data_dir=None):
        # Allow tests or alternative deployments to specify a different directory.
        self._dir  = Path(data_dir) if data_dir else DATA_DIR
        self._path = self._dir / SAVE_FILE

        # In-memory score list -- loaded from disk immediately.
        self._scores: list[dict] = []
        self._load()

    # -- Public API -----------------------------------------------------------

    def submit(self, player_name: str, score: int, avg_rt_ms: int) -> tuple:
        """
        Record a new run result and update the leaderboard.

        player_name -- display name (empty string becomes "Unknown")
        score       -- number of correct throws in the session
        avg_rt_ms   -- average reaction time in milliseconds

        Returns (is_new_highscore, rank_1indexed):
            is_new_highscore -- True only if this score beats the previous all-time best
            rank_1indexed    -- position in the leaderboard (1 = top)
        """
        entry = {
            "player": player_name.strip() or "Unknown",
            "score":  score,
            "avg_rt": avg_rt_ms,
            "ts":     time.strftime("%Y-%m-%d %H:%M"),
        }

        # Snapshot the current best score before we add the new entry.
        prev_best = self._scores[0]["score"] if self._scores else -1

        self._scores.append(entry)

        # Sort: highest score first; on a tie, lowest reaction time wins.
        self._scores.sort(key=lambda e: (-e["score"], e["avg_rt"]))

        # Trim to the top TOP_N entries (discard anything below the cutoff).
        self._scores = self._scores[:TOP_N]

        # Find this entry's rank. We try identity (same object) first, then
        # fall back to matching all three fields in case sort rebuilt the list.
        rank = next(
            (i + 1 for i, e in enumerate(self._scores)
             if e is entry or
             (e["player"] == entry["player"] and
              e["score"]  == entry["score"]  and
              e["ts"]     == entry["ts"])),
            TOP_N  # default to last place if the entry was trimmed off the board
        )

        is_new_best = score > prev_best
        self._save()
        return is_new_best, rank

    def get_top(self) -> List[dict]:
        """Return a copy of the leaderboard list (best first, up to TOP_N entries)."""
        return list(self._scores)

    def get_best(self) -> Optional[dict]:
        """Return the all-time best entry, or None if the leaderboard is empty."""
        return self._scores[0] if self._scores else None

    def clear(self):
        """Wipe all scores from the leaderboard and disk (useful for testing)."""
        self._scores = []
        self._save()

    # -- Internal helpers -----------------------------------------------------

    def _load(self):
        """
        Load scores from disk.

        If the file is missing or corrupt we start with an empty list
        rather than crashing -- the file will be created on the next save.
        """
        try:
            if self._path.exists():
                with open(self._path, "r") as f:
                    data = json.load(f)
                # Guard against a corrupted file that isn't a list.
                self._scores = data if isinstance(data, list) else []
        except Exception:
            self._scores = []

    def _save(self):
        """
        Persist the current leaderboard to disk as JSON.
        Creates the parent directory first if it doesn't exist yet.
        """
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w") as f:
                json.dump(self._scores, f, indent=2)
        except Exception as e:
            print(f"[ReflexHighscoreStore] Could not save: {e}")

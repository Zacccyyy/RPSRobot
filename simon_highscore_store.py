"""
simon_highscore_store.py
========================
Persistent high-score storage for the Simon Says solo mode.

Scores are ranked by the longest gesture sequence the player completed
(seq_length), with rounds completed as a tiebreaker.  The top TOP_N
results are kept on disk.

Saved at:
    ~/Desktop/CapStone/simon_highscores.json

Typical usage:
    store = SimonHighscoreStore()
    is_new_best, rank = store.submit("Zac", score=12, seq_length=8)
    top  = store.get_top()    # list of dicts, best first
    best = store.get_best()   # single dict or None
"""

import json
import time
from pathlib import Path
from typing import Optional, List

# How many scores the leaderboard holds.
TOP_N = 10

# Filename and default storage directory.
SAVE_FILE = "simon_highscores.json"
DATA_DIR  = Path.home() / "Desktop" / "CapStone"


class SimonHighscoreStore:
    """
    Manages the Simon Says leaderboard.

    Rankings are determined by:
      1. Longest sequence chain reached (higher is better).
      2. Rounds completed as a tiebreaker (higher is better).
    """

    def __init__(self, data_dir=None):
        # Allow tests or alternative deployments to pass in a different directory.
        self._dir   = Path(data_dir) if data_dir else DATA_DIR
        self._path  = self._dir / SAVE_FILE
        # In-memory score list — loaded from disk right away.
        self._scores: list = []
        self._load()

    def submit(self, player_name: str, score: int, seq_length: int) -> tuple:
        """
        Record a new Simon Says result and update the leaderboard.

        Parameters
        ----------
        player_name : str  -- display name (empty string becomes "Unknown")
        score       : int  -- total rounds the player completed
        seq_length  : int  -- longest correct gesture chain reached

        Returns
        -------
        (is_new_best, rank_1indexed)
            is_new_best   -- True only if seq_length beats the previous all-time best
            rank_1indexed -- leaderboard position (1 = top)
        """
        entry = {
            "player":     player_name.strip() or "Unknown",
            "score":      score,
            "seq_length": seq_length,
            "ts":         time.strftime("%Y-%m-%d %H:%M"),
        }

        # Snapshot the current best sequence length before adding the new entry.
        prev_best = self._scores[0]["seq_length"] if self._scores else -1

        self._scores.append(entry)
        # Sort: longest sequence first; on a tie, most rounds completed wins.
        self._scores.sort(key=lambda e: (-e["seq_length"], -e["score"]))
        # Keep only the top TOP_N entries.
        self._scores = self._scores[:TOP_N]

        # Find the rank.  Match by identity first, then by all three fields.
        rank = next((i + 1 for i, e in enumerate(self._scores)
                     if e is entry or (e["player"] == entry["player"]
                                       and e["score"]  == entry["score"]
                                       and e["ts"]     == entry["ts"])), TOP_N)

        is_new_best = seq_length > prev_best
        self._save()
        return is_new_best, rank

    def get_top(self) -> List[dict]:
        """Return a copy of the leaderboard (best first, up to TOP_N entries)."""
        return list(self._scores)

    def get_best(self) -> Optional[dict]:
        """Return the all-time best entry, or None if no scores exist yet."""
        return self._scores[0] if self._scores else None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load(self):
        """
        Load scores from disk.  Starts with an empty list if the file is
        missing or corrupt — the file will be created on the next save.
        """
        try:
            if self._path.exists():
                with open(self._path) as f:
                    data = json.load(f)
                # Guard against a corrupt file that isn't a list.
                self._scores = data if isinstance(data, list) else []
        except Exception:
            self._scores = []

    def _save(self):
        """
        Write the current leaderboard to disk as JSON.
        Creates the parent directory if needed.
        """
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w") as f:
                json.dump(self._scores, f, indent=2)
        except Exception as e:
            print(f"[SimonHighscoreStore] Could not save: {e}")

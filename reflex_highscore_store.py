"""
reflex_highscore_store.py
=========================
Persistent highscore storage for Speed Reflex solo mode.

Stores up to TOP_N all-time scores with player name, score, avg reaction time
and timestamp.  Saved as JSON at ~/Desktop/CapStone/reflex_highscores.json.

Usage:
    store = ReflexHighscoreStore()
    updated, rank = store.submit(player_name="Zac", score=18, avg_rt_ms=312)
    top = store.get_top()       # list of dicts, best first
    best = store.get_best()     # single dict or None
"""

import json
import time
from pathlib import Path
from typing import Optional, List

TOP_N       = 10          # how many scores to keep
SAVE_FILE   = "reflex_highscores.json"
DATA_DIR    = Path.home() / "Desktop" / "CapStone"


class ReflexHighscoreStore:

    def __init__(self, data_dir=None):
        self._dir  = Path(data_dir) if data_dir else DATA_DIR
        self._path = self._dir / SAVE_FILE
        self._scores: list[dict] = []
        self._load()

    # ── Public API ────────────────────────────────────────────────────────

    def submit(self, player_name: str, score: int, avg_rt_ms: int) -> tuple:
        """
        Add a new run result.  Returns (is_new_highscore, rank_1indexed).
        is_new_highscore is True only if this score is strictly better than
        the previous all-time best (or the board was empty).
        """
        entry = {
            "player": player_name.strip() or "Unknown",
            "score":  score,
            "avg_rt": avg_rt_ms,
            "ts":     time.strftime("%Y-%m-%d %H:%M"),
        }

        prev_best = self._scores[0]["score"] if self._scores else -1

        self._scores.append(entry)
        # Sort: highest score first; ties broken by lowest avg reaction time
        self._scores.sort(key=lambda e: (-e["score"], e["avg_rt"]))
        self._scores = self._scores[:TOP_N]

        rank = next((i + 1 for i, e in enumerate(self._scores)
                     if e is entry or
                     (e["player"] == entry["player"] and
                      e["score"] == entry["score"] and
                      e["ts"] == entry["ts"])), TOP_N)

        is_new_best = score > prev_best
        self._save()
        return is_new_best, rank

    def get_top(self) -> List[dict]:
        """Return up to TOP_N scores, best first."""
        return list(self._scores)

    def get_best(self) -> Optional[dict]:
        """Return the all-time best entry, or None if no scores yet."""
        return self._scores[0] if self._scores else None

    def clear(self):
        """Wipe all scores (useful for testing)."""
        self._scores = []
        self._save()

    # ── Internal ──────────────────────────────────────────────────────────

    def _load(self):
        try:
            if self._path.exists():
                with open(self._path, "r") as f:
                    data = json.load(f)
                self._scores = data if isinstance(data, list) else []
        except Exception:
            self._scores = []

    def _save(self):
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w") as f:
                json.dump(self._scores, f, indent=2)
        except Exception as e:
            print(f"[ReflexHighscoreStore] Could not save: {e}")

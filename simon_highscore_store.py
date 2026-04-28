"""
simon_highscore_store.py
========================
Persistent highscore storage for Simon Says solo mode.
Scores ranked by longest sequence length, then rounds completed.
Saved at ~/Desktop/CapStone/simon_highscores.json.
"""

import json
import time
from pathlib import Path
from typing import Optional, List

TOP_N     = 10
SAVE_FILE = "simon_highscores.json"
DATA_DIR  = Path.home() / "Desktop" / "CapStone"


class SimonHighscoreStore:

    def __init__(self, data_dir=None):
        self._dir   = Path(data_dir) if data_dir else DATA_DIR
        self._path  = self._dir / SAVE_FILE
        self._scores: list = []
        self._load()

    def submit(self, player_name: str, score: int, seq_length: int) -> tuple:
        """score = rounds completed, seq_length = longest chain reached.
        Returns (is_new_best, rank_1indexed)."""
        entry = {
            "player":     player_name.strip() or "Unknown",
            "score":      score,
            "seq_length": seq_length,
            "ts":         time.strftime("%Y-%m-%d %H:%M"),
        }
        prev_best = self._scores[0]["seq_length"] if self._scores else -1
        self._scores.append(entry)
        self._scores.sort(key=lambda e: (-e["seq_length"], -e["score"]))
        self._scores = self._scores[:TOP_N]
        rank = next((i + 1 for i, e in enumerate(self._scores)
                     if e is entry or (e["player"] == entry["player"]
                     and e["score"] == entry["score"]
                     and e["ts"] == entry["ts"])), TOP_N)
        is_new_best = seq_length > prev_best
        self._save()
        return is_new_best, rank

    def get_top(self) -> List[dict]:
        return list(self._scores)

    def get_best(self) -> Optional[dict]:
        return self._scores[0] if self._scores else None

    def _load(self):
        try:
            if self._path.exists():
                with open(self._path) as f:
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
            print(f"[SimonHighscoreStore] Could not save: {e}")

"""
Front-On Landmark Data Collector

Collects normalised hand landmark data for training the front-on
gesture classifier. Designed to be used from Diagnostic mode.

Usage (from main.py):
    Press 'F' to toggle collection mode ON/OFF
    While collection mode is ON:
        Press '7' to record current landmarks as Rock
        Press '8' to record current landmarks as Scissors
        Press '9' to record current landmarks as Paper

Data is saved to ~/Desktop/CapStone/front_on_training_data.csv

Each row contains:
    label, x0, y0, x1, y1, ... x20, y20  (43 values)

Coordinates are normalised:
    - Translated so wrist is at origin (0,0)
    - Scaled by palm size so hand distance doesn't matter
"""

import csv
import os
from pathlib import Path


LABEL_MAP = {
    ord("7"): "Rock",
    ord("8"): "Scissors",
    ord("9"): "Paper",
}

CSV_HEADER = ["label"] + [f"{ax}{i}" for i in range(21) for ax in ("x", "y")]


class LandmarkCollector:

    def __init__(self, output_dir=None):
        self.base_dir = Path(output_dir) if output_dir else Path.home() / "Desktop" / "CapStone"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.base_dir / "front_on_training_data.csv"

        self.active = False
        self.last_landmarks = None
        self.sample_counts = {"Rock": 0, "Paper": 0, "Scissors": 0}

        self._ensure_csv()
        self._count_existing()

    def _ensure_csv(self):
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADER)

    def _count_existing(self):
        """Count existing samples per label."""
        if not self.csv_path.exists():
            return

        try:
            with open(self.csv_path, "r") as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                for row in reader:
                    if row and row[0] in self.sample_counts:
                        self.sample_counts[row[0]] += 1
        except Exception:
            pass

    def toggle(self):
        self.active = not self.active
        state = "ON" if self.active else "OFF"
        print(f"[Collector] Data collection {state}")
        print(f"[Collector] Samples: {self.sample_counts}")
        return self.active

    def update_landmarks(self, hand_landmarks):
        """Call every frame with the current hand landmarks (or None)."""
        self.last_landmarks = hand_landmarks

    def try_record(self, key):
        """
        Try to record current landmarks with the given key.
        Returns (recorded: bool, label: str or None, message: str).
        """
        if not self.active:
            return False, None, ""

        if key not in LABEL_MAP:
            return False, None, ""

        if self.last_landmarks is None:
            return False, None, "No hand detected"

        label = LABEL_MAP[key]
        row = self._normalise_landmarks(self.last_landmarks)

        if row is None:
            return False, None, "Could not normalise landmarks"

        try:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([label] + row)

            self.sample_counts[label] += 1
            total = sum(self.sample_counts.values())
            msg = (
                f"Saved {label} (R:{self.sample_counts['Rock']} "
                f"S:{self.sample_counts['Scissors']} "
                f"P:{self.sample_counts['Paper']} "
                f"total:{total})"
            )
            print(f"[Collector] {msg}")
            return True, label, msg

        except Exception as exc:
            return False, None, f"Save error: {exc}"

    def _normalise_landmarks(self, hand_landmarks):
        """
        Normalise 21 landmarks relative to wrist, scaled by palm size.
        Returns flat list of 42 floats [x0, y0, x1, y1, ...] or None.
        """
        lm = hand_landmarks.landmark

        if len(lm) < 21:
            return None

        wrist_x = lm[0].x
        wrist_y = lm[0].y

        # Palm size: average distance from wrist to MCP joints.
        import math
        def _d(a, b):
            return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

        d1 = _d(lm[0], lm[5])
        d2 = _d(lm[0], lm[9])
        d3 = _d(lm[0], lm[17])
        palm_scale = max((d1 + d2 + d3) / 3.0, 1e-6)

        row = []
        for i in range(21):
            nx = (lm[i].x - wrist_x) / palm_scale
            ny = (lm[i].y - wrist_y) / palm_scale
            row.append(round(nx, 5))
            row.append(round(ny, 5))

        return row

    def get_status_text(self):
        if not self.active:
            return ""

        return (
            f"COLLECTING: 7=Rock 8=Scissors 9=Paper | "
            f"R:{self.sample_counts['Rock']} "
            f"S:{self.sample_counts['Scissors']} "
            f"P:{self.sample_counts['Paper']}"
        )

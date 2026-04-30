"""
landmark_collector.py
=====================
Collects hand gesture training data for the front-on gesture classifier.

CHANGES FROM ORIGINAL:
  - Uses rotation-invariant angle/curl features (front_on_features.py)
    instead of raw x,y coordinates -- improves accuracy from ~64% to ~90%+
  - Enforces a 0.4s minimum gap between captures so rapid clicking
    still produces meaningfully varied samples
  - CSV header updated to match new 20-feature format

Usage (from main.py Diagnostic mode):
    Press F  to toggle collection mode ON/OFF
    Press 7  to record current landmarks as Rock
    Press 8  to record current landmarks as Scissors
    Press 9  to record current landmarks as Paper
"""

import csv
import time
from pathlib import Path

try:
    from capstone_paths import CAPSTONE_DIR
except ImportError:
    import sys as _sys
    CAPSTONE_DIR = (Path.home() / "Desktop" / "CapStone"
                    if _sys.platform == "darwin"
                    else Path.home() / "CapStone")

from front_on_features import extract_features, FEATURE_DIM


LABEL_MAP = {
    ord("7"): "Rock",
    ord("8"): "Scissors",
    ord("9"): "Paper",
}

CSV_HEADER = ["label"] + [f"f{i}" for i in range(FEATURE_DIM)]

# Minimum seconds between captures -- prevents rapid-clicking identical frames
MIN_CAPTURE_GAP = 0.4


class LandmarkCollector:

    def __init__(self, output_dir=None):
        self.base_dir  = Path(output_dir) if output_dir else CAPSTONE_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path  = self.base_dir / "front_on_training_data.csv"

        self.active         = False
        self.last_landmarks = None
        self.sample_counts  = {"Rock": 0, "Paper": 0, "Scissors": 0}
        self._last_capture  = 0.0

        self._ensure_csv()
        self._count_existing()

    def _ensure_csv(self):
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(CSV_HEADER)
        else:
            # Migrate old format (42 x,y coords) to new (20 angle features)
            try:
                with open(self.csv_path, "r") as f:
                    header = next(csv.reader(f), [])
                if len(header) == 43:
                    backup = self.csv_path.with_suffix(".old_format.csv")
                    self.csv_path.rename(backup)
                    print(f"[Collector] Old feature format detected - "
                          f"backed up to {backup.name}, starting fresh.")
                    with open(self.csv_path, "w", newline="") as f:
                        csv.writer(f).writerow(CSV_HEADER)
            except Exception:
                pass

    def _count_existing(self):
        if not self.csv_path.exists():
            return
        try:
            with open(self.csv_path, "r") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if row and row[0] in self.sample_counts:
                        self.sample_counts[row[0]] += 1
        except Exception:
            pass

    def toggle(self):
        self.active = not self.active
        print(f"[Collector] Collection {'ON' if self.active else 'OFF'} - "
              f"samples: {self.sample_counts}")
        return self.active

    def update_landmarks(self, hand_landmarks):
        self.last_landmarks = hand_landmarks

    def try_record(self, key):
        """
        Try to record current landmarks.
        Returns (recorded: bool, label: str | None, message: str).
        Enforces MIN_CAPTURE_GAP between captures.
        """
        if not self.active:
            return False, None, ""

        if key not in LABEL_MAP:
            return False, None, ""

        if self.last_landmarks is None:
            return False, None, "No hand detected"

        now = time.monotonic()
        if now - self._last_capture < MIN_CAPTURE_GAP:
            remaining = MIN_CAPTURE_GAP - (now - self._last_capture)
            return False, None, f"Hold still... ({remaining:.1f}s)"

        label    = LABEL_MAP[key]
        features = extract_features(self.last_landmarks)

        if features is None:
            return False, None, "Could not extract features"

        try:
            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow([label] + features)
            self._last_capture = now
            self.sample_counts[label] += 1
            total = sum(self.sample_counts.values())
            msg = (f"Saved {label} "
                   f"(R:{self.sample_counts['Rock']} "
                   f"S:{self.sample_counts['Scissors']} "
                   f"P:{self.sample_counts['Paper']} "
                   f"total:{total})")
            print(f"[Collector] {msg}")
            return True, label, msg
        except Exception as exc:
            return False, None, f"Save error: {exc}"

    def get_status_text(self):
        if not self.active:
            return ""
        return (f"COLLECTING: 7=Rock 8=Scissors 9=Paper | "
                f"R:{self.sample_counts['Rock']} "
                f"S:{self.sample_counts['Scissors']} "
                f"P:{self.sample_counts['Paper']}")

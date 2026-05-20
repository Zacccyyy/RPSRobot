# landmark_collector.py
# ----------------------
# Collects hand gesture training data for the front-on gesture classifier.
# Run from Diagnostic mode in main.py — press F to toggle collection on/off,
# then press 7/8/9 to save the current hand pose as Rock/Scissors/Paper.
#
# Samples are saved to a CSV file that the classifier is later trained from.
# A minimum 0.4s gap between captures stops rapid clicking from recording
# identical frames and polluting the training data.
#
# CHANGES FROM ORIGINAL:
#   - Uses rotation-invariant angle/curl features (front_on_features.py)
#     instead of raw x,y coordinates -- improves accuracy from ~64% to ~90%+
#   - Enforces a 0.4s minimum gap between captures so rapid clicking
#     still produces meaningfully varied samples
#   - CSV header updated to match new 20-feature format
#
# Key bindings:
#   F  -- toggle collection mode ON / OFF
#   7  -- record current pose as Rock
#   8  -- record current pose as Scissors
#   9  -- record current pose as Paper

import csv
import time
from pathlib import Path

# Try the shared path config first; fall back to the standard Desktop location.
try:
    from capstone_paths import CAPSTONE_DIR
except ImportError:
    import sys as _sys
    CAPSTONE_DIR = (
        Path.home() / "Desktop" / "CapStone"
        if _sys.platform == "darwin"
        else Path.home() / "CapStone"
    )

from front_on_features import extract_features, FEATURE_DIM


# Maps key codes to gesture labels so try_record() doesn't need any if/elif chains.
LABEL_MAP = {
    ord("7"): "Rock",
    ord("8"): "Scissors",
    ord("9"): "Paper",
}

# CSV header: "label" followed by one column per feature (f0, f1, ..., fN).
CSV_HEADER = ["label"] + [f"f{i}" for i in range(FEATURE_DIM)]

# Minimum time between captures in seconds. Prevents rapid clicking from saving
# frames that are too similar to be useful for training.
MIN_CAPTURE_GAP = 0.4


class LandmarkCollector:
    """
    Manages gesture sample collection in Diagnostic mode.

    Keeps a running count of saved samples per label, writes to a CSV file,
    and enforces a minimum gap between captures. The caller passes in
    landmarks each frame via update_landmarks(), then triggers recording
    via try_record() when a key is pressed.
    """

    def __init__(self, output_dir=None):
        # Default to the shared CapStone folder; allow override for testing.
        self.base_dir = Path(output_dir) if output_dir else CAPSTONE_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.base_dir / "front_on_training_data.csv"

        self.active         = False    # True while the user has collection turned on
        self.last_landmarks = None     # the most recent hand landmarks from the camera
        self.sample_counts  = {"Rock": 0, "Paper": 0, "Scissors": 0}
        self._last_capture  = 0.0     # monotonic timestamp of the last saved sample

        # Prepare the CSV file (create it or migrate old format if needed).
        self._ensure_csv()
        # Count any samples already in the file so the counter starts correctly.
        self._count_existing()

    def _ensure_csv(self):
        """
        Create the CSV file with the correct header if it doesn't exist.
        If an old-format file is detected (42-column x,y format), rename it
        as a backup and start a fresh file — the old data isn't compatible
        with the new feature extractor.
        """
        if not self.csv_path.exists():
            # New file — just write the header.
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(CSV_HEADER)
            return

        # File already exists — check whether it uses the old 42-column format.
        try:
            with open(self.csv_path, "r") as f:
                header = next(csv.reader(f), [])

            # Old format had 42 columns (1 label + 21 landmarks * 2 coords).
            if len(header) == 43:
                backup = self.csv_path.with_suffix(".old_format.csv")
                self.csv_path.rename(backup)
                print(f"[Collector] Old feature format detected - "
                      f"backed up to {backup.name}, starting fresh.")
                with open(self.csv_path, "w", newline="") as f:
                    csv.writer(f).writerow(CSV_HEADER)
        except Exception:
            pass  # if anything goes wrong reading, leave the file alone

    def _count_existing(self):
        """
        Read through the CSV and count how many samples exist per label.
        This makes the sample count display accurate when re-opening the app
        with an already-populated training file.
        """
        if not self.csv_path.exists():
            return
        try:
            with open(self.csv_path, "r") as f:
                reader = csv.reader(f)
                next(reader, None)  # skip the header row
                for row in reader:
                    # Each row starts with the label name.
                    if row and row[0] in self.sample_counts:
                        self.sample_counts[row[0]] += 1
        except Exception:
            pass  # don't crash if the file is somehow unreadable

    def toggle(self):
        """
        Flip the active flag on or off.
        Prints the new state and current counts so the user can see what's happening.
        Returns the new active state.
        """
        self.active = not self.active
        print(f"[Collector] Collection {'ON' if self.active else 'OFF'} - "
              f"samples: {self.sample_counts}")
        return self.active

    def update_landmarks(self, hand_landmarks):
        """Store the most recent hand landmarks so try_record() can use them."""
        self.last_landmarks = hand_landmarks

    def try_record(self, key):
        """
        Attempt to save a training sample for the gesture mapped to `key`.

        Returns a tuple (recorded: bool, label: str | None, message: str):
            recorded -- True if a sample was successfully saved
            label    -- the gesture label ("Rock", "Scissors", or "Paper"), or None
            message  -- a status string suitable for displaying in the UI
        """
        # Do nothing if collection mode is off.
        if not self.active:
            return False, None, ""

        # Ignore key presses that aren't mapped to a gesture.
        if key not in LABEL_MAP:
            return False, None, ""

        # Can't record without a hand in frame.
        if self.last_landmarks is None:
            return False, None, "No hand detected"

        # Enforce the minimum gap between captures.
        now = time.monotonic()
        if now - self._last_capture < MIN_CAPTURE_GAP:
            remaining = MIN_CAPTURE_GAP - (now - self._last_capture)
            return False, None, f"Hold still... ({remaining:.1f}s)"

        label    = LABEL_MAP[key]
        features = extract_features(self.last_landmarks)

        # Feature extraction can fail if the hand is partially visible.
        if features is None:
            return False, None, "Could not extract features"

        # Guard against the feature extractor returning the wrong number of values.
        if len(features) != FEATURE_DIM:
            return False, None, f"Feature dimension mismatch: {len(features)} != {FEATURE_DIM}"

        # Append the new row to the CSV file.
        try:
            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow([label] + features)

            # Update tracking state.
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
        """
        Return a one-line status string to overlay on the camera feed.
        Returns an empty string when collection mode is off (nothing to show).
        """
        if not self.active:
            return ""
        return (f"COLLECTING: 7=Rock 8=Scissors 9=Paper | "
                f"R:{self.sample_counts['Rock']} "
                f"S:{self.sample_counts['Scissors']} "
                f"P:{self.sample_counts['Paper']}")

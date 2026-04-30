"""
calibration_state.py
====================
Guided front-on gesture calibration for new players.

Walks the player through collecting 15 samples each of Rock, Paper,
and Scissors (45 total), then automatically trains the ML model.

This runs ONCE per player on first launch before any game starts.
The model is saved to CapStone/front_on_gesture_model.pkl and reused
on all subsequent launches.

Phases:
    INTRO       -> explains what's happening, press ENTER to start
    COLLECTING  -> show one gesture at a time, SPACE to capture frame
    COUNTDOWN   -> 3-second hold countdown before auto-capture
    TRAINING    -> model is being trained (brief)
    DONE        -> success, ENTER to start playing
    FAILED      -> not enough samples, ENTER to retry
"""

import time
from pathlib import Path

try:
    from capstone_paths import CAPSTONE_DIR
except ImportError:
    import sys
    CAPSTONE_DIR = (Path.home() / "Desktop" / "CapStone"
                    if sys.platform == "darwin"
                    else Path.home() / "CapStone")

# How many samples needed per gesture
SAMPLES_PER_GESTURE = 20

# Minimum gap between captures (same as LandmarkCollector)
# Ensures each sample is meaningfully different
MIN_CAPTURE_GAP = 0.4

# Variation hints shown in rotation to encourage different hand positions
VARIATION_HINTS = [
    "Slightly tilt your hand left",
    "Slightly tilt your hand right", 
    "Move a little closer to camera",
    "Move a little further away",
    "Rotate wrist slightly clockwise",
    "Rotate wrist slightly anti-clockwise",
    "Spread fingers a bit wider",
    "Hold as naturally as possible",
    "Try a slightly different angle",
    "Keep fingers relaxed",
]
GESTURES            = ["Rock", "Paper", "Scissors"]

# Instructions shown for each gesture
GESTURE_INSTRUCTIONS = {
    "Rock":     "Make a FIST - curl all fingers, thumb over fingers",
    "Paper":    "Open your hand FLAT - fingers together, palm facing camera",
    "Scissors": "Show SCISSORS - index and middle fingers extended, others curled",
}

# Auto-capture countdown seconds
COUNTDOWN_SECS = 3.0


def model_exists() -> bool:
    """Return True if a trained model already exists for this machine."""
    return (CAPSTONE_DIR / "front_on_gesture_model.pkl").exists()


class CalibrationController:
    """
    Guides a new player through gesture calibration.
    Call update() every frame, read phase/progress from the returned dict.
    """

    def __init__(self):
        from landmark_collector import LandmarkCollector
        self._collector  = LandmarkCollector(output_dir=str(CAPSTONE_DIR))
        self._collector.active = True   # always collecting during calibration

        self.phase          = "INTRO"
        self._gesture_idx   = 0         # index into GESTURES list
        self._counts        = {g: 0 for g in GESTURES}  # samples captured this session
        self._last_landmarks = None
        self._countdown_start = 0.0
        self._last_capture    = 0.0   # enforces MIN_CAPTURE_GAP
        self._variation_idx   = 0
        self._training_result = None    # accuracy float or None
        self._status_msg    = ""

    @property
    def current_gesture(self):
        if self._gesture_idx < len(GESTURES):
            return GESTURES[self._gesture_idx]
        return None

    def update(self, hand_state, now=None):
        if now is None:
            now = time.monotonic()

        lm = hand_state.get("_landmarks") if hand_state else None
        self._collector.update_landmarks(lm)
        self._last_landmarks = lm

        if self.phase == "TRAINING":
            self._do_train()

        return self._build_output(now)

    def handle_key(self, key):
        """
        Returns "done" when calibration is complete.
        ENTER/SPACE trigger actions depending on phase.
        """
        KEY_ENTER = (13, 10)
        KEY_SPACE = (32,)

        if self.phase == "INTRO":
            if key in KEY_ENTER or key in KEY_SPACE:
                self.phase = "COLLECTING"

        elif self.phase == "COLLECTING":
            if key in KEY_SPACE or key in KEY_ENTER:
                self._capture_one()

        elif self.phase == "DONE":
            if key in KEY_ENTER or key in KEY_SPACE:
                return "done"

        elif self.phase == "FAILED":
            if key in KEY_ENTER:
                self._reset()

        return None

    def _capture_one(self):
        """Capture the current frame as a sample for the current gesture."""
        if self._last_landmarks is None:
            self._status_msg = "No hand detected - hold your hand up clearly"
            return

        now = time.monotonic()
        if now - self._last_capture < MIN_CAPTURE_GAP:
            remaining = MIN_CAPTURE_GAP - (now - self._last_capture)
            self._status_msg = f"Hold still... ({remaining:.1f}s)"
            return

        gesture = self.current_gesture
        if gesture is None:
            return

        # Map gesture to collector key (7=Rock, 8=Scissors, 9=Paper)
        key_map = {"Rock": ord("7"), "Scissors": ord("8"), "Paper": ord("9")}
        key = key_map.get(gesture)
        if key is None:
            return

        recorded, label, msg = self._collector.try_record(key)
        if recorded:
            self._counts[gesture] = self._counts.get(gesture, 0) + 1
            self._status_msg = f"Captured! ({self._counts[gesture]}/{SAMPLES_PER_GESTURE})"

            # Move to next gesture when enough samples collected
            if self._counts[gesture] >= SAMPLES_PER_GESTURE:
                self._gesture_idx += 1
                if self._gesture_idx >= len(GESTURES):
                    self.phase = "TRAINING"
                else:
                    self._status_msg = ""
        else:
            self._status_msg = msg or "Try again"

    def _do_train(self):
        """Train the model from collected data."""
        try:
            from front_on_trainer import train_and_save
            accuracy = train_and_save()
            if accuracy is not None:
                self._training_result = accuracy
                self.phase = "DONE"
            else:
                self.phase = "FAILED"
        except Exception as e:
            print(f"[Calibration] Training error: {e}")
            self.phase = "FAILED"

    def _reset(self):
        """Reset to try again from scratch."""
        self._gesture_idx = 0
        self._counts      = {g: 0 for g in GESTURES}
        self._status_msg  = ""
        self.phase        = "INTRO"

    def _build_output(self, now):
        gesture   = self.current_gesture
        collected = self._counts.get(gesture, 0) if gesture else 0
        total     = sum(self._counts.values())

        return {
            "phase":            self.phase,
            "gesture":          gesture,
            "gesture_idx":      self._gesture_idx,
            "gesture_count":    len(GESTURES),
            "samples_this":     collected,
            "samples_needed":   SAMPLES_PER_GESTURE,
            "samples_total":    total,
            "counts":           dict(self._counts),
            "instruction":      GESTURE_INSTRUCTIONS.get(gesture, ""),
            "status_msg":       self._status_msg,
            "variation_hint":   VARIATION_HINTS[self._variation_idx],
            "hand_visible":     self._last_landmarks is not None,
            "training_result":  self._training_result,
        }

"""
calibration_state.py
====================
Guided front-on gesture calibration for new players.

Walks the player through collecting 20 samples each of Rock, Paper,
and Scissors (60 total), then automatically trains the ML model in the
background.

This runs once per player on first launch, before any game starts.
The trained model is saved to CapStone/front_on_gesture_model.pkl and
reused on all subsequent launches.

Phases:
    INTRO       → explains what's about to happen, press ENTER or SPACE to start
    COLLECTING  → shows one gesture at a time; press SPACE or ENTER to capture a frame
    TRAINING    → model is being trained (brief background operation)
    DONE        → training succeeded, ENTER or SPACE to start playing
    FAILED      → not enough samples or training error, ENTER to retry from scratch
"""

import time
import threading
from pathlib import Path

# Try to import the canonical CapStone directory path; fall back to a platform default
try:
    from capstone_paths import CAPSTONE_DIR
except ImportError:
    import sys
    CAPSTONE_DIR = (Path.home() / "Desktop" / "CapStone"
                    if sys.platform == "darwin"
                    else Path.home() / "CapStone")

# How many samples we need for each gesture before training
SAMPLES_PER_GESTURE = 20

# Minimum gap between captures — keeps each sample meaningfully different
MIN_CAPTURE_GAP = 0.4

# Rotation of hints shown to the player to encourage varied hand positions.
# A bit of variety in the training data makes the model more robust.
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

# The three gestures we calibrate, in the order we collect them
GESTURES = ["Rock", "Paper", "Scissors"]

# Plain-English instructions shown for each gesture
GESTURE_INSTRUCTIONS = {
    "Rock":     "Make a FIST - curl all fingers, thumb over fingers",
    "Paper":    "Open your hand FLAT - fingers together, palm facing camera",
    "Scissors": "Show SCISSORS - index and middle fingers extended, others curled",
}

# How long the auto-capture countdown lasts (currently unused, kept for reference)
COUNTDOWN_SECS = 3.0


def model_exists() -> bool:
    """Return True if a trained model file already exists on this machine."""
    return (CAPSTONE_DIR / "front_on_gesture_model.pkl").exists()


class CalibrationController:
    """
    Guides a new player through the calibration process.
    Call update() every frame; read phase/progress from the returned dict.
    """

    def __init__(self):
        # LandmarkCollector handles actually saving the landmark data to disk
        from landmark_collector import LandmarkCollector
        self._collector       = LandmarkCollector(output_dir=str(CAPSTONE_DIR))
        self._collector.active = True  # always collecting during calibration

        self.phase             = "INTRO"
        self._gesture_idx      = 0           # index into GESTURES — which gesture we're on
        self._counts           = {g: 0 for g in GESTURES}  # samples captured so far
        self._last_landmarks   = None        # most recent hand landmark data
        self._last_capture     = 0.0         # timestamp of the last accepted capture
        self._variation_idx    = 0           # cycles through VARIATION_HINTS
        self._training_result  = None        # accuracy float set after training finishes
        self._status_msg       = ""          # short feedback message shown under the gesture

    @property
    def current_gesture(self):
        """Return the gesture we're currently collecting samples for, or None if done."""
        if self._gesture_idx < len(GESTURES):
            return GESTURES[self._gesture_idx]
        return None

    def update(self, hand_state, now=None):
        """
        Main tick — call once per frame.

        hand_state : dict from the hand tracker (we read "_landmarks" from it)
        now        : optional monotonic timestamp
        """
        if now is None:
            now = time.monotonic()

        # Pull the latest landmark data from the tracker and keep the collector updated
        lm = hand_state.get("_landmarks") if hand_state else None
        self._collector.update_landmarks(lm)
        self._last_landmarks = lm

        # If we just entered TRAINING, kick off the background training thread
        if self.phase == "TRAINING":
            self._start_training()

        return self._build_output(now)

    def handle_key(self, key):
        """
        Handle a key press from the main loop.

        Returns "done" when calibration is successfully completed, None otherwise.
        ENTER and SPACE both trigger the primary action in most phases.
        """
        KEY_ENTER = (13, 10)
        KEY_SPACE = (32,)

        if self.phase == "INTRO":
            # Any confirmation key starts the collection process
            if key in KEY_ENTER or key in KEY_SPACE:
                self.phase = "COLLECTING"

        elif self.phase == "COLLECTING":
            # Capture one sample frame for the current gesture
            if key in KEY_SPACE or key in KEY_ENTER:
                self._capture_one()

        elif self.phase == "DONE":
            # Calibration finished — signal to the main loop that we're ready to play
            if key in KEY_ENTER or key in KEY_SPACE:
                return "done"

        elif self.phase == "FAILED":
            # Something went wrong — let the player retry from scratch
            if key in KEY_ENTER:
                self._reset()

        return None

    def _capture_one(self):
        """
        Try to capture the current frame as a training sample for the current gesture.
        Enforces MIN_CAPTURE_GAP to ensure each sample is meaningfully different.
        """
        if self._last_landmarks is None:
            self._status_msg = "No hand detected - hold your hand up clearly"
            return

        now = time.monotonic()

        # Enforce the minimum gap between captures
        if now - self._last_capture < MIN_CAPTURE_GAP:
            remaining = MIN_CAPTURE_GAP - (now - self._last_capture)
            self._status_msg = f"Hold still... ({remaining:.1f}s)"
            return

        gesture = self.current_gesture
        if gesture is None:
            return  # all gestures collected — shouldn't reach here in normal flow

        # Map each gesture to the keyboard key the collector expects
        # (7=Rock, 8=Scissors, 9=Paper — legacy key assignments)
        key_map = {"Rock": ord("7"), "Scissors": ord("8"), "Paper": ord("9")}
        key = key_map.get(gesture)
        if key is None:
            return

        # Ask the collector to record the current landmarks
        recorded, label, msg = self._collector.try_record(key)

        if recorded:
            self._counts[gesture] = self._counts.get(gesture, 0) + 1
            self._last_capture    = now
            self._status_msg      = f"Captured! ({self._counts[gesture]}/{SAMPLES_PER_GESTURE})"

            # Rotate the variation hint to encourage different hand positions
            self._variation_idx = (self._variation_idx + 1) % len(VARIATION_HINTS)

            # Move to the next gesture once we have enough samples for this one
            if self._counts[gesture] >= SAMPLES_PER_GESTURE:
                self._gesture_idx += 1
                if self._gesture_idx >= len(GESTURES):
                    # All gestures done — kick off training
                    self.phase = "TRAINING"
                else:
                    self._status_msg = ""  # clear message when moving to a new gesture
        else:
            self._status_msg = msg or "Try again"

    def _start_training(self):
        """
        Start the background training thread if it isn't already running.
        The guard prevents us from spawning multiple threads if update() is
        called several times before training finishes.
        """
        if getattr(self, "_training_in_progress", False):
            return  # already started — don't spawn another thread
        self._training_in_progress = True
        threading.Thread(target=self._do_train, daemon=True).start()

    def _do_train(self):
        """
        Run model training in a background thread so the camera loop doesn't freeze.
        Sets self.phase to "DONE" on success or "FAILED" on any error.
        """
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
        finally:
            self._training_in_progress = False  # allow a retry if needed

    def _reset(self):
        """Reset all progress so the player can try calibration again from scratch."""
        self._gesture_idx = 0
        self._counts      = {g: 0 for g in GESTURES}
        self._status_msg  = ""
        self.phase        = "INTRO"

    def _build_output(self, now):
        """
        Package the current calibration state into a flat dict for the renderer.
        Called at the end of every update() so the view always gets a fresh snapshot.
        """
        gesture   = self.current_gesture
        collected = self._counts.get(gesture, 0) if gesture else 0
        total     = sum(self._counts.values())  # total samples across all gestures

        return {
            "phase":           self.phase,
            "gesture":         gesture,
            "gesture_idx":     self._gesture_idx,
            "gesture_count":   len(GESTURES),
            "samples_this":    collected,
            "samples_needed":  SAMPLES_PER_GESTURE,
            "samples_total":   total,
            "counts":          dict(self._counts),
            "instruction":     GESTURE_INSTRUCTIONS.get(gesture, ""),
            "status_msg":      self._status_msg,
            "variation_hint":  VARIATION_HINTS[self._variation_idx],
            "hand_visible":    self._last_landmarks is not None,
            "training_result": self._training_result,
        }

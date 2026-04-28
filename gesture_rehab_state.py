"""
gesture_rehab_state.py
======================
Gesture Trainer / Rehabilitation Exergame.

A guided exercise session: the system cycles through Rock, Paper, Scissors
gestures and prompts the player to hold each one for a set duration.
Tracks accuracy, completion rate, and session stats.

Designed for: dexterity training, warm-up before playing RPS, accessibility.

States: INTRO → EXERCISE → REST → COMPLETE
"""

import time
import random

VALID_GESTURES  = ["Rock", "Paper", "Scissors"]
HOLD_SECS       = 3.0    # hold time per gesture
REST_SECS       = 0.8    # brief rest between gestures
REPS_PER_GEST   = 3      # how many times each gesture is requested


class GestureRehabController:
    """
    Gesture Trainer controller.
    INTRO — explanation screen, waits for Enter to begin.
    EXERCISE → REST → COMPLETE
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.state          = "INTRO"
        self._sequence      = self._build_sequence()
        self._seq_idx       = 0
        self._dwell_start   = None
        self._rest_until    = 0.0
        self._held_gesture  = ""
        self.completed      = 0
        self.missed         = 0
        self.session_log    = []

    def start_session(self):
        """Called when player presses Enter on the INTRO screen."""
        if self.state == "INTRO":
            self.state         = "EXERCISE"
            self._dwell_start  = None
            self._held_gesture = ""

    def _build_sequence(self):
        seq = VALID_GESTURES * REPS_PER_GEST
        random.shuffle(seq)
        return seq

    def _build_output(self, now):
        target = ""
        dwell_pct = 0.0
        if self.state == "EXERCISE" and self._seq_idx < len(self._sequence):
            target = self._sequence[self._seq_idx]
            if self._dwell_start is not None and self._held_gesture == target:
                dwell_pct = min(1.0, (now - self._dwell_start) / HOLD_SECS)

        total = len(self._sequence)
        return {
            "play_mode_label": "Gesture Trainer",
            "state":           self.state,
            "target":          target,
            "held_gesture":    self._held_gesture,
            "dwell_pct":       dwell_pct,
            "step":            self._seq_idx,
            "total_steps":     total,
            "completed":       self.completed,
            "missed":          self.missed,
            "session_log":     list(self.session_log),
            "accuracy":        self.completed / max(1, self.completed + self.missed),
            "hold_secs":       HOLD_SECS,
            "reps_per_gest":   REPS_PER_GEST,
        }

    def update(self, tracker_state, now=None):
        if now is None:
            now = time.monotonic()

        confirmed = tracker_state.get("confirmed_gesture", "Unknown")

        if self.state == "INTRO":
            # Waits for start_session() via Enter key in main.py
            return self._build_output(now)

        if self.state == "REST":
            if now >= self._rest_until:
                self._seq_idx += 1
                if self._seq_idx >= len(self._sequence):
                    self.state = "COMPLETE"
                else:
                    self.state        = "EXERCISE"
                    self._dwell_start = None
                    self._held_gesture = ""
            return self._build_output(now)

        if self.state == "COMPLETE":
            return self._build_output(now)

        if self.state == "EXERCISE":
            target = self._sequence[self._seq_idx]

            if confirmed in VALID_GESTURES:
                if confirmed != self._held_gesture:
                    # Gesture changed — reset dwell
                    self._held_gesture = confirmed
                    self._dwell_start  = now
                else:
                    # Same gesture held — check dwell
                    dwell = now - self._dwell_start
                    if dwell >= HOLD_SECS:
                        # Locked in
                        success = (confirmed == target)
                        self.session_log.append({
                            "target":     target,
                            "held":       confirmed,
                            "success":    success,
                            "dwell_secs": round(dwell, 2),
                        })
                        if success:
                            self.completed += 1
                        else:
                            self.missed += 1
                        self._held_gesture = ""
                        self._dwell_start  = None
                        self.state         = "REST"
                        self._rest_until   = now + REST_SECS
            else:
                # No gesture — reset dwell
                if self._held_gesture:
                    self._held_gesture = ""
                    self._dwell_start  = None

        return self._build_output(now)

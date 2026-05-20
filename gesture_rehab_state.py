"""
gesture_rehab_state.py
======================
Gesture Trainer / Rehabilitation Exergame.

The system cycles through Rock, Paper, Scissors and prompts the player to
hold each gesture for a set duration. Once a gesture is held continuously
for HOLD_SECS seconds, it locks in and the session advances to the next prompt.

This mode is designed for:
  - Dexterity training and hand rehabilitation
  - Warm-up before playing RPS
  - Accessibility / assistive use

States: INTRO -> EXERCISE -> REST -> COMPLETE

The sequence is all three gestures repeated REPS_PER_GEST times, shuffled
randomly so the order is different every session.
"""

import time
import random

# Valid gestures this mode works with
VALID_GESTURES = ["Rock", "Paper", "Scissors"]

# Session configuration
HOLD_SECS     = 3.0   # how long a gesture must be held continuously before it locks in
REST_SECS     = 0.8   # short pause between gestures (gives the hand a moment to relax)
REPS_PER_GEST = 3     # how many times each gesture is prompted per session
UNKNOWN_GRACE = 0.30  # seconds of Unknown before the hold timer resets (avoids interrupting
                      # a real hold due to brief tracking glitches between frames)


class GestureRehabController:
    """
    Gesture Trainer controller.

    INTRO   — explanation screen; waits for start_session() (Enter key) to begin.
    EXERCISE — prompts a gesture and tracks how long the player holds it.
    REST     — brief pause between gestures.
    COMPLETE — session finished; shows stats.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all state back to INTRO so a new session can begin."""
        self.state          = "INTRO"
        self._sequence      = self._build_sequence()  # randomised list of target gestures
        self._seq_idx       = 0       # index into _sequence for the current prompt
        self._dwell_start   = None    # timestamp when the current hold started
        self._rest_until    = 0.0     # timestamp when the REST pause ends
        self._held_gesture  = ""      # which gesture the player is currently holding
        self._unknown_since = 0.0     # timestamp when Unknown started; 0 = not in Unknown
        self.completed      = 0       # number of successfully held gestures
        self.missed         = 0       # number of incorrectly held gestures
        self.session_log    = []      # per-gesture log entry for the COMPLETE screen

    def start_session(self):
        """
        Called when the player presses Enter on the INTRO screen.
        Transitions from INTRO to EXERCISE and arms the dwell timer.
        """
        if self.state == "INTRO":
            self.state         = "EXERCISE"
            self._dwell_start  = None
            self._held_gesture = ""

    def _build_sequence(self):
        """
        Build a shuffled list of target gestures for the full session.
        Each gesture appears REPS_PER_GEST times, so the total length is
        len(VALID_GESTURES) * REPS_PER_GEST.
        """
        seq = VALID_GESTURES * REPS_PER_GEST  # e.g. [Rock, Paper, Scissors, Rock, ...]
        random.shuffle(seq)                    # randomise so it's different every time
        return seq

    def _build_output(self, now):
        """
        Build the output dict the UI renderer reads every frame.

        Includes the current target gesture and dwell progress (0.0–1.0)
        so the UI can show a fill bar indicating hold progress.
        """
        target    = ""
        dwell_pct = 0.0

        # Only compute target and progress during an active EXERCISE step
        if self.state == "EXERCISE" and self._seq_idx < len(self._sequence):
            target = self._sequence[self._seq_idx]
            # dwell_pct = fraction of HOLD_SECS elapsed while holding the correct gesture
            if self._dwell_start is not None and self._held_gesture == target:
                dwell_pct = min(1.0, (now - self._dwell_start) / HOLD_SECS)

        return {
            "play_mode_label": "Gesture Trainer",
            "state":           self.state,
            "target":          target,          # the gesture the player should be making
            "held_gesture":    self._held_gesture,
            "dwell_pct":       dwell_pct,       # 0.0 = just started, 1.0 = locked in
            "step":            self._seq_idx,   # how far through the sequence we are
            "total_steps":     len(self._sequence),
            "completed":       self.completed,
            "missed":          self.missed,
            "session_log":     list(self.session_log),
            "accuracy":        self.completed / max(1, self.completed + self.missed),
            "hold_secs":       HOLD_SECS,
            "reps_per_gest":   REPS_PER_GEST,
        }

    def update(self, tracker_state, now=None):
        """
        Main frame update. Called every frame by the game loop.

        tracker_state must contain 'confirmed_gesture' from the hand tracker.
        """
        if now is None:
            now = time.monotonic()

        confirmed = tracker_state.get("confirmed_gesture", "Unknown")

        # INTRO: just wait here until start_session() is called via Enter key
        if self.state == "INTRO":
            return self._build_output(now)

        # REST: pause between gestures; advance to the next step when the timer expires
        if self.state == "REST":
            if now >= self._rest_until:
                self._seq_idx += 1
                if self._seq_idx >= len(self._sequence):
                    # All steps done — session is complete
                    self.state = "COMPLETE"
                else:
                    # Move to the next target gesture
                    self.state         = "EXERCISE"
                    self._dwell_start  = None
                    self._held_gesture = ""
            return self._build_output(now)

        # COMPLETE: session finished; nothing left to do, just display stats
        if self.state == "COMPLETE":
            return self._build_output(now)

        # EXERCISE: prompt the player to hold the target gesture for HOLD_SECS
        if self.state == "EXERCISE":
            target = self._sequence[self._seq_idx]

            if confirmed in VALID_GESTURES:
                # A valid RPS gesture is being shown — reset the Unknown grace timer
                self._unknown_since = 0.0

                if confirmed != self._held_gesture:
                    # Gesture changed (or just started) — restart the dwell timer
                    self._held_gesture = confirmed
                    self._dwell_start  = now
                else:
                    # Same gesture held continuously — check if we've hit HOLD_SECS
                    dwell = now - self._dwell_start
                    if dwell >= HOLD_SECS:
                        # Gesture has been held long enough — log it and move to REST
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
                        # Clear the hold state and start the rest pause
                        self._held_gesture = ""
                        self._dwell_start  = None
                        self.state         = "REST"
                        self._rest_until   = now + REST_SECS
            else:
                # No valid gesture detected (hand transitioning or out of frame).
                # Don't reset the dwell timer immediately — small Unknown blips are
                # normal as a hand moves between gestures and shouldn't break an
                # ongoing hold. Only reset after UNKNOWN_GRACE seconds of continuous Unknown.
                if self._unknown_since == 0.0:
                    self._unknown_since = now  # start timing the Unknown streak
                elif now - self._unknown_since >= UNKNOWN_GRACE:
                    # Unknown held too long — the player must have dropped their gesture
                    self._held_gesture  = ""
                    self._dwell_start   = None
                    self._unknown_since = 0.0

        return self._build_output(now)

# =============================================================================
# gesture_state.py
# ----------------
# Adds a multi-frame confirmation layer between raw gesture detection and
# any downstream action (UI update, robot command, etc.).
#
# The problem it solves:
#   MediaPipe can flicker — a "Rock" might briefly read as "Paper" for one
#   frame and then snap back. Without filtering, that flicker would cause
#   a wrong command to fire. This file fixes that by only confirming a
#   gesture once it has held steady for several frames in a row.
#
# Pipeline:
#   raw gesture  -->  majority vote over recent history  -->  stable gesture
#                -->  held steady for N frames            -->  confirmed gesture
#                -->  all checks pass                     -->  robot_ready / command
#
# Where it fits:
#   Created once per player in the game loop. update() is called every frame
#   with the raw gesture string from hand_landmarks.process_hand_frame().
# =============================================================================

from collections import Counter, deque


# Valid RPS/RPSLS throws. "Unknown" is intentionally excluded — it means
# the detector couldn't figure out what the hand is doing.
VALID_GESTURES = {"Rock", "Paper", "Scissors", "Spock", "Lizard"}

# Maps each gesture to the command string that the robot/UI expects.
COMMAND_MAP = {
    "Rock":     "CMD_ROCK",
    "Paper":    "CMD_PAPER",
    "Scissors": "CMD_SCISSORS",
    "Spock":    "CMD_SPOCK",
    "Lizard":   "CMD_LIZARD",
}


class GestureStateTracker:
    """
    Per-player gesture confirmation tracker.

    Parameters:
        history_size         -- how many recent raw frames to keep in the buffer
                                (used for majority voting)
        confirm_frames       -- how many consecutive frames the stable gesture
                                must hold before it's "confirmed"
        invalid_reset_frames -- how many consecutive "Unknown" frames before
                                we wipe history and start over
    """

    def __init__(self, history_size=7, confirm_frames=3, invalid_reset_frames=6):
        self.history_size         = history_size
        self.confirm_frames       = confirm_frames
        self.invalid_reset_frames = invalid_reset_frames
        # Initialise all state to clean defaults
        self.reset()

    def clear_for_new_throw(self):
        """
        Wipe recent gesture memory before a new throw phase begins.

        During the countdown, the player holds "Rock" as part of the pump
        animation. Without this clear, that Rock would linger in the buffer
        and bleed into the actual throw at the start of SHOOT — potentially
        giving the player the wrong result.
        """
        self.raw_history      = deque(maxlen=self.history_size)
        self.last_raw_gesture = "Unknown"
        self.stable_gesture   = "Unknown"
        self.confirmed_gesture = "Unknown"
        self.stable_streak    = 0
        self.invalid_frame_count = 0

    def reset(self):
        """Full reset — called on __init__ and between rounds."""
        self.raw_history         = deque(maxlen=self.history_size)
        self.last_raw_gesture    = "Unknown"
        self.stable_gesture      = "Unknown"
        self.confirmed_gesture   = "Unknown"
        self.stable_streak       = 0
        self.invalid_frame_count = 0

    def _get_majority_gesture(self):
        """
        Return whichever gesture appears most in the recent history buffer.
        Returns "Unknown" if the buffer is empty.
        """
        if not self.raw_history:
            return "Unknown"
        # Counter.most_common(1) gives us the top-voted gesture in one call
        counts = Counter(self.raw_history)
        return counts.most_common(1)[0][0]

    def update(self, raw_gesture):
        """
        Feed one frame's raw gesture into the tracker and advance state.

        raw_gesture should be one of: "Rock", "Paper", "Scissors",
        "Spock", "Lizard", or "Unknown".

        Returns a state dict with:
            raw_gesture         -- the value passed in this frame
            stable_gesture      -- majority-voted gesture from recent history
            confirmed_gesture   -- stable gesture that has held for N frames
            stable_streak       -- consecutive frames the stable gesture has held
            history_size        -- current number of entries in the buffer
            invalid_frame_count -- consecutive frames of "Unknown"
            robot_ready         -- True only when it's safe to act on the gesture
            command             -- "CMD_ROCK" etc., or "CMD_UNKNOWN" if not ready
        """
        self.last_raw_gesture = raw_gesture

        # If the gesture is a valid throw, add it to the sliding window buffer
        # and reset the invalid-frame counter since we got something good.
        if raw_gesture in VALID_GESTURES:
            self.raw_history.append(raw_gesture)
            self.invalid_frame_count = 0
        else:
            # Bad/unknown frame — increment the counter
            self.invalid_frame_count += 1

        # If we've had too many bad frames in a row, wipe everything and restart
        if self.invalid_frame_count >= self.invalid_reset_frames:
            self.raw_history.clear()
            self.stable_gesture    = "Unknown"
            self.confirmed_gesture = "Unknown"
            self.stable_streak     = 0
        else:
            # Figure out the most common gesture in the recent buffer
            new_stable = self._get_majority_gesture()

            if new_stable == "Unknown":
                # Buffer has no clear winner — reset stable tracking
                self.stable_gesture = "Unknown"
                self.stable_streak  = 0
            elif new_stable != self.stable_gesture:
                # Gesture changed — reset the streak to 1 for the new gesture
                self.stable_gesture = new_stable
                self.stable_streak  = 1
            else:
                # Same gesture as last frame — extend the streak
                self.stable_streak += 1

            # Promote to "confirmed" once the gesture has held long enough
            if self.stable_gesture in VALID_GESTURES and self.stable_streak >= self.confirm_frames:
                self.confirmed_gesture = self.stable_gesture

        # "Robot ready" requires all four conditions at once:
        #   1. A confirmed gesture exists
        #   2. The stable and confirmed gestures agree (no recent switch)
        #   3. The streak is long enough
        #   4. No invalid frames this frame (hand is visible and readable)
        robot_ready = (
            self.confirmed_gesture in VALID_GESTURES
            and self.stable_gesture == self.confirmed_gesture
            and self.stable_streak >= self.confirm_frames
            and self.invalid_frame_count == 0
        )

        # Only emit the real command when the robot is ready; otherwise unknown
        command = COMMAND_MAP.get(self.confirmed_gesture, "CMD_UNKNOWN")
        if not robot_ready:
            command = "CMD_UNKNOWN"

        return {
            "raw_gesture":         self.last_raw_gesture,
            "stable_gesture":      self.stable_gesture,
            "confirmed_gesture":   self.confirmed_gesture,
            "stable_streak":       self.stable_streak,
            "history_size":        len(self.raw_history),
            "invalid_frame_count": self.invalid_frame_count,
            "robot_ready":         robot_ready,
            "command":             command,
        }

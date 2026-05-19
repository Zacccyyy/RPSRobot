# =============================================================================
# gesture_state.py
# ----------------
# Adds a multi-frame confirmation layer between raw gesture detection and
# any downstream action (UI update, robot command, etc.).
#
# What it does:
#   - Buffers recent raw gesture detections in a sliding window.
#   - Promotes the most frequent gesture in that window to "stable".
#   - Promotes a stable gesture to "confirmed" only once it has held
#     steady for `confirm_frames` consecutive frames.
#   - Resets everything if the gesture is "Unknown" or missing for too
#     many frames in a row (configurable via `invalid_reset_frames`).
#
# Why this matters:
#   MediaPipe can produce single-frame noise (a "Rock" that flickers to
#   "Paper" for one frame and back).  Without this buffer, downstream
#   code would need to handle that noise itself.  The pipeline here
#   ensures a command is only emitted when the player has clearly held
#   a gesture for at least `confirm_frames` frames.
#
# Pipeline:
#   raw gesture  -->  majority vote over history  -->  stable gesture
#                -->  held for N frames            -->  confirmed gesture
#                -->  all checks pass              -->  robot_ready / command
#
# Where it fits:
#   - Created per-player in the game loop.
#   - update() is called once per frame with the raw_gesture string from
#     hand_landmarks.process_hand_frame().
# =============================================================================

from collections import Counter, deque


# Gestures that are valid RPS/RPSLS throws.
# "Unknown" is intentionally excluded — it means "we don't know yet".
VALID_GESTURES = {"Rock", "Paper", "Scissors", "Spock", "Lizard"}

# Maps each valid gesture to the command string the robot/UI expects
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
        history_size         — how many recent raw gestures to keep
                               (sliding window for majority vote)
        confirm_frames       — how many consecutive frames the stable gesture
                               must hold before it's "confirmed"
        invalid_reset_frames — how many consecutive "Unknown" frames before
                               we wipe the history and start over
    """

    def __init__(self, history_size=7, confirm_frames=3, invalid_reset_frames=6):
        self.history_size        = history_size
        self.confirm_frames      = confirm_frames
        self.invalid_reset_frames = invalid_reset_frames
        self.reset()

    def clear_for_new_throw(self):
        """
        Wipe recent gesture memory before a new throw phase begins.

        During the countdown the player holds "Rock" as part of the pump
        animation.  Without this clear, the Rock from the countdown would
        linger in the history buffer and bleed into the player's actual throw
        at the start of SHOOT — potentially causing a wrong result.
        """
        self.raw_history.clear()
        self.last_raw_gesture  = "Unknown"
        self.stable_gesture    = "Unknown"
        self.confirmed_gesture = "Unknown"
        self.stable_streak     = 0
        self.invalid_frame_count = 0

    def reset(self):
        """Full reset — called on init and between rounds."""
        self.raw_history       = deque(maxlen=self.history_size)
        self.last_raw_gesture  = "Unknown"
        self.stable_gesture    = "Unknown"
        self.confirmed_gesture = "Unknown"
        self.stable_streak     = 0
        self.invalid_frame_count = 0

    def _get_majority_gesture(self):
        """
        Return whichever gesture appears most often in the recent history.
        Returns "Unknown" if the history is empty.
        """
        if not self.raw_history:
            return "Unknown"
        counts = Counter(self.raw_history)
        return counts.most_common(1)[0][0]

    def update(self, raw_gesture):
        """
        Feed one frame's raw gesture into the tracker.

        raw_gesture should be one of: "Rock", "Paper", "Scissors",
        "Spock", "Lizard", or "Unknown".

        Returns a state dict with:
            raw_gesture        — the value passed in this frame
            stable_gesture     — majority-voted gesture from recent history
            confirmed_gesture  — stable gesture that has held for N frames
            stable_streak      — consecutive frames the stable gesture has held
            history_size       — current number of entries in the buffer
            invalid_frame_count — consecutive frames of "Unknown"
            robot_ready        — True only when it's safe to act on the gesture
            command            — "CMD_ROCK" etc. or "CMD_UNKNOWN"
        """
        self.last_raw_gesture = raw_gesture

        # Valid gesture: add to history and reset the invalid counter
        if raw_gesture in VALID_GESTURES:
            self.raw_history.append(raw_gesture)
            self.invalid_frame_count = 0
        else:
            # Invalid/unknown gesture this frame
            self.invalid_frame_count += 1

        if self.invalid_frame_count >= self.invalid_reset_frames:
            # Too many bad frames in a row — start fresh
            self.raw_history.clear()
            self.stable_gesture    = "Unknown"
            self.confirmed_gesture = "Unknown"
            self.stable_streak     = 0
        else:
            # Work out the majority gesture from the recent buffer
            new_stable = self._get_majority_gesture()

            if new_stable == "Unknown":
                # No clear winner in the buffer
                self.stable_gesture = "Unknown"
                self.stable_streak  = 0
            elif new_stable != self.stable_gesture:
                # Gesture changed — reset the streak counter
                self.stable_gesture = new_stable
                self.stable_streak  = 1
            else:
                # Same gesture as last frame — extend the streak
                self.stable_streak += 1

            # Promote to confirmed once the streak is long enough
            if (self.stable_gesture in VALID_GESTURES
                    and self.stable_streak >= self.confirm_frames):
                self.confirmed_gesture = self.stable_gesture

        # "Robot ready" requires all four conditions simultaneously:
        #   - a confirmed gesture exists
        #   - the stable and confirmed gestures agree (no recent change)
        #   - the streak is long enough
        #   - no invalid frames this frame (hand is visible and clean)
        robot_ready = (
            self.confirmed_gesture in VALID_GESTURES
            and self.stable_gesture == self.confirmed_gesture
            and self.stable_streak >= self.confirm_frames
            and self.invalid_frame_count == 0
        )

        # Only emit the real command when robot_ready; otherwise CMD_UNKNOWN
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

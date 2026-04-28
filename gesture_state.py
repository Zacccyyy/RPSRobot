from collections import Counter, deque


VALID_GESTURES = {"Rock", "Paper", "Scissors", "Spock", "Lizard"}

COMMAND_MAP = {
    "Rock":     "CMD_ROCK",
    "Paper":    "CMD_PAPER",
    "Scissors": "CMD_SCISSORS",
    "Spock":    "CMD_SPOCK",
    "Lizard":   "CMD_LIZARD",
}


class GestureStateTracker:
    """
    Adds a safer confirmation layer between raw gesture detection
    and any future robot command output.

    Pipeline:
        raw gesture -> stable gesture -> confirmed gesture -> safe command
    """

    def __init__(self, history_size=7, confirm_frames=3, invalid_reset_frames=6):
        self.history_size = history_size
        self.confirm_frames = confirm_frames
        self.invalid_reset_frames = invalid_reset_frames
        self.reset()

    def clear_for_new_throw(self):
        """
        Clears recent gesture memory so countdown Rock does not spill
        into the player's actual throw during SHOOT.
        """
        self.raw_history.clear()
        self.last_raw_gesture = "Unknown"
        self.stable_gesture = "Unknown"
        self.confirmed_gesture = "Unknown"
        self.stable_streak = 0
        self.invalid_frame_count = 0

    def reset(self):
        self.raw_history = deque(maxlen=self.history_size)
        self.last_raw_gesture = "Unknown"
        self.stable_gesture = "Unknown"
        self.confirmed_gesture = "Unknown"
        self.stable_streak = 0
        self.invalid_frame_count = 0

    def _get_majority_gesture(self):
        if not self.raw_history:
            return "Unknown"

        counts = Counter(self.raw_history)
        return counts.most_common(1)[0][0]

    def update(self, raw_gesture):
        """
        Update the tracker once per frame.

        raw_gesture should be:
            "Rock", "Paper", "Scissors", or "Unknown"
        """
        self.last_raw_gesture = raw_gesture

        if raw_gesture in VALID_GESTURES:
            self.raw_history.append(raw_gesture)
            self.invalid_frame_count = 0
        else:
            self.invalid_frame_count += 1

        if self.invalid_frame_count >= self.invalid_reset_frames:
            self.raw_history.clear()
            self.stable_gesture = "Unknown"
            self.confirmed_gesture = "Unknown"
            self.stable_streak = 0
        else:
            new_stable_gesture = self._get_majority_gesture()

            if new_stable_gesture == "Unknown":
                self.stable_gesture = "Unknown"
                self.stable_streak = 0
            elif new_stable_gesture != self.stable_gesture:
                self.stable_gesture = new_stable_gesture
                self.stable_streak = 1
            else:
                self.stable_streak += 1

            if (
                self.stable_gesture in VALID_GESTURES
                and self.stable_streak >= self.confirm_frames
            ):
                self.confirmed_gesture = self.stable_gesture

        robot_ready = (
            self.confirmed_gesture in VALID_GESTURES
            and self.stable_gesture == self.confirmed_gesture
            and self.stable_streak >= self.confirm_frames
            and self.invalid_frame_count == 0
        )

        command = COMMAND_MAP.get(self.confirmed_gesture, "CMD_UNKNOWN")
        if not robot_ready:
            command = "CMD_UNKNOWN"

        return {
            "raw_gesture": self.last_raw_gesture,
            "stable_gesture": self.stable_gesture,
            "confirmed_gesture": self.confirmed_gesture,
            "stable_streak": self.stable_streak,
            "history_size": len(self.raw_history),
            "invalid_frame_count": self.invalid_frame_count,
            "robot_ready": robot_ready,
            "command": command,
        }
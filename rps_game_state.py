import time


VALID_GESTURES = {"Rock", "Paper", "Scissors"}

WIN_MAP = {
    "Rock": "Paper",
    "Paper": "Scissors",
    "Scissors": "Rock",
}


class RPSGameController:
    """
    Cheat Mode:
    - counts the player's throw
    - instantly outputs the winning counter-move
    """

    def __init__(
        self,
        robot_output=None,
        down_threshold=0.045,
        up_threshold=0.035,
        beat_cooldown=0.18,
        rock_grace_period=0.50,
        shoot_window_seconds=0.55,
        shoot_change_guard_seconds=0.05,
        rock_assume_seconds=0.13,
        result_display_seconds=1.80
    ):
        self.robot_output = robot_output

        self.down_threshold = down_threshold
        self.up_threshold = up_threshold
        self.beat_cooldown = beat_cooldown
        self.rock_grace_period = rock_grace_period
        self.shoot_window_seconds = shoot_window_seconds
        self.shoot_change_guard_seconds = shoot_change_guard_seconds
        self.rock_assume_seconds = rock_assume_seconds
        self.result_display_seconds = result_display_seconds

        self.reset_round()

    def reset(self):
        self._voice_mode = False
        self.reset_round()

    def set_voice_mode(self, enabled):
        self._voice_mode = bool(enabled)

    def inject_voice_beat(self, word, now=None):
        """Advance the countdown by one step when a beat word is spoken."""
        if now is None:
            now = time.monotonic()

        cooldown_ok = (now - self.last_beat_time) >= self.beat_cooldown

        if self.state == "WAITING_FOR_ROCK" and word == "ready":
            self.state = "COUNTDOWN"
            self.beat_count = 0
            self.last_beat_time = now
            return

        if self.state != "COUNTDOWN":
            return

        if word in ("one", "two") and cooldown_ok:
            self.beat_count += 1
            self.last_beat_time = now

        elif word == "three" and cooldown_ok:
            self.beat_count = 3
            self.last_beat_time = now
            self.beat_count = 4
            self.state = "SHOOT_WINDOW"
            self.shoot_open_time  = now
            self.shoot_close_time = now + max(self.shoot_window_seconds, 2.50)

    def inject_voice_throw(self, gesture, now=None):
        """Resolve the current round with a spoken throw."""
        if now is None:
            now = time.monotonic()

        if self.state == "SHOOT_WINDOW" and gesture in VALID_GESTURES:
            self._lock_round(gesture, now)

    def reset_round(self):
        self._voice_mode = getattr(self, "_voice_mode", False)  # preserve across rounds
        self.state = "WAITING_FOR_ROCK"
        self.beat_count = 0
        self.phase = "ready_for_down"
        self.top_y = None
        self.bottom_y = None
        self.last_beat_time = 0.0
        self.last_rock_time = 0.0

        self.shoot_open_time = None
        self.shoot_close_time = None

        self.player_gesture = "Unknown"
        self.computer_gesture = "Unknown"
        self.robot_move_command = "PENDING"
        self.result_banner = ""
        self.result_until = None

    def _lock_round(self, player_gesture, now):
        self.player_gesture = player_gesture
        self.computer_gesture = WIN_MAP[player_gesture]
        self.robot_move_command = f"ROBOT_PLAY_{self.computer_gesture.upper()}"
        self.result_banner = "ROBOT TAKES THE ROUND"

        self.state = "ROUND_RESULT"
        self.result_until = now + self.result_display_seconds

        if self.robot_output is not None:
            self.robot_output.publish_round_result(
                command=self.robot_move_command,
                game_mode="Cheat",
                round_result="robot_win",
                player_gesture=self.player_gesture,
                robot_gesture=self.computer_gesture,
                metadata={
                    "banner": self.result_banner,
                }
            )


    def _get_fallback_throw(self, tracker_state):
        stable_gesture = tracker_state.get("stable_gesture", "Unknown")
        confirmed_gesture = tracker_state.get("confirmed_gesture", "Unknown")
        raw_gesture = tracker_state.get("raw_gesture", "Unknown")

        if stable_gesture in VALID_GESTURES:
            return stable_gesture

        if confirmed_gesture in VALID_GESTURES:
            return confirmed_gesture

        if raw_gesture in VALID_GESTURES:
            return raw_gesture

        return "Unknown"

    def _build_output(self, now):
        base = {
            "play_mode_label": "Cheat Mode",
            "state": self.state,
            "beat_count": self.beat_count,
            "time_left": 0.0,
            "player_gesture": self.player_gesture,
            "computer_gesture": self.computer_gesture,
            "robot_move_command": self.robot_move_command,
            "result_banner": self.result_banner,
            "score_text": "",
            "round_text": "",
            "round_number": 0,
            "player_score": 0,
            "robot_score": 0,
        }

        if self.state == "WAITING_FOR_ROCK":
            base.update({
                "state_label": "Waiting",
                "main_text": "MAKE A FIST" if not self._voice_mode else 'Say  "READY"',
                "sub_text": "Hold Rock, then pump downward 4 times" if not self._voice_mode else "to start the countdown",
            })
            return base

        if self.state == "COUNTDOWN":
            if self._voice_mode:
                next_words = {0: '"ONE"', 1: '"TWO"', 2: '"THREE"'}
                _nw        = next_words.get(self.beat_count, '"THREE"')
                main_text  = f"Say  {_nw}"
            else:
                main_text = "READY" if self.beat_count == 0 else str(min(self.beat_count, 3))
            base.update({
                "state_label": "Countdown",
                "main_text": main_text,
                "sub_text": "Cheat mode counters after SHOOT",
            })
            return base

        if self.state == "SHOOT_WINDOW":
            base.update({
                "state_label": "Shoot Window",
                "main_text": "SHOOT!",
                "sub_text": "Throw Rock, Paper, or Scissors now",
                "time_left": 0.0 if self._voice_mode else max(0.0, self.shoot_close_time - now),
            })
            return base

        if self.state == "ROUND_RESULT":
            base.update({
                "state_label": "Round Result",
                "main_text": self.result_banner,
                "sub_text": "Cheat mode always counters your throw",
                "time_left": max(0.0, self.result_until - now),
            })
            return base

        base.update({
            "state_label": "Unknown",
            "main_text": "UNKNOWN",
            "sub_text": "",
        })
        return base

    def update(self, wrist_y, tracker_state, now=None):
        if now is None:
            now = time.monotonic()

        confirmed_gesture = tracker_state.get("confirmed_gesture", "Unknown")
        stable_gesture = tracker_state.get("stable_gesture", "Unknown")

        if self.state == "ROUND_RESULT":
            if now >= self.result_until:
                self.reset_round()
            return self._build_output(now)

        confirmed_rock = confirmed_gesture == "Rock"
        stable_rock = stable_gesture == "Rock"

        if self.state == "WAITING_FOR_ROCK":
            if self._voice_mode:
                # Voice mode: "ready" spoken → inject_voice_beat handles transition.
                return self._build_output(now)
            if confirmed_rock and wrist_y is not None:
                self.state = "COUNTDOWN"
                self.phase = "ready_for_down"
                self.top_y = wrist_y
                self.bottom_y = wrist_y
                self.last_rock_time = now
            return self._build_output(now)

        if self.state == "COUNTDOWN":
            # Voice mode: each beat advances only when the next word is spoken.
            # No timeout — the player can take as long as they need between words.
            if self._voice_mode:
                return self._build_output(now)
            rock_detected = (confirmed_rock or stable_rock) and wrist_y is not None
            within_grace = (now - self.last_rock_time) <= self.rock_grace_period
            can_track = rock_detected or (within_grace and wrist_y is not None and self.beat_count > 0)

            if rock_detected:
                self.last_rock_time = now

            if can_track:
                if self.phase == "ready_for_down":
                    if self.top_y is None:
                        self.top_y = wrist_y

                    self.top_y = min(self.top_y, wrist_y)

                    moved_down_enough = (wrist_y - self.top_y) >= self.down_threshold
                    cooldown_ok = (now - self.last_beat_time) >= self.beat_cooldown

                    if moved_down_enough and cooldown_ok:
                        self.beat_count += 1
                        self.last_beat_time = now
                        self.phase = "waiting_for_up"
                        self.bottom_y = wrist_y

                        if self.beat_count >= 4:
                            self.state = "SHOOT_WINDOW"
                            self.shoot_open_time = now
                            self.shoot_close_time = now + self.shoot_window_seconds

                elif self.phase == "waiting_for_up":
                    if self.bottom_y is None:
                        self.bottom_y = wrist_y

                    self.bottom_y = max(self.bottom_y, wrist_y)

                    moved_up_enough = (self.bottom_y - wrist_y) >= self.up_threshold

                    if moved_up_enough:
                        self.phase = "ready_for_down"
                        self.top_y = wrist_y

            else:
                if not within_grace:
                    self.reset_round()

            return self._build_output(now)

        if self.state == "SHOOT_WINDOW":
            time_since_open = now - self.shoot_open_time

            # Voice mode: window stays open until inject_voice_throw fires.
            if self._voice_mode:
                return self._build_output(now)

            if time_since_open >= self.shoot_change_guard_seconds:
                if stable_gesture in {"Paper", "Scissors"}:
                    self._lock_round(stable_gesture, now)
                    return self._build_output(now)

            if time_since_open >= self.rock_assume_seconds:
                self._lock_round("Rock", now)
                return self._build_output(now)

            if now >= self.shoot_close_time:
                fallback_throw = self._get_fallback_throw(tracker_state)

                if fallback_throw in VALID_GESTURES:
                    self._lock_round(fallback_throw, now)
                else:
                    self.reset_round()

            return self._build_output(now)

        return self._build_output(now)
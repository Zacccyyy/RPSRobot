import time

from challenge_ai import ChallengeAI
from fair_play_ai import UPGRADE_MOVE, DOWNGRADE_MOVE


VALID_GESTURES = {"Rock", "Paper", "Scissors"}

BEATS = {
    "Rock": "Scissors",
    "Paper": "Rock",
    "Scissors": "Paper",
}


def compare_rps(player_move, robot_move):
    if player_move == robot_move:
        return "draw"

    if BEATS[player_move] == robot_move:
        return "win"

    return "lose"


class ChallengeController:
    """
    Challenge Mode:
    - endless run until first loss
    - score = consecutive wins
    - persistent high score supported through stats logger
    - robot locks on beat 3
    - player throws on SHOOT
    - draws replay the round
    - AI ramps up as streak increases
    """

    def __init__(
        self,
        robot_output=None,
        ai=None,
        stats_logger=None,
        down_threshold=0.045,
        up_threshold=0.035,
        beat_cooldown=0.18,
        rock_grace_period=0.50,
        shoot_window_seconds=0.55,
        shoot_change_guard_seconds=0.05,
        rock_assume_seconds=0.14,
        round_intro_seconds=1.00,
        round_result_seconds=1.80,
        game_over_seconds=2.70
    ):
        self.robot_output = robot_output
        self._voice_mode = False
        self.ai = ai or ChallengeAI()
        self.stats_logger = stats_logger

        self.down_threshold = down_threshold
        self.up_threshold = up_threshold
        self.beat_cooldown = beat_cooldown
        self.rock_grace_period = rock_grace_period
        self.shoot_window_seconds = shoot_window_seconds
        self.shoot_change_guard_seconds = shoot_change_guard_seconds
        self.rock_assume_seconds = rock_assume_seconds
        self.round_intro_seconds = round_intro_seconds
        self.round_result_seconds = round_result_seconds
        self.game_over_seconds = game_over_seconds

        self.high_score = self.stats_logger.get_high_score() if self.stats_logger else 0
        self.reset_run()

    def reset(self):
        """
        Reset the active run but keep persistent high score.
        """
        if self.stats_logger is not None:
            self.high_score = self.stats_logger.get_high_score()
        self.reset_run()

    def reset_run(self, now=None):
        if now is None:
            now = time.monotonic()

        self.ai.reset()
        self.history = []

        self.streak = 0
        self.round_number = 1

        self.match_result_banner = ""
        self.match_until = None
        self.last_round_result = None
        self.emotion_snapshot = None

        self._reset_round_motion()
        self.state = "ROUND_INTRO"
        self.intro_until = now + self.round_intro_seconds

    def _reset_round_motion(self):
        self.beat_count = 0
        self.phase = "ready_for_down"
        self.top_y = None
        self.bottom_y = None
        self.last_beat_time = 0.0
        self.last_rock_time = 0.0

        self.shoot_open_time = None
        self.shoot_close_time = None

        self.robot_locked_move = None
        self.robot_move_command = "PENDING"

        self.player_gesture = "Unknown"
        self.computer_gesture = "Unknown"

        self.result_banner = ""
        self.result_until = None

        self.tracker_reset_requested = False

        if self.robot_output is not None:
            self.robot_output.clear_pending_locked()

    def consume_tracker_reset_request(self):
        self.tracker_reset_requested = False

    # ------------------------------------------------------------------ #
    # Voice input injection                                                #
    # ------------------------------------------------------------------ #

    def set_voice_mode(self, enabled):
        """Enable or disable voice-based input."""
        self._voice_mode = bool(enabled)

    def inject_voice_beat(self, word, now=None):
        """Advance the countdown via a spoken word.
        Protocol: ready → one → two → three → [say gesture]
        """
        if now is None:
            now = time.monotonic()

        if self.state == "WAITING_FOR_ROCK" and word == "ready":
            self.state = "COUNTDOWN"
            self.phase = "ready_for_down"
            self.beat_count = 0
            self.last_beat_time = now
            self.last_rock_time = now
            return

        if self.state != "COUNTDOWN":
            return

        self.last_rock_time = now
        cooldown_ok = (now - self.last_beat_time) >= self.beat_cooldown

        if word in ("one", "two") and cooldown_ok:
            self.beat_count += 1
            self.last_beat_time = now

        elif word == "three" and cooldown_ok:
            self.beat_count = 3
            self.last_beat_time = now
            self._lock_robot_move()
            self.beat_count = 4
            self.state = "SHOOT_WINDOW"
            self.shoot_open_time  = now
            self.shoot_close_time = now + max(self.shoot_window_seconds, 2.50)
            self.tracker_reset_requested = True

    def inject_voice_throw(self, gesture, now=None):
        """Resolve the current round with a spoken throw."""
        if now is None:
            now = time.monotonic()

        if self.state == "SHOOT_WINDOW" and gesture in VALID_GESTURES:
            self._resolve_round(gesture, now)

    def _prepare_next_round(self, now):
        self._reset_round_motion()
        self.state = "ROUND_INTRO"
        self.intro_until = now + self.round_intro_seconds

    def _lock_robot_move(self):
        if self.robot_locked_move is not None:
            return

        # Feed emotion state to AI so it can adjust confidence.
        if hasattr(self.ai, "set_emotion"):
            self.ai.set_emotion(self.emotion_snapshot)

        self.robot_locked_move = self.ai.choose_robot_move(
            history=self.history,
            streak=self.streak,
            round_number=self.round_number
        )
        self.robot_move_command = f"ROBOT_PLAY_{self.robot_locked_move.upper()}"

        if self.robot_output is not None:
            self.robot_output.stage_locked_move(
                command=self.robot_move_command,
                game_mode="Challenge",
                metadata={
                    "round_number": self.round_number,
                    "streak": self.streak,
                    "high_score": self.high_score,
                }
            )

    def _fallback_throw(self, tracker_state):
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

    def set_emotion_snapshot(self, snapshot):
        """
        Called by main loop each frame so that when a round resolves,
        the emotion at decision time is captured.
        """
        self.emotion_snapshot = snapshot

    def _log_round(self, round_result, reaction_time_ms=None):
        if self.stats_logger is None:
            return

        # --- Derive previous gesture and response type ---
        # History already has the current round appended by now,
        # so "previous" is the second-to-last entry.
        previous_player_gesture = None
        player_response_type = None

        if len(self.history) >= 2:
            prev_round = self.history[-2]
            previous_player_gesture = prev_round["player_gesture"]
            current_gesture = self.player_gesture

            if current_gesture == previous_player_gesture:
                player_response_type = "stay"
            elif UPGRADE_MOVE.get(previous_player_gesture) == current_gesture:
                player_response_type = "upgrade"
            elif DOWNGRADE_MOVE.get(previous_player_gesture) == current_gesture:
                player_response_type = "downgrade"

        # --- AI prediction metadata ---
        prediction = self.ai.last_prediction or {}

        # --- Emotion data ---
        em = self.emotion_snapshot or {}

        self.stats_logger.log_round(
            round_number=self.round_number,
            player_gesture=self.player_gesture,
            robot_gesture=self.computer_gesture,
            round_result=round_result,
            streak_after_round=self.streak,
            high_score_after_round=self.high_score,
            ai_predicted_move=prediction.get("top_predicted_move"),
            ai_effective_skill=prediction.get("effective_skill"),
            reaction_time_ms=reaction_time_ms,
            previous_player_gesture=previous_player_gesture,
            player_response_type=player_response_type,
            emotion=em.get("emotion"),
            emotion_confidence=em.get("emotion_confidence"),
            smile_score=em.get("smile_score"),
            surprise_score=em.get("surprise_score"),
            frustration_score=em.get("frustration_score"),
        )

    def _resolve_round(self, player_gesture, now):
        if self.robot_locked_move is None:
            self._lock_robot_move()

        # Capture reaction time: ms between SHOOT opening and gesture lock.
        reaction_time_ms = None
        if self.shoot_open_time is not None:
            reaction_time_ms = round((now - self.shoot_open_time) * 1000, 1)

        self.player_gesture = player_gesture
        self.computer_gesture = self.robot_locked_move

        outcome = compare_rps(self.player_gesture, self.computer_gesture)

        if outcome == "win":
            self.streak += 1
            self.high_score = max(self.high_score, self.streak)
            self.result_banner = "YOU SURVIVE"
            round_result = "player_win"
            player_outcome_for_history = "win"

            self.history.append({
                "round_number": self.round_number,
                "player_gesture": self.player_gesture,
                "robot_gesture": self.computer_gesture,
                "player_outcome": player_outcome_for_history,
            })

            if self.robot_output is not None:
                self.robot_output.publish_round_result(
                    command=self.robot_move_command,
                    game_mode="Challenge",
                    round_result=round_result,
                    player_gesture=self.player_gesture,
                    robot_gesture=self.computer_gesture,
                    metadata={
                        "round_number": self.round_number,
                        "streak": self.streak,
                        "high_score": self.high_score,
                        "banner": self.result_banner,
                    }
                )

            self._log_round(round_result, reaction_time_ms=reaction_time_ms)

            self.last_round_result = round_result
            self.state = "ROUND_RESULT"
            self.result_until = now + self.round_result_seconds
            return

        if outcome == "draw":
            self.result_banner = "DRAW - GO AGAIN"
            round_result = "draw"
            player_outcome_for_history = "draw"

            self.history.append({
                "round_number": self.round_number,
                "player_gesture": self.player_gesture,
                "robot_gesture": self.computer_gesture,
                "player_outcome": player_outcome_for_history,
            })

            if self.robot_output is not None:
                self.robot_output.publish_round_result(
                    command=self.robot_move_command,
                    game_mode="Challenge",
                    round_result=round_result,
                    player_gesture=self.player_gesture,
                    robot_gesture=self.computer_gesture,
                    metadata={
                        "round_number": self.round_number,
                        "streak": self.streak,
                        "high_score": self.high_score,
                        "banner": self.result_banner,
                    }
                )

            self._log_round(round_result, reaction_time_ms=reaction_time_ms)

            self.last_round_result = round_result
            self.state = "ROUND_RESULT"
            self.result_until = now + self.round_result_seconds
            return

        # Player loses -> run ends immediately
        self.result_banner = "GAME OVER"
        round_result = "robot_win"
        player_outcome_for_history = "lose"

        self.history.append({
            "round_number": self.round_number,
            "player_gesture": self.player_gesture,
            "robot_gesture": self.computer_gesture,
            "player_outcome": player_outcome_for_history,
        })

        if self.robot_output is not None:
            self.robot_output.publish_round_result(
                command=self.robot_move_command,
                game_mode="Challenge",
                round_result=round_result,
                player_gesture=self.player_gesture,
                robot_gesture=self.computer_gesture,
                metadata={
                    "round_number": self.round_number,
                    "streak": self.streak,
                    "high_score": self.high_score,
                    "banner": self.result_banner,
                }
            )

        self._log_round(round_result, reaction_time_ms=reaction_time_ms)

        if self.stats_logger is not None:
            self.stats_logger.finalize_run(
                final_streak=self.streak,
                status="completed"
            )

        self.last_round_result = round_result
        self.match_result_banner = "GAME OVER"
        self.state = "MATCH_RESULT"
        self.match_until = now + self.game_over_seconds

    def _build_output(self, now):
        score_text = f"STREAK {self.streak} | HIGH {self.high_score}"
        round_text = f"ROUND {self.round_number}"

        base = {
            "play_mode_label": "Challenge",
            "state": self.state,
            "beat_count": self.beat_count,
            "time_left": 0.0,
            "player_gesture": self.player_gesture,
            "computer_gesture": self.computer_gesture,
            "robot_move_command": self.robot_move_command,
            "result_banner": self.result_banner,
            "score_text": score_text,
            "round_text": round_text,
            "round_number": self.round_number,
            "player_score": self.streak,
            "robot_score": self.high_score,
            "request_tracker_reset": self.tracker_reset_requested,
        }

        if self.state == "ROUND_INTRO":
            base.update({
                "state_label": "Round Intro",
                "main_text": round_text,
                "sub_text": score_text,
            })
            return base

        if self.state == "WAITING_FOR_ROCK":
            base.update({
                "state_label": "Waiting",
                "main_text": "MAKE A FIST" if not self._voice_mode else "VOICE MODE",
                "sub_text": (
                    "Say READY  then  ONE  TWO  THREE"
                    if self._voice_mode
                    else "KEEP THE STREAK ALIVE"
                ),
            })
            return base

        if self.state == "COUNTDOWN":
            main_text = "READY" if self.beat_count == 0 else str(min(self.beat_count, 3))
            base.update({
                "state_label": "Countdown",
                "main_text": main_text,
                "sub_text": (
                    "Say ONE  TWO  THREE  SHOOT"
                    if self._voice_mode
                    else "AI gets stronger as your streak rises"
                ),
            })
            return base

        if self.state == "SHOOT_WINDOW":
            base.update({
                "state_label": "Shoot Window",
                "main_text": "SHOOT!",
                "sub_text": (
                    "Say ROCK, PAPER, or SCISSORS"
                    if self._voice_mode
                    else "One loss ends the run"
                ),
                "time_left": 0.0 if self._voice_mode else max(0.0, self.shoot_close_time - now),
            })
            return base

        if self.state == "ROUND_RESULT":
            base.update({
                "state_label": "Round Result",
                "main_text": self.result_banner,
                "sub_text": score_text,
                "time_left": max(0.0, self.result_until - now),
            })
            return base

        if self.state == "MATCH_RESULT":
            base.update({
                "state_label": "Game Over",
                "main_text": self.match_result_banner,
                "sub_text": f"FINAL STREAK {self.streak} | HIGH {self.high_score}",
                "result_banner": self.match_result_banner,
                "time_left": max(0.0, self.match_until - now),
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

        if self.state == "ROUND_INTRO":
            if now >= self.intro_until:
                self.state = "WAITING_FOR_ROCK"
            return self._build_output(now)

        if self.state == "ROUND_RESULT":
            if now >= self.result_until:
                if self.last_round_result == "player_win":
                    self.round_number += 1
                self._prepare_next_round(now)
            return self._build_output(now)

        if self.state == "MATCH_RESULT":
            if now >= self.match_until:
                if self.stats_logger is not None:
                    self.high_score = self.stats_logger.get_high_score()
                self.reset_run(now)
            return self._build_output(now)

        confirmed_rock = confirmed_gesture == "Rock"
        stable_rock = stable_gesture == "Rock"

        if self.state == "WAITING_FOR_ROCK":
            if self._voice_mode:
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

                        if self.beat_count >= 3:
                            self._lock_robot_move()

                        if self.beat_count >= 4:
                            self.state = "SHOOT_WINDOW"
                            self.shoot_open_time = now
                            self.shoot_close_time = now + self.shoot_window_seconds
                            self.tracker_reset_requested = True

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
                    self._prepare_next_round(now)

            return self._build_output(now)

        if self.state == "SHOOT_WINDOW":
            time_since_open = now - self.shoot_open_time

            # Voice mode: window stays open indefinitely until inject_voice_throw fires.
            if self._voice_mode:
                return self._build_output(now)

            if time_since_open >= self.shoot_change_guard_seconds:
                if confirmed_gesture in {"Paper", "Scissors"}:
                    self._resolve_round(confirmed_gesture, now)
                    return self._build_output(now)

                if stable_gesture in {"Paper", "Scissors"}:
                    self._resolve_round(stable_gesture, now)
                    return self._build_output(now)

            if time_since_open >= self.rock_assume_seconds:
                self._resolve_round("Rock", now)
                return self._build_output(now)

            if now >= self.shoot_close_time:
                fallback_throw = self._fallback_throw(tracker_state)

                if fallback_throw in VALID_GESTURES:
                    self._resolve_round(fallback_throw, now)
                else:
                    self._prepare_next_round(now)

            return self._build_output(now)

        return self._build_output(now)
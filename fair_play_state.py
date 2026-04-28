import time

from fair_play_ai import FairPlayAI


VALID_GESTURES = {"Rock", "Paper", "Scissors"}

COUNTER_MOVE = {
    "Rock": "Paper",
    "Paper": "Scissors",
    "Scissors": "Rock",
}

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


class FairPlayController:
    """
    Fair Play Mode:
    - first to 2 wins
    - robot locks on beat 3
    - player throws on SHOOT
    - draws replay the round
    """

    def __init__(
        self,
        robot_output=None,
        ai=None,
        win_target=2,
        play_mode_label="Fair Play",
        down_threshold=0.045,
        up_threshold=0.035,
        beat_cooldown=0.18,
        rock_grace_period=0.50,
        shoot_window_seconds=0.55,
        shoot_change_guard_seconds=0.05,
        rock_assume_seconds=0.14,
        round_intro_seconds=1.00,
        round_result_seconds=2.00,
        match_result_seconds=2.40
    ):
        self.robot_output = robot_output
        self._voice_mode = False
        self.ai = ai or FairPlayAI()
        self.win_target = win_target
        self.play_mode_label = play_mode_label
        self.opponent_label = "ROBOT"

        self.down_threshold = down_threshold
        self.up_threshold = up_threshold
        self.beat_cooldown = beat_cooldown
        self.rock_grace_period = rock_grace_period
        self.shoot_window_seconds = shoot_window_seconds
        self.shoot_change_guard_seconds = shoot_change_guard_seconds
        self.rock_assume_seconds = rock_assume_seconds
        self.round_intro_seconds = round_intro_seconds
        self.round_result_seconds = round_result_seconds
        self.match_result_seconds = match_result_seconds

        self.reset_match()

    def reset(self):
        self.reset_match()

    def reset_match(self, now=None):
        if now is None:
            now = time.monotonic()

        self.ai.reset()
        self.history = []

        self.player_score = 0
        self.robot_score = 0
        self.round_number = 1

        self.match_result_banner = ""
        self.match_until = None

        # Session-level stats (survive across rounds within a match)
        self._session_reaction_times = []
        self._session_gestures       = []
        self._last_round_player_gest = None
        self._last_round_robot_gest  = None
        self._last_round_banner      = ""

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
        self.last_round_result = None
        self.result_until = None

        # One-shot flag used by the main loop.
        # When True, the tracker should clear its recent history exactly once.
        self.tracker_reset_requested = False

        if self.robot_output is not None:
            self.robot_output.clear_pending_locked()

    def consume_tracker_reset_request(self):
        """
        Called by the main loop after it has cleared the tracker.
        """
        self.tracker_reset_requested = False

    # ------------------------------------------------------------------ #
    # Voice input injection                                                #
    # ------------------------------------------------------------------ #

    def set_voice_mode(self, enabled):
        """Enable or disable voice-based input.  Must be set before a round starts."""
        self._voice_mode = bool(enabled)

    def inject_voice_beat(self, word, now=None):
        """
        Advance the countdown via a spoken word.
        Protocol: "ready" → "one" → "two" → "three" → [say gesture]
        "three" locks the robot and opens SHOOT_WINDOW immediately.
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
            # "three" = final beat: lock robot + open SHOOT immediately.
            self.beat_count = 3
            self.last_beat_time = now
            self._lock_robot_move()
            self.beat_count = 4
            self.state = "SHOOT_WINDOW"
            self.shoot_open_time  = now
            # Voice needs more time than a physical throw — use 2.5s
            self.shoot_close_time = now + max(self.shoot_window_seconds, 2.50)
            self.tracker_reset_requested = True

    def inject_voice_throw(self, gesture, now=None):
        """
        Resolve the current round with a spoken throw.

        Called by the main loop when a "throw" voice event is received during
        the SHOOT_WINDOW.
        """
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

        self.robot_locked_move = self.ai.choose_robot_move(
            history=self.history,
            round_number=self.round_number
        )
        self.robot_move_command = f"ROBOT_PLAY_{self.robot_locked_move.upper()}"

        if self.robot_output is not None:
            self.robot_output.stage_locked_move(
                command=self.robot_move_command,
                game_mode="FairPlay",
                metadata={
                    "round_number": self.round_number,
                    "player_score": self.player_score,
                    "robot_score": self.robot_score,
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

    def _resolve_round(self, player_gesture, now):
        if self.robot_locked_move is None:
            self._lock_robot_move()

        # Reaction time — time from SHOOT opening to gesture resolved
        reaction_ms = None
        if self.shoot_open_time is not None:
            reaction_ms = round((now - self.shoot_open_time) * 1000)
            if not hasattr(self, "_session_reaction_times"):
                self._session_reaction_times = []
            if 0 < reaction_ms < 5000:  # sanity bounds
                self._session_reaction_times.append(reaction_ms)

        # Track gesture sequence for summary
        if not hasattr(self, "_session_gestures"):
            self._session_gestures = []
        if player_gesture in ("Rock", "Paper", "Scissors"):
            self._session_gestures.append(player_gesture)

        # Store last round gestures for replay display
        self._last_round_player_gest = player_gesture
        self._last_round_robot_gest  = self.robot_locked_move
        self._last_round_banner      = ""  # filled in below

        self.player_gesture = player_gesture
        self.computer_gesture = self.robot_locked_move

        outcome = compare_rps(self.player_gesture, self.computer_gesture)

        if outcome == "win":
            self.player_score += 1
            self.result_banner = "YOU WIN THE ROUND"
            round_result = "player_win"
            player_outcome_for_history = "win"

        elif outcome == "lose":
            self.robot_score += 1
            self.result_banner = f"{self.opponent_label} TAKES THE ROUND"
            round_result = "robot_win"
            player_outcome_for_history = "lose"

        else:
            self.result_banner = "DRAW - THROW AGAIN"
            round_result = "draw"
            player_outcome_for_history = "draw"

        self.history.append({
            "round_number": self.round_number,
            "player_gesture": self.player_gesture,
            "robot_gesture": self.computer_gesture,
            "player_outcome": player_outcome_for_history,
        })

        # Update bandit layer weights based on whether the AI predicted correctly
        if hasattr(self.ai, "update_bandit") and hasattr(self.ai, "last_prediction"):
            pred = self.ai.last_prediction or {}
            predicted_player = pred.get("used_predicted_move")
            if predicted_player:
                self.ai.update_bandit(predicted_player, self.player_gesture)

        if self.robot_output is not None:
            self.robot_output.publish_round_result(
                command=self.robot_move_command,
                game_mode="FairPlay",
                round_result=round_result,
                player_gesture=self.player_gesture,
                robot_gesture=self.computer_gesture,
                metadata={
                    "round_number": self.round_number,
                    "player_score": self.player_score,
                    "robot_score": self.robot_score,
                    "banner": self.result_banner,
                }
            )

        self.last_round_result = round_result
        if hasattr(self, "_last_round_banner"):
            self._last_round_banner = self.result_banner
        self._last_reaction_ms = reaction_ms
        self.state = "ROUND_RESULT"
        self.result_until = now + self.round_result_seconds

    def _round_is_over(self):
        return self.player_score >= self.win_target or self.robot_score >= self.win_target

    def _build_output(self, now):
        score_text = f"YOU {self.player_score} - {self.opponent_label} {self.robot_score}"
        round_text = f"ROUND {self.round_number}"

        # Opponent type from AI last prediction
        pred = getattr(self.ai, "last_prediction", None) or {}
        opp_type    = pred.get("opponent_type", "")
        personality = pred.get("personality", getattr(self.ai, "personality", "Normal"))

        base = {
            "play_mode_label": self.play_mode_label,
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
            "player_score": self.player_score,
            "robot_score": self.robot_score,
            "request_tracker_reset": self.tracker_reset_requested,
            "opponent_type": opp_type,
            "ai_personality": personality,
            "reaction_ms": getattr(self, "_last_reaction_ms", None),
            # Replay: last round gestures (shown briefly in WAITING_FOR_ROCK)
            "last_player_gesture": getattr(self, "_last_round_player_gest", None),
            "last_robot_gesture":  getattr(self, "_last_round_robot_gest",  None),
            "last_banner":         getattr(self, "_last_round_banner", ""),
            # Session summary data
            "session_reaction_times": list(getattr(self, "_session_reaction_times", [])),
            "session_gestures":       list(getattr(self, "_session_gestures", [])),
        }

        if self.state == "ROUND_INTRO":
            base.update({
                "state_label": "Round Intro",
                "main_text": round_text,
                "sub_text": f"FIRST TO {self.win_target} | {score_text}",
            })
            return base

        if self.state == "WAITING_FOR_ROCK":
            base.update({
                "state_label": "Waiting",
                "main_text": "MAKE A FIST" if not self._voice_mode else "VOICE MODE",
                "sub_text": (
                    "Say READY  then  ONE  TWO  THREE"
                    if self._voice_mode
                    else f"{round_text} | {score_text}"
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
                    else "Robot locks on beat 3"
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
                    else "Robot already locked its move"
                ),
                "time_left": 0.0 if self._voice_mode else max(0.0, self.shoot_close_time - now),
            })
            return base

        if self.state == "ROUND_RESULT":
            rxn = getattr(self, "_last_reaction_ms", None)
            rxn_text = f"Reaction: {rxn}ms" if rxn and rxn < 3000 else ""
            base.update({
                "state_label": "Round Result",
                "main_text": self.result_banner,
                "sub_text": rxn_text if rxn_text else score_text,
                "time_left": max(0.0, self.result_until - now),
            })
            return base

        if self.state == "MATCH_RESULT":
            # Build session summary for the summary screen
            rt_list = getattr(self, "_session_reaction_times", [])
            avg_rt = round(sum(rt_list) / len(rt_list)) if rt_list else None
            gestures = getattr(self, "_session_gestures", [])
            from collections import Counter
            gest_counts = Counter(gestures)
            top_gest = gest_counts.most_common(1)[0][0] if gest_counts else "?"
            total_rounds = self.round_number
            player_won = self.player_score > self.robot_score

            base.update({
                "state_label": "Match Result",
                "main_text": self.match_result_banner,
                "sub_text": f"FINAL SCORE | {score_text}",
                "result_banner": self.match_result_banner,
                "time_left": max(0.0, self.match_until - now),
                "session_summary": {
                    "player_won":   player_won,
                    "player_score": self.player_score,
                    "robot_score":  self.robot_score,
                    "total_rounds": total_rounds,
                    "win_rate":     self.player_score / max(total_rounds, 1),
                    "avg_reaction_ms": avg_rt,
                    "top_gesture":  top_gest,
                    "opponent_type": opp_type,
                },
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
                if self._round_is_over():
                    self.state = "MATCH_RESULT"
                    self.match_result_banner = (
                        "YOU WIN THE MATCH"
                        if self.player_score > self.robot_score
                        else f"{self.opponent_label} WINS THE MATCH"
                    )
                    self.match_until = now + self.match_result_seconds
                else:
                    if self.last_round_result != "draw":
                        self.round_number += 1
                    self._prepare_next_round(now)

            return self._build_output(now)

        if self.state == "MATCH_RESULT":
            if now >= self.match_until:
                self.reset_match(now)
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

            # Accept either confirmed or stable Rock for wrist tracking.
            # During fast pumping, confirmed can drop briefly but stable
            # usually holds. Also continue tracking wrist motion if we
            # have a wrist_y and are within the grace window, even if
            # gesture is temporarily Unknown.
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

                            # Important:
                            # ask the main loop to clear the gesture tracker once,
                            # so countdown Rock does not spill into SHOOT.
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

            # Voice mode: throw is resolved by inject_voice_throw().
            # Voice mode: window stays open indefinitely until inject_voice_throw fires.
            # No timer reset — the player speaks when ready.
            if self._voice_mode:
                return self._build_output(now)

            # After the tracker reset, we only want to accept a genuine
            # Paper or Scissors once the new throw has had a moment to form.
            if time_since_open >= self.shoot_change_guard_seconds:
                if confirmed_gesture in {"Paper", "Scissors"}:
                    self._resolve_round(confirmed_gesture, now)
                    return self._build_output(now)

                if stable_gesture in {"Paper", "Scissors"}:
                    self._resolve_round(stable_gesture, now)
                    return self._build_output(now)

            # Slightly fairer Rock assumption timing.
            # This still keeps Rock responsive, but gives Paper/Scissors
            # more time to become the new stable throw.
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
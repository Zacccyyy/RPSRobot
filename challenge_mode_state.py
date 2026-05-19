# ============================================================
# challenge_mode_state.py
#
# Game-state machine for "Challenge Mode".
#
# What this file does:
#   Manages an endless run of RPS where the player tries to build the
#   longest consecutive-win streak before losing.  One loss ends the run.
#   The AI gets progressively harder as the streak grows.
#
# Where it fits:
#   The main loop creates a ChallengeController, then calls
#   controller.update() every frame to get back a UI data dict.
#   ChallengeAI (challenge_ai.py) provides the robot's move.
#   ChallengeStatsLogger (challenge_stats_logger.py) records every round
#   to an Excel workbook for later analysis.
#
# State flow:
#   ROUND_INTRO → WAITING_FOR_ROCK → COUNTDOWN → SHOOT_WINDOW
#     → ROUND_RESULT → (MATCH_RESULT on loss) → reset → ROUND_INTRO
# ============================================================

import time

from challenge_ai import ChallengeAI
from fair_play_ai import UPGRADE_MOVE, DOWNGRADE_MOVE


# Only these three gesture names are valid for gameplay purposes.
VALID_GESTURES = {"Rock", "Paper", "Scissors"}

# What each gesture beats — used to decide the round outcome.
BEATS = {
    "Rock": "Scissors",
    "Paper": "Rock",
    "Scissors": "Paper",
}


def compare_rps(player_move, robot_move):
    """
    Determine the outcome of a single round from the player's perspective.

    Returns "draw", "win", or "lose".
    """
    if player_move == robot_move:
        return "draw"
    if BEATS[player_move] == robot_move:
        return "win"
    return "lose"


class ChallengeController:
    """
    Challenge Mode state machine.

    Rules:
      - Endless run until the player loses a round.
      - Score = consecutive wins (the streak).
      - Persistent high score tracked via ChallengeStatsLogger.
      - Robot locks its move on beat 3 of the physical countdown.
      - Player throws during the SHOOT window.
      - Draws replay the same round (streak unchanged, round not incremented).
      - AI ramps difficulty with the streak via ChallengeAI.ramp_per_win.
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
        # robot_output: optional hardware/BLE bridge (can be None).
        self.robot_output = robot_output
        self._voice_mode = False
        # Use the provided AI or create a default ChallengeAI.
        self.ai = ai or ChallengeAI()
        # stats_logger: ChallengeStatsLogger (or None if logging is disabled).
        self.stats_logger = stats_logger

        # Wrist pump detection thresholds.
        self.down_threshold = down_threshold
        self.up_threshold = up_threshold
        # Minimum gap between beats to prevent double-counting.
        self.beat_cooldown = beat_cooldown
        # Grace window: keep tracking even if Rock briefly disappears.
        self.rock_grace_period = rock_grace_period

        # SHOOT window timing.
        self.shoot_window_seconds = shoot_window_seconds
        # Guard: ignore gestures briefly after window opens to avoid Rock bleed-through.
        self.shoot_change_guard_seconds = shoot_change_guard_seconds
        # Assume Rock if player doesn't switch gestures within this time.
        self.rock_assume_seconds = rock_assume_seconds

        # Display phase durations.
        self.round_intro_seconds  = round_intro_seconds
        self.round_result_seconds = round_result_seconds
        self.game_over_seconds    = game_over_seconds

        # Load the stored high score (0 if no logger or first run).
        self.high_score = self.stats_logger.get_high_score() if self.stats_logger else 0
        self.reset_run()

    def reset(self):
        """
        Reset the active run and reload the high score from storage.
        Keeps the logger reference intact so the next run is still tracked.
        """
        if self.stats_logger is not None:
            self.high_score = self.stats_logger.get_high_score()
        self.reset_run()

    def reset_run(self, now=None):
        """
        Start a brand-new run.
        Clears the streak, history, and all round state, then jumps to ROUND_INTRO.
        """
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
        """
        Clear all per-round motion and gesture tracking state.
        Does NOT touch match-level data (streak, high_score, round_number).
        """
        self.beat_count = 0
        self.phase = "ready_for_down"   # alternates with "waiting_for_up"
        self.top_y    = None            # highest wrist Y in current beat cycle
        self.bottom_y = None            # lowest wrist Y in current beat cycle
        self.last_beat_time = 0.0
        self.last_rock_time = 0.0

        self.shoot_open_time  = None
        self.shoot_close_time = None

        self.robot_locked_move = None
        self.robot_move_command = "PENDING"

        self.player_gesture   = "Unknown"
        self.computer_gesture = "Unknown"

        self.result_banner = ""
        self.result_until  = None

        # One-shot flag: main loop clears the tracker once when this is True.
        self.tracker_reset_requested = False
        self.gesture_assumed = False

        # Last-round gestures for the brief replay overlay.
        self._last_round_player_gest = None
        self._last_round_robot_gest  = None
        self._last_round_banner      = ""

        if self.robot_output is not None:
            self.robot_output.clear_pending_locked()

    def consume_tracker_reset_request(self):
        """
        Called by the main loop after it has flushed the gesture tracker.
        Clears the one-shot flag to prevent repeated resets.
        """
        self.tracker_reset_requested = False

    # ------------------------------------------------------------------ #
    # Voice input injection                                                #
    # ------------------------------------------------------------------ #

    def set_voice_mode(self, enabled):
        """Enable or disable voice-based input for this session."""
        self._voice_mode = bool(enabled)

    def inject_voice_beat(self, word, now=None):
        """
        Advance the countdown via a spoken word.

        Expected sequence: "ready" → "one" → "two" → "three" → [gesture]
        "three" locks the robot and opens the SHOOT window immediately.
        """
        if now is None:
            now = time.monotonic()

        # "ready" moves from the idle waiting state into the countdown.
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
            # Lock robot and open the shoot window — same logic as beat 4 in physical mode.
            self.last_beat_time = now
            self._lock_robot_move()
            self.beat_count = 4
            self.state = "SHOOT_WINDOW"
            self.shoot_open_time  = now
            # Voice mode needs extra time — enforce a 2.5s minimum window.
            self.shoot_close_time = now + max(self.shoot_window_seconds, 2.50)
            self.tracker_reset_requested = True

    def inject_voice_throw(self, gesture, now=None):
        """Resolve the current round with a spoken gesture (called by the main loop)."""
        if now is None:
            now = time.monotonic()

        if self.state == "SHOOT_WINDOW" and gesture in VALID_GESTURES:
            self._resolve_round(gesture, now)

    def _prepare_next_round(self, now):
        """Reset per-round state and enter the intro pause for the next round."""
        self._reset_round_motion()
        self.state = "ROUND_INTRO"
        self.intro_until = now + self.round_intro_seconds

    def _lock_robot_move(self):
        """
        Ask ChallengeAI to choose the robot's move and commit it.

        Also feeds the current emotion snapshot to the AI so it can
        adjust its confidence based on the player's detected mood.
        Guard: if already locked (shouldn't happen), silently returns.
        """
        if self.robot_locked_move is not None:
            return

        # Give the AI the latest emotion reading before it decides.
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
        """
        Last-resort gesture read when the SHOOT window expires.
        Prefers stable → confirmed → raw, returns "Unknown" if all fail.
        """
        stable_gesture    = tracker_state.get("stable_gesture", "Unknown")
        confirmed_gesture = tracker_state.get("confirmed_gesture", "Unknown")
        raw_gesture       = tracker_state.get("raw_gesture", "Unknown")

        if stable_gesture in VALID_GESTURES:
            return stable_gesture
        if confirmed_gesture in VALID_GESTURES:
            return confirmed_gesture
        if raw_gesture in VALID_GESTURES:
            return raw_gesture

        return "Unknown"

    def set_emotion_snapshot(self, snapshot):
        """
        Store the latest emotion snapshot so it's available when the robot locks in.
        Called by the main loop every frame with the current emotion data.
        """
        self.emotion_snapshot = snapshot

    def _log_round(self, round_result, reaction_time_ms=None):
        """
        Write this round's data to the Excel log via the stats_logger.

        Also derives the player's response type (how they transitioned from
        their last gesture) and pulls AI prediction metadata and emotion data
        so the logger has the full picture for each round.

        Does nothing if no stats_logger is configured.
        """
        if self.stats_logger is None:
            return

        # Work out how the player transitioned from the previous gesture.
        # history already has the current round appended, so [-2] is the previous round.
        previous_player_gesture = None
        player_response_type    = None

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
            else:
                # Shouldn't happen with three-gesture RPS, but label it just in case.
                player_response_type = "lateral"

        prediction = self.ai.last_prediction or {}
        em         = self.emotion_snapshot or {}

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
        """
        Lock in both gestures, determine the outcome, and branch on win/draw/loss.

        Win:  streak++, record, go to ROUND_RESULT.
        Draw: no change to streak, record, go to ROUND_RESULT.
        Loss: end the run, go to MATCH_RESULT (game over screen).
        """
        if self.robot_locked_move is None:
            self._lock_robot_move()

        # Measure how fast the player threw after the SHOOT window opened.
        reaction_time_ms = None
        if self.shoot_open_time is not None:
            reaction_time_ms = int(round((now - self.shoot_open_time) * 1000))

        self.player_gesture   = player_gesture
        self.computer_gesture = self.robot_locked_move

        # Save for the replay overlay.
        self._last_round_player_gest = player_gesture
        self._last_round_robot_gest  = self.robot_locked_move or "Unknown"

        outcome = compare_rps(self.player_gesture, self.computer_gesture)

        if outcome == "win":
            self.streak += 1
            # Update the in-memory high score (persistent save happens via logger).
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

            self.last_round_result  = round_result
            self._last_round_banner = self.result_banner
            self.state = "ROUND_RESULT"
            self.result_until = now + self.round_result_seconds
            return

        if outcome == "draw":
            # Draw: streak unchanged, same round replays.
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

            self.last_round_result  = round_result
            self._last_round_banner = self.result_banner
            self.state = "ROUND_RESULT"
            self.result_until = now + self.round_result_seconds
            return

        # Player loses — run ends immediately, jump to game-over screen.
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

        # Finalise the run in the logger so the workbook is fully saved.
        if self.stats_logger is not None:
            self.stats_logger.finalize_run(
                final_streak=self.streak,
                status="completed"
            )

        self.last_round_result  = round_result
        self._last_round_banner = self.result_banner
        self.match_result_banner = "GAME OVER"
        self.state = "MATCH_RESULT"
        self.match_until = now + self.game_over_seconds

    def _build_output(self, now):
        """
        Build the per-frame UI data dict for this state.

        Returns a dict with at minimum:
          state, state_label, main_text, sub_text, time_left,
          score_text, round_text, plus all common gameplay fields.
        """
        score_text = f"STREAK {self.streak} | HIGH {self.high_score}"
        round_text = f"ROUND {self.round_number}"

        # Fields common to every state.
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
            # For Challenge mode, player_score = current streak, robot_score = high score.
            "player_score": self.streak,
            "robot_score": self.high_score,
            "request_tracker_reset": self.tracker_reset_requested,
            "gesture_assumed": self.gesture_assumed,
            "last_player_gesture": self._last_round_player_gest,
            "last_robot_gesture":  self._last_round_robot_gest,
            "last_banner":         self._last_round_banner,
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

        # Catch-all for any unrecognised state.
        base.update({
            "state_label": "Unknown",
            "main_text": "UNKNOWN",
            "sub_text": "",
        })
        return base

    def update(self, wrist_y, tracker_state, now=None):
        """
        Main per-frame update — call this every game loop tick.

        wrist_y:       normalised wrist Y position (None if no hand detected).
        tracker_state: dict from the gesture tracker.
        now:           monotonic timestamp (defaults to time.monotonic()).

        Returns the UI data dict from _build_output().
        """
        if now is None:
            now = time.monotonic()

        confirmed_gesture = tracker_state.get("confirmed_gesture", "Unknown")
        stable_gesture    = tracker_state.get("stable_gesture", "Unknown")

        # ── ROUND_INTRO ──
        if self.state == "ROUND_INTRO":
            if now >= self.intro_until:
                self.state = "WAITING_FOR_ROCK"
            return self._build_output(now)

        # ── ROUND_RESULT ──
        if self.state == "ROUND_RESULT":
            if now >= self.result_until:
                if self.last_round_result == "player_win":
                    # Advance round counter only on a real win (not after a draw).
                    self.round_number += 1
                self._prepare_next_round(now)
            return self._build_output(now)

        # ── MATCH_RESULT (game over screen) ──
        if self.state == "MATCH_RESULT":
            if now >= self.match_until:
                # Reload the high score from storage before starting a new run.
                if self.stats_logger is not None:
                    self.high_score = self.stats_logger.get_high_score()
                self.reset_run(now)
            return self._build_output(now)

        confirmed_rock = confirmed_gesture == "Rock"
        stable_rock    = stable_gesture == "Rock"

        # ── WAITING_FOR_ROCK ──
        if self.state == "WAITING_FOR_ROCK":
            if self._voice_mode:
                # Voice: "ready" spoken → inject_voice_beat() handles transition.
                return self._build_output(now)
            if confirmed_rock and wrist_y is not None:
                # Physical fist detected — start the countdown.
                self.state = "COUNTDOWN"
                self.phase = "ready_for_down"
                self.top_y = wrist_y
                self.bottom_y = wrist_y
                self.last_rock_time = now
            return self._build_output(now)

        # ── COUNTDOWN ──
        if self.state == "COUNTDOWN":
            if self._voice_mode:
                # Voice countdown is fully handled by inject_voice_beat().
                return self._build_output(now)

            # Keep tracking even if Rock flickers off briefly during fast pumping.
            rock_detected = (confirmed_rock or stable_rock) and wrist_y is not None
            within_grace  = (now - self.last_rock_time) <= self.rock_grace_period
            can_track = rock_detected or (within_grace and wrist_y is not None and self.beat_count > 0)

            if rock_detected:
                self.last_rock_time = now

            if can_track:
                if self.phase == "ready_for_down":
                    # Track the peak wrist position to measure downward movement from.
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
                            # Lock robot on beat 3 so it's committed before the throw.
                            self._lock_robot_move()

                        if self.beat_count >= 4:
                            # Beat 4 opens the SHOOT window.
                            self.state = "SHOOT_WINDOW"
                            self.shoot_open_time  = now
                            self.shoot_close_time = now + self.shoot_window_seconds
                            # Flush the tracker so the countdown Rock doesn't leak through.
                            self.tracker_reset_requested = True

                elif self.phase == "waiting_for_up":
                    # Track the lowest point of this pump cycle.
                    if self.bottom_y is None:
                        self.bottom_y = wrist_y
                    self.bottom_y = max(self.bottom_y, wrist_y)

                    moved_up_enough = (self.bottom_y - wrist_y) >= self.up_threshold

                    if moved_up_enough:
                        # Return detected — ready to count the next downward beat.
                        self.phase = "ready_for_down"
                        self.top_y = wrist_y

            else:
                # Hand disappeared and grace period expired — abort this round.
                if not within_grace:
                    self._prepare_next_round(now)

            return self._build_output(now)

        # ── SHOOT_WINDOW ──
        if self.state == "SHOOT_WINDOW":
            time_since_open = now - self.shoot_open_time

            if self._voice_mode:
                # Voice: throw resolved by inject_voice_throw().
                return self._build_output(now)

            # Accept Paper or Scissors after the change guard expires.
            if time_since_open >= self.shoot_change_guard_seconds:
                if confirmed_gesture in {"Paper", "Scissors"}:
                    self._resolve_round(confirmed_gesture, now)
                    return self._build_output(now)

                if stable_gesture in {"Paper", "Scissors"}:
                    self._resolve_round(stable_gesture, now)
                    return self._build_output(now)

            # Assume Rock if nothing else shows up within the rock_assume window.
            if time_since_open >= self.rock_assume_seconds:
                self.gesture_assumed = True
                self._resolve_round("Rock", now)
                return self._build_output(now)

            # Note: this mode does not have a fallback_throw / window expiry path —
            # the rock_assume always fires first. Return and keep waiting.
            return self._build_output(now)

        return self._build_output(now)

# ============================================================
# challenge_mode_state.py
#
# Purpose:
#   State machine for "Challenge Mode" — an endless survival run
#   where the player tries to build the longest consecutive-win
#   streak before losing a single round.
#
# How it works:
#   The main game loop calls ChallengeController.update() every
#   frame with the player's wrist Y position and the gesture
#   tracker state. The controller advances through states and
#   returns a plain dict of UI values for the renderer.
#
#   The robot locks its move on beat 3 of the countdown so it
#   can't react to the player's throw. ChallengeAI picks the
#   move and ramps its difficulty as the streak grows.
#   ChallengeStatsLogger records every round to an Excel file.
#
# State flow:
#   ROUND_INTRO → WAITING_FOR_ROCK → COUNTDOWN
#     → SHOOT_WINDOW → ROUND_RESULT
#     → (MATCH_RESULT on loss)
#     → loops back to ROUND_INTRO for a new run
# ============================================================

import time

from challenge_ai import ChallengeAI
from fair_play_ai import UPGRADE_MOVE, DOWNGRADE_MOVE


# The only three gesture names the game accepts. Anything else is "Unknown".
VALID_GESTURES = {"Rock", "Paper", "Scissors"}

# Maps each gesture to the one it beats — used to score the round.
BEATS = {
    "Rock": "Scissors",
    "Paper": "Rock",
    "Scissors": "Paper",
}


def compare_rps(player_move, robot_move):
    """
    Work out who won a single round from the player's point of view.
    Returns "draw", "win", or "lose".
    """
    if player_move == robot_move:
        return "draw"
    # If the thing the player's move beats is the robot's move, the player wins.
    if BEATS[player_move] == robot_move:
        return "win"
    return "lose"


class ChallengeController:
    """
    Challenge Mode state machine.

    Rules:
      - Endless run until the player loses one round.
      - Score = consecutive wins (the streak).
      - Persistent high score is stored via ChallengeStatsLogger.
      - Robot locks its move on beat 3 so it can't react to the throw.
      - Draws replay the same round without changing the streak.
      - AI increases in difficulty as the streak grows.
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
        # Hardware/BLE bridge — can be None when running without a physical robot.
        self.robot_output = robot_output

        # Whether the player is using voice instead of wrist pumps.
        self._voice_mode = False

        # AI that decides the robot's move each round.
        self.ai = ai or ChallengeAI()

        # Optional stats logger — if None, round logging is silently skipped.
        self.stats_logger = stats_logger

        # --- Wrist-pump detection thresholds ---
        # down_threshold: normalised Y drop needed to register a beat.
        # up_threshold: normalised Y rise needed to confirm the return stroke.
        self.down_threshold = down_threshold
        self.up_threshold   = up_threshold

        # Minimum seconds between two registered beats — prevents jitter double-counting.
        self.beat_cooldown = beat_cooldown

        # How long to keep tracking after the gesture classifier loses "Rock".
        # Fast pumping can cause the classifier to flicker off for a frame.
        self.rock_grace_period = rock_grace_period

        # --- SHOOT window timing ---
        # How long the window stays open for the player to throw.
        self.shoot_window_seconds = shoot_window_seconds

        # Ignore gestures for this many seconds right after the window opens,
        # so the Rock from the final pump beat doesn't instantly resolve as a throw.
        self.shoot_change_guard_seconds = shoot_change_guard_seconds

        # If the player holds Rock for this long without switching, assume they threw Rock.
        self.rock_assume_seconds = rock_assume_seconds

        # --- Display phase durations ---
        self.round_intro_seconds  = round_intro_seconds
        self.round_result_seconds = round_result_seconds
        self.game_over_seconds    = game_over_seconds

        # Load the stored high score from the logger (0 if logging is disabled).
        self.high_score = self.stats_logger.get_high_score() if self.stats_logger else 0

        self.reset_run()

    def reset(self):
        """
        Reset the active run and reload the high score from storage.
        Called by the menu system when the player navigates away and back.
        """
        if self.stats_logger is not None:
            self.high_score = self.stats_logger.get_high_score()
        self.reset_run()

    def reset_run(self, now=None):
        """
        Start a brand-new run from scratch.
        Clears the streak, history, and all round state, then enters ROUND_INTRO.
        """
        if now is None:
            now = time.monotonic()

        # Tell the AI to forget previous rounds so it starts fresh.
        self.ai.reset()

        # Complete history of every round — the AI reads this to predict the next move.
        self.history = []

        self.streak       = 0
        self.round_number = 1

        # Set when the run ends — used for the game-over screen.
        self.match_result_banner = ""
        self.match_until         = None

        # Tracks what happened last round so ROUND_RESULT knows what to do next.
        self.last_round_result = None

        # Latest emotion data from the camera — fed to the AI before it locks its move.
        self.emotion_snapshot = None

        self._reset_round_motion()
        self.state       = "ROUND_INTRO"
        self.intro_until = now + self.round_intro_seconds

    def _reset_round_motion(self):
        """
        Clear per-round motion tracking and gesture state.
        Called at the start of each round without touching streak or high score.
        """
        # Beat counter and which stroke we're waiting for.
        self.beat_count = 0
        self.phase      = "ready_for_down"   # alternates: "ready_for_down" / "waiting_for_up"

        # Peak and trough wrist Y for the current pump cycle.
        self.top_y    = None
        self.bottom_y = None

        # Timestamps used for cooldown and grace-period calculations.
        self.last_beat_time = 0.0
        self.last_rock_time = 0.0

        # Open/close timestamps for the SHOOT window.
        self.shoot_open_time  = None
        self.shoot_close_time = None

        # Robot's move — None until beat 3 locks it in.
        self.robot_locked_move  = None
        self.robot_move_command = "PENDING"

        # What each side threw this round (shown in result and replay overlays).
        self.player_gesture   = "Unknown"
        self.computer_gesture = "Unknown"

        # Result banner text for this round ("YOU SURVIVE", "GAME OVER", etc.).
        self.result_banner = ""
        self.result_until  = None

        # One-shot flag: when True the main loop should flush the gesture tracker.
        self.tracker_reset_requested = False

        # True if Rock was assumed rather than explicitly thrown.
        self.gesture_assumed = False

        # Previous round's gestures for the brief replay overlay.
        self._last_round_player_gest = None
        self._last_round_robot_gest  = None
        self._last_round_banner      = ""

        # Tell the hardware bridge to cancel any staged-but-unsent command.
        if self.robot_output is not None:
            self.robot_output.clear_pending_locked()

    def consume_tracker_reset_request(self):
        """
        Called by the main loop after it has flushed the gesture tracker.
        Clears the one-shot flag so we don't keep requesting resets every frame.
        """
        self.tracker_reset_requested = False

    # ------------------------------------------------------------------ #
    # Voice input                                                          #
    # ------------------------------------------------------------------ #

    def set_voice_mode(self, enabled):
        """Enable or disable voice-based input for this session."""
        self._voice_mode = bool(enabled)

    def inject_voice_beat(self, word, now=None):
        """
        Advance the countdown from a recognised spoken word.

        Expected sequence: "ready" → "one" → "two" → "three"

        "three" simultaneously locks the robot's move and opens the SHOOT window.
        The window is at least 2.5 s so the player has time to speak the gesture.
        """
        if now is None:
            now = time.monotonic()

        # "ready" kicks us out of the idle waiting state and starts the countdown.
        if self.state == "WAITING_FOR_ROCK" and word == "ready":
            self.state          = "COUNTDOWN"
            self.phase          = "ready_for_down"
            self.beat_count     = 0
            self.last_beat_time = now
            self.last_rock_time = now
            return

        # Everything else only matters during an active countdown.
        if self.state != "COUNTDOWN":
            return

        # Keep the grace timer alive so the countdown doesn't abort between words.
        self.last_rock_time = now

        cooldown_ok = (now - self.last_beat_time) >= self.beat_cooldown

        if word in ("one", "two") and cooldown_ok:
            # Each beat word advances the counter.
            self.beat_count    += 1
            self.last_beat_time = now

        elif word == "three" and cooldown_ok:
            # Final beat: lock the robot and open SHOOT immediately.
            self.last_beat_time = now
            self._lock_robot_move()
            self.beat_count = 4   # jump past the normal 4-beat threshold
            self.state = "SHOOT_WINDOW"
            self.shoot_open_time  = now
            # Voice mode gets extra time because speaking takes longer than gesturing.
            self.shoot_close_time = now + max(self.shoot_window_seconds, 2.50)
            self.tracker_reset_requested = True

    def inject_voice_throw(self, gesture, now=None):
        """
        Resolve the current round with a spoken gesture name.
        The main loop calls this when the voice recogniser fires during SHOOT_WINDOW.
        """
        if now is None:
            now = time.monotonic()

        if self.state == "SHOOT_WINDOW" and gesture in VALID_GESTURES:
            self._resolve_round(gesture, now)

    # ------------------------------------------------------------------ #
    # Internal state helpers                                               #
    # ------------------------------------------------------------------ #

    def _prepare_next_round(self, now):
        """Clear per-round state and enter the intro pause before the next round begins."""
        self._reset_round_motion()
        self.state       = "ROUND_INTRO"
        self.intro_until = now + self.round_intro_seconds

    def _lock_robot_move(self):
        """
        Ask ChallengeAI to pick the robot's move and commit it for this round.

        The move is locked on beat 3 (before the player throws) so the robot
        can't react to what it sees. The emotion snapshot is passed to the AI
        so it can adjust its confidence based on the player's detected mood.
        Guard prevents locking twice in one round.
        """
        # Already locked — nothing to do.
        if self.robot_locked_move is not None:
            return

        # Share the latest emotion reading so the AI can factor in player mood.
        if hasattr(self.ai, "set_emotion"):
            self.ai.set_emotion(self.emotion_snapshot)

        self.robot_locked_move = self.ai.choose_robot_move(
            history=self.history,
            streak=self.streak,
            round_number=self.round_number
        )
        self.robot_move_command = f"ROBOT_PLAY_{self.robot_locked_move.upper()}"

        # Stage the move with the physical robot if one is connected.
        if self.robot_output is not None:
            self.robot_output.stage_locked_move(
                command=self.robot_move_command,
                game_mode="Challenge",
                metadata={
                    "round_number": self.round_number,
                    "streak":       self.streak,
                    "high_score":   self.high_score,
                }
            )

    def _fallback_throw(self, tracker_state):
        """
        Last-resort gesture lookup when the SHOOT window expires.
        Tries stable → confirmed → raw in order of reliability.
        Returns "Unknown" if all three fail.
        """
        for key in ("stable_gesture", "confirmed_gesture", "raw_gesture"):
            gesture = tracker_state.get(key, "Unknown")
            if gesture in VALID_GESTURES:
                return gesture
        return "Unknown"

    def set_emotion_snapshot(self, snapshot):
        """
        Store the latest emotion snapshot so it's ready when the robot locks its move.
        The main loop calls this every frame with the current emotion data from the camera.
        """
        self.emotion_snapshot = snapshot

    def _log_round(self, round_result, reaction_time_ms=None):
        """
        Write this round's data to the Excel log via stats_logger.

        Derives the player's response type (did they repeat, upgrade, or downgrade
        their gesture compared to last round?) and pulls AI prediction metadata
        and emotion readings so the log has the full picture for each round.

        Does nothing if no stats_logger is configured.
        """
        if self.stats_logger is None:
            return

        # Work out how the player transitioned from their previous gesture.
        # At this point history already has the current round appended, so
        # index -2 is the previous round (if it exists).
        previous_player_gesture = None
        player_response_type    = None

        if len(self.history) >= 2:
            previous_player_gesture = self.history[-2]["player_gesture"]
            current_gesture         = self.player_gesture

            if current_gesture == previous_player_gesture:
                player_response_type = "stay"
            elif UPGRADE_MOVE.get(previous_player_gesture) == current_gesture:
                player_response_type = "upgrade"
            elif DOWNGRADE_MOVE.get(previous_player_gesture) == current_gesture:
                player_response_type = "downgrade"
            else:
                # Shouldn't happen in standard RPS, but label it just in case.
                player_response_type = "lateral"

        prediction = self.ai.last_prediction or {}
        em         = self.emotion_snapshot    or {}

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
        Finalise the round: score it, update the streak, log it, and transition state.

        Win:  streak++, ROUND_RESULT → next round.
        Draw: streak unchanged, ROUND_RESULT → replay.
        Loss: run ends, MATCH_RESULT (game-over screen).
        """
        # Edge case: lock the robot now if beat 3 was somehow skipped.
        if self.robot_locked_move is None:
            self._lock_robot_move()

        # Measure how quickly the player threw after the window opened.
        reaction_time_ms = None
        if self.shoot_open_time is not None:
            reaction_time_ms = int(round((now - self.shoot_open_time) * 1000))

        # Lock in both gestures for display and history.
        self.player_gesture   = player_gesture
        self.computer_gesture = self.robot_locked_move

        # Cache for the replay overlay shown at the start of the next round.
        self._last_round_player_gest = player_gesture
        self._last_round_robot_gest  = self.robot_locked_move or "Unknown"

        outcome = compare_rps(self.player_gesture, self.computer_gesture)

        # --- Determine outcome-specific values ---

        if outcome == "win":
            self.streak   += 1
            # Keep the in-memory high score up to date (logger persists it on disk).
            self.high_score = max(self.high_score, self.streak)
            self.result_banner         = "YOU SURVIVE"
            round_result               = "player_win"
            player_outcome_for_history = "win"

        elif outcome == "draw":
            # Draw: streak is unaffected, same round will replay.
            self.result_banner         = "DRAW - GO AGAIN"
            round_result               = "draw"
            player_outcome_for_history = "draw"

        else:
            # Loss: run is over.
            self.result_banner         = "GAME OVER"
            round_result               = "robot_win"
            player_outcome_for_history = "lose"

        # --- Shared steps for all outcomes ---

        # Append this round to history so the AI can learn from it.
        self.history.append({
            "round_number":   self.round_number,
            "player_gesture": self.player_gesture,
            "robot_gesture":  self.computer_gesture,
            "player_outcome": player_outcome_for_history,
        })

        # Push the result to the physical robot if one is connected.
        if self.robot_output is not None:
            self.robot_output.publish_round_result(
                command=self.robot_move_command,
                game_mode="Challenge",
                round_result=round_result,
                player_gesture=self.player_gesture,
                robot_gesture=self.computer_gesture,
                metadata={
                    "round_number": self.round_number,
                    "streak":       self.streak,
                    "high_score":   self.high_score,
                    "banner":       self.result_banner,
                }
            )

        # Write the round to the Excel log.
        self._log_round(round_result, reaction_time_ms=reaction_time_ms)

        # Cache the banner so the replay overlay can show it next round.
        self.last_round_result  = round_result
        self._last_round_banner = self.result_banner

        # --- State transition ---

        if outcome == "lose":
            # Finalise the run in the logger so the workbook is fully saved.
            if self.stats_logger is not None:
                self.stats_logger.finalize_run(
                    final_streak=self.streak,
                    status="completed"
                )
            # Jump straight to the game-over screen instead of ROUND_RESULT.
            self.match_result_banner = "GAME OVER"
            self.state       = "MATCH_RESULT"
            self.match_until = now + self.game_over_seconds

        else:
            # Win or draw — show the round result briefly then continue.
            self.state        = "ROUND_RESULT"
            self.result_until = now + self.round_result_seconds

    def _build_output(self, now):
        """
        Assemble the per-frame UI data dict for the current state.

        Returns a dict with at minimum: state, state_label, main_text, sub_text,
        time_left, score_text, round_text, plus all common gameplay fields.
        The renderer reads this dict directly — no rendering happens here.
        """
        score_text = f"STREAK {self.streak} | HIGH {self.high_score}"
        round_text = f"ROUND {self.round_number}"

        # Fields sent to the renderer on every frame regardless of state.
        base = {
            "play_mode_label":  "Challenge",
            "state":            self.state,
            "beat_count":       self.beat_count,
            "time_left":        0.0,
            "player_gesture":   self.player_gesture,
            "computer_gesture": self.computer_gesture,
            "robot_move_command": self.robot_move_command,
            "result_banner":    self.result_banner,
            "score_text":       score_text,
            "round_text":       round_text,
            "round_number":     self.round_number,
            # player_score maps to the streak; robot_score maps to the high score
            # so the renderer can display them in the same score widget as Fair Play.
            "player_score":     self.streak,
            "robot_score":      self.high_score,
            "request_tracker_reset": self.tracker_reset_requested,
            "gesture_assumed":  self.gesture_assumed,
            "last_player_gesture": self._last_round_player_gest,
            "last_robot_gesture":  self._last_round_robot_gest,
            "last_banner":         self._last_round_banner,
        }

        # Each if-block adds the state-specific text fields.

        if self.state == "ROUND_INTRO":
            base.update({
                "state_label": "Round Intro",
                "main_text":   round_text,
                "sub_text":    score_text,
            })
            return base

        if self.state == "WAITING_FOR_ROCK":
            base.update({
                "state_label": "Waiting",
                "main_text": "VOICE MODE" if self._voice_mode else "MAKE A FIST",
                "sub_text": (
                    "Say READY  then  ONE  TWO  THREE"
                    if self._voice_mode
                    else "KEEP THE STREAK ALIVE"
                ),
            })
            return base

        if self.state == "COUNTDOWN":
            # Show "READY" before the first beat, then show the beat number.
            main_text = "READY" if self.beat_count == 0 else str(min(self.beat_count, 3))
            base.update({
                "state_label": "Countdown",
                "main_text":   main_text,
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
                "main_text":   "SHOOT!",
                "sub_text": (
                    "Say ROCK, PAPER, or SCISSORS"
                    if self._voice_mode
                    else "One loss ends the run"
                ),
                # In voice mode there's no countdown bar, so time_left stays 0.
                "time_left": 0.0 if self._voice_mode else max(0.0, self.shoot_close_time - now),
            })
            return base

        if self.state == "ROUND_RESULT":
            base.update({
                "state_label": "Round Result",
                "main_text":   self.result_banner,
                "sub_text":    score_text,
                "time_left":   max(0.0, self.result_until - now),
            })
            return base

        if self.state == "MATCH_RESULT":
            base.update({
                "state_label":   "Game Over",
                "main_text":     self.match_result_banner,
                "sub_text":      f"FINAL STREAK {self.streak} | HIGH {self.high_score}",
                "result_banner": self.match_result_banner,
                "time_left":     max(0.0, self.match_until - now),
            })
            return base

        # Catch-all — should never be reached with a valid state value.
        base.update({
            "state_label": "Unknown",
            "main_text":   "UNKNOWN",
            "sub_text":    "",
        })
        return base

    def update(self, wrist_y, tracker_state, now=None):
        """
        Main per-frame update — call this every game loop tick.

        wrist_y:       normalised Y position of the player's wrist
                       (None when no hand is visible).
        tracker_state: dict from the gesture tracker, expected keys:
                         "confirmed_gesture", "stable_gesture", "raw_gesture"
        now:           monotonic timestamp; defaults to time.monotonic().

        Returns the UI data dict produced by _build_output().
        """
        if now is None:
            now = time.monotonic()

        # Pull the two most-used gesture fields up front.
        confirmed_gesture = tracker_state.get("confirmed_gesture", "Unknown")
        stable_gesture    = tracker_state.get("stable_gesture",    "Unknown")

        # ── ROUND_INTRO: wait for the intro timer, then show the fist prompt. ──
        if self.state == "ROUND_INTRO":
            if now >= self.intro_until:
                self.state = "WAITING_FOR_ROCK"
            return self._build_output(now)

        # ── ROUND_RESULT: hold the result banner, then move to the next round. ──
        if self.state == "ROUND_RESULT":
            if now >= self.result_until:
                # Only increment the round counter on a real win, not after a draw.
                if self.last_round_result == "player_win":
                    self.round_number += 1
                self._prepare_next_round(now)
            return self._build_output(now)

        # ── MATCH_RESULT: hold the game-over screen, then start a new run. ──
        if self.state == "MATCH_RESULT":
            if now >= self.match_until:
                # Reload the high score from storage in case it changed during the run.
                if self.stats_logger is not None:
                    self.high_score = self.stats_logger.get_high_score()
                self.reset_run(now)
            return self._build_output(now)

        # Pre-compute Rock booleans — used in both WAITING_FOR_ROCK and COUNTDOWN.
        confirmed_rock = confirmed_gesture == "Rock"
        stable_rock    = stable_gesture    == "Rock"

        # ── WAITING_FOR_ROCK: hold until the player makes a fist. ──
        if self.state == "WAITING_FOR_ROCK":
            if self._voice_mode:
                # Voice: inject_voice_beat("ready") handles the transition.
                return self._build_output(now)
            if confirmed_rock and wrist_y is not None:
                # Physical fist detected — start the countdown.
                self.state          = "COUNTDOWN"
                self.phase          = "ready_for_down"
                self.top_y          = wrist_y
                self.bottom_y       = wrist_y
                self.last_rock_time = now
            return self._build_output(now)

        # ── COUNTDOWN: count wrist pumps (beats) up to 4. ──
        if self.state == "COUNTDOWN":
            if self._voice_mode:
                # Voice countdown is fully handled by inject_voice_beat().
                return self._build_output(now)

            # Keep tracking even if Rock flickers off briefly during fast pumping.
            rock_detected = (confirmed_rock or stable_rock) and wrist_y is not None
            within_grace  = (now - self.last_rock_time) <= self.rock_grace_period

            # can_track: True while a rock is visible OR during the grace window
            # (only after the first beat so we don't continue from a stale position).
            can_track = rock_detected or (within_grace and wrist_y is not None and self.beat_count > 0)

            if rock_detected:
                # Reset the grace timer while Rock is actively visible.
                self.last_rock_time = now

            if can_track:
                if self.phase == "ready_for_down":
                    # Track the highest wrist position to measure the drop from.
                    # (Y grows downward in screen coords, so "highest" = smallest Y.)
                    if self.top_y is None:
                        self.top_y = wrist_y
                    self.top_y = min(self.top_y, wrist_y)

                    # A beat fires when the wrist drops far enough AND the cooldown is clear.
                    moved_down_enough = (wrist_y - self.top_y) >= self.down_threshold
                    cooldown_ok       = (now - self.last_beat_time) >= self.beat_cooldown

                    if moved_down_enough and cooldown_ok:
                        # Downward pump counted — advance the beat counter.
                        self.beat_count    += 1
                        self.last_beat_time = now
                        self.phase          = "waiting_for_up"
                        self.bottom_y       = wrist_y

                        if self.beat_count >= 3:
                            # Beat 3: lock the robot's move now, before the player throws.
                            self._lock_robot_move()

                        if self.beat_count >= 4:
                            # Beat 4: open the SHOOT window.
                            self.state            = "SHOOT_WINDOW"
                            self.shoot_open_time  = now
                            self.shoot_close_time = now + self.shoot_window_seconds
                            # Ask the main loop to flush the tracker so the countdown
                            # Rock doesn't instantly resolve as the player's throw.
                            self.tracker_reset_requested = True

                elif self.phase == "waiting_for_up":
                    # Track the lowest wrist position (the trough of the pump).
                    if self.bottom_y is None:
                        self.bottom_y = wrist_y
                    self.bottom_y = max(self.bottom_y, wrist_y)

                    # Wait until the wrist rises far enough to confirm the upstroke.
                    moved_up_enough = (self.bottom_y - wrist_y) >= self.up_threshold

                    if moved_up_enough:
                        # Upstroke confirmed — ready to count the next downward beat.
                        self.phase = "ready_for_down"
                        self.top_y = wrist_y

            else:
                # Rock disappeared and the grace period has expired — player dropped out.
                # Cancel the countdown and go back to WAITING_FOR_ROCK.
                if not within_grace:
                    self._prepare_next_round(now)

            return self._build_output(now)

        # ── SHOOT_WINDOW: watch for the player's throw gesture. ──
        if self.state == "SHOOT_WINDOW":
            time_since_open = now - self.shoot_open_time

            if self._voice_mode:
                # Voice throw is handled by inject_voice_throw() — nothing to do here.
                return self._build_output(now)

            # Change guard: ignore gestures for a brief moment right after the window
            # opens to prevent the Rock from the final pump beat from resolving instantly.
            if time_since_open >= self.shoot_change_guard_seconds:
                # Paper or Scissors seen — resolve immediately.
                if confirmed_gesture in {"Paper", "Scissors"}:
                    self._resolve_round(confirmed_gesture, now)
                    return self._build_output(now)
                if stable_gesture in {"Paper", "Scissors"}:
                    self._resolve_round(stable_gesture, now)
                    return self._build_output(now)

            # Rock assumption: if the player hasn't changed gesture by this point,
            # treat it as a deliberate Rock throw. The threshold is slightly larger
            # than the change guard so Paper/Scissors still get a fair window.
            if time_since_open >= self.rock_assume_seconds:
                self.gesture_assumed = True
                self._resolve_round("Rock", now)
                return self._build_output(now)

            # Note: unlike Fair Play mode there is no explicit window-expiry fallback
            # here — rock_assume_seconds always fires first, so the window never
            # closes without a resolution.
            return self._build_output(now)

        # Should never reach here, but always return a valid dict.
        return self._build_output(now)

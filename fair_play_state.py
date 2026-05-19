# ============================================================
# fair_play_state.py
#
# Game-state machine for "Fair Play" mode.
#
# What this file does:
#   Manages everything that happens during a Fair Play match —
#   the countdown (physical pump or voice), the SHOOT window,
#   round resolution, score tracking, and the match-result screen.
#
# Where it fits:
#   The main loop calls FairPlayController.update() every frame,
#   passing the current wrist Y position and tracker state.
#   The controller returns a dict of UI data to be rendered.
#   FairPlayAI (from fair_play_ai.py) is used internally to
#   choose the robot's locked move before the player throws.
# ============================================================

import time
from collections import Counter

from fair_play_ai import FairPlayAI


# Gestures the game accepts — anything else is treated as "Unknown".
VALID_GESTURES = {"Rock", "Paper", "Scissors"}

# What each gesture beats — used to decide the round outcome.
COUNTER_MOVE = {
    "Rock": "Paper",
    "Paper": "Scissors",
    "Scissors": "Rock",
}

# The reverse: for each gesture, which gesture does it beat?
BEATS = {
    "Rock": "Scissors",
    "Paper": "Rock",
    "Scissors": "Paper",
}


def compare_rps(player_move, robot_move):
    """
    Determine the outcome of a single round.

    Returns:
      "draw"  — both threw the same thing
      "win"   — player's gesture beats the robot's
      "lose"  — robot's gesture beats the player's
    """
    if player_move == robot_move:
        return "draw"
    if BEATS[player_move] == robot_move:
        return "win"
    return "lose"


class FairPlayController:
    """
    Fair Play Mode state machine.

    Rules:
      - First player to win `win_target` rounds (default 2) wins the match.
      - The robot locks in its move on beat 3 of the countdown.
      - The player throws during the SHOOT window after beat 3.
      - Draws replay the same round (round number doesn't increment).

    State flow (normal physical path):
      ROUND_INTRO
        → WAITING_FOR_ROCK   (player makes a fist)
        → COUNTDOWN          (pump fist up/down for beats 1, 2, 3)
        → SHOOT_WINDOW       (player throws — resolved instantly)
        → ROUND_RESULT       (brief result display)
        → MATCH_RESULT       (if match is over)
        → ROUND_INTRO        (next round, or reset if match ended)

    Voice mode path uses inject_voice_beat() and inject_voice_throw() instead
    of wrist-motion detection — the state names stay the same.
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
        # robot_output: optional hardware/BLE output bridge (can be None).
        self.robot_output = robot_output
        self._voice_mode = False
        # Use the provided AI or create a default FairPlayAI instance.
        self.ai = ai or FairPlayAI()
        self.win_target = win_target
        self.play_mode_label = play_mode_label
        self.opponent_label = "ROBOT"

        # Wrist motion thresholds — how far the wrist must move to register a beat.
        self.down_threshold = down_threshold
        self.up_threshold = up_threshold
        # Minimum seconds between beats so rapid micro-movements don't double-count.
        self.beat_cooldown = beat_cooldown
        # How long after the last Rock detection to keep tracking motion.
        self.rock_grace_period = rock_grace_period

        # How long the SHOOT window stays open.
        self.shoot_window_seconds = shoot_window_seconds
        # Guard: ignore gestures for this many seconds right after the window opens
        # so the Rock from the final beat doesn't immediately resolve as a throw.
        self.shoot_change_guard_seconds = shoot_change_guard_seconds
        # If this many seconds pass with no Paper/Scissors, assume the player kept Rock.
        self.rock_assume_seconds = rock_assume_seconds

        # Timing for each display phase.
        self.round_intro_seconds = round_intro_seconds
        self.round_result_seconds = round_result_seconds
        self.match_result_seconds = match_result_seconds

        self.reset_match()

    def reset(self):
        """Public alias for reset_match() — called by the menu system."""
        self.reset_match()

    def reset_match(self, now=None):
        """
        Fully reset the controller for a new match.
        Clears all scores, history, and per-round state.
        """
        if now is None:
            now = time.monotonic()

        self.ai.reset()
        self.history = []

        self.player_score = 0
        self.robot_score = 0
        self.round_number = 1

        self.match_result_banner = ""
        self.match_until = None

        # Session-level stats that survive across rounds within a match.
        self._session_reaction_times = []
        self._session_gestures       = []
        self._last_round_player_gest = None
        self._last_round_robot_gest  = None
        # NOTE: _last_round_banner is intentionally NOT reset here.
        # Keeping it lets the ESC overlay and replay display show the previous
        # round's outcome banner while the new round's countdown starts.
        # Only reset_match() clears it — and this IS reset_match, so it's set below.

        self._reset_round_motion()
        self.state = "ROUND_INTRO"
        self.intro_until = now + self.round_intro_seconds

    def _reset_round_motion(self):
        """
        Reset only the per-round motion-tracking and gesture state.
        Called at the start of each round without touching match-level data.
        """
        # Countdown pump tracking.
        self.beat_count = 0
        self.phase = "ready_for_down"   # alternates between "ready_for_down" and "waiting_for_up"
        self.top_y = None               # highest Y seen during this beat cycle
        self.bottom_y = None            # lowest Y seen during this beat cycle
        self.last_beat_time = 0.0
        self.last_rock_time = 0.0

        # SHOOT window timing.
        self.shoot_open_time = None
        self.shoot_close_time = None

        # Robot move state — PENDING until beat 3 locks it.
        self.robot_locked_move = None
        self.robot_move_command = "PENDING"

        # Current-round gesture display values (set when the round resolves).
        self.player_gesture = "Unknown"
        self.computer_gesture = "Unknown"

        # Result display for this round.
        self.result_banner = ""
        self.last_round_result = None
        self.result_until = None

        # One-shot flag: True = main loop should clear the gesture tracker once.
        # Prevents the Rock from countdown from leaking into the SHOOT window.
        self.tracker_reset_requested = False
        self.gesture_assumed = False
        self._last_reaction_ms = None

        # Tell the hardware bridge to cancel any staged-but-not-sent command.
        if self.robot_output is not None:
            self.robot_output.clear_pending_locked()

    def consume_tracker_reset_request(self):
        """
        Called by the main loop after it has cleared the gesture tracker.
        Clears the one-shot flag so we don't keep asking for resets.
        """
        self.tracker_reset_requested = False

    # ------------------------------------------------------------------ #
    # Voice input injection                                                #
    # ------------------------------------------------------------------ #

    def set_voice_mode(self, enabled):
        """
        Enable or disable voice-based input.
        In voice mode the wrist-pump detection is skipped entirely — the player
        speaks "ready / one / two / three / [gesture]" instead.
        Must be set before a round starts for it to take effect cleanly.
        """
        self._voice_mode = bool(enabled)

    def inject_voice_beat(self, word, now=None):
        """
        Advance the countdown from a spoken word.

        Expected sequence: "ready" → "one" → "two" → "three" → [gesture]

        "three" immediately locks the robot and opens the SHOOT window.
        Voice windows are wider (2.5s minimum) so the player has time to speak.
        """
        if now is None:
            now = time.monotonic()

        # "ready" in WAITING_FOR_ROCK kicks off the countdown.
        if self.state == "WAITING_FOR_ROCK" and word == "ready":
            self.state = "COUNTDOWN"
            self.phase = "ready_for_down"
            self.beat_count = 0
            self.last_beat_time = now
            self.last_rock_time = now
            return

        # All other beats only work during COUNTDOWN.
        if self.state != "COUNTDOWN":
            return

        self.last_rock_time = now
        cooldown_ok = (now - self.last_beat_time) >= self.beat_cooldown

        if word in ("one", "two") and cooldown_ok:
            # Each spoken beat advances the counter.
            self.beat_count += 1
            self.last_beat_time = now

        elif word == "three" and cooldown_ok:
            # Final beat: lock the robot's move and open the shoot window now.
            self.last_beat_time = now
            self._lock_robot_move()
            self.beat_count = 4
            self.state = "SHOOT_WINDOW"
            self.shoot_open_time  = now
            # Voice needs more time to say the gesture — enforce a 2.5s minimum.
            self.shoot_close_time = now + max(self.shoot_window_seconds, 2.50)
            self.tracker_reset_requested = True

    def inject_voice_throw(self, gesture, now=None):
        """
        Resolve the current round with a spoken gesture.
        Called by the main loop when it receives a voice "throw" event during SHOOT_WINDOW.
        """
        if now is None:
            now = time.monotonic()

        if self.state == "SHOOT_WINDOW" and gesture in VALID_GESTURES:
            self._resolve_round(gesture, now)

    def _prepare_next_round(self, now):
        """Reset per-round state and transition to ROUND_INTRO for the next round."""
        self._reset_round_motion()
        self.state = "ROUND_INTRO"
        self.intro_until = now + self.round_intro_seconds

    def _lock_robot_move(self):
        """
        Ask the AI to choose the robot's move and lock it in.

        Guard: if already locked (e.g. called twice in one round), do nothing.
        Also stages the move with the hardware bridge so the physical arm
        can prepare before the player throws.
        """
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
        """
        Last-resort gesture picker when the SHOOT window closes without a clear throw.

        Checks stable_gesture → confirmed_gesture → raw_gesture in order of reliability.
        Returns "Unknown" if none of them are valid (round is then replayed).
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

    def _resolve_round(self, player_gesture, now):
        """
        Lock in both gestures, determine the outcome, update scores, and
        record the round in history for the AI to learn from.

        Also updates the Thompson Sampling bandit inside the AI so it can
        adjust which prediction layers it trusts going forward.
        """
        # If the robot never got locked (e.g. edge case), lock it now.
        if self.robot_locked_move is None:
            self._lock_robot_move()

        # Calculate how fast the player threw after the SHOOT window opened.
        reaction_ms = None
        if self.shoot_open_time is not None:
            reaction_ms = round((now - self.shoot_open_time) * 1000)
            if not hasattr(self, "_session_reaction_times"):
                self._session_reaction_times = []
            # Sanity filter: only record plausible reaction times.
            if 0 < reaction_ms < 5000:
                self._session_reaction_times.append(reaction_ms)

        # Track the gesture sequence for the end-of-match summary.
        if not hasattr(self, "_session_gestures"):
            self._session_gestures = []
        if player_gesture in ("Rock", "Paper", "Scissors"):
            self._session_gestures.append(player_gesture)

        # Store the round's gestures so the replay overlay can show them.
        self._last_round_player_gest = player_gesture
        self._last_round_robot_gest  = self.robot_locked_move

        self.player_gesture   = player_gesture
        self.computer_gesture = self.robot_locked_move

        outcome = compare_rps(self.player_gesture, self.computer_gesture)

        # Update scores and set the result banner text.
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
            # Draw — round replays, scores unchanged.
            self.result_banner = "DRAW - THROW AGAIN"
            round_result = "draw"
            player_outcome_for_history = "draw"

        # Append this round to the AI's history so it can learn from it.
        self.history.append({
            "round_number": self.round_number,
            "player_gesture": self.player_gesture,
            "robot_gesture": self.computer_gesture,
            "player_outcome": player_outcome_for_history,
        })

        # Let the AI bandit know whether its prediction was accurate.
        if hasattr(self.ai, "update_bandit") and hasattr(self.ai, "last_prediction"):
            pred = self.ai.last_prediction or {}
            predicted_player = pred.get("used_predicted_move")
            if predicted_player:
                self.ai.update_bandit(predicted_player, self.player_gesture)

        # Notify the hardware bridge of the round result.
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

        self.last_round_result  = round_result
        self._last_round_banner = self.result_banner
        self._last_reaction_ms  = reaction_ms
        self.state = "ROUND_RESULT"
        self.result_until = now + self.round_result_seconds

    def _round_is_over(self):
        """Return True if either player has reached the win target."""
        return self.player_score >= self.win_target or self.robot_score >= self.win_target

    def _build_output(self, now):
        """
        Build the dict of UI data for the current frame.

        Every state returns the same base fields plus state-specific
        main_text / sub_text / time_left. The renderer reads this dict
        directly — nothing is rendered here.
        """
        score_text = f"YOU {self.player_score} - {self.opponent_label} {self.robot_score}"
        round_text = f"ROUND {self.round_number}"

        # Pull AI metadata for display (opponent type, personality name).
        pred = getattr(self.ai, "last_prediction", None) or {}
        opp_type    = pred.get("opponent_type", "")
        personality = pred.get("personality", getattr(self.ai, "personality", "Normal"))

        # Fields common to every state.
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
            "gesture_assumed": self.gesture_assumed,
            "opponent_type": opp_type,
            "ai_personality": personality,
            "reaction_ms": self._last_reaction_ms,
            # Last-round gestures for the brief replay shown in WAITING_FOR_ROCK.
            "last_player_gesture": getattr(self, "_last_round_player_gest", None),
            "last_robot_gesture":  getattr(self, "_last_round_robot_gest",  None),
            "last_banner":         getattr(self, "_last_round_banner", ""),
            # Full session lists for the summary screen.
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
            # Show "READY" until the first beat, then show the beat number.
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
                # In voice mode the window never times out, so time_left stays 0.
                "time_left": 0.0 if self._voice_mode else max(0.0, self.shoot_close_time - now),
            })
            return base

        if self.state == "ROUND_RESULT":
            rxn = getattr(self, "_last_reaction_ms", None)
            # Only show reaction time if it looks plausible (under 3 seconds).
            rxn_text = f"Reaction: {rxn}ms" if rxn and rxn < 3000 else ""
            base.update({
                "state_label": "Round Result",
                "main_text": self.result_banner,
                "sub_text": rxn_text if rxn_text else score_text,
                "time_left": max(0.0, self.result_until - now),
            })
            return base

        if self.state == "MATCH_RESULT":
            # Build session summary stats for the summary screen.
            rt_list = getattr(self, "_session_reaction_times", [])
            avg_rt = round(sum(rt_list) / len(rt_list)) if rt_list else None
            gestures = getattr(self, "_session_gestures", [])
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

        # Catch-all for any unexpected state value.
        base.update({
            "state_label": "Unknown",
            "main_text": "UNKNOWN",
            "sub_text": "",
        })
        return base

    def update(self, wrist_y, tracker_state, now=None):
        """
        Main per-frame update — call this every game loop tick.

        wrist_y:       normalised Y position of the player's wrist (None if no hand).
        tracker_state: dict from the gesture tracker with "confirmed_gesture",
                       "stable_gesture", "raw_gesture" keys.
        now:           monotonic timestamp (defaults to time.monotonic()).

        Returns the UI output dict from _build_output().
        """
        if now is None:
            now = time.monotonic()

        confirmed_gesture = tracker_state.get("confirmed_gesture", "Unknown")
        stable_gesture    = tracker_state.get("stable_gesture", "Unknown")

        # ── ROUND_INTRO: wait for the intro timer, then ask for a Rock. ──
        if self.state == "ROUND_INTRO":
            if now >= self.intro_until:
                self.state = "WAITING_FOR_ROCK"
            return self._build_output(now)

        # ── ROUND_RESULT: wait for the display timer, then move on. ──
        if self.state == "ROUND_RESULT":
            if now >= self.result_until:
                if self._round_is_over():
                    # Match decided — show the final banner.
                    self.state = "MATCH_RESULT"
                    self.match_result_banner = (
                        "YOU WIN THE MATCH"
                        if self.player_score > self.robot_score
                        else f"{self.opponent_label} WINS THE MATCH"
                    )
                    self.match_until = now + self.match_result_seconds
                else:
                    # Round finished, match still going — increment round number
                    # unless it was a draw (draws replay the same round).
                    if self.last_round_result != "draw":
                        self.round_number += 1
                    self._prepare_next_round(now)
            return self._build_output(now)

        # ── MATCH_RESULT: auto-reset to a new match after the display time. ──
        if self.state == "MATCH_RESULT":
            if now >= self.match_until:
                self.reset_match(now)
            return self._build_output(now)

        # Precompute rock booleans — used in WAITING_FOR_ROCK and COUNTDOWN.
        confirmed_rock = confirmed_gesture == "Rock"
        stable_rock    = stable_gesture == "Rock"

        # ── WAITING_FOR_ROCK: hold until the player makes a fist. ──
        if self.state == "WAITING_FOR_ROCK":
            if self._voice_mode:
                # In voice mode, "ready" spoken → inject_voice_beat() handles the transition.
                return self._build_output(now)
            if confirmed_rock and wrist_y is not None:
                # Fist detected — kick off the countdown.
                self.state = "COUNTDOWN"
                self.phase = "ready_for_down"
                self.top_y = wrist_y
                self.bottom_y = wrist_y
                self.last_rock_time = now
            return self._build_output(now)

        # ── COUNTDOWN: track wrist pumps to count beats. ──
        if self.state == "COUNTDOWN":
            if self._voice_mode:
                # Voice countdown advances only when spoken — nothing to track here.
                return self._build_output(now)

            # We keep tracking even if the gesture briefly drops (fast pumping can
            # cause the classifier to lose the Rock for a frame). The grace window
            # (rock_grace_period) lets us continue if the wrist is still moving.
            rock_detected = (confirmed_rock or stable_rock) and wrist_y is not None
            within_grace  = (now - self.last_rock_time) <= self.rock_grace_period
            can_track = rock_detected or (within_grace and wrist_y is not None and self.beat_count > 0)

            if rock_detected:
                # Reset the grace timer while a solid Rock is visible.
                self.last_rock_time = now

            if can_track:
                if self.phase == "ready_for_down":
                    # Track the highest wrist position seen (the "top" of the pump).
                    if self.top_y is None:
                        self.top_y = wrist_y
                    self.top_y = min(self.top_y, wrist_y)

                    moved_down_enough = (wrist_y - self.top_y) >= self.down_threshold
                    cooldown_ok = (now - self.last_beat_time) >= self.beat_cooldown

                    if moved_down_enough and cooldown_ok:
                        # Downward pump detected — count the beat.
                        self.beat_count += 1
                        self.last_beat_time = now
                        self.phase = "waiting_for_up"
                        self.bottom_y = wrist_y

                        if self.beat_count >= 3:
                            # Beat 3 reached — lock the robot's move now so it's
                            # committed before the player throws.
                            self._lock_robot_move()

                        if self.beat_count >= 4:
                            # Beat 4 — open the SHOOT window.
                            self.state = "SHOOT_WINDOW"
                            self.shoot_open_time  = now
                            self.shoot_close_time = now + self.shoot_window_seconds
                            # Ask the main loop to flush the tracker so the Rock
                            # from the final beat doesn't resolve as the throw.
                            self.tracker_reset_requested = True

                elif self.phase == "waiting_for_up":
                    # Track the lowest wrist position seen (the "bottom" of the pump).
                    if self.bottom_y is None:
                        self.bottom_y = wrist_y
                    self.bottom_y = max(self.bottom_y, wrist_y)

                    moved_up_enough = (self.bottom_y - wrist_y) >= self.up_threshold

                    if moved_up_enough:
                        # Upward return detected — ready for the next downward beat.
                        self.phase = "ready_for_down"
                        self.top_y = wrist_y

            else:
                # Neither rock detected nor within grace — the player dropped out.
                # Cancel this round and go back to WAITING_FOR_ROCK.
                if not within_grace:
                    self._prepare_next_round(now)

            return self._build_output(now)

        # ── SHOOT_WINDOW: watch for the player's throw. ──
        if self.state == "SHOOT_WINDOW":
            time_since_open = now - self.shoot_open_time

            if self._voice_mode:
                # Voice throw is handled externally by inject_voice_throw().
                return self._build_output(now)

            # Wait for the change guard to pass before accepting a new gesture.
            # This prevents the Rock from the final countdown beat from counting.
            if time_since_open >= self.shoot_change_guard_seconds:
                if confirmed_gesture in {"Paper", "Scissors"}:
                    self._resolve_round(confirmed_gesture, now)
                    return self._build_output(now)

                if stable_gesture in {"Paper", "Scissors"}:
                    self._resolve_round(stable_gesture, now)
                    return self._build_output(now)

            # If the player holds Rock past the rock_assume threshold, count it as Rock.
            # This is intentionally slightly later than the Paper/Scissors check so the
            # player has a fair window to form those gestures first.
            if time_since_open >= self.rock_assume_seconds:
                self.gesture_assumed = True
                self._resolve_round("Rock", now)
                return self._build_output(now)

            # Window expired — try one final fallback read from the tracker.
            if now >= self.shoot_close_time:
                fallback_throw = self._fallback_throw(tracker_state)
                if fallback_throw in VALID_GESTURES:
                    self._resolve_round(fallback_throw, now)
                else:
                    # No valid gesture found — replay the round.
                    self._prepare_next_round(now)

            return self._build_output(now)

        # Should never reach here, but return a valid output just in case.
        return self._build_output(now)

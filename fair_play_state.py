# ============================================================
# fair_play_state.py
#
# Purpose:
#   State machine for "Fair Play" mode — a best-of-N match
#   between the player and the robot.
#
# How it works:
#   The main game loop calls FairPlayController.update() every
#   frame, passing the wrist Y position and a gesture tracker
#   dict. The controller advances through a sequence of states
#   (countdown → shoot window → result) and hands back a plain
#   dict of UI values for the renderer to display.
#
#   The robot's move is locked in on beat 3 of the countdown so
#   it can't cheat by watching the player's final gesture.
#   FairPlayAI (fair_play_ai.py) picks that move.
#
# State flow:
#   ROUND_INTRO → WAITING_FOR_ROCK → COUNTDOWN
#     → SHOOT_WINDOW → ROUND_RESULT
#     → MATCH_RESULT (when the match is decided)
#     → loops back to ROUND_INTRO for the next run
# ============================================================

import time
from collections import Counter

from fair_play_ai import FairPlayAI


# The three gestures the game recognises. Anything else is "Unknown".
VALID_GESTURES = {"Rock", "Paper", "Scissors"}

# Maps each gesture to the one that beats it (used for outcome checks).
BEATS = {
    "Rock": "Scissors",
    "Paper": "Rock",
    "Scissors": "Paper",
}


def compare_rps(player_move, robot_move):
    """
    Work out who won a single round from the player's point of view.

    Returns:
      "draw"  — same gesture thrown by both sides
      "win"   — the player's gesture beats the robot's
      "lose"  — the robot's gesture beats the player's
    """
    if player_move == robot_move:
        return "draw"
    # If the thing the player's move beats equals the robot's move, the player wins.
    if BEATS[player_move] == robot_move:
        return "win"
    return "lose"


class FairPlayController:
    """
    Fair Play Mode state machine.

    Rules:
      - First to win `win_target` rounds (default 2) wins the match.
      - The robot locks its move on beat 3 so it can't react to the throw.
      - Draws replay the same round (round number stays the same).

    Physical path:
      Player makes a fist → pumps wrist 1-2-3 → throws on SHOOT.

    Voice path (enabled with set_voice_mode(True)):
      Speaks "ready", "one", "two", "three", then the gesture name.
      inject_voice_beat() / inject_voice_throw() drive those transitions
      instead of wrist-motion detection.
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
        # Hardware/BLE bridge — can be None when running without a physical robot.
        self.robot_output = robot_output

        # Whether the player is using voice instead of wrist pumps.
        self._voice_mode = False

        # AI that chooses the robot's move each round.
        self.ai = ai or FairPlayAI()

        # How many round wins are needed to claim the match.
        self.win_target = win_target

        # Label shown in the UI for this game mode.
        self.play_mode_label = play_mode_label

        # Label shown next to the robot's score.
        self.opponent_label = "ROBOT"

        # --- Wrist-pump detection thresholds ---
        # down_threshold: how far (in normalised Y) the wrist must drop to count as a beat.
        # up_threshold: how far it must rise to signal the end of a beat, ready for the next.
        self.down_threshold = down_threshold
        self.up_threshold = up_threshold

        # Minimum seconds between two registered beats — prevents rapid jitter from
        # accidentally counting as two separate beats.
        self.beat_cooldown = beat_cooldown

        # After the gesture classifier loses "Rock", keep counting beats for this long.
        # Fast pumping can cause the classifier to flicker off for a frame or two.
        self.rock_grace_period = rock_grace_period

        # --- SHOOT window timing ---
        # How long the window stays open for the player to throw.
        self.shoot_window_seconds = shoot_window_seconds

        # Right after the window opens, ignore gestures for this many seconds.
        # This prevents the Rock from the final pump beat from instantly resolving as Rock.
        self.shoot_change_guard_seconds = shoot_change_guard_seconds

        # If the player holds Rock for this long without switching, assume they threw Rock.
        # Slightly longer than the change guard so Paper/Scissors still get a fair window.
        self.rock_assume_seconds = rock_assume_seconds

        # --- Display phase durations ---
        self.round_intro_seconds  = round_intro_seconds
        self.round_result_seconds = round_result_seconds
        self.match_result_seconds = match_result_seconds

        # Kick off the first match straight away.
        self.reset_match()

    def reset(self):
        """
        Public alias for reset_match(). The menu system calls this when the
        player navigates back to the mode selection screen.
        """
        self.reset_match()

    def reset_match(self, now=None):
        """
        Fully reset everything for a brand-new match.
        Clears scores, round history, and all round state, then starts ROUND_INTRO.
        """
        if now is None:
            now = time.monotonic()

        # Tell the AI to forget previous rounds so it starts fresh.
        self.ai.reset()

        # Full history of every round — the AI reads this to predict the next move.
        self.history = []

        self.player_score = 0
        self.robot_score  = 0
        self.round_number = 1

        # Banner text shown on the MATCH_RESULT screen (set when match ends).
        self.match_result_banner = ""
        self.match_until = None

        # Session-level stats that accumulate across all rounds of one match.
        self._session_reaction_times = []   # how fast the player threw each round (ms)
        self._session_gestures       = []   # which gesture the player threw each round

        # Cached for the "last round" replay overlay.
        self._last_round_player_gest = None
        self._last_round_robot_gest  = None

        # NOTE: _last_round_banner is intentionally set here (not just in
        # _reset_round_motion) so the ESC overlay still shows it during play.
        self._last_round_banner = ""

        # Clear per-round motion state and jump to the first state.
        self._reset_round_motion()
        self.state = "ROUND_INTRO"
        self.intro_until = now + self.round_intro_seconds

    def _reset_round_motion(self):
        """
        Reset only the motion-tracking and gesture fields for a new round.
        Called at the start of each round so we don't accidentally carry
        wrist position or gesture data from the previous round.
        """
        # Beat counter and which phase of the pump we're waiting for next.
        self.beat_count = 0
        self.phase = "ready_for_down"   # alternates: "ready_for_down" / "waiting_for_up"

        # Peak and trough wrist Y during the current beat cycle.
        self.top_y    = None
        self.bottom_y = None

        # Timestamps for cooldown and grace-period calculations.
        self.last_beat_time = 0.0
        self.last_rock_time = 0.0

        # Open/close timestamps for the SHOOT window.
        self.shoot_open_time  = None
        self.shoot_close_time = None

        # Robot's chosen move — None until beat 3 locks it in.
        self.robot_locked_move  = None
        self.robot_move_command = "PENDING"

        # What each side threw this round (shown in the result overlay).
        self.player_gesture   = "Unknown"
        self.computer_gesture = "Unknown"

        # Banner text for the round result ("YOU WIN THE ROUND" etc.).
        self.result_banner    = ""
        self.last_round_result = None
        self.result_until      = None

        # One-shot flag: when True, the main loop should flush the gesture tracker
        # once then call consume_tracker_reset_request() to clear this flag.
        self.tracker_reset_requested = False

        # True if Rock was assumed (player didn't explicitly throw it).
        self.gesture_assumed = False

        # Reaction time for this round (ms from shoot-window open to throw).
        self._last_reaction_ms = None

        # Tell the hardware bridge to drop any staged-but-unsent command.
        if self.robot_output is not None:
            self.robot_output.clear_pending_locked()

    def consume_tracker_reset_request(self):
        """
        The main loop calls this after it has flushed the gesture tracker.
        Clears the one-shot flag so we don't keep asking for resets every frame.
        """
        self.tracker_reset_requested = False

    # ------------------------------------------------------------------ #
    # Voice input                                                          #
    # ------------------------------------------------------------------ #

    def set_voice_mode(self, enabled):
        """
        Switch between physical (wrist-pump) and voice input.
        Should be called before a round starts for a clean transition.
        In voice mode the wrist-pump detection is skipped entirely.
        """
        self._voice_mode = bool(enabled)

    def inject_voice_beat(self, word, now=None):
        """
        Advance the countdown from a recognised spoken word.

        Expected spoken sequence: "ready" → "one" → "two" → "three"

        "three" simultaneously locks the robot's move and opens the SHOOT window.
        The window is at least 2.5 seconds long in voice mode so the player
        has enough time to say the gesture name.
        """
        if now is None:
            now = time.monotonic()

        # "ready" is the trigger word that gets us out of the idle wait state.
        if self.state == "WAITING_FOR_ROCK" and word == "ready":
            self.state = "COUNTDOWN"
            self.phase = "ready_for_down"
            self.beat_count = 0
            self.last_beat_time = now
            self.last_rock_time = now
            return

        # All other voice beats only matter during an active countdown.
        if self.state != "COUNTDOWN":
            return

        # Keep the grace timer alive so the countdown doesn't time out between words.
        self.last_rock_time = now

        # Enforce the same cooldown as physical beats to avoid double-counting.
        cooldown_ok = (now - self.last_beat_time) >= self.beat_cooldown

        if word in ("one", "two") and cooldown_ok:
            # "one" and "two" just advance the beat counter.
            self.beat_count += 1
            self.last_beat_time = now

        elif word == "three" and cooldown_ok:
            # "three" is the final beat — lock the robot and open SHOOT immediately.
            self.last_beat_time = now
            self._lock_robot_move()
            self.beat_count = 4   # jump past the normal 4-beat threshold
            self.state = "SHOOT_WINDOW"
            self.shoot_open_time  = now
            # Voice mode gets a bigger minimum window because speaking takes longer than gesturing.
            self.shoot_close_time = now + max(self.shoot_window_seconds, 2.50)
            self.tracker_reset_requested = True

    def inject_voice_throw(self, gesture, now=None):
        """
        Resolve the current round with a spoken gesture name.
        The main loop calls this when the voice recogniser fires during SHOOT_WINDOW.
        """
        if now is None:
            now = time.monotonic()

        # Only accept a valid gesture during an open shoot window.
        if self.state == "SHOOT_WINDOW" and gesture in VALID_GESTURES:
            self._resolve_round(gesture, now)

    # ------------------------------------------------------------------ #
    # Internal state helpers                                               #
    # ------------------------------------------------------------------ #

    def _prepare_next_round(self, now):
        """Clear per-round state and start the intro pause before the next round begins."""
        self._reset_round_motion()
        self.state = "ROUND_INTRO"
        self.intro_until = now + self.round_intro_seconds

    def _lock_robot_move(self):
        """
        Ask the AI to pick the robot's move and commit it for this round.

        The move is locked on beat 3 (before the player throws) so the robot
        can't react to what it sees. Guard prevents locking twice in one round.
        The hardware bridge is notified so the physical arm can start preparing.
        """
        # If already locked (shouldn't happen normally), do nothing.
        if self.robot_locked_move is not None:
            return

        # Let the AI decide based on match history and current round number.
        self.robot_locked_move = self.ai.choose_robot_move(
            history=self.history,
            round_number=self.round_number
        )

        # Build the command string the hardware bridge understands.
        self.robot_move_command = f"ROBOT_PLAY_{self.robot_locked_move.upper()}"

        # Stage the move with the physical robot if one is connected.
        if self.robot_output is not None:
            self.robot_output.stage_locked_move(
                command=self.robot_move_command,
                game_mode="FairPlay",
                metadata={
                    "round_number": self.round_number,
                    "player_score": self.player_score,
                    "robot_score":  self.robot_score,
                }
            )

    def _fallback_throw(self, tracker_state):
        """
        Last-resort gesture lookup when the SHOOT window closes without a clear throw.

        Tries three tracker fields in order of confidence:
          stable_gesture → confirmed_gesture → raw_gesture
        Returns "Unknown" if none of them hold a valid gesture.
        """
        for key in ("stable_gesture", "confirmed_gesture", "raw_gesture"):
            gesture = tracker_state.get(key, "Unknown")
            if gesture in VALID_GESTURES:
                return gesture
        return "Unknown"

    def _resolve_round(self, player_gesture, now):
        """
        Finalise the round: record both gestures, score the outcome, update
        the AI's learning model, and set the result banner.

        Also pushes the result to the hardware bridge and appends the round
        to history so the AI can learn from it next round.
        """
        # Edge case: if the robot never got locked (e.g. countdown was skipped), lock now.
        if self.robot_locked_move is None:
            self._lock_robot_move()

        # Measure how quickly the player threw after the window opened.
        reaction_ms = None
        if self.shoot_open_time is not None:
            reaction_ms = round((now - self.shoot_open_time) * 1000)
            # Only keep plausible values — anything over 5 seconds is probably a fluke.
            if 0 < reaction_ms < 5000:
                self._session_reaction_times.append(reaction_ms)

        # Track which gestures the player used over the whole match for the summary.
        if player_gesture in VALID_GESTURES:
            self._session_gestures.append(player_gesture)

        # Cache both gestures for the replay overlay and the result screen.
        self.player_gesture          = player_gesture
        self.computer_gesture        = self.robot_locked_move
        self._last_round_player_gest = player_gesture
        self._last_round_robot_gest  = self.robot_locked_move

        outcome = compare_rps(self.player_gesture, self.computer_gesture)

        # Set the score change, banner text, and history label based on outcome.
        if outcome == "win":
            self.player_score += 1
            self.result_banner         = "YOU WIN THE ROUND"
            round_result               = "player_win"
            player_outcome_for_history = "win"

        elif outcome == "lose":
            self.robot_score += 1
            self.result_banner         = f"{self.opponent_label} TAKES THE ROUND"
            round_result               = "robot_win"
            player_outcome_for_history = "lose"

        else:
            # Draw — replay the same round, don't change any scores.
            self.result_banner         = "DRAW - THROW AGAIN"
            round_result               = "draw"
            player_outcome_for_history = "draw"

        # Append this round to the match history so the AI can learn from it.
        self.history.append({
            "round_number":   self.round_number,
            "player_gesture": self.player_gesture,
            "robot_gesture":  self.computer_gesture,
            "player_outcome": player_outcome_for_history,
        })

        # Tell the AI's Thompson Sampling bandit which prediction layer was right,
        # so it can up-weight accurate layers in future rounds.
        if hasattr(self.ai, "update_bandit") and hasattr(self.ai, "last_prediction"):
            pred = self.ai.last_prediction or {}
            predicted_player = pred.get("used_predicted_move")
            if predicted_player:
                self.ai.update_bandit(predicted_player, self.player_gesture)

        # Push the round result to the physical robot if connected.
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
                    "robot_score":  self.robot_score,
                    "banner":       self.result_banner,
                }
            )

        # Store state needed for the next frame before transitioning.
        self.last_round_result  = round_result
        self._last_round_banner = self.result_banner
        self._last_reaction_ms  = reaction_ms
        self.state = "ROUND_RESULT"
        self.result_until = now + self.round_result_seconds

    def _round_is_over(self):
        """Return True if either side has reached the win target (match is decided)."""
        return self.player_score >= self.win_target or self.robot_score >= self.win_target

    def _build_output(self, now):
        """
        Assemble the UI data dict for the current frame.

        Every state returns the same base fields. Each state branch then
        adds state-specific keys (main_text, sub_text, time_left).
        The renderer reads this dict directly — no rendering happens here.
        """
        score_text = f"YOU {self.player_score} - {self.opponent_label} {self.robot_score}"
        round_text = f"ROUND {self.round_number}"

        # Pull AI metadata so the UI can show who the player is facing.
        pred        = getattr(self.ai, "last_prediction", None) or {}
        opp_type    = pred.get("opponent_type", "")
        personality = pred.get("personality", getattr(self.ai, "personality", "Normal"))

        # Fields sent to the renderer on every frame regardless of state.
        base = {
            "play_mode_label":  self.play_mode_label,
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
            "player_score":     self.player_score,
            "robot_score":      self.robot_score,
            "request_tracker_reset": self.tracker_reset_requested,
            "gesture_assumed":  self.gesture_assumed,
            "opponent_type":    opp_type,
            "ai_personality":   personality,
            "reaction_ms":      self._last_reaction_ms,
            # Previous round's gestures — the "replay" overlay during WAITING_FOR_ROCK.
            "last_player_gesture": getattr(self, "_last_round_player_gest", None),
            "last_robot_gesture":  getattr(self, "_last_round_robot_gest",  None),
            "last_banner":         getattr(self, "_last_round_banner", ""),
            # Accumulated session data for the end-of-match summary screen.
            "session_reaction_times": list(getattr(self, "_session_reaction_times", [])),
            "session_gestures":       list(getattr(self, "_session_gestures", [])),
        }

        # Each if-block below adds the state-specific text fields.

        if self.state == "ROUND_INTRO":
            base.update({
                "state_label": "Round Intro",
                "main_text":   round_text,
                "sub_text":    f"FIRST TO {self.win_target} | {score_text}",
            })
            return base

        if self.state == "WAITING_FOR_ROCK":
            base.update({
                "state_label": "Waiting",
                # Different prompt depending on input mode.
                "main_text": "VOICE MODE" if self._voice_mode else "MAKE A FIST",
                "sub_text": (
                    "Say READY  then  ONE  TWO  THREE"
                    if self._voice_mode
                    else f"{round_text} | {score_text}"
                ),
            })
            return base

        if self.state == "COUNTDOWN":
            # Show "READY" before the first beat lands, then the beat number.
            main_text = "READY" if self.beat_count == 0 else str(min(self.beat_count, 3))
            base.update({
                "state_label": "Countdown",
                "main_text":   main_text,
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
                "main_text":   "SHOOT!",
                "sub_text": (
                    "Say ROCK, PAPER, or SCISSORS"
                    if self._voice_mode
                    else "Robot already locked its move"
                ),
                # In voice mode there's no real countdown bar — the window stays
                # open until the player speaks, so time_left is always 0.
                "time_left": 0.0 if self._voice_mode else max(0.0, self.shoot_close_time - now),
            })
            return base

        if self.state == "ROUND_RESULT":
            rxn = self._last_reaction_ms
            # Only show reaction time if it looks plausible (under 3 seconds).
            rxn_text = f"Reaction: {rxn}ms" if rxn and rxn < 3000 else ""
            base.update({
                "state_label": "Round Result",
                "main_text":   self.result_banner,
                "sub_text":    rxn_text or score_text,
                "time_left":   max(0.0, self.result_until - now),
            })
            return base

        if self.state == "MATCH_RESULT":
            # Build summary stats for the end-of-match screen.
            rt_list = self._session_reaction_times
            avg_rt  = round(sum(rt_list) / len(rt_list)) if rt_list else None

            gestures  = self._session_gestures
            top_gest  = Counter(gestures).most_common(1)[0][0] if gestures else "?"

            base.update({
                "state_label":    "Match Result",
                "main_text":      self.match_result_banner,
                "sub_text":       f"FINAL SCORE | {score_text}",
                "result_banner":  self.match_result_banner,
                "time_left":      max(0.0, self.match_until - now),
                "session_summary": {
                    "player_won":       self.player_score > self.robot_score,
                    "player_score":     self.player_score,
                    "robot_score":      self.robot_score,
                    "total_rounds":     self.round_number,
                    "win_rate":         self.player_score / max(self.round_number, 1),
                    "avg_reaction_ms":  avg_rt,
                    "top_gesture":      top_gest,
                    "opponent_type":    opp_type,
                },
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

        Returns the UI output dict produced by _build_output().
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

        # ── ROUND_RESULT: hold the result banner, then decide next state. ──
        if self.state == "ROUND_RESULT":
            if now >= self.result_until:
                if self._round_is_over():
                    # Someone reached the win target — show the final banner.
                    self.state = "MATCH_RESULT"
                    self.match_result_banner = (
                        "YOU WIN THE MATCH"
                        if self.player_score > self.robot_score
                        else f"{self.opponent_label} WINS THE MATCH"
                    )
                    self.match_until = now + self.match_result_seconds
                else:
                    # Match still going — increment round unless it was a draw.
                    if self.last_round_result != "draw":
                        self.round_number += 1
                    self._prepare_next_round(now)
            return self._build_output(now)

        # ── MATCH_RESULT: hold the game-over screen, then start a new match. ──
        if self.state == "MATCH_RESULT":
            if now >= self.match_until:
                self.reset_match(now)
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
                self.state     = "COUNTDOWN"
                self.phase     = "ready_for_down"
                self.top_y     = wrist_y
                self.bottom_y  = wrist_y
                self.last_rock_time = now
            return self._build_output(now)

        # ── COUNTDOWN: count wrist pumps (beats) up to 4. ──
        if self.state == "COUNTDOWN":
            if self._voice_mode:
                # Voice countdown is fully handled by inject_voice_beat().
                return self._build_output(now)

            # We keep tracking even if Rock flickers off for a frame during a fast pump.
            # The grace window (rock_grace_period) gives a short forgiveness period.
            rock_detected = (confirmed_rock or stable_rock) and wrist_y is not None
            within_grace  = (now - self.last_rock_time) <= self.rock_grace_period

            # can_track: True if there's an active rock OR we're inside the grace window
            # (and the player has already started pumping — beat_count > 0).
            can_track = rock_detected or (within_grace and wrist_y is not None and self.beat_count > 0)

            if rock_detected:
                # Reset the grace timer while Rock is actively visible.
                self.last_rock_time = now

            if can_track:
                if self.phase == "ready_for_down":
                    # Track the highest wrist Y so we can measure how far it drops.
                    # (Y increases downward in screen coords, so "highest" = smallest Y.)
                    if self.top_y is None:
                        self.top_y = wrist_y
                    self.top_y = min(self.top_y, wrist_y)

                    # A beat fires when the wrist drops far enough AND the cooldown has passed.
                    moved_down_enough = (wrist_y - self.top_y) >= self.down_threshold
                    cooldown_ok       = (now - self.last_beat_time) >= self.beat_cooldown

                    if moved_down_enough and cooldown_ok:
                        # Downward pump counted — advance the beat counter.
                        self.beat_count    += 1
                        self.last_beat_time = now
                        self.phase          = "waiting_for_up"
                        self.bottom_y       = wrist_y

                        if self.beat_count >= 3:
                            # Beat 3: lock the robot's move now so it's committed
                            # before the player forms their throw gesture.
                            self._lock_robot_move()

                        if self.beat_count >= 4:
                            # Beat 4: open the SHOOT window.
                            self.state            = "SHOOT_WINDOW"
                            self.shoot_open_time  = now
                            self.shoot_close_time = now + self.shoot_window_seconds
                            # Ask the main loop to flush the gesture tracker so the
                            # Rock from the final beat doesn't leak into SHOOT detection.
                            self.tracker_reset_requested = True

                elif self.phase == "waiting_for_up":
                    # Track the lowest wrist Y (the bottom of the current pump).
                    if self.bottom_y is None:
                        self.bottom_y = wrist_y
                    self.bottom_y = max(self.bottom_y, wrist_y)

                    # Wait until the wrist rises far enough back up.
                    moved_up_enough = (self.bottom_y - wrist_y) >= self.up_threshold

                    if moved_up_enough:
                        # Upward return confirmed — ready to count the next downward beat.
                        self.phase = "ready_for_down"
                        self.top_y = wrist_y

            else:
                # Rock disappeared AND the grace period has expired — player dropped out.
                # Reset the round so they can start over from WAITING_FOR_ROCK.
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
            # opens. This stops the Rock from the final pump beat from instantly resolving.
            if time_since_open >= self.shoot_change_guard_seconds:
                # Paper or Scissors thrown — resolve immediately.
                if confirmed_gesture in {"Paper", "Scissors"}:
                    self._resolve_round(confirmed_gesture, now)
                    return self._build_output(now)
                if stable_gesture in {"Paper", "Scissors"}:
                    self._resolve_round(stable_gesture, now)
                    return self._build_output(now)

            # Rock assumption: if the player hasn't changed gesture by this point,
            # count it as Rock. The threshold is slightly larger than shoot_change_guard
            # so Paper/Scissors still get a fair window to be recognised first.
            if time_since_open >= self.rock_assume_seconds:
                self.gesture_assumed = True
                self._resolve_round("Rock", now)
                return self._build_output(now)

            # Window expired without any of the above triggering — do one last
            # grab from the tracker and resolve whatever we find.
            if now >= self.shoot_close_time:
                fallback = self._fallback_throw(tracker_state)
                if fallback in VALID_GESTURES:
                    self._resolve_round(fallback, now)
                else:
                    # Nothing usable — replay the round.
                    self._prepare_next_round(now)

            return self._build_output(now)

        # Should never reach here, but always return a valid dict.
        return self._build_output(now)

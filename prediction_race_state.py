"""
prediction_race_state.py
========================
Prediction Race Mode.

The AI shows its prediction BEFORE the round.  The player wins by throwing
anything OTHER than what was predicted.  Uses identical pump/beat/shoot
mechanics to FairPlay — no custom detection code.

Implemented as a thin wrapper around FairPlayController: we inherit all
the beat detection, shoot window, and state machine, then override just
the round resolution to apply Prediction Race scoring.

Where this fits in the codebase:
  - Inherits from fair_play_state.FairPlayController
  - Uses fair_play_ai.FairPlayAI for predictions
  - Renderer calls draw_prediction_race_view() with the dict from _build_output()
  - Main loop calls update() every frame and confirm_match_end() on Enter
"""

import random
from fair_play_state import FairPlayController
from fair_play_ai import FairPlayAI, VALID_GESTURES, COUNTER_MOVE

# First to WIN_TARGET round wins wins the match
WIN_TARGET = 5


class PredictionRaceController(FairPlayController):
    """
    Prediction Race wraps FairPlayController exactly.

    Everything (pump detection, shoot window, state machine, tracker
    resets, config params) is inherited unchanged.  The only difference
    is _resolve_round: instead of comparing player vs robot, we compare
    player vs AI's prediction.

    Bluff mechanic:
      At beat 2 of the countdown, there is a 25% chance the AI switches
      to a fake (decoy) prediction to try to trick the player.  The bluff
      is locked in once applied — it won't change again within the round.
    """

    def __init__(self, robot_output=None, ai=None, **kwargs):
        super().__init__(
            robot_output=robot_output,
            ai=ai or FairPlayAI(difficulty="Normal"),
            win_target=WIN_TARGET,
            play_mode_label="Prediction Race",
            **kwargs,
        )
        self.opponent_label = "AI"
        self._last_insight  = ""       # explanatory text shown after each round
        self._ai_prediction = None     # the prediction currently displayed on screen

    def reset(self):
        """Reset all match-level state including the prediction fields."""
        super().reset()
        self._last_insight       = ""
        self._ai_prediction      = None
        self._result_prediction  = None    # locked-in prediction at SHOOT time
        self._bluffed_this_round = False   # whether the AI has already bluffed this round
        # Ensure match_until is never None — parent's _build_output uses it
        if self.match_until is None:
            self.match_until = 0.0

    # ── Live prediction refresh ───────────────────────────────────────────────

    def update(self, wrist_y, tracker_state, now=None):
        """
        Override of FairPlayController.update().

        We do three things on top of the parent:
          1. Clear the locked-in prediction and bluff flag at round start
          2. Update the live displayed prediction every frame
          3. Handle the MATCH_RESULT and SHOOT_WINDOW states ourselves so we
             can use Prediction Race scoring instead of FairPlay scoring
        """
        import time as _time
        if now is None:
            now = _time.monotonic()

        # When a new round begins, clear the previous result so the
        # live prediction starts updating again from scratch
        if self.state == "WAITING_FOR_ROCK":
            self._result_prediction  = None
            self._bluffed_this_round = False

        # Update the live prediction during the lead-up and countdown phases.
        # We do this every frame so the display is always current.
        if self.state in ("WAITING_FOR_ROCK", "COUNTDOWN", "SHOOT_WINDOW"):
            if self.history:
                # Ask the AI what it thinks the player will throw next
                scores = self.ai._predict_player_scores(self.history)
                best = max(scores, key=scores.get)

                # Bluff: at beat 2 exactly, 25% chance to flip to a decoy prediction.
                # Once flipped, we don't change it again within this round.
                if (self.state == "COUNTDOWN"
                        and self.beat_count == 2
                        and not getattr(self, "_bluffed_this_round", False)
                        and random.random() < 0.25):
                    others = [g for g in VALID_GESTURES if g != best]
                    self._ai_prediction      = random.choice(others)
                    self._bluffed_this_round = True
                elif not getattr(self, "_bluffed_this_round", False):
                    # No bluff yet — show the genuine best prediction
                    self._ai_prediction = best
                # If we already bluffed this round, leave the display unchanged
            elif self._ai_prediction is None:
                # No history yet (early rounds) — show a random gesture so
                # the display isn't just blank
                self._ai_prediction = random.choice(list(VALID_GESTURES))

        # Safety: match_until must be a float before calling the parent
        if self.match_until is None:
            self.match_until = 0.0

        # Handle MATCH_RESULT ourselves — don't let the parent auto-reset on
        # a timer.  We wait for the player to press Enter instead.
        if self.state == "MATCH_RESULT":
            return self._build_output(now)

        # Handle SHOOT_WINDOW ourselves so we can use Prediction Race scoring.
        # We accept stable_gesture immediately once the guard period has passed,
        # giving the tracker as much time as possible to re-confirm after reset.
        if self.state == "SHOOT_WINDOW":
            time_since_open = now - self.shoot_open_time

            if time_since_open >= self.shoot_change_guard_seconds:
                confirmed = tracker_state.get("confirmed_gesture", "Unknown")
                stable    = tracker_state.get("stable_gesture", "Unknown")

                # Try confirmed first (higher confidence), then stable
                throw = None
                if confirmed in VALID_GESTURES:
                    throw = confirmed
                elif stable in VALID_GESTURES:
                    throw = stable

                if throw:
                    self._resolve_round(throw, now)
                    return self._build_output(now)

            # Backstop: if the player takes too long, assume Rock
            if time_since_open >= self.rock_assume_seconds:
                self._resolve_round("Rock", now)
                return self._build_output(now)

            return self._build_output(now)

        # All other states delegate to the parent FairPlayController
        return super().update(wrist_y=wrist_y, tracker_state=tracker_state, now=now)

    # ── Round resolution with Prediction Race rules ───────────────────────────

    def _resolve_round(self, player_gesture, now):
        """
        Prediction Race scoring rules:

        - Whatever prediction was shown on screen at SHOOT is the contract.
        - If the player threw it  → AI wins (it predicted correctly, or its
          bluff fooled the player into throwing what it wanted).
        - If the player threw anything else → player wins (they fooled the AI).

        We use the displayed prediction directly — no recomputation at
        resolution time — so the player's experience matches what they saw.
        """
        import time as _time
        if now is None:
            now = _time.monotonic()

        # The prediction the player was trying to avoid
        displayed = self._ai_prediction or random.choice(list(VALID_GESTURES))
        bluffed   = getattr(self, "_bluffed_this_round", False)

        if player_gesture == displayed:
            # Player threw exactly what was shown — AI wins this round
            if bluffed:
                self.result_banner = "BLUFF WORKED!"
                self._last_insight = (
                    f"AI bluffed {displayed} and you fell for it. Tricked!"
                )
            else:
                self.result_banner = "PREDICTED!"
                self._last_insight = f"AI predicted {displayed}. You threw it."
            self.robot_score += 1
        else:
            # Player threw something different — player wins this round
            self.result_banner = "FOOLED IT!"
            if bluffed:
                self._last_insight = (
                    f"AI bluffed {displayed}, you threw {player_gesture}. Saw through it!"
                )
            else:
                self._last_insight = (
                    f"AI predicted {displayed}, you threw {player_gesture}. Fooled!"
                )
            self.player_score += 1

        self.player_gesture   = player_gesture
        self.computer_gesture = displayed

        # Determine the outcome label for the history log
        player_outcome = "lose" if player_gesture == displayed else "win"
        self.history.append({
            "round_number":   self.round_number,
            "player_gesture": player_gesture,
            "robot_gesture":  displayed,
            "player_outcome": player_outcome,
        })

        # Update the bandit model if the AI supports it
        if hasattr(self.ai, "update_bandit") and hasattr(self.ai, "last_prediction"):
            pred = self.ai.last_prediction or {}
            pm   = pred.get("used_predicted_move")
            if pm:
                self.ai.update_bandit(pm, player_gesture)

        # Lock in the prediction for the result screen, then clear the live one
        self._result_prediction  = displayed
        self._ai_prediction      = None

        self.state        = "ROUND_RESULT"
        self.result_until = now + self.round_result_seconds

        # Check if either side has reached the win target
        if self.player_score >= WIN_TARGET or self.robot_score >= WIN_TARGET:
            self.state               = "MATCH_RESULT"
            self.match_result_banner = (
                "YOU WIN THE MATCH!" if self.player_score >= WIN_TARGET
                else "AI WINS THE MATCH!"
            )
            self.match_until  = now + 3.0
            self.result_until = self.match_until

    # ── Output dict ──────────────────────────────────────────────────────────

    def _build_output(self, now):
        """
        Extend the parent's output dict with Prediction Race-specific fields.

        During result states we show the locked-in prediction (what was
        actually on screen at SHOOT); during active play we show the live
        updating prediction.
        """
        base = super()._build_output(now)

        # Switch between the live prediction and the locked-in one
        display_prediction = (
            self._result_prediction
            if self.state in ("ROUND_RESULT", "MATCH_RESULT")
            else self._ai_prediction
        ) or ""

        base["ai_prediction"]    = display_prediction
        base["last_insight"]     = self._last_insight
        base["win_target"]       = WIN_TARGET
        base["player_score"]     = self.player_score
        base["ai_score"]         = self.robot_score
        base["score_text"]       = f"YOU {self.player_score}  -  AI {self.robot_score}"
        base["play_mode_label"]  = "Prediction Race"
        # Renderer uses this flag to know when to prompt for Enter
        base["waiting_for_enter"] = (self.state == "MATCH_RESULT")
        return base

    def confirm_match_end(self):
        """
        Called by the main loop when the player presses Enter on the match
        result screen.  Resets the match so a new one can begin.
        """
        if self.state == "MATCH_RESULT":
            self.reset_match()

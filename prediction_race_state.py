"""
prediction_race_state.py
========================
Prediction Race Mode.

The AI shows its prediction BEFORE each round. The player wins by throwing
anything OTHER than what was predicted. If you throw what it predicted,
the AI wins.

There's also a bluff mechanic: at beat 2 of the countdown, the AI has a
25% chance to switch to a fake "decoy" prediction to try to trick the player.
Once a bluff is applied it stays locked for that round.

This mode is a thin wrapper around FairPlayController: we inherit all the
beat detection, shoot window, and state machine, then override just two
methods — update() and _resolve_round() — to apply Prediction Race scoring.

How it fits into the project:
  - Inherits from fair_play_state.FairPlayController.
  - Uses fair_play_ai.FairPlayAI for predictions.
  - Renderer calls draw_prediction_race_view() with the dict from _build_output().
  - Main loop calls update() every frame and confirm_match_end() on Enter.
"""

import time
import random
from fair_play_state import FairPlayController
from fair_play_ai import FairPlayAI, VALID_GESTURES, COUNTER_MOVE

# First to this many round wins takes the match
WIN_TARGET = 5


class PredictionRaceController(FairPlayController):
    """
    Prediction Race wraps FairPlayController almost entirely.

    All pump detection, shoot window, and state machine logic is inherited
    unchanged. The only differences are:
      - update():        refreshes the live prediction display and handles bluffing
      - _resolve_round(): applies Prediction Race scoring instead of FairPlay scoring
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
        self._last_insight  = ""    # explanatory text shown after each round result
        self._ai_prediction = None  # the prediction currently displayed on screen

    def reset(self):
        """Reset all match-level state including prediction fields."""
        super().reset()
        self._last_insight       = ""
        self._ai_prediction      = None
        self._result_prediction  = None    # prediction locked in at SHOOT time
        self._bluffed_this_round = False   # whether the AI has already bluffed this round

        # The parent's _build_output() reads match_until — make sure it's always a float
        if self.match_until is None:
            self.match_until = 0.0

    # ── Override: add live prediction logic on top of the parent's update() ───

    def update(self, wrist_y, tracker_state, now=None):
        """
        Override of FairPlayController.update().

        We add three things on top of the parent:
          1. Clear the locked-in prediction and bluff flag at the start of each round.
          2. Refresh the live displayed prediction every frame during lead-up states.
          3. Handle MATCH_RESULT and SHOOT_WINDOW ourselves to apply Prediction Race
             scoring instead of FairPlay scoring.
        """
        if now is None:
            now = time.monotonic()

        # At the start of a new round, clear leftover prediction state from last round
        if self.state == "WAITING_FOR_ROCK":
            self._result_prediction  = None
            self._bluffed_this_round = False

        # Keep the displayed prediction current during all lead-up phases
        if self.state in ("WAITING_FOR_ROCK", "COUNTDOWN", "SHOOT_WINDOW"):
            if self.history:
                # Ask the AI for its confidence scores across all gestures
                scores = self.ai._predict_player_scores(self.history)
                best   = max(scores, key=scores.get)  # most likely gesture the player will throw

                # Bluff: at exactly beat 2, 25% chance to flip to a decoy prediction.
                # Once bluffed, we don't change the display again for the rest of the round.
                if (self.state == "COUNTDOWN"
                        and self.beat_count == 2
                        and not self._bluffed_this_round
                        and random.random() < 0.25):
                    # Pick any gesture that isn't the genuine prediction
                    others = [g for g in VALID_GESTURES if g != best]
                    self._ai_prediction      = random.choice(others)
                    self._bluffed_this_round = True
                elif not self._bluffed_this_round:
                    # No bluff applied yet — show the genuine best prediction
                    self._ai_prediction = best
                # If already bluffed this round, leave the display unchanged

            elif self._ai_prediction is None:
                # No history yet (first few rounds) — show a random gesture so
                # the display isn't blank
                self._ai_prediction = random.choice(list(VALID_GESTURES))

        # Safety: match_until must be a float before the parent can use it
        if self.match_until is None:
            self.match_until = 0.0

        # Handle MATCH_RESULT ourselves — we wait for Enter instead of auto-resetting
        if self.state == "MATCH_RESULT":
            return self._build_output(now)

        # Handle SHOOT_WINDOW ourselves — use Prediction Race scoring, not FairPlay
        if self.state == "SHOOT_WINDOW":
            time_since_open = now - self.shoot_open_time

            if time_since_open >= self.shoot_change_guard_seconds:
                confirmed = tracker_state.get("confirmed_gesture", "Unknown")
                stable    = tracker_state.get("stable_gesture",    "Unknown")

                # Prefer confirmed (higher confidence), fall back to stable
                throw = None
                if confirmed in VALID_GESTURES:
                    throw = confirmed
                elif stable in VALID_GESTURES:
                    throw = stable

                if throw:
                    self._resolve_round(throw, now)
                    return self._build_output(now)

            # Backstop: if the player takes too long, assume they threw Rock
            if time_since_open >= self.rock_assume_seconds:
                self._resolve_round("Rock", now)
                return self._build_output(now)

            return self._build_output(now)

        # All other states (WAITING_FOR_ROCK, COUNTDOWN, etc.) — delegate to parent
        return super().update(wrist_y=wrist_y, tracker_state=tracker_state, now=now)

    # ── Override: Prediction Race round scoring ───────────────────────────────

    def _resolve_round(self, player_gesture, now):
        """
        Prediction Race scoring rules:
          - The prediction shown on screen at SHOOT is the contract.
          - If the player threw it → AI wins (it predicted correctly, or bluffed
            them into throwing the predicted gesture anyway).
          - If the player threw anything else → player wins (they fooled the AI).

        We use the displayed prediction directly — no recomputation — so the
        outcome always matches what the player actually saw on screen.
        """
        if now is None:
            now = time.monotonic()

        # Whatever was shown on screen is what we score against
        displayed = self._ai_prediction or random.choice(list(VALID_GESTURES))

        if player_gesture == displayed:
            # Player threw exactly what was shown — AI wins this round
            if self._bluffed_this_round:
                self.result_banner = "BLUFF WORKED!"
                self._last_insight = f"AI bluffed {displayed} and you fell for it. Tricked!"
            else:
                self.result_banner = "PREDICTED!"
                self._last_insight = f"AI predicted {displayed}. You threw it."
            self.robot_score += 1
        else:
            # Player threw something different — player wins this round
            self.result_banner = "FOOLED IT!"
            if self._bluffed_this_round:
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

        # "lose" from the player's perspective means the AI's prediction was correct
        player_outcome = "lose" if player_gesture == displayed else "win"
        self.history.append({
            "round_number":   self.round_number,
            "player_gesture": player_gesture,
            "robot_gesture":  displayed,
            "player_outcome": player_outcome,
        })

        # Update the AI's internal bandit model if it supports it
        if hasattr(self.ai, "update_bandit") and hasattr(self.ai, "last_prediction"):
            pred = self.ai.last_prediction or {}
            pm   = pred.get("used_predicted_move")
            if pm:
                self.ai.update_bandit(pm, player_gesture)

        # Lock in the prediction for the result screen, then clear the live display
        self._result_prediction = displayed
        self._ai_prediction     = None

        self.state        = "ROUND_RESULT"
        self.result_until = now + self.round_result_seconds

        # Check if either side has now reached the win target
        if self.player_score >= WIN_TARGET or self.robot_score >= WIN_TARGET:
            self.state               = "MATCH_RESULT"
            self.match_result_banner = (
                "YOU WIN THE MATCH!" if self.player_score >= WIN_TARGET
                else "AI WINS THE MATCH!"
            )
            self.match_until  = now + 3.0
            self.result_until = self.match_until  # keep result displayed until match screen ends

    # ── Override: extend the parent's output dict ─────────────────────────────

    def _build_output(self, now):
        """
        Add Prediction Race-specific fields to the parent's output dict.

        During result states we show the locked-in prediction (what was
        actually on screen when SHOOT opened). During play we show the
        live updating prediction.
        """
        base = super()._build_output(now)

        # Show the locked-in prediction during result screens, live otherwise
        display_prediction = (
            self._result_prediction
            if self.state in ("ROUND_RESULT", "MATCH_RESULT")
            else self._ai_prediction
        ) or ""

        base["ai_prediction"]     = display_prediction
        base["last_insight"]      = self._last_insight
        base["win_target"]        = WIN_TARGET
        base["player_score"]      = self.player_score
        base["ai_score"]          = self.robot_score
        base["score_text"]        = f"YOU {self.player_score}  -  AI {self.robot_score}"
        base["play_mode_label"]   = "Prediction Race"
        # The renderer uses this flag to know when to prompt the player to press Enter
        base["waiting_for_enter"] = (self.state == "MATCH_RESULT")
        return base

    def confirm_match_end(self):
        """
        Called by the main loop when the player presses Enter on the match
        result screen. Resets everything so a new match can begin.
        """
        if self.state == "MATCH_RESULT":
            self.reset_match()

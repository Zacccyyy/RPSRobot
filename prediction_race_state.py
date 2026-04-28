"""
prediction_race_state.py
========================
Prediction Race Mode.

The AI shows its prediction BEFORE the round. The player wins by throwing
anything OTHER than what was predicted. Uses identical pump/beat/shoot
mechanics to FairPlay — no custom detection code.

Implemented as a thin wrapper around FairPlayController: we inherit all
the beat detection, shoot window, and state machine, then override just
the round resolution to apply Prediction Race scoring.
"""

import random
from fair_play_state import FairPlayController
from fair_play_ai import FairPlayAI, VALID_GESTURES, COUNTER_MOVE

WIN_TARGET = 5


class PredictionRaceController(FairPlayController):
    """
    Prediction Race wraps FairPlayController exactly.

    Everything (pump detection, shoot window, state machine, tracker
    resets, config params) is inherited unchanged. The only difference
    is _resolve_round: instead of comparing player vs robot, we compare
    player vs AI's prediction.
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
        self._last_insight  = ""
        self._ai_prediction = None   # live prediction shown during countdown

    def reset(self):
        super().reset()
        self._last_insight       = ""
        self._ai_prediction      = None
        self._result_prediction  = None
        self._bluffed_this_round = False
        # Ensure match_until is never None — parent's _build_output uses it
        if self.match_until is None:
            self.match_until = 0.0

    # ── Refresh prediction on each beat ────────────────────────────────────
    def update(self, wrist_y, tracker_state, now=None):
        import time as _time
        if now is None:
            now = _time.monotonic()

        # Clear result prediction and bluff flag when a new round begins
        if self.state == "WAITING_FOR_ROCK":
            self._result_prediction  = None
            self._bluffed_this_round = False

        # Update live prediction every frame from WAITING_FOR_ROCK onward.
        # Uses the full history so it genuinely adapts round by round.
        # During early rounds with no history: randomly cycle each beat so
        # the displayed prediction isn't static.
        if self.state in ("WAITING_FOR_ROCK", "COUNTDOWN", "SHOOT_WINDOW"):
            if self.history:
                scores = self.ai._predict_player_scores(self.history)
                best = max(scores, key=scores.get)
                # Bluff: at beat 2 exactly, 25% chance to show a decoy
                # Only flips once per round — locked in after beat 2
                if (self.state == "COUNTDOWN"
                        and self.beat_count == 2
                        and not getattr(self, "_bluffed_this_round", False)
                        and random.random() < 0.25):
                    others = [g for g in VALID_GESTURES if g != best]
                    self._ai_prediction = random.choice(others)
                    self._bluffed_this_round = True
                elif not getattr(self, "_bluffed_this_round", False):
                    self._ai_prediction = best
                # If already bluffed this round, leave the prediction as-is
            elif self._ai_prediction is None:
                self._ai_prediction = random.choice(list(VALID_GESTURES))

        # Ensure match_until is always a float before calling parent
        if self.match_until is None:
            self.match_until = 0.0

        # Intercept MATCH_RESULT — don't let parent auto-reset on timer.
        # We wait for Enter (confirm_match_end) instead.
        if self.state == "MATCH_RESULT":
            return self._build_output(now)

        # Override SHOOT_WINDOW: after the guard, accept stable_gesture
        # immediately (don't require confirmed_gesture) since we need all
        # the time we can give the tracker to re-confirm after reset.
        if self.state == "SHOOT_WINDOW":
            import time as _t
            if now is None:
                now = _t.monotonic()
            time_since_open = now - self.shoot_open_time
            if time_since_open >= self.shoot_change_guard_seconds:
                confirmed = tracker_state.get("confirmed_gesture", "Unknown")
                stable    = tracker_state.get("stable_gesture", "Unknown")
                # Accept confirmed first, then stable — both valid
                throw = None
                if confirmed in VALID_GESTURES:
                    throw = confirmed
                elif stable in VALID_GESTURES:
                    throw = stable
                if throw:
                    self._resolve_round(throw, now)
                    return self._build_output(now)
            # Rock assumption still applies as a backstop
            if time_since_open >= self.rock_assume_seconds:
                self._resolve_round("Rock", now)
                return self._build_output(now)
            return self._build_output(now)

        return super().update(wrist_y=wrist_y, tracker_state=tracker_state, now=now)

    # ── Override resolution — prediction race scoring ──────────────────────
    def _resolve_round(self, player_gesture, now):
        """
        In Prediction Race the rule is simple:
        - Whatever was shown on screen at SHOOT is what the player must avoid.
        - If the player threw it = AI wins (predicted correctly or bluff succeeded).
        - If the player threw something else = player wins.
        No recomputation at resolution — the displayed prediction is the contract.
        """
        import time as _time
        if now is None:
            now = _time.monotonic()

        # The displayed prediction is the one the player was trying to avoid.
        # Use it directly — no recomputation.
        displayed = self._ai_prediction or random.choice(list(VALID_GESTURES))
        bluffed   = getattr(self, "_bluffed_this_round", False)

        if player_gesture == displayed:
            # Player threw exactly what was shown — AI wins
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
            # Player threw something different — player wins
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

        player_outcome = "lose" if player_gesture == displayed else "win"
        self.history.append({
            "round_number":  self.round_number,
            "player_gesture": player_gesture,
            "robot_gesture":  displayed,
            "player_outcome": player_outcome,
        })

        if hasattr(self.ai, "update_bandit") and hasattr(self.ai, "last_prediction"):
            pred = self.ai.last_prediction or {}
            pm   = pred.get("used_predicted_move")
            if pm:
                self.ai.update_bandit(pm, player_gesture)

        self._result_prediction  = displayed
        self._ai_prediction      = None

        self.state        = "ROUND_RESULT"
        self.result_until = now + self.round_result_seconds

        if self.player_score >= WIN_TARGET or self.robot_score >= WIN_TARGET:
            self.state               = "MATCH_RESULT"
            self.match_result_banner = (
                "YOU WIN THE MATCH!" if self.player_score >= WIN_TARGET
                else "AI WINS THE MATCH!"
            )
            self.match_until  = now + 3.0
            self.result_until = self.match_until

    # ── Expose extra fields for renderer ──────────────────────────────────
    def _build_output(self, now):
        base = super()._build_output(now)
        display_prediction = (
            self._result_prediction
            if self.state in ("ROUND_RESULT", "MATCH_RESULT")
            else self._ai_prediction
        ) or ""
        base["ai_prediction"]   = display_prediction
        base["last_insight"]    = self._last_insight
        base["win_target"]      = WIN_TARGET
        base["player_score"]    = self.player_score
        base["ai_score"]        = self.robot_score
        base["score_text"]      = f"YOU {self.player_score}  -  AI {self.robot_score}"
        base["play_mode_label"] = "Prediction Race"
        base["waiting_for_enter"] = (self.state == "MATCH_RESULT")
        return base

    def confirm_match_end(self):
        """Called when player presses Enter on the match result screen."""
        if self.state == "MATCH_RESULT":
            self.reset_match()

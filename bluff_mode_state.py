"""
bluff_mode_state.py
===================
Bluff Mode — AI declares its intended move before every round, then plays.

The AI declares STRATEGICALLY (Option B):
  - Default 60% of the time the AI bluffs:
    it declares the move it wants the player to COUNTER, then plays
    the move that BEATS that counter.
    Example: AI will play Rock. It declares "Scissors" — hoping the
    player throws Paper (to beat Scissors), which Rock then beats.
  - 40% of the time the AI tells the truth.

Research angle: does prior knowledge of opponent's declared move
change player behaviour? Each round logs:
  - declared move, actual AI move, player move, outcome
  - whether the declaration was a bluff or truth
  - whether the player "followed" the declaration (played to beat it)

The bluff_rate can be tuned at construction time.

The game uses the existing pump-beat countdown (3 beats → SHOOT).
Best-of-5 (first to 3).
"""

import time
import random
import csv
import os

from fair_play_ai import FairPlayAI

VALID_GESTURES  = ("Rock", "Paper", "Scissors")
VALID_SET       = frozenset(VALID_GESTURES)

BEATS = {
    "Rock":     "Scissors",
    "Paper":    "Rock",
    "Scissors": "Paper",
}
COUNTER = {v: k for k, v in BEATS.items()}   # what beats each move

INTRO_SECS        = 1.20
ROUND_RESULT_SECS = 2.60
MATCH_RESULT_SECS = 2.40
SHOOT_WINDOW      = 0.90
BEAT_COOLDOWN     = 0.25   # longer than FairPlay to prevent double-counting
DOWN_THRESHOLD    = 0.060  # requires more deliberate downstroke than FairPlay's 0.045
UP_THRESHOLD      = 0.045  # requires clear upstroke recovery
ROCK_GRACE        = 0.50
DECLARATION_SECS  = 1.20   # how long to show declaration before countdown


def compare_rps(p, c):
    if p == c:
        return "draw"
    if BEATS.get(p) == c:
        return "win"
    return "lose"


class BluffModeController:
    """
    Bluff Mode controller.

    update() signature:
        controller.update(tracker_state=..., wrist_y=..., now=...)
    """

    def __init__(self, ai=None, win_target=3, bluff_rate=0.60,
                 beat_cooldown=BEAT_COOLDOWN,
                 shoot_window_seconds=SHOOT_WINDOW):
        self.ai              = ai or FairPlayAI()
        self.win_target      = win_target
        self.bluff_rate      = bluff_rate
        self.BEAT_COOLDOWN   = beat_cooldown
        self.SHOOT_WINDOW    = shoot_window_seconds
        self._log: list[dict] = []
        self.reset_match()

    def reset(self):
        self.reset_match()

    def reset_match(self, now=None):
        if now is None:
            now = time.monotonic()
        self.player_score = 0
        self.robot_score  = 0
        self.round_number = 1
        self._history: list[dict] = []
        self.ai.reset()
        self._reset_round(now)

    def _reset_round(self, now=None):
        if now is None:
            now = time.monotonic()
        self.state          = "ROUND_INTRO"
        self.intro_until    = now + INTRO_SECS
        self.beat_count     = 0
        self.phase          = "ready_for_down"
        self.top_y          = None
        self.bottom_y       = None
        self.last_beat_time = 0.0
        self.last_rock_time = 0.0
        self.shoot_open_time  = None
        self.shoot_close_time = None
        self.tracker_reset_requested = False

        # Bluff decision
        self._ai_actual      = None
        self._ai_declared    = None
        self._is_bluff       = False
        self._declaration_until = None

        self.player_gesture  = "Unknown"
        self.result_banner   = ""
        self.last_round_result = None
        self.result_until    = None
        self.match_until     = None

    def _plan_declaration(self):
        """
        Decide the AI's actual move (via FairPlayAI), then decide whether
        to bluff and compute the declared move.
        """
        actual = self.ai.choose_robot_move(
            history=self._history,
            round_number=self.round_number,
        )
        if actual not in VALID_SET:
            actual = random.choice(VALID_GESTURES)

        self._ai_actual   = actual
        self._is_bluff    = (random.random() < self.bluff_rate)

        if self._is_bluff:
            # Declare the move the player would need to counter actual,
            # i.e. declare BEATS[actual] — so player throws counter of BEATS[actual],
            # but AI plays actual which beats that.
            # Example: actual=Rock → declare Scissors (player throws Paper to beat Scissors → Rock beats Paper)
            self._ai_declared = BEATS[actual]
        else:
            self._ai_declared = actual

    def _update_beat(self, wrist_y, confirmed, stable, now):
        """Exact pump logic from FairPlayController."""
        confirmed_rock = confirmed == "Rock"
        stable_rock    = stable    == "Rock"
        rock_detected  = (confirmed_rock or stable_rock) and wrist_y is not None
        within_grace   = (now - self.last_rock_time) <= ROCK_GRACE
        can_track      = rock_detected or (within_grace and wrist_y is not None
                                           and self.beat_count > 0)

        if rock_detected:
            self.last_rock_time = now

        if not can_track:
            # Grace expired with no rock — reset countdown (match FairPlay)
            if not within_grace and self.beat_count > 0:
                self.beat_count = 0
                self.phase      = "ready_for_down"
                self.top_y      = None
                self.bottom_y   = None
            return

        cooldown_ok = (now - self.last_beat_time) >= self.BEAT_COOLDOWN

        if self.phase == "ready_for_down":
            if self.top_y is None:
                self.top_y = wrist_y
            self.top_y = min(self.top_y, wrist_y)
            if (wrist_y - self.top_y) >= DOWN_THRESHOLD and cooldown_ok:
                self.beat_count    += 1
                self.last_beat_time = now
                self.phase          = "waiting_for_up"
                self.bottom_y       = wrist_y

        elif self.phase == "waiting_for_up":
            if self.bottom_y is None:
                self.bottom_y = wrist_y
            self.bottom_y = max(self.bottom_y, wrist_y)
            if (self.bottom_y - wrist_y) >= UP_THRESHOLD:
                self.phase = "ready_for_down"
                self.top_y = wrist_y

    def _resolve_round(self, player_g, now):
        self.player_gesture = player_g
        outcome = compare_rps(player_g, self._ai_actual)

        if outcome == "win":
            self.player_score += 1
            self.result_banner = "YOU WIN THE ROUND"
        elif outcome == "lose":
            self.robot_score  += 1
            self.result_banner = "AI WINS THE ROUND"
        else:
            self.result_banner = "DRAW"

        self.last_round_result = outcome

        # Research log entry
        player_followed = (player_g == COUNTER.get(self._ai_declared, ""))
        self._log.append({
            "round":           self.round_number,
            "ai_declared":     self._ai_declared,
            "ai_actual":       self._ai_actual,
            "is_bluff":        self._is_bluff,
            "player_move":     player_g,
            "outcome":         outcome,
            "player_followed_declaration": player_followed,
        })

        self._history.append({
            "round_number":   self.round_number,
            "player_gesture": player_g,
            "robot_gesture":  self._ai_actual,
            "player_outcome": outcome,
        })

        # Update bandit weights if the AI supports it
        if hasattr(self.ai, "update_bandit") and hasattr(self.ai, "last_prediction"):
            pred = self.ai.last_prediction or {}
            predicted_player = pred.get("used_predicted_move")
            if predicted_player:
                self.ai.update_bandit(predicted_player, player_g)

        self.state        = "ROUND_RESULT"
        self.result_until = now + ROUND_RESULT_SECS

    def _round_is_over(self):
        return (self.player_score >= self.win_target or
                self.robot_score  >= self.win_target)

    def _build_output(self, now):
        tl = max(0.0, self.shoot_close_time - now) if self.shoot_close_time else 0.0
        return {
            "play_mode_label":   "Bluff Mode",
            "state":             self.state,
            "beat_count":        self.beat_count,
            "time_left":         tl,
            "player_gesture":    self.player_gesture,
            # Keys expected by the renderer
            "ai_declared_move":  self._ai_declared or "",
            "ai_actual_move":    self._ai_actual   or "",
            # Aliases for round-logger and other shared code
            "ai_declared":       self._ai_declared or "",
            "ai_actual":         self._ai_actual   or "",
            "computer_gesture":  self._ai_actual   or "Unknown",
            "is_bluff":          self._is_bluff,
            "result_banner":     self.result_banner,
            "score_text":        f"You: {self.player_score}  |  AI: {self.robot_score}",
            "round_text":        f"ROUND {self.round_number}",
            "player_score":      self.player_score,
            "robot_score":       self.robot_score,
            "win_target":        self.win_target,
            "round_number":      self.round_number,
            "request_tracker_reset": self.tracker_reset_requested,
            "bluff_rate":        self.bluff_rate,
            "bluff_pct_so_far":  (sum(1 for r in self._log if r["is_bluff"]) / max(len(self._log), 1)),
            "research_log":      self._log,
            "two_player":        False,
            "opponent_type":     "AI",
        }

    def get_research_log(self):
        return list(self._log)

    def update(self, tracker_state, wrist_y=None, now=None):
        if now is None:
            now = time.monotonic()

        confirmed = tracker_state.get("confirmed_gesture", "Unknown")
        stable    = tracker_state.get("stable_gesture",   "Unknown")

        # ── ROUND_INTRO ────────────────────────────────────────────────────
        if self.state == "ROUND_INTRO":
            if now >= self.intro_until:
                # Plan the declaration before showing anything
                self._plan_declaration()
                self._declaration_until = now + DECLARATION_SECS
                self.state = "DECLARATION"
            return self._build_output(now)

        # ── DECLARATION ────────────────────────────────────────────────────
        if self.state == "DECLARATION":
            if now >= self._declaration_until:
                self.state = "WAITING_FOR_ROCK"
            return self._build_output(now)

        # ── ROUND_RESULT ───────────────────────────────────────────────────
        if self.state == "ROUND_RESULT":
            if now >= self.result_until:
                if self._round_is_over():
                    winner = "YOU WIN!" if self.player_score >= self.win_target else "AI WINS"
                    self.result_banner = winner
                    self.state         = "MATCH_RESULT"
                    self.match_until   = now + MATCH_RESULT_SECS
                else:
                    if self.last_round_result != "draw":
                        self.round_number += 1
                    self._reset_round(now)
            return self._build_output(now)

        # ── MATCH_RESULT ───────────────────────────────────────────────────
        if self.state == "MATCH_RESULT":
            if now >= self.match_until:
                self.reset_match(now)
            return self._build_output(now)

        # ── WAITING_FOR_ROCK ───────────────────────────────────────────────
        if self.state == "WAITING_FOR_ROCK":
            if confirmed == "Rock" and wrist_y is not None:
                self.last_rock_time = now
                self.state          = "COUNTDOWN"
                self.beat_count     = 0
                self.phase          = "ready_for_down"
                self.top_y          = wrist_y   # match FairPlayController exactly
                self.bottom_y       = wrist_y
            return self._build_output(now)

        # ── COUNTDOWN ──────────────────────────────────────────────────────
        if self.state == "COUNTDOWN":
            self._update_beat(wrist_y, confirmed, stable, now)
            # Beat 4 = SHOOT (display shows 1,2,3 via min(beat_count,3))
            # This matches FairPlayController: 4 pumps total, displayed as 1-2-3-SHOOT
            if self.beat_count >= 4:
                self.state            = "SHOOT_WINDOW"
                self.shoot_open_time  = now
                self.shoot_close_time = now + self.SHOOT_WINDOW
                self.tracker_reset_requested = True
            return self._build_output(now)

        # ── SHOOT_WINDOW ───────────────────────────────────────────────────
        if self.state == "SHOOT_WINDOW":
            thrown = confirmed if confirmed in VALID_SET else \
                     (stable   if stable   in VALID_SET else None)
            if now >= self.shoot_close_time or thrown:
                self._resolve_round(thrown or "Rock", now)
            return self._build_output(now)

        return self._build_output(now)

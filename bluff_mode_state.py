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

Where this fits in the codebase:
  - Standalone controller (does NOT inherit FairPlayController)
  - Uses FairPlayAI for the underlying move-choice logic
  - Renderer calls draw_bluff_mode_view() with the dict from _build_output()
  - Main loop calls update() every frame
"""

import time
import random
import csv
import os

from fair_play_ai import FairPlayAI

# The three valid gestures for this mode
VALID_GESTURES  = ("Rock", "Paper", "Scissors")
VALID_SET       = frozenset(VALID_GESTURES)

# What each gesture beats (used for outcome resolution)
BEATS = {
    "Rock":     "Scissors",
    "Paper":    "Rock",
    "Scissors": "Paper",
}
# Inverse of BEATS: what gesture beats each key
COUNTER = {v: k for k, v in BEATS.items()}

# ── Timing constants — DO NOT change, calibrated values ──────────────────────
INTRO_SECS        = 1.20   # brief "get ready" screen before the declaration
ROUND_RESULT_SECS = 2.60   # how long to show the round result
MATCH_RESULT_SECS = 2.40   # how long to show the match result before auto-resetting
SHOOT_WINDOW      = 0.90   # seconds the SHOOT window stays open
BEAT_COOLDOWN     = 0.25   # minimum time between beats (longer than FairPlay to prevent double-counting)
DOWN_THRESHOLD    = 0.060  # wrist must drop this far (normalised) to register a beat
UP_THRESHOLD      = 0.045  # wrist must rise this far after a beat to reset for the next one
ROCK_GRACE        = 0.50   # seconds the beat counter persists after Rock is no longer detected
DECLARATION_SECS  = 1.20   # how long to display the AI's declaration before countdown


def compare_rps(p, c):
    """
    Compare two RPS gestures from the player's perspective.
    Returns "win", "lose", or "draw".
    """
    if p == c:
        return "draw"
    if BEATS.get(p) == c:
        return "win"
    return "lose"


class BluffModeController:
    """
    Bluff Mode controller.

    State machine:
      ROUND_INTRO → DECLARATION → WAITING_FOR_ROCK → COUNTDOWN
        → SHOOT_WINDOW → ROUND_RESULT → (MATCH_RESULT or next round)

    update() signature:
        controller.update(tracker_state=..., wrist_y=..., now=...)
    """

    def __init__(self, ai=None, win_target=3, bluff_rate=0.60,
                 beat_cooldown=BEAT_COOLDOWN,
                 shoot_window_seconds=SHOOT_WINDOW):
        self.ai              = ai or FairPlayAI()
        self.win_target      = win_target
        self.bluff_rate      = bluff_rate       # probability of bluffing each round
        self.BEAT_COOLDOWN   = beat_cooldown
        self.SHOOT_WINDOW    = shoot_window_seconds
        self._log: list[dict] = []              # full research log across all rounds
        self.reset_match()

    def reset(self):
        """Alias so the main loop can call reset() without knowing internals."""
        self.reset_match()

    def reset_match(self, now=None):
        """Wipe scores and history, then reset the first round."""
        if now is None:
            now = time.monotonic()
        self.player_score = 0
        self.robot_score  = 0
        self.round_number = 1
        self._history: list[dict] = []   # round-by-round history fed back to the AI
        self.ai.reset()
        self._reset_round(now)

    def _reset_round(self, now=None):
        """
        Reset per-round state.  Called at the start of every round and after
        a draw (draws don't increment the round counter but still reset state).
        """
        if now is None:
            now = time.monotonic()
        self.state          = "ROUND_INTRO"
        self.intro_until    = now + INTRO_SECS
        self.beat_count     = 0
        self.phase          = "ready_for_down"   # pump detection phase
        self.top_y          = None               # highest wrist position seen in this beat cycle
        self.bottom_y       = None               # lowest wrist position seen in this beat cycle
        self.last_beat_time = 0.0
        self.last_rock_time = 0.0
        self.shoot_open_time  = None
        self.shoot_close_time = None
        self.tracker_reset_requested = False

        # Bluff decision fields — set by _plan_declaration()
        self._ai_actual      = None   # what the AI will actually play
        self._ai_declared    = None   # what the AI tells the player it will play
        self._is_bluff       = False  # whether the declaration is a lie
        self._declaration_until = None

        self.player_gesture    = "Unknown"
        self.result_banner     = ""
        self.last_round_result = None
        self.result_until      = None
        self.match_until       = None

    def _plan_declaration(self):
        """
        Decide what the AI will actually play (via FairPlayAI), then decide
        whether to bluff and compute what to declare.

        Bluff logic:
          actual = Rock  →  BEATS[Rock] = Scissors  →  declare "Scissors"
          Player thinks: "I need Paper to beat Scissors"
          AI plays Rock, which beats Paper  →  AI wins
        """
        actual = self.ai.choose_robot_move(
            history=self._history,
            round_number=self.round_number,
        )
        # Fallback in case the AI returns something unexpected
        if actual not in VALID_SET:
            actual = random.choice(VALID_GESTURES)

        self._ai_actual = actual
        self._is_bluff  = (random.random() < self.bluff_rate)

        if self._is_bluff:
            # Declare what beats actual, not actual itself
            self._ai_declared = BEATS[actual]
        else:
            # Tell the truth
            self._ai_declared = actual

    def _update_beat(self, wrist_y, confirmed, stable, now):
        """
        Pump detection — mirrors the logic in FairPlayController exactly.

        A "beat" is one full down-then-up wrist pump while Rock is held.
        We track the wrist Y position across frames and register a beat when
        the wrist moves down by DOWN_THRESHOLD then back up by UP_THRESHOLD.

        If Rock disappears for longer than ROCK_GRACE the countdown resets.
        """
        confirmed_rock = confirmed == "Rock"
        stable_rock    = stable    == "Rock"
        rock_detected  = (confirmed_rock or stable_rock) and wrist_y is not None
        within_grace   = (now - self.last_rock_time) <= ROCK_GRACE

        # We can track wrist movement if Rock is visible OR we're in the grace window
        can_track = rock_detected or (within_grace and wrist_y is not None
                                       and self.beat_count > 0)

        if rock_detected:
            self.last_rock_time = now

        if not can_track:
            # Grace expired with no Rock — reset the countdown to zero
            if not within_grace and self.beat_count > 0:
                self.beat_count = 0
                self.phase      = "ready_for_down"
                self.top_y      = None
                self.bottom_y   = None
            return

        cooldown_ok = (now - self.last_beat_time) >= self.BEAT_COOLDOWN

        if self.phase == "ready_for_down":
            # Track the highest point the wrist reaches between beats
            if self.top_y is None:
                self.top_y = wrist_y
            self.top_y = min(self.top_y, wrist_y)

            # Register a beat when the wrist has dropped far enough
            if (wrist_y - self.top_y) >= DOWN_THRESHOLD and cooldown_ok:
                self.beat_count    += 1
                self.last_beat_time = now
                self.phase          = "waiting_for_up"
                self.bottom_y       = wrist_y

        elif self.phase == "waiting_for_up":
            # Track the lowest point after the downstroke
            if self.bottom_y is None:
                self.bottom_y = wrist_y
            self.bottom_y = max(self.bottom_y, wrist_y)

            # Reset to ready_for_down once the wrist has come back up
            if (self.bottom_y - wrist_y) >= UP_THRESHOLD:
                self.phase = "ready_for_down"
                self.top_y = wrist_y

    def _resolve_round(self, player_g, now):
        """
        Compare player_g against the AI's actual move (not the declared one)
        and update scores.  Also append to the research log.
        """
        self.player_gesture = player_g
        outcome = compare_rps(player_g, self._ai_actual)

        # Update scores and set the result banner
        if outcome == "win":
            self.player_score += 1
            self.result_banner = "YOU WIN THE ROUND"
        elif outcome == "lose":
            self.robot_score  += 1
            self.result_banner = "AI WINS THE ROUND"
        else:
            self.result_banner = "DRAW"

        self.last_round_result = outcome

        # Research log: did the player throw the move that would have beaten
        # the declared (possibly fake) move?
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

        # Feed the round into the AI's history for future predictions
        self._history.append({
            "round_number":   self.round_number,
            "player_gesture": player_g,
            "robot_gesture":  self._ai_actual,
            "player_outcome": outcome,
        })

        # Update the bandit model if the AI supports it (improves future predictions)
        if hasattr(self.ai, "update_bandit") and hasattr(self.ai, "last_prediction"):
            pred = self.ai.last_prediction or {}
            predicted_player = pred.get("used_predicted_move")
            if predicted_player:
                self.ai.update_bandit(predicted_player, player_g)

        self.state        = "ROUND_RESULT"
        self.result_until = now + ROUND_RESULT_SECS

    def _round_is_over(self):
        """Return True if either player has reached the win target."""
        return (self.player_score >= self.win_target or
                self.robot_score  >= self.win_target)

    def _build_output(self, now):
        """
        Package all current state into a flat dict for the renderer.
        Several fields are duplicated under different key names to satisfy
        both the dedicated bluff renderer and shared logging/round code.
        """
        # Shoot window countdown — only valid during SHOOT_WINDOW state
        tl = max(0.0, self.shoot_close_time - now) if self.shoot_close_time else 0.0

        # What fraction of rounds so far have been bluffs (for research display)
        bluff_pct = sum(1 for r in self._log if r["is_bluff"]) / max(len(self._log), 1)

        return {
            "play_mode_label":   "Bluff Mode",
            "state":             self.state,
            "beat_count":        self.beat_count,
            "time_left":         tl,
            "player_gesture":    self.player_gesture,
            # Keys expected by the renderer
            "ai_declared_move":  self._ai_declared or "",
            "ai_actual_move":    self._ai_actual   or "",
            # Aliases for the round-logger and other shared code
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
            "bluff_pct_so_far":  bluff_pct,
            # Last 6 rounds shown as a declaration history panel
            "declaration_history": [
                {"declared": r["ai_declared"], "actual": r["ai_actual"],
                 "outcome": r["outcome"], "is_bluff": r["is_bluff"]}
                for r in self._log[-6:]
            ],
            "research_log":      self._log,
            "two_player":        False,
            "opponent_type":     "AI",
        }

    def get_research_log(self):
        """Return a copy of the full research log (list of round dicts)."""
        return list(self._log)

    def update(self, tracker_state, wrist_y=None, now=None):
        """
        Main tick — call once per frame.

        tracker_state : dict from the gesture tracker
        wrist_y       : normalised wrist Y coordinate for beat detection
        now           : optional monotonic timestamp
        """
        if now is None:
            now = time.monotonic()

        confirmed = tracker_state.get("confirmed_gesture", "Unknown")
        stable    = tracker_state.get("stable_gesture",   "Unknown")

        # ── ROUND_INTRO: brief "get ready" pause ─────────────────────────────
        if self.state == "ROUND_INTRO":
            if now >= self.intro_until:
                # Plan the bluff/truth declaration before showing anything
                self._plan_declaration()
                self._declaration_until = now + DECLARATION_SECS
                self.state = "DECLARATION"
            return self._build_output(now)

        # ── DECLARATION: show the AI's declared move ──────────────────────────
        if self.state == "DECLARATION":
            if now >= self._declaration_until:
                # Declaration period over — player can now make their fist
                self.state = "WAITING_FOR_ROCK"
            return self._build_output(now)

        # ── ROUND_RESULT: show outcome, then move to next round or match end ──
        if self.state == "ROUND_RESULT":
            if now >= self.result_until:
                if self._round_is_over():
                    # Someone has reached the win target — end the match
                    winner = "YOU WIN!" if self.player_score >= self.win_target else "AI WINS"
                    self.result_banner = winner
                    self.state         = "MATCH_RESULT"
                    self.match_until   = now + MATCH_RESULT_SECS
                else:
                    # Only increment the round counter if it wasn't a draw
                    if self.last_round_result != "draw":
                        self.round_number += 1
                    self._reset_round(now)
            return self._build_output(now)

        # ── MATCH_RESULT: show winner, auto-reset after timer ─────────────────
        if self.state == "MATCH_RESULT":
            if now >= self.match_until:
                self.reset_match(now)
            return self._build_output(now)

        # ── WAITING_FOR_ROCK: player must form a fist to start the countdown ──
        if self.state == "WAITING_FOR_ROCK":
            if confirmed == "Rock" and wrist_y is not None:
                # Rock detected — initialise beat tracking and start countdown
                self.last_rock_time = now
                self.state          = "COUNTDOWN"
                self.beat_count     = 0
                self.phase          = "ready_for_down"
                self.top_y          = wrist_y   # starting position for beat measurement
                self.bottom_y       = wrist_y
            return self._build_output(now)

        # ── COUNTDOWN: count pump beats 1-2-3, then open SHOOT window ────────
        if self.state == "COUNTDOWN":
            self._update_beat(wrist_y, confirmed, stable, now)
            # 4 total pump beats map to the display sequence 1, 2, 3, SHOOT
            if self.beat_count >= 4:
                self.state            = "SHOOT_WINDOW"
                self.shoot_open_time  = now
                self.shoot_close_time = now + self.SHOOT_WINDOW
                self.tracker_reset_requested = True
            return self._build_output(now)

        # ── SHOOT_WINDOW: accept whatever gesture the player throws ──────────
        if self.state == "SHOOT_WINDOW":
            # Accept confirmed first, then stable as a fallback
            thrown = (confirmed if confirmed in VALID_SET
                      else (stable if stable in VALID_SET else None))

            # Close the window either on a detected gesture or on timeout
            if now >= self.shoot_close_time or thrown:
                self._resolve_round(thrown or "Rock", now)
            return self._build_output(now)

        return self._build_output(now)

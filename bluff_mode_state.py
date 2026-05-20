"""
bluff_mode_state.py
===================
Bluff Mode — the AI declares its "intended" move before every round, but it
might be lying. This is a research mode studying whether knowing the opponent's
declared move changes how a player behaves.

How the bluff works:
  - 60% of the time (configurable): the AI bluffs.
    It declares the gesture it wants the player to COUNTER, then plays the
    move that BEATS that counter.
    Example: AI plans to play Rock. It declares "Scissors", hoping the player
    throws Paper (to beat Scissors). Rock then beats Paper — AI wins.
  - 40% of the time: the AI tells the truth and plays what it declared.

Every round is logged with:
  - What was declared vs what was actually played
  - Whether the declaration was a bluff
  - Whether the player "followed" the declaration (played to beat it)
  - The final outcome

State flow:
  ROUND_INTRO -> DECLARATION -> WAITING_FOR_ROCK -> COUNTDOWN
    -> SHOOT_WINDOW -> ROUND_RESULT -> (MATCH_RESULT or next round)

Best-of-5 (first to 3 wins).
"""

import time
import random

from fair_play_ai import FairPlayAI

# The three valid RPS gestures
VALID_GESTURES = ("Rock", "Paper", "Scissors")
VALID_SET      = frozenset(VALID_GESTURES)

# What each gesture beats
BEATS = {
    "Rock":     "Scissors",
    "Paper":    "Rock",
    "Scissors": "Paper",
}

# Reverse of BEATS: what gesture beats each key (used to check if player "followed")
COUNTER = {v: k for k, v in BEATS.items()}

# Timing constants — calibrated, don't change without re-testing
INTRO_SECS        = 1.20  # "get ready" pause before the declaration is shown
ROUND_RESULT_SECS = 2.60  # how long the round result screen stays up
MATCH_RESULT_SECS = 2.40  # how long the match result screen stays before auto-reset
SHOOT_WINDOW      = 0.90  # seconds the SHOOT window stays open
BEAT_COOLDOWN     = 0.25  # minimum seconds between beats (slightly longer than FairPlay)
DOWN_THRESHOLD    = 0.060 # wrist must drop this far (normalised) to register a beat
UP_THRESHOLD      = 0.045 # wrist must rise this far after a beat to reset for the next one
ROCK_GRACE        = 0.50  # seconds the beat counter persists after Rock is no longer detected
DECLARATION_SECS  = 1.20  # how long the AI's declaration is shown before the countdown


def compare_rps(p, c):
    """
    Compare two RPS gestures from the player's perspective.
    Returns "win" if p beats c, "lose" if c beats p, or "draw" if they match.
    """
    if p == c:
        return "draw"
    if BEATS.get(p) == c:
        return "win"
    return "lose"


class BluffModeController:
    """
    Bluff Mode game controller.

    Key difference from standard mode: the AI announces its move in advance
    via the DECLARATION state, but the declaration may be a lie (a bluff).
    The player sees the declaration and must decide whether to trust it.
    """

    def __init__(self, ai=None, win_target=3, bluff_rate=0.60,
                 beat_cooldown=BEAT_COOLDOWN,
                 shoot_window_seconds=SHOOT_WINDOW):
        self.ai            = ai or FairPlayAI()
        self.win_target    = win_target
        self.bluff_rate    = bluff_rate     # probability (0–1) of bluffing each round
        self.BEAT_COOLDOWN = beat_cooldown
        self.SHOOT_WINDOW  = shoot_window_seconds
        self._log: list[dict] = []          # full research log across all rounds
        self.reset_match()

    def reset(self):
        """Alias so the main loop can call reset() without knowing internals."""
        self.reset_match()

    def reset_match(self, now=None):
        """Wipe scores and history, then start round 1."""
        if now is None:
            now = time.monotonic()
        self.player_score = 0
        self.robot_score  = 0
        self.round_number = 1
        self._history: list[dict] = []  # round history fed to the AI for its predictions
        self.ai.reset()
        self._reset_round(now)

    def _reset_round(self, now=None):
        """
        Reset per-round state. Called at the start of every round.
        Draws don't increment round_number but do call this to reset everything else.
        """
        if now is None:
            now = time.monotonic()

        # State machine start and timing
        self.state       = "ROUND_INTRO"
        self.intro_until = now + INTRO_SECS

        # Pump / beat detection state (single hand, so simpler than two_player_state.py)
        self.beat_count     = 0
        self.phase          = "ready_for_down"  # current phase of the pump state machine
        self.top_y          = None              # highest wrist position seen between beats
        self.bottom_y       = None              # lowest wrist position during current stroke
        self.last_beat_time = 0.0
        self.last_rock_time = 0.0

        # Shoot window timing
        self.shoot_open_time  = None
        self.shoot_close_time = None

        self.tracker_reset_requested = False

        # AI declaration fields — populated by _plan_declaration()
        self._ai_actual      = None   # what the AI will actually play
        self._ai_declared    = None   # what the AI says it will play (may be a lie)
        self._is_bluff       = False  # True if the declaration is a lie
        self._declaration_until = None

        # Per-round display state
        self.player_gesture    = "Unknown"
        self.result_banner     = ""
        self.last_round_result = None
        self.result_until      = None
        self.match_until       = None

    def _plan_declaration(self):
        """
        Decide what the AI will actually play (via FairPlayAI), then decide
        whether to bluff and what to declare.

        Bluff example:
          AI decides to play Rock.
          BEATS["Rock"] = "Scissors", so it declares "Scissors".
          The player thinks: "I need Paper to beat Scissors."
          AI plays Rock, which beats Paper -> AI wins.

        Truth: AI just declares what it's actually going to play.
        """
        # Let the AI pick a move based on the player's history
        actual = self.ai.choose_robot_move(
            history=self._history,
            round_number=self.round_number,
        )

        # Fallback if the AI returns something unexpected
        if actual not in VALID_SET:
            actual = random.choice(VALID_GESTURES)

        self._ai_actual = actual
        self._is_bluff  = (random.random() < self.bluff_rate)

        if self._is_bluff:
            # Declare what actual BEATS, not actual itself
            # This leads the player to think they should counter that weaker gesture
            self._ai_declared = BEATS[actual]
        else:
            # Honest declaration
            self._ai_declared = actual

    def _update_beat(self, wrist_y, confirmed, stable, now):
        """
        Pump detection for a single hand. Called every frame during COUNTDOWN.

        A "beat" is one complete down-then-up wrist pump while Rock is held.
        The detector tracks wrist Y position across frames:
          - Phase "ready_for_down": wait for the wrist to drop DOWN_THRESHOLD from its peak
          - Phase "waiting_for_up": wait for it to rise back UP_THRESHOLD before re-arming

        If Rock disappears for longer than ROCK_GRACE seconds, the countdown resets to 0.
        """
        confirmed_rock = confirmed == "Rock"
        stable_rock    = stable    == "Rock"
        rock_detected  = (confirmed_rock or stable_rock) and wrist_y is not None
        within_grace   = (now - self.last_rock_time) <= ROCK_GRACE

        # We can track the wrist if Rock is currently visible, or if we're inside the grace window
        can_track = rock_detected or (within_grace and wrist_y is not None
                                      and self.beat_count > 0)

        if rock_detected:
            self.last_rock_time = now  # refresh the grace period timer

        if not can_track:
            # Grace expired with no Rock — reset the countdown back to zero
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

            # Register a beat once the wrist has dropped far enough from the peak
            if (wrist_y - self.top_y) >= DOWN_THRESHOLD and cooldown_ok:
                self.beat_count    += 1
                self.last_beat_time = now
                self.phase          = "waiting_for_up"
                self.bottom_y       = wrist_y

        elif self.phase == "waiting_for_up":
            # Track the lowest point during the downstroke
            if self.bottom_y is None:
                self.bottom_y = wrist_y
            self.bottom_y = max(self.bottom_y, wrist_y)

            # Once the wrist has risen back up enough, re-arm for the next beat
            if (self.bottom_y - wrist_y) >= UP_THRESHOLD:
                self.phase = "ready_for_down"
                self.top_y = wrist_y

    def _resolve_round(self, player_g, now):
        """
        Compare the player's gesture against the AI's actual move (not the declared one).
        Update scores, set the result banner, and append to the research log.
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

        # Research log: did the player throw the gesture that would have beaten
        # the declared (possibly fake) move? This measures how much the declaration influenced them.
        player_followed = (player_g == COUNTER.get(self._ai_declared, ""))
        self._log.append({
            "round":                      self.round_number,
            "ai_declared":                self._ai_declared,
            "ai_actual":                  self._ai_actual,
            "is_bluff":                   self._is_bluff,
            "player_move":                player_g,
            "outcome":                    outcome,
            "player_followed_declaration": player_followed,
        })

        # Feed this round into the AI's history so it can improve future predictions
        self._history.append({
            "round_number":   self.round_number,
            "player_gesture": player_g,
            "robot_gesture":  self._ai_actual,
            "player_outcome": outcome,
        })

        # Update the bandit model if the AI supports it (tunes prediction weights)
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
        Package all current state into a flat dict for the renderer to read.

        Some fields are duplicated under different names to satisfy both the
        dedicated bluff renderer and shared logging/round code.
        """
        # How long is left in the shoot window (only meaningful during SHOOT_WINDOW)
        time_left = max(0.0, self.shoot_close_time - now) if self.shoot_close_time else 0.0

        # What fraction of rounds so far have been bluffs (shown as a research stat)
        bluff_pct = sum(1 for r in self._log if r["is_bluff"]) / max(len(self._log), 1)

        return {
            "play_mode_label":   "Bluff Mode",
            "state":             self.state,
            "beat_count":        self.beat_count,
            "time_left":         time_left,
            "player_gesture":    self.player_gesture,
            # Both naming conventions for the declared/actual move (different parts of the UI use each)
            "ai_declared_move":  self._ai_declared or "",
            "ai_actual_move":    self._ai_actual   or "",
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
            # Last 6 rounds shown as a declaration history panel in the UI
            "declaration_history": [
                {"declared": r["ai_declared"], "actual": r["ai_actual"],
                 "outcome": r["outcome"], "is_bluff": r["is_bluff"]}
                for r in self._log[-6:]
            ],
            "research_log":  self._log,
            "two_player":    False,
            "opponent_type": "AI",
        }

    def get_research_log(self):
        """Return a copy of the full research log (list of round dicts)."""
        return list(self._log)

    def update(self, tracker_state, wrist_y=None, now=None):
        """
        Main tick — call once per frame from the game loop.

        tracker_state : dict from the gesture tracker with 'confirmed_gesture' etc.
        wrist_y       : normalised wrist Y coordinate for beat detection
        now           : optional monotonic timestamp (uses time.monotonic() if omitted)
        """
        if now is None:
            now = time.monotonic()

        confirmed = tracker_state.get("confirmed_gesture", "Unknown")
        stable    = tracker_state.get("stable_gesture",   "Unknown")

        # ROUND_INTRO: brief "get ready" pause before we show the declaration
        if self.state == "ROUND_INTRO":
            if now >= self.intro_until:
                # Plan the bluff/truth decision before revealing anything to the player
                self._plan_declaration()
                self._declaration_until = now + DECLARATION_SECS
                self.state = "DECLARATION"
            return self._build_output(now)

        # DECLARATION: show the AI's declared move; wait for the timer to expire
        if self.state == "DECLARATION":
            if now >= self._declaration_until:
                self.state = "WAITING_FOR_ROCK"  # player can now form a fist to start
            return self._build_output(now)

        # ROUND_RESULT: show the outcome, then advance or end the match
        if self.state == "ROUND_RESULT":
            if now >= self.result_until:
                if self._round_is_over():
                    # Someone reached the win target — declare a match winner
                    winner = "YOU WIN!" if self.player_score >= self.win_target else "AI WINS"
                    self.result_banner = winner
                    self.state       = "MATCH_RESULT"
                    self.match_until = now + MATCH_RESULT_SECS
                else:
                    # Don't increment round number on a draw — replay the same round number
                    if self.last_round_result != "draw":
                        self.round_number += 1
                    self._reset_round(now)
            return self._build_output(now)

        # MATCH_RESULT: show the winner, then auto-reset for a new match
        if self.state == "MATCH_RESULT":
            if now >= self.match_until:
                self.reset_match(now)
            return self._build_output(now)

        # WAITING_FOR_ROCK: player must form a fist to begin the countdown
        if self.state == "WAITING_FOR_ROCK":
            if confirmed == "Rock" and wrist_y is not None:
                # Rock detected — start the beat detector and move to COUNTDOWN
                self.last_rock_time = now
                self.state          = "COUNTDOWN"
                self.beat_count     = 0
                self.phase          = "ready_for_down"
                self.top_y          = wrist_y  # use current position as the baseline
                self.bottom_y       = wrist_y
            return self._build_output(now)

        # COUNTDOWN: count 4 pump beats, then open the SHOOT window
        if self.state == "COUNTDOWN":
            self._update_beat(wrist_y, confirmed, stable, now)
            # 4 beats map to the display sequence: 1, 2, 3, SHOOT
            if self.beat_count >= 4:
                self.state            = "SHOOT_WINDOW"
                self.shoot_open_time  = now
                self.shoot_close_time = now + self.SHOOT_WINDOW
                self.tracker_reset_requested = True  # flush tracker history so SHOOT reads fresh
            return self._build_output(now)

        # SHOOT_WINDOW: accept the player's throw, or close on timeout
        if self.state == "SHOOT_WINDOW":
            # Use confirmed gesture first, fall back to stable if not yet confirmed
            thrown = (confirmed if confirmed in VALID_SET
                      else (stable if stable in VALID_SET else None))

            # Resolve as soon as a gesture is detected, or when the window timer expires
            if now >= self.shoot_close_time or thrown:
                self._resolve_round(thrown or "Rock", now)  # default Rock if no gesture thrown
            return self._build_output(now)

        return self._build_output(now)

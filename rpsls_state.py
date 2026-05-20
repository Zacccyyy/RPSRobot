"""
rpsls_state.py
==============
Rock Paper Scissors Lizard Spock — best-of-5 (first to 3 wins) vs FairPlayAI.

RPSLS rules (15 outcomes instead of the usual 3):
  Rock     beats Lizard and Scissors
  Paper    beats Rock and Spock
  Scissors beats Paper and Lizard
  Lizard   beats Spock and Paper
  Spock    beats Scissors and Rock

Gesture detection:
  Rock     — closed fist
  Paper    — open palm, all fingers up
  Scissors — index + middle up only
  Lizard   — fingers curled like a sock-puppet mouth (thumb + index spread)
  Spock    — Vulcan salute: index+middle up, ring+pinky up, gap between them

Lizard/Spock are tricky to detect in side-view, so RPSLS works best in
Front-On orientation with the trained MLP model.

The AI uses a simple frequency-bias strategy: look at what the player has
thrown most recently, then pick a move that beats that gesture.

How it fits into the project:
  - Standalone controller (does NOT inherit from FairPlayController).
  - Renderer calls draw_rpsls_view() with the dict from _build_output().
  - Main loop calls update() every frame.
"""

import time
import random
from fair_play_ai import FairPlayAI

# All five valid gestures in the classic "Sheldon" order
VALID_RPSLS     = ("Rock", "Spock", "Paper", "Lizard", "Scissors")
VALID_RPSLS_SET = frozenset(VALID_RPSLS)  # used for fast "in" checks

# Maps each gesture to the two gestures it beats
BEATS = {
    "Scissors": ("Paper",   "Lizard"),
    "Paper":    ("Rock",    "Spock"),
    "Rock":     ("Lizard",  "Scissors"),
    "Lizard":   ("Spock",   "Paper"),
    "Spock":    ("Scissors","Rock"),
}

# Flavour text for the result banner, e.g. "Rock crushes Scissors"
BEAT_VERBS = {
    ("Scissors", "Paper"):   "Scissors cuts Paper",
    ("Scissors", "Lizard"):  "Scissors decapitates Lizard",
    ("Paper",    "Rock"):    "Paper covers Rock",
    ("Paper",    "Spock"):   "Paper disproves Spock",
    ("Rock",     "Lizard"):  "Rock crushes Lizard",
    ("Rock",     "Scissors"):"Rock crushes Scissors",
    ("Lizard",   "Spock"):   "Lizard poisons Spock",
    ("Lizard",   "Paper"):   "Lizard eats Paper",
    ("Spock",    "Scissors"):"Spock smashes Scissors",
    ("Spock",    "Rock"):    "Spock vaporizes Rock",
}

# Build the reverse of BEATS: for each gesture, which gestures beat it?
# (Each gesture in RPSLS loses to exactly 2 others.)
COUNTER_RPSLS: dict[str, list[str]] = {}
for _winner, _losers in BEATS.items():
    for _loser in _losers:
        COUNTER_RPSLS.setdefault(_loser, []).append(_winner)


def compare_rpsls(p, c):
    """
    Determine the outcome of a round from the player's perspective.
    Returns "win", "lose", or "draw".
    """
    if p == c:
        return "draw"
    if c in BEATS.get(p, ()):
        return "win"
    return "lose"


def beat_verb(winner, loser):
    """
    Return the flavour-text phrase for a win, e.g. "Rock crushes Scissors".
    Falls back to a plain "<winner> beats <loser>" if the pair isn't listed.
    """
    return BEAT_VERBS.get((winner, loser), f"{winner} beats {loser}")


# ── Timing constants (calibrated — do not change) ────────────────────────────
INTRO_DURATION       = 1.20   # seconds of "GET READY" before the fist prompt
ROUND_RESULT_SECONDS = 2.80   # longer than normal RPS so players can read the verb
MATCH_RESULT_SECONDS = 2.40   # show the match winner before auto-resetting
SHOOT_WINDOW         = 1.20   # shoot window is longer for RPSLS (more gestures to choose)
BEAT_COOLDOWN        = 0.18   # minimum time between pump beats
DOWN_THRESHOLD       = 0.045  # wrist drop required to register a beat (normalised Y)
UP_THRESHOLD         = 0.035  # wrist rise required to reset the beat detector
ROCK_GRACE           = 0.50   # seconds the beat counter persists after fist disappears

# Minimum time the player must hold their final gesture before we accept it.
# This prevents the transit pose (Rock → Spock) from being read as the wrong gesture.
MIN_SHOT_DWELL = 0.25


class RPSLSController:
    """
    RPSLS vs FairPlayAI, best-of-N (first to win_target wins).
    Uses the same pump-beat detection as FairPlayController.

    State machine per round:
      ROUND_INTRO → WAITING_FOR_ROCK → COUNTDOWN → SHOOT_WINDOW
        → ROUND_RESULT → (MATCH_RESULT or next ROUND_INTRO)
    """

    def __init__(self, ai=None, win_target=3,
                 beat_cooldown=0.18, shoot_window_seconds=SHOOT_WINDOW,
                 robot_output=None):
        self.ai            = ai or FairPlayAI()
        self.win_target    = win_target
        self.BEAT_COOLDOWN = beat_cooldown
        self.SHOOT_WINDOW  = shoot_window_seconds
        self.robot_output  = robot_output  # optional hardware interface
        self.reset_match()

    def reset(self):
        """Public alias so the main loop can call reset() without knowing internals."""
        self.reset_match()

    def reset_match(self, now=None):
        """Wipe scores, history, and AI state, then start a fresh first round."""
        if now is None:
            now = time.monotonic()
        self.player_score        = 0
        self.robot_score         = 0
        self.round_number        = 1
        self.history: list[dict] = []
        self.match_result_banner = ""
        self.match_until         = None
        self.ai.reset()
        self._reset_round(now)

    def _reset_round(self, now=None):
        """Reset all per-round state — called at the start of every round."""
        if now is None:
            now = time.monotonic()
        self.state          = "ROUND_INTRO"
        self.intro_until    = now + INTRO_DURATION
        self.beat_count     = 0
        self.phase          = "ready_for_down"   # beat detector phase
        self.top_y          = None
        self.bottom_y       = None
        self.last_beat_time = 0.0
        self.last_rock_time = 0.0
        self.shoot_open_time  = None
        self.shoot_close_time = None
        self.tracker_reset_requested = False
        self.ai_locked      = None      # AI's gesture, locked in at beat 3
        self.player_gesture = "Unknown"
        self.ai_gesture     = "Unknown"
        self.result_banner  = ""
        self.result_verb    = ""        # e.g. "Spock vaporizes Rock"
        self.last_round_result = None
        self.result_until   = None

    def consume_tracker_reset_request(self):
        """
        Called by the main loop once it has acted on the tracker reset request.
        Clears the flag so we don't request another reset next frame.
        """
        self.tracker_reset_requested = False

    def _lock_ai(self):
        """
        Lock in the AI's chosen gesture at beat 3 so it can't change before SHOOT.
        This method is a no-op if the AI has already been locked this round.
        """
        if self.ai_locked is not None:
            return  # already locked — don't overwrite
        self.ai_locked = self._ai_choose()

    def _ai_choose(self):
        """
        Simple frequency-bias AI: count what the player threw in recent rounds
        and return a gesture that beats the most common one.

        Falls back to a random choice for the first 3 rounds (not enough
        history to make a meaningful prediction yet).
        """
        if len(self.history) < 4:
            return random.choice(VALID_RPSLS)

        # Count how often each gesture appeared in the last 10 rounds
        freq = {g: 0 for g in VALID_RPSLS}
        for round_record in self.history[-10:]:
            g = round_record.get("player_gesture")
            if g in freq:
                freq[g] += 1

        # Assume the player will repeat their most frequent recent gesture
        predicted = max(freq, key=freq.get)

        # Pick a random gesture from those that beat the predicted move
        counters = COUNTER_RPSLS.get(predicted, [])
        if counters:
            return random.choice(counters)
        return random.choice(VALID_RPSLS)  # fallback if counters is empty

    def _update_beat(self, wrist_y, confirmed, now):
        """
        Pump-beat detector — same algorithm as FairPlayController.

        One full beat = wrist drops DOWN_THRESHOLD then rises UP_THRESHOLD.

        Both Rock and Spock count as valid starting gestures here because
        the player uses a similar fist position before spreading into Spock.
        """
        # Either Rock or Spock is acceptable as the "holding" gesture
        rock_held = confirmed in ("Rock", "Spock")
        if rock_held:
            self.last_rock_time = now  # refresh the grace window

        grace_ok    = (now - self.last_rock_time) <= ROCK_GRACE
        cooldown_ok = (now - self.last_beat_time) >= self.BEAT_COOLDOWN

        # If Rock/Spock is gone and grace expired, stop tracking
        if not grace_ok or wrist_y is None:
            return

        # First frame of tracking — initialise positions
        if self.top_y is None:
            self.top_y = self.bottom_y = wrist_y

        # Always keep the highest/lowest positions updated
        self.top_y    = min(self.top_y, wrist_y)
        self.bottom_y = max(self.bottom_y, wrist_y)

        if self.phase == "ready_for_down":
            # Wait for the wrist to drop far enough to count as a downstroke
            if cooldown_ok and (wrist_y - self.top_y) >= DOWN_THRESHOLD:
                self.phase          = "waiting_for_up"
                self.bottom_y       = wrist_y
                self.last_beat_time = now
                self.beat_count    += 1
                self.top_y          = wrist_y  # reset ceiling for the next beat

        elif self.phase == "waiting_for_up":
            self.bottom_y = max(self.bottom_y, wrist_y)
            # Wait for the wrist to come back up high enough
            if (self.bottom_y - wrist_y) >= UP_THRESHOLD:
                self.phase  = "ready_for_down"
                self.top_y  = self.bottom_y = wrist_y

    def _resolve_round(self, player_g, now):
        """
        Score the round, set the result banner and flavour verb, and
        record the result in the history list for the AI.
        """
        self.player_gesture = player_g
        self.ai_gesture     = self.ai_locked or "Rock"  # fallback if lock failed
        outcome = compare_rpsls(player_g, self.ai_gesture)

        if outcome == "win":
            self.player_score += 1
            self.result_banner = "YOU WIN THE ROUND"
            self.result_verb   = beat_verb(player_g, self.ai_gesture)
        elif outcome == "lose":
            self.robot_score += 1
            self.result_banner = "AI WINS THE ROUND"
            self.result_verb   = beat_verb(self.ai_gesture, player_g)
        else:
            self.result_banner = "DRAW"
            self.result_verb   = "Same gesture — no winner"

        self.last_round_result = outcome

        # Store a minimal record so the AI can analyse frequency trends
        self.history.append({
            "round_number":   self.round_number,
            "player_gesture": player_g,
            "player_outcome": outcome,
        })

        self.state        = "ROUND_RESULT"
        self.result_until = now + ROUND_RESULT_SECONDS

    def _round_is_over(self):
        """Return True if either side has reached the score needed to win the match."""
        return (self.player_score >= self.win_target or
                self.robot_score  >= self.win_target)

    def _build_output(self, now):
        """
        Package the current state into a flat dict for the renderer.
        The renderer reads this every frame — it never touches internal state directly.
        """
        # Shoot window countdown bar — only meaningful during SHOOT_WINDOW
        tl = max(0.0, self.shoot_close_time - now) if self.shoot_close_time else 0.0

        # Big centre label for each state
        main_text = {
            "ROUND_INTRO":      "GET READY",
            "WAITING_FOR_ROCK": "MAKE A FIST",
            "COUNTDOWN":        "READY" if self.beat_count == 0 else str(min(self.beat_count, 3)),
            "SHOOT_WINDOW":     "SHOOT!",
            "ROUND_RESULT":     self.result_banner,
            "MATCH_RESULT":     self.match_result_banner,
        }.get(self.state, self.state)

        return {
            "play_mode_label":       "RPSLS",
            "state":                 self.state,
            "state_label":           self.state.replace("_", " ").title(),
            "beat_count":            self.beat_count,
            "time_left":             tl,
            "main_text":             main_text,
            "sub_text":              "5-gesture variant",
            "player_gesture":        self.player_gesture,
            "computer_gesture":      self.ai_gesture,
            "ai_gesture":            self.ai_gesture,
            "result_banner":         self.result_banner,
            "result_verb":           self.result_verb,
            "score_text":            f"You: {self.player_score}  |  AI: {self.robot_score}",
            "round_text":            f"ROUND {self.round_number}",
            "round_number":          self.round_number,
            "player_score":          self.player_score,
            "robot_score":           self.robot_score,
            "win_target":            self.win_target,
            "request_tracker_reset": self.tracker_reset_requested,
            # "PENDING" until the AI locks in its gesture at beat 3
            "robot_move_command":    f"ROBOT_PLAY_{self.ai_locked.upper()}" if self.ai_locked else "PENDING",
            "two_player":            False,
            "opponent_type":         "",
        }

    def update(self, tracker_state, wrist_y=None, now=None):
        """
        Main tick — call once per frame.

        tracker_state : dict from the gesture tracker (needs "confirmed_gesture" and "stable_gesture")
        wrist_y       : normalised wrist Y coordinate (used for beat detection)
        now           : optional monotonic timestamp
        """
        if now is None:
            now = time.monotonic()

        confirmed = tracker_state.get("confirmed_gesture", "Unknown")
        stable    = tracker_state.get("stable_gesture",   "Unknown")

        # ── ROUND_INTRO: brief "get ready" pause before each round ───────────
        if self.state == "ROUND_INTRO":
            if now >= self.intro_until:
                self.state = "WAITING_FOR_ROCK"
            return self._build_output(now)

        # ── ROUND_RESULT: show outcome, then advance to next round or match end
        if self.state == "ROUND_RESULT":
            if now >= self.result_until:
                if self._round_is_over():
                    # Match finished — show the winner
                    winner = "YOU WIN!" if self.player_score >= self.win_target else "AI WINS"
                    self.match_result_banner = winner
                    self.state       = "MATCH_RESULT"
                    self.match_until = now + MATCH_RESULT_SECONDS
                else:
                    # Match continues — increment round counter (skip on draws)
                    if self.last_round_result != "draw":
                        self.round_number += 1
                    self._reset_round(now)
            return self._build_output(now)

        # ── MATCH_RESULT: show winner, then auto-reset the whole match ────────
        if self.state == "MATCH_RESULT":
            if now >= self.match_until:
                self.reset_match(now)
            return self._build_output(now)

        # ── WAITING_FOR_ROCK: player must form a fist or Spock hand ──────────
        if self.state == "WAITING_FOR_ROCK":
            # Both Rock and Spock are valid starting positions for the countdown
            if confirmed in ("Rock", "Spock") or stable in ("Rock", "Spock"):
                self.last_rock_time = now
                self.state          = "COUNTDOWN"
                self.beat_count     = 0
                self.phase          = "ready_for_down"
                self.top_y          = self.bottom_y = wrist_y
            return self._build_output(now)

        # ── COUNTDOWN: accumulate 4 pump beats, then lock AI and open SHOOT ──
        if self.state == "COUNTDOWN":
            self._update_beat(wrist_y, confirmed, now)

            # Lock the AI's move at beat 3 so it's decided before SHOOT opens
            if self.beat_count >= 3:
                self._lock_ai()

            # 4 beats = open the shoot window and request a tracker reset
            if self.beat_count >= 4:
                self.state                   = "SHOOT_WINDOW"
                self.shoot_open_time         = now
                self.shoot_close_time        = now + self.SHOOT_WINDOW
                self.tracker_reset_requested = True

            return self._build_output(now)

        # ── SHOOT_WINDOW: read the player's final gesture ────────────────────
        if self.state == "SHOOT_WINDOW":
            # Prefer confirmed (more reliable), fall back to stable
            thrown = (confirmed if confirmed in VALID_RPSLS_SET
                      else (stable if stable in VALID_RPSLS_SET else None))

            # Only accept the gesture after MIN_SHOT_DWELL seconds have elapsed.
            # This gives the hand time to fully transition from Rock to Spock/Lizard
            # without the transit pose being read as the final gesture.
            elapsed      = now - self.shoot_open_time
            gesture_ready = thrown and elapsed >= MIN_SHOT_DWELL

            if now >= self.shoot_close_time or gesture_ready:
                # Either we detected a clear gesture or time ran out — lock it in
                self._resolve_round(thrown or "Rock", now)

            return self._build_output(now)

        return self._build_output(now)

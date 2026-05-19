"""
rpsls_state.py
==============
Rock Paper Scissors Lizard Spock — best-of-5 (first to 3) vs FairPlayAI.

RPSLS rules (10 outcomes, cyclic modular arithmetic):
  Numbered: Rock=0, Spock=1, Paper=2, Lizard=3, Scissors=4
  Player wins if (player_num - computer_num) % 5 in {1, 2}

Gesture detection:
  Rock     — closed fist                 (existing)
  Paper    — open palm, all fingers up   (existing)
  Scissors — index + middle up only      (existing)
  Lizard   — fingers curled like a sock-puppet mouth:
             thumb + index spread, ring/pinky down, middle down
             -> detected as "Lizard" by front_on or dedicated curl logic
  Spock    — Vulcan salute: index+middle up, ring+pinky up, split between them
             -> index+middle separated from ring+pinky by a gap

Because Lizard/Spock share landmark patterns with Scissors/Paper in
side-view, RPSLS works best in Front-On orientation with the trained MLP.
In side-view mode we use curl analysis heuristics.

The AI is the standard FairPlayAI extended to a 5-gesture action space.

Where this fits in the codebase:
  - Standalone controller (does NOT inherit FairPlayController)
  - Uses a frequency-bias AI rather than the full FairPlayAI bandit model
  - Renderer calls draw_rpsls_view() with the dict from _build_output()
  - Main loop calls update() every frame
"""

import time
import random
from fair_play_ai import FairPlayAI

# All valid gestures in RPSLS — order matches the classic "Sheldon" circle
VALID_RPSLS     = ("Rock", "Spock", "Paper", "Lizard", "Scissors")
VALID_RPSLS_SET = frozenset(VALID_RPSLS)

# Maps each gesture to the two gestures it beats
BEATS = {
    "Scissors": ("Paper", "Lizard"),
    "Paper":    ("Rock",  "Spock"),
    "Rock":     ("Lizard","Scissors"),
    "Lizard":   ("Spock", "Paper"),
    "Spock":    ("Scissors","Rock"),
}

# Human-readable win phrases shown in the result banner
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

# Build the reverse map: for each gesture, which gestures beat it?
# Each gesture in RPSLS loses to exactly 2 others.
COUNTER_RPSLS: dict[str, list[str]] = {}
for _winner, _losers in BEATS.items():
    for _loser in _losers:
        COUNTER_RPSLS.setdefault(_loser, []).append(_winner)


def compare_rpsls(p, c):
    """Returns 'win', 'lose', or 'draw' from the player's perspective (p vs computer c)."""
    if p == c:
        return "draw"
    if c in BEATS.get(p, ()):
        return "win"
    return "lose"


def beat_verb(winner, loser):
    """
    Return the flavour-text phrase for a win, e.g. "Rock crushes Scissors".
    Falls back to a generic string if the pair isn't in BEAT_VERBS.
    """
    return BEAT_VERBS.get((winner, loser), f"{winner} beats {loser}")


# ── Timing constants — DO NOT change, calibrated values ──────────────────────
INTRO_DURATION       = 1.20   # seconds of "GET READY" before the fist prompt
ROUND_RESULT_SECONDS = 2.80   # longer than normal RPS so players can read the verb
MATCH_RESULT_SECONDS = 2.40   # show the match winner before auto-resetting
SHOOT_WINDOW         = 1.20   # seconds the SHOOT window stays open (longer for 5 gestures)
BEAT_COOLDOWN        = 0.18   # minimum time between pump beats
DOWN_THRESHOLD       = 0.045  # wrist drop required to register a beat (normalised Y)
UP_THRESHOLD         = 0.035  # wrist rise required to reset beat detector
ROCK_GRACE           = 0.50   # seconds the beat counter persists after fist disappears

# Minimum time the player must hold their final gesture before we accept it.
# Prevents Rock-to-Spock transit poses from being read as the wrong gesture.
MIN_SHOT_DWELL = 0.25


class RPSLSController:
    """
    RPSLS vs FairPlayAI, best-of-N (first to win_target wins).
    Uses the same pump-beat detection as FairPlayController.

    State machine per round:
      ROUND_INTRO -> WAITING_FOR_ROCK -> COUNTDOWN -> SHOOT_WINDOW
        -> ROUND_RESULT -> (MATCH_RESULT or next ROUND_INTRO)
    """

    def __init__(self, ai=None, win_target=3,
                 beat_cooldown=0.18, shoot_window_seconds=SHOOT_WINDOW,
                 robot_output=None):
        self.ai            = ai or FairPlayAI()
        self.win_target    = win_target
        self.BEAT_COOLDOWN = beat_cooldown
        self.SHOOT_WINDOW  = shoot_window_seconds
        self.robot_output  = robot_output   # optional hardware robot output interface
        self.reset_match()

    def reset(self):
        """Alias so the main loop can call reset() without knowing internals."""
        self.reset_match()

    def reset_match(self, now=None):
        """Wipe scores, history, and AI state, then start the first round."""
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
        self.result_verb    = ""        # flavour text, e.g. "Spock vaporizes Rock"
        self.last_round_result = None
        self.result_until   = None

    def consume_tracker_reset_request(self):
        """
        Clear the tracker reset flag once the main loop has acted on it.
        Calling this prevents repeated resets on consecutive frames.
        """
        self.tracker_reset_requested = False

    def _lock_ai(self):
        """
        Lock in the AI's gesture at beat 3 so it can't change before SHOOT.
        Only runs once per round — subsequent calls are no-ops.
        """
        if self.ai_locked is not None:
            return
        self.ai_locked = self._ai_choose()

    def _ai_choose(self):
        """
        Simple RPSLS AI: look at what the player has thrown in recent rounds
        and return a gesture that beats the most frequent one.

        Falls back to a random gesture for the first 3 rounds (not enough
        data for a meaningful frequency count yet).
        """
        if len(self.history) < 4:
            return random.choice(VALID_RPSLS)

        # Count how often each gesture appeared in the last 10 rounds
        freq = {g: 0 for g in VALID_RPSLS}
        for r in self.history[-10:]:
            g = r.get("player_gesture")
            if g in freq:
                freq[g] += 1

        # Predict the player will throw their most frequent recent gesture
        predicted = max(freq, key=freq.get)

        # Pick randomly from the gestures that beat the predicted move
        counters = COUNTER_RPSLS.get(predicted, [])
        if counters:
            return random.choice(counters)
        return random.choice(VALID_RPSLS)

    def _update_beat(self, wrist_y, confirmed, now):
        """
        Pump-beat detector — same algorithm as FairPlayController.

        Spock counts as a valid "rock" for the purpose of starting the
        countdown because the player uses a similar fist-shaped starting
        position before spreading into the Vulcan salute.

        One full beat = wrist drops DOWN_THRESHOLD then rises UP_THRESHOLD.
        """
        # Both Rock and Spock are valid "start" gestures for the countdown
        rock_held = confirmed in ("Rock", "Spock")
        if rock_held:
            self.last_rock_time = now

        grace_ok    = (now - self.last_rock_time) <= ROCK_GRACE
        cooldown_ok = (now - self.last_beat_time) >= self.BEAT_COOLDOWN

        # If the player has stopped holding Rock/Spock and the grace window expired, stop tracking
        if not grace_ok or wrist_y is None:
            return

        # Initialise tracking positions on first frame
        if self.top_y is None:
            self.top_y = self.bottom_y = wrist_y

        self.top_y    = min(self.top_y, wrist_y)
        self.bottom_y = max(self.bottom_y, wrist_y)

        if self.phase == "ready_for_down":
            # Wait for the wrist to drop enough to register a downstroke
            if cooldown_ok and (wrist_y - self.top_y) >= DOWN_THRESHOLD:
                self.phase          = "waiting_for_up"
                self.bottom_y       = wrist_y
                self.last_beat_time = now
                self.beat_count     += 1
                self.top_y          = wrist_y   # reset ceiling for next beat

        elif self.phase == "waiting_for_up":
            self.bottom_y = max(self.bottom_y, wrist_y)
            # Wait for the wrist to come back up
            if (self.bottom_y - wrist_y) >= UP_THRESHOLD:
                self.phase  = "ready_for_down"
                self.top_y  = self.bottom_y = wrist_y

    def _resolve_round(self, player_g, now):
        """
        Score the round, set the result banner and flavour verb, and
        append to history.
        """
        self.player_gesture = player_g
        self.ai_gesture     = self.ai_locked or "Rock"
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

        # Append a minimal history entry for the AI's frequency analysis
        self.history.append({
            "round_number":   self.round_number,
            "player_gesture": player_g,
            "player_outcome": outcome,
        })

        self.state        = "ROUND_RESULT"
        self.result_until = now + ROUND_RESULT_SECONDS

    def _round_is_over(self):
        """Return True if either side has reached the win target."""
        return (self.player_score >= self.win_target or
                self.robot_score  >= self.win_target)

    def _build_output(self, now):
        """
        Package the current state into a flat dict for the renderer.
        main_text is what the big centre label shows each state.
        """
        # Shoot window countdown — only meaningful during SHOOT_WINDOW
        tl = max(0.0, self.shoot_close_time - now) if self.shoot_close_time else 0.0

        # Human-readable label for the current state
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
            # Robot output command — "PENDING" until the AI has locked in
            "robot_move_command":    f"ROBOT_PLAY_{self.ai_locked.upper()}" if self.ai_locked else "PENDING",
            "two_player":            False,
            "opponent_type":         "",
        }

    def update(self, tracker_state, wrist_y=None, now=None):
        """
        Main tick — call once per frame.

        tracker_state : dict from the gesture tracker (needs confirmed and stable)
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
                self.state = "WAITING_FOR_ROCK"
            return self._build_output(now)

        # ── ROUND_RESULT: show outcome, then advance ──────────────────────────
        if self.state == "ROUND_RESULT":
            if now >= self.result_until:
                if self._round_is_over():
                    # End the match
                    winner = "YOU WIN!" if self.player_score >= self.win_target else "AI WINS"
                    self.match_result_banner = winner
                    self.state       = "MATCH_RESULT"
                    self.match_until = now + MATCH_RESULT_SECONDS
                else:
                    # Continue — only count the round if it wasn't a draw
                    if self.last_round_result != "draw":
                        self.round_number += 1
                    self._reset_round(now)
            return self._build_output(now)

        # ── MATCH_RESULT: show winner, then auto-reset ────────────────────────
        if self.state == "MATCH_RESULT":
            if now >= self.match_until:
                self.reset_match(now)
            return self._build_output(now)

        # ── WAITING_FOR_ROCK: player must form a fist or Spock hand ──────────
        if self.state == "WAITING_FOR_ROCK":
            # Both Rock and Spock are accepted as the starting fist gesture
            if confirmed in ("Rock", "Spock") or stable in ("Rock", "Spock"):
                self.last_rock_time = now
                self.state          = "COUNTDOWN"
                self.beat_count     = 0
                self.phase          = "ready_for_down"
                self.top_y          = self.bottom_y = wrist_y
            return self._build_output(now)

        # ── COUNTDOWN: accumulate pump beats 1-2-3 then lock AI and SHOOT ────
        if self.state == "COUNTDOWN":
            self._update_beat(wrist_y, confirmed, now)

            # Lock the AI's gesture in at beat 3 so it's decided before SHOOT
            if self.beat_count >= 3:
                self._lock_ai()

            # 4 beats = open the SHOOT window
            if self.beat_count >= 4:
                self.state             = "SHOOT_WINDOW"
                self.shoot_open_time   = now
                self.shoot_close_time  = now + self.SHOOT_WINDOW
                self.tracker_reset_requested = True

            return self._build_output(now)

        # ── SHOOT_WINDOW: read the player's throw ────────────────────────────
        if self.state == "SHOOT_WINDOW":
            # Try confirmed first (more reliable), then stable as fallback
            thrown = (confirmed if confirmed in VALID_RPSLS_SET
                      else (stable if stable in VALID_RPSLS_SET else None))

            # Only accept the gesture after MIN_SHOT_DWELL seconds in the window.
            # This gives the hand time to transition from Rock to Spock/Lizard
            # without the transit pose being read as the final gesture.
            elapsed_in_window = now - self.shoot_open_time
            gesture_ready     = thrown and elapsed_in_window >= MIN_SHOT_DWELL

            if now >= self.shoot_close_time or gesture_ready:
                self._resolve_round(thrown or "Rock", now)

            return self._build_output(now)

        return self._build_output(now)

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
             → detected as "Lizard" by front_on or dedicated curl logic
  Spock    — Vulcan salute: index+middle up, ring+pinky up, split between them
             → index+middle separated from ring+pinky by a gap

Because Lizard/Spock share landmark patterns with Scissors/Paper in
side-view, RPSLS works best in Front-On orientation with the trained MLP.
In side-view mode we use curl analysis heuristics.

The AI is the standard FairPlayAI extended to a 5-gesture action space.
"""

import time
import random
from fair_play_ai import FairPlayAI

VALID_RPSLS = ("Rock", "Spock", "Paper", "Lizard", "Scissors")
VALID_RPSLS_SET = frozenset(VALID_RPSLS)

# Maps each gesture to the two it beats, and the verb describing the win
BEATS = {
    "Scissors": ("Paper", "Lizard"),
    "Paper":    ("Rock",  "Spock"),
    "Rock":     ("Lizard","Scissors"),
    "Lizard":   ("Spock", "Paper"),
    "Spock":    ("Scissors","Rock"),
}

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

COUNTER_RPSLS = {}
for winner, losers in BEATS.items():
    for loser in losers:
        COUNTER_RPSLS[loser] = winner   # keeps last set; fine for symmetry


def compare_rpsls(p, c):
    """Returns 'win', 'lose', or 'draw' from player p vs computer c."""
    if p == c:
        return "draw"
    if c in BEATS.get(p, ()):
        return "win"
    return "lose"


def beat_verb(winner, loser):
    return BEAT_VERBS.get((winner, loser), f"{winner} beats {loser}")


INTRO_DURATION       = 1.20
ROUND_RESULT_SECONDS = 2.80   # extra time to read the verb
MATCH_RESULT_SECONDS = 2.40
SHOOT_WINDOW         = 1.20
BEAT_COOLDOWN        = 0.18
DOWN_THRESHOLD       = 0.045
UP_THRESHOLD         = 0.035
ROCK_GRACE           = 0.50


class RPSLSController:
    """
    RPSLS vs FairPlayAI, best-of-N (first to win_target wins).
    Uses the same pump-beat detection as FairPlayController.
    """

    def __init__(self, ai=None, win_target=3,
                 beat_cooldown=0.18, shoot_window_seconds=SHOOT_WINDOW,
                 robot_output=None):
        self.ai            = ai or FairPlayAI()
        self.win_target    = win_target
        self.BEAT_COOLDOWN = beat_cooldown
        self.SHOOT_WINDOW  = shoot_window_seconds
        self.robot_output  = robot_output
        self.reset_match()

    def reset(self):
        self.reset_match()

    def reset_match(self, now=None):
        if now is None:
            now = time.monotonic()
        self.player_score = 0
        self.robot_score  = 0
        self.round_number = 1
        self.history: list[dict] = []
        self.match_result_banner = ""
        self.match_until = None
        self.ai.reset()
        self._reset_round(now)

    def _reset_round(self, now=None):
        if now is None:
            now = time.monotonic()
        self.state          = "ROUND_INTRO"
        self.intro_until    = now + INTRO_DURATION
        self.beat_count     = 0
        self.phase          = "ready_for_down"
        self.top_y          = None
        self.bottom_y       = None
        self.last_beat_time = 0.0
        self.last_rock_time = 0.0
        self.shoot_open_time  = None
        self.shoot_close_time = None
        self.tracker_reset_requested = False
        self.ai_locked      = None
        self.player_gesture = "Unknown"
        self.ai_gesture     = "Unknown"
        self.result_banner  = ""
        self.result_verb    = ""
        self.last_round_result = None
        self.result_until   = None

    def consume_tracker_reset_request(self):
        self.tracker_reset_requested = False

    def _lock_ai(self):
        if self.ai_locked is not None:
            return
        # AI chooses from RPSLS space using adapted history
        # Build a mini-history the AI can read; map RPSLS outcomes to
        # standard win/lose/draw for the base FairPlayAI
        self.ai_locked = self._ai_choose()

    def _ai_choose(self):
        """
        Simple RPSLS AI: use frequency + transition patterns from history,
        then pick the gesture that beats the predicted player move.
        Falls back to FairPlayAI in the 3-gesture subspace when history is thin.
        """
        if len(self.history) < 4:
            return random.choice(VALID_RPSLS)

        # Frequency bias
        freq = {g: 0 for g in VALID_RPSLS}
        for r in self.history[-10:]:
            g = r.get("player_gesture")
            if g in freq:
                freq[g] += 1
        predicted = max(freq, key=freq.get)

        # Pick a move that beats the predicted gesture
        for winner, losers in BEATS.items():
            if predicted in losers:
                return winner
        return random.choice(VALID_RPSLS)

    def _update_beat(self, wrist_y, confirmed, now):
        rock_held = confirmed in ("Rock", "Spock")  # Spock also uses a fist-ish motion
        if rock_held:
            self.last_rock_time = now
        grace_ok    = (now - self.last_rock_time) <= ROCK_GRACE
        cooldown_ok = (now - self.last_beat_time) >= self.BEAT_COOLDOWN
        if not grace_ok or wrist_y is None:
            return
        if self.top_y is None:
            self.top_y = self.bottom_y = wrist_y
        self.top_y    = min(self.top_y, wrist_y)
        self.bottom_y = max(self.bottom_y, wrist_y)
        if self.phase == "ready_for_down":
            if cooldown_ok and (wrist_y - self.top_y) >= DOWN_THRESHOLD:
                self.phase = "waiting_for_up"
                self.bottom_y = wrist_y
                self.last_beat_time = now
                self.beat_count += 1
                self.top_y = wrist_y
        elif self.phase == "waiting_for_up":
            self.bottom_y = max(self.bottom_y, wrist_y)
            if (self.bottom_y - wrist_y) >= UP_THRESHOLD:
                self.phase = "ready_for_down"
                self.top_y = self.bottom_y = wrist_y

    def _resolve_round(self, player_g, now):
        self.player_gesture = player_g
        self.ai_gesture     = self.ai_locked or "Rock"
        outcome = compare_rpsls(player_g, self.ai_gesture)

        if outcome == "win":
            self.player_score += 1
            verb = beat_verb(player_g, self.ai_gesture)
            self.result_banner = "YOU WIN THE ROUND"
            self.result_verb   = verb
        elif outcome == "lose":
            self.robot_score += 1
            verb = beat_verb(self.ai_gesture, player_g)
            self.result_banner = "AI WINS THE ROUND"
            self.result_verb   = verb
        else:
            self.result_banner = "DRAW"
            self.result_verb   = "Same gesture — no winner"

        self.last_round_result = outcome
        self.history.append({
            "round_number":   self.round_number,
            "player_gesture": player_g,
            "player_outcome": outcome,
        })
        self.state        = "ROUND_RESULT"
        self.result_until = now + ROUND_RESULT_SECONDS

    def _round_is_over(self):
        return (self.player_score >= self.win_target or
                self.robot_score  >= self.win_target)

    def _build_output(self, now):
        tl = max(0.0, self.shoot_close_time - now) if self.shoot_close_time else 0.0
        main_text = {
            "ROUND_INTRO":     "GET READY",
            "WAITING_FOR_ROCK":"MAKE A FIST",
            "COUNTDOWN":       "READY" if self.beat_count == 0 else str(min(self.beat_count, 3)),
            "SHOOT_WINDOW":    "SHOOT!",
            "ROUND_RESULT":    self.result_banner,
            "MATCH_RESULT":    self.match_result_banner,
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
            "robot_move_command":    f"ROBOT_PLAY_{self.ai_locked.upper()}" if self.ai_locked else "PENDING",
            "two_player":            False,
            "opponent_type":         "",
        }

    def update(self, tracker_state, wrist_y=None, now=None):
        if now is None:
            now = time.monotonic()

        # Accept all 5 RPSLS gestures
        confirmed = tracker_state.get("confirmed_gesture", "Unknown")
        stable    = tracker_state.get("stable_gesture",   "Unknown")

        if self.state == "ROUND_INTRO":
            if now >= self.intro_until:
                self.state = "WAITING_FOR_ROCK"
            return self._build_output(now)

        if self.state == "ROUND_RESULT":
            if now >= self.result_until:
                if self._round_is_over():
                    winner = "YOU WIN!" if self.player_score >= self.win_target else "AI WINS"
                    self.match_result_banner = winner
                    self.state       = "MATCH_RESULT"
                    self.match_until = now + MATCH_RESULT_SECONDS
                else:
                    if self.last_round_result != "draw":
                        self.round_number += 1
                    self._reset_round(now)
            return self._build_output(now)

        if self.state == "MATCH_RESULT":
            if now >= self.match_until:
                self.reset_match(now)
            return self._build_output(now)

        if self.state == "WAITING_FOR_ROCK":
            if confirmed in ("Rock", "Spock") or stable in ("Rock", "Spock"):
                self.last_rock_time = now
                self.state = "COUNTDOWN"
                self.beat_count = 0
                self.phase = "ready_for_down"
                self.top_y = self.bottom_y = wrist_y
            return self._build_output(now)

        if self.state == "COUNTDOWN":
            self._update_beat(wrist_y, confirmed, now)
            if self.beat_count >= 3:
                self._lock_ai()
            if self.beat_count >= 4:
                self.state             = "SHOOT_WINDOW"
                self.shoot_open_time   = now
                self.shoot_close_time  = now + self.SHOOT_WINDOW
                self.tracker_reset_requested = True
            return self._build_output(now)

        if self.state == "SHOOT_WINDOW":
            thrown = confirmed if confirmed in VALID_RPSLS_SET else \
                     (stable    if stable    in VALID_RPSLS_SET else None)
            # Require the gesture to be held for at least MIN_SHOT_DWELL seconds
            # before accepting, so the hand has time to transition from Rock
            # to Spock / Lizard without the window closing on the transit pose.
            MIN_SHOT_DWELL = 0.25
            elapsed_in_window = now - self.shoot_open_time
            gesture_ready = thrown and elapsed_in_window >= MIN_SHOT_DWELL
            if now >= self.shoot_close_time or gesture_ready:
                self._resolve_round(thrown or "Rock", now)
            return self._build_output(now)

        return self._build_output(now)

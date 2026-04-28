"""
arcade_snake_state.py
=====================
Gesture-controlled Snake game.

Rock  = straight (neutral)
Scissors = turn left
Paper = turn right

Persistent high score saved to ~/Desktop/CapStone/snake_highscore.json
"""

import time
import json
import random
from collections import deque, Counter
from pathlib import Path

GRID_W      = 20
GRID_H      = 15
TICK_SECS   = 0.10
VOTE_FRAMES = 5
SCORE_PATH  = Path.home() / "Desktop" / "CapStone" / "snake_highscore.json"

RIGHT = ( 1,  0)
LEFT  = (-1,  0)
UP    = ( 0, -1)
DOWN  = ( 0,  1)

TURN_LEFT  = {RIGHT: UP,   UP: LEFT,   LEFT: DOWN,  DOWN: RIGHT}
TURN_RIGHT = {RIGHT: DOWN, DOWN: LEFT, LEFT: UP,    UP: RIGHT}
_OPPOSITE  = {RIGHT: LEFT, LEFT: RIGHT, UP: DOWN,   DOWN: UP}


def _load_high_score():
    try:
        data = json.loads(SCORE_PATH.read_text())
        return int(data.get("high_score", 0)), data.get("entries", [])
    except Exception:
        return 0, []


def _save_high_score(score, entries):
    try:
        SCORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        SCORE_PATH.write_text(json.dumps(
            {"high_score": score, "entries": entries}, indent=2))
    except Exception:
        pass


class ArcadeSnakeController:

    def __init__(self):
        self._persistent_hs, self._leaderboard = _load_high_score()
        self._session_score = 0   # score from most recent completed game
        self._is_new_record = False
        self._full_reset()

    def _full_reset(self):
        """Reset everything except persistent high score."""
        self.state            = "INTRO"
        self.score            = 0
        self._snake           = deque()
        self._direction       = RIGHT
        self._apple           = (0, 0)
        self._last_tick       = 0.0
        self._last_gesture    = "Unknown"
        self._voted_gesture   = "Unknown"
        self._game_over_until = 0.0
        self._vote_buffer     = deque(maxlen=VOTE_FRAMES)
        self._turn_used       = False
        self._init_snake()

    def reset(self):
        """Called by main loop reset_all_modes — preserve persistent hs."""
        self._persistent_hs, self._leaderboard = _load_high_score()
        self._session_score = 0
        self._is_new_record = False
        self._full_reset()

    def _init_snake(self):
        cx, cy = GRID_W // 2, GRID_H // 2
        self._snake = deque([(cx, cy), (cx - 1, cy), (cx - 2, cy)])
        self._direction = RIGHT
        self._place_apple()

    def _place_apple(self):
        occupied = set(self._snake)
        free = [(x, y) for x in range(GRID_W) for y in range(GRID_H)
                if (x, y) not in occupied]
        if free:
            self._apple = random.choice(free)

    def _record_score(self, score):
        """Persist score and update leaderboard (top 5)."""
        self._is_new_record = score > self._persistent_hs
        if self._is_new_record:
            self._persistent_hs = score
        # Add to leaderboard entries
        import datetime
        entry = {"score": score, "date": datetime.date.today().isoformat()}
        self._leaderboard.append(entry)
        # Keep top 5 unique scores
        self._leaderboard = sorted(
            self._leaderboard, key=lambda e: -e["score"])[:5]
        _save_high_score(self._persistent_hs, self._leaderboard)

    def _resolve_voted_gesture(self):
        if not self._vote_buffer:
            return "Unknown"
        counts = Counter(self._vote_buffer)
        real = {g: c for g, c in counts.items() if g != "Unknown"}
        if not real:
            return "Unknown"
        top_gest, top_count = max(real.items(), key=lambda x: x[1])
        threshold = max(2, len(self._vote_buffer) * 0.45)
        return top_gest if top_count >= threshold else "Unknown"

    def _build_output(self):
        return {
            "play_mode_label": "Gesture Snake",
            "state":           self.state,
            "snake":           list(self._snake),
            "apple":           self._apple,
            "direction":       self._direction,
            "score":           self.score,
            "high_score":      self._persistent_hs,
            "session_score":   self._session_score,
            "is_new_record":   self._is_new_record,
            "leaderboard":     list(self._leaderboard),
            "grid_w":          GRID_W,
            "grid_h":          GRID_H,
            "last_gesture":    self._last_gesture,
            "voted_gesture":   self._voted_gesture,
        }

    def update(self, tracker_state, now=None):
        if now is None:
            now = time.monotonic()

        confirmed = tracker_state.get("confirmed_gesture", "Unknown")
        stable    = tracker_state.get("stable_gesture", "Unknown")
        raw = (confirmed if confirmed in ("Rock", "Paper", "Scissors")
               else (stable if stable in ("Rock", "Paper", "Scissors")
                     else "Unknown"))

        self._vote_buffer.append(raw)
        if raw in ("Rock", "Paper", "Scissors"):
            self._last_gesture = raw
        self._voted_gesture = self._resolve_voted_gesture()

        # ── INTRO ──────────────────────────────────────────────────────────
        if self.state == "INTRO":
            # Wait for Rock to start
            if self._voted_gesture == "Rock":
                self.state      = "PLAYING"
                self._last_tick = now
                self._turn_used = True   # prevent immediate turn on first tick
            return self._build_output()

        # ── GAME OVER ──────────────────────────────────────────────────────
        if self.state == "GAME_OVER":
            # Wait for Rock to restart
            if self._voted_gesture == "Rock":
                self._full_reset()
                self._is_new_record = False
                self.state      = "PLAYING"
                self._last_tick = now
                self._turn_used = True
            return self._build_output()

        # ── PLAYING ────────────────────────────────────────────────────────
        if self.state == "PLAYING":
            voted = self._voted_gesture

            if voted == "Rock":
                self._turn_used = False

            if now - self._last_tick >= TICK_SECS:
                self._last_tick = now

                if not self._turn_used:
                    if voted == "Scissors":
                        new_dir = TURN_LEFT[self._direction]
                        if new_dir != _OPPOSITE[self._direction]:
                            self._direction = new_dir
                        self._turn_used = True
                    elif voted == "Paper":
                        new_dir = TURN_RIGHT[self._direction]
                        if new_dir != _OPPOSITE[self._direction]:
                            self._direction = new_dir
                        self._turn_used = True

                head     = self._snake[0]
                nx       = (head[0] + self._direction[0]) % GRID_W
                ny       = (head[1] + self._direction[1]) % GRID_H
                new_head = (nx, ny)

                if new_head in self._snake:
                    self._session_score = self.score
                    self._record_score(self.score)
                    self.state = "GAME_OVER"
                    return self._build_output()

                self._snake.appendleft(new_head)
                if new_head == self._apple:
                    self.score += 10
                    self._place_apple()
                else:
                    self._snake.pop()

        return self._build_output()

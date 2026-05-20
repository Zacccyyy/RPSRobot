"""
arcade_snake_state.py
=====================
Gesture-controlled Snake game — the classic Nokia game but you steer
with your hand instead of a d-pad.

Gesture → action mapping:
  Rock     = go straight (no turn, neutral hold)
  Scissors = turn left relative to current direction
  Paper    = turn right relative to current direction

The snake moves on a regular tick that gets faster as the score increases.
Eating an apple scores +10 and places a new one at a random empty cell.
Hitting your own body ends the game.

The high score is saved to ~/Desktop/CapStone/snake_highscore.json.
Writes are done on a background thread so the camera loop never stalls.

This module owns only game logic — all drawing is done by draw_snake_view()
in the renderer.
"""

import time
import json
import random
import datetime
import threading
from collections import deque, Counter
from pathlib import Path

# ── Grid and timing constants ─────────────────────────────────────────────────
GRID_W      = 20    # number of columns
GRID_H      = 15    # number of rows
TICK_SECS   = 0.10  # base time between snake moves in seconds (calibrated)
VOTE_FRAMES = 5     # how many frames to majority-vote before committing a gesture
SCORE_PATH  = Path.home() / "Desktop" / "CapStone" / "snake_highscore.json"

# Direction vectors — (dx, dy) where y increases downward (screen coordinates)
RIGHT = ( 1,  0)
LEFT  = (-1,  0)
UP    = ( 0, -1)
DOWN  = ( 0,  1)

# Given the current direction, what direction is a left or right turn?
TURN_LEFT  = {RIGHT: UP,   UP: LEFT,   LEFT: DOWN,  DOWN: RIGHT}
TURN_RIGHT = {RIGHT: DOWN, DOWN: LEFT, LEFT: UP,    UP: RIGHT}

# Used to prevent the snake from immediately reversing into itself
_OPPOSITE  = {RIGHT: LEFT, LEFT: RIGHT, UP: DOWN, DOWN: UP}


# ── Highscore helpers ─────────────────────────────────────────────────────────

def _load_high_score():
    """
    Read the highscore JSON file from disk.
    Returns (high_score_int, entries_list).
    Falls back to (0, []) on any error so the game works without the file.
    """
    try:
        data = json.loads(SCORE_PATH.read_text())
        return int(data.get("high_score", 0)), data.get("entries", [])
    except Exception:
        return 0, []


def _save_high_score(score, entries):
    """
    Write the highscore JSON to disk.
    Called from a daemon thread so it never blocks the main camera loop.
    Silently ignores errors (e.g. permission issues on the Desktop path).
    """
    try:
        SCORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        SCORE_PATH.write_text(json.dumps(
            {"high_score": score, "entries": entries}, indent=2))
    except Exception:
        pass


# ── Main controller ───────────────────────────────────────────────────────────

class ArcadeSnakeController:
    """
    Manages the full Snake game state machine.

    States: INTRO → PLAYING → GAME_OVER → (PLAYING again on Rock)

    Call update(tracker_state) every frame.
    Use the returned dict to drive the renderer.
    """

    def __init__(self):
        # Load the persistent high score from disk on startup
        self._persistent_hs, self._leaderboard = _load_high_score()
        self._session_score = 0     # score from the most recently completed game
        self._is_new_record = False
        self._full_reset()

    def _full_reset(self):
        """Reset all per-game state. Does NOT touch the persistent high score."""
        self.state          = "INTRO"
        self.score          = 0
        self._snake         = deque()       # deque of (x, y) cells, head at index 0
        self._direction     = RIGHT
        self._apple         = (0, 0)
        self._last_tick     = 0.0           # timestamp of the last snake move
        self._last_gesture  = "Unknown"     # most recent valid gesture seen
        self._voted_gesture = "Unknown"     # majority-vote result over last VOTE_FRAMES
        self._game_over_until = 0.0
        self._vote_buffer   = deque(maxlen=VOTE_FRAMES)  # rolling gesture window
        self._turn_used     = False         # prevents multiple turns per single tick
        self._init_snake()

    def reset(self):
        """
        Called by the main loop's reset_all_modes().
        Reloads the high score from disk first (in case another session changed it),
        then does a full game reset.
        """
        self._persistent_hs, self._leaderboard = _load_high_score()
        self._session_score = 0
        self._is_new_record = False
        self._full_reset()

    def _init_snake(self):
        """Spawn a 3-cell snake in the centre of the grid, heading right."""
        cx, cy = GRID_W // 2, GRID_H // 2
        self._snake = deque([(cx, cy), (cx - 1, cy), (cx - 2, cy)])
        self._direction = RIGHT
        self._place_apple()

    def _place_apple(self):
        """Pick a random empty cell for the apple. Does nothing if the grid is full."""
        occupied = set(self._snake)
        free = [(x, y) for x in range(GRID_W) for y in range(GRID_H)
                if (x, y) not in occupied]
        if free:
            self._apple = random.choice(free)

    def _record_score(self, score):
        """
        Update the in-memory leaderboard and flush it to disk in the background.
        Keeps only the top 5 entries, sorted by score descending.
        """
        self._is_new_record = score > self._persistent_hs
        if self._is_new_record:
            self._persistent_hs = score

        # Add today's run to the leaderboard, sort, and trim to top 5
        self._leaderboard.append({"score": score, "date": datetime.date.today().isoformat()})
        self._leaderboard = sorted(self._leaderboard, key=lambda e: -e["score"])[:5]

        # Write to disk in a background daemon thread — won't block the camera loop
        _hs, _lb = self._persistent_hs, list(self._leaderboard)
        threading.Thread(target=_save_high_score, args=(_hs, _lb), daemon=True).start()

    def _resolve_voted_gesture(self):
        """
        Majority-vote over the last VOTE_FRAMES readings to decide a stable gesture.

        A gesture must appear in at least 45% of frames (or at least 2 frames,
        whichever is larger) to win the vote. This prevents brief flickering
        between gestures from causing unintended turns.

        Returns the winning gesture string, or "Unknown" if nothing clears the bar.
        """
        if not self._vote_buffer:
            return "Unknown"

        counts = Counter(self._vote_buffer)

        # Exclude "Unknown" — those frames just represent no clear detection
        real = {g: c for g, c in counts.items() if g != "Unknown"}
        if not real:
            return "Unknown"

        top_gest, top_count = max(real.items(), key=lambda x: x[1])
        threshold = max(2, len(self._vote_buffer) * 0.45)  # at least 45% or 2 frames
        return top_gest if top_count >= threshold else "Unknown"

    def _build_output(self):
        """
        Snapshot all game state into a flat dict for the renderer.
        tick_secs is recalculated here so the renderer can show the current speed.
        """
        return {
            "play_mode_label": "Gesture Snake",
            "state":           self.state,
            "snake":           list(self._snake),  # copy — renderer must not mutate this
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
            # Speed increases by 0.0002 s per point (snake moves faster as score grows)
            "tick_secs":       max(0.055, TICK_SECS - self.score * 0.00020),
        }

    def update(self, tracker_state, now=None):
        """
        Main tick — call once per frame.

        tracker_state : dict from the gesture tracker. We read both
                        "confirmed_gesture" and "stable_gesture" for the
                        widest possible input signal.
        now           : optional monotonic timestamp (injectable for testing).
        """
        if now is None:
            now = time.monotonic()

        # Pull both confidence levels from the tracker
        confirmed = tracker_state.get("confirmed_gesture", "Unknown")
        stable    = tracker_state.get("stable_gesture",    "Unknown")

        # Use confirmed first (highest confidence), fall back to stable,
        # then "Unknown" if neither gave a valid RPS gesture
        raw = (confirmed if confirmed in ("Rock", "Paper", "Scissors")
               else (stable if stable in ("Rock", "Paper", "Scissors")
                     else "Unknown"))

        # Add the raw reading to the rolling vote buffer every frame
        self._vote_buffer.append(raw)
        if raw in ("Rock", "Paper", "Scissors"):
            self._last_gesture = raw  # keep the HUD showing the last valid gesture

        # Re-evaluate the majority vote
        self._voted_gesture = self._resolve_voted_gesture()

        # ── INTRO: wait for Rock to start the game ────────────────────────────
        if self.state == "INTRO":
            if self._voted_gesture == "Rock":
                self.state      = "PLAYING"
                self._last_tick = now
                # Mark the turn as used so the first tick doesn't register a spurious turn
                self._turn_used = True
            return self._build_output()

        # ── GAME_OVER: wait for Rock to restart ──────────────────────────────
        if self.state == "GAME_OVER":
            if self._voted_gesture == "Rock":
                self._full_reset()
                self._is_new_record = False
                self.state      = "PLAYING"
                self._last_tick = now
                self._turn_used = True
            return self._build_output()

        # ── PLAYING: move the snake on each tick ─────────────────────────────
        if self.state == "PLAYING":
            voted = self._voted_gesture

            # Rock = go straight — clear the turn flag so the player can turn again
            # once they switch to Scissors or Paper
            if voted == "Rock":
                self._turn_used = False

            # How long between snake moves (gets shorter as score increases)
            tick = max(0.055, TICK_SECS - self.score * 0.00020)

            if now - self._last_tick >= tick:
                self._last_tick = now

                # Apply at most one turn per tick to prevent double-turns
                if not self._turn_used:
                    if voted == "Scissors":
                        new_dir = TURN_LEFT[self._direction]
                        # Safety check: never let the snake reverse directly into itself
                        if new_dir != _OPPOSITE[self._direction]:
                            self._direction = new_dir
                        self._turn_used = True
                    elif voted == "Paper":
                        new_dir = TURN_RIGHT[self._direction]
                        if new_dir != _OPPOSITE[self._direction]:
                            self._direction = new_dir
                        self._turn_used = True

                # Calculate the new head position, wrapping around grid edges
                head     = self._snake[0]
                nx       = (head[0] + self._direction[0]) % GRID_W
                ny       = (head[1] + self._direction[1]) % GRID_H
                new_head = (nx, ny)

                # Colliding with own body = game over
                if new_head in self._snake:
                    self._session_score = self.score
                    self._record_score(self.score)
                    self.state = "GAME_OVER"
                    return self._build_output()

                # Move the snake: add new head...
                self._snake.appendleft(new_head)

                if new_head == self._apple:
                    # Ate the apple — grow the snake (don't remove the tail) and score
                    self.score += 10
                    self._place_apple()
                else:
                    # Normal move — remove the tail to keep length constant
                    self._snake.pop()

        return self._build_output()

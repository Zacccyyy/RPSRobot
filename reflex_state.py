"""
reflex_state.py
===============
Speed Reflex game mode — two controllers:

  ReflexSoloController
    - Random target gesture flashes on screen
    - Player must match it as fast as possible
    - 30-second sprint, score = number of hits
    - No penalty for misses; timeout after 3s = miss, next target

  ReflexTwoPlayerController
    - Same shared target displayed centre-screen
    - First player to correctly match the gesture wins the point
    - First to win_target (default 10) wins the match
"""

import time
import random
from reflex_highscore_store import ReflexHighscoreStore

VALID_GESTURES = ["Rock", "Paper", "Scissors"]

SOLO_DURATION       = 30.0   # seconds total
TARGET_TIMEOUT      = 3.0    # seconds before a target times out (miss)
RESULT_FLASH_SECS   = 0.55   # brief flash before next target
INTRO_SECS          = 2.0
GAME_OVER_SECS      = 4.0    # show final score before resetting

TWO_PLAYER_TARGET   = 10     # first to this many wins the match
RESULT_FLASH_2P     = 0.70


# ─────────────────────────────────────────────────────────────────────────────
# Solo Controller
# ─────────────────────────────────────────────────────────────────────────────

class ReflexSoloController:
    """
    30-second sprint.  Target appears → player matches → score+1 → next target.
    Timeout after TARGET_TIMEOUT seconds = miss, next target immediately.

    update() signature:
        controller.update(tracker_state=..., now=..., player_name="")

    Returns a dict consumed by draw_reflex_solo_view().
    """

    def __init__(self):
        self._hs_store = ReflexHighscoreStore()
        self.reset()

    def reset(self):
        self.state           = "INTRO"
        self.target          = ""
        self.score           = 0
        self.misses          = 0
        self.reaction_times  = []
        self.target_shown    = 0.0
        self.result_until    = 0.0
        self.game_over_until = 0.0
        self.last_result     = ""
        self.last_rt_ms      = 0
        self.game_end_time   = 0.0
        self._intro_until    = time.monotonic() + INTRO_SECS
        # Highscore result for this run (set on game over)
        self._is_new_best    = False
        self._run_rank       = 0
        self._next_target()

    def _next_target(self):
        self.target       = random.choice(VALID_GESTURES)
        self.target_shown = time.monotonic()

    def _avg_rt(self):
        if not self.reaction_times:
            return 0
        return int(sum(self.reaction_times) / len(self.reaction_times))

    def _build_output(self, now):
        time_left = 0.0
        if self.game_end_time > 0:
            time_left = max(0.0, self.game_end_time - now)
        best = self._hs_store.get_best()
        return {
            "play_mode_label":  "Speed Reflex",
            "state":            self.state,
            "target":           self.target,
            "score":            self.score,
            "misses":           self.misses,
            "time_left":        time_left,
            "last_result":      self.last_result,
            "last_rt_ms":       self.last_rt_ms,
            "avg_reaction_ms":  self._avg_rt(),
            "two_player":       False,
            # Highscore fields
            "best_score":       best["score"] if best else 0,
            "best_player":      best["player"] if best else "",
            "best_avg_rt":      best["avg_rt"] if best else 0,
            "is_new_best":      self._is_new_best,
            "run_rank":         self._run_rank,
            "top_scores":       self._hs_store.get_top(),
        }

    def update(self, tracker_state, now=None, player_name=""):
        if now is None:
            now = time.monotonic()

        confirmed = tracker_state.get("confirmed_gesture", "Unknown")

        # ── INTRO ──────────────────────────────────────────────────────────
        if self.state == "INTRO":
            if now >= self._intro_until:
                self.state        = "PLAYING"
                self.game_end_time = now + SOLO_DURATION
                self._next_target()
            return self._build_output(now)

        # ── GAME_OVER ──────────────────────────────────────────────────────
        if self.state == "GAME_OVER":
            # Stay on results screen until player presses Enter
            return self._build_output(now)

        # ── RESULT_FLASH ───────────────────────────────────────────────────
        if self.state == "RESULT_FLASH":
            if now >= self.result_until:
                if now >= self.game_end_time:
                    self._submit_score(player_name)
                    self.state           = "GAME_OVER"
                    self.game_over_until = now + GAME_OVER_SECS
                else:
                    self.state = "PLAYING"
                    self._next_target()
            return self._build_output(now)

        # ── PLAYING ────────────────────────────────────────────────────────
        if self.state == "PLAYING":
            if now >= self.game_end_time:
                self._submit_score(player_name)
                self.state           = "GAME_OVER"
                self.game_over_until = now + GAME_OVER_SECS
                return self._build_output(now)

            elapsed = now - self.target_shown
            if elapsed >= TARGET_TIMEOUT:
                self.misses      += 1
                self.last_result  = "timeout"
                self.last_rt_ms   = 0
                self.state        = "RESULT_FLASH"
                self.result_until = now + RESULT_FLASH_SECS
                return self._build_output(now)

            if confirmed == self.target:
                rt_ms = int((now - self.target_shown) * 1000)
                self.reaction_times.append(rt_ms)
                self.score       += 1
                self.last_result  = "hit"
                self.last_rt_ms   = rt_ms
                self.state        = "RESULT_FLASH"
                self.result_until = now + RESULT_FLASH_SECS
                return self._build_output(now)

        return self._build_output(now)

    def _submit_score(self, player_name: str):
        """Submit the completed run to the highscore store."""
        name = (player_name or "Unknown").strip()
        self._is_new_best, self._run_rank = self._hs_store.submit(
            player_name=name,
            score=self.score,
            avg_rt_ms=self._avg_rt(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Two-Player Controller
# ─────────────────────────────────────────────────────────────────────────────

class ReflexTwoPlayerController:
    """
    Shared target centre-screen.  First player to match the gesture scores.
    First to win_target wins the match.

    update() signature:
        controller.update(p1_tracker=..., p2_tracker=..., now=...)

    Returns a dict consumed by draw_reflex_two_player_view().
    """

    def __init__(self, win_target=TWO_PLAYER_TARGET):
        self.win_target = win_target
        self.reset()

    def reset(self):
        self.state          = "INTRO"
        self.target         = ""
        self.p1_score       = 0
        self.p2_score       = 0
        self.last_winner    = ""
        self.match_winner   = ""
        self.result_until   = 0.0
        self.target_shown   = 0.0
        self._intro_until   = time.monotonic() + INTRO_SECS
        self._next_target()

    def _next_target(self):
        self.target       = random.choice(VALID_GESTURES)
        self.target_shown = time.monotonic()
        self.last_winner  = ""

    def _build_output(self, now):
        tl = max(0.0, TARGET_TIMEOUT - (now - self.target_shown)) \
             if self.state == "PLAYING" else 0.0
        return {
            "play_mode_label": "Reflex Race",
            "state":           self.state,
            "target":          self.target,
            "p1_score":        self.p1_score,
            "p2_score":        self.p2_score,
            "win_target":      self.win_target,
            "last_winner":     self.last_winner,
            "match_winner":    self.match_winner,
            "time_left":       tl,
            "two_player":      True,
        }

    def update(self, p1_tracker, p2_tracker, now=None):
        if now is None:
            now = time.monotonic()

        p1_confirmed = p1_tracker.get("confirmed_gesture", "Unknown")
        p2_confirmed = p2_tracker.get("confirmed_gesture", "Unknown")

        # ── INTRO ──────────────────────────────────────────────────────────
        if self.state == "INTRO":
            if now >= self._intro_until:
                self.state = "PLAYING"
                self._next_target()
            return self._build_output(now)

        # ── MATCH_OVER ─────────────────────────────────────────────────────
        if self.state == "MATCH_OVER":
            if now >= self.result_until:
                self.reset()
            return self._build_output(now)

        # ── RESULT_FLASH ───────────────────────────────────────────────────
        if self.state == "RESULT_FLASH":
            if now >= self.result_until:
                if self.p1_score >= self.win_target or self.p2_score >= self.win_target:
                    winner = "P1 WINS!" if self.p1_score >= self.win_target else "P2 WINS!"
                    self.match_winner = winner
                    self.state        = "MATCH_OVER"
                    self.result_until = now + 4.0
                else:
                    self.state = "PLAYING"
                    self._next_target()
            return self._build_output(now)

        # ── PLAYING ────────────────────────────────────────────────────────
        if self.state == "PLAYING":
            # Target timeout — no winner, just next target
            elapsed = now - self.target_shown
            if elapsed >= TARGET_TIMEOUT:
                self.state        = "RESULT_FLASH"
                self.result_until = now + RESULT_FLASH_2P
                self.last_winner  = "—"
                return self._build_output(now)

            # Check both players simultaneously; P1 wins ties
            p1_hit = (p1_confirmed == self.target)
            p2_hit = (p2_confirmed == self.target)

            if p1_hit or p2_hit:
                if p1_hit:
                    self.p1_score    += 1
                    self.last_winner  = "P1"
                else:
                    self.p2_score    += 1
                    self.last_winner  = "P2"
                self.state        = "RESULT_FLASH"
                self.result_until = now + RESULT_FLASH_2P

        return self._build_output(now)

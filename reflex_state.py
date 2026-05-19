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

This file lives entirely in game-logic space — it owns no camera or
drawing code.  The main loop calls update() every frame and passes the
returned dict straight to the appropriate draw_reflex_*_view() function.
"""

import time
import random
from reflex_highscore_store import ReflexHighscoreStore

# The only three gestures the camera tracker can confirm
VALID_GESTURES = ["Rock", "Paper", "Scissors"]

# ── Timing constants — DO NOT change, these are calibrated values ─────────────
SOLO_DURATION       = 30.0   # seconds total for one solo sprint
TARGET_TIMEOUT      = 3.0    # seconds before an unmatched target counts as a miss
RESULT_FLASH_SECS   = 0.55   # brief "HIT / MISS" flash before the next target appears
INTRO_SECS          = 2.0    # countdown intro shown before the sprint begins
GAME_OVER_SECS      = 4.0    # show final score before resetting (unused directly, kept for reference)

TWO_PLAYER_TARGET   = 10     # first player to this score wins the 2P match
RESULT_FLASH_2P     = 0.70   # slightly longer flash for 2P so both players can register what happened


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
        # Load the persistent highscore store so we can compare at game-over
        self._hs_store = ReflexHighscoreStore()
        self.reset()

    def reset(self):
        """Wipe all per-run state and go back to the INTRO countdown."""
        self.state           = "INTRO"
        self.target          = ""          # current gesture the player must match
        self.score           = 0
        self.misses          = 0
        self.reaction_times  = []          # list of per-hit reaction times in ms
        self.target_shown    = 0.0         # monotonic timestamp when current target appeared
        self.result_until    = 0.0         # timestamp when the RESULT_FLASH ends
        self.last_result     = ""          # "hit" or "timeout" — drives the flash label
        self.last_rt_ms      = 0           # reaction time for the most recent hit
        self.game_end_time   = 0.0         # timestamp when the 30-second sprint ends
        self._intro_until    = time.monotonic() + INTRO_SECS
        # Highscore metadata set at game-over — used by the results screen
        self._is_new_best    = False
        self._run_rank       = 0
        self._next_target()

    def _next_target(self):
        """Pick a new random gesture and record when it was shown."""
        self.target       = random.choice(VALID_GESTURES)
        self.target_shown = time.monotonic()

    def _avg_rt(self):
        """Return average reaction time in milliseconds, or 0 if no hits yet."""
        if not self.reaction_times:
            return 0
        return int(sum(self.reaction_times) / len(self.reaction_times))

    def _build_output(self, now):
        """
        Package the current game state into a flat dict for the renderer.
        Called at the end of every update() branch so the view always gets
        a consistent snapshot — never a partial update.
        """
        # Time remaining is only meaningful while the sprint is active
        time_left = 0.0
        if self.game_end_time > 0:
            time_left = max(0.0, self.game_end_time - now)

        # Pull the all-time best from disk so the results screen is up to date
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
            # Highscore fields (renderer shows these on the GAME_OVER screen)
            "best_score":       best["score"]  if best else 0,
            "best_player":      best["player"] if best else "",
            "best_avg_rt":      best["avg_rt"] if best else 0,
            "is_new_best":      self._is_new_best,
            "run_rank":         self._run_rank,
            "top_scores":       self._hs_store.get_top(),
        }

    def update(self, tracker_state, now=None, player_name=""):
        """
        Main tick function — call once per frame.

        tracker_state  : dict from the gesture tracker (needs "confirmed_gesture")
        now            : optional monotonic timestamp (uses time.monotonic() if omitted)
        player_name    : used when submitting the run to the highscore store
        """
        if now is None:
            now = time.monotonic()

        # The tracker's confirmed gesture — "Unknown" when nothing is detected
        confirmed = tracker_state.get("confirmed_gesture", "Unknown")

        # ── INTRO: show a countdown splash before the sprint starts ──────────
        if self.state == "INTRO":
            if now >= self._intro_until:
                # Intro finished — start the clock and show the first target
                self.state         = "PLAYING"
                self.game_end_time = now + SOLO_DURATION
                self._next_target()
            return self._build_output(now)

        # ── GAME_OVER: show results screen, wait for the player to press Enter
        if self.state == "GAME_OVER":
            # Nothing to do here — the main loop handles the Enter key
            return self._build_output(now)

        # ── RESULT_FLASH: brief pause after a hit or timeout ─────────────────
        if self.state == "RESULT_FLASH":
            if now >= self.result_until:
                # Flash is done — check if the game is also over
                if now >= self.game_end_time:
                    self._submit_score(player_name)
                    self.state = "GAME_OVER"
                else:
                    # Game still running — show the next target
                    self.state = "PLAYING"
                    self._next_target()
            return self._build_output(now)

        # ── PLAYING: the core reflex loop ─────────────────────────────────────
        if self.state == "PLAYING":
            # Check if time has run out first (avoids processing an extra gesture)
            if now >= self.game_end_time:
                self._submit_score(player_name)
                self.state           = "GAME_OVER"
                self.game_over_until = now + GAME_OVER_SECS
                return self._build_output(now)

            elapsed = now - self.target_shown

            # Target timed out — the player was too slow, count a miss
            if elapsed >= TARGET_TIMEOUT:
                self.misses      += 1
                self.last_result  = "timeout"
                self.last_rt_ms   = 0
                self.state        = "RESULT_FLASH"
                self.result_until = now + RESULT_FLASH_SECS
                return self._build_output(now)

            # Player matched the target gesture — count a hit and record how fast
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
        """Submit the completed run to the highscore store and save rank info."""
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
        """Wipe all match state and go back to INTRO."""
        self.state          = "INTRO"
        self.target         = ""
        self.p1_score       = 0
        self.p2_score       = 0
        self.last_winner    = ""     # "P1", "P2", or "NONE" (timeout)
        self.match_winner   = ""     # "P1 WINS!" or "P2 WINS!" — set at match end
        self.result_until   = 0.0
        self.target_shown   = 0.0
        self._intro_until   = time.monotonic() + INTRO_SECS
        self._next_target()

    def _next_target(self):
        """Pick a new random target and clear the last-winner label."""
        self.target       = random.choice(VALID_GESTURES)
        self.target_shown = time.monotonic()
        self.last_winner  = ""

    def _build_output(self, now):
        """
        Package current state for the renderer.
        time_left counts down from TARGET_TIMEOUT only while actively playing.
        """
        # Only show a time countdown during the active PLAYING state
        if self.state == "PLAYING":
            tl = max(0.0, TARGET_TIMEOUT - (now - self.target_shown))
        else:
            tl = 0.0

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
        """
        Main tick — called once per frame with both players' tracker states.
        P1 wins ties (both players match on the same frame).
        """
        if now is None:
            now = time.monotonic()

        # Pull the latest confirmed gesture for each player
        p1_confirmed = p1_tracker.get("confirmed_gesture", "Unknown")
        p2_confirmed = p2_tracker.get("confirmed_gesture", "Unknown")

        # ── INTRO: short splash before the first target appears ──────────────
        if self.state == "INTRO":
            if now >= self._intro_until:
                self.state = "PLAYING"
                self._next_target()
            return self._build_output(now)

        # ── MATCH_OVER: show the winner for a few seconds then auto-reset ────
        if self.state == "MATCH_OVER":
            if now >= self.result_until:
                self.reset()
            return self._build_output(now)

        # ── RESULT_FLASH: brief pause after each point is scored ─────────────
        if self.state == "RESULT_FLASH":
            if now >= self.result_until:
                # Check if either player has reached the win target
                if self.p1_score >= self.win_target or self.p2_score >= self.win_target:
                    winner = "P1 WINS!" if self.p1_score >= self.win_target else "P2 WINS!"
                    self.match_winner = winner
                    self.state        = "MATCH_OVER"
                    self.result_until = now + 4.0
                else:
                    # Match is still going — show the next target
                    self.state = "PLAYING"
                    self._next_target()
            return self._build_output(now)

        # ── PLAYING: check both players each frame ────────────────────────────
        if self.state == "PLAYING":
            # Nobody matched in time — skip to the next target
            elapsed = now - self.target_shown
            if elapsed >= TARGET_TIMEOUT:
                self.state        = "RESULT_FLASH"
                self.result_until = now + RESULT_FLASH_2P
                self.last_winner  = "NONE"
                return self._build_output(now)

            # Evaluate both players on the same frame; P1 wins on a tie
            p1_hit = (p1_confirmed == self.target)
            p2_hit = (p2_confirmed == self.target)

            if p1_hit or p2_hit:
                # Award the point and enter the flash state
                if p1_hit:
                    self.p1_score    += 1
                    self.last_winner  = "P1"
                else:
                    self.p2_score    += 1
                    self.last_winner  = "P2"
                self.state        = "RESULT_FLASH"
                self.result_until = now + RESULT_FLASH_2P

        return self._build_output(now)

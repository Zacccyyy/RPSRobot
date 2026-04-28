"""
simon_says_state.py
===================
Gesture Simon Says — two controllers:

  SimonSaysSoloController
    - System generates a sequence of gestures
    - Player must reproduce each gesture in order within TIME_PER_STEP seconds
    - After 2s the current gesture locks in (whatever is showing)
    - Correct sequence → next round with one more gesture added
    - Wrong gesture at any step → GAME_OVER
    - Starts with 3 gestures; grows by 1 each successful round

  SimonSaysTwoPlayerController
    - P1 shows a gesture — locked after TIME_PER_STEP seconds
    - P2 must copy it
    - Then P2 adds a new gesture to the growing chain
    - P1 must replay the full sequence
    - One wrong step = game over for that player
    - Score = longest sequence completed before failing
"""

import time
import random
from simon_highscore_store import SimonHighscoreStore

VALID_GESTURES   = ["Rock", "Paper", "Scissors"]
TIME_PER_STEP    = 2.0       # seconds per gesture step (lock-in window)
INTRO_SECS       = 2.0
RESULT_SECS      = 1.80
GAME_OVER_SECS   = 4.0
STARTING_LENGTH  = 3
PLAYBACK_STEP    = 1.20      # how long each gesture is shown during playback


class SimonSaysSoloController:
    """
    Solo Simon Says.

    States:
      INTRO → PLAYBACK → PLAYER_INPUT → ROUND_WIN → GAME_OVER

    update() signature:
        controller.update(tracker_state=..., now=...)
    """

    def __init__(self):
        self._hs_store = SimonHighscoreStore()
        self._is_new_best = False
        self._run_rank    = 0
        self.reset()

    def reset(self):
        self.state           = "INTRO"
        self.sequence        = []
        self.input_index     = 0
        self.playback_index  = 0
        self.score           = 0
        self.seq_length      = STARTING_LENGTH
        self._intro_until    = 0.0
        self._step_start     = 0.0
        self._playback_start = 0.0
        self._result_until   = 0.0
        self._game_over_until= 0.0
        self.last_result     = ""
        self.fail_at_step    = -1
        self._held_gesture   = ""   # which gesture is currently being held
        self._is_new_best    = False
        self._run_rank       = 0
        self._generate_sequence()

    def start_playback(self, now=None):
        """Called when player presses Enter on the INTRO screen."""
        if self.state == "INTRO":
            if now is None:
                now = time.monotonic()
            self.state           = "PLAYBACK"
            self._playback_start = now
            self.playback_index  = 0

    def _generate_sequence(self):
        self.sequence = [random.choice(VALID_GESTURES)
                         for _ in range(self.seq_length)]
        self.input_index    = 0
        self.playback_index = 0

    def _build_output(self, now):
        # Dwell progress: fill based on how long current gesture has been held
        dwell_pct = 0.0
        if self.state == "PLAYER_INPUT" and self._held_gesture in VALID_GESTURES:
            dwell_pct = min(1.0, (now - self._step_start) / TIME_PER_STEP)

        playback_gesture = ""
        if self.state == "PLAYBACK" and self.playback_index < len(self.sequence):
            playback_gesture = self.sequence[self.playback_index]

        best = self._hs_store.get_best()
        return {
            "play_mode_label":   "Simon Says",
            "state":             self.state,
            "sequence":          list(self.sequence),
            "seq_length":        self.seq_length,
            "input_index":       self.input_index,
            "playback_index":    self.playback_index,
            "playback_gesture":  playback_gesture,
            "score":             self.score,
            "step_time_left":    (1.0 - dwell_pct) * TIME_PER_STEP,
            "dwell_pct":         dwell_pct,
            "held_gesture":      self._held_gesture,
            "last_result":       self.last_result,
            "fail_at_step":      self.fail_at_step,
            "two_player":        False,
            "best_seq":          best["seq_length"] if best else 0,
            "best_player":       best["player"]     if best else "",
            "best_score":        best["score"]      if best else 0,
            "is_new_best":       self._is_new_best,
            "run_rank":          self._run_rank,
            "top_scores":        self._hs_store.get_top(),
        }

    def update(self, tracker_state, now=None, player_name=""):
        if now is None:
            now = time.monotonic()

        confirmed = tracker_state.get("confirmed_gesture", "Unknown")

        if self.state == "INTRO":
            return self._build_output(now)

        if self.state == "GAME_OVER":
            if now >= self._game_over_until:
                self.reset()
            return self._build_output(now)

        if self.state == "PLAYBACK":
            elapsed = now - self._playback_start
            step    = int(elapsed / PLAYBACK_STEP)
            if step >= len(self.sequence):
                self.state       = "PLAYER_INPUT"
                self.input_index = 0
                self._step_start = now
            else:
                self.playback_index = step
            return self._build_output(now)

        if self.state == "ROUND_WIN":
            if now >= self._result_until:
                self.score      += 1
                self.seq_length += 1
                self._generate_sequence()
                self.state           = "PLAYBACK"
                self._playback_start = now
                self.playback_index  = 0
            return self._build_output(now)

        if self.state == "PLAYER_INPUT":
            expected = self.sequence[self.input_index]

            # Dwell-to-confirm: track how long the current gesture has been held.
            # The timer resets any time the gesture changes.
            # Only when the same gesture has been held for TIME_PER_STEP seconds
            # does it "lock in" — then we check if it matches expected.
            # This prevents transitional fist poses from firing instantly.

            if confirmed in VALID_GESTURES:
                if confirmed != self._held_gesture:
                    # Gesture changed — reset dwell timer
                    self._held_gesture = confirmed
                    self._step_start   = now
                else:
                    # Same gesture held — check dwell time
                    dwell = now - self._step_start
                    if dwell >= TIME_PER_STEP:
                        if confirmed == expected:
                            self.last_result   = "correct"
                            self._held_gesture = ""
                            self._advance_input(now)
                        else:
                            self.last_result      = "wrong"
                            self.fail_at_step     = self.input_index
                            self._submit_score(player_name, now)
                            self.state            = "GAME_OVER"
                            self._game_over_until = now + GAME_OVER_SECS
            else:
                # Unknown / no gesture — reset the dwell timer (pauses it)
                if self._held_gesture != "":
                    self._held_gesture = ""
                    self._step_start   = now

        return self._build_output(now)

    def _submit_score(self, player_name: str, now=None):
        name = (player_name or "Unknown").strip()
        self._is_new_best, self._run_rank = self._hs_store.submit(
            player_name=name,
            score=self.score,
            seq_length=self.seq_length,
        )

    def _advance_input(self, now):
        self.input_index  += 1
        self._held_gesture = ""
        if self.input_index >= len(self.sequence):
            self.state         = "ROUND_WIN"
            self._result_until = now + RESULT_SECS
        else:
            self._step_start = now


# ─────────────────────────────────────────────────────────────────────────────
# Two-Player Controller
# ─────────────────────────────────────────────────────────────────────────────

class SimonSaysTwoPlayerController:
    """
    Two-player chain Simon Says.

    P1 adds a gesture (held 2s to lock in) → system plays back chain →
    P2 repeats full chain (each step held 2s) then adds one new gesture →
    system plays back → P1 repeats and adds → ...

    States: INTRO → ADD_GESTURE → PLAYBACK → PLAYER_INPUT → GAME_OVER
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.state            = "INTRO"
        self.sequence         = []
        self.input_index      = 0
        self.playback_index   = 0
        self.current_player   = 1
        self.loser            = 0
        self.rounds_completed = 0
        self._step_start      = 0.0
        self._playback_start  = 0.0
        self._game_over_until = 0.0
        self._held_gesture    = ""
        self._waiting_for_neutral = False
        self._tracker_reset_req   = False
        self.last_result      = ""

    def start_playback(self, now=None):
        """Called on Enter from INTRO."""
        if self.state == "INTRO":
            if now is None:
                now = time.monotonic()
            self.state         = "ADD_GESTURE"
            self._step_start   = now
            self._held_gesture = ""

    def _build_output(self, now):
        dwell_pct = 0.0
        if self.state in ("ADD_GESTURE", "PLAYER_INPUT") and self._held_gesture in VALID_GESTURES:
            dwell_pct = min(1.0, (now - self._step_start) / TIME_PER_STEP)

        playback_gesture = ""
        if self.state == "PLAYBACK" and self.playback_index < len(self.sequence):
            playback_gesture = self.sequence[self.playback_index]

        return {
            "play_mode_label":   "Simon Says 2P",
            "state":             self.state,
            "sequence":          list(self.sequence),
            "seq_length":        len(self.sequence),
            "input_index":       self.input_index,
            "playback_index":    self.playback_index,
            "playback_gesture":  playback_gesture,
            "current_player":    self.current_player,
            "loser":             self.loser,
            "rounds_completed":  self.rounds_completed,
            "dwell_pct":         dwell_pct,
            "held_gesture":      self._held_gesture,
            "last_result":       self.last_result,
            "two_player":        True,
            "waiting_for_neutral":         self._waiting_for_neutral,
            "tracker_reset_requested":     getattr(self, "_tracker_reset_req", False),
        }

    def _tracker_for_current(self, p1_tracker, p2_tracker):
        """
        Return the best available tracker for the active player.
        Since players take turns with a single hand in frame, we use
        whichever tracker has an actual hand detected rather than
        relying on the fixed left/right spatial assignment.
        """
        p1_has_hand = p1_tracker.get("confirmed_gesture", "Unknown") != "Unknown" or \
                      p1_tracker.get("stable_gesture", "Unknown") != "Unknown"
        p2_has_hand = p2_tracker.get("confirmed_gesture", "Unknown") != "Unknown" or \
                      p2_tracker.get("stable_gesture", "Unknown") != "Unknown"

        # If only one tracker has a hand, use that one regardless of player number
        if p1_has_hand and not p2_has_hand:
            return p1_tracker
        if p2_has_hand and not p1_has_hand:
            return p2_tracker

        # Both or neither have a hand — fall back to nominal assignment
        return p1_tracker if self.current_player == 1 else p2_tracker

    def update(self, p1_tracker, p2_tracker, now=None):
        if now is None:
            now = time.monotonic()

        active_tracker = self._tracker_for_current(p1_tracker, p2_tracker)
        confirmed = active_tracker.get("confirmed_gesture", "Unknown")

        # ── INTRO ──────────────────────────────────────────────────────────
        if self.state == "INTRO":
            return self._build_output(now)

        # ── GAME_OVER ──────────────────────────────────────────────────────
        if self.state == "GAME_OVER":
            if now >= self._game_over_until:
                self.reset()
            return self._build_output(now)

        # ── ADD_GESTURE ────────────────────────────────────────────────────
        # No neutral gate — player can immediately show their gesture
        if self.state == "ADD_GESTURE":
            if confirmed in VALID_GESTURES:
                if confirmed != self._held_gesture:
                    self._held_gesture = confirmed
                    self._step_start   = now
                else:
                    if now - self._step_start >= TIME_PER_STEP:
                        self.sequence.append(confirmed)
                        self._held_gesture   = ""
                        self.state           = "PLAYBACK"
                        self._playback_start = now
                        self.playback_index  = 0
            else:
                self._held_gesture = ""
            return self._build_output(now)

        # ── PLAYBACK ───────────────────────────────────────────────────────
        if self.state == "PLAYBACK":
            elapsed = now - self._playback_start
            step    = int(elapsed / PLAYBACK_STEP)
            if step >= len(self.sequence):
                other               = 2 if self.current_player == 1 else 1
                self.current_player = other
                self.state          = "PLAYER_INPUT"
                self.input_index    = 0
                self._step_start    = now
                self._held_gesture  = ""
                self._waiting_for_neutral = True
                self._tracker_reset_req   = True   # clear both trackers on turn change
            else:
                self.playback_index = step
            return self._build_output(now)

        # ── PLAYER_INPUT ───────────────────────────────────────────────────
        if self.state == "PLAYER_INPUT":
            active_tracker = self._tracker_for_current(p1_tracker, p2_tracker)
            confirmed      = active_tracker.get("confirmed_gesture", "Unknown")
            expected       = self.sequence[self.input_index]

            # Wait for neutral hand before accepting any gesture
            if self._waiting_for_neutral:
                if confirmed not in VALID_GESTURES:
                    self._waiting_for_neutral = False
                return self._build_output(now)

            if confirmed in VALID_GESTURES:
                if confirmed != self._held_gesture:
                    self._held_gesture = confirmed
                    self._step_start   = now
                else:
                    if now - self._step_start >= TIME_PER_STEP:
                        self._held_gesture = ""
                        if confirmed == expected:
                            self.last_result = "correct"
                            self._next_input_step(now)
                        else:
                            self.last_result      = "wrong"
                            self.loser            = self.current_player
                            self.state            = "GAME_OVER"
                            self._game_over_until = now + GAME_OVER_SECS
            else:
                self._held_gesture = ""

        return self._build_output(now)

    def _next_input_step(self, now):
        self.input_index  += 1
        self._held_gesture = ""
        if self.input_index >= len(self.sequence):
            # Full chain reproduced — the player who just finished adds the next gesture.
            # Don't switch current_player here; the switch happens at PLAYBACK→PLAYER_INPUT.
            self.rounds_completed += 1
            self.state             = "ADD_GESTURE"
            self._step_start       = now
            self._held_gesture     = ""
            # No neutral gate — can go straight into showing a gesture
        else:
            self._step_start   = now
            self._held_gesture = ""

    def _fail(self, now):
        self.loser            = self.current_player
        self.last_result      = "wrong"
        self.state            = "GAME_OVER"
        self._game_over_until = now + GAME_OVER_SECS

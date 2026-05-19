"""
simon_says_state.py
===================
Gesture Simon Says -- two controllers:

  SimonSaysSoloController
    - System generates a sequence of gestures
    - Player must reproduce each gesture in order within TIME_PER_STEP seconds
    - The gesture "locks in" once it has been held continuously for TIME_PER_STEP
    - Correct sequence -> next round with one more gesture added
    - Wrong gesture at any step -> GAME_OVER
    - Starts with 3 gestures; grows by 1 each successful round

  SimonSaysTwoPlayerController
    - P1 shows a gesture (held 2s to lock in) -> system plays it back
    - P2 must copy the growing chain, then adds one new gesture
    - P1 must replay the full updated chain, then adds one new gesture
    - One wrong step = game over for that player
    - Score = longest sequence completed before failing
"""

import time
import random
from simon_highscore_store import SimonHighscoreStore

VALID_GESTURES   = ["Rock", "Paper", "Scissors"]
TIME_PER_STEP    = 2.0       # seconds a gesture must be held before it locks in
UNKNOWN_GRACE    = 0.30      # seconds of Unknown before the dwell timer resets (calibrated)
INTRO_SECS       = 2.0
RESULT_SECS      = 1.80      # how long the round-win flash shows
GAME_OVER_SECS   = 4.0
STARTING_LENGTH  = 3         # how many gestures in the first round's sequence
PLAYBACK_STEP    = 1.20      # how long each gesture is shown during system playback


class SimonSaysSoloController:
    """
    Solo Simon Says.

    States:
      INTRO -> PLAYBACK -> PLAYER_INPUT -> ROUND_WIN -> GAME_OVER

    update() signature:
        controller.update(tracker_state=..., now=...)
    """

    def __init__(self):
        self._hs_store    = SimonHighscoreStore()
        self._is_new_best = False
        self._run_rank    = 0
        self.reset()

    def reset(self):
        """Reset all state back to the beginning (new game)."""
        self.state            = "INTRO"
        self.sequence         = []       # current round's gesture sequence
        self.input_index      = 0        # which gesture in the sequence we're expecting next
        self.playback_index   = 0        # which gesture is currently highlighted during playback
        self.score            = 0        # number of rounds successfully completed
        self.seq_length       = STARTING_LENGTH
        self._intro_until     = 0.0
        self._step_start      = 0.0      # when the current dwell timer started
        self._playback_start  = 0.0
        self._result_until    = 0.0
        self._game_over_until = 0.0
        self.last_result      = ""       # "correct" | "wrong" | ""
        self.fail_at_step     = -1       # index of the step that failed (for UI highlight)
        self._held_gesture    = ""       # which gesture is being held right now
        self._unknown_since   = 0.0     # timestamp when Unknown started; 0 = not in Unknown
        self._is_new_best     = False
        self._run_rank        = 0
        self._generate_sequence()

    def start_playback(self, now=None):
        """
        Begin the playback phase.  Called when the player presses Enter
        on the INTRO screen to start the round.
        """
        if self.state == "INTRO":
            if now is None:
                now = time.monotonic()
            self.state           = "PLAYBACK"
            self._playback_start = now
            self.playback_index  = 0

    def _generate_sequence(self):
        """Create a fresh random gesture sequence of the current length."""
        self.sequence = [random.choice(VALID_GESTURES)
                         for _ in range(self.seq_length)]
        self.input_index    = 0
        self.playback_index = 0

    def _build_output(self, now):
        """
        Build the output dict the UI reads every frame.

        Includes dwell progress (how long the current gesture has been held)
        so the UI can show a fill bar.
        """
        # Dwell progress: how far into the TIME_PER_STEP lock-in window we are
        dwell_pct = 0.0
        if self.state == "PLAYER_INPUT" and self._held_gesture in VALID_GESTURES:
            dwell_pct = min(1.0, (now - self._step_start) / TIME_PER_STEP)

        # The gesture currently being highlighted during playback
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
        """
        Main frame update.  Drives the Simon Says state machine.

        tracker_state must contain 'confirmed_gesture' (from the hand tracker).
        player_name is used for high-score submission on GAME_OVER.
        """
        if now is None:
            now = time.monotonic()

        confirmed = tracker_state.get("confirmed_gesture", "Unknown")

        # ── INTRO: show instructions until start_playback() is called ──
        if self.state == "INTRO":
            return self._build_output(now)

        # ── GAME_OVER: linger, then auto-reset ──
        if self.state == "GAME_OVER":
            if now >= self._game_over_until:
                self.reset()
            return self._build_output(now)

        # ── PLAYBACK: animate through the sequence at PLAYBACK_STEP seconds each ──
        if self.state == "PLAYBACK":
            elapsed = now - self._playback_start
            step    = int(elapsed / PLAYBACK_STEP)
            if step >= len(self.sequence):
                # Finished showing the whole sequence — let the player input it
                self.state       = "PLAYER_INPUT"
                self.input_index = 0
                self._step_start = now
            else:
                # Advance the highlighted gesture as time passes
                self.playback_index = step
            return self._build_output(now)

        # ── ROUND_WIN: brief celebration, then advance to the next round ──
        if self.state == "ROUND_WIN":
            if now >= self._result_until:
                self.score      += 1
                self.seq_length += 1
                self._generate_sequence()
                self.state           = "PLAYBACK"
                self._playback_start = now
                self.playback_index  = 0
            return self._build_output(now)

        # ── PLAYER_INPUT: dwell-to-confirm input ──────────────────────────
        if self.state == "PLAYER_INPUT":
            expected = self.sequence[self.input_index]

            # Dwell-to-confirm mechanic:
            #   - The player must hold the SAME gesture continuously for TIME_PER_STEP
            #   - If the gesture changes, the dwell timer resets
            #   - Brief Unknown frames (hand transitioning) don't reset the timer;
            #     only Unknown held for > UNKNOWN_GRACE resets it
            # This prevents a fist passing through "Rock" on the way to "Paper"
            # from being mistakenly locked in as Rock.

            if confirmed in VALID_GESTURES:
                # A valid gesture is being shown
                self._unknown_since = 0.0   # cancel any pending Unknown reset

                if confirmed != self._held_gesture:
                    # Gesture changed — restart the dwell timer
                    self._held_gesture = confirmed
                    self._step_start   = now
                else:
                    # Same gesture held — check if dwell time has elapsed
                    dwell = now - self._step_start
                    if dwell >= TIME_PER_STEP:
                        if confirmed == expected:
                            # Correct! Clear held gesture and advance
                            self.last_result   = "correct"
                            self._held_gesture = ""
                            self._advance_input(now)
                        else:
                            # Wrong gesture locked in — game over
                            self.last_result      = "wrong"
                            self.fail_at_step     = self.input_index
                            self._submit_score(player_name, now)
                            self.state            = "GAME_OVER"
                            self._game_over_until = now + GAME_OVER_SECS
            else:
                # Unknown / no gesture detected.
                # Only reset the dwell timer after UNKNOWN_GRACE seconds of
                # continuous Unknown, so brief hand-transition frames don't
                # interrupt a gesture the player is in the middle of holding.
                if self._unknown_since == 0.0:
                    self._unknown_since = now   # start timing the Unknown streak
                elif now - self._unknown_since >= UNKNOWN_GRACE:
                    # Unknown held too long — reset dwell
                    self._held_gesture  = ""
                    self._unknown_since = 0.0

        return self._build_output(now)

    def _submit_score(self, player_name: str, now=None):
        """Submit this run's score to the high-score store."""
        name = (player_name or "Unknown").strip()
        self._is_new_best, self._run_rank = self._hs_store.submit(
            player_name=name,
            score=self.score,
            seq_length=self.seq_length,
        )

    def _advance_input(self, now):
        """
        Move to the next gesture in the sequence.
        If all gestures were reproduced correctly, transition to ROUND_WIN.
        """
        self.input_index  += 1
        self._held_gesture = ""
        if self.input_index >= len(self.sequence):
            # All gestures completed — round won!
            self.state         = "ROUND_WIN"
            self._result_until = now + RESULT_SECS
        else:
            # More gestures to go — reset the dwell timer for the next step
            self._step_start = now


# ─────────────────────────────────────────────────────────────────────────────
# Two-Player Controller
# ─────────────────────────────────────────────────────────────────────────────

class SimonSaysTwoPlayerController:
    """
    Two-player chain Simon Says.

    P1 adds a gesture (held 2s to lock in) -> system plays back the chain ->
    P2 repeats the full chain (each step held 2s) then adds one new gesture ->
    system plays back -> P1 repeats and adds -> ...

    States: INTRO -> ADD_GESTURE -> PLAYBACK -> PLAYER_INPUT -> GAME_OVER
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all state to the beginning."""
        self.state            = "INTRO"
        self.sequence         = []        # growing chain of gestures
        self.input_index      = 0         # which step we expect from the active player
        self.playback_index   = 0         # which step is highlighted during playback
        self.current_player   = 1         # whose turn it is (1 or 2)
        self.loser            = 0         # set to 1 or 2 when someone fails
        self.rounds_completed = 0         # how many full chain reproductions happened
        self._step_start      = 0.0
        self._playback_start  = 0.0
        self._game_over_until = 0.0
        self._held_gesture    = ""
        self._unknown_since   = 0.0
        self._waiting_for_neutral  = False   # True while waiting for player to clear their hand
        self._tracker_reset_req    = False   # signal to reset both hand trackers on turn change
        self.last_result      = ""

    def start_playback(self, now=None):
        """Called when the player presses Enter on the INTRO screen."""
        if self.state == "INTRO":
            if now is None:
                now = time.monotonic()
            self.state         = "ADD_GESTURE"
            self._step_start   = now
            self._held_gesture = ""

    def _build_output(self, now):
        """Build the output dict the UI reads every frame."""
        # Dwell progress for the ADD_GESTURE and PLAYER_INPUT phases
        dwell_pct = 0.0
        if self.state in ("ADD_GESTURE", "PLAYER_INPUT") and self._held_gesture in VALID_GESTURES:
            dwell_pct = min(1.0, (now - self._step_start) / TIME_PER_STEP)

        # The gesture currently shown during system playback
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
        Return the best available tracker for the currently active player.

        Because players take turns with a single hand in frame, we pick
        whichever tracker actually has a hand detected rather than relying on
        the fixed left/right spatial assignment.  This is more robust when a
        player moves around or uses either hand.

        If both (or neither) have a hand, fall back to the nominal assignment.
        """
        p1_has_hand = (p1_tracker.get("confirmed_gesture", "Unknown") != "Unknown" or
                       p1_tracker.get("stable_gesture",    "Unknown") != "Unknown")
        p2_has_hand = (p2_tracker.get("confirmed_gesture", "Unknown") != "Unknown" or
                       p2_tracker.get("stable_gesture",    "Unknown") != "Unknown")

        if p1_has_hand and not p2_has_hand:
            return p1_tracker   # only P1 visible — use them regardless of turn
        if p2_has_hand and not p1_has_hand:
            return p2_tracker   # only P2 visible

        # Both or neither visible — fall back to nominal player assignment
        return p1_tracker if self.current_player == 1 else p2_tracker

    def update(self, p1_tracker, p2_tracker, now=None):
        """
        Main frame update.  Drives the two-player Simon Says state machine.

        p1_tracker / p2_tracker are tracker state dicts for each hand.
        The active tracker is selected by _tracker_for_current().
        """
        if now is None:
            now = time.monotonic()

        active_tracker = self._tracker_for_current(p1_tracker, p2_tracker)
        confirmed = active_tracker.get("confirmed_gesture", "Unknown")

        # ── INTRO: wait for start_playback() ──
        if self.state == "INTRO":
            return self._build_output(now)

        # ── GAME_OVER: linger, then auto-reset ──
        if self.state == "GAME_OVER":
            if now >= self._game_over_until:
                self.reset()
            return self._build_output(now)

        # ── ADD_GESTURE: active player adds a new gesture to the chain ──
        # No neutral gate here — player can go straight into showing their gesture.
        if self.state == "ADD_GESTURE":
            if confirmed in VALID_GESTURES:
                if confirmed != self._held_gesture:
                    # Gesture changed — restart dwell timer
                    self._held_gesture = confirmed
                    self._step_start   = now
                else:
                    # Same gesture held — check dwell
                    if now - self._step_start >= TIME_PER_STEP:
                        # Locked in — append to chain and start playback
                        self.sequence.append(confirmed)
                        self._held_gesture   = ""
                        self.state           = "PLAYBACK"
                        self._playback_start = now
                        self.playback_index  = 0
            else:
                # No valid gesture shown — reset dwell
                self._held_gesture = ""
            return self._build_output(now)

        # ── PLAYBACK: show the chain to the next player, then hand over ──
        if self.state == "PLAYBACK":
            elapsed = now - self._playback_start
            step    = int(elapsed / PLAYBACK_STEP)
            if step >= len(self.sequence):
                # Playback finished — switch to the other player's input turn
                other               = 2 if self.current_player == 1 else 1
                self.current_player = other
                self.state          = "PLAYER_INPUT"
                self.input_index    = 0
                self._step_start    = now
                self._held_gesture  = ""
                # Require the new player to show a neutral hand before we accept input.
                # This prevents the gesture they were just watching playback with from
                # accidentally counting as their first input.
                self._waiting_for_neutral = True
                self._tracker_reset_req   = True   # tell the tracker to clear history on turn change
            else:
                self.playback_index = step
            return self._build_output(now)

        # ── PLAYER_INPUT: active player reproduces the sequence ──
        if self.state == "PLAYER_INPUT":
            # Re-read active tracker since current_player may have just changed
            active_tracker = self._tracker_for_current(p1_tracker, p2_tracker)
            confirmed      = active_tracker.get("confirmed_gesture", "Unknown")
            expected       = self.sequence[self.input_index]

            # Wait for a neutral (non-gesture) hand before accepting any input.
            # This prevents residual gestures from the turn transition from firing.
            if self._waiting_for_neutral:
                if confirmed not in VALID_GESTURES:
                    self._waiting_for_neutral = False   # hand is clear, ready to accept input
                return self._build_output(now)

            # Same dwell-to-confirm mechanic as solo mode
            if confirmed in VALID_GESTURES:
                self._unknown_since = 0.0
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
                            # Wrong gesture — this player loses
                            self.last_result      = "wrong"
                            self.loser            = self.current_player
                            self.state            = "GAME_OVER"
                            self._game_over_until = now + GAME_OVER_SECS
            else:
                # Unknown — only reset dwell after UNKNOWN_GRACE seconds
                if self._unknown_since == 0.0:
                    self._unknown_since = now
                elif now - self._unknown_since >= UNKNOWN_GRACE:
                    self._held_gesture  = ""
                    self._unknown_since = 0.0

        return self._build_output(now)

    def _next_input_step(self, now):
        """
        Advance to the next gesture in the input sequence.

        If all gestures were reproduced, the active player adds a new gesture.
        Note: the current_player switch happens at PLAYBACK -> PLAYER_INPUT,
        not here, so the player who just finished adding gets to ADD again.
        """
        self.input_index  += 1
        self._held_gesture = ""
        if self.input_index >= len(self.sequence):
            # Full chain successfully reproduced
            self.rounds_completed += 1
            # Now the same player adds a new gesture to extend the chain
            self.state         = "ADD_GESTURE"
            self._step_start   = now
            self._held_gesture = ""
            # No neutral gate needed here — player can go straight into adding
        else:
            # More steps to go — reset dwell timer for the next expected gesture
            self._step_start   = now
            self._held_gesture = ""

    def _fail(self, now):
        """Helper to trigger game-over for the current player."""
        self.loser            = self.current_player
        self.last_result      = "wrong"
        self.state            = "GAME_OVER"
        self._game_over_until = now + GAME_OVER_SECS

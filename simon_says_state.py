"""
simon_says_state.py
===================
Gesture Simon Says — two controllers:

  SimonSaysSoloController
    The system shows a growing sequence of RPS gestures and the player must
    reproduce them in order. Each gesture is confirmed by holding it for
    TIME_PER_STEP seconds (dwell-to-confirm, so brief hand transitions don't
    accidentally lock in a gesture mid-movement).

    Round flow: INTRO -> PLAYBACK -> PLAYER_INPUT -> ROUND_WIN -> (repeat, longer sequence)
    Fail: any wrong gesture -> GAME_OVER
    Score = number of rounds completed before failing.
    Starts at 3 gestures; grows by 1 each round.

  SimonSaysTwoPlayerController
    Players take turns extending a shared gesture chain.
    P1 adds a gesture -> system plays back the full chain -> P2 repeats it
    and adds a new one -> system plays back -> P1 repeats and adds -> ...
    First player to make a mistake loses.
    Score = length of the chain when the mistake happened.
"""

import time
import random
from simon_highscore_store import SimonHighscoreStore

VALID_GESTURES  = ["Rock", "Paper", "Scissors"]
TIME_PER_STEP   = 2.0   # seconds a gesture must be held before it locks in
UNKNOWN_GRACE   = 0.30  # seconds of Unknown before the dwell timer resets (calibrated to
                        # ignore brief tracking glitches between gestures)
INTRO_SECS      = 2.0
RESULT_SECS     = 1.80  # how long the "round won" flash shows
GAME_OVER_SECS  = 4.0
STARTING_LENGTH = 3     # number of gestures in the very first round
PLAYBACK_STEP   = 1.20  # seconds each gesture is displayed during system playback


# ─────────────────────────────────────────────────────────────────────────────
# Solo Controller
# ─────────────────────────────────────────────────────────────────────────────

class SimonSaysSoloController:
    """
    Solo Simon Says.

    State flow:
      INTRO -> PLAYBACK -> PLAYER_INPUT -> ROUND_WIN -> PLAYBACK (next round)
                                        -> GAME_OVER (wrong gesture)
    """

    def __init__(self):
        self._hs_store    = SimonHighscoreStore()
        self._is_new_best = False
        self._run_rank    = 0
        self.reset()

    def reset(self):
        """Reset all state back to a new game."""
        self.state            = "INTRO"
        self.sequence         = []              # the current round's gesture sequence
        self.input_index      = 0              # which position in the sequence we're waiting on
        self.playback_index   = 0              # which gesture is highlighted during playback
        self.score            = 0              # rounds completed so far
        self.seq_length       = STARTING_LENGTH
        self._intro_until     = 0.0
        self._step_start      = 0.0            # when the current dwell timer started
        self._playback_start  = 0.0
        self._result_until    = 0.0
        self._game_over_until = 0.0
        self.last_result      = ""             # "correct" | "wrong" | ""
        self.fail_at_step     = -1             # which step triggered game over (for UI highlight)
        self._held_gesture    = ""             # gesture currently being held
        self._unknown_since   = 0.0            # timestamp when Unknown started; 0 = not in Unknown
        self._is_new_best     = False
        self._run_rank        = 0
        self._generate_sequence()

    def start_playback(self, now=None):
        """
        Begin the playback phase.
        Called when the player presses Enter on the INTRO screen.
        """
        if self.state == "INTRO":
            if now is None:
                now = time.monotonic()
            self.state           = "PLAYBACK"
            self._playback_start = now
            self.playback_index  = 0

    def _generate_sequence(self):
        """Create a fresh random gesture sequence of the current length."""
        self.sequence     = [random.choice(VALID_GESTURES) for _ in range(self.seq_length)]
        self.input_index  = 0
        self.playback_index = 0

    def _build_output(self, now):
        """
        Build the output dict the UI reads every frame.

        Includes dwell_pct (0.0–1.0) so the UI can show a fill bar showing
        how long the current gesture has been held.
        """
        # How far through the TIME_PER_STEP window the current gesture hold is
        dwell_pct = 0.0
        if self.state == "PLAYER_INPUT" and self._held_gesture in VALID_GESTURES:
            dwell_pct = min(1.0, (now - self._step_start) / TIME_PER_STEP)

        # Which gesture is currently highlighted during playback
        playback_gesture = ""
        if self.state == "PLAYBACK" and self.playback_index < len(self.sequence):
            playback_gesture = self.sequence[self.playback_index]

        best = self._hs_store.get_best()
        return {
            "play_mode_label":  "Simon Says",
            "state":            self.state,
            "sequence":         list(self.sequence),
            "seq_length":       self.seq_length,
            "input_index":      self.input_index,
            "playback_index":   self.playback_index,
            "playback_gesture": playback_gesture,
            "score":            self.score,
            "step_time_left":   (1.0 - dwell_pct) * TIME_PER_STEP,  # time remaining to lock in
            "dwell_pct":        dwell_pct,
            "held_gesture":     self._held_gesture,
            "last_result":      self.last_result,
            "fail_at_step":     self.fail_at_step,
            "two_player":       False,
            "best_seq":         best["seq_length"] if best else 0,
            "best_player":      best["player"]     if best else "",
            "best_score":       best["score"]      if best else 0,
            "is_new_best":      self._is_new_best,
            "run_rank":         self._run_rank,
            "top_scores":       self._hs_store.get_top(),
        }

    def update(self, tracker_state, now=None, player_name=""):
        """
        Main frame update. Drives the Solo Simon Says state machine.

        tracker_state must contain 'confirmed_gesture' from the hand tracker.
        player_name is used for high-score submission when the game ends.
        """
        if now is None:
            now = time.monotonic()

        confirmed = tracker_state.get("confirmed_gesture", "Unknown")

        # INTRO: just sit here until start_playback() is called via Enter key
        if self.state == "INTRO":
            return self._build_output(now)

        # GAME_OVER: linger on the fail screen, then auto-reset to a new game
        if self.state == "GAME_OVER":
            if now >= self._game_over_until:
                self.reset()
            return self._build_output(now)

        # PLAYBACK: animate through the sequence, highlighting one gesture per PLAYBACK_STEP
        if self.state == "PLAYBACK":
            elapsed = now - self._playback_start
            step    = int(elapsed / PLAYBACK_STEP)  # which gesture we should be showing
            if step >= len(self.sequence):
                # Finished showing the whole sequence — hand over to the player
                self.state       = "PLAYER_INPUT"
                self.input_index = 0
                self._step_start = now
            else:
                self.playback_index = step  # advance the highlighted gesture
            return self._build_output(now)

        # ROUND_WIN: brief celebration, then extend the sequence and start the next round
        if self.state == "ROUND_WIN":
            if now >= self._result_until:
                self.score      += 1
                self.seq_length += 1
                self._generate_sequence()
                self.state           = "PLAYBACK"
                self._playback_start = now
                self.playback_index  = 0
            return self._build_output(now)

        # PLAYER_INPUT: the player reproduces the sequence using dwell-to-confirm
        if self.state == "PLAYER_INPUT":
            expected = self.sequence[self.input_index]

            if confirmed in VALID_GESTURES:
                # A valid gesture is visible — reset the Unknown grace timer
                self._unknown_since = 0.0

                if confirmed != self._held_gesture:
                    # Gesture changed — restart the dwell timer from scratch
                    # This prevents a hand passing through "Rock" on the way to "Paper"
                    # from being accidentally locked in as Rock
                    self._held_gesture = confirmed
                    self._step_start   = now
                else:
                    # Same gesture held continuously — check if dwell time is up
                    dwell = now - self._step_start
                    if dwell >= TIME_PER_STEP:
                        if confirmed == expected:
                            # Correct gesture locked in — advance to the next step
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
                # Unknown / no gesture. Only reset the dwell timer after UNKNOWN_GRACE
                # seconds of continuous Unknown — brief glitches between gestures
                # shouldn't interrupt an ongoing hold.
                if self._unknown_since == 0.0:
                    self._unknown_since = now  # start timing the Unknown streak
                elif now - self._unknown_since >= UNKNOWN_GRACE:
                    # Unknown has been held too long — player must have dropped their gesture
                    self._held_gesture  = ""
                    self._unknown_since = 0.0

        return self._build_output(now)

    def _submit_score(self, player_name: str, now=None):
        """Submit this run's final score to the high-score store."""
        name = (player_name or "Unknown").strip()
        self._is_new_best, self._run_rank = self._hs_store.submit(
            player_name=name,
            score=self.score,
            seq_length=self.seq_length,
        )

    def _advance_input(self, now):
        """
        Move to the next gesture in the input sequence after a correct lock-in.
        If all gestures were reproduced correctly, transition to ROUND_WIN.
        """
        self.input_index  += 1
        self._held_gesture = ""
        if self.input_index >= len(self.sequence):
            # All gestures in the sequence reproduced — round complete
            self.state         = "ROUND_WIN"
            self._result_until = now + RESULT_SECS
        else:
            # Still more gestures to go — reset the dwell timer for the next step
            self._step_start = now


# ─────────────────────────────────────────────────────────────────────────────
# Two-Player Controller
# ─────────────────────────────────────────────────────────────────────────────

class SimonSaysTwoPlayerController:
    """
    Two-player chain Simon Says.

    Players take turns adding to and reproducing a shared gesture chain.
    The flow goes:
      Active player adds a gesture (held 2s to lock in)
        -> system plays back the full chain
          -> other player reproduces the whole chain and adds one new gesture
            -> repeat, chain grows by 1 each turn

    One wrong step = game over for that player.
    Score = length of the chain when the mistake happened.

    State flow: INTRO -> ADD_GESTURE -> PLAYBACK -> PLAYER_INPUT -> GAME_OVER
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all state to the beginning."""
        self.state            = "INTRO"
        self.sequence         = []      # growing chain of gestures, shared between players
        self.input_index      = 0      # which step in the chain the active player should show next
        self.playback_index   = 0      # which step is highlighted during system playback
        self.current_player   = 1      # whose turn it is (1 or 2)
        self.loser            = 0      # set to 1 or 2 when someone fails
        self.rounds_completed = 0      # how many full chain reproductions have happened
        self._step_start      = 0.0
        self._playback_start  = 0.0
        self._game_over_until = 0.0
        self._held_gesture    = ""
        self._unknown_since   = 0.0
        # After a player turn ends, we wait for a neutral hand before accepting new input.
        # This prevents the last gesture from the previous turn from counting as the first
        # gesture of the new turn.
        self._waiting_for_neutral = False
        # Signal sent to the hand tracker to clear its history on turn changes
        self._tracker_reset_req   = False
        self.last_result      = ""

    def start_playback(self, now=None):
        """Called when the player presses Enter on the INTRO screen to start the game."""
        if self.state == "INTRO":
            if now is None:
                now = time.monotonic()
            self.state         = "ADD_GESTURE"
            self._step_start   = now
            self._held_gesture = ""

    def _build_output(self, now):
        """Build the output dict the UI reads every frame."""
        # Dwell progress during ADD_GESTURE and PLAYER_INPUT (0.0 = just started, 1.0 = locked in)
        dwell_pct = 0.0
        if self.state in ("ADD_GESTURE", "PLAYER_INPUT") and self._held_gesture in VALID_GESTURES:
            dwell_pct = min(1.0, (now - self._step_start) / TIME_PER_STEP)

        # Which gesture is highlighted during system playback
        playback_gesture = ""
        if self.state == "PLAYBACK" and self.playback_index < len(self.sequence):
            playback_gesture = self.sequence[self.playback_index]

        return {
            "play_mode_label":         "Simon Says 2P",
            "state":                   self.state,
            "sequence":                list(self.sequence),
            "seq_length":              len(self.sequence),
            "input_index":             self.input_index,
            "playback_index":          self.playback_index,
            "playback_gesture":        playback_gesture,
            "current_player":          self.current_player,
            "loser":                   self.loser,
            "rounds_completed":        self.rounds_completed,
            "dwell_pct":               dwell_pct,
            "held_gesture":            self._held_gesture,
            "last_result":             self.last_result,
            "two_player":              True,
            "waiting_for_neutral":     self._waiting_for_neutral,
            "tracker_reset_requested": getattr(self, "_tracker_reset_req", False),
        }

    def _tracker_for_current(self, p1_tracker, p2_tracker):
        """
        Return the tracker for the currently active player.

        Because players take turns with a single hand in frame, we pick
        whichever tracker has a hand detected rather than relying purely on
        the fixed left/right spatial assignment. This is more robust when a
        player moves around or uses either hand.

        If both or neither tracker sees a hand, fall back to the nominal assignment.
        """
        p1_has_hand = (p1_tracker.get("confirmed_gesture", "Unknown") != "Unknown" or
                       p1_tracker.get("stable_gesture",    "Unknown") != "Unknown")
        p2_has_hand = (p2_tracker.get("confirmed_gesture", "Unknown") != "Unknown" or
                       p2_tracker.get("stable_gesture",    "Unknown") != "Unknown")

        if p1_has_hand and not p2_has_hand:
            return p1_tracker  # only P1 is visible — use them regardless of whose turn it is
        if p2_has_hand and not p1_has_hand:
            return p2_tracker  # only P2 is visible

        # Both or neither visible — fall back to the nominal player assignment
        return p1_tracker if self.current_player == 1 else p2_tracker

    def update(self, p1_tracker, p2_tracker, now=None):
        """
        Main frame update. Drives the two-player Simon Says state machine.

        p1_tracker / p2_tracker are tracker state dicts for each hand.
        The active player's tracker is selected by _tracker_for_current().
        """
        if now is None:
            now = time.monotonic()

        active_tracker = self._tracker_for_current(p1_tracker, p2_tracker)
        confirmed = active_tracker.get("confirmed_gesture", "Unknown")

        # INTRO: wait here until start_playback() is called via Enter key
        if self.state == "INTRO":
            return self._build_output(now)

        # GAME_OVER: linger on the fail screen, then auto-reset
        if self.state == "GAME_OVER":
            if now >= self._game_over_until:
                self.reset()
            return self._build_output(now)

        # ADD_GESTURE: active player shows a new gesture to append to the chain
        # No neutral gate here — the player can go straight into showing their gesture
        if self.state == "ADD_GESTURE":
            if confirmed in VALID_GESTURES:
                if confirmed != self._held_gesture:
                    # Gesture changed — restart dwell timer
                    self._held_gesture = confirmed
                    self._step_start   = now
                else:
                    # Same gesture held — check if dwell time is up
                    if now - self._step_start >= TIME_PER_STEP:
                        # Gesture locked in — append it to the chain and start playback
                        self.sequence.append(confirmed)
                        self._held_gesture   = ""
                        self.state           = "PLAYBACK"
                        self._playback_start = now
                        self.playback_index  = 0
            else:
                # No valid gesture — reset dwell timer
                self._held_gesture = ""
            return self._build_output(now)

        # PLAYBACK: show the chain to the next player, one gesture per PLAYBACK_STEP seconds
        if self.state == "PLAYBACK":
            elapsed = now - self._playback_start
            step    = int(elapsed / PLAYBACK_STEP)  # which gesture to highlight
            if step >= len(self.sequence):
                # Playback done — switch to the other player's input turn
                other               = 2 if self.current_player == 1 else 1
                self.current_player = other
                self.state          = "PLAYER_INPUT"
                self.input_index    = 0
                self._step_start    = now
                self._held_gesture  = ""
                # Require a neutral hand before accepting input. Without this, the
                # gesture the player was just watching during playback could accidentally
                # count as their first input step.
                self._waiting_for_neutral = True
                self._tracker_reset_req   = True  # tell the tracker to clear history on turn change
            else:
                self.playback_index = step
            return self._build_output(now)

        # PLAYER_INPUT: active player reproduces the sequence step by step
        if self.state == "PLAYER_INPUT":
            # Re-read the active tracker because current_player may have just changed
            active_tracker = self._tracker_for_current(p1_tracker, p2_tracker)
            confirmed      = active_tracker.get("confirmed_gesture", "Unknown")
            expected       = self.sequence[self.input_index]

            # Wait for the player to clear their hand before accepting any input.
            # This prevents residual gestures from the turn transition from firing immediately.
            if self._waiting_for_neutral:
                if confirmed not in VALID_GESTURES:
                    self._waiting_for_neutral = False  # hand is clear, ready to accept input
                return self._build_output(now)

            # Same dwell-to-confirm mechanic as solo mode
            if confirmed in VALID_GESTURES:
                self._unknown_since = 0.0  # reset Unknown grace timer

                if confirmed != self._held_gesture:
                    # Gesture changed — restart dwell timer
                    self._held_gesture = confirmed
                    self._step_start   = now
                else:
                    # Same gesture held — check if dwell time is up
                    if now - self._step_start >= TIME_PER_STEP:
                        self._held_gesture = ""
                        if confirmed == expected:
                            # Correct — advance to the next step
                            self.last_result = "correct"
                            self._next_input_step(now)
                        else:
                            # Wrong gesture locked in — this player loses
                            self.last_result      = "wrong"
                            self.loser            = self.current_player
                            self.state            = "GAME_OVER"
                            self._game_over_until = now + GAME_OVER_SECS
            else:
                # Unknown — only reset dwell after UNKNOWN_GRACE seconds of continuous Unknown
                if self._unknown_since == 0.0:
                    self._unknown_since = now
                elif now - self._unknown_since >= UNKNOWN_GRACE:
                    self._held_gesture  = ""
                    self._unknown_since = 0.0

        return self._build_output(now)

    def _next_input_step(self, now):
        """
        Advance to the next gesture in the input sequence after a correct lock-in.

        If the active player has reproduced the full chain, they now get to add
        a new gesture (ADD_GESTURE). Note: the player switch happens at
        PLAYBACK -> PLAYER_INPUT, not here — so whoever just finished the chain
        is the one who adds to it next.
        """
        self.input_index  += 1
        self._held_gesture = ""
        if self.input_index >= len(self.sequence):
            # Full chain reproduced successfully — now add a new gesture
            self.rounds_completed += 1
            self.state         = "ADD_GESTURE"
            self._step_start   = now
            self._held_gesture = ""
            # No neutral gate needed here — player can go straight into adding
        else:
            # More steps remaining — reset the dwell timer for the next expected gesture
            self._step_start   = now
            self._held_gesture = ""

    def _fail(self, now):
        """Helper to trigger game-over for the current player."""
        self.loser            = self.current_player
        self.last_result      = "wrong"
        self.state            = "GAME_OVER"
        self._game_over_until = now + GAME_OVER_SECS

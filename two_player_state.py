"""
two_player_state.py
===================
Two-player game modes for the RPS robot game.

  TwoPlayerPvPController   — Player 1 (left hand) vs Player 2 (right hand),
                             no AI involved. Best-of-N rounds.

  PvPvAIController         — Player 1 vs Player 2 vs FairPlayAI, everyone
                             for themselves. Points awarded per opponent beaten.

Both controllers use a pump-based beat-detection system inherited from _BeatMixin.
A "beat" is counted when both players pump their wrists downward in sync.
Four beats open the SHOOT window where players reveal their throws.
"""

import time
from fair_play_state import compare_rps
from fair_play_ai    import FairPlayAI

# Gestures we consider valid throws — Unknown, None, etc. are ignored
VALID_GESTURES = frozenset({"Rock", "Paper", "Scissors"})

# What each gesture beats, and what it loses to (used for AI planning)
UPGRADE   = {"Rock": "Paper",    "Paper": "Scissors", "Scissors": "Rock"}
DOWNGRADE = {"Rock": "Scissors", "Paper": "Rock",     "Scissors": "Paper"}


# ─────────────────────────────────────────────────────────────────────────────
# Shared beat-detection mixin
# ─────────────────────────────────────────────────────────────────────────────

class _BeatMixin:
    """
    Provides pump-based countdown beat detection for two-player modes.

    Each player's wrist is tracked independently. A beat is only counted
    when BOTH players have pumped their wrist within SYNC_WINDOW seconds
    of each other. This means neither player can carry the countdown solo —
    they have to pump together.

    If players fall out of sync mid-countdown, the count pauses (no reset)
    until they sync up again on the next pump.
    """

    # Calibrated timing values — don't change these without re-testing
    DOWN_THRESHOLD   = 0.035  # how far the wrist must drop (normalised) to register a downstroke
    UP_THRESHOLD     = 0.025  # how far it must rise back up before we arm the next downstroke
    BEAT_COOLDOWN    = 0.20   # minimum seconds between registered beats (avoids double-counting)
    SYNC_WINDOW      = 0.65   # both players must pump within this window to count as one beat
    ROCK_GRACE_PERIOD = 0.50  # keep accepting beats this many seconds after Rock disappears
    SHOOT_WINDOW     = 0.55   # how long the SHOOT window stays open
    ROCK_ASSUME      = 0.14   # assume Rock if wrist is visible but gesture is unclassified

    def _init_beat(self):
        """Reset all beat-detection state. Call this at the start of every new countdown."""
        self.beat_count      = 0
        self.last_beat_time  = 0.0
        self.last_rock_time  = 0.0
        self.shoot_open_time  = None
        self.shoot_close_time = None

        # Each player gets their own pump state machine so they can't carry each other
        self._p1_phase  = "ready_for_down"  # waiting for P1 to pump down
        self._p1_top_y  = None              # highest wrist position P1 has reached
        self._p1_bot_y  = None              # lowest wrist position during the current stroke
        self._p1_pump_t = 0.0              # timestamp of P1's most recent completed pump

        self._p2_phase  = "ready_for_down"
        self._p2_top_y  = None
        self._p2_bot_y  = None
        self._p2_pump_t = 0.0

    def _track_hand_pump(self, wrist_y, phase_attr, top_attr, bot_attr, now):
        """
        Run one frame of pump detection for a single hand.

        Returns True the instant a downstroke is completed (wrist moved down
        by at least DOWN_THRESHOLD from its recent peak). We fire on the DOWN
        because that matches the natural RPS rhythm: 1-2-3-SHOOT.

        After the downstroke fires, we wait for the wrist to recover upward
        by UP_THRESHOLD before arming the next downstroke detector.

        Args:
            wrist_y    : normalised Y coordinate (0 = top of frame, 1 = bottom)
            phase_attr : name of the instance attribute holding this hand's phase
            top_attr   : name of the attribute holding this hand's peak Y
            bot_attr   : name of the attribute holding this hand's trough Y
            now        : current timestamp (available for future use)
        """
        # Load this hand's current state from instance attributes
        phase = getattr(self, phase_attr)
        top_y = getattr(self, top_attr)
        bot_y = getattr(self, bot_attr)

        pumped = False  # will be set True if a downstroke fires this frame

        if phase == "ready_for_down":
            # Track the highest (smallest Y) position seen since the last beat
            if top_y is None:
                top_y = wrist_y
            top_y = min(top_y, wrist_y)

            # If the wrist has dropped far enough from the peak, register a beat
            if (wrist_y - top_y) >= self.DOWN_THRESHOLD:
                setattr(self, phase_attr, "waiting_for_up")
                bot_y  = wrist_y
                pumped = True  # fire the beat on the downstroke

        elif phase == "waiting_for_up":
            # Track the lowest point of this downstroke
            bot_y = max(bot_y if bot_y is not None else wrist_y, wrist_y)

            # Once the wrist rises back up enough, re-arm for the next downstroke
            if (bot_y - wrist_y) >= self.UP_THRESHOLD:
                setattr(self, phase_attr, "ready_for_down")
                top_y = wrist_y
                bot_y = wrist_y

        # Save state back to instance attributes
        setattr(self, top_attr, top_y)
        setattr(self, bot_attr, bot_y)
        return pumped

    def _update_beat(self, wrist_y1, wrist_y2, confirmed1, confirmed2, now):
        """
        Called every frame during the countdown. Runs pump detection on both
        hands and advances beat_count when both players have pumped in sync.

        Both players must contribute a pump — one player alone doesn't count.
        We use timestamps from the most recent pump per hand, so players don't
        have to hit the exact same frame, just within SYNC_WINDOW of each other.

        Returns True if a new beat was registered this frame, False otherwise.
        """
        # A hand is "active" only if Rock is visible and the wrist position is known
        p1_active = wrist_y1 is not None and confirmed1 == "Rock"
        p2_active = wrist_y2 is not None and confirmed2 == "Rock"

        # Update the "Rock seen" timestamp for the grace period logic
        if confirmed1 == "Rock" or confirmed2 == "Rock":
            self.last_rock_time = now

        # Only process beats while Rock was seen recently (or within grace window)
        grace_ok    = (now - self.last_rock_time) <= self.ROCK_GRACE_PERIOD
        cooldown_ok = (now - self.last_beat_time) >= self.BEAT_COOLDOWN

        if not grace_ok:
            return False

        # Run each hand's individual pump detector
        p1_pumped = False
        p2_pumped = False

        if p1_active:
            p1_pumped = self._track_hand_pump(
                wrist_y1, "_p1_phase", "_p1_top_y", "_p1_bot_y", now)
            if p1_pumped:
                self._p1_pump_t = now  # record when P1 last pumped

        if p2_active:
            p2_pumped = self._track_hand_pump(
                wrist_y2, "_p2_phase", "_p2_top_y", "_p2_bot_y", now)
            if p2_pumped:
                self._p2_pump_t = now  # record when P2 last pumped

        # Count a beat only if cooldown has elapsed AND both players pumped recently
        if cooldown_ok:
            p1_recent = (now - self._p1_pump_t) <= self.SYNC_WINDOW
            p2_recent = (now - self._p2_pump_t) <= self.SYNC_WINDOW
            if p1_recent and p2_recent and (p1_pumped or p2_pumped):
                self.beat_count    += 1
                self.last_beat_time = now
                # Reset both timestamps so neither pump double-counts next frame
                self._p1_pump_t = 0.0
                self._p2_pump_t = 0.0
                return True

        return False


# ─────────────────────────────────────────────────────────────────────────────
# Two-Player PvP Controller
# ─────────────────────────────────────────────────────────────────────────────

class TwoPlayerPvPController(_BeatMixin):
    """
    Player 1 (left hand) vs Player 2 (right hand). No AI. Best-of-N rounds.

    State flow:
      ROUND_INTRO -> WAITING_FOR_ROCK -> COUNTDOWN -> SHOOT_WINDOW
        -> ROUND_RESULT -> (MATCH_RESULT or next round)

    Beat 4 opens the SHOOT window. Both players lock in their throw
    independently as soon as a valid gesture is detected.
    """

    def __init__(
        self,
        robot_output=None,
        win_target=3,
        beat_cooldown=0.18,
        shoot_window_seconds=1.20,  # wider than solo — two players need more time to react
        rock_assume_seconds=0.14,
        round_intro_seconds=1.00,
        round_result_seconds=2.40,
        match_result_seconds=2.40,
    ):
        self.robot_output         = robot_output
        self.win_target           = win_target
        # Override the mixin's class-level defaults with whatever was passed in
        self.BEAT_COOLDOWN        = beat_cooldown
        self.SHOOT_WINDOW         = shoot_window_seconds
        self.ROCK_ASSUME          = rock_assume_seconds
        self.round_intro_seconds  = round_intro_seconds
        self.round_result_seconds = round_result_seconds
        self.match_result_seconds = match_result_seconds
        self._voice_mode = False
        self.reset_match()

    def reset(self):
        """Full reset — same as starting a brand-new match from scratch."""
        self.reset_match()

    def set_voice_mode(self, enabled: bool):
        """Toggle voice-command mode (voice input replaces gesture input)."""
        self._voice_mode = enabled

    def reset_match(self, now=None):
        """Reset scores and start round 1. Called at init and after a match ends."""
        if now is None:
            now = time.monotonic()
        self.p1_score            = 0
        self.p2_score            = 0
        self.round_number        = 1
        self.history: list[dict] = []  # record of every round played, used for post-game review
        self.match_result_banner = ""
        self.match_until         = None  # timestamp when the match-result screen expires
        self._reset_round(now)

    def _reset_round(self, now=None):
        """Reset per-round state while keeping match scores intact."""
        if now is None:
            now = time.monotonic()
        self._init_beat()
        self.p1_gesture          = "Unknown"
        self.p2_gesture          = "Unknown"
        self._p1_shoot_locked    = None  # P1's locked-in throw (None = not thrown yet)
        self._p2_shoot_locked    = None  # P2's locked-in throw
        self.result_banner       = ""
        self.last_round_result   = None
        self.result_until        = None
        self.tracker_reset_requested = False
        self.state               = "ROUND_INTRO"
        self.intro_until         = now + self.round_intro_seconds

    def consume_tracker_reset_request(self):
        """
        Called by the hand-tracker after it clears its state.
        Clears the flag so the reset only fires once.
        """
        self.tracker_reset_requested = False

    def _resolve_round(self, p1g, p2g, now):
        """
        Compare both gestures, update scores, and move to ROUND_RESULT.

        Uses compare_rps() from fair_play_state.py:
          'win'  = p1g beats p2g
          'lose' = p1g loses to p2g
          'draw' = tie
        """
        self.p1_gesture = p1g
        self.p2_gesture = p2g
        outcome = compare_rps(p1g, p2g)

        # Update the appropriate score and set the display banner
        if outcome == "win":
            self.p1_score     += 1
            self.result_banner = "PLAYER 1 WINS THE ROUND"
        elif outcome == "lose":
            self.p2_score     += 1
            self.result_banner = "PLAYER 2 WINS THE ROUND"
        else:
            self.result_banner = "DRAW - THROW AGAIN"

        self.last_round_result = outcome

        # Save this round to history so the UI can display a match recap
        self.history.append({
            "round_number": self.round_number,
            "p1_gesture":   p1g,
            "p2_gesture":   p2g,
            "outcome":      outcome,
        })

        self.state        = "ROUND_RESULT"
        self.result_until = now + self.round_result_seconds

    def _round_is_over(self):
        """Return True if either player has reached the win target."""
        return self.p1_score >= self.win_target or self.p2_score >= self.win_target

    def _build_output(self, now):
        """
        Build the output dict that the UI renderer reads every frame.

        Includes compatibility keys so the existing single-player renderer can
        display this mode without needing its own special code path.
        """
        score_text = f"P1  {self.p1_score}  -  {self.p2_score}  P2"
        round_text = f"ROUND {self.round_number}"

        base = {
            "play_mode_label":        "2 Player PvP",
            "state":                  self.state,
            "beat_count":             self.beat_count,
            "time_left":              0.0,
            "p1_gesture":             self.p1_gesture,
            "p2_gesture":             self.p2_gesture,
            # Single-player renderer compatibility aliases
            "player_gesture":         self.p1_gesture,
            "computer_gesture":       self.p2_gesture,
            "robot_move_command":     "PENDING",
            "result_banner":          self.result_banner,
            "score_text":             score_text,
            "round_text":             round_text,
            "round_number":           self.round_number,
            "player_score":           self.p1_score,
            "robot_score":            self.p2_score,
            "request_tracker_reset":  self.tracker_reset_requested,
            "opponent_type":          "",
            "reaction_ms":            None,
            "last_player_gesture":    None,
            "last_robot_gesture":     None,
            "last_banner":            "",
            "session_reaction_times": [],
            "session_gestures":       [],
            "streak_label":           "",
            "p1_name":                "PLAYER 1",
            "p2_name":                "PLAYER 2",
            "two_player":             True,
            "coop_mode":              False,
            # Per-player throw status during SHOOT_WINDOW (None = not thrown yet)
            "p1_shoot_locked":        getattr(self, "_p1_shoot_locked", None),
            "p2_shoot_locked":        getattr(self, "_p2_shoot_locked", None),
        }

        # Add state-specific display strings based on what's happening right now
        if self.state == "ROUND_INTRO":
            base.update({"state_label": "Round Intro",
                         "main_text":   round_text,
                         "sub_text":    f"FIRST TO {self.win_target} | {score_text}"})
        elif self.state == "WAITING_FOR_ROCK":
            base.update({"state_label": "Waiting",
                         "main_text":   "BOTH MAKE A FIST",
                         "sub_text":    f"{round_text} | {score_text}"})
        elif self.state == "COUNTDOWN":
            # Show "READY" before the first beat, then count up 1-2-3
            main = "READY" if self.beat_count == 0 else str(min(self.beat_count, 3))
            base.update({"state_label": "Countdown",
                         "main_text":   main,
                         "sub_text":    "Pump together!"})
        elif self.state == "SHOOT_WINDOW":
            # Show a live countdown of the remaining throw window
            time_left = max(0.0, self.shoot_close_time - now) if self.shoot_close_time else 0.0
            base.update({"state_label": "Shoot Window",
                         "main_text":   "SHOOT!",
                         "sub_text":    "Both throw NOW",
                         "time_left":   time_left})
        elif self.state == "ROUND_RESULT":
            base.update({"state_label": "Round Result",
                         "main_text":   self.result_banner,
                         "sub_text":    score_text,
                         "time_left":   max(0.0, self.result_until - now)})
        elif self.state == "MATCH_RESULT":
            base.update({"state_label":   "Match Result",
                         "main_text":     self.match_result_banner,
                         "sub_text":      f"FINAL | {score_text}",
                         "result_banner": self.match_result_banner,
                         "time_left":     max(0.0, self.match_until - now)})
        else:
            base.update({"state_label": "Unknown", "main_text": "UNKNOWN", "sub_text": ""})

        return base

    def update(self, p1_tracker_state, p2_tracker_state,
               p1_wrist_y=None, p2_wrist_y=None, now=None):
        """
        Main frame update. Called every frame by the game loop.

        p1 = left hand, p2 = right hand.
        tracker_state dicts come from the hand-tracking pipeline and must
        contain 'confirmed_gesture' and 'stable_gesture' keys.
        """
        if now is None:
            now = time.monotonic()

        # Pull the gesture readings we need from each tracker
        p1_conf = p1_tracker_state.get("confirmed_gesture", "Unknown")
        p2_conf = p2_tracker_state.get("confirmed_gesture", "Unknown")
        p1_stab = p1_tracker_state.get("stable_gesture",   "Unknown")
        p2_stab = p2_tracker_state.get("stable_gesture",   "Unknown")

        # ROUND_INTRO: show "Round N" briefly before the countdown starts
        if self.state == "ROUND_INTRO":
            if now >= self.intro_until:
                self.state = "WAITING_FOR_ROCK"
            return self._build_output(now)

        # ROUND_RESULT: show the outcome, then advance to the next round or end the match
        if self.state == "ROUND_RESULT":
            if now >= self.result_until:
                if self._round_is_over():
                    # Someone reached win_target — declare a match winner
                    winner = "PLAYER 1" if self.p1_score >= self.win_target else "PLAYER 2"
                    self.match_result_banner = f"{winner} WINS THE MATCH"
                    self.state       = "MATCH_RESULT"
                    self.match_until = now + self.match_result_seconds
                else:
                    # Don't advance the round number on a draw — replay the same round
                    if self.last_round_result != "draw":
                        self.round_number += 1
                    self._reset_round(now)
            return self._build_output(now)

        # MATCH_RESULT: display the winner banner, then auto-reset for a new match
        if self.state == "MATCH_RESULT":
            if now >= self.match_until:
                self.reset_match(now)
            return self._build_output(now)

        # WAITING_FOR_ROCK: hold here until both players form a fist
        if self.state == "WAITING_FOR_ROCK":
            # Accept Rock from either confirmed or stable gesture, or assume Rock if
            # the wrist was visible and showed Rock very recently (within ROCK_ASSUME seconds)
            p1_rock = (p1_conf == "Rock" or p1_stab == "Rock" or
                       (p1_wrist_y is not None and
                        (now - self.last_rock_time) < self.ROCK_ASSUME))
            p2_rock = (p2_conf == "Rock" or p2_stab == "Rock" or
                       (p2_wrist_y is not None and
                        (now - self.last_rock_time) < self.ROCK_ASSUME))
            if p1_rock and p2_rock:
                self.last_rock_time = now
                self.state = "COUNTDOWN"
                self._init_beat()  # fresh beat detector for this countdown
            return self._build_output(now)

        # COUNTDOWN: count pump beats until beat 4 triggers the SHOOT window
        if self.state == "COUNTDOWN":
            self._update_beat(p1_wrist_y, p2_wrist_y, p1_conf, p2_conf, now)
            if self.beat_count >= 4:
                # Beat 4 = SHOOT — open the throw window
                self.state            = "SHOOT_WINDOW"
                self.shoot_open_time  = now
                self.shoot_close_time = now + self.SHOOT_WINDOW
                # Ask the hand tracker to flush its history so SHOOT reads fresh gestures
                self.tracker_reset_requested = True
            return self._build_output(now)

        # SHOOT_WINDOW: lock in each player's throw as soon as it's detected
        if self.state == "SHOOT_WINDOW":
            # Use confirmed gesture first; fall back to stable if not yet confirmed
            p1_thrown = (p1_conf if p1_conf in VALID_GESTURES else
                         (p1_stab if p1_stab in VALID_GESTURES else None))
            p2_thrown = (p2_conf if p2_conf in VALID_GESTURES else
                         (p2_stab if p2_stab in VALID_GESTURES else None))

            # Lock in the first gesture detected — players can't change their throw
            if p1_thrown and self._p1_shoot_locked is None:
                self._p1_shoot_locked = p1_thrown
            if p2_thrown and self._p2_shoot_locked is None:
                self._p2_shoot_locked = p2_thrown

            # Resolve when both players have thrown, or when the window timer runs out
            time_up = now >= self.shoot_close_time
            if time_up or (self._p1_shoot_locked and self._p2_shoot_locked):
                # Default to Rock if a player never threw (rare — window expired without a gesture)
                p1g = self._p1_shoot_locked or "Rock"
                p2g = self._p2_shoot_locked or "Rock"
                self._resolve_round(p1g, p2g, now)
            return self._build_output(now)

        return self._build_output(now)


# ─────────────────────────────────────────────────────────────────────────────
# Three-Way PvPvAI Controller
# ─────────────────────────────────────────────────────────────────────────────

class PvPvAIController(_BeatMixin):
    """
    1v1v1: Player 1 vs Player 2 vs FairPlayAI — everyone plays for themselves.

    Scoring per round:
      Beat 1 opponent  -> +1 point
      Beat 2 opponents -> +2 points
      3-way draw       -> no points for anyone
      2-way draw       -> the non-drawing player still scores normally

    First to win_target (default 5) points wins the match.

    The AI commits to its move at beat 3 (before the SHOOT window opens),
    so it's locked in and can't react to what the humans throw.
    """

    def __init__(
        self,
        robot_output=None,
        ai=None,
        win_target=5,
        beat_cooldown=0.18,
        shoot_window_seconds=1.20,
        rock_assume_seconds=0.14,
        round_intro_seconds=1.00,
        round_result_seconds=2.80,
        match_result_seconds=2.40,
    ):
        self.robot_output         = robot_output
        self.ai                   = ai or FairPlayAI()
        self.win_target           = win_target
        self.BEAT_COOLDOWN        = beat_cooldown
        self.SHOOT_WINDOW         = shoot_window_seconds
        self.ROCK_ASSUME          = rock_assume_seconds
        self.round_intro_seconds  = round_intro_seconds
        self.round_result_seconds = round_result_seconds
        self.match_result_seconds = match_result_seconds
        self._voice_mode = False
        self.reset_match()

    def reset(self):
        """Full reset — same as starting a brand-new match."""
        self.reset_match()

    def set_voice_mode(self, enabled: bool):
        """Toggle voice-command mode."""
        self._voice_mode = enabled

    def reset_match(self, now=None):
        """Reset all scores and start round 1."""
        if now is None:
            now = time.monotonic()
        self.p1_score     = 0
        self.p2_score     = 0
        self.ai_score     = 0
        self.round_number = 1
        # Separate histories let the AI model each player's tendencies independently
        self.p1_history: list[dict] = []
        self.p2_history: list[dict] = []
        self.match_result_banner = ""
        self.match_until = None
        self.ai.reset()
        self._reset_round(now)

    def _reset_round(self, now=None):
        """Reset per-round state for the next round."""
        if now is None:
            now = time.monotonic()
        self._init_beat()
        self.p1_gesture        = "Unknown"
        self.p2_gesture        = "Unknown"
        self.ai_gesture        = "Unknown"
        self.ai_locked         = None  # AI's chosen move (set at beat 3, before SHOOT opens)
        self.result_banner     = ""
        self.last_round_result = None
        # Points each participant earned this round (for the result display)
        self.p1_pts_this_round = 0
        self.p2_pts_this_round = 0
        self.ai_pts_this_round = 0
        self.result_until      = None
        self.tracker_reset_requested = False
        self.state       = "ROUND_INTRO"
        self.intro_until = now + self.round_intro_seconds

    def consume_tracker_reset_request(self):
        """Called by the hand-tracker after it clears state. Clears the request flag."""
        self.tracker_reset_requested = False

    def _lock_ai(self):
        """
        Ask the AI to choose its move for this round and store it in ai_locked.

        The AI looks at both players' histories to predict what each will throw,
        then picks whichever gesture scores the most expected points against them.
        We call this at beat 3, before the SHOOT window opens, so the AI commits
        before it can see the humans' gestures — keeps things fair.
        """
        if self.ai_locked is not None:
            return  # already locked this round, don't call again

        # Ask the AI to predict what each player is likely to throw
        pred1 = self.ai.choose_robot_move(history=self.p1_history, round_number=self.round_number)
        pred2 = self.ai.choose_robot_move(history=self.p2_history, round_number=self.round_number)

        # Try all three gestures and pick the one that earns the most total points
        GESTURES = ("Rock", "Paper", "Scissors")
        best_move, best_pts = pred1, -1
        for g in GESTURES:
            # Count how many of the two predictions this gesture beats
            pts = ((1 if compare_rps(g, pred1) == "win" else 0) +
                   (1 if compare_rps(g, pred2) == "win" else 0))
            if pts > best_pts:
                best_pts, best_move = pts, g

        self.ai_locked = best_move

    @staticmethod
    def _score_three_way(g1, g2, g3):
        """
        Calculate points for a 3-player round.

        Each player earns 1 point for each opponent they beat.
        In a 3-way draw (all same gesture), everyone scores 0.

        Returns (pts1, pts2, pts3).
        """
        # Each participant checks their gesture against both opponents
        pts1 = ((1 if compare_rps(g1, g2) == "win" else 0) +
                (1 if compare_rps(g1, g3) == "win" else 0))
        pts2 = ((1 if compare_rps(g2, g1) == "win" else 0) +
                (1 if compare_rps(g2, g3) == "win" else 0))
        pts3 = ((1 if compare_rps(g3, g1) == "win" else 0) +
                (1 if compare_rps(g3, g2) == "win" else 0))
        return pts1, pts2, pts3

    def _resolve_round(self, p1g, p2g, now):
        """Score the round, update histories, and transition to ROUND_RESULT."""
        self.p1_gesture = p1g
        self.p2_gesture = p2g
        self.ai_gesture = self.ai_locked or "Rock"  # fall back to Rock if AI never locked in

        # Calculate how many points each participant earns this round
        p1p, p2p, aip = self._score_three_way(p1g, p2g, self.ai_gesture)
        self.p1_pts_this_round = p1p
        self.p2_pts_this_round = p2p
        self.ai_pts_this_round = aip

        self.p1_score += p1p
        self.p2_score += p2p
        self.ai_score += aip

        self.last_round_result = (p1p, p2p, aip)

        # Build the result banner showing who scored and how much
        if p1p == 0 and p2p == 0 and aip == 0:
            self.result_banner = "3-WAY DRAW  -  NO POINTS"
        else:
            # Only list participants who actually scored
            parts = []
            if p1p: parts.append(f"P1 +{p1p}")
            if p2p: parts.append(f"P2 +{p2p}")
            if aip: parts.append(f"AI +{aip}")
            self.result_banner = "  |  ".join(parts)

        # Update the AI's prediction history for each player separately.
        # A player "won" if they beat at least one opponent this round.
        for hist, pg, opp1, opp2 in [
            (self.p1_history, p1g, p2g, self.ai_gesture),
            (self.p2_history, p2g, p1g, self.ai_gesture),
        ]:
            outcome = ("win" if compare_rps(pg, opp1) == "win" or
                                compare_rps(pg, opp2) == "win" else "lose")
            hist.append({
                "round_number":   self.round_number,
                "player_gesture": pg,
                "player_outcome": outcome,
            })

        self.state        = "ROUND_RESULT"
        self.result_until = now + self.round_result_seconds

    def _round_is_over(self):
        """Return True if any participant has reached the win target."""
        return (self.p1_score >= self.win_target or
                self.p2_score >= self.win_target or
                self.ai_score >= self.win_target)

    def _match_winner_text(self):
        """Return the match-winner announcement string."""
        if self.p1_score >= self.win_target:
            return "PLAYER 1 WINS!"
        if self.p2_score >= self.win_target:
            return "PLAYER 2 WINS!"
        return "AI WINS THE MATCH"

    def _build_output(self, now):
        """Build the output dict that the UI renderer reads every frame."""
        score_text = f"P1: {self.p1_score}  |  AI: {self.ai_score}  |  P2: {self.p2_score}"
        round_text = f"ROUND {self.round_number}"
        pred = getattr(self.ai, "last_prediction", None) or {}

        base = {
            "play_mode_label":        "PvPvAI",
            "state":                  self.state,
            "beat_count":             self.beat_count,
            "time_left":              0.0,
            "p1_gesture":             self.p1_gesture,
            "p2_gesture":             self.p2_gesture,
            "ai_gesture":             self.ai_gesture,
            # Single-player renderer compatibility aliases
            "player_gesture":         self.p1_gesture,
            "computer_gesture":       self.ai_gesture,
            # Send the robot command once the AI locks in so the physical robot can prepare
            "robot_move_command":     (f"ROBOT_PLAY_{self.ai_locked.upper()}"
                                       if self.ai_locked else "PENDING"),
            "result_banner":          self.result_banner,
            "score_text":             score_text,
            "round_text":             round_text,
            "round_number":           self.round_number,
            "player_score":           self.p1_score,
            "robot_score":            self.ai_score,
            "p1_score":               self.p1_score,
            "p2_score":               self.p2_score,
            "ai_score":               self.ai_score,
            "p1_pts_this_round":      self.p1_pts_this_round,
            "p2_pts_this_round":      self.p2_pts_this_round,
            "ai_pts_this_round":      self.ai_pts_this_round,
            "win_target":             self.win_target,
            "request_tracker_reset":  self.tracker_reset_requested,
            "opponent_type":          pred.get("opponent_type", ""),
            "reaction_ms":            None,
            "last_player_gesture":    None,
            "last_robot_gesture":     None,
            "last_banner":            "",
            "session_reaction_times": [],
            "session_gestures":       [],
            "streak_label":           "",
            "two_player":             True,
            "coop_mode":              False,
        }

        # Add state-specific display strings
        if self.state == "ROUND_INTRO":
            base.update({"state_label": "Round Intro",
                         "main_text":   round_text,
                         "sub_text":    f"First to {self.win_target} pts | {score_text}"})
        elif self.state == "WAITING_FOR_ROCK":
            base.update({"state_label": "Waiting",
                         "main_text":   "BOTH MAKE A FIST",
                         "sub_text":    f"{round_text} | {score_text}"})
        elif self.state == "COUNTDOWN":
            main = "READY" if self.beat_count == 0 else str(min(self.beat_count, 3))
            base.update({"state_label": "Countdown",
                         "main_text":   main,
                         "sub_text":    "AI locks on beat 3"})
        elif self.state == "SHOOT_WINDOW":
            time_left = max(0.0, self.shoot_close_time - now) if self.shoot_close_time else 0.0
            base.update({"state_label": "Shoot Window",
                         "main_text":   "SHOOT!",
                         "sub_text":    "Everyone throws NOW",
                         "time_left":   time_left})
        elif self.state == "ROUND_RESULT":
            base.update({"state_label": "Round Result",
                         "main_text":   self.result_banner,
                         "sub_text":    score_text,
                         "time_left":   max(0.0, self.result_until - now)})
        elif self.state == "MATCH_RESULT":
            base.update({"state_label":   "Match Result",
                         "main_text":     self.match_result_banner,
                         "sub_text":      f"FINAL | {score_text}",
                         "result_banner": self.match_result_banner,
                         "time_left":     max(0.0, self.match_until - now)})
        else:
            base.update({"state_label": "Unknown", "main_text": "UNKNOWN", "sub_text": ""})

        return base

    def update(self, p1_tracker_state, p2_tracker_state,
               p1_wrist_y=None, p2_wrist_y=None, now=None):
        """
        Main frame update. Called every frame by the game loop.

        Mirrors TwoPlayerPvPController.update() but also manages the AI's move
        (locked in at beat 3) and uses 3-way scoring in _resolve_round().
        """
        if now is None:
            now = time.monotonic()

        p1_conf = p1_tracker_state.get("confirmed_gesture", "Unknown")
        p2_conf = p2_tracker_state.get("confirmed_gesture", "Unknown")
        p1_stab = p1_tracker_state.get("stable_gesture",   "Unknown")
        p2_stab = p2_tracker_state.get("stable_gesture",   "Unknown")

        # ROUND_INTRO: brief display of the round number before play starts
        if self.state == "ROUND_INTRO":
            if now >= self.intro_until:
                self.state = "WAITING_FOR_ROCK"
            return self._build_output(now)

        # ROUND_RESULT: show the outcome, then advance to next round or end the match
        if self.state == "ROUND_RESULT":
            if now >= self.result_until:
                if self._round_is_over():
                    self.match_result_banner = self._match_winner_text()
                    self.state       = "MATCH_RESULT"
                    self.match_until = now + self.match_result_seconds
                else:
                    # PvPvAI has no draws — always increment the round number
                    self.round_number += 1
                    self._reset_round(now)
            return self._build_output(now)

        # MATCH_RESULT: display the winner, then auto-reset for a new match
        if self.state == "MATCH_RESULT":
            if now >= self.match_until:
                self.reset_match(now)
            return self._build_output(now)

        # WAITING_FOR_ROCK: both human players must show a fist to start
        if self.state == "WAITING_FOR_ROCK":
            p1_rock = p1_conf == "Rock" or p1_stab == "Rock"
            p2_rock = p2_conf == "Rock" or p2_stab == "Rock"
            if p1_rock and p2_rock:
                self.last_rock_time = now
                self.state = "COUNTDOWN"
                self._init_beat()
            return self._build_output(now)

        # COUNTDOWN: pump beats 1-3; AI locks in at beat 3; SHOOT opens at beat 4
        if self.state == "COUNTDOWN":
            self._update_beat(p1_wrist_y, p2_wrist_y, p1_conf, p2_conf, now)
            if self.beat_count >= 3:
                self._lock_ai()  # AI picks its move before the SHOOT window opens
            if self.beat_count >= 4:
                self.state            = "SHOOT_WINDOW"
                self.shoot_open_time  = now
                self.shoot_close_time = now + self.SHOOT_WINDOW
                self.tracker_reset_requested = True
            return self._build_output(now)

        # SHOOT_WINDOW: lock in throws, resolve when both humans have thrown or time runs out
        if self.state == "SHOOT_WINDOW":
            p1_thrown = (p1_conf if p1_conf in VALID_GESTURES else
                         (p1_stab if p1_stab in VALID_GESTURES else None))
            p2_thrown = (p2_conf if p2_conf in VALID_GESTURES else
                         (p2_stab if p2_stab in VALID_GESTURES else None))
            if now >= self.shoot_close_time or (p1_thrown and p2_thrown):
                self._resolve_round(p1_thrown or "Rock", p2_thrown or "Rock", now)
            return self._build_output(now)

        return self._build_output(now)

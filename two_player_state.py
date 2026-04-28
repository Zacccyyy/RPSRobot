"""
two_player_state.py
===================
Two-player game modes:

  TwoPlayerPvPController   — Player 1 (left hand) vs Player 2 (right hand).
                             No AI.  Best-of-N match.

  CoopVsAIController       — Player 1 + Player 2 team vs FairPlayAI.
                             Both humans throw simultaneously; AI throws one
                             move.  Round outcomes:
                               • Both beat AI  → human team scores
                               • AI beats both → AI scores
                               • Split         → draw (round replays)

Both controllers share the same pump-based beat-detection state machine as
FairPlayController.  Beat advances when EITHER player's wrist pumps.
"""

import time
from fair_play_state import compare_rps
from fair_play_ai    import FairPlayAI

# ── Constants ─────────────────────────────────────────────────────────────────
VALID_GESTURES = frozenset({"Rock", "Paper", "Scissors"})

# Rock-Paper-Scissors cycle for response-type labelling
UPGRADE   = {"Rock": "Paper",    "Paper": "Scissors", "Scissors": "Rock"}
DOWNGRADE = {"Rock": "Scissors", "Paper": "Rock",     "Scissors": "Paper"}


# ═══════════════════════════════════════════════════════════════════════════════
# Shared beat-detection mixin
# ═══════════════════════════════════════════════════════════════════════════════
class _BeatMixin:
    """
    Pump-based countdown beat detection for two-player modes.

    Each hand's pump is tracked independently.  A beat is counted only
    when BOTH hands have each completed a pump stroke within SYNC_WINDOW
    seconds of each other.  This means players must pump together —
    one player carrying the countdown alone is impossible.

    If players fall out of sync mid-countdown the count pauses at its
    current value (no reset) until they sync up again on the next pump.
    """

    DOWN_THRESHOLD = 0.035   # slightly more sensitive — easier to trigger
    UP_THRESHOLD   = 0.025   # recovery threshold (just needs to come back up a little)
    BEAT_COOLDOWN  = 0.20    # minimum time between beats
    SYNC_WINDOW    = 0.65    # widened — hands pumping within 0.65s counts as together
    ROCK_GRACE_PERIOD = 0.50
    SHOOT_WINDOW   = 0.55
    ROCK_ASSUME    = 0.14

    def _init_beat(self):
        self.beat_count       = 0
        self.last_beat_time   = 0.0
        self.last_rock_time   = 0.0
        self.shoot_open_time  = None
        self.shoot_close_time = None
        # Independent pump state per hand
        self._p1_phase    = "ready_for_down"
        self._p1_top_y    = None
        self._p1_bot_y    = None
        self._p1_pump_t   = 0.0   # timestamp of last completed p1 pump
        self._p2_phase    = "ready_for_down"
        self._p2_top_y    = None
        self._p2_bot_y    = None
        self._p2_pump_t   = 0.0   # timestamp of last completed p2 pump

    def _track_hand_pump(self, wrist_y, phase_attr, top_attr, bot_attr, now):
        """
        Track one hand's pump state machine.
        Returns True the moment a downstroke is detected (hand moves DOWN).
        Counting on the down matches natural RPS rhythm: 1-2-3-SHOOT.
        The up-recovery just resets the tracker ready for the next down.
        """
        phase = getattr(self, phase_attr)
        top_y = getattr(self, top_attr)
        bot_y = getattr(self, bot_attr)

        pumped = False

        if phase == "ready_for_down":
            if top_y is None:
                top_y = wrist_y
            top_y = min(top_y, wrist_y)
            # Downstroke detected — fire the beat NOW, then wait for recovery
            if (wrist_y - top_y) >= self.DOWN_THRESHOLD:
                setattr(self, phase_attr, "waiting_for_up")
                bot_y = wrist_y
                pumped = True   # ← beat fires on the DOWN, not the up

        elif phase == "waiting_for_up":
            bot_y = max(bot_y if bot_y is not None else wrist_y, wrist_y)
            # Recovery up — reset ready for next downstroke
            if (bot_y - wrist_y) >= self.UP_THRESHOLD:
                setattr(self, phase_attr, "ready_for_down")
                top_y = wrist_y
                bot_y = wrist_y

        setattr(self, top_attr, top_y)
        setattr(self, bot_attr, bot_y)
        return pumped

    def _update_beat(self, wrist_y1, wrist_y2, confirmed1, confirmed2, now):
        """
        Update beat counter.  Both hands must independently pump within
        SYNC_WINDOW seconds of each other to advance the beat count.
        """
        p1_active = wrist_y1 is not None and confirmed1 in ("Rock", "Unknown")
        p2_active = wrist_y2 is not None and confirmed2 in ("Rock", "Unknown")

        rock_held = confirmed1 == "Rock" or confirmed2 == "Rock"
        if rock_held:
            self.last_rock_time = now

        grace_ok    = (now - self.last_rock_time) <= self.ROCK_GRACE_PERIOD
        cooldown_ok = (now - self.last_beat_time) >= self.BEAT_COOLDOWN

        if not grace_ok:
            return False

        # Track each hand independently
        p1_pumped = False
        p2_pumped = False

        if p1_active:
            p1_pumped = self._track_hand_pump(
                wrist_y1, "_p1_phase", "_p1_top_y", "_p1_bot_y", now)
            if p1_pumped:
                self._p1_pump_t = now

        if p2_active:
            p2_pumped = self._track_hand_pump(
                wrist_y2, "_p2_phase", "_p2_top_y", "_p2_bot_y", now)
            if p2_pumped:
                self._p2_pump_t = now

        # Count a beat only when both hands have pumped within SYNC_WINDOW
        # and the cooldown has elapsed since the last beat
        if cooldown_ok:
            p1_recent = (now - self._p1_pump_t) <= self.SYNC_WINDOW
            p2_recent = (now - self._p2_pump_t) <= self.SYNC_WINDOW
            if p1_recent and p2_recent and (p1_pumped or p2_pumped):
                self.beat_count   += 1
                self.last_beat_time = now
                # Clear pump timestamps so they can't double-count
                self._p1_pump_t = 0.0
                self._p2_pump_t = 0.0
                return True

        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Two-Player PvP Controller
# ═══════════════════════════════════════════════════════════════════════════════
class TwoPlayerPvPController(_BeatMixin):
    """
    Player 1 (left hand) vs Player 2 (right hand).
    Best-of win_target rounds.  No AI.
    Beat 4 opens SHOOT (same timing as FairPlayController).
    """

    def __init__(
        self,
        robot_output=None,
        win_target=3,
        beat_cooldown=0.18,
        shoot_window_seconds=1.20,   # wider than solo (was 0.55) — two players need more time
        rock_assume_seconds=0.14,
        round_intro_seconds=1.00,
        round_result_seconds=2.40,
        match_result_seconds=2.40,
    ):
        self.robot_output          = robot_output
        self.win_target            = win_target
        self.BEAT_COOLDOWN         = beat_cooldown
        self.SHOOT_WINDOW          = shoot_window_seconds
        self.ROCK_ASSUME           = rock_assume_seconds
        self.round_intro_seconds   = round_intro_seconds
        self.round_result_seconds  = round_result_seconds
        self.match_result_seconds  = match_result_seconds

        self._voice_mode = False
        self.reset_match()

    def reset(self):
        self.reset_match()

    def set_voice_mode(self, enabled: bool):
        self._voice_mode = enabled

    def reset_match(self, now=None):
        if now is None:
            now = time.monotonic()
        self.p1_score = 0
        self.p2_score = 0
        self.round_number = 1
        self.history: list[dict] = []
        self.match_result_banner = ""
        self.match_until = None
        self._reset_round(now)

    def _reset_round(self, now=None):
        if now is None:
            now = time.monotonic()
        self._init_beat()
        self.p1_gesture = "Unknown"
        self.p2_gesture = "Unknown"
        self.result_banner = ""
        self.last_round_result = None
        self.result_until = None
        self.tracker_reset_requested = False
        self.state = "ROUND_INTRO"
        self.intro_until = now + self.round_intro_seconds

    def consume_tracker_reset_request(self):
        self.tracker_reset_requested = False

    def _resolve_round(self, p1g, p2g, now):
        """Compare both gestures and record the round result."""
        self.p1_gesture = p1g
        self.p2_gesture = p2g
        outcome = compare_rps(p1g, p2g)

        if outcome == "win":
            self.p1_score += 1
            self.result_banner = "PLAYER 1 WINS THE ROUND"
        elif outcome == "lose":
            self.p2_score += 1
            self.result_banner = "PLAYER 2 WINS THE ROUND"
        else:
            self.result_banner = "DRAW - THROW AGAIN"

        self.last_round_result = outcome
        self.history.append({
            "round_number": self.round_number,
            "p1_gesture":   p1g,
            "p2_gesture":   p2g,
            "outcome":      outcome,
        })
        self.state        = "ROUND_RESULT"
        self.result_until = now + self.round_result_seconds

    def _round_is_over(self):
        return self.p1_score >= self.win_target or self.p2_score >= self.win_target

    def _build_output(self, now):
        score_text = f"P1  {self.p1_score}  -  {self.p2_score}  P2"
        round_text = f"ROUND {self.round_number}"
        base = {
            "play_mode_label":        "2 Player PvP",
            "state":                  self.state,
            "beat_count":             self.beat_count,
            "time_left":              0.0,
            "p1_gesture":             self.p1_gesture,
            "p2_gesture":             self.p2_gesture,
            # Compat keys expected by single-player renderer
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
        }

        if self.state == "ROUND_INTRO":
            base.update({"state_label": "Round Intro",
                         "main_text":   round_text,
                         "sub_text":    f"FIRST TO {self.win_target} | {score_text}"})
        elif self.state == "WAITING_FOR_ROCK":
            base.update({"state_label": "Waiting",
                         "main_text":   "BOTH MAKE A FIST",
                         "sub_text":    f"{round_text} | {score_text}"})
        elif self.state == "COUNTDOWN":
            mt = "READY" if self.beat_count == 0 else str(min(self.beat_count, 3))
            base.update({"state_label": "Countdown",
                         "main_text":   mt,
                         "sub_text":    "Pump together!"})
        elif self.state == "SHOOT_WINDOW":
            tl = max(0.0, self.shoot_close_time - now) if self.shoot_close_time else 0.0
            base.update({"state_label": "Shoot Window",
                         "main_text":   "SHOOT!",
                         "sub_text":    "Both throw NOW",
                         "time_left":   tl})
        elif self.state == "ROUND_RESULT":
            base.update({"state_label": "Round Result",
                         "main_text":   self.result_banner,
                         "sub_text":    score_text,
                         "time_left":   max(0.0, self.result_until - now)})
        elif self.state == "MATCH_RESULT":
            base.update({"state_label":    "Match Result",
                         "main_text":      self.match_result_banner,
                         "sub_text":       f"FINAL | {score_text}",
                         "result_banner":  self.match_result_banner,
                         "time_left":      max(0.0, self.match_until - now)})
        else:
            base.update({"state_label": "Unknown", "main_text": "UNKNOWN", "sub_text": ""})

        return base

    def update(self, p1_tracker_state, p2_tracker_state,
               p1_wrist_y=None, p2_wrist_y=None, now=None):
        """
        Called every frame.  p1 = left hand, p2 = right hand.
        """
        if now is None:
            now = time.monotonic()

        p1_conf = p1_tracker_state.get("confirmed_gesture", "Unknown")
        p2_conf = p2_tracker_state.get("confirmed_gesture", "Unknown")
        p1_stab = p1_tracker_state.get("stable_gesture",   "Unknown")
        p2_stab = p2_tracker_state.get("stable_gesture",   "Unknown")

        if self.state == "ROUND_INTRO":
            if now >= self.intro_until:
                self.state = "WAITING_FOR_ROCK"
            return self._build_output(now)

        if self.state == "ROUND_RESULT":
            if now >= self.result_until:
                if self._round_is_over():
                    winner = "PLAYER 1" if self.p1_score >= self.win_target else "PLAYER 2"
                    self.match_result_banner = f"{winner} WINS THE MATCH"
                    self.state       = "MATCH_RESULT"
                    self.match_until = now + self.match_result_seconds
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
            # Both need to show Rock (or be assumed Rock after timeout)
            p1_rock = p1_conf == "Rock" or p1_stab == "Rock" or \
                      (p1_wrist_y is not None and
                       (now - self.last_rock_time) < self.ROCK_ASSUME)
            p2_rock = p2_conf == "Rock" or p2_stab == "Rock" or \
                      (p2_wrist_y is not None and
                       (now - self.last_rock_time) < self.ROCK_ASSUME)
            if p1_rock and p2_rock:
                self.last_rock_time = now
                self.state = "COUNTDOWN"
                self._init_beat()
            return self._build_output(now)

        if self.state == "COUNTDOWN":
            self._update_beat(p1_wrist_y, p2_wrist_y, p1_conf, p2_conf, now)
            if self.beat_count >= 4:
                self.state           = "SHOOT_WINDOW"
                self.shoot_open_time  = now
                self.shoot_close_time = now + self.SHOOT_WINDOW
                self.tracker_reset_requested = True
            return self._build_output(now)

        if self.state == "SHOOT_WINDOW":
            p1_thrown = p1_conf if p1_conf in VALID_GESTURES else \
                        (p1_stab if p1_stab in VALID_GESTURES else None)
            p2_thrown = p2_conf if p2_conf in VALID_GESTURES else \
                        (p2_stab if p2_stab in VALID_GESTURES else None)

            # Both thrown or window expired
            time_up = now >= self.shoot_close_time
            if time_up or (p1_thrown and p2_thrown):
                p1g = p1_thrown or "Rock"
                p2g = p2_thrown or "Rock"
                self._resolve_round(p1g, p2g, now)
            return self._build_output(now)

        return self._build_output(now)


# ═══════════════════════════════════════════════════════════════════════════════
# Cooperative 2-vs-AI Controller
# ═══════════════════════════════════════════════════════════════════════════════
class PvPvAIController(_BeatMixin):
    """
    1v1v1: Player 1 vs Player 2 vs FairPlayAI — everyone for themselves.

    Scoring per round:
      Beat 1 opponent  → +1 point
      Beat 2 opponents → +2 points
      3-way draw       → +0 points (all same gesture)
      2-way draw       → the non-drawing player still scores normally

    First to win_target (default 5) points wins the match.
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
        self.robot_output          = robot_output
        self.ai                    = ai or FairPlayAI()
        self.win_target            = win_target
        self.BEAT_COOLDOWN         = beat_cooldown
        self.SHOOT_WINDOW          = shoot_window_seconds
        self.ROCK_ASSUME           = rock_assume_seconds
        self.round_intro_seconds   = round_intro_seconds
        self.round_result_seconds  = round_result_seconds
        self.match_result_seconds  = match_result_seconds
        self._voice_mode = False
        self.reset_match()

    def reset(self):
        self.reset_match()

    def set_voice_mode(self, enabled: bool):
        self._voice_mode = enabled

    def reset_match(self, now=None):
        if now is None:
            now = time.monotonic()
        self.p1_score     = 0
        self.p2_score     = 0
        self.ai_score     = 0
        self.round_number = 1
        self.p1_history: list[dict] = []
        self.p2_history: list[dict] = []
        self.match_result_banner = ""
        self.match_until = None
        self.ai.reset()
        self._reset_round(now)

    def _reset_round(self, now=None):
        if now is None:
            now = time.monotonic()
        self._init_beat()
        self.p1_gesture        = "Unknown"
        self.p2_gesture        = "Unknown"
        self.ai_gesture        = "Unknown"
        self.ai_locked         = None
        self.result_banner     = ""
        self.last_round_result = None
        self.p1_pts_this_round = 0
        self.p2_pts_this_round = 0
        self.ai_pts_this_round = 0
        self.result_until      = None
        self.tracker_reset_requested = False
        self.state       = "ROUND_INTRO"
        self.intro_until = now + self.round_intro_seconds

    def consume_tracker_reset_request(self):
        self.tracker_reset_requested = False

    def _lock_ai(self):
        """Pick the move that maximises expected points vs both players."""
        if self.ai_locked is not None:
            return
        pred1 = self.ai.choose_robot_move(
            history=self.p1_history, round_number=self.round_number)
        pred2 = self.ai.choose_robot_move(
            history=self.p2_history, round_number=self.round_number)
        # Score each possible AI move against both predictions
        GESTURES = ("Rock", "Paper", "Scissors")
        best_move, best_pts = pred1, -1
        for g in GESTURES:
            pts = (1 if compare_rps(g, pred1) == "win" else 0) +                   (1 if compare_rps(g, pred2) == "win" else 0)
            if pts > best_pts:
                best_pts, best_move = pts, g
        self.ai_locked = best_move

    @staticmethod
    def _score_three_way(g1, g2, g3):
        """
        Returns (pts1, pts2, pts3) points for each player in a 3-way round.
        Each player earns 1 point per opponent they beat.
        All three same gesture → all score 0.
        """
        pts1 = (1 if compare_rps(g1, g2) == "win" else 0) +                (1 if compare_rps(g1, g3) == "win" else 0)
        pts2 = (1 if compare_rps(g2, g1) == "win" else 0) +                (1 if compare_rps(g2, g3) == "win" else 0)
        pts3 = (1 if compare_rps(g3, g1) == "win" else 0) +                (1 if compare_rps(g3, g2) == "win" else 0)
        return pts1, pts2, pts3

    def _resolve_round(self, p1g, p2g, now):
        self.p1_gesture = p1g
        self.p2_gesture = p2g
        self.ai_gesture = self.ai_locked or "Rock"

        p1p, p2p, aip = self._score_three_way(p1g, p2g, self.ai_gesture)
        self.p1_pts_this_round = p1p
        self.p2_pts_this_round = p2p
        self.ai_pts_this_round = aip

        self.p1_score += p1p
        self.p2_score += p2p
        self.ai_score += aip

        self.last_round_result = (p1p, p2p, aip)

        # Build banner
        if p1p == 0 and p2p == 0 and aip == 0:
            self.result_banner = "3-WAY DRAW  -  NO POINTS"
        else:
            parts = []
            if p1p: parts.append(f"P1 +{p1p}")
            if p2p: parts.append(f"P2 +{p2p}")
            if aip: parts.append(f"AI +{aip}")
            self.result_banner = "  |  ".join(parts)

        # Update AI prediction histories
        for hist, pg, opp1, opp2 in [
            (self.p1_history, p1g, p2g, self.ai_gesture),
            (self.p2_history, p2g, p1g, self.ai_gesture),
        ]:
            outcome = "win" if compare_rps(pg, opp1) == "win" or                                compare_rps(pg, opp2) == "win" else "lose"
            hist.append({
                "round_number":   self.round_number,
                "player_gesture": pg,
                "player_outcome": outcome,
            })

        self.state        = "ROUND_RESULT"
        self.result_until = now + self.round_result_seconds

    def _round_is_over(self):
        return (self.p1_score >= self.win_target or
                self.p2_score >= self.win_target or
                self.ai_score >= self.win_target)

    def _match_winner_text(self):
        if self.p1_score >= self.win_target:
            return "PLAYER 1 WINS!"
        if self.p2_score >= self.win_target:
            return "PLAYER 2 WINS!"
        return "AI WINS THE MATCH"

    def _build_output(self, now):
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
            "player_gesture":         self.p1_gesture,
            "computer_gesture":       self.ai_gesture,
            "robot_move_command":     f"ROBOT_PLAY_{self.ai_locked.upper()}" if self.ai_locked else "PENDING",
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

        if self.state == "ROUND_INTRO":
            base.update({"state_label": "Round Intro",
                         "main_text":   round_text,
                         "sub_text":    f"First to {self.win_target} pts | {score_text}"})
        elif self.state == "WAITING_FOR_ROCK":
            base.update({"state_label": "Waiting",
                         "main_text":   "BOTH MAKE A FIST",
                         "sub_text":    f"{round_text} | {score_text}"})
        elif self.state == "COUNTDOWN":
            mt = "READY" if self.beat_count == 0 else str(min(self.beat_count, 3))
            base.update({"state_label": "Countdown",
                         "main_text":   mt,
                         "sub_text":    "AI locks on beat 3"})
        elif self.state == "SHOOT_WINDOW":
            tl = max(0.0, self.shoot_close_time - now) if self.shoot_close_time else 0.0
            base.update({"state_label": "Shoot Window",
                         "main_text":   "SHOOT!",
                         "sub_text":    "Everyone throws NOW",
                         "time_left":   tl})
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
        if now is None:
            now = time.monotonic()

        p1_conf = p1_tracker_state.get("confirmed_gesture", "Unknown")
        p2_conf = p2_tracker_state.get("confirmed_gesture", "Unknown")
        p1_stab = p1_tracker_state.get("stable_gesture",   "Unknown")
        p2_stab = p2_tracker_state.get("stable_gesture",   "Unknown")

        if self.state == "ROUND_INTRO":
            if now >= self.intro_until:
                self.state = "WAITING_FOR_ROCK"
            return self._build_output(now)

        if self.state == "ROUND_RESULT":
            if now >= self.result_until:
                if self._round_is_over():
                    self.match_result_banner = self._match_winner_text()
                    self.state       = "MATCH_RESULT"
                    self.match_until = now + self.match_result_seconds
                else:
                    self.round_number += 1
                    self._reset_round(now)
            return self._build_output(now)

        if self.state == "MATCH_RESULT":
            if now >= self.match_until:
                self.reset_match(now)
            return self._build_output(now)

        if self.state == "WAITING_FOR_ROCK":
            p1_rock = p1_conf == "Rock" or p1_stab == "Rock"
            p2_rock = p2_conf == "Rock" or p2_stab == "Rock"
            if p1_rock and p2_rock:
                self.last_rock_time = now
                self.state = "COUNTDOWN"
                self._init_beat()
            return self._build_output(now)

        if self.state == "COUNTDOWN":
            self._update_beat(p1_wrist_y, p2_wrist_y, p1_conf, p2_conf, now)
            if self.beat_count >= 3:
                self._lock_ai()
            if self.beat_count >= 4:
                self.state            = "SHOOT_WINDOW"
                self.shoot_open_time  = now
                self.shoot_close_time = now + self.SHOOT_WINDOW
                self.tracker_reset_requested = True
            return self._build_output(now)

        if self.state == "SHOOT_WINDOW":
            p1_thrown = p1_conf if p1_conf in VALID_GESTURES else                         (p1_stab if p1_stab in VALID_GESTURES else None)
            p2_thrown = p2_conf if p2_conf in VALID_GESTURES else                         (p2_stab if p2_stab in VALID_GESTURES else None)
            if now >= self.shoot_close_time or (p1_thrown and p2_thrown):
                self._resolve_round(p1_thrown or "Rock", p2_thrown or "Rock", now)
            return self._build_output(now)

        return self._build_output(now)

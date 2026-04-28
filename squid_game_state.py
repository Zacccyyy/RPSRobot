"""
squid_game_state.py
===================
Squid Game — Red Light, Green Light (gesture navigation variant).

A dot appears at a random position on screen.
The player steers their index finger tip toward the dot.
When the finger tip dwells inside the dot radius → DOT CAPTURED → score + new dot.

Meanwhile the system alternates between:
  GREEN LIGHT — player can move freely
  RED LIGHT   — player must freeze. Any substantial movement = GAME OVER.

Red/green intervals start slow and become increasingly sporadic over time.

Score = (dots_collected * 100) + int(seconds_survived)

The controller receives `hand_state` (the full dict from process_hand_frame)
so it can read both the normalised index-tip position AND velocity.

Finger tip = landmark index 8 (INDEX_FINGER_TIP).
"""

import time
import random
import math

# ── Tuning constants ──────────────────────────────────────────────────────────
INTRO_SECS           = 2.0
DOT_RADIUS_NORM      = 0.055     # dot radius as fraction of frame width
CAPTURE_DWELL_SECS   = 1.00      # seconds of dwell inside dot to capture (increased for fingerprint quality)
RESULT_FLASH_SECS    = 0.60      # brief flash after capture

MOVE_THRESHOLD_NORM  = 0.030     # normalised coord delta that counts as "moved"
                                  # during red light
FRAME_HISTORY        = 4         # frames to average for movement detection

# Green / Red light timing progression
# Each phase:  green_secs, red_secs
# After PHASE_COUNT cycles the interval shrinks toward MIN_INTERVAL
GREEN_START      = 5.0
RED_START        = 3.0
SHRINK_PER_DOT   = 0.25          # seconds removed per dot collected
MIN_GREEN        = 1.40
MIN_RED          = 0.90

GAME_OVER_SECS   = 4.0

INDEX_TIP = 8    # MediaPipe landmark index


def _landmark_pos(hand_state):
    """Return (norm_x, norm_y) of index tip, or None if no hand."""
    lm_obj = hand_state.get("_landmarks")
    if lm_obj is None:
        return None
    lm = lm_obj.landmark
    return (lm[INDEX_TIP].x, lm[INDEX_TIP].y)


class SquidGameController:
    """
    Squid Game controller.

    update() signature:
        controller.update(hand_state=..., now=...)

    hand_state is the dict returned by process_hand_frame / process_two_hands_frame
    for the main player.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.state             = "INTRO"
        self.dots_collected    = 0
        self.survived_secs     = 0.0
        self.score             = 0
        self._start_time       = 0.0
        self._game_over_time   = 0.0
        self._intro_until      = time.monotonic() + INTRO_SECS

        # Dot position (normalised 0-1)
        self._dot_x            = 0.5
        self._dot_y            = 0.5
        self._dwell_start      = None   # when finger entered the dot
        self._capture_flash    = 0.0    # until time for flash

        # Light state
        self._light            = "GREEN"    # "GREEN" | "RED"
        self._light_until      = 0.0
        self._eliminated       = False
        self.game_over_reason  = ""

        # Movement history for red-light detection
        self._pos_history: list = []   # list of (x, y)

        self._place_dot()

    def _place_dot(self):
        """Place dot at a random position, avoiding edges."""
        margin = 0.15
        self._dot_x   = random.uniform(margin, 1.0 - margin)
        self._dot_y   = random.uniform(margin + 0.10, 1.0 - margin)
        self._dwell_start = None

    def _green_duration(self):
        return max(MIN_GREEN, GREEN_START - self.dots_collected * SHRINK_PER_DOT)

    def _red_duration(self):
        # Add some randomness to make it unpredictable
        base = max(MIN_RED, RED_START - self.dots_collected * SHRINK_PER_DOT * 0.5)
        jitter = random.uniform(-0.30, 0.60)
        return max(MIN_RED, base + jitter)

    def _start_green(self, now):
        self._light       = "GREEN"
        self._light_until = now + self._green_duration()

    def _start_red(self, now):
        self._light       = "RED"
        self._light_until = now + self._red_duration()
        # Clear movement history at the moment red starts
        self._pos_history.clear()

    def _check_movement(self):
        """
        Return True if the player moved substantially during red light.
        Compares oldest and newest positions in the history buffer.
        """
        if len(self._pos_history) < FRAME_HISTORY:
            return False
        oldest = self._pos_history[0]
        newest = self._pos_history[-1]
        dx = abs(newest[0] - oldest[0])
        dy = abs(newest[1] - oldest[1])
        return (dx > MOVE_THRESHOLD_NORM or dy > MOVE_THRESHOLD_NORM)

    def _dist_to_dot(self, x, y):
        return math.sqrt((x - self._dot_x) ** 2 + (y - self._dot_y) ** 2)

    def _compute_score(self, now):
        survived = now - self._start_time if self._start_time > 0 else 0.0
        return int(self.dots_collected * 100 + survived)

    def _build_output(self, now):
        survived = max(0.0, now - self._start_time) if self._start_time > 0 else 0.0
        score    = self._compute_score(now)

        dwell_pct = 0.0
        if self._dwell_start is not None and self.state == "PLAYING":
            dwell_pct = min(1.0, (now - self._dwell_start) / CAPTURE_DWELL_SECS)

        return {
            "play_mode_label":  "Red Light Green Light",
            "state":            self.state,
            "light":            self._light,
            "dot_x":            self._dot_x,
            "dot_y":            self._dot_y,
            "dot_radius":       DOT_RADIUS_NORM,
            "dwell_pct":        dwell_pct,
            "dots_collected":   self.dots_collected,
            "survived_secs":    survived,
            "score":            score,
            "capture_flash":    now < self._capture_flash,
            "game_over_reason": self.game_over_reason,
            "eliminated":       self._eliminated,
            "two_player":       False,
        }

    def update(self, hand_state, now=None):
        if now is None:
            now = time.monotonic()

        # ── INTRO ──────────────────────────────────────────────────────────
        if self.state == "INTRO":
            if now >= self._intro_until:
                self.state       = "PLAYING"
                self._start_time = now
                self._start_green(now)
            return self._build_output(now)

        # ── GAME_OVER ──────────────────────────────────────────────────────
        if self.state == "GAME_OVER":
            if now >= self._game_over_time:
                self.reset()
            return self._build_output(now)

        # ── PLAYING ────────────────────────────────────────────────────────
        if self.state == "PLAYING":
            tip = _landmark_pos(hand_state)

            # ── Light state machine ─────────────────────────────────────
            if now >= self._light_until:
                if self._light == "GREEN":
                    self._start_red(now)
                else:
                    self._start_green(now)

            # ── Update position history ─────────────────────────────────
            if tip is not None:
                self._pos_history.append(tip)
                if len(self._pos_history) > FRAME_HISTORY:
                    self._pos_history.pop(0)

            # ── RED LIGHT movement check ────────────────────────────────
            if self._light == "RED" and tip is not None:
                if self._check_movement():
                    self._eliminated      = True
                    self.game_over_reason = "YOU MOVED!"
                    self.state            = "GAME_OVER"
                    self._game_over_time  = now + GAME_OVER_SECS
                    return self._build_output(now)

            # ── DOT capture (green light only) ─────────────────────────
            if self._light == "GREEN" and tip is not None:
                dist = self._dist_to_dot(tip[0], tip[1])
                if dist <= DOT_RADIUS_NORM:
                    # Inside dot
                    if self._dwell_start is None:
                        self._dwell_start = now
                    elif (now - self._dwell_start) >= CAPTURE_DWELL_SECS:
                        # Captured!
                        self.dots_collected  += 1
                        self._capture_flash   = now + RESULT_FLASH_SECS
                        self._place_dot()
                else:
                    self._dwell_start = None
            elif self._light == "RED":
                # Can't capture during red light
                self._dwell_start = None

        return self._build_output(now)


# ─────────────────────────────────────────────────────────────────────────────
# Two-Player Red Light Green Light
# ─────────────────────────────────────────────────────────────────────────────

WIN_DOTS_2P = 5   # first to collect this many wins

class SquidGame2PController:
    """
    Two-player Red Light Green Light.

    Both players play simultaneously with the same shared light.
    P1 (cyan) chases their own dot; P2 (magenta) chases their own dot.
    Moving on RED = that player is eliminated.
    First to WIN_DOTS_2P dots wins.

    update() signature:
        controller.update(p1_hand=..., p2_hand=..., now=...)

    p1_hand / p2_hand are hand_state dicts from process_two_hands_frame.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.state           = "INTRO"
        self._intro_until    = time.monotonic() + INTRO_SECS
        self._start_time     = 0.0
        self._game_over_time = 0.0

        # Shared light state
        self._light       = "GREEN"
        self._light_until = 0.0

        # Per-player state
        self._p = [
            self._make_player(),   # P1 index 0
            self._make_player(),   # P2 index 1
        ]
        self._place_dot(0)
        self._place_dot(1)

        self.winner        = 0   # 1 or 2
        self.loser         = 0   # 1 or 2 (if eliminated)
        self.game_over_reason = ""

    def _make_player(self):
        return {
            "dots":       0,
            "eliminated": False,
            "dwell_start": None,
            "capture_flash": 0.0,
            "dot_x":  0.5,
            "dot_y":  0.5,
            "pos_history": [],
        }

    def _place_dot(self, idx):
        margin = 0.12
        p = self._p[idx]
        # P1 (idx 0) → left half, P2 (idx 1) → right half
        if idx == 0:
            p["dot_x"] = random.uniform(margin, 0.48)
        else:
            p["dot_x"] = random.uniform(0.52, 1.0 - margin)
        p["dot_y"]      = random.uniform(margin + 0.10, 1.0 - margin)
        p["dwell_start"] = None

    def _green_duration(self):
        total = self._p[0]["dots"] + self._p[1]["dots"]
        return max(MIN_GREEN, GREEN_START - total * SHRINK_PER_DOT * 0.5)

    def _red_duration(self):
        total = self._p[0]["dots"] + self._p[1]["dots"]
        base  = max(MIN_RED, RED_START - total * SHRINK_PER_DOT * 0.25)
        return max(MIN_RED, base + random.uniform(-0.30, 0.60))

    def _start_green(self, now):
        self._light       = "GREEN"
        self._light_until = now + self._green_duration()

    def _start_red(self, now):
        self._light = "RED"
        self._light_until = now + self._red_duration()
        for p in self._p:
            p["pos_history"].clear()

    def _check_movement(self, idx):
        hist = self._p[idx]["pos_history"]
        if len(hist) < FRAME_HISTORY:
            return False
        dx = abs(hist[-1][0] - hist[0][0])
        dy = abs(hist[-1][1] - hist[0][1])
        return dx > MOVE_THRESHOLD_NORM or dy > MOVE_THRESHOLD_NORM

    def _update_player(self, idx, hand_state, now):
        p = self._p[idx]
        if p["eliminated"]:
            return

        tip = _landmark_pos(hand_state)

        if tip is not None:
            p["pos_history"].append(tip)
            if len(p["pos_history"]) > FRAME_HISTORY:
                p["pos_history"].pop(0)

        if self._light == "RED" and tip is not None:
            if self._check_movement(idx):
                p["eliminated"] = True
                return

        if self._light == "GREEN" and tip is not None:
            dx = tip[0] - p["dot_x"]
            dy = tip[1] - p["dot_y"]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist <= DOT_RADIUS_NORM:
                if p["dwell_start"] is None:
                    p["dwell_start"] = now
                elif (now - p["dwell_start"]) >= CAPTURE_DWELL_SECS:
                    p["dots"] += 1
                    p["capture_flash"] = now + RESULT_FLASH_SECS
                    self._place_dot(idx)
            else:
                p["dwell_start"] = None
        elif self._light == "RED":
            p["dwell_start"] = None

    def _build_output(self, now):
        survived = max(0.0, now - self._start_time) if self._start_time > 0 else 0.0
        p1, p2   = self._p[0], self._p[1]

        def dwell_pct(p):
            if p["dwell_start"] is None or self.state != "PLAYING":
                return 0.0
            return min(1.0, (now - p["dwell_start"]) / CAPTURE_DWELL_SECS)

        return {
            "play_mode_label":    "Red Light Green Light 2P",
            "state":              self.state,
            "light":              self._light,
            "light_time_left":    max(0.0, self._light_until - now),
            "survived_secs":      survived,
            "winner":             self.winner,
            "loser":              self.loser,
            "game_over_reason":   self.game_over_reason,
            # P1
            "p1_dot_x":      p1["dot_x"],
            "p1_dot_y":      p1["dot_y"],
            "p1_dots":       p1["dots"],
            "p1_eliminated": p1["eliminated"],
            "p1_dwell_pct":  dwell_pct(p1),
            "p1_flash":      now < p1["capture_flash"],
            # P2
            "p2_dot_x":      p2["dot_x"],
            "p2_dot_y":      p2["dot_y"],
            "p2_dots":       p2["dots"],
            "p2_eliminated": p2["eliminated"],
            "p2_dwell_pct":  dwell_pct(p2),
            "p2_flash":      now < p2["capture_flash"],
            "win_dots":      WIN_DOTS_2P,
            "two_player":    True,
        }

    def update(self, p1_hand, p2_hand, now=None):
        if now is None:
            now = time.monotonic()

        if self.state == "INTRO":
            if now >= self._intro_until:
                self.state       = "PLAYING"
                self._start_time = now
                self._start_green(now)
            return self._build_output(now)

        if self.state == "GAME_OVER":
            if now >= self._game_over_time:
                self.reset()
            return self._build_output(now)

        if self.state == "PLAYING":
            # Update shared light
            if now >= self._light_until:
                if self._light == "GREEN":
                    self._start_red(now)
                else:
                    self._start_green(now)

            # Update each player
            self._update_player(0, p1_hand, now)
            self._update_player(1, p2_hand, now)

            p1, p2 = self._p[0], self._p[1]

            # Check win conditions
            if p1["dots"] >= WIN_DOTS_2P and not p1["eliminated"]:
                self.winner = 1; self.loser = 2
                self.game_over_reason = "P1 collected all dots!"
                self.state = "GAME_OVER"
                self._game_over_time = now + GAME_OVER_SECS
            elif p2["dots"] >= WIN_DOTS_2P and not p2["eliminated"]:
                self.winner = 2; self.loser = 1
                self.game_over_reason = "P2 collected all dots!"
                self.state = "GAME_OVER"
                self._game_over_time = now + GAME_OVER_SECS
            elif p1["eliminated"] and p2["eliminated"]:
                self.winner = 0; self.loser = 0
                self.game_over_reason = "Both eliminated!"
                self.state = "GAME_OVER"
                self._game_over_time = now + GAME_OVER_SECS
            elif p1["eliminated"]:
                self.winner = 2; self.loser = 1
                self.game_over_reason = "P1 moved on RED!"
                self.state = "GAME_OVER"
                self._game_over_time = now + GAME_OVER_SECS
            elif p2["eliminated"]:
                self.winner = 1; self.loser = 2
                self.game_over_reason = "P2 moved on RED!"
                self.state = "GAME_OVER"
                self._game_over_time = now + GAME_OVER_SECS

        return self._build_output(now)

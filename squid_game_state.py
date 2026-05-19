"""
squid_game_state.py
===================
Squid Game -- Red Light, Green Light (gesture navigation variant).

A dot appears at a random position on screen.
The player steers their index finger tip toward the dot.
When the finger tip dwells inside the dot radius -> DOT CAPTURED -> score + new dot.

Meanwhile the system alternates between:
  GREEN LIGHT -- player can move freely
  RED LIGHT   -- player must freeze. Any substantial movement = GAME OVER.

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
DOT_RADIUS_NORM      = 0.055     # dot radius as a fraction of frame width (normalised)
CAPTURE_DWELL_SECS   = 1.00      # how long finger must stay inside dot to capture it
RESULT_FLASH_SECS    = 0.60      # how long the capture-flash effect shows

# How much the finger tip must move (in normalised coords) to count as "moved"
# during red light — this threshold is calibrated, do not change
MOVE_THRESHOLD_NORM  = 0.030
FRAME_HISTORY        = 4         # frames to average for movement detection

# Green / Red light timing progression.
# Each phase:  green_secs, red_secs
# Intervals shrink as the player collects more dots, making the game harder.
GREEN_START      = 5.0
RED_START        = 3.0
SHRINK_PER_DOT   = 0.25          # seconds removed per dot collected
MIN_GREEN        = 1.40          # green light will never be shorter than this
MIN_RED          = 0.90          # red light will never be shorter than this

GAME_OVER_SECS   = 4.0           # how long the GAME_OVER screen lingers before auto-reset

INDEX_TIP = 8    # MediaPipe landmark index for the index finger tip


def _landmark_pos(hand_state):
    """
    Extract the normalised (x, y) position of the index finger tip from a
    hand_state dict.  Returns None if no hand is detected this frame.

    The hand_state dict comes from process_hand_frame and stores the raw
    MediaPipe landmark object under the '_landmarks' key.
    """
    lm_obj = hand_state.get("_landmarks")
    if lm_obj is None:
        return None
    lm = lm_obj.landmark
    return (lm[INDEX_TIP].x, lm[INDEX_TIP].y)


class SquidGameController:
    """
    Single-player Squid Game controller (Red Light, Green Light).

    update() signature:
        controller.update(hand_state=..., now=...)

    hand_state is the dict returned by process_hand_frame / process_two_hands_frame
    for the main player.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all game state back to the initial INTRO screen."""
        self.state             = "INTRO"
        self.dots_collected    = 0
        self.survived_secs     = 0.0
        self.score             = 0
        self._start_time       = 0.0      # set when PLAYING begins
        self._game_over_time   = 0.0      # when the GAME_OVER screen will auto-reset
        self._intro_until      = time.monotonic() + INTRO_SECS

        # Dot tracking
        self._dot_x            = 0.5
        self._dot_y            = 0.5
        self._dwell_start      = None     # timestamp when finger first entered the dot
        self._capture_flash    = 0.0      # timestamp until which capture-flash shows

        # Light state
        self._light            = "GREEN"   # "GREEN" or "RED"
        self._light_until      = 0.0       # when the current light phase ends
        self._eliminated       = False
        self.game_over_reason  = ""

        # Ring buffer of recent finger tip positions, used to detect movement on RED
        self._pos_history: list = []

        self._place_dot()

    def _place_dot(self):
        """Place a new dot at a random position, avoiding the screen edges."""
        margin = 0.15
        self._dot_x   = random.uniform(margin, 1.0 - margin)
        # Extra top margin (+0.10) so the dot doesn't get hidden behind UI headers
        self._dot_y   = random.uniform(margin + 0.10, 1.0 - margin)
        self._dwell_start = None   # reset dwell so the player has to re-enter the new dot

    def _green_duration(self):
        """
        Calculate how long the next GREEN phase lasts.
        Gets shorter as dots are collected, never below MIN_GREEN.
        """
        return max(MIN_GREEN, GREEN_START - self.dots_collected * SHRINK_PER_DOT)

    def _red_duration(self):
        """
        Calculate how long the next RED phase lasts.
        Gets shorter as dots are collected AND has random jitter to keep players on edge.
        """
        base   = max(MIN_RED, RED_START - self.dots_collected * SHRINK_PER_DOT * 0.5)
        jitter = random.uniform(-0.30, 0.60)
        return max(MIN_RED, base + jitter)

    def _start_green(self, now):
        """Switch to GREEN light and schedule when it ends."""
        self._light       = "GREEN"
        self._light_until = now + self._green_duration()

    def _start_red(self, now):
        """
        Switch to RED light, schedule when it ends, and clear the movement
        history buffer so we don't carry old motion into the new RED phase.
        """
        self._light       = "RED"
        self._light_until = now + self._red_duration()
        # Clear history so movement from the last GREEN phase can't trigger an
        # immediate elimination at the start of RED
        self._pos_history.clear()

    def _check_movement(self):
        """
        Return True if the player moved substantially during red light.
        Compares oldest and newest positions in the history buffer —
        if either the X or Y delta exceeds MOVE_THRESHOLD_NORM, they moved.
        """
        # Need enough history to make a reliable comparison
        if len(self._pos_history) < FRAME_HISTORY:
            return False
        oldest = self._pos_history[0]
        newest = self._pos_history[-1]
        dx = abs(newest[0] - oldest[0])
        dy = abs(newest[1] - oldest[1])
        return (dx > MOVE_THRESHOLD_NORM or dy > MOVE_THRESHOLD_NORM)

    def _dist_to_dot(self, x, y):
        """Euclidean distance from point (x, y) to the current dot centre."""
        return math.sqrt((x - self._dot_x) ** 2 + (y - self._dot_y) ** 2)

    def _compute_score(self, now):
        """Score = (dots * 100) + whole seconds survived."""
        survived = now - self._start_time if self._start_time > 0 else 0.0
        return int(self.dots_collected * 100 + survived)

    def _build_output(self, now):
        """
        Build the output dict that the UI renderer reads every frame.
        Called from every branch of update() so the renderer always has current data.
        """
        survived = max(0.0, now - self._start_time) if self._start_time > 0 else 0.0
        score    = self._compute_score(now)

        # Dwell progress (0.0-1.0) shown as a fill ring around the dot
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
            "capture_flash":    now < self._capture_flash,  # True while flash is showing
            "game_over_reason": self.game_over_reason,
            "eliminated":       self._eliminated,
            "two_player":       False,
        }

    def update(self, hand_state, now=None):
        """
        Main frame update.  Called every frame by the game loop.

        Drives the INTRO -> PLAYING -> GAME_OVER state machine.
        All gesture detection and light-state logic lives here.
        """
        if now is None:
            now = time.monotonic()

        # ── INTRO: show countdown before game starts ──────────────────────
        if self.state == "INTRO":
            if now >= self._intro_until:
                self.state       = "PLAYING"
                self._start_time = now
                self._start_green(now)   # begin with GREEN light
            return self._build_output(now)

        # ── GAME_OVER: linger on screen, then auto-reset ──────────────────
        if self.state == "GAME_OVER":
            if now >= self._game_over_time:
                self.reset()
            return self._build_output(now)

        # ── PLAYING: all gameplay logic ───────────────────────────────────
        if self.state == "PLAYING":
            tip = _landmark_pos(hand_state)

            # -- Light state machine: toggle GREEN <-> RED when time expires --
            if now >= self._light_until:
                if self._light == "GREEN":
                    self._start_red(now)
                else:
                    self._start_green(now)

            # -- Maintain a rolling history of finger positions --
            if tip is not None:
                self._pos_history.append(tip)
                # Cap the buffer at FRAME_HISTORY frames; drop oldest
                if len(self._pos_history) > FRAME_HISTORY:
                    self._pos_history.pop(0)

            # -- RED LIGHT: eliminate player if they moved --
            if self._light == "RED" and tip is not None:
                if self._check_movement():
                    self._eliminated      = True
                    self.game_over_reason = "YOU MOVED!"
                    self.state            = "GAME_OVER"
                    self._game_over_time  = now + GAME_OVER_SECS
                    return self._build_output(now)

            # -- GREEN LIGHT: allow dot captures --
            if self._light == "GREEN" and tip is not None:
                dist = self._dist_to_dot(tip[0], tip[1])
                if dist <= DOT_RADIUS_NORM:
                    # Finger is inside the dot — start or continue dwell timer
                    if self._dwell_start is None:
                        self._dwell_start = now
                    elif (now - self._dwell_start) >= CAPTURE_DWELL_SECS:
                        # Dwell complete — capture the dot!
                        self.dots_collected  += 1
                        self._capture_flash   = now + RESULT_FLASH_SECS
                        self._place_dot()   # spawn a new dot
                else:
                    # Finger drifted outside the dot — reset dwell timer
                    self._dwell_start = None
            elif self._light == "RED":
                # Can't capture during red light — reset dwell so you have to
                # re-enter the dot from scratch when green comes back
                self._dwell_start = None

        return self._build_output(now)


# ─────────────────────────────────────────────────────────────────────────────
# Two-Player Red Light Green Light
# ─────────────────────────────────────────────────────────────────────────────

# First player to collect this many dots wins
WIN_DOTS_2P = 5

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
        """Reset all game state to the initial INTRO screen."""
        self.state           = "INTRO"
        self._intro_until    = time.monotonic() + INTRO_SECS
        self._start_time     = 0.0
        self._game_over_time = 0.0

        # Shared light — both players obey the same traffic light
        self._light       = "GREEN"
        self._light_until = 0.0

        # Per-player state stored in a 2-element list (index 0 = P1, index 1 = P2)
        self._p = [
            self._make_player(),   # P1
            self._make_player(),   # P2
        ]
        self._place_dot(0)
        self._place_dot(1)

        self.winner        = 0   # set to 1 or 2 on GAME_OVER
        self.loser         = 0   # set to 1 or 2 on elimination
        self.game_over_reason = ""

    def _make_player(self):
        """Create the per-player state dict with defaults."""
        return {
            "dots":         0,
            "eliminated":   False,
            "dwell_start":  None,       # timestamp when finger entered the dot
            "capture_flash": 0.0,       # until time for capture flash
            "dot_x":        0.5,
            "dot_y":        0.5,
            "pos_history":  [],         # rolling history for red-light movement detection
        }

    def _place_dot(self, idx):
        """
        Place a new dot for player idx at a random position.
        P1 (idx 0) stays in the left half; P2 (idx 1) in the right half
        so they don't chase each other's targets.
        """
        margin = 0.12
        p = self._p[idx]
        if idx == 0:
            p["dot_x"] = random.uniform(margin, 0.48)          # left half
        else:
            p["dot_x"] = random.uniform(0.52, 1.0 - margin)    # right half
        p["dot_y"]      = random.uniform(margin + 0.10, 1.0 - margin)
        p["dwell_start"] = None   # must re-enter dot to start dwell

    def _green_duration(self):
        """
        Green duration scales down with total dots collected by both players.
        Uses a gentler shrink rate than single-player (0.5x) since two players
        collect dots faster.
        """
        total = self._p[0]["dots"] + self._p[1]["dots"]
        return max(MIN_GREEN, GREEN_START - total * SHRINK_PER_DOT * 0.5)

    def _red_duration(self):
        """Red duration with randomness; uses 0.25x shrink so red doesn't end too fast."""
        total = self._p[0]["dots"] + self._p[1]["dots"]
        base  = max(MIN_RED, RED_START - total * SHRINK_PER_DOT * 0.25)
        return max(MIN_RED, base + random.uniform(-0.30, 0.60))

    def _start_green(self, now):
        """Switch to GREEN light and schedule when it ends."""
        self._light       = "GREEN"
        self._light_until = now + self._green_duration()

    def _start_red(self, now):
        """Switch to RED light and clear both players' movement histories."""
        self._light = "RED"
        self._light_until = now + self._red_duration()
        # Clear history for both players so lingering movement from GREEN can't
        # immediately eliminate them at the start of RED
        for p in self._p:
            p["pos_history"].clear()

    def _check_movement(self, idx):
        """
        Return True if player idx moved too much during red light.
        Same logic as single-player version, but reads from per-player history.
        """
        hist = self._p[idx]["pos_history"]
        if len(hist) < FRAME_HISTORY:
            return False
        dx = abs(hist[-1][0] - hist[0][0])
        dy = abs(hist[-1][1] - hist[0][1])
        return dx > MOVE_THRESHOLD_NORM or dy > MOVE_THRESHOLD_NORM

    def _update_player(self, idx, hand_state, now):
        """
        Update one player's state for this frame.

        Handles:
          - Rolling position history (for red-light movement detection)
          - Elimination check during RED
          - Dot capture during GREEN (with dwell timer)
        """
        p = self._p[idx]

        # Skip any processing for eliminated players
        if p["eliminated"]:
            return

        tip = _landmark_pos(hand_state)

        # Update rolling position history
        if tip is not None:
            p["pos_history"].append(tip)
            if len(p["pos_history"]) > FRAME_HISTORY:
                p["pos_history"].pop(0)

        # RED light: check if this player moved
        if self._light == "RED" and tip is not None:
            if self._check_movement(idx):
                p["eliminated"] = True
                return   # no further processing needed for an eliminated player

        # GREEN light: check dot capture (dwell mechanic)
        if self._light == "GREEN" and tip is not None:
            dx   = tip[0] - p["dot_x"]
            dy   = tip[1] - p["dot_y"]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist <= DOT_RADIUS_NORM:
                # Inside the dot — start or continue dwell timer
                if p["dwell_start"] is None:
                    p["dwell_start"] = now
                elif (now - p["dwell_start"]) >= CAPTURE_DWELL_SECS:
                    # Dwell complete — capture!
                    p["dots"]          += 1
                    p["capture_flash"]  = now + RESULT_FLASH_SECS
                    self._place_dot(idx)
            else:
                # Finger left the dot area — reset dwell
                p["dwell_start"] = None
        elif self._light == "RED":
            # Reset dwell during red so the player must re-enter once green returns
            p["dwell_start"] = None

    def _build_output(self, now):
        """Build the output dict that the UI renderer reads every frame."""
        survived = max(0.0, now - self._start_time) if self._start_time > 0 else 0.0
        p1, p2   = self._p[0], self._p[1]

        def dwell_pct(p):
            """Dwell progress (0.0-1.0) for one player, shown as a ring fill."""
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
            # P1 state
            "p1_dot_x":      p1["dot_x"],
            "p1_dot_y":      p1["dot_y"],
            "p1_dots":       p1["dots"],
            "p1_eliminated": p1["eliminated"],
            "p1_dwell_pct":  dwell_pct(p1),
            "p1_flash":      now < p1["capture_flash"],
            # P2 state
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
        """
        Main frame update for the 2-player version.

        Drives INTRO -> PLAYING -> GAME_OVER.
        Both players share a single light; win/elimination logic checks both.
        """
        if now is None:
            now = time.monotonic()

        # ── INTRO ──
        if self.state == "INTRO":
            if now >= self._intro_until:
                self.state       = "PLAYING"
                self._start_time = now
                self._start_green(now)
            return self._build_output(now)

        # ── GAME_OVER: linger, then auto-reset ──
        if self.state == "GAME_OVER":
            if now >= self._game_over_time:
                self.reset()
            return self._build_output(now)

        # ── PLAYING ──
        if self.state == "PLAYING":
            # Advance the shared light state machine
            if now >= self._light_until:
                if self._light == "GREEN":
                    self._start_red(now)
                else:
                    self._start_green(now)

            # Update each player independently
            self._update_player(0, p1_hand, now)
            self._update_player(1, p2_hand, now)

            p1, p2 = self._p[0], self._p[1]

            # Check win/elimination conditions.
            # Order matters: dot-win beats elimination so a simultaneous
            # "P1 wins AND P2 eliminated" correctly counts as P1 winning.
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
                # Both moved on red — draw/no winner
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

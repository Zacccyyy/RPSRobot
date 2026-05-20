"""
gesture_nav.py  --  Gesture-based menu navigation with +/- zone support.

How the interaction works:
  - Moving the hand up/down highlights different menu items instantly
    (raw Y position, no smoothing lag on the selection itself).
  - Holding the hand in the CENTER zone (X 0.40-0.60) for 2 seconds selects.
  - Holding the hand in the LEFT zone  (X < 0.40)   for 1 second fires adjust -1,
    then repeats every 0.5s while the hand stays there.
  - Holding the hand in the RIGHT zone (X > 0.60)   for 1 second fires adjust +1,
    with the same repeat behaviour.

The +/- adjust zones only activate after the hand has been on the same item
for ITEM_STABLE seconds, so scrolling quickly between rows doesn't
accidentally fire adjustments.

Events emitted:
  {"type": "hover",  "item_index": int}
  {"type": "select"}
  {"type": "adjust", "direction": -1 | +1}
  {"type": "swipe_left"}   -- wave hand left quickly to go back / ESC
"""

import time

# How long (seconds) the hand must dwell in a zone before it triggers
DWELL_SELECT  = 2.0    # center zone → select
DWELL_ADJUST  = 1.0    # left/right zone → first adjust fire
ADJUST_REPEAT = 0.5    # how often adjust fires while hand stays in zone

# How long the hand must be on the same item before +/- zones activate
ITEM_STABLE = 0.35

# Exponential smoothing factor for cursor display position (0=no update, 1=instant)
SMOOTHING = 0.60

# Number of consecutive frames with a hand present before the controller activates
CONFIRM_FRAMES = 5

# Swipe-left gesture detection tuning
SWIPE_LEFT_THRESHOLD = 0.25   # hand must move this far left (normalised X) ...
SWIPE_TIME_WINDOW    = 0.6    # ... within this many seconds
SWIPE_COOLDOWN       = 1.5    # don't fire a swipe again this soon after the last one

# Button hit zones for the +/- buttons -- must match draw_settings_screen in ui_renderer.py.
# These numbers come from: x2=0.935, btn_w=0.055, btn_gap=0.010, x2_offset=0.018
_BTN_W    = 0.055
_BTN_GAP  = 0.010
_PLUS_X2  = 0.935 - 0.018          # right edge of plus button  ≈ 0.917
_PLUS_X1  = _PLUS_X2 - _BTN_W      # left edge of plus button   ≈ 0.862
_MINUS_X2 = _PLUS_X1 - _BTN_GAP    # right edge of minus button ≈ 0.852
_MINUS_X1 = _MINUS_X2 - _BTN_W     # left edge of minus button  ≈ 0.797
_ZONE_SEP = (_MINUS_X2 + _PLUS_X1) / 2  # midpoint between the two buttons ≈ 0.857

# Expand each zone slightly inward toward the panel centre so they're easier to hit
ZONE_MINUS = (_MINUS_X1 - 0.020, _ZONE_SEP)   # ≈ (0.777, 0.857)
ZONE_PLUS  = (_ZONE_SEP, _PLUS_X2 + 0.020)    # ≈ (0.857, 0.937)

# MediaPipe landmark index for the tip of the index finger (used as the cursor)
_INDEX_TIP = 8


class GestureNavController:
    """
    Translates live hand position into menu navigation events.

    The controller goes through three states:
      inactive   -- no hand present, nothing happening
      warming_up -- hand just appeared, counting up to CONFIRM_FRAMES
      active     -- hand confirmed, emitting hover/select/adjust events
    """

    def __init__(self):
        # Overall active state
        self._active        = False
        self._warming_up    = False
        self._warmup_frames = 0

        # Which menu item the hand is currently over, and when we entered it
        self._last_item_idx = -1
        self._item_since    = 0.0

        # Timers for the select dwell (center zone)
        self._select_start  = None

        # Which X zone the hand is in, and the adjust-fire timers
        self._x_zone        = "center"   # "minus" | "center" | "plus"
        self._adjust_start  = None
        self._adjust_last   = 0.0

        # Smoothed X/Y for the cursor display (not used for logic)
        self._smooth_x      = None
        self._smooth_y      = None

        # Layout info set each update() call
        self._item_count    = 1
        self._content_top   = 0.44
        self._content_bot   = 0.83

        # Set of item indices that have +/- buttons (suppresses center-dwell on those rows)
        self._adjust_items  = set()

        # Swipe-left tracking
        self._swipe_x_start   = None
        self._swipe_t_start   = None
        self._swipe_last_fire = 0.0

    def update(self, hand_state, now=None, item_count=1,
               content_top=0.44, content_bottom=0.83,
               adjust_items=None):
        """
        Call once per frame with the latest hand state dict.

        hand_state      -- dict that must contain "_landmarks" (MediaPipe result or None)
        now             -- current time (time.monotonic()); if None we fetch it here
        item_count      -- how many rows are in the menu
        content_top     -- normalised Y of the topmost menu item
        content_bottom  -- normalised Y of the bottommost menu item
        adjust_items    -- set of row indices that have +/- buttons;
                           those rows suppress the center-dwell select behaviour.
                           Pass None or empty set to disable adjust zones.

        Returns a list of event dicts (may be empty).
        """
        if now is None:
            now = time.monotonic()

        # Store layout info so get_cursor_info() can use it too
        self._item_count   = max(item_count, 1)
        self._content_top  = content_top
        self._content_bot  = content_bottom
        self._adjust_items = adjust_items or set()

        # If there's more than one item, expand the top/bottom by half a gap so
        # the topmost and bottommost items are reachable without extreme Y positions
        if self._item_count > 1:
            gap = (content_bottom - content_top) / (self._item_count - 1)
            self._content_top = content_top  - gap * 0.5
            self._content_bot = content_bottom + gap * 0.5

        events = []

        # Pull the landmark object out of the hand state
        lm_obj = hand_state.get("_landmarks")
        lm     = lm_obj.landmark if lm_obj is not None else None

        hand_present = lm is not None
        raw_x = lm[_INDEX_TIP].x if hand_present else None
        raw_y = lm[_INDEX_TIP].y if hand_present else None

        # Smooth the cursor position for display purposes only --
        # we use the raw position for all zone/row logic to avoid lag
        if raw_x is not None:
            if self._smooth_x is None:
                # First frame: jump straight to the real position
                self._smooth_x, self._smooth_y = raw_x, raw_y
            else:
                # Exponential smoothing: blend toward the new position
                self._smooth_x = SMOOTHING * raw_x + (1 - SMOOTHING) * self._smooth_x
                self._smooth_y = SMOOTHING * raw_y + (1 - SMOOTHING) * self._smooth_y

        # ── Swipe-left detection (wave hand quickly left = go back) ────────
        if hand_present and raw_x is not None:
            if self._swipe_x_start is None:
                # Start of a new potential swipe — record starting position and time
                self._swipe_x_start = raw_x
                self._swipe_t_start = now
            else:
                elapsed  = now - self._swipe_t_start
                distance = self._swipe_x_start - raw_x   # positive = moved left

                if elapsed <= SWIPE_TIME_WINDOW:
                    # Still within the time window — check if we've moved far enough
                    if (distance >= SWIPE_LEFT_THRESHOLD
                            and now - self._swipe_last_fire >= SWIPE_COOLDOWN):
                        events.append({"type": "swipe_left"})
                        self._swipe_last_fire = now
                        # Reset so the next swipe starts fresh
                        self._swipe_x_start = None
                        self._swipe_t_start = None
                else:
                    # Time window expired without a swipe -- start tracking again
                    self._swipe_x_start = raw_x
                    self._swipe_t_start = now
        else:
            # No hand -- reset swipe tracking
            self._swipe_x_start = None
            self._swipe_t_start = None

        # ── Main state machine ─────────────────────────────────────────────
        if not self._active:
            # Not yet active: wait for the hand to appear for CONFIRM_FRAMES frames
            if hand_present:
                self._warming_up    = True
                self._warmup_frames += 1
                if self._warmup_frames >= CONFIRM_FRAMES:
                    self._activate(now)
            else:
                # Hand disappeared before we confirmed -- reset the warmup count
                self._reset_warmup()
        else:
            # Active: process hand position into events
            if hand_present and raw_y is not None:
                # Map the Y position to a row index
                span     = self._content_bot - self._content_top
                fraction = (raw_y - self._content_top) / span if span > 0 else 0.5
                fraction = max(0.0, min(1.0, fraction))  # clamp to [0, 1]
                item_idx = min(int(fraction * self._item_count), self._item_count - 1)

                events.append({"type": "hover", "item_index": item_idx})

                # If the user moved to a new row, reset all zone timers
                if item_idx != self._last_item_idx:
                    self._last_item_idx = item_idx
                    self._item_since    = now
                    self._select_start  = now
                    self._adjust_start  = None
                    self._adjust_last   = 0.0
                    self._x_zone        = "center"

                # Determine which X zone the hand is in
                item_is_adj = item_idx in self._adjust_items   # does this row have +/- buttons?
                item_age    = now - self._item_since
                zones_on    = item_is_adj and item_age >= ITEM_STABLE   # only activate after settling
                sx          = self._smooth_x
                new_zone    = "center"

                if zones_on and sx is not None:
                    # Check if the smoothed X falls inside the minus or plus button zone
                    if ZONE_MINUS[0] <= sx <= ZONE_MINUS[1]:
                        new_zone = "minus"
                    elif ZONE_PLUS[0] <= sx <= ZONE_PLUS[1]:
                        new_zone = "plus"

                # If the zone changed, reset the relevant dwell timers
                if new_zone != self._x_zone:
                    self._x_zone       = new_zone
                    self._adjust_start = now if new_zone != "center" else None
                    self._select_start = now if new_zone == "center" else None

                # ── Fire events based on which zone the hand is in ─────────
                if self._x_zone == "center":
                    # Center zone -- fire select after the dwell time.
                    # But skip dwell-to-select for rows that have +/- buttons
                    # (those rows are controlled by the side zones instead).
                    if not item_is_adj:
                        if self._select_start and (now - self._select_start) >= DWELL_SELECT:
                            events.append({"type": "select"})
                            self._deactivate()

                elif self._adjust_start is not None:
                    # Left or right zone -- fire adjust after DWELL_ADJUST,
                    # then repeat every ADJUST_REPEAT while the hand stays there
                    direction = -1 if self._x_zone == "minus" else +1
                    elapsed   = now - self._adjust_start

                    if elapsed >= DWELL_ADJUST:
                        since_last = now - self._adjust_last
                        # First fire: immediately once the dwell threshold is crossed.
                        # Subsequent fires: every ADJUST_REPEAT seconds.
                        if self._adjust_last == 0.0 or since_last >= ADJUST_REPEAT:
                            events.append({"type": "adjust", "direction": direction})
                            self._adjust_last = now

            else:
                # Hand disappeared while active -- deactivate
                self._deactivate()

        return events

    def reset(self):
        """Fully reset all state (call when switching screens)."""
        self._active          = False
        self._warming_up      = False
        self._warmup_frames   = 0
        self._last_item_idx   = -1
        self._item_since      = 0.0
        self._select_start    = None
        self._x_zone          = "center"
        self._adjust_start    = None
        self._adjust_last     = 0.0
        self._smooth_x        = None
        self._smooth_y        = None
        self._adjust_items    = set()
        self._swipe_x_start   = None
        self._swipe_t_start   = None
        self._swipe_last_fire = 0.0

    def is_active(self):
        """Return True if a hand has been confirmed and navigation is live."""
        return self._active

    def is_warming_up(self):
        """Return True if a hand was just detected but hasn't been confirmed yet."""
        return self._warming_up and not self._active

    def get_cursor_info(self):
        """
        Return a dict of display info for drawing the navigation cursor overlay.
        The UI uses this to draw the cursor dot, progress arcs, etc.
        """
        now = time.monotonic()

        # Calculate progress percentages for the dwell arcs (0.0 to 1.0)
        select_pct = 0.0
        adjust_pct = 0.0
        if self._active:
            current_is_adj = self._last_item_idx in self._adjust_items
            if self._x_zone == "center" and self._select_start and not current_is_adj:
                select_pct = min((now - self._select_start) / DWELL_SELECT, 1.0)
            elif self._x_zone != "center" and self._adjust_start:
                adjust_pct = min((now - self._adjust_start) / DWELL_ADJUST, 1.0)

        return {
            "active":      self._active,
            "warming_up":  self.is_warming_up(),
            "warmup_pct":  min(self._warmup_frames / CONFIRM_FRAMES, 1.0),
            "index_tip_x": self._smooth_x,
            "index_tip_y": self._smooth_y,
            "item_index":  self._last_item_idx,
            "item_count":  self._item_count,
            "dwell_pct":   select_pct,   # how far through the select dwell
            "adjust_pct":  adjust_pct,   # how far through the adjust dwell
            "x_zone":      self._x_zone,
            "has_adjust":  bool(self._adjust_items),
        }

    def _activate(self, now):
        """Transition from warming_up to fully active."""
        self._active        = True
        self._warming_up    = False
        self._warmup_frames = 0
        self._select_start  = now
        self._last_item_idx = -1

    def _deactivate(self):
        """Hand disappeared or a select fired -- go back to inactive."""
        self._active        = False
        self._warming_up    = False
        self._warmup_frames = 0
        self._select_start  = None
        self._adjust_start  = None
        self._adjust_last   = 0.0
        self._last_item_idx = -1
        self._x_zone        = "center"

    def _reset_warmup(self):
        """Hand disappeared before we confirmed -- cancel the warmup."""
        self._warming_up    = False
        self._warmup_frames = 0

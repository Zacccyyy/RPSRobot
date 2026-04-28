"""
gesture_nav.py  —  Gesture-based menu navigation with +/- zone support.

INTERACTION
-----------
  Y position → which item is highlighted (instant, raw Y, no lag)
  Hand center (X 0.40–0.60) → 2s dwell → select
  Hand left   (X < 0.40)    → 1s dwell → adjust −1, repeats every 0.5s
  Hand right  (X > 0.60)    → 1s dwell → adjust +1, repeats every 0.5s

Adjust zones only activate after ITEM_STABLE_TIME on the same item,
preventing accidental triggers while scrolling between rows.

EVENTS
------
  {"type": "hover",  "item_index": int}
  {"type": "select"}
  {"type": "adjust", "direction": -1 | +1}
"""

import time

DWELL_SELECT   = 2.0
DWELL_ADJUST   = 1.0
ADJUST_REPEAT  = 0.5
ITEM_STABLE    = 0.35
SMOOTHING      = 0.60
CONFIRM_FRAMES = 5

# Button hit zones — must match draw_settings_screen in ui_renderer.py.
# Computed from: x2=0.935, btn_w=0.055, btn_gap=0.010, x2_offset=0.018
_BTN_W    = 0.055
_BTN_GAP  = 0.010
_PLUS_X2  = 0.935 - 0.018                      # 0.917
_PLUS_X1  = _PLUS_X2 - _BTN_W                  # 0.862
_MINUS_X2 = _PLUS_X1 - _BTN_GAP                # 0.852
_MINUS_X1 = _MINUS_X2 - _BTN_W                 # 0.797
_ZONE_SEP = (_MINUS_X2 + _PLUS_X1) / 2         # 0.857 — mid-gap boundary

# Expand each zone 0.02 inward toward panel centre for easier targeting
ZONE_MINUS = (_MINUS_X1 - 0.020, _ZONE_SEP)    # ≈ (0.777, 0.857)
ZONE_PLUS  = (_ZONE_SEP, _PLUS_X2 + 0.020)     # ≈ (0.857, 0.937)

_INDEX_TIP = 8


class GestureNavController:

    def __init__(self):
        self._active        = False
        self._warming_up    = False
        self._warmup_frames = 0
        # Item tracking
        self._last_item_idx = -1
        self._item_since    = 0.0   # when current item was first entered
        # Select dwell
        self._select_start  = None
        # Adjust zone tracking
        self._x_zone        = "center"   # "minus" | "center" | "plus"
        self._adjust_start  = None
        self._adjust_last   = 0.0
        # Smoothed coords for display
        self._smooth_x      = None
        self._smooth_y      = None
        # Content mapping
        self._item_count    = 1
        self._content_top   = 0.44
        self._content_bot   = 0.83
        # Whether this screen supports +/− (set per-call)
        self._adjust_items  = set()

    # ------------------------------------------------------------------ #

    def update(self, hand_state, now=None, item_count=1,
               content_top=0.44, content_bottom=0.83,
               adjust_items=None):
        """
        adjust_items : set of item indices that have +/− buttons.
                       Those items suppress center-dwell; all others behave normally.
                       Pass None or empty set to disable adjust zones entirely.
        """
        if now is None:
            now = time.monotonic()

        self._item_count  = max(item_count, 1)
        self._content_top = content_top
        self._content_bot = content_bottom
        self._adjust_items = adjust_items or set()
        has_adjust = bool(self._adjust_items)

        if self._item_count > 1:
            _gap = (content_bottom - content_top) / (self._item_count - 1)
            self._content_top = content_top - _gap * 0.5
            self._content_bot = content_bottom + _gap * 0.5

        events = []
        lm_obj = hand_state.get("_landmarks")
        lm     = lm_obj.landmark if lm_obj is not None else None

        hand_present = lm is not None
        raw_x = lm[_INDEX_TIP].x if hand_present else None
        raw_y = lm[_INDEX_TIP].y if hand_present else None

        # Smooth for display only
        if raw_x is not None:
            if self._smooth_x is None:
                self._smooth_x, self._smooth_y = raw_x, raw_y
            else:
                self._smooth_x = SMOOTHING * raw_x + (1 - SMOOTHING) * self._smooth_x
                self._smooth_y = SMOOTHING * raw_y + (1 - SMOOTHING) * self._smooth_y

        if not self._active:
            if hand_present:
                self._warming_up    = True
                self._warmup_frames += 1
                if self._warmup_frames >= CONFIRM_FRAMES:
                    self._activate(now)
            else:
                self._reset_warmup()
        else:
            if hand_present and raw_y is not None:
                span     = self._content_bot - self._content_top
                fraction = (raw_y - self._content_top) / span if span > 0 else 0.5
                fraction = max(0.0, min(1.0, fraction))
                item_idx = min(int(fraction * self._item_count), self._item_count - 1)

                events.append({"type": "hover", "item_index": item_idx})

                # Track how long we've been on this item
                if item_idx != self._last_item_idx:
                    self._last_item_idx = item_idx
                    self._item_since    = now
                    self._select_start  = now
                    self._adjust_start  = None
                    self._adjust_last   = 0.0
                    self._x_zone        = "center"

                # Is the current item one with +/− buttons?
                item_is_adj = item_idx in self._adjust_items
                item_age    = now - self._item_since
                zones_on    = item_is_adj and item_age >= ITEM_STABLE
                sx          = self._smooth_x
                new_zone    = "center"
                if zones_on and sx is not None:
                    if ZONE_MINUS[0] <= sx <= ZONE_MINUS[1]:
                        new_zone = "minus"
                    elif ZONE_PLUS[0] <= sx <= ZONE_PLUS[1]:
                        new_zone = "plus"

                if new_zone != self._x_zone:
                    self._x_zone       = new_zone
                    self._adjust_start = now if new_zone != "center" else None
                    self._select_start = now if new_zone == "center" else None

                # Fire events based on zone
                if self._x_zone == "center":
                    # Suppress dwell-to-select only for items that have +/− buttons
                    if not item_is_adj:
                        if self._select_start and (now - self._select_start) >= DWELL_SELECT:
                            events.append({"type": "select"})
                            self._deactivate()

                elif self._adjust_start is not None:
                    direction = -1 if self._x_zone == "minus" else +1
                    elapsed   = now - self._adjust_start
                    if elapsed >= DWELL_ADJUST:
                        since_last = now - self._adjust_last
                        # First fire: once dwell is reached
                        # Repeat fires: every ADJUST_REPEAT after that
                        if self._adjust_last == 0.0 or since_last >= ADJUST_REPEAT:
                            events.append({"type": "adjust", "direction": direction})
                            self._adjust_last = now

            else:
                self._deactivate()

        return events

    def reset(self):
        self._active        = False
        self._warming_up    = False
        self._warmup_frames = 0
        self._last_item_idx = -1
        self._item_since    = 0.0
        self._select_start  = None
        self._x_zone        = "center"
        self._adjust_start  = None
        self._adjust_last   = 0.0
        self._smooth_x      = None
        self._smooth_y      = None
        self._adjust_items  = set()

    def is_active(self):     return self._active
    def is_warming_up(self): return self._warming_up and not self._active

    def get_cursor_info(self):
        now = time.monotonic()

        select_pct = 0.0
        adjust_pct = 0.0
        if self._active:
            current_is_adj = self._last_item_idx in getattr(self, "_adjust_items", set())
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
            "dwell_pct":   select_pct,    # select dwell (for center zone)
            "adjust_pct":  adjust_pct,    # adjust dwell (for +/- zones)
            "x_zone":      self._x_zone,
            "has_adjust":  bool(self._adjust_items),
        }

    def _activate(self, now):
        self._active        = True
        self._warming_up    = False
        self._warmup_frames = 0
        self._select_start  = now
        self._last_item_idx = -1

    def _deactivate(self):
        self._active        = False
        self._warming_up    = False
        self._warmup_frames = 0
        self._select_start  = None
        self._adjust_start  = None
        self._adjust_last   = 0.0
        self._last_item_idx = -1
        self._x_zone        = "center"

    def _reset_warmup(self):
        self._warming_up    = False
        self._warmup_frames = 0

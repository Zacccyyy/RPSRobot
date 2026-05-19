"""
ui_base.py -- Colours, layout helpers, drawing primitives, text utilities.

This is the foundation layer of the UI. Every other ui_*.py module imports
from here via `from ui_base import *`. Think of it as the shared design system:
  - BGR colour constants for every element
  - Font and scale constants so all text sizes are consistent
  - Layout helpers that compute pixel regions as percentages of frame size
  - Core draw_ functions for panels, text, bars, icons, etc.

All public names are re-exported by ui_renderer.py so callers can import
from a single place if they prefer.
"""
import cv2
import math
import time

__all__ = [
    # Spec palette constants
    'COL_PANEL_BG', 'COL_PANEL_ALPHA', 'COL_ROW_SELECTED', 'COL_BORDER_HAIR',
    'COL_ACCENT', 'COL_TEXT_PRIMARY', 'COL_TEXT_SECONDARY', 'COL_TEXT_DIM',
    'COL_AMBER', 'COL_GREEN', 'COL_RED',
    'COL_BEAT_FILL', 'COL_BEAT_RING', 'COL_ON_ACTIVE', 'COL_INACTIVE',
    # Legacy aliases (old names, new values -- sibling modules use these)
    'COL_BG_DARK', 'COL_BG_PANEL', 'COL_BG_PANEL_LIGHT',
    'COL_CYAN', 'COL_MAGENTA', 'COL_YELLOW', 'COL_ORANGE',
    'COL_TEXT', 'COL_TEXT_ACCENT',
    # Colourblind variants
    '_COL_CB_WIN', '_COL_CB_LOSE', '_COL_CB_DRAW',
    # Colour helpers
    '_result_colour', '_get_emotion_color',
    # Layout helpers
    '_ix', '_fit_rect', '_frame_size',
    '_game_layout', '_menu_layout', '_settings_layout',
    # Glow helper stub (backward compat)
    '_draw_glow_border',
    # Typography constants
    'FONT_PRIMARY', 'FONT_DISPLAY',
    'SCALE_DISPLAY_XL', 'SCALE_DISPLAY_L', 'SCALE_HEADING',
    'SCALE_BODY', 'SCALE_CAPTION', 'SCALE_MICRO',
    # Geometry constants
    'TOP_BAR_PCT', 'BOTTOM_BAR_PCT', 'PANEL_INSET_X', 'PANEL_INSET_Y',
    # Public drawing functions
    'get_gesture_color', 'draw_panel', 'draw_gesture_icon', 'draw_result_flash',
    'draw_gesture_confidence_bar', 'draw_quality_warnings', 'draw_round_history_dots',
    'draw_help_overlay', 'get_fit_scale', 'draw_outlined_text',
    'draw_centered_text', 'draw_centered_text_in_rect',
    'draw_top_bar', 'draw_bottom_bar',
    'draw_status_chip', 'get_result_banner_color',
    # New spec primitives
    'draw_gesture_glyph', 'draw_gesture_badge', 'draw_beat_track',
    'draw_progress_bar', 'draw_selected_row', 'draw_row',
]

# ============================================================
# SPEC PALETTE -- all BGR tuples for cv2
# OpenCV uses BGR order, NOT RGB. So (0, 200, 220) is yellow, etc.
# ============================================================

COL_PANEL_BG       = (20, 15, 10)     # #0A0F14 frosted panel background
COL_PANEL_ALPHA    = 0.78             # default transparency for panels
COL_ROW_SELECTED   = (45, 38, 30)     # #1E262D selected row fill
COL_BORDER_HAIR    = (36, 36, 36)     # hairline border (opaque equivalent of 14% white)
COL_ACCENT         = (210, 160, 60)   # #3CA0D2 soft blue (BGR)
COL_TEXT_PRIMARY   = (255, 255, 255)  # #FFFFFF
COL_TEXT_SECONDARY = (160, 160, 160)  # #A0A0A0
COL_TEXT_DIM       = (80, 80, 80)     # #505050
COL_AMBER          = (40, 160, 220)   # #DCA028 warning / bluff
COL_GREEN          = (80, 180, 80)    # #50B450 win / success
COL_RED            = (60, 60, 200)    # #C83C3C lose / error

# Legacy aliases -- old names now point to spec values so sibling modules
# pick up the new palette without any changes to their source.
COL_BG_DARK        = COL_PANEL_BG
COL_BG_PANEL       = COL_PANEL_BG
COL_BG_PANEL_LIGHT = (28, 22, 16)
COL_CYAN           = (200, 200,   0)   # teal-cyan  (BGR)
COL_MAGENTA        = (180,   0, 180)   # purple-magenta (BGR)
COL_YELLOW         = (  0, 200, 220)   # warm yellow (BGR)
COL_ORANGE         = (  0, 130, 255)   # orange (BGR)
COL_TEXT           = COL_TEXT_PRIMARY
COL_TEXT_ACCENT    = COL_ACCENT

# UI primitive colours (shared across beat-track, indicator rows, toggles)
COL_BEAT_FILL  = (24, 24, 24)    # inactive beat circle fill
COL_BEAT_RING  = (56, 56, 56)    # inactive beat circle ring
COL_ON_ACTIVE  = (10, 10, 10)    # dark label text on filled/active elements
COL_INACTIVE   = (40, 40, 60)    # unselected indicator / dim border

# Colourblind-safe variants used when colourblind mode is enabled
_COL_CB_WIN  = COL_GREEN
_COL_CB_LOSE = COL_RED
_COL_CB_DRAW = COL_TEXT_SECONDARY

# ============================================================
# TYPOGRAPHY CONSTANTS
# These are used everywhere text is drawn -- consistent sizes mean
# the whole UI feels coherent even across different frame resolutions.
# ============================================================

FONT_PRIMARY     = cv2.FONT_HERSHEY_SIMPLEX
FONT_DISPLAY     = cv2.FONT_HERSHEY_DUPLEX  # heavier weight for big titles
SCALE_DISPLAY_XL = 1.7   # very large: countdown numbers, big result text
SCALE_DISPLAY_L  = 1.2   # large: section headings, SHOOT! prompt
SCALE_HEADING    = 0.90  # medium-large: panel titles
SCALE_BODY       = 0.55  # normal body text
SCALE_CAPTION    = 0.45  # smaller captions
SCALE_MICRO      = 0.40  # tiny labels, hints

# ============================================================
# GEOMETRY CONSTANTS
# Positions are stored as fractions of frame height/width so the
# layout automatically scales to any camera resolution.
# ============================================================

TOP_BAR_PCT    = 0.06    # top bar takes 6% of frame height
BOTTOM_BAR_PCT = 0.06    # same for the bottom bar
PANEL_INSET_X  = 0.036   # 3.6% left/right inset for full-width panels
PANEL_INSET_Y  = 0.105   # below the top bar

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def _frame_size(frame):
    """Return (width, height) from an OpenCV frame (shape is h,w,c)."""
    h, w = frame.shape[:2]
    return w, h

def _ix(value):
    """Convert a float pixel position to the nearest int. Used everywhere
    so sub-pixel maths stays clean until the final cv2 call."""
    return int(round(value))

def _fit_rect(x1, y1, x2, y2):
    """Convert a float bounding box to a 4-tuple of ints."""
    return (_ix(x1), _ix(y1), _ix(x2), _ix(y2))

def _game_layout(frame):
    """
    Compute all named pixel regions for the main game screen.
    Returns a dict with keys like 'hero', 'beat_track', 'result', etc.
    Every value is already rounded to int via _ix() or _fit_rect().
    """
    w, h = _frame_size(frame)
    top_bar_h    = _ix(h * TOP_BAR_PCT)
    bottom_bar_h = _ix(h * BOTTOM_BAR_PCT)
    return {
        "w": w, "h": h,
        "top_bar_h":      top_bar_h,
        "top_row_h":      top_bar_h,     # legacy key kept for backward compat
        "second_row_h":   0,             # removed -- single top bar now
        "header_total_h": top_bar_h,     # legacy key kept for backward compat
        "bottom_bar_h":   bottom_bar_h,
        "arcade_title_y": _ix(h * 0.17),
        "arcade_lights_y":_ix(h * 0.20),
        # Named zones as (x1, y1, x2, y2) pixel rects
        "status_strip": _fit_rect(w * 0.08, h * 0.18, w * 0.92, h * 0.24),
        "hero":        _fit_rect(w * 0.08, h * 0.26, w * 0.92, h * 0.68),
        "beat_track":  _fit_rect(w * 0.15, h * 0.76, w * 0.85, h * 0.90),
        "result":      _fit_rect(w * 0.08, h * 0.26, w * 0.92, h * 0.90),
        "gesture_row": _fit_rect(w * 0.08, h * 0.09, w * 0.92, h * 0.17),
    }

def _menu_layout(frame):
    """Compute pixel regions for a full-page menu screen."""
    w, h = _frame_size(frame)
    return {
        "w": w, "h": h,
        "panel": _fit_rect(w * PANEL_INSET_X, h * 0.10,
                           w * (1 - PANEL_INSET_X), h * 0.92),
        "bottom_bar_h": _ix(h * BOTTOM_BAR_PCT),
    }

def _settings_layout(frame):
    """Compute pixel regions for the settings/features screens.
    Shorter panel height than menus to leave room for a description box below."""
    w, h = _frame_size(frame)
    return {
        "w": w, "h": h,
        "panel": _fit_rect(w * PANEL_INSET_X, h * 0.10,
                           w * (1 - PANEL_INSET_X), h * 0.75),
        "bottom_bar_h": _ix(h * BOTTOM_BAR_PCT),
    }

# ============================================================
# COLOUR HELPERS
# ============================================================

def _result_colour(result_str, colourblind=False):
    """Map a result string like 'YOU WIN' or 'DRAW' to a display colour."""
    r = result_str.upper()
    if "WIN" in r or "SURVIVE" in r:
        return COL_GREEN
    if "DRAW" in r or "AGAIN" in r:
        return COL_TEXT_SECONDARY
    return COL_RED

def _get_emotion_color(emotion):
    """Return the colour associated with a detected facial emotion."""
    if emotion == "Happy":      return COL_GREEN
    if emotion == "Surprised":  return COL_AMBER
    if emotion == "Frustrated": return COL_RED
    return COL_TEXT_DIM

def get_gesture_color(gesture):
    """Return white for valid RPS(LS) gestures, dim grey for anything else."""
    if gesture in ("Rock", "Paper", "Scissors", "Lizard", "Spock"):
        return COL_TEXT_PRIMARY
    return COL_TEXT_DIM

# ============================================================
# CORE DRAWING PRIMITIVES
# ============================================================

def draw_panel(frame, x1, y1, x2, y2, fill=None, alpha=None,
               border=None, border_thickness=1):
    """
    Draw a semi-transparent filled rectangle with an optional border.
    Uses addWeighted to blend the fill colour over the camera image underneath,
    giving a frosted-glass look rather than a solid opaque box.
    Clamps coordinates to frame bounds so callers never need to guard for that.
    """
    if fill  is None: fill  = COL_PANEL_BG
    if alpha is None: alpha = COL_PANEL_ALPHA
    if border is None: border = COL_BORDER_HAIR
    h, w = frame.shape[:2]
    # Clamp to frame bounds to avoid out-of-bounds slice
    x1i = max(0, int(x1));  y1i = max(0, int(y1))
    x2i = min(w - 1, int(x2)); y2i = min(h - 1, int(y2))
    if x2i <= x1i or y2i <= y1i:
        return  # nothing to draw
    roi     = frame[y1i:y2i, x1i:x2i]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (x2i - x1i, y2i - y1i), fill, -1)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)
    if border_thickness > 0:
        cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), border, border_thickness)

def _draw_glow_border(frame, x1, y1, x2, y2, color, thickness=1):
    """
    Backward-compat stub that used to draw a multi-pass glow.
    Now just draws a single solid border (glow effect was removed for performance).
    """
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

def draw_outlined_text(frame, text, x, y, scale, color, thickness=1, outline=2):
    """
    Draw text with a black drop-shadow/outline pass first, then the coloured text on top.
    The outline pass uses a thicker stroke so it bleeds around the coloured text,
    making it readable against any background colour.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Black background pass
    cv2.putText(frame, text, (int(x), int(y)), font, scale,
                (0, 0, 0), thickness + outline, cv2.LINE_AA)
    # Coloured foreground pass
    cv2.putText(frame, text, (int(x), int(y)), font, scale,
                color, thickness, cv2.LINE_AA)

def get_fit_scale(text, max_width, base_scale=1.0, thickness=2, min_scale=0.35):
    """
    Binary-search for the largest font scale that fits 'text' within max_width pixels.
    Starts at base_scale; if it already fits, returns it unchanged.
    Runs 8 iterations of bisection -- good enough precision for any UI text.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Fast path: already fits at base scale
    if cv2.getTextSize(text, font, base_scale, thickness)[0][0] <= max_width:
        return base_scale
    # Binary search between min_scale and base_scale
    lo, hi = min_scale, base_scale
    for _ in range(8):
        mid = (lo + hi) / 2
        if cv2.getTextSize(text, font, mid, thickness)[0][0] <= max_width:
            lo = mid
        else:
            hi = mid
    return lo

def draw_centered_text(frame, text, center_y, scale, color, thickness=2, outline=4):
    """Draw text horizontally centred on the frame at the given y position."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, _), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (frame.shape[1] - text_w) // 2
    draw_outlined_text(frame, text, x, center_y, scale, color, thickness, outline)

def draw_centered_text_in_rect(frame, text, rect, base_scale, color,
                                thickness=2, outline=4):
    """
    Draw text centred both horizontally and vertically within a pixel rect.
    Shrinks the font scale if the text is too wide for the rect.
    """
    x1, y1, x2, y2 = rect
    max_width = max(40, (x2 - x1) - 20)
    scale = get_fit_scale(text, max_width, base_scale=base_scale,
                          thickness=thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x = x1 + ((x2 - x1) - text_w) // 2
    y = y1 + ((y2 - y1) + text_h) // 2
    draw_outlined_text(frame, text, x, y, scale, color, thickness, outline)

# ============================================================
# SPEC DRAWING PRIMITIVES
# ============================================================

def draw_gesture_glyph(frame, gesture, rect, color=None):
    """
    Draw a flat, outline-only gesture icon inside the given rect.
    Each gesture maps to a distinct simple shape so they're recognisable
    even at small sizes or in peripheral vision:
      Rock     -> circle
      Paper    -> square
      Scissors -> X
      Lizard   -> wide ellipse
      Spock    -> three vertical lines + one crossbar
    Unknown/invalid gesture draws a small dim square as a placeholder.
    """
    x1, y1, x2, y2 = rect
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    side  = min(x2 - x1, y2 - y1)
    col   = color or COL_TEXT_PRIMARY
    thick = max(2, _ix(side * 0.04))  # line thickness scales with icon size

    if gesture == "Rock":
        r = _ix(side * 0.40)
        cv2.circle(frame, (cx, cy), r, col, thick)
    elif gesture == "Paper":
        half = _ix(side * 0.38)
        cv2.rectangle(frame, (cx - half, cy - half), (cx + half, cy + half),
                      col, thick)
    elif gesture == "Scissors":
        d = _ix(side * 0.38)
        cv2.line(frame, (cx - d, cy - d), (cx + d, cy + d), col, thick)
        cv2.line(frame, (cx + d, cy - d), (cx - d, cy + d), col, thick)
    elif gesture == "Lizard":
        axes = (_ix(side * 0.42), _ix(side * 0.22))
        cv2.ellipse(frame, (cx, cy), axes, 0, 0, 360, col, thick)
    elif gesture == "Spock":
        # Three vertical lines + one horizontal crossbar to suggest a Vulcan salute
        gap = _ix(side * 0.14)
        top = cy - _ix(side * 0.36)
        bot = cy + _ix(side * 0.36)
        bar = cy - _ix(side * 0.08)
        for dx in (-gap, 0, gap):
            cv2.line(frame, (cx + dx, top), (cx + dx, bot), col, thick)
        cv2.line(frame, (cx - gap, bar), (cx + gap, bar), col, thick)
    else:
        # Unknown gesture: dim placeholder square
        half = _ix(side * 0.28)
        cv2.rectangle(frame, (cx - half, cy - half), (cx + half, cy + half),
                      COL_TEXT_DIM, 1)

def draw_gesture_icon(frame, gesture, cx, cy, size):
    """Backward-compatible wrapper: draw a gesture glyph centred at (cx, cy)."""
    rect = (cx - size, cy - size, cx + size, cy + size)
    draw_gesture_glyph(frame, gesture, rect, color=COL_TEXT_PRIMARY)

def draw_gesture_badge(frame, gesture, confidence, x, y,
                       threshold=0.70, show_confidence=True):
    """
    Draw a small info badge showing the detected gesture name and a confidence dot.
    The border glows accent-coloured when confidence is above the threshold,
    stays dim/hairline when below it -- giving instant visual feedback on
    whether the reading is trustworthy.
    """
    h, fw = frame.shape[:2]
    pad_x, pad_y = 14, 8
    font   = cv2.FONT_HERSHEY_SIMPLEX
    g_text = gesture.upper() if gesture else "NONE"
    (gw, gh), _ = cv2.getTextSize(g_text, font, SCALE_BODY, 1)

    det_text = "DETECTED"
    (dw, dh), _ = cv2.getTextSize(det_text, font, SCALE_MICRO, 1)

    # Compute box dimensions to fit both lines and the status dot
    dot_r   = 4
    inner_w = dot_r * 2 + 6 + max(gw, dw)
    box_w   = inner_w + pad_x * 2
    box_h   = gh + dh + pad_y * 2 + 6

    x2 = x + box_w
    y2 = y + box_h

    # Border colour signals confidence level: accent=good, hairline=low
    confident  = confidence >= threshold
    borderline = 0.50 <= confidence < threshold
    border_col = COL_ACCENT if confident else COL_BORDER_HAIR

    draw_panel(frame, x, y, x2, y2, fill=COL_PANEL_BG, alpha=0.88,
               border=border_col, border_thickness=1)

    # Status dot: green=confirmed, amber=borderline, dim=too uncertain
    dot_col = COL_GREEN if confident else (COL_AMBER if borderline else COL_TEXT_DIM)
    dot_cx  = x + pad_x + dot_r
    dot_cy  = y + box_h // 2
    cv2.circle(frame, (dot_cx, dot_cy), dot_r, dot_col, -1)

    # "DETECTED" micro label above the gesture name
    text_x = dot_cx + dot_r + 6
    cv2.putText(frame, det_text,
                (text_x, y + pad_y + dh),
                font, SCALE_MICRO, COL_TEXT_DIM, 1, cv2.LINE_AA)

    # Gesture name with optional confidence value appended
    conf_suffix = ""
    if show_confidence and confidence > 0:
        conf_suffix = f"  {confidence:.2f}"
    label = g_text + conf_suffix
    draw_outlined_text(frame, label, text_x, y + pad_y + dh + 6 + gh,
                       SCALE_BODY, COL_TEXT_PRIMARY, thickness=1, outline=2)

def draw_beat_track(frame, beat_count, num_beats=4, state="", x1=None, y1=None,
                    x2=None, y2=None):
    """
    Draw the beat-track row: a horizontal strip of num_beats numbered circles.
    Circles fill in left-to-right as beats are counted.
    The final beat turns red during the SHOOT_WINDOW to warn the player to throw.
    Inactive circles show a dim ring; active ones fill with accent colour.
    Falls back to _game_layout() coordinates if no explicit rect is provided.
    """
    w, h = _frame_size(frame)
    if x1 is None:
        bx1, by1, bx2, by2 = _game_layout(frame)["beat_track"]
    else:
        bx1, by1, bx2, by2 = x1, y1, x2, y2

    draw_panel(frame, bx1, by1, bx2, by2,
               fill=COL_PANEL_BG, alpha=0.78, border=COL_BORDER_HAIR, border_thickness=1)

    ph = by2 - by1
    pw = bx2 - bx1

    # "BEAT TRACK" caption centred near the top of the strip
    cap_y = by1 + _ix(ph * 0.28)
    (cap_lw, _), _ = cv2.getTextSize("BEAT TRACK", FONT_PRIMARY, SCALE_MICRO, 1)
    cv2.putText(frame, "BEAT TRACK", (bx1 + (pw - cap_lw) // 2, cap_y),
                FONT_PRIMARY, SCALE_MICRO, COL_TEXT_SECONDARY, 1, cv2.LINE_AA)

    # Circle geometry: size relative to frame height, capped at 38px diameter
    diameter = min(_ix(h * 0.052), 38)
    radius   = diameter // 2
    gap      = _ix(h * 0.037)
    total_w  = num_beats * diameter + (num_beats - 1) * gap
    start_x  = bx1 + (pw - total_w) // 2 + radius
    cy       = by1 + _ix(ph * 0.62)

    is_shoot = state in ("SHOOT_WINDOW",)
    for i in range(num_beats):
        cx  = start_x + i * (diameter + gap)
        act = i < beat_count
        # Last beat goes red during the shoot window as an urgency cue
        shoot_beat = is_shoot and (i == num_beats - 1)

        if shoot_beat:
            col = COL_RED
        elif act:
            col = COL_ACCENT
        else:
            col = COL_BEAT_RING

        if act or shoot_beat:
            cv2.circle(frame, (cx, cy), radius, col, -1)        # filled circle
            num_col = COL_ON_ACTIVE                              # dark number on filled
        else:
            cv2.circle(frame, (cx, cy), radius, col, 2)         # ring only
            num_col = COL_TEXT_DIM                               # dim number in ring

        # Centred beat number inside the circle
        label = str(i + 1)
        (lw, lh), _ = cv2.getTextSize(label, FONT_PRIMARY, SCALE_MICRO, 1)
        cv2.putText(frame, label, (cx - lw // 2, cy + lh // 2),
                    FONT_PRIMARY, SCALE_MICRO, num_col, 1, cv2.LINE_AA)

    # Hint text below circles changes based on whether shoot window is active
    hint = "4th beat opens SHOOT" if not is_shoot else "MAKE YOUR THROW"
    hint_y = by1 + _ix(ph * 0.90)
    (hw, _), _ = cv2.getTextSize(hint, FONT_PRIMARY, SCALE_MICRO, 1)
    cv2.putText(frame, hint, (bx1 + (pw - hw) // 2, hint_y),
                FONT_PRIMARY, SCALE_MICRO, COL_TEXT_DIM, 1, cv2.LINE_AA)

def draw_progress_bar(frame, x1, y1, x2, y2, value, color=None, track_height=4):
    """
    Draw a simple horizontal progress bar inside the given rect.
    value should be in the range 0.0 to 1.0; it is clamped internally.
    The bar is vertically centred within the rect using the track_height.
    """
    col = color or COL_ACCENT
    bh  = max(3, min(track_height, y2 - y1))
    mid = y1 + (y2 - y1 - bh) // 2
    # Dark track background
    cv2.rectangle(frame, (x1, mid), (x2, mid + bh), (28, 28, 28), -1)
    # Filled portion, proportional to value
    fill_x = x1 + _ix((x2 - x1) * max(0.0, min(1.0, value)))
    if fill_x > x1:
        cv2.rectangle(frame, (x1, mid), (fill_x, mid + bh), col, -1)

def draw_selected_row(frame, x1, y1, x2, y2, accent_bar=True):
    """
    Highlight a selected list row with a translucent fill and an optional
    2-px accent-coloured left edge bar (the 'active indicator').
    """
    roi     = frame[y1:y2, x1:x2]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (x2 - x1, y2 - y1), COL_ROW_SELECTED, -1)
    cv2.addWeighted(overlay, 0.95, roi, 0.05, 0, roi)
    if accent_bar:
        cv2.rectangle(frame, (x1, y1), (x1 + 2, y2), COL_ACCENT, -1)

def draw_row(frame, x1, y1, x2, y2, label, selected=False,
             sub_label='', right_hint=''):
    """
    Draw a single list row with optional selection highlight.
    When selected:
      - Row gets the highlight fill from draw_selected_row
      - sub_label appears beneath the main label in dim text
      - right_hint appears right-aligned (e.g. 'ENTER FIGHT')
    When not selected, just the label is drawn in a dim colour.
    """
    row_h = y2 - y1
    if selected:
        # Slightly inset fill so it doesn't bleed into the panel border
        roi = frame[y1:y2, x1+2:x2-2]
        overlay = roi.copy()
        cv2.rectangle(overlay, (0, 0), (x2 - x1 - 4, row_h), COL_ROW_SELECTED, -1)
        cv2.addWeighted(overlay, 0.95, roi, 0.05, 0, roi)
        cv2.rectangle(frame, (x1, y1), (x1 + 2, y2), COL_ACCENT, -1)
    text_color = COL_TEXT_PRIMARY if selected else COL_TEXT_SECONDARY
    pad = _ix((x2 - x1) * 0.025)
    text_x = x1 + pad + (4 if selected else 0)
    # Shift label up slightly when there is a sub_label to make room
    label_y = y1 + _ix(row_h * (0.48 if not sub_label else 0.38))
    draw_outlined_text(frame, label, text_x, label_y, SCALE_BODY, text_color,
                       thickness=1, outline=2)
    # Sub-label and right hint are only visible when the row is selected
    if sub_label and selected:
        draw_outlined_text(frame, sub_label, text_x, y1 + _ix(row_h * 0.72),
                           SCALE_CAPTION, COL_TEXT_DIM, thickness=1, outline=2)
    if right_hint and selected:
        (tw, _), _ = cv2.getTextSize(right_hint, FONT_PRIMARY, SCALE_MICRO, 1)
        draw_outlined_text(frame, right_hint, x2 - pad - tw, label_y,
                           SCALE_MICRO, COL_ACCENT, thickness=1, outline=2)

# ============================================================
# BARS (top and bottom HUD strips)
# ============================================================

def draw_top_bar(frame, left_label, right_hints=''):
    """
    Draw the HUD top bar: y=0, height=6% of frame.
    left_label is drawn in accent colour (typically the mode or screen name).
    right_hints is drawn in dim micro text on the right edge (keyboard shortcuts).
    A hairline separator runs along the bottom edge of the bar.
    """
    w, h  = _frame_size(frame)
    bar_h = _ix(h * 0.06)

    # Semi-transparent dark fill blended over the camera feed
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), COL_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    cv2.line(frame, (0, bar_h - 1), (w, bar_h - 1), COL_BORDER_HAIR, 1)

    text_y = _ix(bar_h * 0.68)  # vertically centred within the bar

    draw_outlined_text(frame, left_label, _ix(w * 0.02), text_y,
                       SCALE_BODY, COL_ACCENT, thickness=2, outline=2)

    if right_hints:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, _), _ = cv2.getTextSize(right_hints, font, SCALE_MICRO, 1)
        rx = w - tw - _ix(w * 0.02)
        draw_outlined_text(frame, right_hints, rx, text_y,
                           SCALE_MICRO, COL_TEXT_DIM, thickness=1, outline=2)

def draw_bottom_bar(frame, hints=''):
    """
    Draw the HUD bottom bar: y=94%, height=6%.
    hints is typically a pipe-separated list of keyboard shortcuts.
    A hairline separator runs along the top edge of the bar.
    """
    w, h  = _frame_size(frame)
    bar_h = _ix(h * 0.06)
    y1    = h - bar_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y1), (w, h), COL_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    cv2.line(frame, (0, y1), (w, y1), COL_BORDER_HAIR, 1)

    if hints:
        text_y = y1 + _ix(bar_h * 0.68)
        draw_outlined_text(frame, hints, _ix(w * 0.02), text_y,
                           SCALE_MICRO, COL_TEXT_DIM, thickness=1, outline=2)

# ============================================================
# STATUS CHIP
# ============================================================

def draw_status_chip(frame, text, y_center, color):
    """
    Draw a small pill/chip with a coloured border centred horizontally.
    Used for things like 'AI' labels and state indicators.
    The chip auto-sizes to fit the text.
    """
    w, h = _frame_size(frame)
    font      = cv2.FONT_HERSHEY_SIMPLEX
    # Shrink font scale if the text would be too wide
    scale     = get_fit_scale(text, _ix(w * 0.52), base_scale=SCALE_BODY, thickness=1)
    thickness = 1
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    pad_x = _ix(w * 0.018)
    pad_y = _ix(h * 0.012)
    x1 = (w - text_w) // 2 - pad_x
    y1 = y_center - text_h - pad_y
    x2 = x1 + text_w + pad_x * 2
    y2 = y_center + pad_y

    draw_panel(frame, x1, y1, x2, y2,
               fill=COL_PANEL_BG, alpha=0.88, border=color, border_thickness=1)
    draw_outlined_text(frame, text, x1 + pad_x, y_center - 2,
                       scale, color, thickness=1, outline=2)

# ============================================================
# RESULT BANNER COLOUR
# ============================================================

def get_result_banner_color(banner, colourblind=False):
    """
    Map a result banner string to its display colour.
    Checked in priority order: win > draw > lose > neutral.
    """
    if banner.startswith("YOU WIN") or banner.startswith("YOU SURVIVE"):
        return COL_GREEN
    if banner.startswith("DRAW"):
        return COL_TEXT_PRIMARY
    if banner.startswith("GAME OVER") or "TAKES" in banner or "WINS" in banner:
        return COL_RED
    return COL_TEXT_PRIMARY

# ============================================================
# COMPLEX HELPERS
# ============================================================

def draw_gesture_confidence_bar(frame, stable_streak, confirm_frames, x, y, width):
    """
    Vertical gesture-lock bar drawn near the left edge of the frame.
    Fills upward as the gesture stabilises (stable_streak approaches confirm_frames).
    Turns green when fully confirmed, stays accent-coloured while building.
    Labelled 'LOCK' with letters stacked vertically above the bar.
    """
    fh, fw = frame.shape[:2]
    pct    = min(1.0, stable_streak / max(confirm_frames, 1))
    bar_w  = max(8, fw // 60)
    bar_h  = _ix(fh * 0.22)
    bar_x  = _ix(fw * 0.012)
    bar_bot = fh - _ix(fh * TOP_BAR_PCT) - 2
    bar_top = bar_bot - bar_h

    # Dark background track
    cv2.rectangle(frame, (bar_x, bar_top), (bar_x + bar_w, bar_bot),
                  COL_BEAT_FILL, -1)
    cv2.rectangle(frame, (bar_x, bar_top), (bar_x + bar_w, bar_bot),
                  COL_BORDER_HAIR, 1)

    # Filled portion grows upward from bar_bot
    if pct > 0:
        fill_h   = _ix(bar_h * pct)
        fill_top = bar_bot - fill_h
        col = COL_GREEN if pct >= 1.0 else COL_ACCENT
        cv2.rectangle(frame, (bar_x, fill_top), (bar_x + bar_w, bar_bot), col, -1)

    # 'LOCK' label with one character per row, centred above the bar
    label = "LOCK"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    lh    = _ix(fh * 0.026)
    for i, ch in enumerate(label):
        cy = bar_top - _ix(fh * 0.004) - (len(label) - 1 - i) * lh
        (cw, _), _ = cv2.getTextSize(ch, font, 0.26, 1)
        cx = bar_x + bar_w // 2 - cw // 2
        cv2.putText(frame, ch, (cx, cy), font, 0.26,
                    COL_TEXT_DIM, 1, cv2.LINE_AA)

def draw_result_flash(frame, result, flash_frame_idx,
                      max_flash_frames=4, colourblind=False):
    """
    Full-screen tinted flash that plays over a few frames after a round result.
    Alpha fades out linearly so the flash doesn't linger too long.
    result should be one of: 'win', 'lose', 'draw'.
    """
    if flash_frame_idx >= max_flash_frames:
        return
    alpha = 0.28 * (1.0 - flash_frame_idx / max_flash_frames)
    color_map = {"win": COL_GREEN, "lose": COL_RED, "draw": COL_TEXT_SECONDARY}
    color   = color_map.get(result, COL_TEXT_SECONDARY)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def draw_quality_warnings(frame, hand_state):
    """
    Draw small amber warning chips along the top-left if the hand state
    indicates a problem (hand too far away, or poor lighting).
    Multiple warnings stack horizontally.
    """
    h, w = frame.shape[:2]
    warn_y = _ix(h * 0.09)
    warnings = []
    if hand_state.get("hand_too_far"):
        warnings.append(("MOVE CLOSER", COL_AMBER))
    if hand_state.get("poor_lighting"):
        warnings.append(("POOR LIGHTING", COL_AMBER))

    x = _ix(w * 0.02)
    for text, col in warnings:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        draw_panel(frame, x - 4, warn_y - th - 4, x + tw + 8, warn_y + 6,
                   fill=COL_PANEL_BG, alpha=0.88, border=col, border_thickness=1)
        cv2.putText(frame, text, (x, warn_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
        x += tw + _ix(w * 0.015)  # offset next warning chip to the right

def draw_round_history_dots(frame, rounds, x1, y, x2):
    """
    Draw a row of small coloured dots representing the last (up to 20) rounds.
    Green = win, red = lose, grey = draw.
    Each dot also shows a single letter for the player's gesture (R/P/S).
    """
    if not rounds:
        return
    recent  = rounds[-20:]
    n       = len(recent)
    # Dot radius auto-sizes to fit all dots in the available width
    dot_r   = max(6, (x2 - x1) // (2 * max(n, 1)) - 2)
    dot_r   = min(dot_r, 10)
    step    = (x2 - x1) / max(n, 1)
    col_map = {"win": COL_GREEN, "lose": COL_RED, "draw": COL_TEXT_SECONDARY}
    gest_ch = {"Rock": "R", "Paper": "P", "Scissors": "S"}
    font    = cv2.FONT_HERSHEY_SIMPLEX
    fscale  = max(0.20, dot_r * 0.038)  # letter scale relative to dot size

    for i, r in enumerate(recent):
        outcome = r.get("outcome", r.get("player_outcome", "draw"))
        gesture = r.get("player_gesture", "")
        col     = col_map.get(outcome, COL_TEXT_SECONDARY)
        cx      = int(x1 + step * i + step / 2)
        cv2.circle(frame, (cx, y), dot_r, col, -1)
        cv2.circle(frame, (cx, y), dot_r, COL_BORDER_HAIR, 1)

        # Draw the gesture initial centred inside the dot
        letter = gest_ch.get(gesture, "")
        if letter:
            (lw, lh), _ = cv2.getTextSize(letter, font, fscale, 1)
            cv2.putText(frame, letter, (cx - lw // 2, y + lh // 2),
                        font, fscale, COL_ON_ACTIVE, 1, cv2.LINE_AA)

def draw_help_overlay(frame, screen_name, voice_mode=False):
    """
    Draw a full-screen semi-transparent help overlay listing keyboard shortcuts
    and voice commands. Content is specific to each screen (GAME, MENU, SETTINGS
    etc.) and whether voice mode is active.
    Two columns are used: left column has keys, right column has their effect.
    """
    h, w = frame.shape[:2]
    # Dim the camera feed behind the overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.86, frame, 0.14, 0, frame)

    # Pick the right content block depending on screen and voice mode
    if screen_name == "GAME" and voice_mode:
        section_title = "VOICE COMMANDS"
        left_rows = [
            ("Countdown:",        ""),
            ("READY / STEADY",    "Start countdown"),
            ("ONE / WON / ON",    "Beat 1"),
            ("TWO / TO / TOO",    "Beat 2"),
            ("THREE / TREE / FREE","Beat 3 + shoot"),
            ("", ""),
            ("Throw:",            ""),
            ("ROCK / LOCK / BLOCK",    "Throw Rock"),
            ("PAPER / FAVOR / PIPER",  "Throw Paper"),
            ("SCISSORS / SISTERS",     "Throw Scissors"),
            ("LIZARD / WIZARD",        "Throw Lizard (RPSLS)"),
            ("SPOCK / SPOT / STOCK",   "Throw Spock (RPSLS)"),
        ]
        right_rows = [
            ("Navigation:",       ""),
            ("BACK / CANCEL",     "Return to menu"),
            ("QUIT / EXIT",       "Quit app"),
            ("COMMENTARY",        "Toggle commentary"),
            ("RESTART / AGAIN",   "Restart game"),
            ("START / BEGIN",     "Start session"),
            ("", ""),
            ("Game shortcuts:",   ""),
            ("SNAKE",      "Gesture Snake"),
            ("SQUID",      "Squid Game"),
            ("SIMON",      "Simon Says"),
            ("BLUFF",      "Bluff Mode"),
            ("REFLEX",     "Reflex"),
            ("REHAB",      "Gesture Trainer"),
            ("RACE",       "Prediction Race"),
        ]
    elif screen_name == "GAME":
        section_title = "SHORTCUTS"
        left_rows = [
            ("Make a FIST",    "Start the countdown"),
            ("Pump 4x",        "Advances countdown beats"),
            ("SHOOT window",   "Change fist to your throw"),
            ("", ""),
            ("ESC",       "Return to Menu"),
            ("M",         "Toggle Diagnostic mode"),
            ("N",         "Toggle sound on / off"),
            ("1 / 2 / 3", "Cheat / Fair Play / Challenge"),
            ("?",         "Close this help"),
        ]
        right_rows = [
            ("Diagnostic only:",""),
            ("F",      "Toggle landmark collection"),
            ("7/8/9",  "Record Rock/Scissors/Paper sample"),
            ("T",      "Train front-on gesture model"),
            ("H",      "Hardware test (ESP32 serial)"),
            ("E",      "Toggle face landmark debug"),
            ("", ""),
            ("Gestures:",""),
            ("Fist",   "Rock"),
            ("Open hand",  "Paper"),
            ("2 fingers",  "Scissors"),
        ]
    elif screen_name == "MENU":
        section_title = "SHORTCUTS"
        left_rows = [
            ("UP / W",       "Move selection up"),
            ("DOWN / S",     "Move selection down"),
            ("ENTER",        "Select item"),
            ("ESC",          "Back / Exit sub-menu"),
            ("Q",            "Quit application"),
            ("?",            "Close this help"),
        ]
        right_rows = [
            ("Voice shortcuts:",""),
            ("GAMES / MODES",  "Open Game Modes"),
            ("FAIR / CHEAT",   "Start game directly"),
            ("CHALLENGE",      "Challenge mode"),
            ("SNAKE / SQUID",  "Start game directly"),
            ("SIMON / BLUFF",  "Start game directly"),
            ("REFLEX / REHAB", "Start game directly"),
            ("RACE / RPSLS",   "Start game directly"),
            ("STATS",          "Player Stats"),
            ("TUTORIAL",       "How to Play"),
            ("SETTINGS",       "Settings"),
            ("SIMULATIONS",    "Simulations Lab"),
        ]
    elif screen_name == "SETTINGS":
        section_title = "SETTINGS"
        left_rows = [
            ("UP / DOWN",    "Navigate items"),
            ("LEFT / RIGHT", "Change value"),
            ("ENTER",        "Edit Player Name field"),
            ("ESC",          "Cancel edit / Back to Menu"),
            ("?",            "Close this help"),
        ]
        right_rows = [
            ("Settings guide:",""),
            ("Player Name",    "Your profile name for stats"),
            ("AI Difficulty",  "Easy / Normal / Hard"),
            ("Voice Model",    "US or Indian English"),
            ("Resolution",     "640x480 recommended"),
            ("Hand Orient.",   "Side (default) or Front"),
            ("Shoot Window",   "Time to throw after SHOOT"),
            ("Beat Cooldown",  "Min time between pumps"),
        ]
    elif screen_name == "FEATURES":
        section_title = "FEATURES"
        left_rows = [
            ("UP / DOWN",    "Navigate"),
            ("ENTER",        "Toggle on / off"),
            ("LEFT / RIGHT", "Change Input Mode"),
            ("ESC",          "Back to Menu"),
            ("?",            "Close this help"),
        ]
        right_rows = [
            ("Features guide:",""),
            ("Input Mode",    "Pump or Voice"),
            ("Emotion Track", "Detect facial emotion"),
            ("Gesture Nav",   "Navigate menus by hand"),
            ("Face Debug",    "Show landmark overlay"),
        ]
    elif screen_name == "GAME_NONRPS":
        section_title = "VOICE COMMANDS"
        left_rows = [
            ("Navigation (all modes):",""),
            ("BACK / CANCEL",   "Return to menu"),
            ("QUIT / EXIT",     "Quit application"),
            ("", ""),
            ("Gesture Trainer:", ""),
            ("START / BEGIN",   "Begin the session"),
            ("BACK",            "Exit to menu"),
            ("", ""),
            ("Speed Reflex:",    ""),
            ("RESTART / AGAIN", "Play again after game over"),
            ("BACK",            "Exit to menu"),
        ]
        right_rows = [
            ("Squid Game:",""),
            ("BACK",            "Exit to menu"),
            ("", ""),
            ("Simon Says:",""),
            ("BACK",            "Exit to menu"),
            ("", ""),
            ("Arcade Snake:",""),
            ("Rock=Straight  Scissors=Left",""),
            ("Paper=Right",""),
            ("ESC",             "Exit to menu"),
            ("", ""),
            ("2-Player modes:",""),
            ("Voice not available","(both hands in use)"),
        ]
    else:
        # Fallback for any screen without a specific help definition
        section_title = "SHORTCUTS"
        left_rows  = [("ESC", "Back"), ("?", "Close help")]
        right_rows = []

    # Section title + horizontal rule
    draw_centered_text(frame, section_title,
                       _ix(h * 0.08), 0.60, COL_ACCENT, thickness=2, outline=3)
    cv2.line(frame, (_ix(w * 0.05), _ix(h * 0.13)),
             (w - _ix(w * 0.05), _ix(h * 0.13)), COL_BORDER_HAIR, 1)

    # Column x-positions for keys (lkey_x, rkey_x) and descriptions (lval_x, rval_x)
    row_y  = _ix(h * 0.17)
    row_h  = _ix(h * 0.056)
    lkey_x = _ix(w * 0.05)
    lval_x = _ix(w * 0.28)
    rkey_x = _ix(w * 0.55)
    rval_x = _ix(w * 0.74)

    # Draw both columns in a single loop, stopping at whichever is shorter
    max_rows = max(len(left_rows), len(right_rows))
    for i in range(max_rows):
        y = row_y + i * row_h
        if i < len(left_rows):
            k, v = left_rows[i]
            if k and v:
                # Key in accent, description in primary
                draw_outlined_text(frame, k, lkey_x, y, SCALE_CAPTION,
                                   COL_ACCENT, thickness=1, outline=2)
                draw_outlined_text(frame, v, lval_x, y, SCALE_MICRO,
                                   COL_TEXT_PRIMARY, thickness=1, outline=2)
            elif k:
                # Section header (no value): dim the text
                draw_outlined_text(frame, k, lkey_x, y, SCALE_MICRO,
                                   COL_TEXT_DIM, thickness=1, outline=2)
        if i < len(right_rows):
            k, v = right_rows[i]
            if k and v:
                draw_outlined_text(frame, k, rkey_x, y, SCALE_CAPTION,
                                   COL_ACCENT, thickness=1, outline=2)
                draw_outlined_text(frame, v, rval_x, y, SCALE_MICRO,
                                   COL_TEXT_PRIMARY, thickness=1, outline=2)
            elif k:
                draw_outlined_text(frame, k, rkey_x, y, SCALE_MICRO,
                                   COL_TEXT_DIM, thickness=1, outline=2)

    # Close hint at the very bottom of the overlay
    draw_centered_text(frame, "Press  ?  again to close",
                       h - _ix(h * 0.06), 0.34, COL_TEXT_DIM,
                       thickness=1, outline=2)

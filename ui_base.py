"""
ui_base.py -- Shared colours, layout helpers, and drawing primitives.

This is the foundation that every other ui_*.py file builds on.
Import it with `from ui_base import *` to get the whole toolkit.

What lives here:
  - BGR colour constants (OpenCV uses BGR order, not RGB!)
  - Font and scale constants so text sizes are consistent everywhere
  - Layout helpers that turn frame percentages into pixel coordinates
  - Core draw_ functions: panels, text, bars, icons, etc.
"""

import cv2
import math
import time

# __all__ controls what gets re-exported when another file does `from ui_base import *`
__all__ = [
    # Colour palette
    'COL_PANEL_BG', 'COL_PANEL_ALPHA', 'COL_ROW_SELECTED', 'COL_BORDER_HAIR',
    'COL_ACCENT', 'COL_TEXT_PRIMARY', 'COL_TEXT_SECONDARY', 'COL_TEXT_DIM',
    'COL_AMBER', 'COL_GREEN', 'COL_RED',
    'COL_BEAT_FILL', 'COL_BEAT_RING', 'COL_ON_ACTIVE', 'COL_INACTIVE',
    # Legacy colour names -- sibling modules still use these old names
    'COL_BG_DARK', 'COL_BG_PANEL', 'COL_BG_PANEL_LIGHT',
    'COL_CYAN', 'COL_MAGENTA', 'COL_YELLOW', 'COL_ORANGE',
    'COL_TEXT', 'COL_TEXT_ACCENT',
    # Colourblind-safe result colours
    '_COL_CB_WIN', '_COL_CB_LOSE', '_COL_CB_DRAW',
    # Colour helpers
    '_result_colour', '_get_emotion_color',
    # Layout helpers
    '_ix', '_fit_rect', '_frame_size',
    '_game_layout', '_menu_layout', '_settings_layout',
    # Backward-compat stub
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
    # Spec primitives
    'draw_gesture_glyph', 'draw_gesture_badge', 'draw_beat_track',
    'draw_progress_bar', 'draw_selected_row', 'draw_row',
]

# ============================================================
# COLOUR PALETTE
# All values are BGR tuples because OpenCV uses blue-green-red order.
# So (0, 200, 220) is a warm yellow, NOT teal. Keep that in mind!
# ============================================================

COL_PANEL_BG       = (20, 15, 10)     # very dark navy -- main panel background
COL_PANEL_ALPHA    = 0.78             # default transparency for panels (78% opaque)
COL_ROW_SELECTED   = (45, 38, 30)     # slightly lighter fill for the selected list row
COL_BORDER_HAIR    = (36, 36, 36)     # nearly invisible hairline border
COL_ACCENT         = (210, 160, 60)   # soft blue -- used for highlights and active states
COL_TEXT_PRIMARY   = (255, 255, 255)  # white -- main readable text
COL_TEXT_SECONDARY = (160, 160, 160)  # mid-grey -- supporting text
COL_TEXT_DIM       = (80, 80, 80)     # dark grey -- hints and inactive labels
COL_AMBER          = (40, 160, 220)   # amber -- warnings and borderline states
COL_GREEN          = (80, 180, 80)    # green -- wins and confirmations
COL_RED            = (60, 60, 200)    # red -- losses and urgent prompts

# Legacy aliases so old code doesn't break -- these just point at the new colours
COL_BG_DARK        = COL_PANEL_BG
COL_BG_PANEL       = COL_PANEL_BG
COL_BG_PANEL_LIGHT = (28, 22, 16)
COL_CYAN           = (200, 200,   0)
COL_MAGENTA        = (180,   0, 180)
COL_YELLOW         = (  0, 200, 220)
COL_ORANGE         = (  0, 130, 255)
COL_TEXT           = COL_TEXT_PRIMARY
COL_TEXT_ACCENT    = COL_ACCENT

# Used for the beat track circles and toggle indicators
COL_BEAT_FILL = (24, 24, 24)   # dark fill for an inactive beat circle
COL_BEAT_RING = (56, 56, 56)   # slightly lighter ring outline
COL_ON_ACTIVE = (10, 10, 10)   # dark text that sits on top of a filled/active element
COL_INACTIVE  = (40, 40, 60)   # dim border for unselected indicators

# Colourblind-safe variants for the result flash -- currently same as main colours
_COL_CB_WIN  = COL_GREEN
_COL_CB_LOSE = COL_RED
_COL_CB_DRAW = COL_TEXT_SECONDARY

# ============================================================
# TYPOGRAPHY CONSTANTS
# Using named scale constants means we only have to change one value
# here if we want to resize text globally, rather than hunting through
# every cv2.putText call in every file.
# ============================================================

FONT_PRIMARY  = cv2.FONT_HERSHEY_SIMPLEX   # standard weight font
FONT_DISPLAY  = cv2.FONT_HERSHEY_DUPLEX    # heavier weight for big titles

SCALE_DISPLAY_XL = 1.7   # countdown numbers, big result text
SCALE_DISPLAY_L  = 1.2   # section headings, SHOOT! prompt
SCALE_HEADING    = 0.90  # panel titles
SCALE_BODY       = 0.55  # normal body text
SCALE_CAPTION    = 0.45  # smaller captions
SCALE_MICRO      = 0.40  # tiny labels and hints

# ============================================================
# GEOMETRY CONSTANTS
# Stored as fractions of the frame so the layout scales automatically
# to whatever resolution the camera is running at.
# ============================================================

TOP_BAR_PCT    = 0.06    # top HUD bar is 6% of frame height
BOTTOM_BAR_PCT = 0.06    # bottom HUD bar is the same
PANEL_INSET_X  = 0.036   # 3.6% side margin for full-width panels
PANEL_INSET_Y  = 0.105   # top margin: starts just below the top bar

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def _frame_size(frame):
    """Return (width, height) from an OpenCV frame.
    frame.shape gives (height, width, channels), so we swap them here."""
    h, w = frame.shape[:2]
    return w, h

def _ix(value):
    """Round a float pixel coordinate to the nearest int.
    We do all our maths in floats and only convert at the last moment
    so sub-pixel errors don't accumulate."""
    return int(round(value))

def _fit_rect(x1, y1, x2, y2):
    """Convert a float bounding box into a tuple of four ints.
    Just a shorthand for calling _ix() on all four corners at once."""
    return (_ix(x1), _ix(y1), _ix(x2), _ix(y2))

def _game_layout(frame):
    """
    Compute all named pixel regions for the main game screen.

    Returns a dict where every value is already rounded to int.
    Keys like 'hero' and 'beat_track' are (x1, y1, x2, y2) rects;
    keys like 'w' and 'top_bar_h' are single numbers.
    """
    w, h = _frame_size(frame)
    top_bar_h    = _ix(h * TOP_BAR_PCT)
    bottom_bar_h = _ix(h * BOTTOM_BAR_PCT)

    return {
        "w": w, "h": h,
        "top_bar_h":      top_bar_h,
        "top_row_h":      top_bar_h,     # legacy key -- same as top_bar_h
        "second_row_h":   0,             # was used for a second header row, now removed
        "header_total_h": top_bar_h,     # legacy key
        "bottom_bar_h":   bottom_bar_h,
        "arcade_title_y": _ix(h * 0.17),
        "arcade_lights_y":_ix(h * 0.20),
        # Named zones as (x1, y1, x2, y2) pixel rects
        "status_strip": _fit_rect(w * 0.08, h * 0.18, w * 0.92, h * 0.24),
        "hero":         _fit_rect(w * 0.08, h * 0.26, w * 0.92, h * 0.68),
        "beat_track":   _fit_rect(w * 0.15, h * 0.76, w * 0.85, h * 0.90),
        "result":       _fit_rect(w * 0.08, h * 0.26, w * 0.92, h * 0.90),
        "gesture_row":  _fit_rect(w * 0.08, h * 0.09, w * 0.92, h * 0.17),
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
    The panel is shorter than a menu panel to leave room for a
    description box below the item list."""
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
    """Map a result string (e.g. 'YOU WIN', 'DRAW') to its display colour.
    The colourblind flag is accepted for API compatibility but not used yet."""
    r = result_str.upper()
    if "WIN" in r or "SURVIVE" in r:
        return COL_GREEN
    if "DRAW" in r or "AGAIN" in r:
        return COL_TEXT_SECONDARY
    # Anything else (loss, error, etc.) is red
    return COL_RED

def _get_emotion_color(emotion):
    """Return the colour that best represents a detected facial emotion."""
    if emotion == "Happy":      return COL_GREEN
    if emotion == "Surprised":  return COL_AMBER
    if emotion == "Frustrated": return COL_RED
    # Unknown or Neutral emotions use the dim grey
    return COL_TEXT_DIM

def get_gesture_color(gesture):
    """White for a valid RPS(LS) gesture, dim grey for anything unrecognised."""
    if gesture in ("Rock", "Paper", "Scissors", "Lizard", "Spock"):
        return COL_TEXT_PRIMARY
    return COL_TEXT_DIM

# ============================================================
# CORE DRAWING PRIMITIVES
# ============================================================

def draw_panel(frame, x1, y1, x2, y2, fill=None, alpha=None,
               border=None, border_thickness=1):
    """
    Draw a semi-transparent filled rectangle -- the frosted-glass panel look.

    How it works: we copy the camera pixels inside the rect (the ROI),
    draw a solid rectangle over the copy, then blend the copy back using
    addWeighted. This lets the camera image bleed through underneath.
    After blending, we draw an opaque border on top of the frame.

    Coordinates are clamped to the frame edge so callers don't need to guard.
    """
    # Apply defaults if the caller didn't specify
    if fill   is None: fill   = COL_PANEL_BG
    if alpha  is None: alpha  = COL_PANEL_ALPHA
    if border is None: border = COL_BORDER_HAIR

    h, w = frame.shape[:2]

    # Clamp to valid frame bounds to avoid index errors on edge panels
    x1i = max(0, int(x1));   y1i = max(0, int(y1))
    x2i = min(w - 1, int(x2)); y2i = min(h - 1, int(y2))

    # Skip if the rect has zero or negative area after clamping
    if x2i <= x1i or y2i <= y1i:
        return

    # Grab the pixel region we're about to overlay
    roi     = frame[y1i:y2i, x1i:x2i]
    overlay = roi.copy()

    # Fill the overlay copy with the panel colour, then blend it back in
    cv2.rectangle(overlay, (0, 0), (x2i - x1i, y2i - y1i), fill, -1)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)

    # Draw the border on the original frame (fully opaque, no blending)
    if border_thickness > 0:
        cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), border, border_thickness)

def _draw_glow_border(frame, x1, y1, x2, y2, color, thickness=1):
    """Backward-compat stub -- used to draw a multi-pass glow effect.
    Now just draws a single solid border (glow was dropped for performance)."""
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

def draw_outlined_text(frame, text, x, y, scale, color, thickness=1, outline=2):
    """
    Draw text with a black drop-shadow outline, then the coloured text on top.

    The black pass uses a thicker stroke, so it bleeds around the letters.
    This makes the text readable against any background colour, which matters
    a lot when text is drawn over the live camera feed.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    # First pass: thick black outline
    cv2.putText(frame, text, (int(x), int(y)), font, scale,
                (0, 0, 0), thickness + outline, cv2.LINE_AA)

    # Second pass: thinner coloured text on top
    cv2.putText(frame, text, (int(x), int(y)), font, scale,
                color, thickness, cv2.LINE_AA)

def get_fit_scale(text, max_width, base_scale=1.0, thickness=2, min_scale=0.35):
    """
    Find the largest font scale that keeps text within max_width pixels.

    Binary search between min_scale and base_scale, running 8 iterations.
    Eight iterations gives sub-1% precision, which is more than enough for UI text.
    Returns base_scale immediately if the text already fits at that size.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Fast path: if it already fits at the requested size, nothing to do
    if cv2.getTextSize(text, font, base_scale, thickness)[0][0] <= max_width:
        return base_scale

    # Binary search: narrow in on the biggest scale that still fits
    lo, hi = min_scale, base_scale
    for _ in range(8):
        mid = (lo + hi) / 2
        if cv2.getTextSize(text, font, mid, thickness)[0][0] <= max_width:
            lo = mid   # fits -- try bigger
        else:
            hi = mid   # too wide -- try smaller
    return lo

def draw_centered_text(frame, text, center_y, scale, color, thickness=2, outline=4):
    """Draw text horizontally centred on the frame at the given y pixel position."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, _), _ = cv2.getTextSize(text, font, scale, thickness)
    # Push x left by half the text width from the frame centre
    x = (frame.shape[1] - text_w) // 2
    draw_outlined_text(frame, text, x, center_y, scale, color, thickness, outline)

def draw_centered_text_in_rect(frame, text, rect, base_scale, color,
                                thickness=2, outline=4):
    """
    Draw text centred both horizontally and vertically inside a pixel rect.
    Shrinks the font automatically if the text would overflow the rect width.
    """
    x1, y1, x2, y2 = rect
    max_width = max(40, (x2 - x1) - 20)  # leave a small side margin

    # Shrink to fit if necessary
    scale = get_fit_scale(text, max_width, base_scale=base_scale, thickness=thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)

    # Centre horizontally: start x so that the text lands in the middle
    x = x1 + ((x2 - x1) - text_w) // 2
    # Centre vertically: cv2 y is the baseline, so add half text height
    y = y1 + ((y2 - y1) + text_h) // 2

    draw_outlined_text(frame, text, x, y, scale, color, thickness, outline)

# ============================================================
# GESTURE GLYPHS
# Each gesture maps to a simple geometric shape so they're still
# recognisable at small sizes or in peripheral vision.
# ============================================================

def draw_gesture_glyph(frame, gesture, rect, color=None):
    """
    Draw a flat, outline-only gesture icon inside the given rect.

    Shape mapping:
      Rock     -> circle
      Paper    -> square
      Scissors -> X
      Lizard   -> wide ellipse
      Spock    -> three vertical lines with a crossbar
      Unknown  -> small dim square placeholder
    """
    x1, y1, x2, y2 = rect
    cx    = (x1 + x2) // 2
    cy    = (y1 + y2) // 2
    side  = min(x2 - x1, y2 - y1)
    col   = color or COL_TEXT_PRIMARY

    # Line thickness scales with the icon so it looks proportional at any size
    thick = max(2, _ix(side * 0.04))

    if gesture == "Rock":
        # Circle represents a closed fist
        r = _ix(side * 0.40)
        cv2.circle(frame, (cx, cy), r, col, thick)

    elif gesture == "Paper":
        # Square represents an open flat hand
        half = _ix(side * 0.38)
        cv2.rectangle(frame, (cx - half, cy - half), (cx + half, cy + half), col, thick)

    elif gesture == "Scissors":
        # Two diagonal lines crossing form an X (the scissor blades)
        d = _ix(side * 0.38)
        cv2.line(frame, (cx - d, cy - d), (cx + d, cy + d), col, thick)
        cv2.line(frame, (cx + d, cy - d), (cx - d, cy + d), col, thick)

    elif gesture == "Lizard":
        # Wide horizontal ellipse suggests a flat lizard head
        axes = (_ix(side * 0.42), _ix(side * 0.22))
        cv2.ellipse(frame, (cx, cy), axes, 0, 0, 360, col, thick)

    elif gesture == "Spock":
        # Three vertical lines with a horizontal crossbar -- Vulcan salute silhouette
        gap = _ix(side * 0.14)
        top = cy - _ix(side * 0.36)
        bot = cy + _ix(side * 0.36)
        bar = cy - _ix(side * 0.08)
        for dx in (-gap, 0, gap):
            cv2.line(frame, (cx + dx, top), (cx + dx, bot), col, thick)
        cv2.line(frame, (cx - gap, bar), (cx + gap, bar), col, thick)

    else:
        # Unknown gesture: draw a tiny dim square as a placeholder
        half = _ix(side * 0.28)
        cv2.rectangle(frame, (cx - half, cy - half), (cx + half, cy + half),
                      COL_TEXT_DIM, 1)

def draw_gesture_icon(frame, gesture, cx, cy, size):
    """Backward-compatible wrapper -- draw a gesture glyph centred at (cx, cy)."""
    rect = (cx - size, cy - size, cx + size, cy + size)
    draw_gesture_glyph(frame, gesture, rect, color=COL_TEXT_PRIMARY)

def draw_gesture_badge(frame, gesture, confidence, x, y,
                       threshold=0.70, show_confidence=True):
    """
    Draw a small info badge showing the detected gesture and a confidence dot.

    The border colour signals confidence level:
      - Accent (blue)  -> at or above threshold (trustworthy reading)
      - Dim (hairline) -> below threshold (don't rely on this yet)

    The dot inside the badge changes colour too:
      - Green  -> confirmed
      - Amber  -> borderline (50-70%)
      - Grey   -> too uncertain
    """
    h, fw = frame.shape[:2]
    pad_x, pad_y = 14, 8
    font   = cv2.FONT_HERSHEY_SIMPLEX
    g_text = gesture.upper() if gesture else "NONE"

    # Measure both lines so we can size the box to fit them
    (gw, gh), _ = cv2.getTextSize(g_text, font, SCALE_BODY, 1)
    det_text = "DETECTED"
    (dw, dh), _ = cv2.getTextSize(det_text, font, SCALE_MICRO, 1)

    # Box needs to fit the wider of the two text lines, plus the dot
    dot_r   = 4
    inner_w = dot_r * 2 + 6 + max(gw, dw)
    box_w   = inner_w + pad_x * 2
    box_h   = gh + dh + pad_y * 2 + 6

    x2 = x + box_w
    y2 = y + box_h

    # Choose border colour based on how confident the reading is
    confident  = confidence >= threshold
    borderline = 0.50 <= confidence < threshold
    border_col = COL_ACCENT if confident else COL_BORDER_HAIR

    draw_panel(frame, x, y, x2, y2,
               fill=COL_PANEL_BG, alpha=0.88, border=border_col, border_thickness=1)

    # Status dot: its colour mirrors the border/confidence level
    dot_col = COL_GREEN if confident else (COL_AMBER if borderline else COL_TEXT_DIM)
    dot_cx  = x + pad_x + dot_r
    dot_cy  = y + box_h // 2
    cv2.circle(frame, (dot_cx, dot_cy), dot_r, dot_col, -1)

    # "DETECTED" label above the gesture name
    text_x = dot_cx + dot_r + 6
    cv2.putText(frame, det_text,
                (text_x, y + pad_y + dh),
                font, SCALE_MICRO, COL_TEXT_DIM, 1, cv2.LINE_AA)

    # Gesture name, optionally followed by the raw confidence value
    conf_suffix = f"  {confidence:.2f}" if show_confidence and confidence > 0 else ""
    label = g_text + conf_suffix
    draw_outlined_text(frame, label, text_x, y + pad_y + dh + 6 + gh,
                       SCALE_BODY, COL_TEXT_PRIMARY, thickness=1, outline=2)

# ============================================================
# BEAT TRACK
# ============================================================

def draw_beat_track(frame, beat_count, num_beats=4, state="", x1=None, y1=None,
                    x2=None, y2=None):
    """
    Draw a horizontal row of numbered circles representing the beat countdown.

    Circles fill in left-to-right as the player pumps their fist.
    The last circle turns red during SHOOT_WINDOW to warn the player to throw.
    Falls back to the layout zone from _game_layout() if no rect is given.
    """
    w, h = _frame_size(frame)

    # Use the layout-computed zone if no explicit rect was provided
    if x1 is None:
        bx1, by1, bx2, by2 = _game_layout(frame)["beat_track"]
    else:
        bx1, by1, bx2, by2 = x1, y1, x2, y2

    # Dark frosted background panel for the beat track strip
    draw_panel(frame, bx1, by1, bx2, by2,
               fill=COL_PANEL_BG, alpha=0.78, border=COL_BORDER_HAIR, border_thickness=1)

    ph = by2 - by1  # panel height
    pw = bx2 - bx1  # panel width

    # "BEAT TRACK" caption centred near the top of the strip
    cap_y = by1 + _ix(ph * 0.28)
    (cap_lw, _), _ = cv2.getTextSize("BEAT TRACK", FONT_PRIMARY, SCALE_MICRO, 1)
    cv2.putText(frame, "BEAT TRACK",
                (bx1 + (pw - cap_lw) // 2, cap_y),
                FONT_PRIMARY, SCALE_MICRO, COL_TEXT_SECONDARY, 1, cv2.LINE_AA)

    # Size circles relative to frame height, capped at 38px diameter so they
    # don't overflow the strip on very tall frames
    diameter = min(_ix(h * 0.052), 38)
    radius   = diameter // 2
    gap      = _ix(h * 0.037)
    total_w  = num_beats * diameter + (num_beats - 1) * gap
    start_x  = bx1 + (pw - total_w) // 2 + radius  # first circle centre x
    cy       = by1 + _ix(ph * 0.62)                 # all circles share this y

    is_shoot = state == "SHOOT_WINDOW"

    # Draw each beat circle
    for i in range(num_beats):
        cx         = start_x + i * (diameter + gap)
        act        = i < beat_count                         # has this beat been counted?
        shoot_beat = is_shoot and (i == num_beats - 1)      # last circle during shoot window

        # Pick the fill/ring colour based on state
        if shoot_beat:
            col = COL_RED     # urgency cue: throw now
        elif act:
            col = COL_ACCENT  # counted beat: accent fill
        else:
            col = COL_BEAT_RING  # future beat: dim ring

        if act or shoot_beat:
            cv2.circle(frame, (cx, cy), radius, col, -1)  # filled circle
            num_col = COL_ON_ACTIVE                        # dark number on the fill
        else:
            cv2.circle(frame, (cx, cy), radius, col, 2)   # ring only
            num_col = COL_TEXT_DIM                         # dim number in the ring

        # Beat number (1-4) centred inside the circle
        label = str(i + 1)
        (lw, lh), _ = cv2.getTextSize(label, FONT_PRIMARY, SCALE_MICRO, 1)
        cv2.putText(frame, label,
                    (cx - lw // 2, cy + lh // 2),
                    FONT_PRIMARY, SCALE_MICRO, num_col, 1, cv2.LINE_AA)

    # Hint text below the circles tells the player what to do next
    hint = "4th beat opens SHOOT" if not is_shoot else "MAKE YOUR THROW"
    hint_y = by1 + _ix(ph * 0.90)
    (hw, _), _ = cv2.getTextSize(hint, FONT_PRIMARY, SCALE_MICRO, 1)
    cv2.putText(frame, hint,
                (bx1 + (pw - hw) // 2, hint_y),
                FONT_PRIMARY, SCALE_MICRO, COL_TEXT_DIM, 1, cv2.LINE_AA)

# ============================================================
# PROGRESS BAR
# ============================================================

def draw_progress_bar(frame, x1, y1, x2, y2, value, color=None, track_height=4):
    """
    Draw a simple horizontal progress bar inside the given rect.
    value is 0.0 to 1.0 (clamped internally).
    The bar is vertically centred within the rect.
    """
    col = color or COL_ACCENT

    # Clamp track height to the available rect height (minimum 3px)
    bh  = max(3, min(track_height, y2 - y1))
    mid = y1 + (y2 - y1 - bh) // 2  # top edge of the bar, centred vertically

    # Dark background track (shows the unfilled portion)
    cv2.rectangle(frame, (x1, mid), (x2, mid + bh), (28, 28, 28), -1)

    # Filled portion: how far along from x1 based on value
    fill_x = x1 + _ix((x2 - x1) * max(0.0, min(1.0, value)))
    if fill_x > x1:
        cv2.rectangle(frame, (x1, mid), (fill_x, mid + bh), col, -1)

# ============================================================
# LIST ROW PRIMITIVES
# ============================================================

def draw_selected_row(frame, x1, y1, x2, y2, accent_bar=True):
    """
    Highlight a list row as 'selected' with a translucent fill.
    Optionally adds a 2px accent-coloured bar on the left edge --
    a visual indicator that this is the active item.
    """
    roi     = frame[y1:y2, x1:x2]
    overlay = roi.copy()
    # Blend a slightly lighter colour over the camera/background pixels
    cv2.rectangle(overlay, (0, 0), (x2 - x1, y2 - y1), COL_ROW_SELECTED, -1)
    cv2.addWeighted(overlay, 0.95, roi, 0.05, 0, roi)

    # Left-edge accent bar so the selection is obvious at a glance
    if accent_bar:
        cv2.rectangle(frame, (x1, y1), (x1 + 2, y2), COL_ACCENT, -1)

def draw_row(frame, x1, y1, x2, y2, label, selected=False,
             sub_label='', right_hint=''):
    """
    Draw a single list row.

    When selected:
      - Row gets the highlight fill and accent left bar
      - sub_label appears below the main label in dim text
      - right_hint appears right-aligned (e.g. 'ENTER FIGHT' shortcut hint)

    When not selected, only the main label is drawn in grey.
    """
    row_h = y2 - y1

    if selected:
        # Inset the fill by 2px on each side so it doesn't overlap the panel border
        roi     = frame[y1:y2, x1+2:x2-2]
        overlay = roi.copy()
        cv2.rectangle(overlay, (0, 0), (x2 - x1 - 4, row_h), COL_ROW_SELECTED, -1)
        cv2.addWeighted(overlay, 0.95, roi, 0.05, 0, roi)
        # Accent bar on the left edge
        cv2.rectangle(frame, (x1, y1), (x1 + 2, y2), COL_ACCENT, -1)

    # Selected rows use white text; unselected use grey
    text_color = COL_TEXT_PRIMARY if selected else COL_TEXT_SECONDARY
    pad  = _ix((x2 - x1) * 0.025)
    # Shift text 4px right on selected rows to clear the accent bar
    text_x = x1 + pad + (4 if selected else 0)

    # If there's a sub_label, shift the main label up to make room below it
    label_y = y1 + _ix(row_h * (0.48 if not sub_label else 0.38))
    draw_outlined_text(frame, label, text_x, label_y, SCALE_BODY, text_color,
                       thickness=1, outline=2)

    # Sub-label and right hint are only shown when this row is selected
    if sub_label and selected:
        draw_outlined_text(frame, sub_label, text_x, y1 + _ix(row_h * 0.72),
                           SCALE_CAPTION, COL_TEXT_DIM, thickness=1, outline=2)
    if right_hint and selected:
        (tw, _), _ = cv2.getTextSize(right_hint, FONT_PRIMARY, SCALE_MICRO, 1)
        draw_outlined_text(frame, right_hint, x2 - pad - tw, label_y,
                           SCALE_MICRO, COL_ACCENT, thickness=1, outline=2)

# ============================================================
# HUD BARS (top and bottom strips)
# ============================================================

def draw_top_bar(frame, left_label, right_hints=''):
    """
    Draw the top HUD bar: spans the full width, 6% of frame height.

    left_label  -- mode name or screen title, shown in accent colour
    right_hints -- keyboard shortcut list, shown in tiny dim text on the right

    A hairline separator line runs along the bottom edge of the bar.
    """
    w, h  = _frame_size(frame)
    bar_h = _ix(h * 0.06)

    # Semi-transparent dark fill over the camera feed (frosted bar effect)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), COL_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # Hairline separator at the bottom edge of the bar
    cv2.line(frame, (0, bar_h - 1), (w, bar_h - 1), COL_BORDER_HAIR, 1)

    text_y = _ix(bar_h * 0.68)  # baseline position, vertically centred in the bar

    # Left: mode/screen name in the accent colour
    draw_outlined_text(frame, left_label, _ix(w * 0.02), text_y,
                       SCALE_BODY, COL_ACCENT, thickness=2, outline=2)

    # Right: keyboard hints in tiny dim text, right-aligned
    if right_hints:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, _), _ = cv2.getTextSize(right_hints, font, SCALE_MICRO, 1)
        rx = w - tw - _ix(w * 0.02)
        draw_outlined_text(frame, right_hints, rx, text_y,
                           SCALE_MICRO, COL_TEXT_DIM, thickness=1, outline=2)

def draw_bottom_bar(frame, hints=''):
    """
    Draw the bottom HUD bar: spans the full width, 6% of frame height, pinned to the bottom.

    hints -- pipe-separated list of shortcuts (e.g. 'ESC Back | ? Help')
    A hairline separator runs along the top edge of the bar.
    """
    w, h  = _frame_size(frame)
    bar_h = _ix(h * 0.06)
    y1    = h - bar_h  # top of the bar, counted from the bottom

    # Semi-transparent dark fill
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y1), (w, h), COL_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # Hairline separator along the top edge of the bar
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
    Draw a small pill/chip badge centred horizontally on the frame.
    The chip auto-sizes to the text and shrinks the font if the text is wide.
    Used for things like 'AI' labels and mode indicators.
    """
    w, h = _frame_size(frame)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Shrink scale if the text would be too wide (more than ~half the frame)
    scale = get_fit_scale(text, _ix(w * 0.52), base_scale=SCALE_BODY, thickness=1)

    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, 1)
    pad_x = _ix(w * 0.018)
    pad_y = _ix(h * 0.012)

    # Centre the chip horizontally
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
    Map a result banner string to the colour it should be displayed in.
    Checked in priority order: win > draw > game-over > neutral.
    """
    if banner.startswith("YOU WIN") or banner.startswith("YOU SURVIVE"):
        return COL_GREEN
    if banner.startswith("DRAW"):
        return COL_TEXT_PRIMARY
    if banner.startswith("GAME OVER") or "TAKES" in banner or "WINS" in banner:
        return COL_RED
    return COL_TEXT_PRIMARY

# ============================================================
# GESTURE LOCK BAR
# ============================================================

def draw_gesture_confidence_bar(frame, stable_streak, confirm_frames, x, y, width):
    """
    Draw a vertical 'lock' bar on the left edge of the frame.

    The bar fills upward as stable_streak approaches confirm_frames.
    Green when fully locked; accent-coloured while still building.
    The letters L-O-C-K are stacked vertically above the bar.
    """
    fh, fw = frame.shape[:2]
    pct    = min(1.0, stable_streak / max(confirm_frames, 1))  # 0.0 to 1.0
    bar_w  = max(8, fw // 60)
    bar_h  = _ix(fh * 0.22)
    bar_x  = _ix(fw * 0.012)
    bar_bot = fh - _ix(fh * TOP_BAR_PCT) - 2
    bar_top = bar_bot - bar_h

    # Dark background track so the fill is visible against any frame content
    cv2.rectangle(frame, (bar_x, bar_top), (bar_x + bar_w, bar_bot),
                  COL_BEAT_FILL, -1)
    cv2.rectangle(frame, (bar_x, bar_top), (bar_x + bar_w, bar_bot),
                  COL_BORDER_HAIR, 1)

    # Filled portion grows upward from the bottom of the bar
    if pct > 0:
        fill_h   = _ix(bar_h * pct)
        fill_top = bar_bot - fill_h
        col = COL_GREEN if pct >= 1.0 else COL_ACCENT
        cv2.rectangle(frame, (bar_x, fill_top), (bar_x + bar_w, bar_bot), col, -1)

    # 'LOCK' stacked vertically above the bar, one character per row
    label = "LOCK"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    lh    = _ix(fh * 0.026)  # vertical step between characters
    for i, ch in enumerate(label):
        # Characters are ordered top-to-bottom, so reverse index for bottom-to-top stacking
        cy = bar_top - _ix(fh * 0.004) - (len(label) - 1 - i) * lh
        (cw, _), _ = cv2.getTextSize(ch, font, 0.26, 1)
        cx = bar_x + bar_w // 2 - cw // 2  # horizontally centred over the bar
        cv2.putText(frame, ch, (cx, cy), font, 0.26, COL_TEXT_DIM, 1, cv2.LINE_AA)

# ============================================================
# RESULT FLASH OVERLAY
# ============================================================

def draw_result_flash(frame, result, flash_frame_idx,
                      max_flash_frames=4, colourblind=False):
    """
    Full-screen tinted flash that plays over a few frames after a round ends.
    The alpha fades out linearly so the flash doesn't linger too long.
    result should be 'win', 'lose', or 'draw'.
    """
    # Skip if the flash has already run its course
    if flash_frame_idx >= max_flash_frames:
        return

    # Alpha starts at 0.28 and drops to 0 by the final frame
    alpha     = 0.28 * (1.0 - flash_frame_idx / max_flash_frames)
    color_map = {"win": COL_GREEN, "lose": COL_RED, "draw": COL_TEXT_SECONDARY}
    color     = color_map.get(result, COL_TEXT_SECONDARY)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

# ============================================================
# QUALITY WARNINGS
# ============================================================

def draw_quality_warnings(frame, hand_state):
    """
    Show small amber warning chips near the top-left if hand_state
    indicates a problem (hand too far away, or poor lighting).
    Multiple warnings stack horizontally from left to right.
    """
    h, w = frame.shape[:2]
    warn_y   = _ix(h * 0.09)
    warnings = []

    # Collect any active warnings from the hand state dict
    if hand_state.get("hand_too_far"):
        warnings.append(("MOVE CLOSER", COL_AMBER))
    if hand_state.get("poor_lighting"):
        warnings.append(("POOR LIGHTING", COL_AMBER))

    x = _ix(w * 0.02)  # starting x; each chip pushes the next one further right

    for text, col in warnings:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)

        # Draw a small panel behind the text so it's readable over the camera
        draw_panel(frame, x - 4, warn_y - th - 4, x + tw + 8, warn_y + 6,
                   fill=COL_PANEL_BG, alpha=0.88, border=col, border_thickness=1)

        cv2.putText(frame, text, (x, warn_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

        # Advance x so the next chip starts after this one
        x += tw + _ix(w * 0.015)

# ============================================================
# ROUND HISTORY DOTS
# ============================================================

def draw_round_history_dots(frame, rounds, x1, y, x2):
    """
    Draw a row of coloured dots showing the outcome of the last 20 rounds.
    Green = win, red = lose, grey = draw.
    Each dot has a single gesture initial (R/P/S) centred inside it.
    """
    if not rounds:
        return

    recent = rounds[-20:]  # only show the most recent 20
    n      = len(recent)

    # Dot radius auto-sizes so all dots fit in the available width
    dot_r  = max(6, (x2 - x1) // (2 * max(n, 1)) - 2)
    dot_r  = min(dot_r, 10)  # cap at 10px so dots don't get too big
    step   = (x2 - x1) / max(n, 1)

    col_map = {"win": COL_GREEN, "lose": COL_RED, "draw": COL_TEXT_SECONDARY}
    gest_ch = {"Rock": "R", "Paper": "P", "Scissors": "S"}
    font    = cv2.FONT_HERSHEY_SIMPLEX
    fscale  = max(0.20, dot_r * 0.038)  # letter scale relative to dot size

    for i, r in enumerate(recent):
        # Support both 'outcome' and legacy 'player_outcome' keys
        outcome = r.get("outcome", r.get("player_outcome", "draw"))
        gesture = r.get("player_gesture", "")
        col     = col_map.get(outcome, COL_TEXT_SECONDARY)
        cx      = int(x1 + step * i + step / 2)  # centre x for this dot

        # Filled dot with a hairline border
        cv2.circle(frame, (cx, y), dot_r, col, -1)
        cv2.circle(frame, (cx, y), dot_r, COL_BORDER_HAIR, 1)

        # Gesture initial centred inside the dot
        letter = gest_ch.get(gesture, "")
        if letter:
            (lw, lh), _ = cv2.getTextSize(letter, font, fscale, 1)
            cv2.putText(frame, letter,
                        (cx - lw // 2, y + lh // 2),
                        font, fscale, COL_ON_ACTIVE, 1, cv2.LINE_AA)

# ============================================================
# HELP OVERLAY
# ============================================================

def draw_help_overlay(frame, screen_name, voice_mode=False):
    """
    Draw a full-screen semi-transparent overlay listing keyboard shortcuts
    and (in voice mode) voice commands.

    The content is screen-specific: GAME, MENU, SETTINGS, FEATURES, etc.
    Two columns are used: left column shows the key/phrase, right shows the effect.
    """
    h, w = frame.shape[:2]

    # Dim the camera feed behind the overlay to ~14% brightness
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.86, frame, 0.14, 0, frame)

    # Pick content based on which screen we're on and whether voice is active
    if screen_name == "GAME" and voice_mode:
        section_title = "VOICE COMMANDS"
        left_rows = [
            ("Countdown:",         ""),
            ("READY / STEADY",     "Start countdown"),
            ("ONE / WON / ON",     "Beat 1"),
            ("TWO / TO / TOO",     "Beat 2"),
            ("THREE / TREE / FREE","Beat 3 + shoot"),
            ("", ""),
            ("Throw:",             ""),
            ("ROCK / LOCK / BLOCK",   "Throw Rock"),
            ("PAPER / FAVOR / PIPER", "Throw Paper"),
            ("SCISSORS / SISTERS",    "Throw Scissors"),
            ("LIZARD / WIZARD",       "Throw Lizard (RPSLS)"),
            ("SPOCK / SPOT / STOCK",  "Throw Spock (RPSLS)"),
        ]
        right_rows = [
            ("Navigation:",      ""),
            ("BACK / CANCEL",    "Return to menu"),
            ("QUIT / EXIT",      "Quit app"),
            ("COMMENTARY",       "Toggle commentary"),
            ("RESTART / AGAIN",  "Restart game"),
            ("START / BEGIN",    "Start session"),
            ("", ""),
            ("Game shortcuts:",  ""),
            ("SNAKE",   "Gesture Snake"),
            ("SQUID",   "Squid Game"),
            ("SIMON",   "Simon Says"),
            ("BLUFF",   "Bluff Mode"),
            ("REFLEX",  "Reflex"),
            ("REHAB",   "Gesture Trainer"),
            ("RACE",    "Prediction Race"),
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
            ("Diagnostic only:", ""),
            ("F",      "Toggle landmark collection"),
            ("7/8/9",  "Record Rock/Scissors/Paper sample"),
            ("T",      "Train front-on gesture model"),
            ("H",      "Hardware test (ESP32 serial)"),
            ("E",      "Toggle face landmark debug"),
            ("", ""),
            ("Gestures:", ""),
            ("Fist",      "Rock"),
            ("Open hand", "Paper"),
            ("2 fingers", "Scissors"),
        ]

    elif screen_name == "MENU":
        section_title = "SHORTCUTS"
        left_rows = [
            ("UP / W",   "Move selection up"),
            ("DOWN / S", "Move selection down"),
            ("ENTER",    "Select item"),
            ("ESC",      "Back / Exit sub-menu"),
            ("Q",        "Quit application"),
            ("?",        "Close this help"),
        ]
        right_rows = [
            ("Voice shortcuts:", ""),
            ("GAMES / MODES",   "Open Game Modes"),
            ("FAIR / CHEAT",    "Start game directly"),
            ("CHALLENGE",       "Challenge mode"),
            ("SNAKE / SQUID",   "Start game directly"),
            ("SIMON / BLUFF",   "Start game directly"),
            ("REFLEX / REHAB",  "Start game directly"),
            ("RACE / RPSLS",    "Start game directly"),
            ("STATS",           "Player Stats"),
            ("TUTORIAL",        "How to Play"),
            ("SETTINGS",        "Settings"),
            ("SIMULATIONS",     "Simulations Lab"),
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
            ("Settings guide:", ""),
            ("Player Name",  "Your profile name for stats"),
            ("AI Difficulty","Easy / Normal / Hard"),
            ("Voice Model",  "US or Indian English"),
            ("Resolution",   "640x480 recommended"),
            ("Hand Orient.", "Side (default) or Front"),
            ("Shoot Window", "Time to throw after SHOOT"),
            ("Beat Cooldown","Min time between pumps"),
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
            ("Features guide:", ""),
            ("Input Mode",   "Pump or Voice"),
            ("Emotion Track","Detect facial emotion"),
            ("Gesture Nav",  "Navigate menus by hand"),
            ("Face Debug",   "Show landmark overlay"),
        ]

    elif screen_name == "GAME_NONRPS":
        section_title = "VOICE COMMANDS"
        left_rows = [
            ("Navigation (all modes):", ""),
            ("BACK / CANCEL",  "Return to menu"),
            ("QUIT / EXIT",    "Quit application"),
            ("", ""),
            ("Gesture Trainer:", ""),
            ("START / BEGIN",  "Begin the session"),
            ("BACK",           "Exit to menu"),
            ("", ""),
            ("Speed Reflex:",   ""),
            ("RESTART / AGAIN","Play again after game over"),
            ("BACK",           "Exit to menu"),
        ]
        right_rows = [
            ("Squid Game:", ""),
            ("BACK",        "Exit to menu"),
            ("", ""),
            ("Simon Says:", ""),
            ("BACK",        "Exit to menu"),
            ("", ""),
            ("Arcade Snake:", ""),
            ("Rock=Straight  Scissors=Left", ""),
            ("Paper=Right",  ""),
            ("ESC",          "Exit to menu"),
            ("", ""),
            ("2-Player modes:", ""),
            ("Voice not available", "(both hands in use)"),
        ]

    else:
        # Fallback for any screen that doesn't have a specific content block
        section_title = "SHORTCUTS"
        left_rows  = [("ESC", "Back"), ("?", "Close help")]
        right_rows = []

    # Section title centred at the top, with a horizontal rule below it
    draw_centered_text(frame, section_title,
                       _ix(h * 0.08), 0.60, COL_ACCENT, thickness=2, outline=3)
    cv2.line(frame,
             (_ix(w * 0.05), _ix(h * 0.13)),
             (w - _ix(w * 0.05), _ix(h * 0.13)),
             COL_BORDER_HAIR, 1)

    # Column x-positions for the two-column layout
    row_y  = _ix(h * 0.17)   # y to start drawing rows
    row_h  = _ix(h * 0.056)  # vertical spacing between rows
    lkey_x = _ix(w * 0.05)   # left column: key/phrase
    lval_x = _ix(w * 0.28)   # left column: description
    rkey_x = _ix(w * 0.55)   # right column: key/phrase
    rval_x = _ix(w * 0.74)   # right column: description

    # Draw both columns together, stopping at the end of whichever is longer
    max_rows = max(len(left_rows), len(right_rows))
    for i in range(max_rows):
        y = row_y + i * row_h

        # Left column row
        if i < len(left_rows):
            k, v = left_rows[i]
            if k and v:
                # Key in accent colour, description in white
                draw_outlined_text(frame, k, lkey_x, y, SCALE_CAPTION,
                                   COL_ACCENT, thickness=1, outline=2)
                draw_outlined_text(frame, v, lval_x, y, SCALE_MICRO,
                                   COL_TEXT_PRIMARY, thickness=1, outline=2)
            elif k:
                # Section header with no value: show dimly
                draw_outlined_text(frame, k, lkey_x, y, SCALE_MICRO,
                                   COL_TEXT_DIM, thickness=1, outline=2)

        # Right column row
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

    # Close hint pinned to the bottom of the overlay
    draw_centered_text(frame, "Press  ?  again to close",
                       h - _ix(h * 0.06), 0.34, COL_TEXT_DIM,
                       thickness=1, outline=2)

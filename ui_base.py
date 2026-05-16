"""
ui_base.py -- Colours, layout helpers, drawing primitives, text utilities.
All public names are re-exported by ui_renderer.py.
"""
import cv2
import math
import time

__all__ = [
    # Spec palette constants
    'COL_PANEL_BG', 'COL_PANEL_ALPHA', 'COL_ROW_SELECTED', 'COL_BORDER_HAIR',
    'COL_ACCENT', 'COL_TEXT_PRIMARY', 'COL_TEXT_SECONDARY', 'COL_TEXT_DIM',
    'COL_AMBER', 'COL_GREEN', 'COL_RED',
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
# ============================================================

COL_PANEL_BG       = (20, 15, 10)     # #0A0F14 frosted panel background
COL_PANEL_ALPHA    = 0.78
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
COL_CYAN           = COL_ACCENT        # was neon cyan, now accent blue
COL_MAGENTA        = COL_AMBER         # was neon magenta, now amber
COL_YELLOW         = COL_ACCENT        # was arcade yellow, now accent blue
COL_ORANGE         = COL_AMBER         # was orange warning, now amber
COL_TEXT           = COL_TEXT_PRIMARY
COL_TEXT_ACCENT    = COL_ACCENT

# Colourblind-safe variants
_COL_CB_WIN  = COL_GREEN
_COL_CB_LOSE = COL_RED
_COL_CB_DRAW = COL_TEXT_SECONDARY

# ============================================================
# TYPOGRAPHY CONSTANTS
# ============================================================

FONT_PRIMARY     = cv2.FONT_HERSHEY_SIMPLEX
FONT_DISPLAY     = cv2.FONT_HERSHEY_DUPLEX
SCALE_DISPLAY_XL = 1.7
SCALE_DISPLAY_L  = 1.2
SCALE_HEADING    = 0.90
SCALE_BODY       = 0.55
SCALE_CAPTION    = 0.45
SCALE_MICRO      = 0.40

# ============================================================
# GEOMETRY CONSTANTS
# ============================================================

TOP_BAR_PCT    = 0.06
BOTTOM_BAR_PCT = 0.06
PANEL_INSET_X  = 0.036   # 3.6% left/right for full-width panels
PANEL_INSET_Y  = 0.105   # below the top bar

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def _frame_size(frame):
    h, w = frame.shape[:2]
    return w, h

def _ix(value):
    return int(round(value))

def _fit_rect(x1, y1, x2, y2):
    return (_ix(x1), _ix(y1), _ix(x2), _ix(y2))

def _game_layout(frame):
    w, h = _frame_size(frame)
    top_bar_h    = _ix(h * TOP_BAR_PCT)
    bottom_bar_h = _ix(h * BOTTOM_BAR_PCT)
    return {
        "w": w, "h": h,
        "top_bar_h":      top_bar_h,
        "top_row_h":      top_bar_h,     # legacy key
        "second_row_h":   0,             # removed -- single top bar now
        "header_total_h": top_bar_h,     # legacy key
        "bottom_bar_h":   bottom_bar_h,
        "arcade_title_y": _ix(h * 0.17),
        "arcade_lights_y":_ix(h * 0.20),
        "status_strip": _fit_rect(w * 0.08, h * 0.18, w * 0.92, h * 0.24),
        "hero":        _fit_rect(w * 0.08, h * 0.26, w * 0.92, h * 0.68),
        "beat_track":  _fit_rect(w * 0.15, h * 0.76, w * 0.85, h * 0.90),
        "result":      _fit_rect(w * 0.08, h * 0.26, w * 0.92, h * 0.90),
        "gesture_row": _fit_rect(w * 0.08, h * 0.09, w * 0.92, h * 0.17),
    }

def _menu_layout(frame):
    w, h = _frame_size(frame)
    return {
        "w": w, "h": h,
        "panel": _fit_rect(w * PANEL_INSET_X, h * 0.10,
                           w * (1 - PANEL_INSET_X), h * 0.92),
        "bottom_bar_h": _ix(h * BOTTOM_BAR_PCT),
    }

def _settings_layout(frame):
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
    r = result_str.upper()
    if "WIN" in r or "SURVIVE" in r:
        return COL_GREEN
    if "DRAW" in r or "AGAIN" in r:
        return COL_TEXT_SECONDARY
    return COL_RED

def _get_emotion_color(emotion):
    if emotion == "Happy":      return COL_GREEN
    if emotion == "Surprised":  return COL_AMBER
    if emotion == "Frustrated": return COL_RED
    return COL_TEXT_DIM

def get_gesture_color(gesture):
    if gesture in ("Rock", "Paper", "Scissors", "Lizard", "Spock"):
        return COL_TEXT_PRIMARY
    return COL_TEXT_DIM

# ============================================================
# CORE DRAWING PRIMITIVES
# ============================================================

def draw_panel(frame, x1, y1, x2, y2,
               fill=COL_PANEL_BG, alpha=COL_PANEL_ALPHA,
               border=COL_BORDER_HAIR, border_thickness=1):
    h, w = frame.shape[:2]
    x1i = max(0, int(x1));  y1i = max(0, int(y1))
    x2i = min(w - 1, int(x2)); y2i = min(h - 1, int(y2))
    if x2i <= x1i or y2i <= y1i:
        return
    roi     = frame[y1i:y2i, x1i:x2i]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (x2i - x1i, y2i - y1i), fill, -1)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)
    if border_thickness > 0:
        cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), border, border_thickness)

def _draw_glow_border(frame, x1, y1, x2, y2, color, thickness=1):
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

def draw_outlined_text(frame, text, x, y, scale, color, thickness=1, outline=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, (int(x), int(y)), font, scale,
                (0, 0, 0), thickness + outline, cv2.LINE_AA)
    cv2.putText(frame, text, (int(x), int(y)), font, scale,
                color, thickness, cv2.LINE_AA)

def get_fit_scale(text, max_width, base_scale=1.0, thickness=2, min_scale=0.35):
    font = cv2.FONT_HERSHEY_SIMPLEX
    if cv2.getTextSize(text, font, base_scale, thickness)[0][0] <= max_width:
        return base_scale
    lo, hi = min_scale, base_scale
    for _ in range(8):
        mid = (lo + hi) / 2
        if cv2.getTextSize(text, font, mid, thickness)[0][0] <= max_width:
            lo = mid
        else:
            hi = mid
    return lo

def draw_centered_text(frame, text, center_y, scale, color, thickness=2, outline=4):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, _), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (frame.shape[1] - text_w) // 2
    draw_outlined_text(frame, text, x, center_y, scale, color, thickness, outline)

def draw_centered_text_in_rect(frame, text, rect, base_scale, color,
                                thickness=2, outline=4):
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
    Draw a flat gesture glyph (outline only) inside rect.
    Rock=circle  Paper=square  Scissors=X  Lizard=ellipse  Spock=3 lines+bar
    """
    x1, y1, x2, y2 = rect
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    side  = min(x2 - x1, y2 - y1)
    col   = color or COL_TEXT_PRIMARY
    thick = max(2, _ix(side * 0.04))

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
        # Three vertical lines + one horizontal crossbar
        gap = _ix(side * 0.14)
        top = cy - _ix(side * 0.36)
        bot = cy + _ix(side * 0.36)
        bar = cy - _ix(side * 0.08)
        for dx in (-gap, 0, gap):
            cv2.line(frame, (cx + dx, top), (cx + dx, bot), col, thick)
        cv2.line(frame, (cx - gap, bar), (cx + gap, bar), col, thick)
    else:
        # Unknown gesture: small dim square
        half = _ix(side * 0.28)
        cv2.rectangle(frame, (cx - half, cy - half), (cx + half, cy + half),
                      COL_TEXT_DIM, 1)

def draw_gesture_icon(frame, gesture, cx, cy, size):
    """Backward-compatible wrapper around draw_gesture_glyph."""
    rect = (cx - size, cy - size, cx + size, cy + size)
    draw_gesture_glyph(frame, gesture, rect, color=COL_TEXT_PRIMARY)

def draw_gesture_badge(frame, gesture, confidence, x, y,
                       threshold=0.70, show_confidence=True):
    """
    Draw a detected-gesture badge starting at (x, y).
    1-px border: accent if confident, hairline otherwise.
    """
    h, fw = frame.shape[:2]
    pad_x, pad_y = 14, 8
    font   = cv2.FONT_HERSHEY_SIMPLEX
    g_text = gesture.upper() if gesture else "NONE"
    (gw, gh), _ = cv2.getTextSize(g_text, font, SCALE_BODY, 1)

    det_text = "DETECTED"
    (dw, dh), _ = cv2.getTextSize(det_text, font, SCALE_MICRO, 1)

    dot_r = 4
    inner_w = dot_r * 2 + 6 + max(gw, dw)
    box_w = inner_w + pad_x * 2
    box_h = gh + dh + pad_y * 2 + 6

    x2 = x + box_w
    y2 = y + box_h

    confident  = confidence >= threshold
    borderline = 0.50 <= confidence < threshold
    border_col = COL_ACCENT if confident else COL_BORDER_HAIR

    draw_panel(frame, x, y, x2, y2, fill=COL_PANEL_BG, alpha=0.88,
               border=border_col, border_thickness=1)

    dot_col = COL_GREEN if confident else (COL_AMBER if borderline else COL_TEXT_DIM)
    dot_cx  = x + pad_x + dot_r
    dot_cy  = y + box_h // 2
    cv2.circle(frame, (dot_cx, dot_cy), dot_r, dot_col, -1)

    text_x = dot_cx + dot_r + 6
    cv2.putText(frame, det_text,
                (text_x, y + pad_y + dh),
                font, SCALE_MICRO, COL_TEXT_DIM, 1, cv2.LINE_AA)

    conf_suffix = ""
    if show_confidence and confidence > 0:
        conf_suffix = f"  {confidence:.2f}"
    label = g_text + conf_suffix
    draw_outlined_text(frame, label, text_x, y + pad_y + dh + 6 + gh,
                       SCALE_BODY, COL_TEXT_PRIMARY, thickness=1, outline=2)

def draw_beat_track(frame, beat_count, num_beats=4, state="", x1=None, y1=None,
                    x2=None, y2=None):
    """
    Draw the beat-track row: num_beats circles, centred horizontally.
    Inactive = 1-px ring (56, 56, 56).  Active = filled accent, dark numeral.
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
    cap_y = by1 + _ix(ph * 0.28)
    cv2.putText(frame, "BEAT TRACK", (bx1 + _ix(pw * 0.30), cap_y),
                FONT_PRIMARY, SCALE_MICRO, COL_TEXT_DIM, 1, cv2.LINE_AA)

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
        shoot_beat = is_shoot and (i == num_beats - 1)
        if shoot_beat:
            col = COL_RED
        elif act:
            col = COL_ACCENT
        else:
            col = (56, 56, 56)

        if act or shoot_beat:
            cv2.circle(frame, (cx, cy), radius, col, -1)
            num_col = (10, 10, 10)
        else:
            cv2.circle(frame, (cx, cy), radius, col, 1)
            num_col = COL_TEXT_DIM

        label = str(i + 1)
        (lw, lh), _ = cv2.getTextSize(label, FONT_PRIMARY, SCALE_MICRO, 1)
        cv2.putText(frame, label, (cx - lw // 2, cy + lh // 2),
                    FONT_PRIMARY, SCALE_MICRO, num_col, 1, cv2.LINE_AA)

    hint = "4th beat opens SHOOT" if not is_shoot else "MAKE YOUR THROW"
    hint_y = by1 + _ix(ph * 0.90)
    (hw, _), _ = cv2.getTextSize(hint, FONT_PRIMARY, SCALE_MICRO, 1)
    cv2.putText(frame, hint, (bx1 + (pw - hw) // 2, hint_y),
                FONT_PRIMARY, SCALE_MICRO, COL_TEXT_DIM, 1, cv2.LINE_AA)

def draw_progress_bar(frame, x1, y1, x2, y2, value, color=None, track_height=4):
    """
    Simple horizontal progress bar.  value is 0.0-1.0.
    """
    col = color or COL_ACCENT
    bh  = max(3, min(track_height, y2 - y1))
    mid = y1 + (y2 - y1 - bh) // 2
    cv2.rectangle(frame, (x1, mid), (x2, mid + bh), (28, 28, 28), -1)
    fill_x = x1 + _ix((x2 - x1) * max(0.0, min(1.0, value)))
    if fill_x > x1:
        cv2.rectangle(frame, (x1, mid), (fill_x, mid + bh), col, -1)

def draw_selected_row(frame, x1, y1, x2, y2, accent_bar=True):
    """
    Draw the selected-row highlight: filled rect + optional 2-px left accent bar.
    """
    roi     = frame[y1:y2, x1:x2]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (x2 - x1, y2 - y1), COL_ROW_SELECTED, -1)
    cv2.addWeighted(overlay, 0.95, roi, 0.05, 0, roi)
    if accent_bar:
        cv2.rectangle(frame, (x1, y1), (x1 + 2, y2), COL_ACCENT, -1)

def draw_row(frame, x1, y1, x2, y2, label, selected=False,
             sub_label='', right_hint=''):
    row_h = y2 - y1
    if selected:
        roi = frame[y1:y2, x1:x2]
        overlay = roi.copy()
        cv2.rectangle(overlay, (0, 0), (x2 - x1, row_h), COL_ROW_SELECTED, -1)
        cv2.addWeighted(overlay, 0.95, roi, 0.05, 0, roi)
        cv2.rectangle(frame, (x1, y1), (x1 + 2, y2), COL_ACCENT, -1)
    text_color = COL_TEXT_PRIMARY if selected else COL_TEXT_SECONDARY
    pad = _ix((x2 - x1) * 0.025)
    text_x = x1 + pad + (4 if selected else 0)
    label_y = y1 + _ix(row_h * (0.48 if not sub_label else 0.38))
    draw_outlined_text(frame, label, text_x, label_y, SCALE_BODY, text_color,
                       thickness=1, outline=2)
    if sub_label and selected:
        draw_outlined_text(frame, sub_label, text_x, y1 + _ix(row_h * 0.72),
                           SCALE_CAPTION, COL_TEXT_DIM, thickness=1, outline=2)
    if right_hint and selected:
        (tw, _), _ = cv2.getTextSize(right_hint, FONT_PRIMARY, SCALE_MICRO, 1)
        draw_outlined_text(frame, right_hint, x2 - pad - tw, label_y,
                           SCALE_MICRO, COL_ACCENT, thickness=1, outline=2)

# ============================================================
# BARS
# ============================================================

def draw_top_bar(frame, left_text, right_text):
    w, h   = _frame_size(frame)
    bar_h  = _ix(h * TOP_BAR_PCT)

    draw_panel(frame, 0, 0, w - 1, bar_h,
               fill=COL_PANEL_BG, alpha=0.72,
               border=COL_PANEL_BG, border_thickness=0)
    cv2.line(frame, (0, bar_h - 1), (w, bar_h - 1), COL_BORDER_HAIR, 1)

    left_scale = get_fit_scale(left_text, _ix(w * 0.44),
                               base_scale=SCALE_BODY, thickness=2, min_scale=0.36)
    draw_outlined_text(frame, left_text, _ix(w * 0.018), _ix(bar_h * 0.70),
                       left_scale, COL_ACCENT, thickness=2, outline=3)

    right_scale = get_fit_scale(right_text, _ix(w * 0.52),
                                base_scale=SCALE_MICRO, thickness=1, min_scale=0.26)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, _), _ = cv2.getTextSize(right_text, font, right_scale, 1)
    draw_outlined_text(frame, right_text, w - text_w - _ix(w * 0.018),
                       _ix(bar_h * 0.70), right_scale, COL_TEXT_SECONDARY,
                       thickness=1, outline=2)

def draw_bottom_bar(frame, text):
    w, h  = _frame_size(frame)
    bar_h = _ix(h * BOTTOM_BAR_PCT)
    y1    = h - bar_h

    draw_panel(frame, 0, y1, w - 1, h - 1,
               fill=COL_PANEL_BG, alpha=0.72,
               border=COL_PANEL_BG, border_thickness=0)
    cv2.line(frame, (0, y1), (w, y1), COL_BORDER_HAIR, 1)

    scale = get_fit_scale(text, _ix(w * 0.96), base_scale=SCALE_MICRO,
                          thickness=1, min_scale=0.26)
    draw_outlined_text(frame, text, _ix(w * 0.018), y1 + _ix(bar_h * 0.70),
                       scale, COL_TEXT_DIM, thickness=1, outline=2)

# ============================================================
# STATUS CHIP
# ============================================================

def draw_status_chip(frame, text, y_center, color):
    w, h = _frame_size(frame)
    font      = cv2.FONT_HERSHEY_SIMPLEX
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
    """Vertical gesture-lock bar, fills upward as gesture stabilises."""
    fh, fw = frame.shape[:2]
    pct    = min(1.0, stable_streak / max(confirm_frames, 1))
    bar_w  = max(8, fw // 60)
    bar_h  = _ix(fh * 0.22)
    bar_x  = _ix(fw * 0.012)
    bar_bot = fh - _ix(fh * TOP_BAR_PCT) - 2
    bar_top = bar_bot - bar_h

    cv2.rectangle(frame, (bar_x, bar_top), (bar_x + bar_w, bar_bot),
                  (24, 24, 24), -1)
    cv2.rectangle(frame, (bar_x, bar_top), (bar_x + bar_w, bar_bot),
                  COL_BORDER_HAIR, 1)

    if pct > 0:
        fill_h   = _ix(bar_h * pct)
        fill_top = bar_bot - fill_h
        col = COL_GREEN if pct >= 1.0 else COL_ACCENT
        cv2.rectangle(frame, (bar_x, fill_top), (bar_x + bar_w, bar_bot), col, -1)

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
    if flash_frame_idx >= max_flash_frames:
        return
    alpha = 0.28 * (1.0 - flash_frame_idx / max_flash_frames)
    color_map = {"win": COL_GREEN, "lose": COL_RED, "draw": COL_TEXT_SECONDARY}
    color   = color_map.get(result, COL_TEXT_SECONDARY)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def draw_quality_warnings(frame, hand_state):
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
        x += tw + _ix(w * 0.015)

def draw_round_history_dots(frame, rounds, x1, y, x2):
    if not rounds:
        return
    recent  = rounds[-20:]
    n       = len(recent)
    dot_r   = max(6, (x2 - x1) // (2 * max(n, 1)) - 2)
    dot_r   = min(dot_r, 10)
    step    = (x2 - x1) / max(n, 1)
    col_map = {"win": COL_GREEN, "lose": COL_RED, "draw": COL_TEXT_SECONDARY}
    gest_ch = {"Rock": "R", "Paper": "P", "Scissors": "S"}
    font    = cv2.FONT_HERSHEY_SIMPLEX
    fscale  = max(0.20, dot_r * 0.038)

    for i, r in enumerate(recent):
        outcome = r.get("outcome", r.get("player_outcome", "draw"))
        gesture = r.get("player_gesture", "")
        col     = col_map.get(outcome, COL_TEXT_SECONDARY)
        cx      = int(x1 + step * i + step / 2)
        cv2.circle(frame, (cx, y), dot_r, col, -1)
        cv2.circle(frame, (cx, y), dot_r, COL_BORDER_HAIR, 1)

        letter = gest_ch.get(gesture, "")
        if letter:
            (lw, lh), _ = cv2.getTextSize(letter, font, fscale, 1)
            cv2.putText(frame, letter, (cx - lw // 2, y + lh // 2),
                        font, fscale, (10, 10, 10), 1, cv2.LINE_AA)

def draw_help_overlay(frame, screen_name, voice_mode=False):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.86, frame, 0.14, 0, frame)

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
        section_title = "SHORTCUTS"
        left_rows  = [("ESC", "Back"), ("?", "Close help")]
        right_rows = []

    draw_centered_text(frame, section_title,
                       _ix(h * 0.08), 0.60, COL_ACCENT, thickness=2, outline=3)
    cv2.line(frame, (_ix(w * 0.05), _ix(h * 0.13)),
             (w - _ix(w * 0.05), _ix(h * 0.13)), COL_BORDER_HAIR, 1)

    row_y  = _ix(h * 0.17)
    row_h  = _ix(h * 0.056)
    lkey_x = _ix(w * 0.05)
    lval_x = _ix(w * 0.28)
    rkey_x = _ix(w * 0.55)
    rval_x = _ix(w * 0.74)

    max_rows = max(len(left_rows), len(right_rows))
    for i in range(max_rows):
        y = row_y + i * row_h
        if i < len(left_rows):
            k, v = left_rows[i]
            if k and v:
                draw_outlined_text(frame, k, lkey_x, y, SCALE_CAPTION,
                                   COL_ACCENT, thickness=1, outline=2)
                draw_outlined_text(frame, v, lval_x, y, SCALE_MICRO,
                                   COL_TEXT_PRIMARY, thickness=1, outline=2)
            elif k:
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

    draw_centered_text(frame, "Press  ?  again to close",
                       h - _ix(h * 0.06), 0.34, COL_TEXT_DIM,
                       thickness=1, outline=2)

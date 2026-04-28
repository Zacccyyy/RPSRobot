"""
UI Renderer — Arcade/Retro Theme

Neon glow aesthetic with deep dark backgrounds, cyan/magenta/yellow accents,
and bold geometric elements. All function signatures preserved for compatibility.
"""

import cv2
import math
import time

# ============================================================
# ARCADE COLOR PALETTE
# ============================================================

COL_BG_DARK = (8, 8, 16)
COL_BG_PANEL = (12, 12, 24)
COL_BG_PANEL_LIGHT = (18, 18, 32)

COL_CYAN = (255, 220, 0)       # BGR for neon cyan
COL_MAGENTA = (200, 50, 255)   # BGR for neon magenta
COL_YELLOW = (0, 220, 255)     # BGR for arcade yellow
COL_GREEN = (100, 255, 0)      # BGR for neon green
COL_RED = (40, 40, 255)        # BGR for neon red
COL_ORANGE = (0, 140, 255)     # BGR for orange

COL_TEXT = (255, 245, 240)     # cool white
COL_TEXT_DIM = (160, 150, 140) # muted
COL_TEXT_ACCENT = (255, 230, 180)  # warm highlight

# Colourblind-safe alternatives (blue/orange instead of red/green)
_COL_CB_WIN  = (220, 150, 0)    # blue-ish
_COL_CB_LOSE = (0, 120, 255)    # orange
_COL_CB_DRAW = (160, 160, 160)


def _result_colour(result_str, colourblind=False):
    """Return the appropriate WIN/LOSE/DRAW colour based on mode."""
    r = result_str.upper()
    if "WIN" in r or "SURVIVE" in r:
        return _COL_CB_WIN  if colourblind else COL_GREEN
    if "DRAW" in r or "AGAIN" in r:
        return _COL_CB_DRAW if colourblind else (160, 160, 160)
    return _COL_CB_LOSE if colourblind else COL_RED

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
    top_row_h = _ix(h * 0.075)
    second_row_h = _ix(h * 0.065)
    header_total_h = top_row_h + second_row_h
    return {
        "w": w, "h": h,
        "top_row_h": top_row_h,
        "second_row_h": second_row_h,
        "header_total_h": header_total_h,
        "bottom_bar_h": _ix(h * 0.065),
        "arcade_title_y": _ix(h * 0.195),
        "arcade_lights_y": _ix(h * 0.232),
        "status_strip": _fit_rect(w * 0.12, h * 0.260, w * 0.88, h * 0.320),
        "hero": _fit_rect(w * 0.09, h * 0.35, w * 0.91, h * 0.65),
        "beat_track": _fit_rect(w * 0.17, h * 0.69, w * 0.83, h * 0.88),
        "result": _fit_rect(w * 0.08, h * 0.32, w * 0.92, h * 0.90),
    }

def _menu_layout(frame):
    w, h = _frame_size(frame)
    return {"w": w, "h": h,
            "panel": _fit_rect(w * 0.09, h * 0.12, w * 0.91, h * 0.92),
            "bottom_bar_h": _ix(h * 0.065)}

def _settings_layout(frame):
    w, h = _frame_size(frame)
    return {"w": w, "h": h,
            "panel": _fit_rect(w * 0.065, h * 0.12, w * 0.935, h * 0.92),
            "bottom_bar_h": _ix(h * 0.065)}

# ============================================================
# DRAWING PRIMITIVES
# ============================================================

def get_gesture_color(gesture):
    if gesture == "Rock":     return COL_YELLOW
    if gesture == "Paper":    return COL_GREEN
    if gesture == "Scissors": return COL_MAGENTA
    return COL_TEXT_DIM

def _get_emotion_color(emotion):
    if emotion == "Happy":      return (0, 255, 200)   # bright green-cyan
    if emotion == "Surprised":  return (0, 200, 255)    # warm yellow-orange
    if emotion == "Frustrated": return (80, 80, 255)    # soft red
    return COL_TEXT_DIM                                 # Neutral

def draw_panel(frame, x1, y1, x2, y2, fill=COL_BG_PANEL, alpha=0.82,
               border=COL_CYAN, border_thickness=2):
    # Clamp to frame bounds
    h, w = frame.shape[:2]
    x1i, y1i = max(0, int(x1)), max(0, int(y1))
    x2i, y2i = min(w - 1, int(x2)), min(h - 1, int(y2))
    if x2i <= x1i or y2i <= y1i:
        return
    # Copy only the ROI sub-rect — ~10-50× cheaper than copying the full frame
    roi = frame[y1i:y2i, x1i:x2i]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (x2i - x1i, y2i - y1i), fill, -1)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)
    if border_thickness > 0:
        cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), border, border_thickness)

def _draw_glow_border(frame, x1, y1, x2, y2, color, thickness=2):
    """Draw a border with a subtle glow effect."""
    # Outer glow.
    glow_color = tuple(max(0, c // 3) for c in color)
    cv2.rectangle(frame, (int(x1)-2, int(y1)-2), (int(x2)+2, int(y2)+2),
                  glow_color, thickness + 2)
    # Main border.
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                  color, thickness)


def draw_gesture_icon(frame, gesture, cx, cy, size):
    """
    Draw a simple polygon icon for Rock / Paper / Scissors.
    cx, cy = centre point; size = approximate radius in pixels.
    """
    import numpy as np
    if gesture == "Rock":
        # Filled circle (fist)
        cv2.circle(frame, (cx, cy), size, COL_YELLOW, -1)
        cv2.circle(frame, (cx, cy), size, (0, 0, 0), 2)
    elif gesture == "Paper":
        # Open rectangle (flat hand)
        s = size
        pts = np.array([
            [cx - s, cy - int(s * 0.6)],
            [cx + s, cy - int(s * 0.6)],
            [cx + s, cy + int(s * 0.6)],
            [cx - s, cy + int(s * 0.6)],
        ], np.int32)
        cv2.fillPoly(frame, [pts], COL_GREEN)
        cv2.polylines(frame, [pts], True, (0, 0, 0), 2)
    elif gesture == "Scissors":
        # Two diagonal lines forming a V
        s = size
        cv2.line(frame, (cx - s, cy - s), (cx, cy), COL_MAGENTA, max(3, size // 5))
        cv2.line(frame, (cx + s, cy - s), (cx, cy), COL_MAGENTA, max(3, size // 5))
        cv2.circle(frame, (cx, cy), size // 3, COL_MAGENTA, -1)
        cv2.circle(frame, (cx - s, cy - s), size // 4, COL_MAGENTA, -1)
        cv2.circle(frame, (cx + s, cy - s), size // 4, COL_MAGENTA, -1)


def draw_result_flash(frame, result, flash_frame_idx, max_flash_frames=4, colourblind=False):
    """Brief coloured flash overlay when a round resolves."""
    if flash_frame_idx >= max_flash_frames:
        return
    alpha = 0.35 * (1.0 - flash_frame_idx / max_flash_frames)
    if colourblind:
        color_map = {"win": _COL_CB_WIN, "lose": _COL_CB_LOSE, "draw": _COL_CB_DRAW}
    else:
        color_map = {"win": (0, 255, 80), "lose": (0, 60, 220), "draw": (180, 180, 180)}
    color = color_map.get(result, (180, 180, 180))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_gesture_confidence_bar(frame, stable_streak, confirm_frames, x, y, width):
    """
    Vertical gesture lock bar on the left edge — fills upward as gesture stabilises.
    x, y, width params are kept for API compatibility but position is recalculated
    to sit above the bottom bar and fill upward.
    """
    fh, fw = frame.shape[:2]
    pct = min(1.0, stable_streak / max(confirm_frames, 1))

    # Dimensions: thin vertical bar left of frame, above bottom bar
    bar_w  = max(10, fw // 60)
    bar_h  = int(fh * 0.22)          # total available height
    bar_x  = _ix(fw * 0.012)
    bar_bot = fh - _ix(fh * 0.075)   # sit above the bottom legend bar
    bar_top = bar_bot - bar_h

    # Background track
    cv2.rectangle(frame, (bar_x, bar_top), (bar_x + bar_w, bar_bot), (35, 35, 35), -1)
    cv2.rectangle(frame, (bar_x, bar_top), (bar_x + bar_w, bar_bot), (80, 80, 80), 1)

    # Fill from bottom upward
    if pct > 0:
        fill_h   = int(bar_h * pct)
        fill_top = bar_bot - fill_h
        col = COL_GREEN if pct >= 1.0 else COL_YELLOW
        cv2.rectangle(frame, (bar_x, fill_top), (bar_x + bar_w, bar_bot), col, -1)

    # Label rotated 90° using individual chars stacked vertically
    label = "LOCK"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    fscale = 0.28
    lh = int(fh * 0.028)
    for i, ch in enumerate(label):
        cy = bar_top - _ix(fh * 0.005) - (len(label) - 1 - i) * lh
        (cw, _), _ = cv2.getTextSize(ch, font, fscale, 1)
        cx = bar_x + bar_w // 2 - cw // 2
        cv2.putText(frame, ch, (cx, cy), font, fscale,
                    (120, 120, 120), 1, cv2.LINE_AA)


def draw_quality_warnings(frame, hand_state):
    """
    Overlay small warning chips if the hand is too far or lighting is poor.
    Positioned just below the top bar.
    """
    h, w = frame.shape[:2]
    warn_y = _ix(h * 0.09)
    warnings = []
    if hand_state.get("hand_too_far"):
        warnings.append(("MOVE CLOSER", COL_ORANGE))
    if hand_state.get("poor_lighting"):
        warnings.append(("POOR LIGHTING", (60, 60, 200)))

    x = _ix(w * 0.02)
    for text, col in warnings:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        draw_panel(frame, x - 4, warn_y - th - 4, x + tw + 8, warn_y + 6,
                   fill=(0, 0, 0), alpha=0.65, border=col, border_thickness=1)
        cv2.putText(frame, text, (x, warn_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, col, 1, cv2.LINE_AA)
        x += tw + _ix(w * 0.015)


def draw_round_history_dots(frame, rounds, x1, y, x2):
    """
    Row of dots for the last 20 rounds.
    Colour = outcome (green=win, red=lose, grey=draw).
    Letter inside = gesture played (R / P / S).
    """
    if not rounds:
        return
    recent  = rounds[-20:]
    n       = len(recent)
    dot_r   = max(6, (x2 - x1) // (2 * max(n, 1)) - 2)
    dot_r   = min(dot_r, 10)
    step    = (x2 - x1) / max(n, 1)
    col_map = {"win": COL_GREEN, "lose": COL_RED, "draw": (140, 140, 140)}
    gest_ch = {"Rock": "R", "Paper": "P", "Scissors": "S"}
    font    = cv2.FONT_HERSHEY_SIMPLEX
    fscale  = max(0.20, dot_r * 0.038)

    for i, r in enumerate(recent):
        outcome = r.get("outcome", r.get("player_outcome", "draw"))
        gesture = r.get("player_gesture", "")
        col     = col_map.get(outcome, (140, 140, 140))
        cx      = int(x1 + step * i + step / 2)
        cv2.circle(frame, (cx, y), dot_r, col, -1)
        cv2.circle(frame, (cx, y), dot_r, (0, 0, 0), 1)

        letter = gest_ch.get(gesture, "")
        if letter:
            (lw, lh), _ = cv2.getTextSize(letter, font, fscale, 1)
            cv2.putText(frame, letter, (cx - lw // 2, y + lh // 2),
                        font, fscale, (10, 10, 10), 1, cv2.LINE_AA)


def draw_help_overlay(frame, screen_name, voice_mode=False):
    """
    ? key — full shortcut overlay, differentiated by screen and input mode.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    # ── Bindings per screen ──────────────────────────────────────────────────
    if screen_name == "GAME" and voice_mode:
        section_title = "SHORTCUTS  |  VOICE MODE"
        left_rows = [
            ("READY",     "Start the countdown"),
            ("ONE",       "First beat"),
            ("TWO",       "Second beat"),
            ("THREE",     "Third beat + opens SHOOT"),
            ("ROCK",      "Throw Rock"),
            ("PAPER",     "Throw Paper"),
            ("SCISSORS",  "Throw Scissors"),
            ("BACK",      "Return to menu"),
            ("CHEAT",     "Switch to Cheat mode"),
            ("FAIR",      "Switch to Fair Play"),
            ("CHALLENGE", "Switch to Challenge"),
        ]
        right_rows = [
            ("Also accepted:",""),
            ("STEADY / FREDDY / BETTY", "-> READY"),
            ("WON / ON / AND",          "-> ONE"),
            ("TO / TOO / DO",           "-> TWO"),
            ("TREE / FREE / THROUGH",   "-> THREE"),
            ("LOCK / KNOCK / WALK",     "-> ROCK"),
            ("FAVOR / TAPER",           "-> PAPER"),
            ("SISTERS / SEIZURES",      "-> SCISSORS"),
            ("", ""),
            ("ESC",  "Return to Menu"),
            ("?",    "Close this help"),
        ]
    elif screen_name == "GAME":
        section_title = "SHORTCUTS  |  PUMP MODE"
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
        section_title = "SHORTCUTS  |  MENU"
        left_rows = [
            ("UP / W",       "Move selection up"),
            ("DOWN / S",     "Move selection down"),
            ("ENTER",        "Select item"),
            ("ESC",          "Back / Exit sub-menu"),
            ("Q",            "Quit application"),
            ("?",            "Close this help"),
        ]
        right_rows = [
            ("Voice nav:",""),
            ("CHEAT",      "Open Cheat mode"),
            ("FAIR",       "Open Fair Play"),
            ("CHALLENGE",  "Open Challenge"),
            ("CLONE",      "Open Clone setup"),
            ("STATS",      "Open Player Stats"),
            ("TUTORIAL",   "How to Play"),
            ("SETTINGS",   "Open Settings"),
            ("BACK / NO",  "Go back"),
            ("SELECT / YES","Select item"),
        ]
    elif screen_name == "SETTINGS":
        section_title = "SHORTCUTS  |  SETTINGS"
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
        section_title = "SHORTCUTS  |  FEATURES"
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
    else:
        section_title = "SHORTCUTS"
        left_rows  = [("ESC", "Back"), ("?", "Close help")]
        right_rows = []

    # ── Title ────────────────────────────────────────────────────────────────
    draw_centered_text(frame, section_title,
                       _ix(h * 0.08), 0.62, COL_YELLOW, thickness=2, outline=3)
    cv2.line(frame, (_ix(w * 0.05), _ix(h * 0.13)), (w - _ix(w * 0.05), _ix(h * 0.13)),
             COL_CYAN, 1)

    # ── Two-column layout ────────────────────────────────────────────────────
    row_y  = _ix(h * 0.17)
    row_h  = _ix(h * 0.058)
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
                draw_outlined_text(frame, k, lkey_x, y, 0.42, COL_CYAN,   thickness=1, outline=2)
                draw_outlined_text(frame, v, lval_x, y, 0.40, COL_TEXT,   thickness=1, outline=2)
            elif k:  # section header
                draw_outlined_text(frame, k, lkey_x, y, 0.38, COL_TEXT_DIM, thickness=1, outline=2)

        if i < len(right_rows):
            k, v = right_rows[i]
            if k and v:
                draw_outlined_text(frame, k, rkey_x, y, 0.42, COL_CYAN,   thickness=1, outline=2)
                draw_outlined_text(frame, v, rval_x, y, 0.38, COL_TEXT,   thickness=1, outline=2)
            elif k:  # section header
                draw_outlined_text(frame, k, rkey_x, y, 0.38, COL_TEXT_DIM, thickness=1, outline=2)

    draw_centered_text(frame, "Press  ?  again to close",
                       h - _ix(h * 0.06), 0.36, COL_TEXT_DIM, thickness=1, outline=2)

def get_fit_scale(text, max_width, base_scale=1.0, thickness=2, min_scale=0.35):
    font = cv2.FONT_HERSHEY_SIMPLEX
    if cv2.getTextSize(text, font, base_scale, thickness)[0][0] <= max_width:
        return base_scale
    lo, hi = min_scale, base_scale
    for _ in range(8):  # binary search — ~3× faster than 0.05 linear steps
        mid = (lo + hi) / 2
        if cv2.getTextSize(text, font, mid, thickness)[0][0] <= max_width:
            lo = mid
        else:
            hi = mid
    return lo

def draw_outlined_text(frame, text, x, y, scale, color, thickness=2, outline=4):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Glow layer (color-tinted, larger).
    glow = tuple(max(0, c // 3) for c in color)
    cv2.putText(frame, text, (int(x), int(y)), font, scale, glow, thickness + outline + 2)
    # Dark outline.
    cv2.putText(frame, text, (int(x), int(y)), font, scale, (0, 0, 0), thickness + outline)
    # Main text.
    cv2.putText(frame, text, (int(x), int(y)), font, scale, color, thickness)

def draw_centered_text(frame, text, center_y, scale, color, thickness=2, outline=4):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, _), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (frame.shape[1] - text_w) // 2
    draw_outlined_text(frame, text, x, center_y, scale, color, thickness, outline)

def draw_centered_text_in_rect(frame, text, rect, base_scale, color, thickness=2, outline=4):
    x1, y1, x2, y2 = rect
    max_width = max(40, (x2 - x1) - 20)
    scale = get_fit_scale(text, max_width, base_scale=base_scale, thickness=thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x = x1 + ((x2 - x1) - text_w) // 2
    y = y1 + ((y2 - y1) + text_h) // 2
    draw_outlined_text(frame, text, x, y, scale, color, thickness, outline)

# ============================================================
# BARS
# ============================================================

def draw_top_bar(frame, left_text, right_text):
    w, h = _frame_size(frame)
    bar_h = _ix(h * 0.10)

    draw_panel(frame, 0, 0, w - 1, bar_h, fill=(6, 6, 14), alpha=0.92,
               border=(6, 6, 14), border_thickness=0)
    # Accent line at bottom of bar.
    cv2.line(frame, (0, bar_h), (w, bar_h), COL_CYAN, 1)

    left_scale = get_fit_scale(left_text, _ix(w * 0.42), base_scale=0.62, thickness=2, min_scale=0.40)
    draw_outlined_text(frame, left_text, _ix(w * 0.02), _ix(bar_h * 0.62),
                       left_scale, COL_CYAN, thickness=2, outline=3)

    right_scale = get_fit_scale(right_text, _ix(w * 0.50), base_scale=0.40, thickness=1, min_scale=0.26)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, _), _ = cv2.getTextSize(right_text, font, right_scale, 1)
    draw_outlined_text(frame, right_text, w - text_w - _ix(w * 0.02),
                       _ix(bar_h * 0.64), right_scale, COL_TEXT_DIM, thickness=1, outline=2)

def draw_bottom_bar(frame, text):
    w, h = _frame_size(frame)
    bar_h = _ix(h * 0.065)
    y1 = h - bar_h

    draw_panel(frame, 0, y1, w - 1, h - 1, fill=(6, 6, 14), alpha=0.92,
               border=(6, 6, 14), border_thickness=0)
    cv2.line(frame, (0, y1), (w, y1), COL_CYAN, 1)
    draw_outlined_text(frame, text, _ix(w * 0.02), y1 + _ix(bar_h * 0.68),
                       0.42, COL_TEXT_DIM, thickness=1, outline=2)

# ============================================================
# STATUS CHIP
# ============================================================

def draw_status_chip(frame, text, y_center, color):
    w, h = _frame_size(frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = get_fit_scale(text, _ix(w * 0.52), base_scale=0.62, thickness=2)
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    pad_x = _ix(w * 0.025)
    pad_y = _ix(h * 0.018)
    x1 = (w - text_w) // 2 - pad_x
    y1 = y_center - text_h - pad_y
    x2 = x1 + text_w + pad_x * 2
    y2 = y_center + pad_y

    draw_panel(frame, x1, y1, x2, y2, fill=(10, 10, 20), alpha=0.90, border=color, border_thickness=2)
    draw_outlined_text(frame, text, x1 + pad_x, y_center - 2, scale, color,
                       thickness=thickness, outline=3)

# ============================================================
# RESULT BANNER COLOR
# ============================================================

def get_result_banner_color(banner, colourblind=False):
    if banner.startswith("YOU WIN") or banner.startswith("YOU SURVIVE"):
        return _COL_CB_WIN  if colourblind else COL_GREEN
    if banner.startswith("DRAW"):
        return _COL_CB_DRAW if colourblind else COL_TEXT
    if banner.startswith("GAME OVER") or "TAKES" in banner or "WINS" in banner:
        return _COL_CB_LOSE if colourblind else COL_RED
    return COL_TEXT

# ============================================================
# GAME HEADER
# ============================================================

def draw_game_header(frame, game_state, voice_mode_active=False, sound_on=True):
    layout = _game_layout(frame)
    w = layout["w"]
    top_row_h = layout["top_row_h"]
    second_row_h = layout["second_row_h"]

    draw_panel(frame, 0, 0, w - 1, top_row_h, fill=(6, 6, 14), alpha=0.94,
               border=(6, 6, 14), border_thickness=0)
    draw_panel(frame, 0, top_row_h, w - 1, top_row_h + second_row_h,
               fill=(10, 10, 20), alpha=0.92, border=(10, 10, 20), border_thickness=0)
    cv2.line(frame, (0, top_row_h + second_row_h), (w, top_row_h + second_row_h), COL_CYAN, 1)

    draw_outlined_text(frame, "RPS ROBOT", _ix(w * 0.02), _ix(top_row_h * 0.72),
                       0.56, COL_CYAN, thickness=2, outline=3)

    # Top right: sound ON indicator (only when active)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if sound_on:
        sound_badge = "[ SOUND ON ]"
        (sw, _), _ = cv2.getTextSize(sound_badge, font, 0.32, 1)
        draw_outlined_text(frame, sound_badge, w - sw - _ix(w * 0.02),
                           _ix(top_row_h * 0.72), 0.32, COL_GREEN, thickness=1, outline=2)

    mode_text = game_state["play_mode_label"].upper()
    mode_scale = get_fit_scale(mode_text, _ix(w * 0.38), base_scale=0.74, thickness=2, min_scale=0.42)
    draw_outlined_text(frame, mode_text, _ix(w * 0.02),
                       top_row_h + _ix(second_row_h * 0.74), mode_scale, COL_YELLOW, thickness=2, outline=3)

    mode_controls = "1 Cheat | 2 Fair | 3 Challenge"
    ctrl_scale = get_fit_scale(mode_controls, _ix(w * 0.44), base_scale=0.32, thickness=1, min_scale=0.22)
    (ctrl_w, _), _ = cv2.getTextSize(mode_controls, font, ctrl_scale, 1)
    draw_outlined_text(frame, mode_controls, w - ctrl_w - _ix(w * 0.02),
                       top_row_h + _ix(second_row_h * 0.72), ctrl_scale, COL_TEXT_DIM, thickness=1, outline=2)

# ============================================================
# GAME STATUS STRIP
# ============================================================

def draw_game_status_strip(frame, game_state):
    layout = _game_layout(frame)
    x1, y1, x2, y2 = layout["status_strip"]

    if game_state["play_mode_label"] in ("Fair Play", "Challenge") or game_state["play_mode_label"].startswith("vs "):
        text = f"{game_state['round_text']}   |   {game_state['score_text']}"
    else:
        text = "Cheat mode counters your throw after SHOOT"

    draw_panel(frame, x1, y1, x2, y2, fill=(10, 10, 22), alpha=0.88,
               border=COL_CYAN, border_thickness=1)
    draw_centered_text_in_rect(frame, text, (x1 + 8, y1 + 2, x2 - 8, y2 - 2),
                               base_scale=0.52, color=COL_TEXT_ACCENT, thickness=1, outline=2)

# ============================================================
# DIAGNOSTIC PANELS (kept functional, not heavily styled)
# ============================================================

def draw_info_panel(frame, tracker_state, game_state, count_text, status_text,
                    reason_text, ambiguous_count, output_summary,
                    emotion_state=None, fps=None):
    w, h = _frame_size(frame)
    x1, y1, x2, y2 = _fit_rect(w * 0.02, h * 0.15, w * 0.55, h * 0.76)
    draw_panel(frame, x1, y1, x2, y2, fill=(8, 8, 16), alpha=0.82, border=COL_CYAN, border_thickness=1)

    raw_gesture = tracker_state["raw_gesture"]
    stable_gesture = tracker_state["stable_gesture"]
    confirmed_gesture = tracker_state["confirmed_gesture"]
    robot_ready = tracker_state["robot_ready"]
    command_text = tracker_state["command"]
    stable_streak = tracker_state["stable_streak"]
    history_size = tracker_state["history_size"]
    play_mode_label = game_state["play_mode_label"]

    fps_line = []
    if fps is not None:
        fps_col = COL_GREEN if fps >= 25 else (COL_YELLOW if fps >= 15 else COL_RED)
        fps_line = [(f"FPS: {fps:.0f}", fps_col, 0.52, 2)]

    lines = fps_line + [
        (f"Mode: {play_mode_label}", COL_TEXT, 0.56, 2),
        (f"Count: {count_text}", COL_GREEN, 0.64, 2),
        (f"Raw: {raw_gesture}", get_gesture_color(raw_gesture), 0.58, 2),
        (f"Stable: {stable_gesture}", get_gesture_color(stable_gesture), 0.58, 2),
        (f"Confirmed: {confirmed_gesture}", get_gesture_color(confirmed_gesture), 0.58, 2),
        (f"Frames: {stable_streak}/3   Buf: {history_size}/7", (210, 200, 200), 0.46, 1),
        (f"Robot Ready: {'YES' if robot_ready else 'NO'}", COL_GREEN if robot_ready else COL_ORANGE, 0.52, 2),
        (f"Safe Cmd: {command_text}", COL_TEXT, 0.46, 1),
        (f"Status: {status_text}", COL_CYAN, 0.42, 1),
        (f"Reason: {reason_text}", COL_TEXT_DIM, 0.40, 1),
        (f"Ambig: {ambiguous_count}", COL_TEXT_DIM, 0.40, 1),
        (f"Output: {output_summary}", COL_TEXT_ACCENT, 0.36, 1),
    ]

    # Emotion rows — [E] hint removed here, it's already in the top bar
    if emotion_state and emotion_state.get("face_detected"):
        em       = emotion_state["stable_emotion"]
        em_color = _get_emotion_color(em)
        sc       = emotion_state["scores"]
        cal      = emotion_state.get("calibrated", True)
        cal_prog = emotion_state.get("calibration_progress", 100)

        if not cal:
            lines.append((f"Emotion: calibrating... {cal_prog}%", (200, 180, 80), 0.46, 1))
            lines.append(("  Look neutral to set baseline", COL_TEXT_DIM, 0.34, 1))
        else:
            em_detail = (
                f"Smile:{sc['smile']:.2f}  "
                f"Surp:{sc['surprise']:.2f}  "
                f"Frust:{sc['frustration']:.2f}"
            )
            lines.append((f"Emotion: {em} ({emotion_state['confidence']:.0%})", em_color, 0.52, 2))
            lines.append((f"  {em_detail}", em_color, 0.36, 1))
    elif emotion_state:
        lines.append(("Emotion: No face detected", COL_TEXT_DIM, 0.42, 1))

    y    = y1 + _ix(h * 0.040)
    step = _ix(h * 0.035)
    for text, color, scale, thickness in lines:
        if y + step > y2 - _ix(h * 0.01):
            break   # never overflow the panel
        draw_outlined_text(frame, text, x1 + _ix(w * 0.02), y, scale, color,
                           thickness=thickness, outline=2)
        y += step

def draw_diagnostic_game_panel(frame, game_state):
    w, h = _frame_size(frame)
    x1, y1, x2, y2 = _fit_rect(w * 0.02, h * 0.72, w * 0.98, h * 0.97)
    draw_panel(frame, x1, y1, x2, y2, fill=(8, 8, 16), alpha=0.84, border=COL_CYAN, border_thickness=1)

    state_label = game_state["state_label"]
    beat_count = game_state["beat_count"]
    time_left = game_state["time_left"]
    main_text = game_state["main_text"]
    sub_text = game_state["sub_text"]
    score_text = game_state["score_text"]
    round_text = game_state["round_text"]

    draw_outlined_text(frame, f"State: {state_label}", x1 + _ix(w * 0.025),
                       y1 + _ix(h * 0.05), 0.60, COL_TEXT, thickness=2, outline=3)

    line2 = f"Beats: {beat_count}/4"
    if round_text: line2 += f"   {round_text}"
    if score_text: line2 += f"   {score_text}"
    if game_state["state"] == "SHOOT_WINDOW": line2 += f"   {time_left:.2f}s"

    draw_outlined_text(frame, line2, x1 + _ix(w * 0.025), y1 + _ix(h * 0.10),
                       0.44, COL_TEXT_DIM, thickness=1, outline=2)

    draw_centered_text_in_rect(frame, main_text,
        (x1 + 20, y1 + _ix(h * 0.11), x2 - 20, y1 + _ix(h * 0.19)),
        base_scale=0.95, color=COL_TEXT, thickness=2, outline=3)

    draw_centered_text_in_rect(frame, sub_text,
        (x1 + 20, y1 + _ix(h * 0.18), x2 - 20, y2 - 8),
        base_scale=0.56, color=COL_TEXT_DIM, thickness=1, outline=2)

# ============================================================
# ARCADE HEADER (title + lights)
# ============================================================

def draw_arcade_header(frame):
    layout = _game_layout(frame)
    w = layout["w"]

    draw_centered_text(frame, "ROCK   PAPER   SCISSORS", layout["arcade_title_y"],
                       0.62, COL_TEXT, thickness=2, outline=3)

    # Animated neon dots.
    t = time.monotonic()
    light_positions = [_ix(w * p) for p in (0.29, 0.39, 0.50, 0.61, 0.71)]
    light_colors = [COL_YELLOW, COL_MAGENTA, COL_GREEN, COL_MAGENTA, COL_YELLOW]
    ly = layout["arcade_lights_y"]

    for i, (x, color) in enumerate(zip(light_positions, light_colors)):
        # Subtle pulse.
        pulse = 0.6 + 0.4 * math.sin(t * 3.5 + i * 1.2)
        r = _ix(w * 0.010)
        bright = tuple(min(255, int(c * pulse)) for c in color)

        # Glow.
        cv2.circle(frame, (x, ly), r + 4, tuple(c // 4 for c in color), -1)
        cv2.circle(frame, (x, ly), r, bright, -1)

# ============================================================
# ARCADE HERO (main game state display)
# ============================================================

def draw_arcade_hero(frame, game_state, voice_mode_active=False):
    layout = _game_layout(frame)
    w, h = layout["w"], layout["h"]
    x1, y1, x2, y2 = layout["hero"]
    state      = game_state["state"]
    main_text  = game_state["main_text"]
    sub_text   = game_state["sub_text"]
    time_left  = game_state["time_left"]
    beat_count = game_state.get("beat_count", 0)

    draw_panel(frame, x1, y1, x2, y2, fill=(8, 8, 18), alpha=0.84,
               border=COL_CYAN, border_thickness=1)

    chip_y = y1 + _ix((y2 - y1) * 0.14)

    if state == "ROUND_INTRO":
        draw_status_chip(frame, game_state["round_text"], chip_y, COL_CYAN)
        draw_centered_text(frame, main_text, y1 + _ix((y2 - y1) * 0.52),
                           1.15, COL_TEXT, thickness=3, outline=4)
        draw_centered_text(frame, sub_text, y1 + _ix((y2 - y1) * 0.82),
                           0.48, COL_TEXT_DIM, thickness=2, outline=3)

    elif state == "WAITING_FOR_ROCK":
        if voice_mode_active:
            draw_status_chip(frame, "VOICE MODE", chip_y, (80, 255, 180))
            draw_centered_text(frame, 'Say  "READY"',
                               y1 + _ix((y2 - y1) * 0.46),
                               1.00, (80, 255, 180), thickness=3, outline=4)
            draw_centered_text(frame, "to start the countdown",
                               y1 + _ix((y2 - y1) * 0.76),
                               0.46, COL_TEXT_DIM, thickness=1, outline=2)
        else:
            draw_status_chip(frame, "READY", chip_y, COL_TEXT)
            draw_centered_text(frame, "MAKE A FIST", y1 + _ix((y2 - y1) * 0.48),
                               1.20, COL_TEXT, thickness=3, outline=4)
            draw_centered_text(frame, "TO START", y1 + _ix((y2 - y1) * 0.70),
                               1.20, COL_TEXT, thickness=3, outline=4)
            draw_centered_text(frame, sub_text, y1 + _ix((y2 - y1) * 0.90),
                               0.44, COL_TEXT_DIM, thickness=2, outline=3)

    elif state == "COUNTDOWN":
        if voice_mode_active:
            next_words = {0: '"ONE"', 1: '"TWO"', 2: '"THREE"'}
            next_word  = next_words.get(beat_count, '"THREE"')
            chip_label = f"BEAT  {beat_count} / 3" if beat_count > 0 else "COUNTING"
            chip_col   = COL_YELLOW if beat_count >= 2 else COL_CYAN
            draw_status_chip(frame, chip_label, chip_y, chip_col)
            draw_centered_text(frame, f"Say  {next_word}",
                               y1 + _ix((y2 - y1) * 0.46),
                               1.10, COL_CYAN, thickness=3, outline=5)
        else:
            draw_status_chip(frame, "COUNTDOWN", chip_y, COL_CYAN)
            # Pulse: scale in from 0.6 → 1.0 over the beat window using sin curve
            t       = time.monotonic()
            pulse   = 0.72 + 0.28 * abs(math.sin(t * math.pi * 1.4))
            num_col = tuple(min(255, int(c * pulse)) for c in COL_CYAN)
            draw_centered_text_in_rect(frame, main_text,
                (x1 + _ix(w * 0.06), y1 + _ix((y2 - y1) * 0.22),
                 x2 - _ix(w * 0.06), y1 + _ix((y2 - y1) * 0.65)),
                base_scale=2.4, color=num_col, thickness=4, outline=6)
            draw_centered_text(frame, sub_text, y1 + _ix((y2 - y1) * 0.88),
                               0.46, COL_TEXT_DIM, thickness=2, outline=3)

    elif state == "SHOOT_WINDOW":
        draw_status_chip(frame, "THROW NOW!", chip_y, COL_RED)
        t     = time.monotonic()
        pulse = 0.7 + 0.3 * math.sin(t * 8)
        pcol  = tuple(min(255, int(c * pulse)) for c in COL_RED)

        if voice_mode_active:
            draw_centered_text(frame, "SAY YOUR THROW",
                               y1 + _ix((y2 - y1) * 0.40),
                               0.90, pcol, thickness=3, outline=4)
            # Three throw options in their own columns — fixed x, not frame-centered
            throws    = [("ROCK", COL_YELLOW), ("PAPER", COL_GREEN), ("SCISSORS", COL_MAGENTA)]
            panel_w   = x2 - x1
            col_w     = panel_w // 3
            row_y     = y1 + _ix((y2 - y1) * 0.74)
            for i, (word, col) in enumerate(throws):
                cx = x1 + col_w * i + col_w // 2
                (tw, _), _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
                draw_outlined_text(frame, word, cx - tw // 2, row_y,
                                   0.60, col, thickness=2, outline=3)
        else:
            draw_centered_text(frame, "SHOOT!", y1 + _ix((y2 - y1) * 0.48),
                               1.70, pcol, thickness=4, outline=5)
            draw_centered_text(frame, f"{time_left:.2f}s", y1 + _ix((y2 - y1) * 0.68),
                               0.65, COL_TEXT_ACCENT, thickness=2, outline=3)
            draw_centered_text(frame, sub_text, y1 + _ix((y2 - y1) * 0.88),
                               0.44, COL_TEXT_DIM, thickness=2, outline=3)

# ============================================================
# ARCADE BEAT TRACK
# ============================================================

def draw_arcade_beat_track(frame, beat_count, state, voice_mode_active=False):
    layout = _game_layout(frame)
    w, h = layout["w"], layout["h"]
    x1, y1, x2, y2 = layout["beat_track"]

    draw_panel(frame, x1, y1, x2, y2, fill=(8, 8, 18), alpha=0.86,
               border=COL_CYAN, border_thickness=1)

    if voice_mode_active:
        # Voice mode: 3 circles labelled ONE / TWO / THREE
        draw_centered_text(frame, "VOICE COUNTDOWN", y1 + _ix((y2 - y1) * 0.22),
                           0.42, (80, 255, 180), thickness=1, outline=2)

        panel_w   = x2 - x1
        positions = [x1 + _ix(panel_w * p) for p in (0.18, 0.50, 0.82)]
        labels    = ["ONE", "TWO", "THREE"]
        colors    = [COL_CYAN, COL_CYAN, COL_YELLOW]
        cy        = y1 + _ix((y2 - y1) * 0.56)
        radius    = _ix(min(w, h) * 0.036)

        for i, (x, label, base_col) in enumerate(zip(positions, labels, colors)):
            active = i < beat_count
            shoot_active = (i == 2 and state == "SHOOT_WINDOW")
            color = COL_RED if shoot_active else (base_col if active else (60, 60, 80))

            if active or shoot_active:
                cv2.circle(frame, (x, cy), radius + 5, tuple(c // 3 for c in color), 2)
                cv2.circle(frame, (x, cy), radius, color, -1)
                text_color = (10, 10, 10)
            else:
                cv2.circle(frame, (x, cy), radius, color, 2)
                text_color = color

            draw_centered_text_in_rect(frame, str(i + 1),
                (x - radius, cy - radius, x + radius, cy + radius),
                base_scale=0.80, color=text_color, thickness=2, outline=0)

            # Word label below circle
            (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
            cv2.putText(frame, label, (x - lw // 2, cy + radius + _ix(h * 0.028)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)

        hint = "THREE opens throw window" if state != "SHOOT_WINDOW" else "Say ROCK  PAPER  or  SCISSORS"
        draw_centered_text(frame, hint, y1 + _ix((y2 - y1) * 0.90),
                           0.38, COL_TEXT_DIM, thickness=1, outline=2)

    else:
        # Physical mode — original 4-beat track
        draw_centered_text(frame, "BEAT TRACK", y1 + _ix((y2 - y1) * 0.22),
                           0.42, COL_TEXT_DIM, thickness=1, outline=2)

        panel_w   = x2 - x1
        positions = [x1 + _ix(panel_w * p) for p in (0.12, 0.37, 0.62, 0.87)]
        labels    = ["1", "2", "3", "4"]
        cy        = y1 + _ix((y2 - y1) * 0.56)
        radius    = _ix(min(w, h) * 0.036)

        for i, (x, label) in enumerate(zip(positions, labels)):
            active   = i < beat_count
            is_shoot = (i == 3 and state == "SHOOT_WINDOW")
            color    = COL_RED if is_shoot else (COL_CYAN if active else (60, 60, 80))

            if active or is_shoot:
                cv2.circle(frame, (x, cy), radius + 5, tuple(c // 3 for c in color), 2)
                cv2.circle(frame, (x, cy), radius, color, -1)
                text_color = (10, 10, 10)
            else:
                cv2.circle(frame, (x, cy), radius, color, 2)
                text_color = color

            draw_centered_text_in_rect(frame, label,
                (x - radius, cy - radius, x + radius, cy + radius),
                base_scale=0.80, color=text_color, thickness=2, outline=0)

        draw_centered_text(frame, "4th beat opens SHOOT", y1 + _ix((y2 - y1) * 0.88),
                           0.38, COL_TEXT_DIM, thickness=1, outline=2)

# ============================================================
# RESULT SCREEN
# ============================================================

def draw_result_screen(frame, game_state, colourblind=False):
    layout = _game_layout(frame)
    w, h = layout["w"], layout["h"]
    x1, y1, x2, y2 = layout["result"]

    banner = game_state["result_banner"] if game_state["result_banner"] else game_state["main_text"]
    banner_color = get_result_banner_color(banner, colourblind=colourblind)

    draw_panel(frame, x1, y1, x2, y2, fill=(8, 8, 18), alpha=0.88,
               border=banner_color, border_thickness=2)
    draw_status_chip(frame, banner, y1 + _ix((y2 - y1) * 0.09), banner_color)

    # Reaction time — shown instead of score_text if available
    rxn_ms = game_state.get("reaction_ms")
    if rxn_ms and rxn_ms < 3000:
        rxn_col = COL_GREEN if rxn_ms < 400 else (COL_YELLOW if rxn_ms < 800 else COL_ORANGE)
        draw_centered_text(frame, f"{rxn_ms}ms reaction",
                           y1 + _ix((y2 - y1) * 0.22), 0.50, rxn_col,
                           thickness=1, outline=2)
    elif game_state["score_text"]:
        draw_centered_text(frame, game_state["score_text"],
                           y1 + _ix((y2 - y1) * 0.22), 0.52, COL_TEXT_ACCENT,
                           thickness=2, outline=3)

    # Colourblind: tint the overall panel background strongly
    if colourblind:
        tint_col = _COL_CB_WIN if "WIN" in banner.upper() or "SURVIVE" in banner.upper() \
                   else (_COL_CB_DRAW if "DRAW" in banner.upper() else _COL_CB_LOSE)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), tint_col, -1)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

    # Player and opponent panels.
    left = _fit_rect(x1 + (x2 - x1) * 0.05, y1 + (y2 - y1) * 0.30,
                     x1 + (x2 - x1) * 0.44, y1 + (y2 - y1) * 0.72)
    right = _fit_rect(x1 + (x2 - x1) * 0.56, y1 + (y2 - y1) * 0.30,
                      x1 + (x2 - x1) * 0.95, y1 + (y2 - y1) * 0.72)

    player_color = get_gesture_color(game_state["player_gesture"])
    robot_color  = get_gesture_color(game_state["computer_gesture"])

    draw_panel(frame, left[0], left[1], left[2], left[3],
               fill=(10, 10, 20), alpha=0.88, border=player_color, border_thickness=2)
    draw_panel(frame, right[0], right[1], right[2], right[3],
               fill=(10, 10, 20), alpha=0.88, border=robot_color, border_thickness=2)

    # Labels.
    draw_centered_text_in_rect(frame, "YOU",
        (left[0], left[1] + 6, left[2], left[1] + _ix((left[3] - left[1]) * 0.20)),
        base_scale=0.52, color=COL_TEXT_DIM, thickness=2, outline=3)

    mode_label = game_state.get("play_mode_label", "")
    opponent_display = mode_label[3:] if mode_label.startswith("vs ") else "CPU"
    draw_centered_text_in_rect(frame, opponent_display,
        (right[0], right[1] + 6, right[2], right[1] + _ix((right[3] - right[1]) * 0.20)),
        base_scale=0.52, color=COL_TEXT_DIM, thickness=2, outline=3)

    _draw_gesture_icon(frame, game_state["player_gesture"], left)
    _draw_gesture_icon(frame, game_state["computer_gesture"], right)

    draw_centered_text_in_rect(frame, game_state["player_gesture"],
        (left[0], left[1] + _ix((left[3] - left[1]) * 0.75), left[2], left[3] - 4),
        base_scale=0.60, color=player_color, thickness=2, outline=3)

    draw_centered_text_in_rect(frame, game_state["computer_gesture"],
        (right[0], right[1] + _ix((right[3] - right[1]) * 0.75), right[2], right[3] - 4),
        base_scale=0.60, color=robot_color, thickness=2, outline=3)

    # In colourblind mode: stamp large WIN / LOSE shape indicator below VS
    vs_y = y1 + _ix((y2 - y1) * 0.50)
    if colourblind:
        if "YOU WIN" in banner.upper() or "SURVIVE" in banner.upper():
            stamp, stamp_col = "WIN", _COL_CB_WIN
        elif "DRAW" in banner.upper():
            stamp, stamp_col = "DRAW", _COL_CB_DRAW
        else:
            stamp, stamp_col = "LOSE", _COL_CB_LOSE
        draw_centered_text(frame, stamp, vs_y, 0.72, stamp_col, thickness=3, outline=4)
    else:
        draw_centered_text(frame, "VS", vs_y, 0.65, COL_MAGENTA, thickness=2, outline=3)

    draw_centered_text_in_rect(frame, game_state.get("play_mode_label", ""),
        (x1, y1 + _ix((y2 - y1) * 0.84), x2, y1 + _ix((y2 - y1) * 0.94)),
        base_scale=0.44, color=COL_TEXT_DIM, thickness=1, outline=2)

def draw_session_summary(frame, summary):
    """
    Post-match summary screen shown briefly when MATCH_RESULT resolves.
    summary: dict from fair_play_state game_state["session_summary"]
    """
    layout = _game_layout(frame)
    w, h = layout["w"], layout["h"]
    x1, y1, x2, y2 = layout["result"]

    won       = summary.get("player_won", False)
    ps        = summary.get("player_score", 0)
    rs        = summary.get("robot_score", 0)
    rds       = summary.get("total_rounds", 0)
    avg_rt    = summary.get("avg_reaction_ms")
    top_g     = summary.get("top_gesture", "?")
    opp_type  = summary.get("opponent_type", "")

    header_col = COL_GREEN if won else COL_RED
    header_txt = "MATCH WON!" if won else "MATCH LOST"

    draw_panel(frame, x1, y1, x2, y2, fill=(5, 10, 5) if won else (10, 5, 5),
               alpha=0.92, border=header_col, border_thickness=2)

    draw_centered_text(frame, header_txt,
                       y1 + _ix((y2 - y1) * 0.10), 0.90, header_col,
                       thickness=3, outline=4)
    draw_centered_text(frame, f"{ps}  -  {rs}",
                       y1 + _ix((y2 - y1) * 0.22), 0.80, COL_TEXT_ACCENT,
                       thickness=2, outline=3)

    stats = []
    if avg_rt:
        stats.append(f"Avg reaction:  {avg_rt}ms")
    if top_g and top_g != "?":
        stats.append(f"Favourite throw:  {top_g}")
    if opp_type and opp_type not in ("random", "grace_period", ""):
        label = opp_type.replace("_", " ").replace("heavy", "player").title()
        stats.append(f"You were profiled as:  {label}")
    stats.append(f"Rounds played:  {rds}")

    stat_y = y1 + _ix((y2 - y1) * 0.38)
    for s in stats[:4]:
        draw_centered_text(frame, s, stat_y, 0.44, COL_TEXT_ACCENT, thickness=1, outline=2)
        stat_y += _ix((y2 - y1) * 0.11)

    draw_centered_text(frame, "Returning to menu...",
                       y2 - _ix((y2 - y1) * 0.07), 0.36, COL_TEXT_DIM,
                       thickness=1, outline=2)


def _draw_gesture_icon(frame, gesture, rect):
    """Draw a simple geometric icon for the gesture."""
    x1, y1, x2, y2 = rect
    cx = (x1 + x2) // 2
    cy = y1 + _ix((y2 - y1) * 0.48)
    size = _ix(min(x2 - x1, y2 - y1) * 0.18)
    color = get_gesture_color(gesture)

    if gesture == "Rock":
        cv2.circle(frame, (cx, cy), size, color, -1)
        cv2.circle(frame, (cx, cy), size + 3, tuple(c // 3 for c in color), 2)
    elif gesture == "Paper":
        half = size
        cv2.rectangle(frame, (cx - half, cy - half), (cx + half, cy + half), color, -1)
        cv2.rectangle(frame, (cx - half - 3, cy - half - 3),
                      (cx + half + 3, cy + half + 3), tuple(c // 3 for c in color), 2)
    elif gesture == "Scissors":
        d = size
        cv2.line(frame, (cx - d, cy - d), (cx + d, cy + d), color, 4)
        cv2.line(frame, (cx + d, cy - d), (cx - d, cy + d), color, 4)

def _draw_last_round_replay(frame, player_gesture, robot_gesture, banner):
    """
    Show last round's gestures briefly at the start of the next WAITING_FOR_ROCK.
    Fades in a small 'Last round' summary panel at the top of the hero area.
    """
    layout = _game_layout(frame)
    x1, y1, x2, y2 = layout["result"]
    ph = y2 - y1
    pw = x2 - x1

    draw_panel(frame, x1, y1, x2, y1 + _ix(ph * 0.30),
               fill=(8, 8, 18), alpha=0.80, border=COL_TEXT_DIM, border_thickness=1)
    draw_outlined_text(frame, "LAST ROUND", x1 + _ix(pw * 0.03),
                       y1 + _ix(ph * 0.08), 0.34, COL_TEXT_DIM, thickness=1, outline=2)
    if banner:
        banner_col = get_result_banner_color(banner)
        draw_centered_text(frame, banner, y1 + _ix(ph * 0.08), 0.38, banner_col,
                           thickness=1, outline=2)

    lx = x1 + _ix(pw * 0.20)
    rx = x1 + _ix(pw * 0.60)
    gy = y1 + _ix(ph * 0.20)
    sz = _ix(min(pw, ph) * 0.06)
    draw_gesture_icon(frame, player_gesture, lx, gy, sz)
    draw_gesture_icon(frame, robot_gesture,  rx, gy, sz)
    draw_outlined_text(frame, player_gesture, lx - _ix(pw * 0.06), gy + sz + 8,
                       0.34, get_gesture_color(player_gesture), thickness=1, outline=2)
    draw_outlined_text(frame, robot_gesture,  rx - _ix(pw * 0.05), gy + sz + 8,
                       0.34, get_gesture_color(robot_gesture),  thickness=1, outline=2)
    draw_centered_text(frame, "vs", gy, 0.44, COL_MAGENTA, thickness=1, outline=2)


# ============================================================
# GAME MODE VIEW (main composite)
# ============================================================


# ============================================================
# TWO-PLAYER SCREENS
# ============================================================

def _draw_tp_hand_panel(frame, x1, y1, x2, y2, label, gesture, tracker_state=None,
                        highlight_col=None, result_col=None):
    """
    Draw a single player's panel in the two-player view.
    Shows label, detected gesture icon + text, and optional lock bar.
    """
    col   = highlight_col or COL_CYAN
    fill  = result_col or (10, 14, 24)

    draw_panel(frame, x1, y1, x2, y2, fill=fill, alpha=0.88,
               border=col, border_thickness=2)

    ph = y2 - y1
    pw = x2 - x1

    # Player label
    draw_centered_text_in_rect(frame, label,
        (x1, y1 + _ix(ph * 0.02), x2, y1 + _ix(ph * 0.16)),
        base_scale=0.58, color=col, thickness=2, outline=3)

    # Gesture icon (large)
    icon_rect = (_ix(x1 + pw * 0.15), _ix(y1 + ph * 0.18),
                 _ix(x2 - pw * 0.15), _ix(y1 + ph * 0.70))
    _draw_gesture_icon(frame, gesture, icon_rect)

    # Gesture name
    g_col = get_gesture_color(gesture)
    draw_centered_text_in_rect(frame, gesture,
        (x1, y1 + _ix(ph * 0.72), x2, y1 + _ix(ph * 0.88)),
        base_scale=0.68, color=g_col, thickness=2, outline=3)

    # LOCK confidence bar (right edge of panel)
    if tracker_state:
        streak   = tracker_state.get("stable_streak", 0)
        bar_w    = max(8, _ix(pw * 0.06))
        bar_h    = _ix(ph * 0.50)
        bar_x    = x2 - bar_w - _ix(pw * 0.02)
        bar_top  = y1 + _ix(ph * 0.20)
        bar_bot  = bar_top + bar_h
        pct      = min(1.0, streak / 3)
        cv2.rectangle(frame, (bar_x, bar_top), (bar_x + bar_w, bar_bot), (35, 35, 35), -1)
        cv2.rectangle(frame, (bar_x, bar_top), (bar_x + bar_w, bar_bot), (80, 80, 80), 1)
        if pct > 0:
            fill_y = bar_bot - int(bar_h * pct)
            fc = COL_GREEN if pct >= 1.0 else COL_YELLOW
            cv2.rectangle(frame, (bar_x, fill_y), (bar_x + bar_w, bar_bot), fc, -1)


def draw_two_player_view(frame, game_state,
                         p1_tracker_state=None, p2_tracker_state=None,
                         colourblind=False):
    """
    Render the two-player PvP game screen.
    Layout: [P1 panel] [Centre: state/score/beat] [P2 panel]
    """
    w, h = frame.shape[1], frame.shape[0]
    t    = time.monotonic()

    # Draw a dark overlay
    draw_panel(frame, 0, 0, w - 1, h - 1, fill=COL_BG_DARK, alpha=0.40,
               border=COL_BG_DARK, border_thickness=0)

    cur_state = game_state["state"]
    banner    = game_state.get("result_banner", "")

    # Panel geometry
    panel_w = _ix(w * 0.28)
    p1_x1, p1_x2 = _ix(w * 0.01), _ix(w * 0.01) + panel_w
    p2_x1, p2_x2 = w - _ix(w * 0.01) - panel_w, w - _ix(w * 0.01)
    pan_y1, pan_y2 = _ix(h * 0.08), _ix(h * 0.88)

    # Determine result colouring
    def _res_col(won):
        if won:
            return _COL_CB_WIN if colourblind else COL_GREEN
        return (10, 14, 24)

    p1_won = "PLAYER 1" in banner.upper() and "WIN" in banner.upper()
    p2_won = "PLAYER 2" in banner.upper() and "WIN" in banner.upper()
    p1_res = _res_col(p1_won) if cur_state in ("ROUND_RESULT", "MATCH_RESULT") else (10, 14, 24)
    p2_res = _res_col(p2_won) if cur_state in ("ROUND_RESULT", "MATCH_RESULT") else (10, 14, 24)

    # Draw player panels — show live tracker gesture during gameplay,
    # resolved gesture only during result screens
    def _live_gest(tracker_st, game_key):
        """Return the live confirmed gesture from tracker, falling back to game_state."""
        if cur_state not in ("ROUND_RESULT", "MATCH_RESULT") and tracker_st:
            conf = tracker_st.get("confirmed_gesture", "Unknown")
            stab = tracker_st.get("stable_gesture", "Unknown")
            if conf in ("Rock", "Paper", "Scissors"):
                return conf
            if stab in ("Rock", "Paper", "Scissors"):
                return stab
        return game_state.get(game_key, "Unknown")

    p1_display = _live_gest(p1_tracker_state, "p1_gesture")
    p2_display = _live_gest(p2_tracker_state, "p2_gesture")

    _draw_tp_hand_panel(frame, p1_x1, pan_y1, p1_x2, pan_y2,
                        "PLAYER 1", p1_display,
                        p1_tracker_state, COL_CYAN, p1_res)
    _draw_tp_hand_panel(frame, p2_x1, pan_y1, p2_x2, pan_y2,
                        "PLAYER 2", p2_display,
                        p2_tracker_state, COL_MAGENTA, p2_res)

    # Centre column
    cx1, cx2 = p1_x2 + _ix(w * 0.01), p2_x1 - _ix(w * 0.01)
    draw_panel(frame, cx1, pan_y1, cx2, pan_y2, fill=(6, 8, 18), alpha=0.88,
               border=COL_YELLOW, border_thickness=2)

    cy_mid   = (pan_y1 + pan_y2) // 2
    cent_ph  = pan_y2 - pan_y1

    # State chip
    chip_y = pan_y1 + _ix(cent_ph * 0.06)
    draw_status_chip(frame, cur_state.replace("_", " "), chip_y, COL_YELLOW)

    # Main text (countdown number, SHOOT, etc.)
    main_text = game_state.get("main_text", "")
    if cur_state == "COUNTDOWN" and main_text not in ("READY", ""):
        pulse = 0.72 + 0.28 * abs(math.sin(t * math.pi * 1.4))
        num_col = tuple(min(255, int(c * pulse)) for c in COL_CYAN)
        draw_centered_text_in_rect(frame, main_text,
            (cx1, pan_y1 + _ix(cent_ph * 0.18), cx2, pan_y1 + _ix(cent_ph * 0.58)),
            base_scale=2.2, color=num_col, thickness=4, outline=6)
    elif cur_state in ("ROUND_RESULT", "MATCH_RESULT"):
        b_col = get_result_banner_color(banner, colourblind=colourblind)
        draw_centered_text_in_rect(frame, banner,
            (cx1 + 4, pan_y1 + _ix(cent_ph * 0.18), cx2 - 4, pan_y1 + _ix(cent_ph * 0.52)),
            base_scale=0.60, color=b_col, thickness=2, outline=3)
    else:
        draw_centered_text_in_rect(frame, main_text,
            (cx1, pan_y1 + _ix(cent_ph * 0.18), cx2, pan_y1 + _ix(cent_ph * 0.52)),
            base_scale=0.72, color=COL_CYAN, thickness=2, outline=3)

    # Score
    draw_centered_text(frame, game_state.get("score_text", ""),
                       pan_y1 + _ix(cent_ph * 0.60), 0.54, COL_TEXT_ACCENT,
                       thickness=2, outline=3)

    # Beat track (mini)
    if cur_state in ("COUNTDOWN", "WAITING_FOR_ROCK", "SHOOT_WINDOW"):
        beat = game_state.get("beat_count", 0)
        dot_y = pan_y1 + _ix(cent_ph * 0.76)
        dot_spacing = _ix((cx2 - cx1) * 0.20)
        dot_start = cx1 + _ix((cx2 - cx1) * 0.10)
        for bi in range(4):
            bx = dot_start + bi * dot_spacing
            active = bi < beat or (bi == 3 and cur_state == "SHOOT_WINDOW")
            col = COL_RED if (bi == 3 and cur_state == "SHOOT_WINDOW") else \
                  (COL_CYAN if active else (50, 50, 70))
            cv2.circle(frame, (bx, dot_y), _ix(w * 0.012),
                       col, -1 if active else 2)

    # Sub text
    draw_centered_text(frame, game_state.get("sub_text", ""),
                       pan_y1 + _ix(cent_ph * 0.90), 0.34, COL_TEXT_DIM,
                       thickness=1, outline=2)

    # Top bar
    draw_top_bar(frame, "2 PLAYER PvP",
                 f"Round {game_state.get('round_number', 1)} | ESC Back | Q Quit")
    draw_bottom_bar(frame, "Both players: make a FIST then pump 4x together | Q Quit")


def draw_pvpvai_view(frame, game_state,
                     p1_tracker_state=None, p2_tracker_state=None,
                     colourblind=False):
    """
    1v1v1: Player 1 vs Player 2 vs AI — everyone for themselves.
    Beat 1 opponent = +1 pt, beat 2 = +2 pts, 3-way draw = +0.
    First to win_target wins.
    Layout: [P1 panel] [Centre: AI + countdown/result] [P2 panel]
    """
    w, h = frame.shape[1], frame.shape[0]
    t    = time.monotonic()

    draw_panel(frame, 0, 0, w - 1, h - 1, fill=COL_BG_DARK, alpha=0.40,
               border=COL_BG_DARK, border_thickness=0)

    cur_state = game_state["state"]
    banner    = game_state.get("result_banner", "")
    p1_pts    = game_state.get("p1_pts_this_round", 0)
    p2_pts    = game_state.get("p2_pts_this_round", 0)
    ai_pts    = game_state.get("ai_pts_this_round", 0)
    win_target = game_state.get("win_target", 5)

    panel_w = _ix(w * 0.26)
    p1_x1, p1_x2 = _ix(w * 0.01), _ix(w * 0.01) + panel_w
    p2_x1, p2_x2 = w - _ix(w * 0.01) - panel_w, w - _ix(w * 0.01)
    pan_y1, pan_y2 = _ix(h * 0.08), _ix(h * 0.88)
    cent_ph = pan_y2 - pan_y1

    # Per-player result tint
    def _panel_fill(pts_this_round, score_key):
        if cur_state not in ("ROUND_RESULT", "MATCH_RESULT"):
            return (10, 14, 24)
        if pts_this_round == 2:
            return _COL_CB_WIN if colourblind else (10, 55, 10)
        if pts_this_round == 1:
            return (10, 35, 10)
        return (10, 14, 24)

    def _live_g(tracker_st, game_key):
        if cur_state not in ("ROUND_RESULT", "MATCH_RESULT") and tracker_st:
            conf = tracker_st.get("confirmed_gesture", "Unknown")
            stab = tracker_st.get("stable_gesture",   "Unknown")
            if conf in ("Rock", "Paper", "Scissors"):
                return conf
            if stab in ("Rock", "Paper", "Scissors"):
                return stab
        return game_state.get(game_key, "Unknown")

    _draw_tp_hand_panel(frame, p1_x1, pan_y1, p1_x2, pan_y2,
                        "PLAYER 1", _live_g(p1_tracker_state, "p1_gesture"),
                        p1_tracker_state, COL_CYAN, _panel_fill(p1_pts, "p1_score"))
    _draw_tp_hand_panel(frame, p2_x1, pan_y1, p2_x2, pan_y2,
                        "PLAYER 2", _live_g(p2_tracker_state, "p2_gesture"),
                        p2_tracker_state, COL_MAGENTA, _panel_fill(p2_pts, "p2_score"))

    # +pts flash on player panels during result
    if cur_state in ("ROUND_RESULT", "MATCH_RESULT"):
        for px, pts, col in [
            ((p1_x1 + p1_x2) // 2, p1_pts, COL_CYAN),
            ((p2_x1 + p2_x2) // 2, p2_pts, COL_MAGENTA),
        ]:
            if pts > 0:
                draw_centered_text(frame, f"+{pts}", pan_y1 + _ix(cent_ph * 0.88),
                                   0.70, col, thickness=2, outline=3)

    # Centre AI panel
    cx1, cx2 = p1_x2 + _ix(w * 0.01), p2_x1 - _ix(w * 0.01)
    ai_fill = _panel_fill(ai_pts, "ai_score") if cur_state in ("ROUND_RESULT","MATCH_RESULT") \
              else (6, 8, 18)
    draw_panel(frame, cx1, pan_y1, cx2, pan_y2,
               fill=ai_fill, alpha=0.88, border=COL_RED, border_thickness=2)

    draw_status_chip(frame, "AI", pan_y1 + _ix(cent_ph * 0.04), COL_RED)

    ai_g = game_state.get("ai_gesture", "Unknown")

    if cur_state in ("ROUND_RESULT", "MATCH_RESULT"):
        # Show AI gesture
        if ai_g in ("Rock", "Paper", "Scissors"):
            ai_icon = (_ix(cx1 + (cx2-cx1)*0.10), _ix(pan_y1 + cent_ph*0.14),
                       _ix(cx2 - (cx2-cx1)*0.10), _ix(pan_y1 + cent_ph*0.52))
            _draw_gesture_icon(frame, ai_g, ai_icon)
            draw_centered_text_in_rect(frame, ai_g,
                (cx1, pan_y1 + _ix(cent_ph*0.54), cx2, pan_y1 + _ix(cent_ph*0.65)),
                base_scale=0.50, color=get_gesture_color(ai_g), thickness=2, outline=3)
            if ai_pts > 0:
                draw_centered_text(frame, f"+{ai_pts}",
                                   pan_y1 + _ix(cent_ph * 0.88), 0.70,
                                   COL_RED, thickness=2, outline=3)

        # Result banner across full width below panels
        b_col = COL_YELLOW if "DRAW" in banner.upper() else \
                (COL_GREEN if "PLAYER" in banner.upper() else COL_RED)
        draw_centered_text(frame, banner,
                           pan_y2 + _ix(h * 0.005), 0.48, b_col,
                           thickness=2, outline=3)
    else:
        # Countdown / waiting
        main_text = game_state.get("main_text", "")
        if cur_state == "COUNTDOWN" and main_text not in ("READY", ""):
            pulse = 0.72 + 0.28 * abs(math.sin(t * math.pi * 1.4))
            num_col = tuple(min(255, int(c * pulse)) for c in COL_CYAN)
            draw_centered_text_in_rect(frame, main_text,
                (cx1, pan_y1 + _ix(cent_ph*0.22), cx2, pan_y1 + _ix(cent_ph*0.62)),
                base_scale=2.0, color=num_col, thickness=4, outline=6)
        else:
            draw_centered_text_in_rect(frame, main_text,
                (cx1, pan_y1 + _ix(cent_ph*0.22), cx2, pan_y1 + _ix(cent_ph*0.58)),
                base_scale=0.68, color=COL_CYAN, thickness=2, outline=3)
        # Beat dots
        beat = game_state.get("beat_count", 0)
        dot_y = pan_y1 + _ix(cent_ph * 0.76)
        dot_sp = _ix((cx2-cx1) * 0.20)
        dot_x0 = cx1 + _ix((cx2-cx1) * 0.10)
        for bi in range(4):
            bx = dot_x0 + bi * dot_sp
            active = bi < beat or (bi == 3 and cur_state == "SHOOT_WINDOW")
            col = COL_RED if (bi == 3 and cur_state == "SHOOT_WINDOW") else \
                  (COL_CYAN if active else (50, 50, 70))
            cv2.circle(frame, (bx, dot_y), _ix(w * 0.012), col, -1 if active else 2)
        draw_centered_text(frame, game_state.get("sub_text", ""),
                           pan_y1 + _ix(cent_ph * 0.90), 0.34, COL_TEXT_DIM,
                           thickness=1, outline=2)

    # 3-score strip at bottom, with progress bar per player
    score_y = pan_y2 + _ix(h * 0.005)
    for label, score, col, sx in [
        ("P1", game_state.get("p1_score", 0), COL_CYAN,    _ix(w * 0.08)),
        ("AI", game_state.get("ai_score", 0), COL_RED,     _ix(w * 0.44)),
        ("P2", game_state.get("p2_score", 0), COL_MAGENTA, _ix(w * 0.80)),
    ]:
        bar_w = _ix(w * 0.14)
        bar_h = _ix(h * 0.018)
        pct   = min(1.0, score / max(win_target, 1))
        cv2.rectangle(frame, (sx, score_y), (sx + bar_w, score_y + bar_h), (30,30,30), -1)
        if pct > 0:
            cv2.rectangle(frame, (sx, score_y),
                          (sx + int(bar_w * pct), score_y + bar_h), col, -1)
        cv2.rectangle(frame, (sx, score_y), (sx + bar_w, score_y + bar_h), (80,80,80), 1)
        draw_outlined_text(frame, f"{label}: {score}/{win_target}",
                           sx, score_y - 4, 0.34, col, thickness=1, outline=2)

    draw_top_bar(frame, "PvPvAI  (1v1v1)",
                 f"Round {game_state.get('round_number',1)} | Beat 1=+1pt  Beat 2=+2pts | ESC")
    draw_bottom_bar(frame, "Fist + pump 4x | Beat opponents to score | First to 5 wins | Q Quit")




# ============================================================
# AI PERSONALITY SETTINGS SCREEN
# ============================================================

def draw_personality_settings(frame, selected_name, descriptions):
    """
    Full-screen AI Personality selector.
    selected_name: currently highlighted personality key.
    descriptions: list of (name, desc) tuples.
    """
    from fair_play_ai import PERSONALITIES, PERSONALITY_NAMES
    w, h = frame.shape[1], frame.shape[0]
    draw_panel(frame, 0, 0, w - 1, h - 1, fill=(6, 8, 18), alpha=0.95,
               border=COL_MAGENTA, border_thickness=2)

    draw_top_bar(frame, "AI PERSONALITIES", "W/S Select  |  Enter Confirm  |  ESC Back")

    # Layout: list of personalities on left, description panel on right
    list_x1, list_x2 = _ix(w * 0.03), _ix(w * 0.52)
    desc_x1, desc_x2 = _ix(w * 0.55), _ix(w * 0.97)
    y_start = _ix(h * 0.12)
    item_h  = _ix(h * 0.10)

    for i, name in enumerate(PERSONALITY_NAMES):
        p    = PERSONALITIES[name]
        iy1  = y_start + i * item_h
        iy2  = iy1 + item_h - _ix(h * 0.008)
        is_sel = (name == selected_name)
        border_col = COL_MAGENTA if is_sel else (40, 40, 60)
        fill_col   = (18, 8, 28) if is_sel else (8, 8, 16)
        draw_panel(frame, list_x1, iy1, list_x2, iy2,
                   fill=fill_col, alpha=0.90, border=border_col, border_thickness=2)
        label_col = COL_MAGENTA if is_sel else COL_TEXT_ACCENT
        draw_centered_text_in_rect(frame, p["label"],
            (list_x1, iy1, list_x2, iy2),
            base_scale=0.52, color=label_col, thickness=2, outline=3)
        if is_sel:
            cv2.rectangle(frame, (list_x1 + 2, iy1 + 2), (list_x1 + 8, iy2 - 2),
                          COL_MAGENTA, -1)

    # Description panel for selected
    sel_p  = PERSONALITIES.get(selected_name, {})
    draw_panel(frame, desc_x1, y_start, desc_x2, y_start + len(PERSONALITY_NAMES) * item_h,
               fill=(8, 5, 18), alpha=0.90, border=COL_CYAN, border_thickness=2)
    ph = len(PERSONALITY_NAMES) * item_h
    draw_centered_text_in_rect(frame, sel_p.get("label", ""),
        (desc_x1, y_start, desc_x2, y_start + _ix(ph * 0.18)),
        base_scale=0.58, color=COL_CYAN, thickness=2, outline=3)
    # Wrap description text manually (simple word wrap)
    desc_text = sel_p.get("desc", "")
    words = desc_text.split()
    lines = []
    cur = ""
    for word in words:
        test = cur + (" " if cur else "") + word
        if len(test) > 28:
            if cur:
                lines.append(cur)
            cur = word
        else:
            cur = test
    if cur:
        lines.append(cur)
    for li, line in enumerate(lines[:4]):
        ty = y_start + _ix(ph * 0.22) + li * _ix(ph * 0.12)
        draw_centered_text(frame, line, ty, 0.36, COL_TEXT_ACCENT, thickness=1, outline=2)

    draw_bottom_bar(frame, "Default: Normal  |  Resets to Normal on app restart  |  ESC Back")


# ============================================================
# SPEED REFLEX SCREENS
# ============================================================

def _draw_reflex_target(frame, target, cx, cy, radius, flash=False):
    """Draw the gesture target circle with icon and name."""
    col = get_gesture_color(target) if target else COL_TEXT_DIM
    if flash:
        col = COL_GREEN
    cv2.circle(frame, (cx, cy), radius, col, 3)
    cv2.circle(frame, (cx, cy), radius - 4, (10, 12, 20), -1)
    icon_rect = (cx - radius + 10, cy - radius + 10,
                 cx + radius - 10, cy + radius - 10)
    _draw_gesture_icon(frame, target or "Unknown", icon_rect)
    if target:
        draw_centered_text(frame, target, cy + radius + _ix(0.04 * frame.shape[0]),
                           0.50, col, thickness=2, outline=3)


def draw_reflex_solo_view(frame, game_state):
    w, h = frame.shape[1], frame.shape[0]
    t    = time.monotonic()
    draw_panel(frame, 0, 0, w - 1, h - 1, fill=COL_BG_DARK, alpha=0.45,
               border=COL_BG_DARK, border_thickness=0)

    cur_state  = game_state["state"]
    target     = game_state.get("target", "")
    score      = game_state.get("score", 0)
    time_left  = game_state.get("time_left", 30.0)
    last_res   = game_state.get("last_result", "")
    avg_rt     = game_state.get("avg_reaction_ms", 0)

    draw_top_bar(frame, "SPEED REFLEX  —  SOLO",
                 f"Match the gesture! Score: {score}  |  Q Quit")

    if cur_state == "INTRO":
        draw_centered_text(frame, "SPEED REFLEX", h // 2 - _ix(h * 0.10),
                           1.0, COL_YELLOW, thickness=3, outline=5)
        draw_centered_text(frame, "Match each gesture as fast as you can!",
                           h // 2 + _ix(h * 0.02), 0.44, COL_TEXT_ACCENT,
                           thickness=1, outline=2)
        draw_centered_text(frame, "30 seconds  —  No penalty for misses",
                           h // 2 + _ix(h * 0.09), 0.38, COL_TEXT_DIM,
                           thickness=1, outline=2)

    elif cur_state == "GAME_OVER":
        draw_centered_text(frame, "TIME'S UP!", h // 2 - _ix(h * 0.15),
                           1.1, COL_RED, thickness=3, outline=5)
        draw_centered_text(frame, f"Final Score: {score}", h // 2,
                           0.80, COL_YELLOW, thickness=2, outline=4)
        draw_centered_text(frame, f"Avg Reaction: {avg_rt}ms",
                           h // 2 + _ix(h * 0.12), 0.50, COL_CYAN, thickness=2, outline=3)

    else:
        # Main target circle
        cx, cy = w // 2, h // 2
        radius = _ix(min(w, h) * 0.18)
        flash  = (last_res == "hit" and cur_state == "RESULT_FLASH")
        _draw_reflex_target(frame, target, cx, cy, radius, flash=flash)

        # Result feedback
        if cur_state == "RESULT_FLASH":
            res_col = COL_GREEN if last_res == "hit" else (COL_ORANGE if last_res == "timeout" else COL_RED)
            res_txt = "✓ HIT!" if last_res == "hit" else ("TIME!" if last_res == "timeout" else "MISS")
            draw_centered_text(frame, res_txt, cy - radius - _ix(h * 0.08),
                               0.70, res_col, thickness=2, outline=3)

        # Timer bar at top
        pct = time_left / 30.0
        bar_x1 = _ix(w * 0.05); bar_x2 = _ix(w * 0.95)
        bar_y  = _ix(h * 0.08)
        bar_h2 = _ix(h * 0.014)
        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + bar_h2), (30, 30, 30), -1)
        col_t = COL_GREEN if pct > 0.5 else (COL_YELLOW if pct > 0.25 else COL_RED)
        cv2.rectangle(frame, (bar_x1, bar_y),
                      (bar_x1 + int((bar_x2 - bar_x1) * pct), bar_y + bar_h2), col_t, -1)
        draw_outlined_text(frame, f"{time_left:.1f}s",
                           bar_x1, bar_y - 4, 0.36, COL_TEXT_DIM, thickness=1, outline=1)

    draw_bottom_bar(frame, f"Score: {score}  |  Avg RT: {avg_rt}ms  |  Q Quit")


def draw_reflex_two_player_view(frame, game_state,
                                p1_tracker_state=None, p2_tracker_state=None):
    w, h = frame.shape[1], frame.shape[0]
    draw_panel(frame, 0, 0, w - 1, h - 1, fill=COL_BG_DARK, alpha=0.45,
               border=COL_BG_DARK, border_thickness=0)

    cur_state  = game_state["state"]
    target     = game_state.get("target", "")
    p1_score   = game_state.get("p1_score", 0)
    p2_score   = game_state.get("p2_score", 0)
    win_target = game_state.get("win_target", 10)
    last_win   = game_state.get("last_winner", "")
    match_win  = game_state.get("match_winner", "")
    time_left  = game_state.get("time_left", 0.0)

    draw_top_bar(frame, "SPEED REFLEX  —  2 PLAYER",
                 f"P1 {p1_score}  |  P2 {p2_score}  |  First to {win_target}  |  Q Quit")

    cx, cy   = w // 2, int(h * 0.48)
    radius   = _ix(min(w, h) * 0.17)
    p1_flash = (last_win == "P1" and cur_state == "RESULT_FLASH")
    p2_flash = (last_win == "P2" and cur_state == "RESULT_FLASH")

    if cur_state == "MATCH_OVER":
        draw_centered_text(frame, match_win, h // 2 - _ix(h * 0.08),
                           0.80, COL_GREEN, thickness=2, outline=4)
        draw_centered_text(frame, f"P1: {p1_score}  |  P2: {p2_score}",
                           h // 2 + _ix(h * 0.06), 0.50, COL_TEXT_ACCENT,
                           thickness=1, outline=3)
    elif cur_state == "INTRO":
        draw_centered_text(frame, "REFLEX RACE", cy - _ix(h * 0.08),
                           0.90, COL_YELLOW, thickness=2, outline=4)
        draw_centered_text(frame, "Match the gesture first to score!",
                           cy + _ix(h * 0.06), 0.40, COL_TEXT_DIM,
                           thickness=1, outline=2)
    else:
        _draw_reflex_target(frame, target, cx, cy, radius)
        # Winner flash
        if cur_state == "RESULT_FLASH" and last_win in ("P1", "P2"):
            col_w = COL_CYAN if last_win == "P1" else COL_MAGENTA
            draw_centered_text(frame, f"{last_win} GOT IT!",
                               cy - radius - _ix(h * 0.06), 0.60, col_w,
                               thickness=2, outline=3)
        # Timeout
        if cur_state == "RESULT_FLASH" and last_win == "timeout":
            draw_centered_text(frame, "TOO SLOW!", cy - radius - _ix(h * 0.06),
                               0.60, COL_RED, thickness=2, outline=3)
        # Timer bar
        pct = time_left / 2.5
        bar_x1, bar_x2 = _ix(w * 0.2), _ix(w * 0.8)
        bar_y = cy + radius + _ix(h * 0.08)
        bh    = _ix(h * 0.012)
        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + bh), (30, 30, 30), -1)
        fc = COL_GREEN if pct > 0.5 else (COL_YELLOW if pct > 0.2 else COL_RED)
        cv2.rectangle(frame, (bar_x1, bar_y),
                      (bar_x1 + int((bar_x2 - bar_x1) * pct), bar_y + bh), fc, -1)

    # Score bars
    for sx, score, col, lbl in [
        (_ix(w * 0.03), p1_score, COL_CYAN, "P1"),
        (_ix(w * 0.80), p2_score, COL_MAGENTA, "P2"),
    ]:
        pct = min(1.0, score / max(win_target, 1))
        bw  = _ix(w * 0.14)
        bh2 = _ix(h * 0.60)
        by1 = int(h * 0.20)
        cv2.rectangle(frame, (sx, by1), (sx + bw, by1 + bh2), (30, 30, 30), -1)
        cv2.rectangle(frame, (sx, by1), (sx + bw, by1 + bh2), (60, 60, 60), 1)
        if pct > 0:
            fy = by1 + bh2 - int(bh2 * pct)
            cv2.rectangle(frame, (sx, fy), (sx + bw, by1 + bh2), col, -1)
        draw_centered_text(frame, f"{lbl}: {score}/{win_target}",
                           by1 + bh2 + _ix(h * 0.02), 0.36, col,
                           thickness=1, outline=2)

    draw_bottom_bar(frame, "P1 = Left hand  |  P2 = Right hand  |  First to match wins the point")


# ============================================================
# BLUFF MODE SCREEN
# ============================================================

def draw_bluff_mode_view(frame, game_state, tracker_state=None, hand_state=None,
                         flash_info=None):
    w, h = frame.shape[1], frame.shape[0]
    t    = time.monotonic()
    draw_panel(frame, 0, 0, w - 1, h - 1, fill=COL_BG_DARK, alpha=0.40,
               border=COL_BG_DARK, border_thickness=0)

    cur_state    = game_state["state"]
    declared     = game_state.get("ai_declared_move", "?")
    actual       = game_state.get("ai_actual_move",   "?")
    score_text   = game_state.get("score_text", "")
    banner       = game_state.get("result_banner", "")
    p_gest       = game_state.get("player_gesture", "Unknown")
    beat_count   = game_state.get("beat_count", 0)
    time_left    = game_state.get("time_left", 0.0)
    bluff_pct    = game_state.get("bluff_pct_so_far", 0.0)
    round_text   = game_state.get("round_text", "")

    draw_top_bar(frame, "BLUFF MODE",
                 f"{round_text}  |  AI bluffs ~{game_state.get('bluff_rate',0.55)*100:.0f}%  |  Q Quit")

    # ── Centre: AI declaration panel ──────────────────────────────────────
    panel_w = _ix(w * 0.44)
    cx1     = (w - panel_w) // 2
    cx2     = cx1 + panel_w
    pan_y1  = _ix(h * 0.10)
    pan_y2  = _ix(h * 0.78)
    ph      = pan_y2 - pan_y1

    # Pulse border when countdown active
    if cur_state in ("COUNTDOWN", "WAITING_FOR_ROCK"):
        pulse = 0.5 + 0.5 * abs(math.sin(t * math.pi * 1.2))
        bcol  = tuple(min(255, int(c * pulse)) for c in COL_YELLOW)
    else:
        bcol = COL_YELLOW

    draw_panel(frame, cx1, pan_y1, cx2, pan_y2,
               fill=(6, 8, 20), alpha=0.90, border=bcol, border_thickness=2)

    # "AI DECLARES" label
    draw_status_chip(frame, "AI DECLARES", pan_y1 + _ix(ph * 0.04), COL_YELLOW)

    # Show declaration from COUNTDOWN onward
    show_decl = cur_state in ("COUNTDOWN", "SHOOT_WINDOW", "ROUND_RESULT", "MATCH_RESULT")
    if show_decl and declared and declared != "?":
        icon_rect = (cx1 + _ix(panel_w * 0.10),
                     pan_y1 + _ix(ph * 0.14),
                     cx2 - _ix(panel_w * 0.10),
                     pan_y1 + _ix(ph * 0.58))
        _draw_gesture_icon(frame, declared, icon_rect)
        decl_col = get_gesture_color(declared)
        draw_centered_text_in_rect(frame, declared,
            (cx1, pan_y1 + _ix(ph * 0.60), cx2, pan_y1 + _ix(ph * 0.73)),
            base_scale=0.60, color=decl_col, thickness=2, outline=3)

        # After round: reveal truth
        if cur_state in ("ROUND_RESULT", "MATCH_RESULT"):
            was_bluff = game_state.get("is_bluff", False)
            truth_txt = "BLUFF! AI played " + actual if was_bluff else "TRUTHFUL"
            truth_col = COL_RED if was_bluff else COL_GREEN
            draw_centered_text_in_rect(frame, truth_txt,
                (cx1 + 4, pan_y1 + _ix(ph * 0.74), cx2 - 4, pan_y1 + _ix(ph * 0.87)),
                base_scale=0.42, color=truth_col, thickness=2, outline=3)
    else:
        # Before declaration: beat countdown in panel
        main_text = "MAKE A FIST" if cur_state == "WAITING_FOR_ROCK" else \
                    ("READY" if beat_count == 0 else str(min(beat_count, 3)))
        pulse = 0.72 + 0.28 * abs(math.sin(t * math.pi * 1.4))
        num_col = tuple(min(255, int(c * pulse)) for c in COL_CYAN) \
                  if cur_state == "COUNTDOWN" and main_text not in ("MAKE A FIST", "READY") \
                  else COL_CYAN
        draw_centered_text_in_rect(frame, main_text,
            (cx1, pan_y1 + _ix(ph * 0.25), cx2, pan_y1 + _ix(ph * 0.65)),
            base_scale=0.72 if cur_state != "COUNTDOWN" else 2.0,
            color=num_col, thickness=2, outline=4)

    # Beat dots
    if cur_state in ("COUNTDOWN", "WAITING_FOR_ROCK", "SHOOT_WINDOW"):
        dot_y    = pan_y2 + _ix(h * 0.03)
        dot_sp   = _ix(w * 0.06)
        dot_x0   = w // 2 - dot_sp * 2
        for bi in range(4):
            bx  = dot_x0 + bi * dot_sp
            act = bi < beat_count or (bi == 3 and cur_state == "SHOOT_WINDOW")
            col = COL_RED if (bi == 3 and cur_state == "SHOOT_WINDOW") else \
                  (COL_CYAN if act else (50, 50, 70))
            cv2.circle(frame, (bx, dot_y), _ix(w * 0.012), col, -1 if act else 2)

    # ── Result banner full-width ──────────────────────────────────────────
    if cur_state in ("ROUND_RESULT", "MATCH_RESULT"):
        b_col = get_result_banner_color(banner)
        draw_centered_text(frame, banner, pan_y2 + _ix(h * 0.10),
                           0.62, b_col, thickness=2, outline=3)

    # ── Score strip ──────────────────────────────────────────────────────
    draw_centered_text(frame, score_text, pan_y2 + _ix(h * 0.19),
                       0.50, COL_TEXT_ACCENT, thickness=2, outline=3)

    # ── Research stat: bluff rate so far ─────────────────────────────────
    research_txt = f"Bluff rate this session: {bluff_pct*100:.0f}%"
    draw_outlined_text(frame, research_txt,
                       _ix(w * 0.03), h - _ix(h * 0.06),
                       0.32, COL_TEXT_DIM, thickness=1, outline=1)

    draw_bottom_bar(frame, "AI may lie about its move  |  Normal RPS rules apply  |  Q Quit")



# ============================================================
# SIMON SAYS SCREENS
# ============================================================

_GESTURE_COLORS = {"Rock": COL_CYAN, "Paper": COL_GREEN, "Scissors": COL_MAGENTA}
_GESTURE_SHORT  = {"Rock": "R", "Paper": "P", "Scissors": "S"}


def _draw_simon_sequence_bar(frame, sequence, step_index, show_step, showing_seq, w, h):
    """Horizontal sequence bar showing all gestures, highlighting current."""
    n   = len(sequence)
    if n == 0:
        return
    bar_y   = _ix(h * 0.82)
    cell_w  = min(_ix(w * 0.08), _ix(w * 0.85 / n))
    total_w = cell_w * n
    start_x = (w - total_w) // 2

    for i, g in enumerate(sequence):
        cx    = start_x + i * cell_w + cell_w // 2
        cy    = bar_y
        col   = _GESTURE_COLORS.get(g, COL_TEXT_DIM)
        r     = _ix(cell_w * 0.38)
        # Current step highlight
        if showing_seq and i == show_step:
            # Currently being shown
            cv2.circle(frame, (cx, cy), r + 4, col, -1)
            draw_outlined_text(frame, _GESTURE_SHORT.get(g, "?"),
                               cx - _ix(r * 0.4), cy + _ix(r * 0.35),
                               0.50, (0, 0, 0), thickness=2, outline=0)
        elif not showing_seq and i < step_index:
            # Already done
            cv2.circle(frame, (cx, cy), r, col, -1)
            draw_outlined_text(frame, _GESTURE_SHORT.get(g, "?"),
                               cx - _ix(r * 0.4), cy + _ix(r * 0.35),
                               0.40, (0, 0, 0), thickness=1, outline=0)
        elif not showing_seq and i == step_index:
            # Current target
            pulse = 0.6 + 0.4 * abs(math.sin(time.monotonic() * math.pi * 2))
            pc    = tuple(min(255, int(c * pulse)) for c in col)
            cv2.circle(frame, (cx, cy), r + 2, pc, 3)
            draw_outlined_text(frame, "?", cx - _ix(r * 0.25), cy + _ix(r * 0.35),
                               0.45, pc, thickness=2, outline=3)
        else:
            cv2.circle(frame, (cx, cy), r, (40, 40, 50), -1)
            cv2.circle(frame, (cx, cy), r, (70, 70, 80), 1)


def draw_simon_says_solo_view(frame, game_state):
    w, h = frame.shape[1], frame.shape[0]
    draw_panel(frame, 0, 0, w - 1, h - 1, fill=COL_BG_DARK, alpha=0.50,
               border=COL_BG_DARK, border_thickness=0)

    showing_seq  = game_state.get("showing_sequence", False)
    show_step    = game_state.get("show_step", 0)
    sequence     = game_state.get("sequence", [])
    step_index   = game_state.get("step_index", 0)
    time_left    = game_state.get("time_left", 2.0)
    target       = game_state.get("current_target", "")
    score        = game_state.get("score", 0)
    game_over    = game_state.get("game_over", False)
    last_result  = game_state.get("last_result", "")
    seq_len      = len(sequence)
    t            = time.monotonic()

    draw_top_bar(frame, "SIMON SAYS",
                 f"Sequence: {seq_len}  |  Score: {score}  |  Q Quit")

    if game_over:
        draw_centered_text(frame, "GAME OVER", h // 2 - _ix(h * 0.14),
                           1.0, COL_RED, thickness=3, outline=5)
        draw_centered_text(frame, game_state.get("game_over_text", ""),
                           h // 2 + _ix(h * 0.02), 0.38, COL_TEXT_ACCENT,
                           thickness=1, outline=2)
        draw_centered_text(frame, f"Press Q to quit  |  ESC for menu",
                           h // 2 + _ix(h * 0.12), 0.34, COL_TEXT_DIM,
                           thickness=1, outline=2)
        return

    cy_mid = h // 2

    if showing_seq:
        # Showing the sequence — display each gesture big
        cur_g = sequence[show_step] if show_step < len(sequence) else ""
        draw_status_chip(frame, "WATCH CAREFULLY", _ix(h * 0.10), COL_YELLOW)
        if cur_g:
            icon_rect = (_ix(w * 0.30), _ix(h * 0.18),
                         _ix(w * 0.70), _ix(h * 0.68))
            _draw_gesture_icon(frame, cur_g, icon_rect)
            col = _GESTURE_COLORS.get(cur_g, COL_TEXT_ACCENT)
            draw_centered_text(frame, cur_g, _ix(h * 0.72), 0.68, col,
                               thickness=2, outline=3)
        draw_centered_text(frame, f"Step {show_step + 1} of {seq_len}",
                           _ix(h * 0.78), 0.36, COL_TEXT_DIM,
                           thickness=1, outline=2)
    elif last_result == "correct" and not game_over:
        # Success flash
        draw_centered_text(frame, "✓ CORRECT!", cy_mid - _ix(h * 0.06),
                           0.90, COL_GREEN, thickness=2, outline=4)
        draw_centered_text(frame, f"Next: {seq_len + 1} gestures",
                           cy_mid + _ix(h * 0.07), 0.44, COL_TEXT_DIM,
                           thickness=1, outline=2)
    else:
        # Player input phase
        draw_status_chip(frame, f"YOUR TURN — Step {step_index + 1} / {seq_len}",
                         _ix(h * 0.10), COL_CYAN)

        # Big target
        if target:
            icon_rect = (_ix(w * 0.32), _ix(h * 0.18),
                         _ix(w * 0.68), _ix(h * 0.62))
            _draw_gesture_icon(frame, target, icon_rect)
            col = _GESTURE_COLORS.get(target, COL_TEXT_ACCENT)
            draw_centered_text(frame, target, _ix(h * 0.65), 0.68, col,
                               thickness=2, outline=3)

        # Timer bar
        pct   = time_left / 2.0
        bx1   = _ix(w * 0.20); bx2 = _ix(w * 0.80)
        by    = _ix(h * 0.74)
        bh    = _ix(h * 0.018)
        cv2.rectangle(frame, (bx1, by), (bx2, by + bh), (30, 30, 30), -1)
        tc = COL_GREEN if pct > 0.5 else (COL_YELLOW if pct > 0.25 else COL_RED)
        cv2.rectangle(frame, (bx1, by),
                      (bx1 + int((bx2 - bx1) * pct), by + bh), tc, -1)

        if last_result == "wrong":
            draw_centered_text(frame, "WRONG!", _ix(h * 0.74),
                               0.60, COL_RED, thickness=2, outline=3)

    _draw_simon_sequence_bar(frame, sequence, step_index, show_step, showing_seq, w, h)
    draw_bottom_bar(frame, "Match each gesture within 2 seconds  |  Q Quit")


def draw_simon_says_two_player_view(frame, game_state,
                                    p1_tracker_state=None, p2_tracker_state=None):
    w, h = frame.shape[1], frame.shape[0]
    draw_panel(frame, 0, 0, w - 1, h - 1, fill=COL_BG_DARK, alpha=0.50,
               border=COL_BG_DARK, border_thickness=0)

    showing_seq  = game_state.get("showing_sequence", False)
    show_step    = game_state.get("show_step", 0)
    chain        = game_state.get("chain", [])
    step_index   = game_state.get("step_index", 0)
    time_left    = game_state.get("time_left", 2.0)
    target       = game_state.get("current_target", "")
    game_over    = game_state.get("game_over", False)
    active       = game_state.get("active_player", "P1")
    phase        = game_state.get("phase", "ADD")
    chain_len    = len(chain)

    draw_top_bar(frame, "SIMON SAYS  —  2 PLAYER CHAIN",
                 f"Chain: {chain_len}  |  {active} is active  |  Q Quit")

    if game_over:
        draw_centered_text(frame, "GAME OVER", h // 2 - _ix(h * 0.14),
                           0.90, COL_RED, thickness=3, outline=5)
        draw_centered_text(frame, game_state.get("game_over_text", ""),
                           h // 2 + _ix(h * 0.02), 0.38, COL_TEXT_ACCENT,
                           thickness=1, outline=2)
        return

    act_col = COL_CYAN if active == "P1" else COL_MAGENTA
    cy_mid  = h // 2

    if showing_seq:
        cur_g = chain[show_step] if show_step < len(chain) else ""
        draw_status_chip(frame, "WATCH THE CHAIN", _ix(h * 0.10), COL_YELLOW)
        if cur_g:
            icon_rect = (_ix(w * 0.30), _ix(h * 0.18),
                         _ix(w * 0.70), _ix(h * 0.64))
            _draw_gesture_icon(frame, cur_g, icon_rect)
            col = _GESTURE_COLORS.get(cur_g, COL_TEXT_ACCENT)
            draw_centered_text(frame, cur_g, _ix(h * 0.68), 0.60, col,
                               thickness=2, outline=3)
        draw_centered_text(frame, f"{show_step + 1} / {chain_len}",
                           _ix(h * 0.76), 0.36, COL_TEXT_DIM,
                           thickness=1, outline=2)
    elif phase == "ADD":
        draw_status_chip(frame, f"{active}  —  ADD A GESTURE", _ix(h * 0.10), act_col)
        draw_centered_text(frame, "Show your gesture now!",
                           cy_mid, 0.50, act_col, thickness=1, outline=3)
        pct = time_left / 2.0
        bx1 = _ix(w * 0.25); bx2 = _ix(w * 0.75)
        by  = _ix(h * 0.68); bh = _ix(h * 0.018)
        cv2.rectangle(frame, (bx1, by), (bx2, by + bh), (30, 30, 30), -1)
        tc = COL_GREEN if pct > 0.5 else (COL_YELLOW if pct > 0.25 else COL_RED)
        cv2.rectangle(frame, (bx1, by),
                      (bx1 + int((bx2 - bx1) * pct), by + bh), tc, -1)
    elif phase == "RECITE":
        draw_status_chip(frame, f"{active}  —  REPEAT STEP {step_index + 1}/{chain_len}",
                         _ix(h * 0.10), act_col)
        if target:
            icon_rect = (_ix(w * 0.33), _ix(h * 0.20),
                         _ix(w * 0.67), _ix(h * 0.62))
            _draw_gesture_icon(frame, target, icon_rect)
            col = _GESTURE_COLORS.get(target, COL_TEXT_ACCENT)
            draw_centered_text(frame, target, _ix(h * 0.66), 0.55, col,
                               thickness=2, outline=3)
        pct = time_left / 2.0
        bx1 = _ix(w * 0.25); bx2 = _ix(w * 0.75)
        by  = _ix(h * 0.76); bh = _ix(h * 0.018)
        cv2.rectangle(frame, (bx1, by), (bx2, by + bh), (30, 30, 30), -1)
        tc = COL_GREEN if pct > 0.5 else (COL_YELLOW if pct > 0.25 else COL_RED)
        cv2.rectangle(frame, (bx1, by),
                      (bx1 + int((bx2 - bx1) * pct), by + bh), tc, -1)

    _draw_simon_sequence_bar(frame, chain, step_index, show_step, showing_seq, w, h)
    draw_bottom_bar(frame, "P1=Left hand  |  P2=Right hand  |  2s per gesture  |  Q Quit")


# ============================================================
# SQUID GAME: RED LIGHT GREEN LIGHT
# ============================================================

def draw_squid_game_view(frame, game_state, hand_state=None):
    w, h = frame.shape[1], frame.shape[0]

    light        = game_state.get("light", "GREEN")
    dot_x        = game_state.get("dot_x", 0.5)
    dot_y        = game_state.get("dot_y", 0.5)
    capture_pct  = game_state.get("capture_progress", 0.0)
    score        = game_state.get("score", 0)
    dots_done    = game_state.get("dots_collected", 0)
    survival     = game_state.get("survival_time", 0.0)
    game_over    = game_state.get("game_over", False)
    light_left   = game_state.get("light_time_left", 0.0)
    t            = time.monotonic()

    # ── Background tint based on light ───────────────────────────────────
    if game_over:
        bg_col = (0, 0, 20)
    elif light == "RED":
        pulse = 0.15 + 0.08 * abs(math.sin(t * math.pi * 3))
        bg_col = (0, 0, int(40 * pulse + 10))
    else:
        bg_col = (0, 20, 0)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), bg_col, -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    if game_state.get("state") == "INTRO":
        draw_centered_text(frame, "RED LIGHT / GREEN LIGHT",
                           h // 2 - _ix(h * 0.10), 0.70, COL_GREEN,
                           thickness=2, outline=4)
        draw_centered_text(frame, "Guide your finger to the dot",
                           h // 2 + _ix(h * 0.04), 0.42, COL_TEXT_ACCENT,
                           thickness=1, outline=2)
        draw_centered_text(frame, "FREEZE on RED LIGHT",
                           h // 2 + _ix(h * 0.12), 0.40, COL_RED,
                           thickness=1, outline=2)
        draw_top_bar(frame, "SQUID GAME", "Q Quit")
        return

    if game_over:
        draw_centered_text(frame, "ELIMINATED!", h // 2 - _ix(h * 0.14),
                           1.0, COL_RED, thickness=3, outline=5)
        go_text = game_state.get("game_over_text", "")
        # Split into two lines at |
        parts = go_text.split("|")
        for i, part in enumerate(parts[:3]):
            draw_centered_text(frame, part.strip(),
                               h // 2 + _ix(h * 0.02) + i * _ix(h * 0.09),
                               0.38, COL_TEXT_ACCENT, thickness=1, outline=2)
        draw_top_bar(frame, "SQUID GAME  —  GAME OVER", "Q Quit  |  ESC Menu")
        return

    # ── Light indicator ───────────────────────────────────────────────────
    light_col = (0, 60, 0) if light == "GREEN" else (60, 0, 0)
    light_txt = "GREEN LIGHT" if light == "GREEN" else "RED LIGHT"
    light_fc  = COL_GREEN if light == "GREEN" else COL_RED
    pulse     = 0.85 + 0.15 * abs(math.sin(t * math.pi * (4 if light == "RED" else 1)))
    pc        = tuple(min(255, int(c * pulse)) for c in light_fc)

    # Big light banner at top
    draw_panel(frame, 0, 0, w, _ix(h * 0.12),
               fill=light_col, alpha=0.70, border=pc, border_thickness=3)
    draw_centered_text(frame, light_txt, _ix(h * 0.07), 0.80, pc,
                       thickness=3, outline=5)

    # Light countdown bar
    max_dur = 5.0
    pct_l   = min(1.0, light_left / max_dur)
    bx1 = _ix(w * 0.05); bx2 = _ix(w * 0.95)
    by  = _ix(h * 0.12); bh  = _ix(h * 0.012)
    cv2.rectangle(frame, (bx1, by), (bx2, by + bh), (30, 30, 30), -1)
    cv2.rectangle(frame, (bx1, by),
                  (bx1 + int((bx2 - bx1) * pct_l), by + bh), pc, -1)

    # ── Dot ───────────────────────────────────────────────────────────────
    # Convert normalised coords to pixel coords (avoid top/bottom bars)
    play_y1 = _ix(h * 0.14)
    play_y2 = _ix(h * 0.90)
    play_h  = play_y2 - play_y1

    dx = int(dot_x * w)
    dy = int(play_y1 + dot_y * play_h)
    dot_r = _ix(min(w, h) * 0.035)

    # Pulsing dot
    dot_pulse = 0.80 + 0.20 * abs(math.sin(t * math.pi * 1.5))
    dot_col   = tuple(min(255, int(c * dot_pulse)) for c in COL_YELLOW)
    cv2.circle(frame, (dx, dy), dot_r + 4, dot_col, 2)
    cv2.circle(frame, (dx, dy), dot_r, (30, 30, 0), -1)
    cv2.circle(frame, (dx, dy), dot_r, dot_col, 2)
    draw_centered_text(frame, "●", dx - 6, dy + 6, 0.30,
                       dot_col, thickness=1, outline=1)

    # Capture progress ring
    if capture_pct > 0:
        angle = int(360 * capture_pct)
        cv2.ellipse(frame, (dx, dy), (dot_r + 10, dot_r + 10), -90,
                    0, angle, COL_GREEN, 4)

    # ── Finger cursor ─────────────────────────────────────────────────────
    if hand_state:
        ix = hand_state.get("index_tip_x")
        iy = hand_state.get("index_tip_y")
        if ix is not None:
            fx = int(ix * w)
            fy = int(play_y1 + iy * play_h)
            cur_col = (0, 200, 0) if light == "GREEN" else (200, 0, 0)
            cv2.circle(frame, (fx, fy), _ix(w * 0.012), cur_col, -1)
            cv2.circle(frame, (fx, fy), _ix(w * 0.018), cur_col, 2)
            # Line to dot
            cv2.line(frame, (fx, fy), (dx, dy), (60, 60, 60), 1)

    # ── Score strip ───────────────────────────────────────────────────────
    score_y = _ix(h * 0.92)
    for txt, sx in [
        (f"Score: {score}", _ix(w * 0.05)),
        (f"Dots: {dots_done}", _ix(w * 0.35)),
        (f"Time: {survival:.0f}s", _ix(w * 0.65)),
    ]:
        draw_outlined_text(frame, txt, sx, score_y, 0.40,
                           COL_TEXT_ACCENT, thickness=1, outline=2)

    draw_bottom_bar(frame, "Move finger to the dot on GREEN  |  FREEZE on RED  |  Q Quit")



# ============================================================
# RPSLS SCREEN
# ============================================================

# Gesture colours for the two new gestures
_RPSLS_COLS = {
    "Rock":     COL_CYAN,
    "Paper":    COL_GREEN,
    "Scissors": COL_MAGENTA,
    "Lizard":   (80, 200, 80),     # lime green
    "Spock":    (255, 200, 0),     # gold
}

# Quick unicode representations for the rule-strip
_RPSLS_ICON = {
    "Rock": "✊", "Paper": "✋", "Scissors": "✌",
    "Lizard": "🤏", "Spock": "🖖",
}

# The 10 win rules as compact strings
_RPSLS_RULES = [
    "Scissors cuts Paper",    "Paper covers Rock",
    "Rock crushes Lizard",    "Lizard poisons Spock",
    "Spock smashes Scissors", "Scissors decapitates Lizard",
    "Lizard eats Paper",      "Paper disproves Spock",
    "Spock vaporizes Rock",   "Rock crushes Scissors",
]


def _draw_rpsls_gesture_icon(frame, gesture, rect):
    """Draw gesture icon with RPSLS-specific colours."""
    col = _RPSLS_COLS.get(gesture, COL_TEXT_DIM)
    x1, y1, x2, y2 = rect
    pw = x2 - x1; ph = y2 - y1
    cx = (x1 + x2) // 2; cy = (y1 + y2) // 2
    r  = _ix(min(pw, ph) * 0.40)

    if gesture == "Rock":
        cv2.circle(frame, (cx, cy), r, col, 3)
    elif gesture == "Paper":
        cv2.rectangle(frame, (cx - r, cy - r), (cx + r, cy + r), col, 3)
    elif gesture == "Scissors":
        cv2.line(frame, (cx - r, cy - r), (cx + r, cy + r), col, 3)
        cv2.line(frame, (cx + r, cy - r), (cx - r, cy + r), col, 3)
    elif gesture == "Lizard":
        # Snake-head shape: two arcs
        cv2.ellipse(frame, (cx, cy), (r, _ix(r * 0.55)), 0, 0, 180, col, 3)
        cv2.ellipse(frame, (cx, cy), (r, _ix(r * 0.55)), 0, 180, 360, col, 2)
        cv2.circle(frame, (cx - _ix(r * 0.35), cy - _ix(r * 0.20)), 4, col, -1)
        cv2.circle(frame, (cx + _ix(r * 0.35), cy - _ix(r * 0.20)), 4, col, -1)
    elif gesture == "Spock":
        # Stylised hand with V-split
        finger_h = _ix(r * 0.85)
        for i, fx in enumerate([-_ix(r*0.55), -_ix(r*0.18), _ix(r*0.18), _ix(r*0.55)]):
            top = cy - finger_h - (_ix(r * 0.15) if i in (0, 3) else 0)
            cv2.line(frame, (cx + fx, cy + _ix(r*0.3)), (cx + fx, top), col, 3)
        cv2.ellipse(frame, (cx, cy + _ix(r*0.3)), (_ix(r*0.55), _ix(r*0.3)),
                    0, 0, 180, col, 3)
    else:
        cv2.circle(frame, (cx, cy), r, COL_TEXT_DIM, 1)

    # Name
    if gesture not in ("Unknown", ""):
        draw_centered_text_in_rect(frame, gesture,
            (x1, y2, x2, y2 + _ix(ph * 0.28)),
            base_scale=0.38, color=col, thickness=1, outline=2)


def draw_rpsls_view(frame, game_state, tracker_state=None, hand_state=None):
    """
    RPSLS game screen.
    Left panel: player gesture. Right panel: AI gesture (revealed after shoot).
    Centre: countdown, result verb, score.
    Bottom: scrolling rule strip.
    """
    w, h = frame.shape[1], frame.shape[0]
    t    = time.monotonic()

    draw_panel(frame, 0, 0, w - 1, h - 1, fill=COL_BG_DARK, alpha=0.42,
               border=COL_BG_DARK, border_thickness=0)

    cur_state  = game_state["state"]
    p_gest     = game_state.get("player_gesture", "Unknown")
    ai_gest    = game_state.get("ai_gesture", "Unknown")
    banner     = game_state.get("result_banner", "")
    verb       = game_state.get("result_verb", "")
    score_text = game_state.get("score_text", "")
    beat_count = game_state.get("beat_count", 0)
    time_left  = game_state.get("time_left", 0.0)
    round_txt  = game_state.get("round_text", "")
    win_target = game_state.get("win_target", 3)
    p_score    = game_state.get("player_score", 0)
    ai_score   = game_state.get("robot_score", 0)

    draw_top_bar(frame, "RPSLS  —  vs AI",
                 f"{round_txt}  |  First to {win_target}  |  Q Quit")

    # ── Panel geometry ────────────────────────────────────────────────────
    pan_w = _ix(w * 0.28)
    py1   = _ix(h * 0.09)
    py2   = _ix(h * 0.82)
    ph    = py2 - py1
    cx1   = _ix(w * 0.01)
    cx2   = cx1 + pan_w
    ax1   = w - _ix(w * 0.01) - pan_w
    ax2   = w - _ix(w * 0.01)
    mid1  = cx2 + _ix(w * 0.01)
    mid2  = ax1 - _ix(w * 0.01)
    mid_w = mid2 - mid1

    # ── Live gesture display (player) ─────────────────────────────────────
    # Show live from tracker during gameplay, locked gesture after shoot
    if cur_state not in ("ROUND_RESULT", "MATCH_RESULT") and tracker_state:
        conf = tracker_state.get("confirmed_gesture", "Unknown")
        stab = tracker_state.get("stable_gesture", "Unknown")
        from rpsls_state import VALID_RPSLS_SET
        live = conf if conf in VALID_RPSLS_SET else (stab if stab in VALID_RPSLS_SET else "Unknown")
    else:
        live = p_gest

    p_col   = _RPSLS_COLS.get(live, COL_TEXT_DIM)
    p_fill  = (10, 14, 24)
    if cur_state in ("ROUND_RESULT", "MATCH_RESULT") and p_gest != "Unknown":
        outcome = game_state.get("last_round_result", "")
        if outcome == "win":   p_fill = (8, 45, 8)
        elif outcome == "lose": p_fill = (40, 8, 8)

    draw_panel(frame, cx1, py1, cx2, py2, fill=p_fill, alpha=0.88,
               border=p_col, border_thickness=2)
    draw_centered_text_in_rect(frame, "YOU",
        (cx1, py1 + _ix(ph*0.01), cx2, py1 + _ix(ph*0.12)),
        base_scale=0.50, color=p_col, thickness=2, outline=3)
    _draw_rpsls_gesture_icon(frame, live,
        (cx1 + _ix(pan_w*0.08), py1 + _ix(ph*0.14),
         cx2 - _ix(pan_w*0.08), py1 + _ix(ph*0.72)))

    # LOCK bar
    streak = tracker_state.get("stable_streak", 0) if tracker_state else 0
    bar_x  = cx2 - _ix(pan_w*0.08)
    bar_y1 = py1 + _ix(ph*0.14)
    bar_y2 = bar_y1 + _ix(ph*0.40)
    bar_w2 = _ix(pan_w*0.06)
    cv2.rectangle(frame, (bar_x, bar_y1), (bar_x+bar_w2, bar_y2), (30,30,30), -1)
    if streak > 0:
        fy = bar_y2 - int((bar_y2-bar_y1) * min(1.0, streak/3))
        fc = COL_GREEN if streak >= 3 else COL_YELLOW
        cv2.rectangle(frame, (bar_x, fy), (bar_x+bar_w2, bar_y2), fc, -1)

    # ── AI panel ──────────────────────────────────────────────────────────
    show_ai = (cur_state in ("ROUND_RESULT", "MATCH_RESULT") and
               ai_gest in ("Rock","Paper","Scissors","Lizard","Spock"))
    ai_col   = _RPSLS_COLS.get(ai_gest, COL_RED)
    ai_fill  = (10, 14, 24)
    if show_ai:
        outcome = game_state.get("last_round_result", "")
        if outcome == "lose":  ai_fill = (8, 45, 8)
        elif outcome == "win": ai_fill = (40, 8, 8)
    draw_panel(frame, ax1, py1, ax2, py2, fill=ai_fill, alpha=0.88,
               border=COL_RED, border_thickness=2)
    draw_centered_text_in_rect(frame, "AI",
        (ax1, py1 + _ix(ph*0.01), ax2, py1 + _ix(ph*0.12)),
        base_scale=0.50, color=COL_RED, thickness=2, outline=3)
    if show_ai:
        _draw_rpsls_gesture_icon(frame, ai_gest,
            (ax1 + _ix(pan_w*0.08), py1 + _ix(ph*0.14),
             ax2 - _ix(pan_w*0.08), py1 + _ix(ph*0.72)))
    else:
        draw_centered_text_in_rect(frame, "?",
            (ax1, py1 + _ix(ph*0.30), ax2, py1 + _ix(ph*0.65)),
            base_scale=1.4, color=(60,60,80), thickness=2, outline=3)

    # ── Centre panel ──────────────────────────────────────────────────────
    draw_panel(frame, mid1, py1, mid2, py2, fill=(6,8,18), alpha=0.88,
               border=COL_YELLOW, border_thickness=2)

    draw_status_chip(frame, cur_state.replace("_"," "),
                     py1 + _ix(ph*0.04), COL_YELLOW)

    if cur_state in ("ROUND_RESULT", "MATCH_RESULT"):
        b_col = get_result_banner_color(banner)
        draw_centered_text_in_rect(frame, banner,
            (mid1+4, py1+_ix(ph*0.14), mid2-4, py1+_ix(ph*0.34)),
            base_scale=0.52, color=b_col, thickness=2, outline=3)
        if verb:
            draw_centered_text_in_rect(frame, verb,
                (mid1+4, py1+_ix(ph*0.36), mid2-4, py1+_ix(ph*0.54)),
                base_scale=0.36, color=COL_TEXT_DIM, thickness=1, outline=2)
    elif cur_state == "COUNTDOWN":
        mt = "READY" if beat_count == 0 else str(min(beat_count, 3))
        pulse = 0.72 + 0.28 * abs(math.sin(t * math.pi * 1.4))
        nc    = tuple(min(255, int(c*pulse)) for c in COL_CYAN)
        draw_centered_text_in_rect(frame, mt,
            (mid1, py1+_ix(ph*0.18), mid2, py1+_ix(ph*0.60)),
            base_scale=2.0 if mt not in ("READY",) else 0.70,
            color=nc, thickness=3, outline=5)
    elif cur_state == "SHOOT_WINDOW":
        draw_centered_text_in_rect(frame, "SHOOT!",
            (mid1, py1+_ix(ph*0.18), mid2, py1+_ix(ph*0.58)),
            base_scale=0.80, color=COL_RED, thickness=2, outline=4)
        draw_centered_text_in_rect(frame, "Rock/Paper/Scissors/Lizard/Spock",
            (mid1+4, py1+_ix(ph*0.60), mid2-4, py1+_ix(ph*0.72)),
            base_scale=0.28, color=COL_TEXT_DIM, thickness=1, outline=2)
    else:
        draw_centered_text_in_rect(frame, "MAKE A FIST",
            (mid1, py1+_ix(ph*0.30), mid2, py1+_ix(ph*0.62)),
            base_scale=0.50, color=COL_CYAN, thickness=2, outline=3)

    # Beat dots
    if cur_state in ("COUNTDOWN", "WAITING_FOR_ROCK", "SHOOT_WINDOW"):
        dot_y   = py1 + _ix(ph*0.80)
        dot_sp  = _ix(mid_w * 0.19)
        dot_x0  = mid1 + _ix(mid_w*0.09)
        for bi in range(4):
            bx  = dot_x0 + bi * dot_sp
            act = bi < beat_count or (bi == 3 and cur_state == "SHOOT_WINDOW")
            col = COL_RED if (bi == 3 and cur_state == "SHOOT_WINDOW") else \
                  (COL_CYAN if act else (50,50,70))
            cv2.circle(frame, (bx, dot_y), _ix(w*0.011), col, -1 if act else 2)

    # Score + progress dots
    draw_centered_text(frame, score_text,
                       py2 + _ix(h*0.007), 0.46, COL_TEXT_ACCENT,
                       thickness=2, outline=3)

    # Mini score pips
    for px_offset, score, col in [
        (-_ix(w*0.15), p_score, COL_CYAN),
        (+_ix(w*0.15), ai_score, COL_RED),
    ]:
        bx0 = w//2 + px_offset - _ix(win_target * 10)
        for i in range(win_target):
            bx = bx0 + i * _ix(20)
            fc = col if i < score else (40, 40, 50)
            cv2.circle(frame, (bx, py2 + _ix(h*0.025)), _ix(w*0.007), fc, -1)

    # ── Scrolling rule strip at bottom ────────────────────────────────────
    strip_y = h - _ix(h*0.050)
    # Slowly scroll rules
    scroll_idx = int(t * 0.5) % len(_RPSLS_RULES)
    rule_txt = _RPSLS_RULES[scroll_idx] + "  •  " + _RPSLS_RULES[(scroll_idx+1) % len(_RPSLS_RULES)]
    draw_centered_text(frame, rule_txt, strip_y, 0.30, COL_TEXT_DIM,
                       thickness=1, outline=1)

    draw_bottom_bar(frame, "5 gestures: Rock  Paper  Scissors  Lizard  Spock  |  Q Quit")


def draw_two_player_diagnostic(frame, game_state,
                               p1_hand_state=None, p2_hand_state=None,
                               p1_tracker_state=None, p2_tracker_state=None,
                               fps=0.0):
    """
    Full diagnostic overlay for two-player modes.
    Shows both hands' complete detection pipeline side by side.
    Toggled with M key, same as single-player diagnostic.
    """
    w, h = frame.shape[1], frame.shape[0]

    # Dark tint over camera feed
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    draw_top_bar(frame, "2-PLAYER DIAGNOSTIC",
                 f"M Normal view  |  FPS {fps:.0f}  |  ESC Back")

    # Two vertical diagnostic panels
    mid   = w // 2
    pad   = 8
    panel_x = [pad, mid + pad]
    panel_w = mid - pad * 2

    labels   = ["PLAYER 1  (left hand)",  "PLAYER 2  (right hand)"]
    colors   = [COL_CYAN,                 COL_MAGENTA]
    h_states = [p1_hand_state,            p2_hand_state]
    t_states = [p1_tracker_state,         p2_tracker_state]

    for col_i in range(2):
        px1 = panel_x[col_i]
        px2 = px1 + panel_w
        py1 = _ix(h * 0.07)
        py2 = _ix(h * 0.95)
        col = colors[col_i]
        hs  = h_states[col_i] or {}
        ts  = t_states[col_i] or {}

        draw_panel(frame, px1, py1, px2, py2,
                   fill=(6, 8, 18), alpha=0.88,
                   border=col, border_thickness=2)

        ph = py2 - py1
        ty = py1 + _ix(ph * 0.04)
        lx = px1 + _ix(panel_w * 0.04)
        line_gap = _ix(ph * 0.065)
        font_sc  = 0.38

        def _row(label_txt, val_txt, val_col=None):
            nonlocal ty
            draw_outlined_text(frame, label_txt, lx, ty, font_sc,
                               COL_TEXT_DIM, thickness=1, outline=2)
            vx = lx + _ix(panel_w * 0.50)
            draw_outlined_text(frame, str(val_txt), vx, ty, font_sc,
                               val_col or COL_TEXT_ACCENT, thickness=1, outline=2)
            ty += line_gap

        # Header
        draw_outlined_text(frame, labels[col_i], lx, ty, 0.44,
                           col, thickness=2, outline=3)
        ty += _ix(line_gap * 1.3)

        # Hand detection
        n_hands = hs.get("hands_detected", 0)
        _row("Hands detected:", str(n_hands),
             COL_GREEN if n_hands > 0 else COL_RED)

        # Pipeline: raw → stable → confirmed
        raw  = hs.get("raw_gesture", "—")
        stab = ts.get("stable_gesture",   "—")
        conf = ts.get("confirmed_gesture","—")
        streak = ts.get("stable_streak", 0)

        _row("Raw gesture:",       raw,
             get_gesture_color(raw) if raw in ("Rock","Paper","Scissors") else COL_TEXT_DIM)
        _row("Stable gesture:",    stab,
             get_gesture_color(stab) if stab in ("Rock","Paper","Scissors") else COL_YELLOW)
        _row("Confirmed gesture:", conf,
             get_gesture_color(conf) if conf in ("Rock","Paper","Scissors") else COL_ORANGE)
        _row("Stable streak:",     f"{streak}/3",
             COL_GREEN if streak >= 3 else COL_YELLOW)

        ty += _ix(line_gap * 0.3)
        cv2.line(frame, (lx, ty), (px2 - pad, ty), (40, 40, 60), 1)
        ty += _ix(line_gap * 0.5)

        # Hand quality
        wrist_y  = hs.get("wrist_y")
        palm_sc  = hs.get("palm_scale", 0.0)
        too_far  = hs.get("hand_too_far", False)
        poor_lit = hs.get("poor_lighting", False)

        _row("Wrist Y:",    f"{wrist_y:.3f}" if wrist_y else "—")
        _row("Palm scale:", f"{palm_sc:.3f}",
             COL_RED if too_far else COL_GREEN)
        _row("Too far:",    "YES" if too_far  else "no",
             COL_RED if too_far else COL_TEXT_DIM)
        _row("Poor light:", "YES" if poor_lit else "no",
             COL_RED if poor_lit else COL_TEXT_DIM)

        ty += _ix(line_gap * 0.3)
        cv2.line(frame, (lx, ty), (px2 - pad, ty), (40, 40, 60), 1)
        ty += _ix(line_gap * 0.5)

        # Detection reason
        reason = hs.get("reason_text", "—")
        _row("Reason:", reason)

        # LOCK bar visual
        bar_x   = lx
        bar_y   = ty
        bar_w   = _ix(panel_w * 0.88)
        bar_h   = _ix(ph * 0.025)
        pct     = min(1.0, streak / 3)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (30, 30, 30), -1)
        if pct > 0:
            fx = bar_x + int(bar_w * pct)
            fc = COL_GREEN if pct >= 1.0 else COL_YELLOW
            cv2.rectangle(frame, (bar_x, bar_y), (fx, bar_y + bar_h), fc, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)
        draw_outlined_text(frame, "LOCK", bar_x, bar_y - 4, 0.28,
                           COL_TEXT_DIM, thickness=1, outline=1)

    # Bottom: shared game state summary
    gs_y = h - _ix(h * 0.042)
    gs_txt = (f"State: {game_state.get('state','?')}  |  "
              f"Beat: {game_state.get('beat_count',0)}  |  "
              f"Score: {game_state.get('score_text','?')}  |  "
              f"Mode: {game_state.get('play_mode_label','?')}")
    draw_bottom_bar(frame, gs_txt)


def draw_game_mode_view(frame, game_state, emotion_state=None, voice_mode_active=False,
                        last_heard_word="", tracker_state=None, hand_state=None,
                        flash_info=None, show_help=False, sound_on=True,
                        colourblind=False, show_session_summary=False):
    draw_game_header(frame, game_state, voice_mode_active=voice_mode_active, sound_on=sound_on)
    if voice_mode_active:
        bottom_hint = "ESC Back  |  Say ROCK / PAPER / SCISSORS  |  ? Help  |  Q Quit"
    else:
        bottom_hint = "ESC Back  |  M Diagnostic  |  N Sound  |  ? Help  |  Q Quit"
    draw_bottom_bar(frame, bottom_hint)
    draw_arcade_header(frame)
    draw_game_status_strip(frame, game_state)

    cur_state = game_state["state"]

    if cur_state in {"ROUND_RESULT", "MATCH_RESULT"}:
        if cur_state == "MATCH_RESULT" and show_session_summary:
            summary = game_state.get("session_summary")
            if summary:
                draw_session_summary(frame, summary)
            else:
                draw_result_screen(frame, game_state, colourblind=colourblind)
        else:
            draw_result_screen(frame, game_state, colourblind=colourblind)
        if flash_info and flash_info.get("active"):
            draw_result_flash(frame, flash_info["result"],
                              flash_info["frame_idx"], max_flash_frames=5,
                              colourblind=colourblind)
    else:
        last_pg = game_state.get("last_player_gesture")
        last_rg = game_state.get("last_robot_gesture")
        replay_active = (cur_state == "WAITING_FOR_ROCK" and last_pg and last_rg
                         and flash_info and flash_info.get("replay_until", 0) > time.monotonic())
        if replay_active:
            _draw_last_round_replay(frame, last_pg, last_rg, game_state.get("last_banner", ""))
        else:
            draw_arcade_hero(frame, game_state, voice_mode_active=voice_mode_active)
        draw_arcade_beat_track(frame, game_state["beat_count"], game_state["state"],
                               voice_mode_active=voice_mode_active)

    w, h = _frame_size(frame)
    layout = _game_layout(frame)

    # Gesture confidence LOCK bar
    # In voice COUNTDOWN the bar is voice-driven — suppress gesture fill
    # so it doesn't visually jump ahead of the spoken words.
    if tracker_state and cur_state not in {"ROUND_RESULT", "MATCH_RESULT"}:
        if voice_mode_active and cur_state == "COUNTDOWN":
            # Show beat progress as LOCK bar fill instead of gesture streak
            beat_count = game_state.get("beat_count", 0)
            draw_gesture_confidence_bar(frame, beat_count, 3,
                                        _ix(w * 0.01), h - _ix(h * 0.045), _ix(w * 0.22))
        else:
            streak = tracker_state.get("stable_streak", 0)
            draw_gesture_confidence_bar(frame, streak, 3,
                                        _ix(w * 0.01), h - _ix(h * 0.045), _ix(w * 0.22))

    # Win/loss streak label
    streak_text = game_state.get("streak_label", "")
    if streak_text:
        streak_col = COL_GREEN if "WIN" in streak_text.upper() else (80, 80, 200)
        draw_outlined_text(frame, streak_text, _ix(w * 0.07), h - _ix(h * 0.08),
                           0.38, streak_col, thickness=1, outline=2)

    # Opponent type chip
    opp_type = game_state.get("opponent_type", "")
    _opp_skip = {"random", "grace_period", "", "unknown", "Unknown"}
    if opp_type and opp_type not in _opp_skip \
            and cur_state not in {"ROUND_RESULT", "MATCH_RESULT"}:
        chip_text = f"[ {opp_type.replace('_', ' ').upper()} DETECTED ]"
        draw_outlined_text(frame, chip_text, _ix(w * 0.02),
                           layout["top_row_h"] + layout["second_row_h"] + _ix(h * 0.038),
                           0.34, COL_MAGENTA, thickness=1, outline=2)

    # Voice badge + mic bar (right-aligned, voice only)
    if voice_mode_active:
        badge_text = f"[ MIC  {last_heard_word.upper()} ]" if last_heard_word else "[ MIC ON ]"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (bw, _), _ = cv2.getTextSize(badge_text, font, 0.38, 1)
        badge_x = w - bw - _ix(w * 0.02)
        badge_y = layout["top_row_h"] + layout["second_row_h"] + _ix(h * 0.038)
        draw_outlined_text(frame, badge_text, badge_x, badge_y, 0.38, (80, 255, 180), thickness=1, outline=2)
        mic_level = flash_info.get("mic_level", 0.0) if flash_info else 0.0
        if mic_level > 0.01:
            bar_y2 = badge_y + _ix(h * 0.018)
            fill_w = int(bw * mic_level)
            cv2.rectangle(frame, (badge_x, bar_y2), (badge_x + bw, bar_y2 + 4), (30, 60, 30), -1)
            col = (0, 255, 100) if mic_level < 0.7 else (0, 200, 255)
            cv2.rectangle(frame, (badge_x, bar_y2), (badge_x + fill_w, bar_y2 + 4), col, -1)

    if hand_state:
        draw_quality_warnings(frame, hand_state)

    if emotion_state and emotion_state.get("face_detected"):
        cal = emotion_state.get("calibrated", True)
        if not cal:
            draw_outlined_text(frame, f"calibrating {emotion_state.get('calibration_progress', 0)}%",
                               w - _ix(w * 0.30), h - _ix(h * 0.11), 0.38, (200, 180, 80), thickness=1, outline=2)
        else:
            em = emotion_state["stable_emotion"]
            em_color = _get_emotion_color(em) if em != "Neutral" else (90, 90, 90)
            label = em if em == "Neutral" else f"{em}  {emotion_state['confidence']:.0%}"
            draw_outlined_text(frame, label, w - _ix(w * 0.26), h - _ix(h * 0.11),
                               0.42, em_color, thickness=1, outline=2)

    if show_help:
        draw_help_overlay(frame, "GAME", voice_mode=voice_mode_active)

# ============================================================
# MENU SCREEN
# ============================================================

def draw_menu_screen(frame, menu_items, selected_index, config,
                     show_help=False, voice_mode_active=False, in_submenu=False,
                     update_label=""):
    layout = _menu_layout(frame)
    w, h = layout["w"], layout["h"]
    x1, y1, x2, y2 = layout["panel"]

    draw_panel(frame, 0, 0, w - 1, h - 1, fill=COL_BG_DARK, alpha=0.75,
               border=COL_BG_DARK, border_thickness=0)

    top_right = "UP/DOWN Navigate | Enter Select | ESC Back | Q Quit"
    draw_top_bar(frame, "RPS ROBOT", top_right)

    # Update banner — shown above menu panel when an update is available
    if update_label:
        banner_y = y1 - _ix(h * 0.055)
        pulse = 0.65 + 0.35 * abs(math.sin(time.monotonic() * math.pi * 1.2))
        bc = tuple(min(255, int(c * pulse)) for c in COL_YELLOW)
        draw_panel(frame, _ix(w*0.03), banner_y - _ix(h*0.008),
                   _ix(w*0.97), banner_y + _ix(h*0.038),
                   fill=(18, 15, 0), alpha=0.90, border=bc, border_thickness=2)
        draw_centered_text(frame, update_label, banner_y + _ix(h*0.016),
                           0.36, bc, thickness=1, outline=2)

    draw_panel(frame, x1, y1, x2, y2, fill=COL_BG_PANEL, alpha=0.92,
               border=COL_CYAN, border_thickness=2)

    if in_submenu:
        draw_centered_text(frame, "GAME MODES", y1 + _ix((y2 - y1) * 0.09),
                           0.80, COL_YELLOW, thickness=2, outline=3)
        draw_centered_text(frame, "Select a mode to play",
                           y1 + _ix((y2 - y1) * 0.17), 0.40, COL_TEXT_DIM, thickness=1, outline=2)
    else:
        draw_centered_text(frame, "ROCK  PAPER  SCISSORS", y1 + _ix((y2 - y1) * 0.09),
                           0.68, COL_TEXT, thickness=2, outline=3)
        draw_centered_text(frame, "ROBOT ARCADE", y1 + _ix((y2 - y1) * 0.19),
                           1.00, COL_YELLOW, thickness=3, outline=4)

    line_y = y1 + _ix((y2 - y1) * 0.26)
    cv2.line(frame, (x1 + _ix(w * 0.10), line_y), (x2 - _ix(w * 0.10), line_y), COL_CYAN, 1)

    subtitle = f"Default: {config['default_play_mode']} | Display: {config['default_display_mode']}"
    draw_centered_text(frame, subtitle, y1 + _ix((y2 - y1) * 0.32),
                       0.42, COL_TEXT_DIM, thickness=1, outline=2)

    item_area_top = y1 + _ix((y2 - y1) * 0.40)
    n_items = len(menu_items)
    # Scale font down automatically when there are many items
    item_gap  = _ix((y2 - y1) * 0.55 / max(n_items, 1))
    font_base = min(0.68, 0.68 * (8 / max(n_items, 8)))

    for i, (label, _) in enumerate(menu_items):
        selected = i == selected_index
        cy = item_area_top + i * item_gap
        bar_h = max(_ix(h * 0.018), item_gap // 2 - 2)

        if selected:
            draw_panel(frame, x1 + _ix(w * 0.04), cy - bar_h, x2 - _ix(w * 0.04), cy + bar_h,
                       fill=(20, 40, 60), alpha=0.70, border=COL_CYAN, border_thickness=1)

        color  = COL_CYAN if selected else COL_TEXT_DIM
        prefix = "> " if selected else "  "
        draw_centered_text_in_rect(frame, f"{prefix}{label}",
            (x1 + _ix(w * 0.06), cy - bar_h, x2 - _ix(w * 0.06), cy + bar_h),
            base_scale=font_base, color=color, thickness=2, outline=3)

    # Dynamic bottom bar
    if voice_mode_active:
        bottom = "ESC Back  |  Voice: CHEAT  FAIR  CHALLENGE  CLONE  STATS  SETTINGS  |  ? Help"
    elif in_submenu:
        bottom = "ESC Back to Main Menu  |  Enter to start mode  |  ? Help"
    else:
        bottom = "ESC Back  |  W/S Navigate  |  Enter Select  |  ? Help  |  Q Quit"
    draw_bottom_bar(frame, bottom)

    if show_help:
        draw_help_overlay(frame, "MENU", voice_mode=voice_mode_active)

# ============================================================
# SIMULATION SCREEN
# ============================================================

def draw_simulation_screen(frame, sim_state):
    """
    In-app simulation progress and results screen.
    sim_state keys:
      status:        "running" | "done" | "error"
      progress:      float 0–1
      progress_text: str  e.g. "random vs fair_play  4/10"
      results:       dict from run_simulation() or None
      error:         str or None
    """
    layout = _menu_layout(frame)
    w, h = layout["w"], layout["h"]
    x1, y1, x2, y2 = layout["panel"]

    draw_panel(frame, 0, 0, w - 1, h - 1, fill=COL_BG_DARK, alpha=0.75,
               border=COL_BG_DARK, border_thickness=0)
    draw_top_bar(frame, "SIMULATION", "ESC Cancel / Back")
    draw_panel(frame, x1, y1, x2, y2, fill=COL_BG_PANEL, alpha=0.94,
               border=COL_YELLOW, border_thickness=2)

    status   = sim_state.get("status", "running")
    progress = sim_state.get("progress", 0.0)
    prog_txt = sim_state.get("progress_text", "")
    results  = sim_state.get("results")
    error    = sim_state.get("error")

    pw, ph = x2 - x1, y2 - y1
    cy = y1 + _ix(ph * 0.10)

    if status == "running":
        draw_centered_text(frame, "RUNNING SIMULATION", cy, 0.70, COL_YELLOW, thickness=2, outline=3)
        cy += _ix(ph * 0.10)
        draw_centered_text(frame, prog_txt, cy, 0.42, COL_TEXT_ACCENT, thickness=1, outline=2)
        cy += _ix(ph * 0.10)

        # Progress bar
        bar_x1 = x1 + _ix(pw * 0.08)
        bar_x2 = x2 - _ix(pw * 0.08)
        bar_y  = cy
        bar_h  = _ix(ph * 0.045)
        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + bar_h), (40, 40, 40), -1)
        fill_x = bar_x1 + int((bar_x2 - bar_x1) * max(0.0, min(1.0, progress)))
        if fill_x > bar_x1:
            cv2.rectangle(frame, (bar_x1, bar_y), (fill_x, bar_y + bar_h), COL_GREEN, -1)
        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + bar_h), COL_CYAN, 1)
        pct_text = f"{progress * 100:.0f}%"
        draw_centered_text(frame, pct_text, bar_y + bar_h + _ix(ph * 0.05),
                           0.48, COL_TEXT, thickness=1, outline=2)

        # Animated dots
        t   = time.monotonic()
        dots = "." * (1 + int(t * 2) % 3)
        draw_centered_text(frame, f"Please wait{dots}", cy + _ix(ph * 0.18),
                           0.42, COL_TEXT_DIM, thickness=1, outline=2)

    elif status == "error":
        draw_centered_text(frame, "SIMULATION ERROR", cy, 0.70, COL_RED, thickness=2, outline=3)
        cy += _ix(ph * 0.12)
        draw_centered_text(frame, str(error or "Unknown error"), cy,
                           0.40, COL_TEXT_ACCENT, thickness=1, outline=2)

    elif status == "done" and results:
        draw_centered_text(frame, "SIMULATION RESULTS", cy, 0.70, COL_YELLOW, thickness=2, outline=3)
        cy += _ix(ph * 0.08)

        total_r   = results.get("total_rounds_actual") or results.get("total_rounds", 0)
        elapsed   = results.get("elapsed_seconds", 0)
        best_ai   = results.get("best_ai",       "?")
        worst_ai  = results.get("worst_ai",      "?")
        best_s    = results.get("best_strategy", "?")
        worst_s   = results.get("worst_strategy","?")
        balanced  = results.get("most_balanced", "?")

        # Headline row
        for line, col in [
            (f"Rounds simulated:  {total_r:,}", COL_TEXT_ACCENT),
            (f"Time elapsed:      {elapsed:.1f}s", COL_TEXT_DIM),
        ]:
            draw_outlined_text(frame, line, x1 + _ix(pw * 0.06), cy,
                               0.44, col, thickness=1, outline=2)
            cy += _ix(ph * 0.065)

        cv2.line(frame, (x1 + _ix(pw * 0.04), cy - _ix(ph * 0.01)),
                 (x2 - _ix(pw * 0.04), cy - _ix(ph * 0.01)), COL_CYAN, 1)

        # Two column layout
        lcol = x1 + _ix(pw * 0.04)
        rcol = x1 + _ix(pw * 0.52)
        col_cy = cy + _ix(ph * 0.02)

        # Left: AI rankings
        draw_outlined_text(frame, "AI DIFFICULTY RANKING",
                           lcol, col_cy, 0.40, COL_CYAN, thickness=1, outline=2)
        col_cy += _ix(ph * 0.06)
        ai_rates = results.get("ai_win_rates", {})
        for ai_name, rate in sorted(ai_rates.items(), key=lambda x: -x[1]):
            bar_w2 = _ix(pw * 0.35)
            bh = _ix(ph * 0.022)
            cv2.rectangle(frame, (lcol, col_cy - bh), (lcol + bar_w2, col_cy + 2), (40, 40, 40), -1)
            fx2 = lcol + int(bar_w2 * rate)
            cv2.rectangle(frame, (lcol, col_cy - bh), (fx2, col_cy + 2), COL_RED, -1)
            draw_outlined_text(frame, f"{ai_name}: {rate:.1%} AI wins",
                               lcol, col_cy - _ix(ph * 0.002),
                               0.36, COL_TEXT, thickness=1, outline=2)
            col_cy += _ix(ph * 0.065)

        col_cy += _ix(ph * 0.02)
        for label, val, col in [
            ("Hardest AI:  ", best_ai,  COL_RED),
            ("Easiest AI:  ", worst_ai, COL_GREEN),
        ]:
            draw_outlined_text(frame, label + val, lcol, col_cy,
                               0.38, col, thickness=1, outline=2)
            col_cy += _ix(ph * 0.06)

        # Right: Strategy rankings
        col_cy2 = cy + _ix(ph * 0.02)
        draw_outlined_text(frame, "PLAYER STRATEGY RANKING",
                           rcol, col_cy2, 0.40, COL_CYAN, thickness=1, outline=2)
        col_cy2 += _ix(ph * 0.06)
        strat_rates = results.get("strategy_win_rates", {})
        for s_name, rate in sorted(strat_rates.items(), key=lambda x: -x[1]):
            bar_w3 = _ix(pw * 0.35)
            bh = _ix(ph * 0.022)
            cv2.rectangle(frame, (rcol, col_cy2 - bh), (rcol + bar_w3, col_cy2 + 2), (40, 40, 40), -1)
            fx3 = rcol + int(bar_w3 * rate)
            col2 = COL_GREEN if rate > 0.35 else COL_ORANGE
            cv2.rectangle(frame, (rcol, col_cy2 - bh), (fx3, col_cy2 + 2), col2, -1)
            draw_outlined_text(frame, f"{s_name}: {rate:.1%}",
                               rcol, col_cy2 - _ix(ph * 0.002),
                               0.36, COL_TEXT, thickness=1, outline=2)
            col_cy2 += _ix(ph * 0.065)

        col_cy2 += _ix(ph * 0.02)
        for label, val, col in [
            ("Best strategy:  ", best_s,  COL_GREEN),
            ("Worst strategy: ", worst_s, COL_ORANGE),
        ]:
            draw_outlined_text(frame, label + val, rcol, col_cy2,
                               0.38, col, thickness=1, outline=2)
            col_cy2 += _ix(ph * 0.06)

        # Footer
        draw_centered_text(frame, f"Most balanced matchup: {balanced}",
                           y2 - _ix(ph * 0.09), 0.36, COL_TEXT_ACCENT, thickness=1, outline=2)
        draw_centered_text(frame, "Full data saved to Desktop/CapStone/simulation_results.xlsx",
                           y2 - _ix(ph * 0.04), 0.32, COL_TEXT_DIM, thickness=1, outline=2)

    draw_bottom_bar(frame, "ESC Back to Menu")

def draw_settings_screen(frame, settings_schema, selected_index, config,
                         format_value_fn, cursor_info=None, text_edit=False):
    layout = _settings_layout(frame)
    w, h = layout["w"], layout["h"]
    x1, y1, x2, y2 = layout["panel"]

    nav_active  = cursor_info is not None and cursor_info.get("active", False)
    x_zone      = cursor_info.get("x_zone", "center")  if nav_active else "center"
    adjust_pct  = cursor_info.get("adjust_pct", 0.0)   if nav_active else 0.0

    draw_panel(frame, 0, 0, w - 1, h - 1, fill=COL_BG_DARK, alpha=0.75,
               border=COL_BG_DARK, border_thickness=0)

    if text_edit:
        hint = "Type name | ENTER confirm | ESC cancel"
    elif nav_active:
        hint = "Move left ◄ −  |  center = select  |  + ► move right"
    else:
        hint = "UP/DOWN Select | LEFT/RIGHT Change | ENTER edit name | BACK"
    draw_top_bar(frame, "SETTINGS", hint)

    draw_panel(frame, x1, y1, x2, y2, fill=COL_BG_PANEL, alpha=0.94,
               border=COL_CYAN, border_thickness=2)

    draw_centered_text(frame, "GAME SETTINGS", y1 + _ix((y2 - y1) * 0.07),
                       0.80, COL_TEXT, thickness=3, outline=4)

    n_items = len(settings_schema)
    # Show a scrolling window of items so the description box is never obscured.
    VISIBLE = 8
    # Centre the window on selected_index, clamped to valid range
    win_start = max(0, min(selected_index - VISIBLE // 2, n_items - VISIBLE))
    win_end   = min(n_items, win_start + VISIBLE)
    visible_items = list(range(win_start, win_end))

    start_y = y1 + _ix((y2 - y1) * 0.15)
    row_gap  = _ix((y2 - y1) * 0.075)

    # Scroll indicator dots top-right when list is longer than window
    if n_items > VISIBLE:
        dot_x = x2 - _ix(w * 0.03)
        for di in range(n_items):
            dot_y = y1 + _ix((y2 - y1) * 0.15) + _ix((y2 - y1) * 0.60 * di / max(n_items - 1, 1))
            col   = COL_CYAN if di == selected_index else (50, 50, 70)
            cv2.circle(frame, (dot_x, dot_y), 3, col, -1)

    for slot, i in enumerate(visible_items):
        item = settings_schema[i]
        selected    = i == selected_index
        y           = start_y + slot * row_gap
        is_adj      = item.get("type") in ("choice", "float")
        is_action   = item.get("type") == "action"
        bar_half_h  = _ix(h * 0.024)

        if selected:
            bar_y1 = y - bar_half_h
            bar_y2 = y + bar_half_h
            draw_panel(frame, x1 + _ix(w * 0.015), bar_y1, x2 - _ix(w * 0.015), bar_y2,
                       fill=(20, 40, 60), alpha=0.60, border=COL_CYAN, border_thickness=1)

        label_color = COL_CYAN if selected else COL_TEXT_DIM
        prefix = "> " if selected else "  "

        draw_outlined_text(frame, f"{prefix}{item['label']}", x1 + _ix(w * 0.025), y,
                           0.54, label_color, thickness=2, outline=2)

        # Value + optional +/- buttons for adjustable items
        value = format_value_fn(item)
        if value and not is_action:
            is_text = item.get("type") == "text"
            if selected and is_text:
                # Text field — blinking cursor when selected
                blink   = int(time.monotonic() * 2) % 2 == 0
                display = f"{value}|" if (blink and text_edit) else value
                font    = cv2.FONT_HERSHEY_SIMPLEX
                # Highlighted box when in edit mode
                field_x1 = x2 - _ix(w * 0.32)
                border_col = COL_YELLOW if text_edit else COL_CYAN
                draw_panel(frame, field_x1, y - bar_half_h + 2,
                           x2 - _ix(w * 0.015), y + bar_half_h - 2,
                           fill=(8, 25, 45), alpha=0.85,
                           border=border_col, border_thickness=1)
                draw_outlined_text(frame, display, field_x1 + _ix(w * 0.01), y,
                                   0.50, COL_TEXT_ACCENT, thickness=1, outline=2)
                if not text_edit:
                    hint_txt = "ENTER to edit"
                    (hw, _), _ = cv2.getTextSize(hint_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
                    cv2.putText(frame, hint_txt,
                                (field_x1 - hw - _ix(w * 0.015), y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.32, COL_TEXT_DIM, 1, cv2.LINE_AA)
            elif selected and is_adj:
                # --- Draw  [−]  value  [+]  trio ---
                btn_w  = _ix(w * 0.055)
                btn_h  = bar_half_h
                gap    = _ix(w * 0.010)
                # [+] on the right
                plus_x2 = x2 - _ix(w * 0.018)
                plus_x1 = plus_x2 - btn_w
                # [−] to the left of [+]
                minus_x2 = plus_x1 - gap
                minus_x1 = minus_x2 - btn_w

                # Highlight based on gesture nav X zone
                minus_active = nav_active and x_zone == "minus"
                plus_active  = nav_active and x_zone == "plus"

                minus_col = COL_YELLOW if minus_active else (50, 80, 50)
                plus_col  = COL_YELLOW if plus_active  else (50, 80, 50)
                minus_text_col = (10, 10, 10) if minus_active else (120, 180, 120)
                plus_text_col  = (10, 10, 10) if plus_active  else (120, 180, 120)

                # Draw dwell arc on active button
                for btn_cx, is_act, col in [
                    (minus_x1 + btn_w // 2, minus_active, minus_col),
                    (plus_x1  + btn_w // 2, plus_active,  plus_col),
                ]:
                    draw_panel(frame,
                               btn_cx - btn_w // 2, y - btn_h,
                               btn_cx + btn_w // 2, y + btn_h,
                               fill=(20, 50, 20) if is_act else (12, 20, 12),
                               alpha=0.85, border=col, border_thickness=1 if not is_act else 2)
                    if is_act and adjust_pct > 0:
                        ang = int(360 * adjust_pct)
                        r = int(80 + 175 * adjust_pct)
                        g = int(255 * (1 - adjust_pct * 0.8))
                        cv2.ellipse(frame, (btn_cx, y), (btn_h, btn_h),
                                    -90, 0, ang, (0, g, r), 2)

                draw_centered_text_in_rect(frame, "-",
                    (minus_x1, y - btn_h, minus_x2, y + btn_h),
                    base_scale=0.70, color=minus_text_col, thickness=2, outline=0)
                draw_centered_text_in_rect(frame, "+",
                    (plus_x1, y - btn_h, plus_x2, y + btn_h),
                    base_scale=0.70, color=plus_text_col, thickness=2, outline=0)

                # Value between label and buttons
                val_x = minus_x1 - gap
                font = cv2.FONT_HERSHEY_SIMPLEX
                (vw, _), _ = cv2.getTextSize(value, font, 0.52, 1)
                draw_outlined_text(frame, value, val_x - vw, y,
                                   0.52, COL_TEXT_ACCENT, thickness=1, outline=2)
            else:
                # Non-selected or non-adjustable: value right-aligned
                font = cv2.FONT_HERSHEY_SIMPLEX
                (text_w, _), _ = cv2.getTextSize(value, font, 0.50, 1)
                draw_outlined_text(frame, value, x2 - text_w - _ix(w * 0.025), y,
                                   0.50, COL_TEXT_ACCENT, thickness=1, outline=2)

    # Description box
    desc_y1 = y2 - _ix((y2 - y1) * 0.22)
    desc_y2 = y2 - _ix((y2 - y1) * 0.03)
    selected_item = settings_schema[selected_index] if selected_index < n_items else None
    desc_text = selected_item.get("desc", "") if selected_item else ""

    if desc_text:
        draw_panel(frame, x1 + _ix(w * 0.015), desc_y1, x2 - _ix(w * 0.015), desc_y2,
                   fill=(8, 20, 35), alpha=0.90, border=COL_CYAN, border_thickness=1)

        max_chars = max(30, int((x2 - x1) / (_ix(w * 0.012) + 1)))
        words, lines, current = desc_text.split(), [], ""
        for word in words:
            test = f"{current} {word}".strip()
            if len(test) <= max_chars:
                current = test
            else:
                if current: lines.append(current)
                current = word
        if current: lines.append(current)

        desc_line_y   = desc_y1 + _ix((desc_y2 - desc_y1) * 0.30)
        desc_line_gap = _ix((desc_y2 - desc_y1) * 0.30)
        for line in lines[:3]:
            scale = get_fit_scale(line, _ix((x2 - x1) * 0.88),
                                  base_scale=0.40, thickness=1, min_scale=0.28)
            draw_outlined_text(frame, line, x1 + _ix(w * 0.035), desc_line_y,
                               scale, COL_TEXT_ACCENT, thickness=1, outline=2)
            desc_line_y += desc_line_gap

    hint_bottom = "Changes save automatically"
    if nav_active and x_zone != "center":
        hint_bottom = f"{'<  Decreasing...' if x_zone == 'minus' else 'Increasing...  >'}  hold to continue"

    # Voice model download hint — shown when voice_model item is selected
    sel_item = settings_schema[selected_index] if selected_index < len(settings_schema) else {}
    if sel_item.get("key") == "voice_model":
        val = config.get("voice_model", "US English")
        if val == "US English":
            hint_bottom = "Download: alphacephei.com/vosk/models  ->  vosk-model-small-en-us-0.15"
        else:
            hint_bottom = "Download: alphacephei.com/vosk/models  ->  vosk-model-small-en-in-0.4"

    draw_bottom_bar(frame, hint_bottom)


# ============================================================
# FEATURES SCREEN
# ============================================================

def draw_features_screen(frame, features_schema, selected_index, config,
                         cursor_info=None):
    """
    Optional feature toggles — separate from program settings.
    All features default to OFF to preserve performance.
    """
    layout = _settings_layout(frame)
    w, h = layout["w"], layout["h"]
    x1, y1, x2, y2 = layout["panel"]

    nav_active = cursor_info is not None and cursor_info.get("active", False)
    x_zone     = cursor_info.get("x_zone", "center") if nav_active else "center"
    adjust_pct = cursor_info.get("adjust_pct", 0.0)  if nav_active else 0.0

    draw_panel(frame, 0, 0, w - 1, h - 1, fill=COL_BG_DARK, alpha=0.75,
               border=COL_BG_DARK, border_thickness=0)

    hint = "UP/DOWN Select | ENTER Toggle | BACK" if not nav_active else \
           "Move left ◄ −  |  center = select  |  + ► move right"
    draw_top_bar(frame, "FEATURES", hint)

    draw_panel(frame, x1, y1, x2, y2, fill=COL_BG_PANEL, alpha=0.94,
               border=COL_YELLOW, border_thickness=2)

    draw_centered_text(frame, "OPTIONAL FEATURES", y1 + _ix((y2 - y1) * 0.07),
                       0.80, COL_YELLOW, thickness=3, outline=4)
    draw_centered_text(frame, "All OFF by default  |  enable what you need",
                       y1 + _ix((y2 - y1) * 0.13), 0.38, COL_TEXT_DIM,
                       thickness=1, outline=2)

    n_items = len(features_schema)
    start_y = y1 + _ix((y2 - y1) * 0.20)
    row_gap = _ix((y2 - y1) * 0.063)
    bar_half = _ix(h * 0.017)

    for i, item in enumerate(features_schema):
        selected  = (i == selected_index)
        y         = start_y + i * row_gap
        key       = item.get("key", "")
        is_back   = (key == "__back__")
        is_choice = item.get("type") == "choice"

        if selected:
            draw_panel(frame, x1 + _ix(w * 0.015), y - bar_half,
                       x2 - _ix(w * 0.015), y + bar_half,
                       fill=(20, 40, 10), alpha=0.60, border=COL_YELLOW, border_thickness=1)

        label_color = COL_YELLOW if selected else COL_TEXT_DIM
        prefix      = "> " if selected else "  "
        draw_outlined_text(frame, f"{prefix}{item['label']}", x1 + _ix(w * 0.025), y,
                           0.48, label_color, thickness=1, outline=2)

        if not is_back:
            if is_choice and selected:
                # [−] value [+] — same as settings screen, yellow theme
                val_text = str(config.get(key, item.get("options", ["?"])[0]))
                btn_w  = _ix(w * 0.055)
                gap    = _ix(w * 0.010)
                plus_x2  = x2 - _ix(w * 0.018)
                plus_x1  = plus_x2 - btn_w
                minus_x2 = plus_x1 - gap
                minus_x1 = minus_x2 - btn_w

                minus_on = nav_active and x_zone == "minus"
                plus_on  = nav_active and x_zone == "plus"

                for btn_x1, btn_x2, label, is_on in [
                    (minus_x1, minus_x2, "-", minus_on),
                    (plus_x1,  plus_x2,  "+", plus_on),
                ]:
                    cx = (btn_x1 + btn_x2) // 2
                    col = COL_YELLOW if is_on else (80, 80, 30)
                    draw_panel(frame, btn_x1, y - bar_half, btn_x2, y + bar_half,
                               fill=(30, 30, 5) if is_on else (10, 10, 5),
                               alpha=0.85, border=col, border_thickness=1 if not is_on else 2)
                    if is_on and adjust_pct > 0:
                        ang = int(360 * adjust_pct)
                        r = int(80 + 175 * adjust_pct); g = int(255 * (1 - adjust_pct * 0.8))
                        cv2.ellipse(frame, (cx, y), (bar_half, bar_half), -90, 0, ang, (0, g, r), 2)
                    txt_col = (10, 10, 10) if is_on else (140, 140, 60)
                    draw_centered_text_in_rect(frame, label,
                        (btn_x1, y - bar_half, btn_x2, y + bar_half),
                        base_scale=0.70, color=txt_col, thickness=2, outline=0)

                (vw, _), _ = cv2.getTextSize(val_text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
                draw_outlined_text(frame, val_text, minus_x1 - gap - vw, y,
                                   0.46, COL_TEXT_ACCENT, thickness=1, outline=2)
            else:
                val_text  = str(config.get(key, item.get("options", ["?"])[0])) \
                            if is_choice else ("ON" if config.get(key, False) else "OFF")
                pill_col  = COL_TEXT_ACCENT if is_choice else \
                            (COL_GREEN if config.get(key, False) else (80, 80, 80))
                (tw, _), _ = cv2.getTextSize(val_text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 2)
                draw_outlined_text(frame, val_text, x2 - tw - _ix(w * 0.025), y,
                                   0.46, pill_col, thickness=2, outline=2)

    # Description box
    desc_y1 = y2 - _ix((y2 - y1) * 0.22)
    desc_y2 = y2 - _ix((y2 - y1) * 0.03)
    sel_item  = features_schema[selected_index] if selected_index < n_items else None
    desc_text = sel_item.get("desc", "") if sel_item else ""

    if desc_text:
        draw_panel(frame, x1 + _ix(w * 0.015), desc_y1, x2 - _ix(w * 0.015), desc_y2,
                   fill=(8, 20, 8), alpha=0.90, border=COL_YELLOW, border_thickness=1)
        max_chars = max(30, int((x2 - x1) / (_ix(w * 0.012) + 1)))
        words, lines, cur = desc_text.split(), [], ""
        for word in words:
            test = f"{cur} {word}".strip()
            if len(test) <= max_chars: cur = test
            else:
                if cur: lines.append(cur)
                cur = word
        if cur: lines.append(cur)
        dy   = desc_y1 + _ix((desc_y2 - desc_y1) * 0.30)
        dgap = _ix((desc_y2 - desc_y1) * 0.30)
        for line in lines[:3]:
            sc = get_fit_scale(line, _ix((x2-x1)*0.88), base_scale=0.40, thickness=1, min_scale=0.28)
            draw_outlined_text(frame, line, x1 + _ix(w * 0.035), dy,
                               sc, COL_TEXT_ACCENT, thickness=1, outline=2)
            dy += dgap

    hint_bottom = "Enter / RIGHT to toggle  |  LEFT / RIGHT for Input Mode  |  Auto-saves"
    if nav_active and x_zone != "center":
        hint_bottom = f"{'◄  Decreasing...' if x_zone == 'minus' else 'Increasing...  ►'}  hold to continue"
    draw_bottom_bar(frame, hint_bottom)


# ============================================================
# CLONE SETUP SCREEN
# ============================================================

def draw_clone_setup_screen(frame, clone_state):
    layout = _menu_layout(frame)
    w, h = layout["w"], layout["h"]
    x1, y1, x2, y2 = layout["panel"]

    draw_panel(frame, 0, 0, w - 1, h - 1, fill=COL_BG_DARK, alpha=0.75,
               border=COL_BG_DARK, border_thickness=0)

    step = clone_state.get("step", "enter_name")

    if step == "enter_name":
        draw_top_bar(frame, "CLONE MODE", "Type name | Enter confirm | ESC Back")
        draw_panel(frame, x1, y1, x2, y2, fill=COL_BG_PANEL, alpha=0.94,
                   border=COL_MAGENTA, border_thickness=2)

        draw_centered_text(frame, "CLONE MODE", y1 + _ix((y2 - y1) * 0.09),
                           0.90, COL_MAGENTA, thickness=3, outline=4)

        draw_centered_text(frame, "Play against an AI clone of a real player.",
                           y1 + _ix((y2 - y1) * 0.22), 0.44, COL_TEXT_DIM, thickness=1, outline=2)
        draw_centered_text(frame, "Your rounds will be recorded so",
                           y1 + _ix((y2 - y1) * 0.30), 0.42, COL_TEXT_DIM, thickness=1, outline=2)
        draw_centered_text(frame, "others can clone YOU too.",
                           y1 + _ix((y2 - y1) * 0.37), 0.42, COL_TEXT_DIM, thickness=1, outline=2)

        draw_centered_text(frame, "YOUR NAME:", y1 + _ix((y2 - y1) * 0.52),
                           0.56, COL_TEXT, thickness=2, outline=3)

        name_text = clone_state.get("text_buffer", "")
        display = f"> {name_text}_"
        draw_centered_text(frame, display, y1 + _ix((y2 - y1) * 0.65),
                           0.78, COL_MAGENTA, thickness=2, outline=3)

        msg = clone_state.get("message", "")
        if msg:
            draw_centered_text(frame, msg, y1 + _ix((y2 - y1) * 0.80),
                               0.42, COL_ORANGE, thickness=1, outline=2)

        draw_bottom_bar(frame, "Type your name, then press Enter")

    elif step == "select_opponent":
        draw_top_bar(frame, "SELECT OPPONENT", "UP/DOWN Move | SELECT | BACK")
        draw_panel(frame, x1, y1, x2, y2, fill=COL_BG_PANEL, alpha=0.94,
                   border=COL_MAGENTA, border_thickness=2)

        player_name = clone_state.get("player_name", "")
        draw_centered_text(frame, f"Playing as: {player_name}", y1 + _ix((y2 - y1) * 0.07),
                           0.48, COL_TEXT_DIM, thickness=1, outline=2)
        draw_centered_text(frame, "SELECT OPPONENT", y1 + _ix((y2 - y1) * 0.17),
                           0.70, COL_MAGENTA, thickness=2, outline=3)
        draw_centered_text(frame, "Best of 5 vs their AI clone",
                           y1 + _ix((y2 - y1) * 0.26), 0.42, COL_TEXT_DIM, thickness=1, outline=2)

        available = clone_state.get("available", [])
        selected_idx = clone_state.get("selected_index", 0)
        item_top = y1 + _ix((y2 - y1) * 0.36)
        item_gap = _ix((y2 - y1) * 0.09)

        for i, (name, count) in enumerate(available):
            selected = i == selected_idx
            cy = item_top + i * item_gap

            if selected:
                bar_y1 = cy - _ix(h * 0.020)
                bar_y2 = cy + _ix(h * 0.020)
                draw_panel(frame, x1 + _ix(w * 0.04), bar_y1, x2 - _ix(w * 0.04), bar_y2,
                           fill=(30, 15, 40), alpha=0.70, border=COL_MAGENTA, border_thickness=1)

            color = COL_MAGENTA if selected else COL_TEXT_DIM
            prefix = "> " if selected else "  "
            label = f"{prefix}{name} ({count} rounds)"
            draw_centered_text_in_rect(frame, label,
                (x1 + _ix(w * 0.06), cy - _ix(h * 0.018), x2 - _ix(w * 0.06), cy + _ix(h * 0.018)),
                base_scale=0.64, color=color, thickness=2, outline=3)

        msg = clone_state.get("message", "")
        if msg:
            draw_centered_text(frame, msg, y2 - _ix((y2 - y1) * 0.08),
                               0.42, COL_ORANGE, thickness=1, outline=2)

        # Progress chip while generate_all_player_reports runs in background
        if clone_state.get("profiles_updating"):
            t = time.monotonic()
            dots = "." * (1 + int(t * 2) % 3)
            draw_outlined_text(frame, f"Updating profiles{dots}",
                               x1 + _ix(w * 0.025), y2 - _ix((y2 - y1) * 0.14),
                               0.36, COL_CYAN, thickness=1, outline=2)

        draw_bottom_bar(frame, "The AI learns from their recorded play style")

    elif step == "no_profiles":
        draw_top_bar(frame, "CLONE MODE", "Enter/ESC to go back")
        draw_panel(frame, x1, y1, x2, y2, fill=COL_BG_PANEL, alpha=0.94,
                   border=COL_MAGENTA, border_thickness=2)

        draw_centered_text(frame, "NO CLONES AVAILABLE YET",
                           y1 + _ix((y2 - y1) * 0.14), 0.72, COL_ORANGE, thickness=2, outline=3)

        draw_centered_text(frame, "To create a clone, a player needs to:",
                           y1 + _ix((y2 - y1) * 0.28), 0.46, COL_TEXT_DIM, thickness=1, outline=2)
        draw_centered_text(frame, "1. Enter their name here",
                           y1 + _ix((y2 - y1) * 0.38), 0.48, COL_TEXT, thickness=1, outline=2)
        draw_centered_text(frame, "2. Play 30+ rounds in any mode",
                           y1 + _ix((y2 - y1) * 0.47), 0.48, COL_TEXT, thickness=1, outline=2)
        draw_centered_text(frame, "3. Patterns are learned automatically",
                           y1 + _ix((y2 - y1) * 0.56), 0.48, COL_TEXT, thickness=1, outline=2)

        all_players = clone_state.get("all_players", [])
        if all_players:
            draw_centered_text(frame, "Players recording:", y1 + _ix((y2 - y1) * 0.70),
                               0.42, COL_TEXT_DIM, thickness=1, outline=2)
            for i, (name, count) in enumerate(all_players[:4]):
                draw_centered_text(frame, f"{name}: {count}/30 rounds",
                    y1 + _ix((y2 - y1) * (0.78 + i * 0.07)),
                    0.46, COL_CYAN, thickness=1, outline=2)

        draw_bottom_bar(frame, "Play Fair Play or Challenge to record rounds")

# ============================================================
# PLAYER STATS SCREEN
# ============================================================

def _draw_bar(frame, x, y, width, height, fill_pct, bar_color, bg_color=(30, 30, 50)):
    """Draw a horizontal progress bar."""
    x, y, width, height = int(x), int(y), int(width), int(height)
    cv2.rectangle(frame, (x, y), (x + width, y + height), bg_color, -1)
    fill_w = max(1, int(width * min(fill_pct, 1.0)))
    if fill_pct > 0.01:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + height), bar_color, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (60, 60, 80), 1)


def draw_player_stats_screen(frame, stats_state):
    """Draw the Player Stats viewer."""
    layout = _menu_layout(frame)
    w, h = layout["w"], layout["h"]
    x1, y1, x2, y2 = layout["panel"]

    draw_panel(frame, 0, 0, w - 1, h - 1, fill=COL_BG_DARK, alpha=0.75,
               border=COL_BG_DARK, border_thickness=0)

    step = stats_state.get("step", "select")

    if step == "select":
        draw_top_bar(frame, "PLAYER STATS", "UP/DOWN Move | SELECT | BACK")
        draw_panel(frame, x1, y1, x2, y2, fill=COL_BG_PANEL, alpha=0.94,
                   border=COL_YELLOW, border_thickness=2)

        draw_centered_text(frame, "SELECT PLAYER", y1 + _ix((y2 - y1) * 0.10),
                           0.80, COL_YELLOW, thickness=3, outline=4)

        players = stats_state.get("players", [])
        selected_idx = stats_state.get("selected_index", 0)
        item_top = y1 + _ix((y2 - y1) * 0.26)
        item_gap = _ix((y2 - y1) * 0.09)

        for i, (name, count) in enumerate(players):
            selected = i == selected_idx
            cy = item_top + i * item_gap

            if selected:
                bar_y1 = cy - _ix(h * 0.020)
                bar_y2 = cy + _ix(h * 0.020)
                draw_panel(frame, x1 + _ix(w * 0.04), bar_y1, x2 - _ix(w * 0.04), bar_y2,
                           fill=(40, 35, 10), alpha=0.70, border=COL_YELLOW, border_thickness=1)

            color = COL_YELLOW if selected else COL_TEXT_DIM
            prefix = "> " if selected else "  "
            label = f"{prefix}{name} ({count} rounds)"
            draw_centered_text_in_rect(frame, label,
                (x1 + _ix(w * 0.06), cy - _ix(h * 0.018), x2 - _ix(w * 0.06), cy + _ix(h * 0.018)),
                base_scale=0.64, color=color, thickness=2, outline=3)

        draw_bottom_bar(frame, "View play patterns and strategy analysis")
        return

    # --- VIEW MODE ---
    data    = stats_state.get("data")
    traits  = stats_state.get("traits", [])
    cur_tab = stats_state.get("tab", "overview")
    cur_flt = stats_state.get("filter", "All")

    # Always draw the panel and top bar with full navigation hints
    player_name_for_header = (data.get("player_name", "Unknown") if data else
                              stats_state.get("player_name_hint", "PLAYER"))
    draw_top_bar(frame, f"STATS: {player_name_for_header.upper()}",
                 "ESC Back | T Tab | A/D Filter | X Export")
    draw_panel(frame, x1, y1, x2, y2, fill=COL_BG_PANEL, alpha=0.94,
               border=COL_YELLOW, border_thickness=2)

    pw, ph = x2 - x1, y2 - y1

    # ── Mode filter strip — ALWAYS drawn so user can still navigate ──────
    _FILTERS = ["All", "FairPlay", "Challenge", "Cheat", "Clone"]
    strip_y1 = y1 + _ix(ph * 0.01)
    strip_y2 = y1 + _ix(ph * 0.10)
    strip_w  = (x2 - x1) // len(_FILTERS)
    for fi, flab in enumerate(_FILTERS):
        fx1 = x1 + fi * strip_w
        fx2 = fx1 + strip_w
        active = (flab == cur_flt)
        fill   = (20, 55, 20) if active else (10, 15, 10)
        border = COL_GREEN if active else (40, 60, 40)
        draw_panel(frame, fx1 + 2, strip_y1 + 2, fx2 - 2, strip_y2 - 2,
                   fill=fill, alpha=0.85, border=border, border_thickness=1)
        col = COL_GREEN if active else COL_TEXT_DIM
        sc  = get_fit_scale(flab, strip_w - 8, base_scale=0.38, thickness=1, min_scale=0.24)
        draw_centered_text_in_rect(frame, flab,
            (fx1 + 2, strip_y1 + 2, fx2 - 2, strip_y2 - 2),
            base_scale=sc, color=col, thickness=1, outline=2)

    # ── Tab strip — ALWAYS drawn ─────────────────────────────────────────
    _TABS  = [("overview", "Overview"), ("history", "Match History")]
    tab_y1 = strip_y2 + _ix(ph * 0.005)
    tab_y2 = tab_y1 + _ix(ph * 0.08)
    tab_w  = (x2 - x1) // len(_TABS)
    for ti, (tid, tlab) in enumerate(_TABS):
        tx1    = x1 + ti * tab_w
        tx2    = tx1 + tab_w
        active = (tid == cur_tab)
        fill   = (15, 45, 55) if active else (8, 15, 20)
        border = COL_CYAN if active else (30, 50, 60)
        draw_panel(frame, tx1 + 2, tab_y1 + 2, tx2 - 2, tab_y2 - 2,
                   fill=fill, alpha=0.90, border=border, border_thickness=1 if active else 0)
        col = COL_CYAN if active else COL_TEXT_DIM
        draw_centered_text_in_rect(frame, tlab,
            (tx1 + 4, tab_y1 + 2, tx2 - 4, tab_y2 - 2),
            base_scale=0.40, color=col, thickness=1, outline=2)

    content_y1 = tab_y2 + _ix(ph * 0.01)
    body_y1    = content_y1 + _ix((y2 - content_y1) * 0.09)

    # ── No data for this filter — show helpful guidance inside content area
    if data is None:
        mid_y = content_y1 + _ix((y2 - content_y1) * 0.30)
        if cur_flt == "All":
            draw_centered_text(frame, "No rounds recorded yet",
                               mid_y, 0.60, COL_ORANGE, thickness=2, outline=3)
            mid_y += _ix((y2 - content_y1) * 0.12)
            for line in [
                "Play Fair Play or Challenge to record rounds",
                "Make sure your Player Name is set in Settings",
                "Stats build up automatically as you play",
            ]:
                draw_centered_text(frame, line, mid_y, 0.40, COL_TEXT_DIM, thickness=1, outline=2)
                mid_y += _ix((y2 - content_y1) * 0.09)
        else:
            mode_hints = {
                "FairPlay":  "Play Fair Play mode to record rounds here",
                "Challenge": "Play Challenge mode to record rounds here",
                "Cheat":     "Play Cheat mode to record rounds here",
                "Clone":     "Play Clone mode to record rounds here",
            }
            draw_centered_text(frame, f"No {cur_flt} data yet",
                               mid_y, 0.60, COL_ORANGE, thickness=2, outline=3)
            mid_y += _ix((y2 - content_y1) * 0.12)
            draw_centered_text(frame, mode_hints.get(cur_flt, "Play some rounds in this mode"),
                               mid_y, 0.42, COL_TEXT_DIM, thickness=1, outline=2)
            mid_y += _ix((y2 - content_y1) * 0.10)
            draw_centered_text(frame, "Use  A / D  to switch to All for combined stats",
                               mid_y, 0.40, COL_TEXT_ACCENT, thickness=1, outline=2)
        draw_bottom_bar(frame, "A / D Filter mode  |  T Switch tab  |  ESC Back")
        return

    # ── Round count + filter note ─────────────────────────────────────────
    flt_note = f"  ({cur_flt})" if cur_flt != "All" else ""
    draw_centered_text(frame, f"{data['round_count']} rounds{flt_note}",
                       content_y1 + _ix((y2 - content_y1) * 0.04),
                       0.42, COL_TEXT_DIM, thickness=1, outline=2)

    name = data.get("player_name", "Unknown")

    # ════════════════════════════════════════════════════════════════════
    # OVERVIEW TAB
    # ════════════════════════════════════════════════════════════════════
    if cur_tab == "overview":
        col_left_x  = x1 + _ix(pw * 0.04)
        col_right_x = x1 + _ix(pw * 0.54)
        bar_w       = _ix(pw * 0.38)
        bar_h       = _ix(ph * 0.028)
        body_ph     = y2 - body_y1

        # Left: Results
        sec_y = body_y1
        draw_outlined_text(frame, "RESULTS", col_left_x, sec_y,
                           0.48, COL_TEXT, thickness=2, outline=3)
        sec_y += _ix(body_ph * 0.07)
        for label, pct, color in [
            ("Win",  data["win_pct"],  COL_GREEN),
            ("Loss", data["loss_pct"], COL_RED),
            ("Draw", data["draw_pct"], COL_TEXT_DIM),
        ]:
            draw_outlined_text(frame, f"{label}: {pct:.0%}", col_left_x, sec_y,
                               0.44, color, thickness=2, outline=3)
            _draw_bar(frame, col_left_x + _ix(pw * 0.18), sec_y - bar_h + 2,
                      bar_w - _ix(pw * 0.18), bar_h, pct, color)
            sec_y += _ix(body_ph * 0.065)

        # Left: Gestures
        sec_y += _ix(body_ph * 0.02)
        draw_outlined_text(frame, "GESTURES", col_left_x, sec_y,
                           0.48, COL_TEXT, thickness=2, outline=3)
        sec_y += _ix(body_ph * 0.07)
        freq = data.get("gesture_freq", {})
        for g in ("Rock", "Paper", "Scissors"):
            pct   = freq.get(g, 0)
            color = get_gesture_color(g)
            draw_outlined_text(frame, f"{g}: {pct:.0%}", col_left_x, sec_y,
                               0.44, color, thickness=2, outline=3)
            _draw_bar(frame, col_left_x + _ix(pw * 0.22), sec_y - bar_h + 2,
                      bar_w - _ix(pw * 0.22), bar_h, pct, color)
            sec_y += _ix(body_ph * 0.065)

        # Right: After-outcome response
        sec_y2 = body_y1
        draw_outlined_text(frame, "AFTER OUTCOME", col_right_x, sec_y2,
                           0.48, COL_TEXT, thickness=2, outline=3)
        sec_y2 += _ix(body_ph * 0.07)
        for outcome in ("win", "lose", "draw"):
            resp = data.get("outcome_response", {}).get(outcome, {})
            stay = resp.get("stay", 0)
            up   = resp.get("upgrade", 0)
            down = resp.get("downgrade", 0)
            line = f"{outcome.title()}: stay {stay:.0%} | up {up:.0%} | dn {down:.0%}"
            avail_w = x2 - col_right_x - _ix(pw * 0.04)
            sc = get_fit_scale(line, avail_w, base_scale=0.40, thickness=1, min_scale=0.26)
            draw_outlined_text(frame, line, col_right_x, sec_y2,
                               sc, COL_TEXT_ACCENT, thickness=1, outline=2)
            sec_y2 += _ix(body_ph * 0.065)

        # Traits
        traits_y = y2 - _ix(ph * 0.26)
        cv2.line(frame, (x1 + _ix(pw * 0.04), traits_y - _ix(ph * 0.01)),
                 (x2 - _ix(pw * 0.04), traits_y - _ix(ph * 0.01)), COL_YELLOW, 1)
        draw_outlined_text(frame, "PLAYER TRAITS", col_left_x, traits_y,
                           0.48, COL_TEXT, thickness=2, outline=3)
        traits_y += _ix(ph * 0.055)
        for trait in traits[:3]:
            sc = get_fit_scale(trait, _ix(pw * 0.90), base_scale=0.40, thickness=1, min_scale=0.26)
            draw_outlined_text(frame, trait, col_left_x, traits_y,
                               sc, COL_CYAN, thickness=1, outline=2)
            traits_y += _ix(ph * 0.048)

        # History dots row
        rounds = stats_state.get("rounds", [])
        if rounds:
            draw_round_history_dots(frame, rounds,
                                    x1 + _ix(pw * 0.04), y2 - _ix(ph * 0.06),
                                    x2 - _ix(pw * 0.04))

    # ════════════════════════════════════════════════════════════════════
    # HISTORY TAB
    # ════════════════════════════════════════════════════════════════════
    else:
        sessions = stats_state.get("sessions", [])
        col_left_x = x1 + _ix(pw * 0.04)
        body_ph    = y2 - body_y1

        if not sessions:
            draw_centered_text(frame, "No session history yet",
                               body_y1 + _ix(body_ph * 0.40),
                               0.55, COL_TEXT_DIM, thickness=1, outline=2)
        else:
            # Header row
            hdr_y = body_y1 + _ix(body_ph * 0.04)
            for txt, xpct in [("Date / Time", 0.04), ("Mode", 0.30), ("Score", 0.47),
                               ("Win%", 0.62), ("Avg RT", 0.76)]:
                draw_outlined_text(frame, txt, x1 + _ix(pw * xpct), hdr_y,
                                   0.36, COL_CYAN, thickness=1, outline=2)
            cv2.line(frame,
                     (x1 + _ix(pw * 0.03), hdr_y + _ix(body_ph * 0.045)),
                     (x2 - _ix(pw * 0.03), hdr_y + _ix(body_ph * 0.045)),
                     COL_CYAN, 1)

            row_h  = _ix(body_ph * 0.13)
            row_y  = hdr_y + _ix(body_ph * 0.07)
            for sess in reversed(sessions):   # most recent first
                w_rate = sess.get("win_rate", 0)
                wins   = sess.get("wins", 0)
                losses = sess.get("losses", 0)
                draws  = sess.get("draws", 0)
                rt     = sess.get("avg_reaction_ms")
                mode   = sess.get("mode", "?")
                date   = sess.get("date", "?")

                row_col = COL_GREEN if w_rate >= 0.5 else (COL_RED if w_rate < 0.35 else COL_TEXT_ACCENT)

                for txt, xpct in [
                    (date, 0.04),
                    (mode, 0.30),
                    (f"{wins}W {losses}L {draws}D", 0.47),
                    (f"{w_rate:.0%}", 0.62),
                    (f"{rt}ms" if rt else "n/a", 0.76),
                ]:
                    sc = get_fit_scale(txt, _ix(pw * 0.22),
                                       base_scale=0.38, thickness=1, min_scale=0.24)
                    draw_outlined_text(frame, txt, x1 + _ix(pw * xpct), row_y,
                                       sc, row_col, thickness=1, outline=2)
                row_y += row_h
                if row_y > y2 - _ix(ph * 0.08):
                    break

    draw_bottom_bar(frame, "A / D Filter  |  T Switch tab  |  X Export CSV  |  ESC Back")

# ============================================================
# TUTORIAL SCREEN
# ============================================================

def draw_tutorial_screen(frame, tut_state):
    """Draw the interactive tutorial overlaid on the camera feed."""
    w, h = _frame_size(frame)

    step = tut_state.get("step", {})
    step_idx = tut_state.get("step_index", 0)
    total = tut_state.get("total_steps", 6)
    detected = tut_state.get("detected_gesture", "Unknown")
    hold = tut_state.get("hold_count", 0)
    hold_needed = tut_state.get("hold_needed", 10)
    pump_count = tut_state.get("pump_count", 0)
    shot = tut_state.get("shot_gesture")
    voice_mode = tut_state.get("voice_mode", False)

    step_id = step.get("id", "")
    title = step.get("title", "")
    instruction = step.get("instruction", "")
    sub = step.get("sub", "")

    mode_label = "VOICE" if voice_mode else "HOW TO PLAY"
    draw_top_bar(frame, mode_label, f"Step {step_idx + 1}/{total} | ESC / BACK to skip")

    # --- Instruction panel ---
    panel_y1 = _ix(h * 0.12)
    panel_y2 = _ix(h * 0.42)
    border_col = (80, 255, 180) if voice_mode else COL_GREEN
    draw_panel(frame, _ix(w * 0.08), panel_y1, _ix(w * 0.92), panel_y2,
               fill=(8, 8, 18), alpha=0.88, border=border_col, border_thickness=2)

    draw_centered_text(frame, title, panel_y1 + _ix((panel_y2 - panel_y1) * 0.18),
                       0.70, border_col, thickness=2, outline=3)

    draw_centered_text(frame, instruction, panel_y1 + _ix((panel_y2 - panel_y1) * 0.50),
                       0.90, COL_TEXT, thickness=2, outline=4)

    draw_centered_text(frame, sub, panel_y1 + _ix((panel_y2 - panel_y1) * 0.78),
                       0.44, COL_TEXT_DIM, thickness=1, outline=2)

    # --- Status panel ---
    status_y1 = _ix(h * 0.78)
    status_y2 = _ix(h * 0.93)
    draw_panel(frame, _ix(w * 0.08), status_y1, _ix(w * 0.92), status_y2,
               fill=(8, 8, 18), alpha=0.88, border=COL_CYAN, border_thickness=1)

    if voice_mode:
        # Voice mode status panel — show listening indicator and word heard
        if step_id in ("rock", "paper", "scissors"):
            # Pulsing "listening" indicator
            t = time.monotonic()
            pulse = 0.5 + 0.5 * math.sin(t * 4)
            mic_color = tuple(int(c * pulse) for c in (80, 255, 180))
            draw_outlined_text(frame, "Listening...", _ix(w * 0.12),
                               status_y1 + _ix((status_y2 - status_y1) * 0.45),
                               0.52, mic_color, thickness=1, outline=2)
            if detected and detected != "Unknown":
                det_color = border_col if detected == step.get("target_gesture") else COL_ORANGE
                draw_outlined_text(frame, f"Heard: {detected}", _ix(w * 0.55),
                                   status_y1 + _ix((status_y2 - status_y1) * 0.45),
                                   0.52, det_color, thickness=2, outline=3)

        elif step_id == "pump":
            # Show 3 countdown circles (ONE TWO THREE)
            labels = ["ONE", "TWO", "THREE"]
            draw_outlined_text(frame, "Words spoken:", _ix(w * 0.12),
                               status_y1 + _ix((status_y2 - status_y1) * 0.45),
                               0.44, COL_TEXT_DIM, thickness=1, outline=2)
            for i, label in enumerate(labels):
                cx = _ix(w * (0.55 + i * 0.12))
                cy = status_y1 + _ix((status_y2 - status_y1) * 0.50)
                r  = _ix(min(w, h) * 0.022)
                active = i < pump_count
                color  = (80, 255, 180) if active else (60, 60, 80)
                thick  = -1 if active else 2
                cv2.circle(frame, (cx, cy), r, color, thick)
                cv2.putText(frame, label, (cx - _ix(w * 0.025), cy + r + _ix(h * 0.025)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                            (80, 255, 180) if active else (80, 80, 80), 1, cv2.LINE_AA)

        elif step_id == "shoot":
            t = time.monotonic()
            pulse = 0.5 + 0.5 * math.sin(t * 4)
            mic_color = tuple(int(c * pulse) for c in (80, 255, 180))
            draw_outlined_text(frame, "Say your throw now...", _ix(w * 0.12),
                               status_y1 + _ix((status_y2 - status_y1) * 0.45),
                               0.52, mic_color, thickness=1, outline=2)

        elif step_id == "done":
            if shot:
                draw_centered_text(frame, f'You said "{shot}"!',
                                   status_y1 + _ix((status_y2 - status_y1) * 0.30),
                                   0.52, get_gesture_color(shot), thickness=2, outline=3)
            draw_centered_text(frame, 'Say "SELECT" to return to menu',
                               status_y1 + _ix((status_y2 - status_y1) * 0.72),
                               0.46, COL_TEXT_DIM, thickness=1, outline=2)

    else:
        # Physical mode — original status panel logic
        if step_id in ("rock", "paper", "scissors"):
            target = step.get("target_gesture", "")
            det_color = COL_GREEN if detected == target else COL_ORANGE

            draw_outlined_text(frame, f"Detected: {detected}", _ix(w * 0.12),
                               status_y1 + _ix((status_y2 - status_y1) * 0.45),
                               0.54, det_color, thickness=2, outline=3)

            bar_x    = _ix(w * 0.55)
            bar_y    = status_y1 + _ix((status_y2 - status_y1) * 0.25)
            bar_w    = _ix(w * 0.30)
            bar_h_px = _ix((status_y2 - status_y1) * 0.35)
            pct = min(hold / max(hold_needed, 1), 1.0)
            _draw_bar(frame, bar_x, bar_y, bar_w, bar_h_px, pct, COL_GREEN)

            if pct >= 1.0:
                draw_outlined_text(frame, "NICE!", _ix(w * 0.58),
                                   status_y1 + _ix((status_y2 - status_y1) * 0.82),
                                   0.50, COL_GREEN, thickness=2, outline=3)
            else:
                draw_outlined_text(frame, f"Hold it...", _ix(w * 0.58),
                                   status_y1 + _ix((status_y2 - status_y1) * 0.82),
                                   0.42, COL_TEXT_DIM, thickness=1, outline=2)

        elif step_id == "pump":
            draw_outlined_text(frame, f"Pumps: {pump_count} / 4", _ix(w * 0.12),
                               status_y1 + _ix((status_y2 - status_y1) * 0.45),
                               0.54, COL_CYAN, thickness=2, outline=3)

            for i in range(4):
                cx = _ix(w * (0.58 + i * 0.08))
                cy = status_y1 + _ix((status_y2 - status_y1) * 0.50)
                r  = _ix(min(w, h) * 0.022)
                active = i < pump_count
                color  = COL_CYAN if active else (60, 60, 80)
                thick  = -1 if active else 2
                cv2.circle(frame, (cx, cy), r, color, thick)

        elif step_id == "shoot":
            det_color = get_gesture_color(detected)
            draw_outlined_text(frame, f"Throw: {detected}", _ix(w * 0.12),
                               status_y1 + _ix((status_y2 - status_y1) * 0.50),
                               0.54, det_color, thickness=2, outline=3)

            # Show countdown during the 2-second wait, then switch to prompt
            shoot_since = tut_state.get("shoot_visible_since")
            if shoot_since is not None:
                elapsed = time.monotonic() - shoot_since
                remaining = max(0.0, 2.0 - elapsed)
                if remaining > 0.05:
                    draw_outlined_text(frame, f"Get ready...  {remaining:.1f}s",
                                       _ix(w * 0.55), status_y1 + _ix((status_y2 - status_y1) * 0.50),
                                       0.44, COL_YELLOW, thickness=1, outline=2)
                else:
                    draw_outlined_text(frame, "THROW NOW!", _ix(w * 0.55),
                                       status_y1 + _ix((status_y2 - status_y1) * 0.50),
                                       0.48, COL_RED, thickness=2, outline=3)
            else:
                draw_outlined_text(frame, "Change from fist NOW!", _ix(w * 0.55),
                                   status_y1 + _ix((status_y2 - status_y1) * 0.50),
                                   0.44, COL_RED, thickness=1, outline=2)

        elif step_id == "done":
            if shot:
                draw_centered_text(frame, f"You threw {shot}!",
                                   status_y1 + _ix((status_y2 - status_y1) * 0.30),
                                   0.52, get_gesture_color(shot), thickness=2, outline=3)

            draw_centered_text(frame, "Press Enter to return to menu",
                               status_y1 + _ix((status_y2 - status_y1) * 0.72),
                               0.46, COL_TEXT_DIM, thickness=1, outline=2)

    bottom_hint = (
        'Say BACK to exit  |  Speak each word clearly'
        if voice_mode
        else "Your camera feed is live — try the gestures!  |  Say BACK to exit"
    )
    draw_bottom_bar(frame, bottom_hint)


# ============================================================
# EMOTION LANDMARK DEBUG OVERLAY
# ============================================================

def draw_emotion_debug(frame, debug_info):
    """
    Overlay emotion landmark dots + score panel on the RIGHT side of the frame.

    Positioned right-side so it never overlaps the Diagnostic info/game panels.
    Shows calibration progress bar during warmup, then live deviation scores.
    """
    if debug_info is None:
        return

    h, w = frame.shape[:2]

    GROUP_COLORS = {
        "mouth":  (80,  220,  80),
        "eyes":   (255, 200,   0),
        "brows":  (0,   200, 255),
        "anchor": (180, 180, 180),
    }
    GROUP_LABELS = {
        "mouth":  "Mouth",
        "eyes":   "Eyes",
        "brows":  "Brows",
        "anchor": "Ref pts",
    }
    BAR_COLORS = {
        "smile":       (80,  220,  80),
        "surprise":    (0,   200, 255),
        "frustration": (80,   80, 255),
    }
    BAR_LABELS = {
        "smile":       "Smile",
        "surprise":    "Surprise",
        "frustration": "Frustration",
    }
    BAR_THRESHOLDS = {
        "smile":       0.38,
        "surprise":    0.40,
        "frustration": 0.42,
    }
    EM_COLORS = {
        "Happy":      (80,  220,  80),
        "Surprised":  (0,   200, 255),
        "Frustrated": (80,   80, 255),
        "Neutral":    (160, 160, 160),
    }

    bar_max_w = 120
    bar_h_px  = 13
    bar_gap   = 20
    pad       = 8
    panel_w   = bar_max_w + 110
    panel_x   = w - panel_w - 6

    calibrated = debug_info.get("calibrated", True)
    cal_prog   = debug_info.get("calibration_progress", 100)

    n_bars   = len(BAR_LABELS)
    panel_h  = 14 + 16 + 16 + 8 + (n_bars * bar_gap) + 42
    if not calibrated:
        panel_h += 24
    panel_y = 6

    # Background panel
    cv2.rectangle(frame,
                  (panel_x - pad, panel_y),
                  (panel_x + panel_w, panel_y + panel_h),
                  (0, 0, 0), -1)
    cv2.rectangle(frame,
                  (panel_x - pad, panel_y),
                  (panel_x + panel_w, panel_y + panel_h),
                  (70, 70, 70), 1)

    cy = panel_y + 14

    # Header
    cv2.putText(frame, "FACE DEBUG  (E to hide)", (panel_x, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
    cy += 16

    # Calibration progress bar (shown until calibrated)
    if not calibrated:
        cv2.putText(frame, f"Calibrating...  {cal_prog}%", (panel_x, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 180, 80), 1, cv2.LINE_AA)
        cy += 14
        # Progress bar
        cv2.rectangle(frame, (panel_x, cy), (panel_x + bar_max_w, cy + 8), (30, 30, 30), -1)
        filled = int(cal_prog / 100 * bar_max_w)
        if filled > 0:
            cv2.rectangle(frame, (panel_x, cy), (panel_x + filled, cy + 8), (200, 180, 80), -1)
        cv2.rectangle(frame, (panel_x, cy), (panel_x + bar_max_w, cy + 8), (80, 80, 80), 1)
        cy += 18
        # Note about looking neutral
        cv2.putText(frame, "Look neutral to calibrate", (panel_x, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 150), 1, cv2.LINE_AA)
        cy += 16
    else:
        cv2.putText(frame, "Calibrated  (personal baseline)", (panel_x, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (80, 200, 80), 1, cv2.LINE_AA)
        cy += 16

    # Dot legend (2 rows of 2)
    groups = list(GROUP_COLORS.items())
    for row in range(2):
        rx = panel_x
        for col in range(2):
            idx = row * 2 + col
            if idx >= len(groups):
                break
            gname, gcol = groups[idx]
            cv2.circle(frame, (rx + 5, cy - 3), 4, gcol, -1)
            cv2.putText(frame, GROUP_LABELS[gname], (rx + 13, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, gcol, 1, cv2.LINE_AA)
            rx += (panel_w // 2)
        cy += 15

    # Separator
    cy += 3
    cv2.line(frame, (panel_x - pad, cy), (panel_x + panel_w, cy), (60, 60, 60), 1)
    cy += 8

    # Score bars
    scores = debug_info["scores"]
    for key in ("smile", "surprise", "frustration"):
        val    = scores.get(key, 0.0)
        color  = BAR_COLORS[key]
        label  = BAR_LABELS[key]
        thresh = BAR_THRESHOLDS[key]

        cv2.putText(frame, label, (panel_x, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1, cv2.LINE_AA)
        bx = panel_x + 78
        by = cy + 2

        cv2.rectangle(frame, (bx, by - bar_h_px), (bx + bar_max_w, by), (30, 30, 30), -1)
        filled = int(val * bar_max_w)
        if filled > 0:
            cv2.rectangle(frame, (bx, by - bar_h_px), (bx + filled, by), color, -1)
        cv2.rectangle(frame, (bx, by - bar_h_px), (bx + bar_max_w, by), (80, 80, 80), 1)
        # Threshold tick
        tx = bx + int(thresh * bar_max_w)
        cv2.line(frame, (tx, by - bar_h_px - 2), (tx, by + 2), (220, 220, 220), 1)
        cv2.putText(frame, f"{val:.2f}", (bx + bar_max_w + 4, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1, cv2.LINE_AA)
        cy += bar_gap

    # Separator
    cv2.line(frame, (panel_x - pad, cy), (panel_x + panel_w, cy), (60, 60, 60), 1)
    cy += 10

    # Current emotion label
    emotion    = debug_info["emotion"]
    confidence = debug_info["confidence"]
    em_color   = EM_COLORS.get(emotion, (200, 200, 200))
    cv2.putText(frame, f"{emotion}  {confidence:.0%}", (panel_x, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, em_color, 1, cv2.LINE_AA)

    # Landmark dots — clipped so they don't paint over the diagnostic info panel.
    # Info panel occupies roughly x: 2%–55%, y: 15%–76% of the frame.
    h_f, w_f = frame.shape[:2]
    clip_x2 = int(w_f * 0.56)   # right edge of info panel + small margin
    clip_y1 = int(h_f * 0.14)
    clip_y2 = int(h_f * 0.77)

    for group, pts in debug_info["points"].items():
        dot_color = GROUP_COLORS.get(group, (255, 255, 255))
        for (px, py) in pts:
            # Skip dots that fall inside the info panel region
            if px < clip_x2 and clip_y1 < py < clip_y2:
                continue
            cv2.circle(frame, (px, py), 4, dot_color, -1)
            cv2.circle(frame, (px, py), 5, (0, 0, 0), 1)


# ============================================================
# GESTURE NAV OVERLAY
# ============================================================

def draw_gesture_nav_overlay(frame, cursor_info):
    """
    Draw the gesture navigation cursor overlay on any nav screen.

    States:
      inactive    — grey dot at fingertip (shows tracking is working)
      warming_up  — white pulsing ring + teal progress arc (counting frames)
      active      — solid cyan circle + dwell arc filling as hover accumulates
    """
    if cursor_info is None:
        return

    h, w = frame.shape[:2]

    active     = cursor_info.get("active", False)
    warming_up = cursor_info.get("warming_up", False)
    warmup_pct = cursor_info.get("warmup_pct", 0.0)
    dwell_pct  = cursor_info.get("dwell_pct", 0.0)
    tip_x      = cursor_info.get("index_tip_x")
    tip_y      = cursor_info.get("index_tip_y")

    if tip_x is None or tip_y is None:
        # No hand detected — just show the hint badge
        badge_text  = "Raise index finger to navigate"
        badge_color = (70, 70, 70)
        badge_x, badge_y = 10, h - 46
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.rectangle(frame, (badge_x - 4, badge_y - th - 4),
                      (badge_x + tw + 6, badge_y + 4), (0, 0, 0), -1)
        cv2.rectangle(frame, (badge_x - 4, badge_y - th - 4),
                      (badge_x + tw + 6, badge_y + 4), badge_color, 1)
        cv2.putText(frame, badge_text, (badge_x, badge_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, badge_color, 1, cv2.LINE_AA)
        return

    px = max(0, min(w - 1, int(tip_x * w)))
    py = max(0, min(h - 1, int(tip_y * h)))

    # ── Cursor circle ──────────────────────────────────────────────────── #
    if active:
        cv2.circle(frame, (px, py), 16, (255, 220, 0), -1)
        cv2.circle(frame, (px, py), 17, (0, 0, 0), 2)

        # Dwell arc: fills as hover time accumulates, shifts cyan → red
        if dwell_pct > 0:
            angle = int(360 * dwell_pct)
            r = int(80  + 175 * dwell_pct)
            g = int(255 * (1 - dwell_pct * 0.8))
            b = int(180 * (1 - dwell_pct))
            cv2.ellipse(frame, (px, py), (22, 22), -90, 0, angle, (b, g, r), 3)

    elif warming_up:
        import time as _t
        pulse    = 0.4 + 0.6 * (_t.monotonic() % 1.0)
        ring_col = tuple(int(c * pulse) for c in (200, 200, 200))
        cv2.circle(frame, (px, py), 16, ring_col, 2)
        if warmup_pct > 0:
            cv2.ellipse(frame, (px, py), (18, 18), -90, 0,
                        int(360 * warmup_pct), (80, 255, 180), 2)

    else:
        # Hand detected but not yet pointing — grey dot so user can
        # see tracking is live and confirm coordinate mapping is correct
        cv2.circle(frame, (px, py), 8, (90, 90, 90), 1)

    # ── Status badge (bottom-left) ─────────────────────────────────────── #
    badge_x, badge_y = 10, h - 46

    if active:
        item_idx   = cursor_info.get("item_index", 0)
        pct_int    = int(dwell_pct * 100)
        badge_text = f"NAV ACTIVE  |  item {item_idx + 1}  |  hover {pct_int}%"
        badge_color = (255, 220, 0)
    elif warming_up:
        badge_text  = f"Activating...  {int(warmup_pct * 100)}%"
        badge_color = (160, 160, 160)
    else:
        badge_text  = "Show hand to navigate  |  hold still 2s to select"
        badge_color = (90, 90, 90)

    (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.rectangle(frame, (badge_x - 4, badge_y - th - 4),
                  (badge_x + tw + 6, badge_y + 4), (0, 0, 0), -1)
    cv2.rectangle(frame, (badge_x - 4, badge_y - th - 4),
                  (badge_x + tw + 6, badge_y + 4), badge_color, 1)
    cv2.putText(frame, badge_text, (badge_x, badge_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, badge_color, 1, cv2.LINE_AA)

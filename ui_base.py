"""
ui_base.py -- Colours, layout helpers, drawing primitives, text utilities.
All public names are re-exported by ui_renderer.py.
"""
import cv2
import math
import time

# Explicit export list — includes private helpers needed by sibling modules.
# 'from ui_base import *' will pick up all names in __all__.
__all__ = [
    # Colour constants
    'COL_BG_DARK', 'COL_BG_PANEL', 'COL_BG_PANEL_LIGHT',
    'COL_CYAN', 'COL_MAGENTA', 'COL_YELLOW', 'COL_GREEN', 'COL_RED', 'COL_ORANGE',
    'COL_TEXT', 'COL_TEXT_DIM', 'COL_TEXT_ACCENT',
    # Private colour helpers (needed by sibling modules)
    '_COL_CB_WIN', '_COL_CB_LOSE', '_COL_CB_DRAW',
    '_result_colour', '_get_emotion_color',
    # Layout helpers (needed by sibling modules)
    '_ix', '_fit_rect', '_frame_size',
    '_game_layout', '_menu_layout', '_settings_layout',
    # Glow helper
    '_draw_glow_border',
    # Public drawing functions
    'get_gesture_color', 'draw_panel', 'draw_gesture_icon', 'draw_result_flash',
    'draw_gesture_confidence_bar', 'draw_quality_warnings', 'draw_round_history_dots',
    'draw_help_overlay', 'get_fit_scale', 'draw_outlined_text',
    'draw_centered_text', 'draw_centered_text_in_rect',
    'draw_top_bar', 'draw_bottom_bar',
    'draw_status_chip', 'get_result_banner_color',
]


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
    # Copy only the ROI sub-rect - ~10-50x cheaper than copying the full frame
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
    Vertical gesture lock bar on the left edge - fills upward as gesture stabilises.
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

    # Label rotated 90deg using individual chars stacked vertically
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
    ? key - full shortcut overlay, differentiated by screen and input mode.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    # -- Bindings per screen --------------------------------------------------
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
            ("SELECT / YES",   "Confirm"),
            ("BACK / NO",      "Go back"),
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
    elif screen_name == "GAME_NONRPS":
        section_title = "VOICE COMMANDS  |  NON-RPS MODES"
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
            ("(Finger only for dot capture)",""),
            ("", ""),
            ("Simon Says:",""),
            ("BACK",            "Exit to menu"),
            ("(Gesture only for matching)",""),
            ("", ""),
            ("Arcade Snake:",""),
            ("(Gesture-only mode)",""),
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

    # -- Title ----------------------------------------------------------------
    draw_centered_text(frame, section_title,
                       _ix(h * 0.08), 0.62, COL_YELLOW, thickness=2, outline=3)
    cv2.line(frame, (_ix(w * 0.05), _ix(h * 0.13)), (w - _ix(w * 0.05), _ix(h * 0.13)),
             COL_CYAN, 1)

    # -- Two-column layout ----------------------------------------------------
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
    for _ in range(8):  # binary search - ~3x faster than 0.05 linear steps
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


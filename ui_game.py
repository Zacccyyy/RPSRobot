"""
ui_game.py -- In-game screen renderers: canonical RPS view, result, session summary,
diagnostic panels.
"""
import cv2
import math
import time
from ui_base import *

# ============================================================
# HELPER: STATE PILL (spec §03 panel header)
# ============================================================

def _draw_state_pill(frame, state_str, cx, cy):
    """
    Bordered pill centred at (cx, cy).
    Maps any game-state / banner string to a short label + colour.
    """
    gs = (state_str or "").upper()
    if any(w in gs for w in ('WIN', 'SURVIVE', 'WON')):
        label, color = 'WIN',   COL_GREEN
    elif any(w in gs for w in ('LOSS', 'LOSE', 'LOST')):
        label, color = 'LOSS',  COL_RED
    elif 'DRAW' in gs:
        label, color = 'DRAW',  COL_TEXT_PRIMARY
    elif any(w in gs for w in ('SHOOT', 'THROW')):
        label, color = 'SHOOT', COL_RED
    elif any(w in gs for w in ('BEAT', 'COUNT')):
        label, color = 'COUNTING', COL_ACCENT
    elif 'VOICE' in gs:
        label, color = 'VOICE', COL_GREEN
    elif any(w in gs for w in ('RESULT', 'REVEAL', 'SHOW')):
        label, color = 'RESULT', COL_TEXT_SECONDARY
    else:
        label, color = 'READY', COL_ACCENT

    w = frame.shape[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, SCALE_CAPTION, 1)
    pad_x = _ix(w * 0.014)
    pad_y = 4
    x1 = cx - tw // 2 - pad_x
    y1 = cy - th - pad_y
    x2 = cx + tw // 2 + pad_x
    y2 = cy + pad_y
    draw_panel(frame, x1, y1, x2, y2,
               fill=COL_PANEL_BG, alpha=0.90, border=color, border_thickness=1)
    draw_centered_text_in_rect(frame, label, (x1, y1, x2, y2),
                               base_scale=SCALE_CAPTION, color=color,
                               thickness=1, outline=2)

# ============================================================
# HELPER: STREAK ROW (challenge mode)
# ============================================================

def _draw_streak_row(frame, px1, py1, px2, py2, streak, high_streak, total_pips=9):
    """9-pip streak indicator at panel y 65%. Used by challenge mode."""
    row_y  = py1 + _ix((py2 - py1) * 0.65)
    pip_r  = 7
    gap    = 14
    total_w = total_pips * (pip_r * 2) + (total_pips - 1) * gap
    start_x = (px1 + px2) // 2 - total_w // 2 + pip_r

    for i in range(total_pips):
        bx = start_x + i * (pip_r * 2 + gap)
        if i < streak:
            cv2.circle(frame, (bx, row_y), pip_r, COL_ACCENT, -1)
        else:
            cv2.circle(frame, (bx, row_y), pip_r, COL_TEXT_DIM, 1)

    streak_str = f"STREAK {streak}  *  HIGH {high_streak}"
    draw_centered_text(frame, streak_str, row_y + 22,
                       SCALE_MICRO, COL_TEXT_DIM, thickness=1, outline=2)

# ============================================================
# GESTURE INDICATOR ROW (spec §03 y 9-17%)
# ============================================================

def draw_gesture_row(frame, detected_gesture="", gestures=None):
    """
    Row of gesture labels with a dot indicator below each.
    Dot filled accent when gesture matches detected_gesture, else dim ring.
    """
    w, h = _frame_size(frame)
    if gestures is None:
        gestures = ["Rock", "Paper", "Scissors"]

    row_y1 = _ix(h * 0.09)
    row_y2 = _ix(h * 0.17)
    cy     = (row_y1 + row_y2) // 2
    n      = len(gestures)
    gap    = _ix(w * 0.056) if n <= 3 else _ix(w * 0.040)

    # Centre the group
    font     = cv2.FONT_HERSHEY_SIMPLEX
    max_w    = max(cv2.getTextSize(g.upper(), font, SCALE_MICRO, 1)[0][0] for g in gestures)
    step     = max_w + gap
    total_w  = n * step - gap
    start_x  = (w - total_w) // 2 + max_w // 2

    _GESTURE_IDENTITY = {
        'rock':     (160, 120,  80),
        'paper':    ( 80, 160, 200),
        'scissors': ( 80,  80, 200),
    }

    dot_r = 4
    for i, g in enumerate(gestures):
        cx     = start_x + i * step
        active = detected_gesture and g.lower() == detected_gesture.lower()
        label  = g.upper()
        base_col = _GESTURE_IDENTITY.get(g.lower(), COL_TEXT_SECONDARY)
        col      = base_col if active else tuple(int(c * 0.35) for c in base_col)
        (lw, lh), _ = cv2.getTextSize(label, font, SCALE_MICRO, 1)
        cv2.putText(frame, label, (cx - lw // 2, cy - 4),
                    font, SCALE_MICRO,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (cx - lw // 2, cy - 4),
                    font, SCALE_MICRO, col, 1, cv2.LINE_AA)
        dot_y = cy + lh // 2 + 6
        if active:
            cv2.circle(frame, (cx, dot_y), dot_r, base_col, -1)
        else:
            cv2.circle(frame, (cx, dot_y), dot_r, col, 1)

# ============================================================
# SCORE BAR (spec §03 y 18-22%)
# ============================================================

def draw_game_status_strip(frame, game_state):
    """Score bar (y 18-22%) -- no panel border, centred dim text."""
    w, h = _frame_size(frame)
    score_y = _ix(h * 0.21)

    mode = game_state.get("play_mode_label", "")
    if mode.lower() in ("cheat", "cheat mode"):
        text = "Cheat mode counters your throw after SHOOT"
    else:
        r_txt = game_state.get("round_text", "")
        s_txt = game_state.get("score_text", "")
        sep   = "  |  "
        text  = f"{r_txt}{sep}{s_txt}" if r_txt and s_txt else (r_txt or s_txt)

    draw_centered_text(frame, text, score_y,
                       SCALE_BODY, COL_TEXT_SECONDARY, thickness=1, outline=2)

# ============================================================
# DIAGNOSTIC PANELS (dev-only)
# ============================================================

def draw_info_panel(frame, tracker_state, game_state, count_text, status_text,
                    reason_text, ambiguous_count, output_summary,
                    emotion_state=None, fps=None):
    w, h = _frame_size(frame)
    x1, y1, x2, y2 = _fit_rect(w * 0.02, h * 0.15, w * 0.55, h * 0.76)
    draw_panel(frame, x1, y1, x2, y2,
               fill=COL_PANEL_BG, alpha=0.88, border=COL_ACCENT, border_thickness=1)

    raw_gesture       = tracker_state.get("raw_gesture",       "?")
    stable_gesture    = tracker_state.get("stable_gesture",    "?")
    confirmed_gesture = tracker_state.get("confirmed_gesture", "?")
    robot_ready       = tracker_state.get("robot_ready",       False)
    command_text      = tracker_state.get("command",           "")
    stable_streak     = tracker_state.get("stable_streak",     0)
    history_size      = tracker_state.get("history_size",      0)
    play_mode_label   = game_state.get("play_mode_label",      "")

    fps_line = []
    if fps is not None:
        fps_col = COL_GREEN if fps >= 25 else (COL_AMBER if fps >= 15 else COL_RED)
        fps_line = [(f"FPS: {fps:.0f}", fps_col, SCALE_CAPTION, 2)]

    lines = fps_line + [
        (f"Mode: {play_mode_label}",         COL_TEXT_PRIMARY, SCALE_BODY,    2),
        (f"Count: {count_text}",              COL_GREEN,         SCALE_BODY,    2),
        (f"Raw: {raw_gesture}",               COL_TEXT_SECONDARY,SCALE_CAPTION, 1),
        (f"Stable: {stable_gesture}",         COL_TEXT_SECONDARY,SCALE_CAPTION, 1),
        (f"Confirmed: {confirmed_gesture}",   COL_TEXT_SECONDARY,SCALE_CAPTION, 1),
        (f"Frames: {stable_streak}/3  Buf: {history_size}/7",
                                              COL_TEXT_DIM,      SCALE_MICRO,   1),
        (f"Robot Ready: {'YES' if robot_ready else 'NO'}",
                                              COL_GREEN if robot_ready else COL_AMBER,
                                              SCALE_CAPTION, 2),
        (f"Safe Cmd: {command_text}",         COL_TEXT_PRIMARY,  SCALE_MICRO,   1),
        (f"Status: {status_text}",            COL_ACCENT,        SCALE_MICRO,   1),
        (f"Reason: {reason_text}",            COL_TEXT_DIM,      SCALE_MICRO,   1),
        (f"Ambig: {ambiguous_count}",         COL_TEXT_DIM,      SCALE_MICRO,   1),
        (f"Output: {output_summary}",         COL_ACCENT,        SCALE_MICRO,   1),
    ]

    if emotion_state and emotion_state.get("face_detected"):
        em       = emotion_state["stable_emotion"]
        em_color = _get_emotion_color(em)
        sc       = emotion_state["scores"]
        cal      = emotion_state.get("calibrated", True)
        cal_prog = emotion_state.get("calibration_progress", 100)
        if not cal:
            lines.append((f"Emotion: calibrating... {cal_prog}%", COL_AMBER, SCALE_CAPTION, 1))
        else:
            em_detail = (f"Smile:{sc['smile']:.2f}  "
                         f"Surp:{sc['surprise']:.2f}  "
                         f"Frust:{sc['frustration']:.2f}")
            lines.append((f"Emotion: {em} ({emotion_state['confidence']:.0%})",
                           em_color, SCALE_CAPTION, 2))
            lines.append((f"  {em_detail}", em_color, SCALE_MICRO, 1))
    elif emotion_state:
        lines.append(("Emotion: No face", COL_TEXT_DIM, SCALE_MICRO, 1))

    y    = y1 + _ix(h * 0.038)
    step = _ix(h * 0.033)
    for text, color, scale, thickness in lines:
        if y + step > y2 - _ix(h * 0.01):
            break
        draw_outlined_text(frame, text, x1 + _ix(w * 0.018), y,
                           scale, color, thickness=thickness, outline=2)
        y += step

def draw_diagnostic_game_panel(frame, game_state):
    w, h = _frame_size(frame)
    x1, y1, x2, y2 = _fit_rect(w * 0.02, h * 0.72, w * 0.98, h * 0.97)
    draw_panel(frame, x1, y1, x2, y2,
               fill=COL_PANEL_BG, alpha=0.88, border=COL_ACCENT, border_thickness=1)

    state_label = game_state.get("state_label", game_state.get("state", "Unknown"))
    beat_count  = game_state.get("beat_count", 0)
    time_left   = game_state.get("time_left", 0.0)
    main_text   = game_state.get("main_text", game_state.get("result_banner", ""))
    sub_text    = game_state.get("sub_text", "")
    score_text  = game_state.get("score_text", "")
    round_text  = game_state.get("round_text", "")

    draw_outlined_text(frame, f"State: {state_label}",
                       x1 + _ix(w * 0.022), y1 + _ix(h * 0.048),
                       SCALE_BODY, COL_TEXT_PRIMARY, thickness=2, outline=3)

    line2 = f"Beats: {beat_count}/4"
    if round_text: line2 += f"   {round_text}"
    if score_text: line2 += f"   {score_text}"
    if game_state.get("state") == "SHOOT_WINDOW":
        line2 += f"   {time_left:.2f}s"
    draw_outlined_text(frame, line2, x1 + _ix(w * 0.022), y1 + _ix(h * 0.095),
                       SCALE_MICRO, COL_TEXT_DIM, thickness=1, outline=2)

    draw_centered_text_in_rect(frame, main_text,
        (x1 + 20, y1 + _ix(h * 0.11), x2 - 20, y1 + _ix(h * 0.19)),
        base_scale=SCALE_HEADING, color=COL_TEXT_PRIMARY, thickness=2, outline=3)
    draw_centered_text_in_rect(frame, sub_text,
        (x1 + 20, y1 + _ix(h * 0.18), x2 - 20, y2 - 8),
        base_scale=SCALE_BODY, color=COL_TEXT_DIM, thickness=1, outline=2)


# ============================================================
# PRIMARY CONTENT PANEL (spec §03 hero area)
# ============================================================

def draw_arcade_hero(frame, game_state, voice_mode_active=False):
    layout     = _game_layout(frame)
    w, h       = layout["w"], layout["h"]
    x1, y1, x2, y2 = layout["hero"]
    state      = game_state.get("state",     "")
    main_text  = game_state.get("main_text", "")
    sub_text   = game_state.get("sub_text",  "")
    time_left  = game_state.get("time_left", 0.0)
    beat_count = game_state.get("beat_count", 0)

    # Border colour reflects win/loss state during result (kept for overlay callers)
    border_col = COL_BORDER_HAIR
    if state in ("ROUND_RESULT", "MATCH_RESULT"):
        banner = game_state.get("result_banner", "")
        border_col = get_result_banner_color(banner)

    draw_panel(frame, x1, y1, x2, y2,
               fill=COL_PANEL_BG, alpha=0.78, border=border_col, border_thickness=1)

    ph     = y2 - y1
    pw     = x2 - x1
    pcx    = (x1 + x2) // 2
    pill_y = y1 + _ix(ph * 0.14)

    if state == "ROUND_INTRO":
        _draw_state_pill(frame, "READY", pcx, pill_y)
        draw_centered_text(frame, main_text, y1 + _ix(ph * 0.52),
                           SCALE_DISPLAY_L, COL_TEXT_PRIMARY, thickness=2, outline=3)
        draw_centered_text(frame, sub_text, y1 + _ix(ph * 0.82),
                           SCALE_CAPTION, COL_TEXT_DIM, thickness=1, outline=2)

    elif state == "WAITING_FOR_ROCK":
        if voice_mode_active:
            _draw_state_pill(frame, "VOICE", pcx, pill_y)
            draw_centered_text(frame, 'Say READY',
                               y1 + _ix(ph * 0.46),
                               SCALE_DISPLAY_L, COL_GREEN, thickness=2, outline=3)
            draw_centered_text(frame, "to start the countdown",
                               y1 + _ix(ph * 0.76),
                               SCALE_CAPTION, COL_TEXT_DIM, thickness=1, outline=2)
        else:
            _draw_state_pill(frame, "READY", pcx, pill_y)
            fist_txt   = "Make a fist to start"
            fist_scale = SCALE_DISPLAY_XL * 0.6
            fist_font  = cv2.FONT_HERSHEY_DUPLEX
            (ftw, _), _ = cv2.getTextSize(fist_txt, fist_font, fist_scale, 2)
            ftx = (w - ftw) // 2
            fty = y1 + _ix(ph * 0.50)
            cv2.putText(frame, fist_txt, (ftx, fty),
                        fist_font, fist_scale, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, fist_txt, (ftx, fty),
                        fist_font, fist_scale, COL_TEXT_PRIMARY, 2, cv2.LINE_AA)
            draw_centered_text(frame, "Pump down 4 times to count down",
                               y1 + _ix(ph * 0.72), SCALE_CAPTION, COL_TEXT_DIM,
                               thickness=1, outline=2)

    elif state == "COUNTDOWN":
        if voice_mode_active:
            next_words = {0: "ONE", 1: "TWO", 2: "THREE"}
            next_word  = next_words.get(beat_count, "THREE")
            _draw_state_pill(frame, "COUNTING", pcx, pill_y)
            draw_centered_text(frame, f"Say  {next_word}",
                               y1 + _ix(ph * 0.46),
                               SCALE_DISPLAY_L, COL_ACCENT, thickness=2, outline=3)
        else:
            _draw_state_pill(frame, f"BEAT {beat_count} OF 4", pcx, pill_y)
            num_str = str(beat_count) if beat_count > 0 else "GO"
            draw_centered_text_in_rect(frame, num_str,
                (x1 + _ix(pw * 0.06), y1 + _ix(ph * 0.28),
                 x2 - _ix(pw * 0.06), y1 + _ix(ph * 0.72)),
                base_scale=SCALE_DISPLAY_XL,
                color=COL_ACCENT, thickness=2, outline=3)
            if sub_text:
                draw_centered_text(frame, sub_text, y1 + _ix(ph * 0.88),
                                   SCALE_CAPTION, COL_TEXT_DIM, thickness=1, outline=2)

    elif state == "SHOOT_WINDOW":
        _draw_state_pill(frame, "SHOOT", pcx, pill_y)
        if voice_mode_active:
            draw_centered_text(frame, "SAY YOUR THROW",
                               y1 + _ix(ph * 0.40),
                               SCALE_DISPLAY_L, COL_RED, thickness=2, outline=3)
            throws  = ["ROCK", "PAPER", "SCISSORS"]
            col_w   = pw // 3
            row_y   = y1 + _ix(ph * 0.74)
            for i, word in enumerate(throws):
                cx = x1 + col_w * i + col_w // 2
                (tw, _), _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, SCALE_BODY, 1)
                draw_outlined_text(frame, word, cx - tw // 2, row_y,
                                   SCALE_BODY, COL_TEXT_PRIMARY, thickness=1, outline=2)
        else:
            draw_centered_text_in_rect(frame, "SHOOT!",
                (x1 + _ix(pw * 0.06), y1 + _ix(ph * 0.28),
                 x2 - _ix(pw * 0.06), y1 + _ix(ph * 0.72)),
                base_scale=SCALE_DISPLAY_XL,
                color=COL_RED, thickness=2, outline=3)
            draw_centered_text(frame, f"{time_left:.2f}s", y1 + _ix(ph * 0.78),
                               SCALE_BODY, COL_TEXT_SECONDARY, thickness=1, outline=2)
            if sub_text:
                draw_centered_text(frame, sub_text, y1 + _ix(ph * 0.90),
                                   SCALE_CAPTION, COL_TEXT_DIM, thickness=1, outline=2)

# ============================================================
# BEAT TRACK (wrapper around ui_base draw_beat_track)
# ============================================================

def draw_arcade_beat_track(frame, beat_count, state, voice_mode_active=False):
    layout = _game_layout(frame)
    w, h   = layout["w"], layout["h"]
    x1, y1, x2, y2 = layout["beat_track"]

    if voice_mode_active:
        # Voice mode: 3 circles labelled ONE / TWO / THREE
        draw_panel(frame, x1, y1, x2, y2,
                   fill=COL_PANEL_BG, alpha=0.78,
                   border=COL_BORDER_HAIR, border_thickness=1)
        ph, pw = y2 - y1, x2 - x1
        cv2.putText(frame, "VOICE COUNTDOWN",
                    (x1 + _ix(pw * 0.28), y1 + _ix(ph * 0.28)),
                    cv2.FONT_HERSHEY_SIMPLEX, SCALE_MICRO, COL_TEXT_DIM, 1, cv2.LINE_AA)

        positions = [x1 + _ix(pw * p) for p in (0.18, 0.50, 0.82)]
        labels    = ["ONE", "TWO", "THREE"]
        cy        = y1 + _ix(ph * 0.62)
        radius    = _ix(min(w, h) * 0.034)

        for i, (x, label) in enumerate(zip(positions, labels)):
            active     = i < beat_count
            shoot_beat = (i == 2 and state == "SHOOT_WINDOW")
            if shoot_beat:
                col = COL_RED
            elif active:
                col = COL_ACCENT
            else:
                col = COL_BEAT_RING

            if active or shoot_beat:
                cv2.circle(frame, (x, cy), radius, col, -1)
                num_col = COL_ON_ACTIVE
            else:
                cv2.circle(frame, (x, cy), radius, col, 1)
                num_col = COL_TEXT_DIM

            draw_centered_text_in_rect(frame, str(i + 1),
                (x - radius, cy - radius, x + radius, cy + radius),
                base_scale=SCALE_MICRO, color=num_col, thickness=1, outline=0)

            (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, SCALE_MICRO, 1)
            cv2.putText(frame, label, (x - lw // 2, cy + radius + _ix(h * 0.026)),
                        cv2.FONT_HERSHEY_SIMPLEX, SCALE_MICRO, col, 1, cv2.LINE_AA)

        hint = "THREE opens throw" if state != "SHOOT_WINDOW" else "Say ROCK PAPER SCISSORS"
        (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, SCALE_MICRO, 1)
        cv2.putText(frame, hint,
                    (x1 + (pw - hw) // 2, y1 + _ix(ph * 0.90)),
                    cv2.FONT_HERSHEY_SIMPLEX, SCALE_MICRO, COL_TEXT_DIM, 1, cv2.LINE_AA)
    else:
        draw_beat_track(frame, beat_count, num_beats=4, state=state,
                        x1=x1, y1=y1, x2=x2, y2=y2)

# ============================================================
# RESULT SCREEN (spec §03 state variants)
# ============================================================

def draw_result_screen(frame, game_state, colourblind=False):
    layout = _game_layout(frame)
    w, h   = layout["w"], layout["h"]
    x1, y1, x2, y2 = layout["result"]

    banner      = game_state.get("result_banner") or game_state.get("main_text", "")
    banner_col  = get_result_banner_color(banner, colourblind=colourblind)

    draw_panel(frame, x1, y1, x2, y2,
               fill=COL_PANEL_BG, alpha=0.88,
               border=banner_col, border_thickness=1)
    gs = banner.upper()
    if any(w in gs for w in ('YOU WIN', 'YOU TAKE', 'SURVIVE')):
        pill_label = 'WIN'
    elif any(w in gs for w in ('ROBOT TAKES', 'ROBOT WIN', 'YOU LOSE', 'LOSS')):
        pill_label = 'LOSS'
    elif 'DRAW' in gs:
        pill_label = 'DRAW'
    else:
        pill_label = 'RESULT'
    _draw_state_pill(frame, pill_label, (x1 + x2) // 2, y1 + _ix((y2 - y1) * 0.09))

    if game_state.get("gesture_assumed"):
        draw_centered_text(frame, "Rock assumed -- no gesture detected",
                           y1 + _ix((y2 - y1) * 0.18), SCALE_CAPTION, COL_AMBER,
                           thickness=1, outline=2)

    # Colourblind tint
    if colourblind:
        tint = _COL_CB_WIN if ("WIN" in banner.upper() or "SURVIVE" in banner.upper()) \
               else (_COL_CB_DRAW if "DRAW" in banner.upper() else _COL_CB_LOSE)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), tint, -1)
        cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)

    # 3-column result grid (no inner sub-panels)
    ph, pw = y2 - y1, x2 - x1
    col_w  = _ix(pw * 0.36)
    left   = _fit_rect(x1 + _ix(pw * 0.04), y1 + _ix(ph * 0.28),
                       x1 + _ix(pw * 0.04) + col_w, y1 + _ix(ph * 0.82))
    right  = _fit_rect(x2 - _ix(pw * 0.04) - col_w, y1 + _ix(ph * 0.28),
                       x2 - _ix(pw * 0.04), y1 + _ix(ph * 0.82))

    mode_label = game_state.get("play_mode_label", "")
    opp_label  = mode_label[3:] if mode_label.startswith("vs ") else "CPU"

    draw_centered_text_in_rect(frame, "YOU",
        (left[0], left[1] + 6, left[2], left[1] + _ix((left[3] - left[1]) * 0.20)),
        base_scale=SCALE_CAPTION, color=COL_TEXT_SECONDARY, thickness=1, outline=2)
    draw_centered_text_in_rect(frame, opp_label.upper(),
        (right[0], right[1] + 6, right[2], right[1] + _ix((right[3] - right[1]) * 0.20)),
        base_scale=SCALE_CAPTION, color=COL_TEXT_SECONDARY, thickness=1, outline=2)

    _GESTURE_COLS = {
        'rock':     (160, 120,  80),
        'paper':    ( 80, 160, 200),
        'scissors': ( 80,  80, 200),
        'lizard':   ( 80, 160,  80),
        'spock':    (200, 100, 200),
    }
    p_gest  = game_state.get("player_gesture",   "")
    ai_gest = game_state.get("computer_gesture", "")
    p_base  = _GESTURE_COLS.get(p_gest.lower(),  COL_TEXT_SECONDARY)
    ai_base = _GESTURE_COLS.get(ai_gest.lower(), COL_TEXT_SECONDARY)
    b_up = banner.upper()
    if any(w in b_up for w in ('YOU WIN', 'YOU TAKE', 'SURVIVE')):
        p_col, ai_col = COL_GREEN, COL_RED
    elif any(w in b_up for w in ('ROBOT TAKES', 'ROBOT WIN', 'YOU LOSE', 'LOSS')):
        p_col, ai_col = COL_RED, COL_GREEN
    else:
        p_col, ai_col = p_base, ai_base

    pgx  = (left[0]  + left[2])  // 2
    pcy  = left[1]  + _ix((left[3]  - left[1])  * 0.48)
    p_sz = _ix(min(left[2]  - left[0],  left[3]  - left[1])  * 0.20)
    draw_gesture_glyph(frame, p_gest,
                       (pgx - p_sz, pcy - p_sz, pgx + p_sz, pcy + p_sz), p_col)

    agx  = (right[0] + right[2]) // 2
    acy  = right[1] + _ix((right[3] - right[1]) * 0.48)
    a_sz = _ix(min(right[2] - right[0], right[3] - right[1]) * 0.20)
    draw_gesture_glyph(frame, ai_gest,
                       (agx - a_sz, acy - a_sz, agx + a_sz, acy + a_sz), ai_col)

    if p_gest:
        draw_centered_text_in_rect(frame, p_gest.upper(),
            (left[0], left[1] + _ix((left[3] - left[1]) * 0.74), left[2], left[3] - 4),
            base_scale=SCALE_CAPTION, color=p_col, thickness=1, outline=2)
    if ai_gest:
        draw_centered_text_in_rect(frame, ai_gest.upper(),
            (right[0], right[1] + _ix((right[3] - right[1]) * 0.74), right[2], right[3] - 4),
            base_scale=SCALE_CAPTION, color=ai_col, thickness=1, outline=2)

    # Colourblind stamp only (centre column otherwise empty)
    if colourblind:
        vs_y = y1 + _ix(ph * 0.54)
        if "YOU WIN" in banner.upper() or "SURVIVE" in banner.upper():
            stamp, stamp_col = "WIN", _COL_CB_WIN
        elif "DRAW" in banner.upper():
            stamp, stamp_col = "DRAW", _COL_CB_DRAW
        else:
            stamp, stamp_col = "LOSE", _COL_CB_LOSE
        draw_centered_text(frame, stamp, vs_y, SCALE_BODY, stamp_col,
                           thickness=2, outline=3)

def draw_session_summary(frame, summary):
    layout = _game_layout(frame)
    w, h   = layout["w"], layout["h"]
    x1, y1, x2, y2 = layout["result"]

    won       = summary.get("player_won", False)
    ps        = summary.get("player_score", 0)
    rs        = summary.get("robot_score", 0)
    rds       = summary.get("total_rounds", 0)
    avg_rt    = summary.get("avg_reaction_ms")
    top_g     = summary.get("top_gesture", "?")
    opp_type  = summary.get("opponent_type", "")

    header_col = COL_GREEN if won else COL_RED
    header_txt = "MATCH WON" if won else "MATCH LOST"

    draw_panel(frame, x1, y1, x2, y2,
               fill=COL_PANEL_BG, alpha=0.92, border=header_col, border_thickness=1)
    draw_centered_text(frame, header_txt, y1 + _ix((y2 - y1) * 0.10),
                       SCALE_DISPLAY_L, header_col, thickness=2, outline=3)
    draw_centered_text(frame, f"{ps}  -  {rs}",
                       y1 + _ix((y2 - y1) * 0.24), SCALE_HEADING,
                       COL_TEXT_PRIMARY, thickness=2, outline=3)

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
        draw_centered_text(frame, s, stat_y, SCALE_BODY, COL_TEXT_PRIMARY,
                           thickness=1, outline=2)
        stat_y += _ix((y2 - y1) * 0.11)

    draw_centered_text(frame, "Returning to menu...",
                       y2 - _ix((y2 - y1) * 0.06), SCALE_MICRO, COL_TEXT_DIM,
                       thickness=1, outline=2)


def _draw_last_round_replay(frame, player_gesture, robot_gesture, banner):
    """Briefly show last round's gestures at the start of the next wait state."""
    layout = _game_layout(frame)
    x1, y1, x2, y2 = layout["result"]
    ph, pw = y2 - y1, x2 - x1

    draw_panel(frame, x1, y1, x2, y1 + _ix(ph * 0.30),
               fill=COL_PANEL_BG, alpha=0.80,
               border=COL_BORDER_HAIR, border_thickness=1)
    draw_outlined_text(frame, "LAST ROUND", x1 + _ix(pw * 0.03),
                       y1 + _ix(ph * 0.08), SCALE_MICRO, COL_TEXT_DIM,
                       thickness=1, outline=2)
    if banner:
        banner_col = get_result_banner_color(banner)
        draw_centered_text(frame, banner, y1 + _ix(ph * 0.08),
                           SCALE_CAPTION, banner_col, thickness=1, outline=2)

    b_up = banner.upper()
    if any(w in b_up for w in ('YOU WIN', 'YOU TAKE', 'SURVIVE')):
        p_col, ai_col = COL_GREEN, COL_RED
    elif any(w in b_up for w in ('ROBOT TAKES', 'ROBOT WIN', 'YOU LOSE', 'LOSS')):
        p_col, ai_col = COL_RED, COL_GREEN
    else:
        p_col, ai_col = COL_TEXT_SECONDARY, COL_TEXT_SECONDARY

    lx = x1 + _ix(pw * 0.22)
    rx = x1 + _ix(pw * 0.62)
    gy = y1 + _ix(ph * 0.20)
    sz = _ix(min(pw, ph) * 0.06)
    if player_gesture:
        draw_gesture_glyph(frame, player_gesture, (lx-sz, gy-sz, lx+sz, gy+sz), p_col)
        draw_outlined_text(frame, player_gesture.upper(), lx - sz, gy + sz + 8,
                           SCALE_MICRO, p_col, thickness=1, outline=2)
    if robot_gesture:
        draw_gesture_glyph(frame, robot_gesture, (rx-sz, gy-sz, rx+sz, gy+sz), ai_col)
        draw_outlined_text(frame, robot_gesture.upper(), rx - sz, gy + sz + 8,
                           SCALE_MICRO, ai_col, thickness=1, outline=2)
    draw_centered_text(frame, "vs", gy, SCALE_CAPTION, COL_TEXT_DIM,
                       thickness=1, outline=2)

# ============================================================
# MAIN COMPOSITE (called from main.py)
# ============================================================

def draw_game_mode_view(frame, game_state, emotion_state=None, voice_mode_active=False,
                        last_heard_word="", tracker_state=None, hand_state=None,
                        flash_info=None, show_help=False, sound_on=True,
                        colourblind=False, show_session_summary=False, diagnostic=False):

    mode_raw   = game_state.get("play_mode_label", "")
    left_label = f"RPS ROBOT  {mode_raw.upper()}" if mode_raw else "RPS ROBOT"
    if voice_mode_active:
        right_hints = "VOICE ON  *  Say READY to start  *  BACK = menu"
        bottom_hint = "Say READY > ONE > TWO > THREE > ROCK/PAPER/SCISSORS  |  BACK = menu  |  ? Help"
    elif diagnostic:
        right_hints = "ESC Back  M Game View  S Sound  ? Help  Q Quit"
        bottom_hint = "M Game View  |  ESC Back  |  S Sound  |  C Commentary  |  ? Help  |  Q Quit"
    else:
        right_hints = "ESC Back  M Diagnostic  S Sound  ? Help  Q Quit"
        bottom_hint = "ESC Back  |  M Diagnostic  |  S Sound  |  C Commentary  |  ? Help  |  Q Quit"
    draw_top_bar(frame, left_label, right_hints)
    draw_bottom_bar(frame, bottom_hint)

    # Gesture indicator row -- show detected gesture if available
    detected = ""
    if tracker_state:
        cg = tracker_state.get("confirmed_gesture", "")
        sg = tracker_state.get("stable_gesture", "")
        detected = cg if cg in ("Rock", "Paper", "Scissors") \
                   else (sg if sg in ("Rock", "Paper", "Scissors") else "")
    draw_gesture_row(frame, detected_gesture=detected)

    draw_game_status_strip(frame, game_state)

    cur_state = game_state.get("state", "")

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
                         and flash_info
                         and flash_info.get("replay_until", 0) > time.monotonic())
        if replay_active:
            _draw_last_round_replay(frame, last_pg, last_rg,
                                    game_state.get("last_banner", ""))
        else:
            draw_arcade_hero(frame, game_state,
                             voice_mode_active=voice_mode_active)
        display_beat = 0 if cur_state == "WAITING_FOR_ROCK" else game_state.get("beat_count", 0)
        draw_arcade_beat_track(frame, display_beat, cur_state,
                               voice_mode_active=voice_mode_active)

    w, h = _frame_size(frame)
    layout = _game_layout(frame)

    # Win/loss streak label
    streak_text = game_state.get("streak_label", "")
    if streak_text:
        streak_col = COL_GREEN if "WIN" in streak_text.upper() else COL_RED
        draw_outlined_text(frame, streak_text, _ix(w * 0.07), h - _ix(h * 0.08),
                           SCALE_MICRO, streak_col, thickness=1, outline=2)

    # Opponent type chip
    opp_type = game_state.get("opponent_type", "")
    _opp_skip = {"random", "grace_period", "", "unknown", "Unknown"}
    if opp_type and opp_type not in _opp_skip \
            and cur_state not in {"ROUND_RESULT", "MATCH_RESULT"}:
        chip_text = f"[ {opp_type.replace('_', ' ').upper()} DETECTED ]"
        draw_outlined_text(frame, chip_text, _ix(w * 0.02),
                           layout["top_bar_h"] + _ix(h * 0.038),
                           SCALE_MICRO, COL_AMBER, thickness=1, outline=2)

    # AI personality chip
    personality = game_state.get("ai_personality", "Normal")
    _P_COLS = {
        "The Psychologist": (180, 80, 255),
        "The Gambler":      (60, 200, 120),
        "The Mirror":       (80, 220, 220),
        "The Ghost":        (160, 160, 200),
        "The Chaos Agent":  (200, 60, 60),
        "The Hustler":      (255, 160, 40),
    }
    if personality and personality != "Normal":
        pcol  = _P_COLS.get(personality, COL_AMBER)
        ptxt  = f"vs  {personality}"
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = SCALE_MICRO
        (tw, th), _ = cv2.getTextSize(ptxt, font, scale, 1)
        chip_cx = w // 2
        chip_y  = layout["top_bar_h"] + _ix(h * 0.004)
        chip_x1 = chip_cx - tw // 2 - _ix(w * 0.012)
        chip_x2 = chip_cx + tw // 2 + _ix(w * 0.012)
        chip_y2 = chip_y + th + _ix(h * 0.018)
        draw_panel(frame, chip_x1, chip_y, chip_x2, chip_y2,
                   fill=tuple(c // 6 for c in pcol), alpha=0.88,
                   border=pcol, border_thickness=1)
        draw_outlined_text(frame, ptxt, chip_cx - tw // 2, chip_y2 - _ix(h * 0.005),
                           scale, pcol, thickness=1, outline=2)

    # Post-round personality insight
    if cur_state in {"ROUND_RESULT", "MATCH_RESULT"} and personality and personality != "Normal":
        _INSIGHTS = {
            "The Psychologist": [
                "Watching for win-stay patterns...",
                "It knows you shifted after that loss.",
                "Outcome-conditioned prediction active.",
                "It predicted your response bias.",
            ],
            "The Gambler": [
                "Wild card incoming - stay sharp.",
                "It rolled the dice this round.",
                "High variance play. Unpredictable.",
                "20% chance it ignores all patterns.",
            ],
            "The Mirror": [
                "It copied your most common gesture.",
                "Switch up your dominant move.",
                "Mirror AI: break the pattern to win.",
                "Its strength is your own habit.",
            ],
            "The Ghost": [
                "It played your previous move.",
                "One step behind - use it against it.",
                "The Ghost echoes your last gesture.",
                "Throw what beats your own last move.",
            ],
            "The Chaos Agent": [
                "Pure Nash equilibrium. No pattern.",
                "33/33/33 - nothing to exploit.",
                "Unreadable by design.",
                "Even the AI does not know.",
            ],
            "The Hustler": [
                "It learned fast. Adapting now.",
                "Pattern locked - it is reading you.",
                "Hustler reads transitions hard.",
                "Change strategy every 3 rounds.",
            ],
        }
        insights = _INSIGHTS.get(personality, [])
        if insights:
            rn      = game_state.get("round_number", 1)
            insight = insights[rn % len(insights)]
            pcol2   = _P_COLS.get(personality, COL_AMBER)
            draw_outlined_text(frame, f"[ {insight} ]",
                               _ix(w * 0.02),
                               layout["top_bar_h"] + _ix(h * 0.038),
                               SCALE_MICRO, pcol2, thickness=1, outline=2)

    # Voice mic badge
    if voice_mode_active:
        badge_text = f"[ MIC  {last_heard_word.upper()} ]" if last_heard_word \
                     else "[ MIC ON ]"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (bw, _), _ = cv2.getTextSize(badge_text, font, SCALE_MICRO, 1)
        badge_x = w - bw - _ix(w * 0.02)
        badge_y = layout["top_bar_h"] + _ix(h * 0.038)
        draw_outlined_text(frame, badge_text, badge_x, badge_y,
                           SCALE_MICRO, COL_GREEN, thickness=1, outline=2)
        mic_level = flash_info.get("mic_level", 0.0) if flash_info else 0.0
        if mic_level > 0.01:
            bar_y2  = badge_y + _ix(h * 0.016)
            fill_w  = int(bw * mic_level)
            cv2.rectangle(frame, (badge_x, bar_y2), (badge_x + bw, bar_y2 + 3),
                          COL_BEAT_FILL, -1)
            col = COL_GREEN if mic_level < 0.7 else COL_AMBER
            cv2.rectangle(frame, (badge_x, bar_y2), (badge_x + fill_w, bar_y2 + 3),
                          col, -1)

    if hand_state:
        draw_quality_warnings(frame, hand_state)

    if emotion_state and emotion_state.get("face_detected"):
        cal = emotion_state.get("calibrated", True)
        if not cal:
            draw_outlined_text(
                frame, f"calibrating {emotion_state.get('calibration_progress', 0)}%",
                w - _ix(w * 0.28), h - _ix(h * 0.10),
                SCALE_MICRO, COL_AMBER, thickness=1, outline=2)
        else:
            em    = emotion_state["stable_emotion"]
            em_col = _get_emotion_color(em) if em != "Neutral" else COL_TEXT_DIM
            label = em if em == "Neutral" else f"{em}  {emotion_state['confidence']:.0%}"
            draw_outlined_text(frame, label,
                               w - _ix(w * 0.24), h - _ix(h * 0.10),
                               SCALE_MICRO, em_col, thickness=1, outline=2)

    if show_help:
        draw_help_overlay(frame, "GAME", voice_mode=voice_mode_active)

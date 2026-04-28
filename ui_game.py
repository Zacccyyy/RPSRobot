"""
ui_game.py -- In-game screen renderers: arcade view, result, session summary, diagnostic.
"""
import cv2
import math
import time
from ui_base import *

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

    # Emotion rows - [E] hint removed here, it's already in the top bar
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

    state_label = game_state.get("state_label", game_state.get("state", "Unknown"))
    beat_count  = game_state.get("beat_count", 0)
    time_left   = game_state.get("time_left", 0.0)
    main_text   = game_state.get("main_text", game_state.get("result_banner", ""))
    sub_text    = game_state.get("sub_text", "")
    score_text  = game_state.get("score_text", "")
    round_text  = game_state.get("round_text", "")

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
            # Pulse: scale in from 0.6 > 1.0 over the beat window using sin curve
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
            # Three throw options in their own columns - fixed x, not frame-centered
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
        # Physical mode - original 4-beat track
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

    # Reaction time - shown instead of score_text if available
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

def draw_game_mode_view(frame, game_state, emotion_state=None, voice_mode_active=False,
                        last_heard_word="", tracker_state=None, hand_state=None,
                        flash_info=None, show_help=False, sound_on=True,
                        colourblind=False, show_session_summary=False):
    draw_game_header(frame, game_state, voice_mode_active=voice_mode_active, sound_on=sound_on)
    if voice_mode_active:
        bottom_hint = "Say READY > ONE > TWO > THREE > ROCK/PAPER/SCISSORS  |  BACK = menu  |  ? Help"
    else:
        bottom_hint = "ESC Back  |  M Diagnostic  |  N Sound  |  C Commentary  |  ? Help  |  Q Quit"
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
    # In voice COUNTDOWN the bar is voice-driven - suppress gesture fill
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

    # ── AI Personality HUD chip ───────────────────────────────────────────
    personality = game_state.get("ai_personality", "Normal")
    if personality and personality != "Normal":
        _P_COLS = {
            "The Psychologist": (180, 80, 255),
            "The Gambler":      (60, 200, 120),
            "The Mirror":       (80, 220, 220),
            "The Ghost":        (160, 160, 200),
            "The Chaos Agent":  (200, 60, 60),
            "The Hustler":      (255, 160, 40),
        }
        pcol  = _P_COLS.get(personality, COL_MAGENTA)
        ptxt  = f"vs  {personality}"
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.38
        (tw, th), _ = cv2.getTextSize(ptxt, font, scale, 1)
        # Centre horizontally, place just below the top bar
        chip_cx = w // 2
        chip_y  = layout["top_row_h"] + _ix(h * 0.004)
        chip_x1 = chip_cx - tw // 2 - _ix(w * 0.012)
        chip_x2 = chip_cx + tw // 2 + _ix(w * 0.012)
        chip_y2 = chip_y + th + _ix(h * 0.018)
        draw_panel(frame, chip_x1, chip_y, chip_x2, chip_y2,
                   fill=tuple(c // 6 for c in pcol), alpha=0.88,
                   border=pcol, border_thickness=1)
        draw_outlined_text(frame, ptxt,
                           chip_cx - tw // 2, chip_y2 - _ix(h * 0.005),
                           scale, pcol, thickness=1, outline=2)

    # ── Post-round personality insight line ───────────────────────────────
    if cur_state in {"ROUND_RESULT", "MATCH_RESULT"} and personality and personality != "Normal":
        _INSIGHTS = {
            "The Psychologist": [
                "Watching for win-stay patterns...",
                "It knows you shifted after that loss.",
                "Outcome-conditioned prediction active.",
                "It predicted your response bias.",
            ],
            "The Gambler": [
                "Wild card incoming — stay sharp.",
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
                "One step behind — use it against it.",
                "The Ghost echoes your last gesture.",
                "Throw what beats your own last move.",
            ],
            "The Chaos Agent": [
                "Pure Nash equilibrium. No pattern.",
                "33/33/33 — nothing to exploit.",
                "Unreadable by design.",
                "Even the AI doesn't know what it played.",
            ],
            "The Hustler": [
                "It learned fast. Adapting now.",
                "Pattern locked — it's reading you.",
                "Hustler reads transitions hard.",
                "Change your strategy every 3 rounds.",
            ],
        }
        import random as _r
        insights = _INSIGHTS.get(personality, [])
        if insights:
            # Pick insight deterministically per round number so it doesn't flicker
            rn = game_state.get("round_number", 1)
            insight = insights[rn % len(insights)]
            insight_y = layout["top_row_h"] + layout["second_row_h"] + _ix(h * 0.038)
            pcol2 = _P_COLS.get(personality, COL_MAGENTA) if personality in _P_COLS else COL_MAGENTA
            draw_outlined_text(frame, f"[ {insight} ]",
                               _ix(w * 0.02), insight_y,
                               0.32, pcol2, thickness=1, outline=2)

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

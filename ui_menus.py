"""
ui_menus.py -- Menu, settings, features, simulation, clone, stats and tutorial screens.
"""
import cv2
import math
import time
from ui_base import *

# MENU SCREEN
# ============================================================

def draw_menu_screen(frame, menu_items, selected_index, config,
                     show_help=False, voice_mode_active=False, in_submenu=False,
                     update_label="", calibration_warning=False):
    w, h = _frame_size(frame)

    top_right = "UP/DOWN Navigate | Enter Select | N Feedback | ESC Back | Q Quit"
    draw_top_bar(frame, "RPS ROBOT", top_right)

    # --- Wordmark block (y 14-28%) ---
    if in_submenu:
        draw_centered_text(frame, "GAME MODES", _ix(h * 0.21),
                           SCALE_HEADING, COL_TEXT_PRIMARY, thickness=1, outline=2)
        draw_centered_text(frame, "Select a mode to play", _ix(h * 0.26),
                           SCALE_CAPTION, COL_TEXT_DIM, thickness=1, outline=2)
    else:
        # Kicker -- accent, caption scale
        draw_centered_text(frame, "ROCK  *  PAPER  *  SCISSORS",
                           _ix(h * 0.19), SCALE_CAPTION, COL_ACCENT,
                           thickness=1, outline=2)
        # Title -- display weight (FONT_HERSHEY_DUPLEX)
        font_disp = cv2.FONT_HERSHEY_DUPLEX
        (tw, _), _ = cv2.getTextSize("RPS ROBOT", font_disp, SCALE_DISPLAY_L, 2)
        tx = (w - tw) // 2
        cv2.putText(frame, "RPS ROBOT", (tx, _ix(h * 0.26)),
                    font_disp, SCALE_DISPLAY_L, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, "RPS ROBOT", (tx, _ix(h * 0.26)),
                    font_disp, SCALE_DISPLAY_L, COL_TEXT_PRIMARY, 2, cv2.LINE_AA)

    # Subtitle -- dim caption, y 32%, sits between wordmark and panel
    subtitle = (f"{config.get('default_play_mode', 'FairPlay')}  |  "
                f"{config.get('default_display_mode', 'Game')}")
    draw_centered_text(frame, subtitle, _ix(h * 0.32),
                       SCALE_CAPTION, COL_TEXT_DIM, thickness=1, outline=2)

    # Calibration warning banner
    if calibration_warning:
        pulse = 0.65 + 0.35 * abs(math.sin(time.monotonic() * math.pi * 1.0))
        bc = tuple(min(255, int(c * pulse)) for c in COL_AMBER)
        draw_panel(frame, _ix(w * 0.255), _ix(h * 0.30), _ix(w * 0.745), _ix(h * 0.36),
                   fill=(18, 12, 0), alpha=0.92, border=bc, border_thickness=1)
        draw_centered_text_in_rect(frame,
            "Gestures not calibrated -- recognition may be inaccurate",
            (_ix(w * 0.265), _ix(h * 0.30), _ix(w * 0.735), _ix(h * 0.36)),
            base_scale=0.32, color=bc, thickness=1, outline=2)

    # Update banner — frosted amber strip directly below top bar
    if update_label:
        bh = _ix(h * 0.048)
        uy1 = _ix(h * 0.06)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, uy1), (w, uy1 + bh), (30, 80, 140), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.line(frame, (0, uy1 + bh), (w, uy1 + bh), COL_BORDER_HAIR, 1)
        draw_centered_text(frame, update_label, uy1 + _ix(bh * 0.68),
                           SCALE_MICRO, COL_AMBER, thickness=1, outline=2)

    # Menu panel -- x 25.5%-74.5%, y 38-86%
    px1 = _ix(w * 0.255)
    py1 = _ix(h * 0.38)
    px2 = _ix(w * 0.745)
    py2 = _ix(h * 0.86)
    draw_panel(frame, px1, py1, px2, py2,
               fill=COL_PANEL_BG, alpha=0.92, border=COL_BORDER_HAIR, border_thickness=1)

    # Menu rows fill the full panel
    n_items = len(menu_items)
    item_area_y1 = py1 + 1
    item_area_y2 = py2 - 1
    row_h = _ix((item_area_y2 - item_area_y1) / max(n_items, 1))

    for i, (label, _) in enumerate(menu_items):
        ry1 = item_area_y1 + i * row_h
        ry2 = ry1 + row_h
        draw_row(frame, px1, ry1, px2, ry2, label, selected=(i == selected_index))
        if i < n_items - 1:
            cv2.line(frame, (px1 + 3, ry2), (px2 - 3, ry2), COL_BORDER_HAIR, 1)

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
      progress:      float 0-1
      progress_text: str  e.g. "random vs fair_play  4/10"
      results:       dict from run_simulation() or None
      error:         str or None
    """
    layout = _menu_layout(frame)
    w, h = layout["w"], layout["h"]
    x1, y1, x2, y2 = layout["panel"]

    status   = sim_state.get("status", "running")
    if status == "running":
        _top_hint = "Simulation running...  ESC disabled"
    elif status == "done":
        _top_hint = "Simulation complete  |  ESC Back"
    else:
        _top_hint = "ESC Back"
    draw_top_bar(frame, "SIMULATION", _top_hint)
    draw_panel(frame, x1, y1, x2, y2, fill=COL_BG_PANEL, alpha=0.94,
               border=COL_ACCENT, border_thickness=1)
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
        draw_centered_text(frame, "ESC is disabled while simulation is running",
                           cy + _ix(ph * 0.26), 0.34, COL_TEXT_DIM, thickness=1, outline=1)

    elif status == "error":
        draw_centered_text(frame, "SIMULATION ERROR", cy, 0.70, COL_RED, thickness=2, outline=3)
        cy += _ix(ph * 0.12)
        draw_centered_text(frame, str(error or "Unknown error"), cy,
                           0.40, COL_TEXT_ACCENT, thickness=1, outline=2)

    elif status == "done" and results:
        mode = results.get("mode", "fairplay")

        # ── Header ─────────────────────────────────────────────────────────
        draw_centered_text_in_rect(frame, "SIMULATION RESULTS",
            (x1, y1, x2, y1 + _ix(ph * 0.10)),
            base_scale=0.70, color=COL_YELLOW, thickness=2, outline=3)

        total_r = results.get("total_rounds_actual") or results.get("total_rounds", 0)
        elapsed = results.get("elapsed_seconds", 0)
        pad_x   = x1 + _ix(pw * 0.05)

        meta_y = y1 + _ix(ph * 0.11)
        draw_outlined_text(frame, f"Rounds simulated: {total_r:,}",
                           pad_x, meta_y, 0.36, COL_TEXT_ACCENT, thickness=1, outline=1)
        draw_outlined_text(frame, f"Time: {elapsed:.1f}s",
                           x1 + _ix(pw * 0.55), meta_y, 0.36, COL_TEXT_DIM, thickness=1, outline=1)

        cv2.line(frame, (x1 + _ix(pw * 0.03), y1 + _ix(ph * 0.17)),
                 (x2 - _ix(pw * 0.03), y1 + _ix(ph * 0.17)), COL_CYAN, 1)

        # ── TOURNAMENT mode ────────────────────────────────────────────────
        if mode == "tournament":
            err_msg = results.get("error_msg", "")
            if err_msg:
                draw_centered_text_in_rect(frame, err_msg,
                    (x1, y1 + _ix(ph * 0.20), x2, y1 + _ix(ph * 0.35)),
                    base_scale=0.40, color=COL_YELLOW, thickness=1, outline=2)
            else:
                champion  = results.get("champion", "?")
                n_players = results.get("n_players", 0)
                rpm       = results.get("rounds_per_match", 200)

                # Champion banner
                draw_centered_text_in_rect(frame,
                    f"Champion:  {champion}",
                    (x1, y1 + _ix(ph * 0.18), x2, y1 + _ix(ph * 0.27)),
                    base_scale=0.58, color=COL_YELLOW, thickness=2, outline=3)
                draw_centered_text_in_rect(frame,
                    f"{n_players} players  |  {rpm} rounds per match",
                    (x1, y1 + _ix(ph * 0.27), x2, y1 + _ix(ph * 0.33)),
                    base_scale=0.34, color=COL_TEXT_DIM, thickness=1, outline=1)

                cv2.line(frame,
                    (x1 + _ix(pw * 0.03), y1 + _ix(ph * 0.34)),
                    (x2 - _ix(pw * 0.03), y1 + _ix(ph * 0.34)), (60, 60, 80), 1)

                # Leaderboard
                draw_outlined_text(frame, "LEADERBOARD",
                                   pad_x, y1 + _ix(ph * 0.37),
                                   0.40, COL_CYAN, thickness=1, outline=2)

                leaderboard = results.get("leaderboard", [])
                row_h  = _ix(ph * 0.073)
                bar_x1 = x1 + _ix(pw * 0.35)
                bar_x2 = x2 - _ix(pw * 0.08)
                bar_w4 = bar_x2 - bar_x1

                for idx, entry in enumerate(leaderboard[:7]):
                    ry  = y1 + _ix(ph * 0.43) + idx * row_h
                    if ry + row_h > y2 - _ix(ph * 0.05):
                        break
                    rank_col = COL_YELLOW if idx == 0 else \
                               (COL_CYAN if idx < 3 else COL_TEXT_DIM)
                    medal = ["1st", "2nd", "3rd"][idx] if idx < 3 else f" #{idx+1}"
                    bh4   = max(4, _ix(ph * 0.020))

                    # Bar bg + fill
                    cv2.rectangle(frame, (bar_x1, ry - bh4), (bar_x2, ry + 2), (30, 30, 40), -1)
                    fx4 = bar_x1 + int(bar_w4 * min(1.0, entry["avg_wr"]))
                    cv2.rectangle(frame, (bar_x1, ry - bh4), (fx4, ry + 2), rank_col, -1)
                    cv2.rectangle(frame, (bar_x1, ry - bh4), (bar_x2, ry + 2), (60, 60, 80), 1)

                    # Label
                    draw_outlined_text(frame,
                        f"{medal}  {entry['player']:<12} {entry['avg_wr']:.0%}  ({entry['rounds']}r)",
                        pad_x, ry, 0.32, rank_col, thickness=1, outline=1)

        # ── FAIRPLAY / PVPVAI mode ─────────────────────────────────────────
        else:
            best_ai  = results.get("best_ai",        "?")
            worst_ai = results.get("worst_ai",       "?")
            best_s   = results.get("best_strategy",  "?")
            worst_s  = results.get("worst_strategy", "?")
            balanced = results.get("most_balanced",  "?")

            # Column boundaries — AI left, Strategy right
            lx1 = x1 + _ix(pw * 0.03)
            lx2 = x1 + _ix(pw * 0.48)
            rx1 = x1 + _ix(pw * 0.52)
            rx2 = x2 - _ix(pw * 0.03)

            section_top = y1 + _ix(ph * 0.19)
            row_h       = _ix(ph * 0.070)
            bar_h       = max(4, _ix(ph * 0.020))

            # ── Left: AI rankings ──────────────────────────────────────────
            draw_outlined_text(frame, "AI DIFFICULTY",
                               lx1, section_top, 0.38, COL_CYAN, thickness=1, outline=2)

            ai_rates = results.get("ai_win_rates", {})
            bar_x1_l = lx1 + _ix((lx2 - lx1) * 0.44)
            bar_x2_l = lx2 - _ix(pw * 0.02)
            bar_w_l  = bar_x2_l - bar_x1_l

            for i, (ai_name, rate) in enumerate(
                    sorted(ai_rates.items(), key=lambda x: -x[1])):
                ry = section_top + _ix(ph * 0.07) + i * row_h
                if ry + row_h > y2 - _ix(ph * 0.12):
                    break
                cv2.rectangle(frame, (bar_x1_l, ry - bar_h), (bar_x2_l, ry + 2), (30,30,40), -1)
                fx  = bar_x1_l + int(bar_w_l * min(1.0, rate))
                cv2.rectangle(frame, (bar_x1_l, ry - bar_h), (fx, ry + 2), COL_RED, -1)
                cv2.rectangle(frame, (bar_x1_l, ry - bar_h), (bar_x2_l, ry + 2), (60,60,80), 1)
                lbl = get_fit_scale(f"{ai_name}", _ix((lx2-lx1)*0.42),
                                    base_scale=0.32, thickness=1, min_scale=0.24)
                draw_outlined_text(frame, f"{ai_name}", lx1, ry,
                                   lbl, COL_TEXT, thickness=1, outline=1)
                draw_outlined_text(frame, f"{rate:.1%}",
                                   bar_x2_l + _ix(pw * 0.005), ry,
                                   0.28, COL_RED, thickness=1, outline=0)

            summary_y_l = section_top + _ix(ph * 0.07) + len(ai_rates) * row_h + _ix(ph * 0.03)
            cv2.line(frame, (lx1, summary_y_l - _ix(ph * 0.01)),
                     (lx2, summary_y_l - _ix(ph * 0.01)), (50, 50, 70), 1)
            draw_outlined_text(frame, f"Hardest:  {best_ai}",
                               lx1, summary_y_l + _ix(ph * 0.020),
                               0.32, COL_RED, thickness=1, outline=1)
            draw_outlined_text(frame, f"Easiest:   {worst_ai}",
                               lx1, summary_y_l + _ix(ph * 0.068),
                               0.32, COL_GREEN, thickness=1, outline=1)

            # ── Right: Strategy rankings ───────────────────────────────────
            draw_outlined_text(frame, "PLAYER STRATEGY",
                               rx1, section_top, 0.38, COL_CYAN, thickness=1, outline=2)

            strat_rates = results.get("strategy_win_rates", {})
            bar_x1_r = rx1 + _ix((rx2 - rx1) * 0.50)
            bar_x2_r = rx2 - _ix(pw * 0.02)
            bar_w_r  = bar_x2_r - bar_x1_r

            for i, (s_name, rate) in enumerate(
                    sorted(strat_rates.items(), key=lambda x: -x[1])):
                ry = section_top + _ix(ph * 0.07) + i * row_h
                if ry + row_h > y2 - _ix(ph * 0.12):
                    break
                bar_col = COL_GREEN if rate > 0.35 else COL_ORANGE
                cv2.rectangle(frame, (bar_x1_r, ry - bar_h), (bar_x2_r, ry + 2), (30,30,40), -1)
                fx  = bar_x1_r + int(bar_w_r * min(1.0, rate))
                cv2.rectangle(frame, (bar_x1_r, ry - bar_h), (fx, ry + 2), bar_col, -1)
                cv2.rectangle(frame, (bar_x1_r, ry - bar_h), (bar_x2_r, ry + 2), (60,60,80), 1)
                lbl = get_fit_scale(f"{s_name}", _ix((rx2-rx1)*0.48),
                                    base_scale=0.30, thickness=1, min_scale=0.22)
                draw_outlined_text(frame, f"{s_name}", rx1, ry,
                                   lbl, COL_TEXT, thickness=1, outline=1)
                draw_outlined_text(frame, f"{rate:.1%}",
                                   bar_x2_r + _ix(pw * 0.005), ry,
                                   0.28, bar_col, thickness=1, outline=0)

            summary_y_r = section_top + _ix(ph * 0.07) + len(strat_rates) * row_h + _ix(ph * 0.03)
            cv2.line(frame, (rx1, summary_y_r - _ix(ph * 0.01)),
                     (rx2, summary_y_r - _ix(ph * 0.01)), (50, 50, 70), 1)
            draw_outlined_text(frame, f"Best:   {best_s}",
                               rx1, summary_y_r + _ix(ph * 0.020),
                               0.32, COL_GREEN, thickness=1, outline=1)
            draw_outlined_text(frame, f"Worst:  {worst_s}",
                               rx1, summary_y_r + _ix(ph * 0.068),
                               0.32, COL_ORANGE, thickness=1, outline=1)

            # Footer strip
            cv2.line(frame, (x1 + _ix(pw * 0.03), y2 - _ix(ph * 0.13)),
                     (x2 - _ix(pw * 0.03), y2 - _ix(ph * 0.13)), (50, 50, 70), 1)
            bal_sc = get_fit_scale(f"Most balanced: {balanced}",
                                   _ix(pw * 0.90), base_scale=0.32, thickness=1, min_scale=0.24)
            draw_centered_text_in_rect(frame, f"Most balanced: {balanced}",
                (x1, y2 - _ix(ph * 0.12), x2, y2 - _ix(ph * 0.06)),
                base_scale=bal_sc, color=COL_TEXT_ACCENT, thickness=1, outline=1)
            draw_centered_text_in_rect(frame,
                "Full data saved to Desktop/CapStone/simulation_results.xlsx",
                (x1, y2 - _ix(ph * 0.07), x2, y2 - _ix(ph * 0.01)),
                base_scale=0.28, color=COL_TEXT_DIM, thickness=1, outline=1)

    draw_bottom_bar(frame, "ESC Back to Simulations Hub  (only when complete)")

def draw_settings_screen(frame, settings_schema, selected_index, config,
                         format_value_fn, cursor_info=None, text_edit=False):
    layout = _settings_layout(frame)
    w, h = layout["w"], layout["h"]
    x1, y1, x2, y2 = layout["panel"]

    nav_active  = cursor_info is not None and cursor_info.get("active", False)
    x_zone      = cursor_info.get("x_zone", "center")  if nav_active else "center"
    adjust_pct  = cursor_info.get("adjust_pct", 0.0)   if nav_active else 0.0

    if text_edit:
        hint = "Type name | ENTER confirm | ESC cancel"
    elif nav_active:
        hint = "Move left [-]  |  center = select  |  [+] move right"
    else:
        hint = "UP/DOWN Select | LEFT/RIGHT Change | ENTER edit name | BACK"
    draw_top_bar(frame, "SETTINGS", hint)

    draw_panel(frame, x1, y1, x2, y2, fill=COL_BG_PANEL, alpha=0.94,
               border=COL_ACCENT, border_thickness=1)

    draw_centered_text(frame, "GAME SETTINGS", y1 + _ix((y2 - y1) * 0.07),
                       SCALE_HEADING, COL_TEXT_PRIMARY, thickness=1, outline=2)

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
            draw_selected_row(frame, x1 + _ix(w * 0.015), bar_y1, x2 - _ix(w * 0.015), bar_y2)

        label_color = COL_ACCENT if selected else COL_TEXT_DIM

        draw_outlined_text(frame, item['label'], x1 + _ix(w * 0.025), y,
                           SCALE_BODY, label_color, thickness=1, outline=2)

        # Value + optional +/- buttons for adjustable items
        value = format_value_fn(item)
        if value and not is_action:
            is_text = item.get("type") == "text"
            if selected and is_text:
                # Text field - blinking cursor when selected
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
                minus_text_col = COL_ON_ACTIVE if minus_active else (120, 180, 120)
                plus_text_col  = COL_ON_ACTIVE if plus_active  else (120, 180, 120)

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
                   fill=(8, 20, 35), alpha=0.90, border=COL_BORDER_HAIR, border_thickness=1)

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

    # Voice model download hint - shown when voice_model item is selected
    sel_item = settings_schema[selected_index] if selected_index < len(settings_schema) else {}
    if sel_item.get("key") == "voice_model":
        val = config.get("voice_model", "US English")
        if val == "US English":
            hint_bottom = "(type in browser) alphacephei.com/vosk/models  ->  vosk-model-small-en-us-0.15"
        else:
            hint_bottom = "(type in browser) alphacephei.com/vosk/models  ->  vosk-model-small-en-in-0.4"

    draw_bottom_bar(frame, hint_bottom)


# ============================================================
# FEATURES SCREEN
# ============================================================

def draw_features_screen(frame, features_schema, selected_index, config,
                         cursor_info=None):
    """
    Optional feature toggles - separate from program settings.
    All features default to OFF to preserve performance.
    """
    layout = _settings_layout(frame)
    w, h = layout["w"], layout["h"]
    x1, y1, x2, y2 = layout["panel"]

    nav_active = cursor_info is not None and cursor_info.get("active", False)
    x_zone     = cursor_info.get("x_zone", "center") if nav_active else "center"
    adjust_pct = cursor_info.get("adjust_pct", 0.0)  if nav_active else 0.0

    hint = "UP/DOWN Select | ENTER Toggle | BACK" if not nav_active else \
           "Move left [-]  |  center = select  |  [+] move right"
    draw_top_bar(frame, "FEATURES", hint)

    draw_panel(frame, x1, y1, x2, y2, fill=COL_BG_PANEL, alpha=0.94,
               border=COL_ACCENT, border_thickness=1)

    draw_centered_text(frame, "OPTIONAL FEATURES", y1 + _ix((y2 - y1) * 0.07),
                       SCALE_HEADING, COL_TEXT_PRIMARY, thickness=1, outline=2)
    draw_centered_text(frame, "All OFF by default  |  enable what you need",
                       y1 + _ix((y2 - y1) * 0.13), SCALE_CAPTION, COL_TEXT_DIM,
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
            draw_selected_row(frame, x1 + _ix(w * 0.015), y - bar_half,
                              x2 - _ix(w * 0.015), y + bar_half)

        label_color = COL_ACCENT if selected else COL_TEXT_DIM
        draw_outlined_text(frame, item['label'], x1 + _ix(w * 0.025), y,
                           SCALE_BODY, label_color, thickness=1, outline=2)

        if not is_back:
            if is_choice and selected:
                # [−] value [+] - same as settings screen, yellow theme
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
                    txt_col = COL_ON_ACTIVE if is_on else (140, 140, 60)
                    draw_centered_text_in_rect(frame, label,
                        (btn_x1, y - bar_half, btn_x2, y + bar_half),
                        base_scale=0.70, color=txt_col, thickness=2, outline=0)

                (vw, _), _ = cv2.getTextSize(val_text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
                draw_outlined_text(frame, val_text, minus_x1 - gap - vw, y,
                                   0.46, COL_TEXT_ACCENT, thickness=1, outline=2)
            elif key == "__personalities__":
                # Show current personality name + [>] indicator (opens sub-screen)
                from fair_play_ai import PERSONALITIES
                cur_name   = config.get("ai_personality", "Normal")
                cur_label  = PERSONALITIES.get(cur_name, {}).get("label", cur_name)
                arrow_col  = COL_MAGENTA if selected else (100, 40, 120)
                (tw, _), _ = cv2.getTextSize(cur_label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                draw_outlined_text(frame, cur_label, x2 - tw - _ix(w * 0.075), y,
                                   0.40, COL_TEXT_DIM, thickness=1, outline=1)
                draw_outlined_text(frame, "[>]", x2 - _ix(w * 0.068), y,
                                   0.44, arrow_col, thickness=2, outline=2)
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
                   fill=(8, 20, 8), alpha=0.90, border=COL_BORDER_HAIR, border_thickness=1)
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
        hint_bottom = f"{'[-] Decreasing...' if x_zone == 'minus' else 'Increasing... [+]'}  hold to continue"
    draw_bottom_bar(frame, hint_bottom)


# ============================================================
# GAME CATEGORY SCREEN  (3-level menu - level 2 + 3)
# ============================================================

def draw_game_category_screen(frame, categories, category_index, mode_index,
                               in_mode_list=False):
    """
    Two-panel game mode selector.
    Left panel  -- scrollable list of game categories.
    Right panel -- category description, or mode list when category opened.
    """
    w, h = _frame_size(frame)

    hint = ("W/S Navigate  |  Enter Open  |  ESC Back" if not in_mode_list
            else "W/S Select Mode  |  Enter Start  |  ESC Back")
    draw_top_bar(frame, "GAME MODES", hint)

    py1 = _ix(h * 0.105)
    py2 = _ix(h * 0.93)
    ph  = py2 - py1

    # Left panel -- x 3.6%, w 31%
    lx1 = _ix(w * 0.036)
    lx2 = lx1 + _ix(w * 0.31)
    draw_panel(frame, lx1, py1, lx2, py2,
               fill=COL_PANEL_BG, alpha=0.92, border=COL_BORDER_HAIR, border_thickness=1)

    header_h = _ix(ph * 0.10)
    draw_centered_text_in_rect(frame, "CATEGORIES",
        (lx1, py1, lx2, py1 + header_h),
        base_scale=SCALE_CAPTION, color=COL_TEXT_SECONDARY, thickness=1, outline=2)
    cv2.line(frame, (lx1, py1 + header_h), (lx2, py1 + header_h), COL_BORDER_HAIR, 1)

    n_cats = len(categories)
    VISIBLE_CATS = 7
    cat_y1 = py1 + header_h + 1
    cat_y2 = py2 - 1
    row_h  = _ix((cat_y2 - cat_y1) / VISIBLE_CATS)

    scroll_off = max(0, min(category_index - VISIBLE_CATS + 1, n_cats - VISIBLE_CATS))
    scroll_off = max(0, scroll_off)

    for i, cat in enumerate(categories):
        vis_idx = i - scroll_off
        if vis_idx < 0 or vis_idx >= VISIBLE_CATS:
            continue
        ry1 = cat_y1 + vis_idx * row_h
        ry2 = ry1 + row_h
        draw_row(frame, lx1, ry1, lx2, ry2, cat['label'], selected=(i == category_index))
        if vis_idx < VISIBLE_CATS - 1:
            cv2.line(frame, (lx1 + 3, ry2), (lx2 - 3, ry2), COL_BORDER_HAIR, 1)

    if scroll_off > 0:
        cv2.putText(frame, "^ more", (lx1 + _ix((lx2 - lx1) * 0.30), cat_y1 - 4),
                    FONT_PRIMARY, SCALE_MICRO, COL_TEXT_DIM, 1, cv2.LINE_AA)
    if scroll_off + VISIBLE_CATS < n_cats:
        cv2.putText(frame, "v more", (lx1 + _ix((lx2 - lx1) * 0.30), cat_y2 + _ix(h * 0.018)),
                    FONT_PRIMARY, SCALE_MICRO, COL_TEXT_DIM, 1, cv2.LINE_AA)

    # Right panel -- x 36.4%, w 60%
    rx1 = _ix(w * 0.364)
    rx2 = rx1 + _ix(w * 0.60)
    draw_panel(frame, rx1, py1, rx2, py2,
               fill=COL_PANEL_BG, alpha=0.92, border=COL_BORDER_HAIR, border_thickness=1)

    sel_cat = categories[category_index]
    rpw   = rx2 - rx1
    pad_x = _ix(rpw * 0.06)

    draw_centered_text_in_rect(frame, sel_cat['label'],
        (rx1, py1, rx2, py1 + header_h),
        base_scale=SCALE_HEADING, color=COL_TEXT_PRIMARY, thickness=1, outline=2)
    cv2.line(frame, (rx1, py1 + header_h), (rx2, py1 + header_h), COL_BORDER_HAIR, 1)

    if not in_mode_list:
        # Description view — dynamic y cursor
        raw_desc = sel_cat.get('desc', '')
        parts    = [p.strip() for p in raw_desc.replace('--', '\n').split('\n') if p.strip()]
        max_px   = rpw - 2 * pad_x
        lines    = []
        for part in parts:
            words = part.split()
            cur   = ''
            for word in words:
                test = f'{cur} {word}'.strip()
                (tw, _), _ = cv2.getTextSize(test, FONT_PRIMARY, SCALE_BODY, 1)
                if tw <= max_px:
                    cur = test
                else:
                    if cur:
                        lines.append(cur)
                    cur = word
            if cur:
                lines.append(cur)
            lines.append('')
        while lines and lines[-1] == '':
            lines.pop()

        cursor_y = py1 + header_h + _ix(ph * 0.04)
        line_gap = _ix(ph * 0.075)
        for line in lines[:8]:
            if line == '':
                cursor_y += _ix(ph * 0.020)
                continue
            draw_outlined_text(frame, line, rx1 + pad_x, cursor_y,
                               SCALE_BODY, COL_TEXT_SECONDARY, thickness=1, outline=2)
            cursor_y += line_gap

        # Separator + Modes section directly after description
        cursor_y += 14
        cv2.line(frame, (rx1 + pad_x, cursor_y), (rx2 - pad_x, cursor_y), COL_BORDER_HAIR, 1)
        cursor_y += 14
        draw_outlined_text(frame, 'MODES', rx1 + pad_x, cursor_y,
                           SCALE_MICRO, COL_TEXT_DIM, thickness=1, outline=1)
        cursor_y += 22
        for (ml, _) in sel_cat['modes'][:4]:
            draw_outlined_text(frame, ml, rx1 + pad_x + _ix(w * 0.01), cursor_y,
                               SCALE_BODY, COL_TEXT_SECONDARY, thickness=1, outline=1)
            cursor_y += 28

        draw_outlined_text(frame, 'Enter to open',
                           rx1 + pad_x, py2 - _ix(ph * 0.03),
                           SCALE_MICRO, COL_ACCENT, thickness=1, outline=2)

    else:
        # Mode list view
        modes   = sel_cat['modes']
        n_modes = len(modes)
        VISIBLE_MODES = 6
        m_y1  = py1 + header_h + 1
        m_y2  = py2 - _ix(ph * 0.08)
        m_row = _ix((m_y2 - m_y1) / VISIBLE_MODES)

        m_scroll = max(0, min(mode_index - VISIBLE_MODES + 1, n_modes - VISIBLE_MODES))
        m_scroll = max(0, m_scroll)

        for j, (ml, _) in enumerate(modes):
            vis_j = j - m_scroll
            if vis_j < 0 or vis_j >= VISIBLE_MODES:
                continue
            ry1 = m_y1 + vis_j * m_row
            ry2 = ry1 + m_row
            draw_row(frame, rx1, ry1, rx2, ry2, ml, selected=(j == mode_index))
            if vis_j < VISIBLE_MODES - 1:
                cv2.line(frame, (rx1 + 3, ry2), (rx2 - 3, ry2), COL_BORDER_HAIR, 1)

        if m_scroll + VISIBLE_MODES < n_modes:
            cv2.putText(frame, 'v more',
                        (rx1 + _ix(rpw * 0.40), m_y2 + _ix(h * 0.018)),
                        FONT_PRIMARY, SCALE_MICRO, COL_TEXT_DIM, 1, cv2.LINE_AA)

        draw_outlined_text(frame, 'ESC to go back',
                           rx1 + pad_x, py2 - _ix(ph * 0.03),
                           SCALE_MICRO, COL_TEXT_DIM, thickness=1, outline=2)

    draw_bottom_bar(frame, 'W/S Navigate  |  Enter Select  |  ESC Back  |  Q Quit')


# ============================================================
# SIMULATIONS HUB SCREEN
# ============================================================

def draw_simulations_hub_screen(frame, selected_index=0, sim_state=None):
    """
    Simulations hub - lists all available simulation types and lets the user
    select which to run with a description panel.
    """
    sim_state  = sim_state or {}
    layout     = _menu_layout(frame)
    w, h       = layout["w"], layout["h"]
    px1, py1, px2, py2 = layout["panel"]
    ph, pw     = py2 - py1, px2 - px1

    SIM_ENTRIES = [
        {
            "label":  "Fair Play vs AI",
            "key":    "fairplay",
            "desc":   (
                "Simulates ~99,000 rounds across 6 player strategies vs Easy / Normal / Hard AI. "
                "Identifies which AI difficulty is hardest to beat and which player strategy is most exploitable."
            ),
            "color":  COL_CYAN,
        },
        {
            "label":  "3-Way  PvPvAI",
            "key":    "pvpvai",
            "desc":   (
                "Simulates every P1 x P2 x AI combination in 1v1v1 format. "
                "First to 5 points wins. Reveals the best AI type and most balanced strategy pairing."
            ),
            "color":  COL_MAGENTA,
        },
        {
            "label":  "Clone Tournament",
            "key":    "tournament",
            "desc":   (
                "Round-robin tournament between all saved player clones. "
                "Each pair plays 200 rounds. Produces a leaderboard ranked by win rate. "
                "Requires 2+ players with 30+ rounds recorded."
            ),
            "color":  COL_YELLOW,
        },
    ]

    draw_top_bar(frame, "SIMULATIONS", "W/S Select  |  Enter Run  |  ESC Back")
    draw_panel(frame, px1, py1, px2, py2, fill=COL_BG_PANEL, alpha=0.94,
               border=COL_ACCENT, border_thickness=1)

    # Title and subtitle — bounded to the outer panel
    draw_centered_text_in_rect(frame, "SIMULATION LAB",
        (px1, py1, px2, py1 + _ix(ph * 0.10)),
        base_scale=SCALE_HEADING, color=COL_TEXT_PRIMARY, thickness=1, outline=2)
    draw_centered_text_in_rect(frame, "Run high-fidelity RPS strategy simulations",
        (px1, py1 + _ix(ph * 0.10), px2, py1 + _ix(ph * 0.18)),
        base_scale=SCALE_CAPTION, color=COL_TEXT_DIM, thickness=1, outline=2)

    cv2.line(frame,
             (px1 + _ix(pw * 0.05), py1 + _ix(ph * 0.19)),
             (px2 - _ix(pw * 0.05), py1 + _ix(ph * 0.19)),
             COL_BORDER_HAIR, 1)

    # Left: sim list
    lx1 = px1 + _ix(pw * 0.02)
    lx2 = px1 + _ix(pw * 0.42)
    for i, entry in enumerate(SIM_ENTRIES):
        bar_y    = py1 + _ix(ph * 0.24) + i * _ix(ph * 0.18)
        bar_half = _ix(ph * 0.07)
        sel      = (i == selected_index)
        if sel:
            draw_selected_row(frame, lx1, bar_y - bar_half, lx2, bar_y + bar_half)
        draw_row(frame, lx1, bar_y - bar_half, lx2, bar_y + bar_half,
                 entry['label'], selected=sel)

    # Right: description panel
    rx1 = px1 + _ix(pw * 0.46)
    rx2 = px2 - _ix(pw * 0.02)
    ry1 = py1 + _ix(ph * 0.20)
    ry2 = py2 - _ix(ph * 0.08)
    rw  = rx2 - rx1
    rh  = ry2 - ry1
    sel_entry = SIM_ENTRIES[selected_index]
    draw_panel(frame, rx1, ry1, rx2, ry2, fill=(8, 16, 8), alpha=0.92,
               border=COL_BORDER_HAIR, border_thickness=1)

    # Label header — within right panel
    draw_centered_text_in_rect(frame, sel_entry["label"],
        (rx1, ry1, rx2, ry1 + _ix(rh * 0.15)),
        base_scale=0.50, color=sel_entry["color"], thickness=2, outline=3)

    # Description — word-wrapped, all lines drawn within right panel
    desc  = sel_entry["desc"]
    words, lines, cur = desc.split(), [], ""
    max_chars = max(20, int(rw / (_ix(w * 0.012) + 1)))
    for word in words:
        test = f"{cur} {word}".strip()
        if len(test) <= max_chars:
            cur = test
        else:
            if cur: lines.append(cur)
            cur = word
    if cur: lines.append(cur)
    dy   = ry1 + _ix(rh * 0.20)
    dgap = _ix(rh * 0.11)
    for line in lines[:5]:
        sc = get_fit_scale(line, _ix(rw * 0.88), base_scale=0.36, thickness=1, min_scale=0.26)
        draw_centered_text_in_rect(frame, line,
            (rx1, dy - _ix(rh * 0.04), rx2, dy + _ix(rh * 0.08)),
            base_scale=sc, color=COL_TEXT_ACCENT, thickness=1, outline=2)
        dy += dgap

    # Previous results summary (if available)
    status = sim_state.get("status", "idle")
    if status == "done" and sim_state.get("results"):
        res = sim_state["results"]
        summary_y = ry2 - _ix(rh * 0.24)
        cv2.line(frame, (rx1 + _ix(rw * 0.04), summary_y - _ix(rh * 0.04)),
                 (rx2 - _ix(rw * 0.04), summary_y - _ix(rh * 0.04)),
                 sel_entry["color"], 1)
        draw_outlined_text(frame, "Last run:", rx1 + _ix(rw * 0.06),
                           summary_y, 0.36, COL_TEXT_DIM, thickness=1, outline=1)
        best_ai = res.get("best_ai", "?")
        draw_outlined_text(frame, f"Best AI: {best_ai}",
                           rx1 + _ix(rw * 0.06), summary_y + _ix(rh * 0.13),
                           0.36, COL_GREEN, thickness=1, outline=1)

    # Footer hint — within outer panel
    draw_centered_text_in_rect(frame, "Press Enter to run selected simulation",
        (px1, py2 - _ix(ph * 0.07), px2, py2),
        base_scale=0.36, color=COL_TEXT_DIM, thickness=1, outline=2)

    draw_bottom_bar(frame, "W/S Select  |  Enter Run Simulation  |  ESC Back")


# ============================================================
# CLONE SETUP SCREEN
# ============================================================

def draw_clone_setup_screen(frame, clone_state):
    import time as _time
    w, h = frame.shape[1], frame.shape[0]
    step = clone_state.get("step", "enter_name")
    msg  = clone_state.get("message", "")

    if step == "enter_name":
        draw_top_bar(frame, "C L O N E   M O D E", "ENTER to confirm  |  ESC Back")

        draw_outlined_text(frame, "CLONE MODE",
                           _ix(w * 0.50) - 80, _ix(h * 0.20),
                           SCALE_MICRO, COL_TEXT_DIM, thickness=1, outline=2)
        heading  = "Who are you playing as?"
        font_d   = cv2.FONT_HERSHEY_DUPLEX
        (hw, _), _ = cv2.getTextSize(heading, font_d, SCALE_DISPLAY_L, 2)
        hx = (w - hw) // 2
        cv2.putText(frame, heading, (hx, _ix(h * 0.30)),
                    font_d, SCALE_DISPLAY_L, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, heading, (hx, _ix(h * 0.30)),
                    font_d, SCALE_DISPLAY_L, COL_TEXT_PRIMARY, 2, cv2.LINE_AA)

        # Input rect
        box_x1 = _ix(w * 0.227)
        box_y1 = _ix(h * 0.43)
        box_x2 = box_x1 + _ix(w * 0.546)
        box_y2 = box_y1 + 70
        draw_panel(frame, box_x1, box_y1, box_x2, box_y2,
                   fill=COL_PANEL_BG, alpha=0.70, border=COL_ACCENT, border_thickness=1)

        name_text = clone_state.get("text_buffer", "")
        font_d2   = cv2.FONT_HERSHEY_DUPLEX
        (tw2, th2), _ = cv2.getTextSize(name_text or "A", font_d2, SCALE_HEADING, 2)
        ty = box_y1 + (70 + th2) // 2
        tx = box_x1 + 14
        if name_text:
            cv2.putText(frame, name_text, (tx, ty), font_d2, SCALE_HEADING,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, name_text, (tx, ty), font_d2, SCALE_HEADING,
                        COL_TEXT_PRIMARY, 2, cv2.LINE_AA)
        t = _time.time()
        if int(t * 2) % 2 == 0:
            (cw, _), _ = cv2.getTextSize(name_text, font_d2, SCALE_HEADING, 2)
            cur_x = tx + cw + 2
            cv2.rectangle(frame, (cur_x, ty - th2 - 2), (cur_x + 2, ty + 4), COL_ACCENT, -1)

        draw_centered_text(frame, "A-Z  *  up to 20 characters  *  ENTER to confirm",
                           _ix(h * 0.60), SCALE_MICRO, COL_TEXT_DIM, thickness=1, outline=2)

        if msg:
            draw_centered_text(frame, msg, _ix(h * 0.68), SCALE_CAPTION,
                               COL_AMBER, thickness=1, outline=2)

        draw_bottom_bar(frame, "Type your name  |  ENTER Confirm  |  ESC Back")

    elif step == "select_opponent":
        draw_top_bar(frame, "C L O N E   M O D E", "UP/DOWN Navigate  |  ENTER Fight  |  ESC Back")

        player_name = clone_state.get("player_name", "")
        draw_centered_text(frame, f"PLAYING AS  {player_name.upper()}",
                           _ix(h * 0.14), SCALE_CAPTION, COL_TEXT_SECONDARY,
                           thickness=1, outline=2)

        px1 = _ix(w * 0.18);  px2 = _ix(w * 0.82)
        py1 = _ix(h * 0.18);  py2 = _ix(h * 0.90)
        ph  = py2 - py1
        pw  = px2 - px1
        draw_panel(frame, px1, py1, px2, py2,
                   fill=COL_PANEL_BG, alpha=0.88, border=COL_BORDER_HAIR, border_thickness=1)

        # Title strip
        strip_y2 = py1 + 38
        draw_panel(frame, px1, py1, px2, strip_y2,
                   fill=COL_PANEL_BG, alpha=0.92, border=COL_BORDER_HAIR, border_thickness=1)
        draw_centered_text_in_rect(frame, "S E L E C T   O P P O N E N T",
            (px1, py1, px2, strip_y2),
            base_scale=SCALE_MICRO, color=COL_TEXT_DIM, thickness=1, outline=1)
        cv2.line(frame, (px1, strip_y2), (px2, strip_y2), COL_BORDER_HAIR, 1)

        available    = clone_state.get("available", [])
        selected_idx = clone_state.get("selected_index", 0)
        row_h        = 64
        ry           = strip_y2

        for i, (name, count) in enumerate(available):
            selected = (i == selected_idx)
            draw_row(frame, px1, ry, px2, ry + row_h,
                     label=name,
                     sub_label=f"{count} rounds recorded",
                     right_hint="ENTER FIGHT",
                     selected=selected)
            ry += row_h

        if clone_state.get("profiles_updating"):
            t2 = time.monotonic()
            dots = "." * (1 + int(t2 * 2) % 3)
            draw_outlined_text(frame, f"Updating profiles{dots}",
                               px1 + _ix(pw * 0.02), py2 - 20,
                               SCALE_CAPTION, COL_ACCENT, thickness=1, outline=2)

        if msg:
            draw_centered_text(frame, msg, py2 + 14, SCALE_CAPTION,
                               COL_AMBER, thickness=1, outline=2)

        draw_bottom_bar(frame, "UP/DOWN Navigate  *  ENTER Fight  *  ESC Back")

    elif step == "no_profiles":
        draw_top_bar(frame, "C L O N E   M O D E", "Enter/ESC to go back")

        px1 = _ix(w * 0.18);  px2 = _ix(w * 0.82)
        py1 = _ix(h * 0.18);  py2 = _ix(h * 0.90)
        ph  = py2 - py1
        draw_panel(frame, px1, py1, px2, py2,
                   fill=COL_PANEL_BG, alpha=0.88, border=COL_BORDER_HAIR, border_thickness=1)

        strip_y2 = py1 + 38
        draw_panel(frame, px1, py1, px2, strip_y2,
                   fill=COL_PANEL_BG, alpha=0.92, border=COL_BORDER_HAIR, border_thickness=1)
        draw_centered_text_in_rect(frame, "S E L E C T   O P P O N E N T",
            (px1, py1, px2, strip_y2),
            base_scale=SCALE_MICRO, color=COL_TEXT_DIM, thickness=1, outline=1)
        cv2.line(frame, (px1, strip_y2), (px2, strip_y2), COL_BORDER_HAIR, 1)

        draw_centered_text_in_rect(frame, "NO CLONES AVAILABLE YET",
            (px1, py1 + _ix(ph * 0.12), px2, py1 + _ix(ph * 0.28)),
            base_scale=SCALE_HEADING, color=COL_TEXT_PRIMARY, thickness=1, outline=3)

        for j, line in enumerate([
            "To create a clone, a player needs to:",
            "1. Enter their name here",
            "2. Play 30+ rounds in any mode",
            "3. Patterns are learned automatically",
        ]):
            draw_centered_text_in_rect(frame, line,
                (px1 + 16, py1 + _ix(ph * (0.32 + j * 0.10)),
                 px2 - 16, py1 + _ix(ph * (0.42 + j * 0.10))),
                base_scale=SCALE_BODY,
                color=COL_TEXT_SECONDARY if j == 0 else COL_TEXT_DIM,
                thickness=1, outline=1)

        all_players = clone_state.get("all_players", [])
        if all_players:
            draw_centered_text_in_rect(frame, "Players recording:",
                (px1, py1 + _ix(ph * 0.72), px2, py1 + _ix(ph * 0.80)),
                base_scale=SCALE_CAPTION, color=COL_TEXT_SECONDARY,
                thickness=1, outline=1)
            for k, (name, count) in enumerate(all_players[:4]):
                draw_centered_text_in_rect(frame, f"{name}: {count}/30 rounds",
                    (px1, py1 + _ix(ph * (0.80 + k * 0.05)),
                     px2, py1 + _ix(ph * (0.85 + k * 0.05))),
                    base_scale=SCALE_CAPTION, color=COL_TEXT_DIM,
                    thickness=1, outline=1)

        draw_bottom_bar(frame, "ENTER Start Fair Play  |  ESC Back to menu")

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

    step = stats_state.get("step", "select")

    if step == "select":
        draw_top_bar(frame, "PLAYER STATS", "UP/DOWN Move | SELECT | BACK")
        px1 = _ix(w * 0.18);  px2 = _ix(w * 0.82)
        py1 = _ix(h * 0.20);  py2 = _ix(h * 0.90)
        draw_panel(frame, px1, py1, px2, py2,
                   fill=COL_PANEL_BG, alpha=0.88, border=COL_BORDER_HAIR, border_thickness=1)
        pw = px2 - px1

        title_h = 38
        sc_t = get_fit_scale("SELECT PLAYER", pw - _ix(pw * 0.10),
                             base_scale=SCALE_HEADING, thickness=1)
        draw_outlined_text(frame, "SELECT PLAYER",
                           px1 + _ix(pw * 0.05), py1 + _ix(title_h * 0.72),
                           sc_t, COL_TEXT_PRIMARY, thickness=1, outline=2)
        cv2.line(frame, (px1, py1 + title_h), (px2, py1 + title_h), COL_BORDER_HAIR, 1)

        players      = stats_state.get("players", [])
        selected_idx = stats_state.get("selected_index", 0)
        row_h   = 64
        row_top = py1 + title_h

        for i, (name, count) in enumerate(players):
            ry1 = row_top + i * row_h
            ry2 = ry1 + row_h
            if ry2 > py2:
                break
            draw_row(frame, px1, ry1, px2, ry2, name,
                     selected=(i == selected_idx),
                     sub_label=f"{count} rounds",
                     right_hint="ENTER OPEN")
            if i < len(players) - 1:
                cv2.line(frame, (px1 + 3, ry2), (px2 - 3, ry2), COL_BORDER_HAIR, 1)

        draw_bottom_bar(frame, "UP/DOWN Navigate  |  ENTER Open  |  ESC Back")
        return

    if step == "no_profiles":
        draw_top_bar(frame, "PLAYER STATS", "ESC Back")
        px1 = _ix(w * 0.20);  px2 = _ix(w * 0.80)
        py1 = _ix(h * 0.25);  py2 = _ix(h * 0.82)
        ph2 = py2 - py1
        draw_panel(frame, px1, py1, px2, py2,
                   fill=COL_PANEL_BG, alpha=0.88, border=COL_BORDER_HAIR, border_thickness=1)
        draw_centered_text_in_rect(frame, "NO STATS YET",
            (px1, py1 + _ix(ph2 * 0.10), px2, py1 + _ix(ph2 * 0.34)),
            base_scale=SCALE_HEADING, color=COL_TEXT_PRIMARY, thickness=1, outline=3)
        draw_centered_text_in_rect(frame, "Play a few rounds first to start tracking your stats.",
            (px1 + 16, py1 + _ix(ph2 * 0.40), px2 - 16, py1 + _ix(ph2 * 0.58)),
            base_scale=SCALE_BODY, color=COL_TEXT_SECONDARY, thickness=1, outline=2)
        draw_centered_text_in_rect(frame, "Head to Game Modes and pick Fair Play or Challenge.",
            (px1 + 16, py1 + _ix(ph2 * 0.58), px2 - 16, py1 + _ix(ph2 * 0.76)),
            base_scale=SCALE_BODY, color=COL_TEXT_DIM, thickness=1, outline=2)
        draw_bottom_bar(frame, "ESC Back to menu")
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
    draw_panel(frame, x1, y1, x2, y2, fill=COL_PANEL_BG, alpha=0.88,
               border=COL_BORDER_HAIR, border_thickness=1)

    pw, ph = x2 - x1, y2 - y1

    # -- Mode filter strip - ALWAYS drawn so user can still navigate ------
    _FILTERS = ["All", "FairPlay", "Challenge", "Cheat", "Clone"]
    strip_y1 = y1 + _ix(ph * 0.01)
    strip_y2 = y1 + _ix(ph * 0.10)
    strip_w  = (x2 - x1) // len(_FILTERS)
    for fi, flab in enumerate(_FILTERS):
        fx1 = x1 + fi * strip_w
        fx2 = fx1 + strip_w
        active = (flab == cur_flt)
        draw_panel(frame, fx1 + 2, strip_y1 + 2, fx2 - 2, strip_y2 - 2,
                   fill=COL_PANEL_BG, alpha=0.85,
                   border=COL_ACCENT if active else COL_BORDER_HAIR, border_thickness=1)
        if active:
            cv2.line(frame, (fx1 + 4, strip_y2 - 4), (fx2 - 4, strip_y2 - 4), COL_ACCENT, 2)
        sc  = get_fit_scale(flab, strip_w - 8, base_scale=SCALE_CAPTION, thickness=1, min_scale=0.24)
        draw_centered_text_in_rect(frame, flab,
            (fx1 + 2, strip_y1 + 2, fx2 - 2, strip_y2 - 2),
            base_scale=sc,
            color=COL_TEXT_PRIMARY if active else COL_TEXT_SECONDARY,
            thickness=1, outline=2)

    # -- View tab strip - ALWAYS drawn ------------------------------------
    _TABS  = [("overview", "Overview"), ("history", "Match History")]
    tab_y1 = strip_y2 + _ix(ph * 0.005)
    tab_y2 = tab_y1 + _ix(ph * 0.08)
    tab_w  = (x2 - x1) // len(_TABS)
    for ti, (tid, tlab) in enumerate(_TABS):
        tx1    = x1 + ti * tab_w
        tx2    = tx1 + tab_w
        active = (tid == cur_tab)
        draw_panel(frame, tx1 + 2, tab_y1 + 2, tx2 - 2, tab_y2 - 2,
                   fill=COL_PANEL_BG, alpha=0.90,
                   border=COL_ACCENT if active else COL_BORDER_HAIR, border_thickness=1)
        if active:
            cv2.line(frame, (tx1 + 4, tab_y2 - 4), (tx2 - 4, tab_y2 - 4), COL_ACCENT, 2)
        draw_centered_text_in_rect(frame, tlab,
            (tx1 + 4, tab_y1 + 2, tx2 - 4, tab_y2 - 2),
            base_scale=SCALE_CAPTION,
            color=COL_TEXT_PRIMARY if active else COL_TEXT_SECONDARY,
            thickness=1, outline=2)

    content_y1 = tab_y2 + _ix(ph * 0.01)
    body_y1    = content_y1 + _ix((y2 - content_y1) * 0.09)

    # -- No data for this filter - show helpful guidance inside content area
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

    # -- Round count + filter note -----------------------------------------
    flt_note = f"  ({cur_flt})" if cur_flt != "All" else ""
    draw_centered_text(frame, f"{data['round_count']} rounds{flt_note}",
                       content_y1 + _ix((y2 - content_y1) * 0.04),
                       0.42, COL_TEXT_DIM, thickness=1, outline=2)

    name = data.get("player_name", "Unknown")

    # ====================================================================
    # OVERVIEW TAB
    # ====================================================================
    if cur_tab == "overview":
        col_left_x  = x1 + _ix(pw * 0.04)
        col_right_x = x1 + _ix(pw * 0.54)
        bar_w       = _ix(pw * 0.38)
        bar_h       = _ix(ph * 0.028)
        body_ph     = y2 - body_y1

        # Left: Results
        sec_y = body_y1
        draw_outlined_text(frame, "RESULTS", col_left_x, sec_y,
                           SCALE_CAPTION, COL_TEXT_DIM, thickness=1, outline=2)
        sec_y += _ix(body_ph * 0.07)
        for label, pct, pct_color in [
            ("Win",  data["win_pct"],  COL_GREEN),
            ("Loss", data["loss_pct"], COL_RED),
            ("Draw", data["draw_pct"], COL_TEXT_DIM),
        ]:
            lbl_str = f"{label}:"
            pct_str = f"{pct:.0%}"
            draw_outlined_text(frame, lbl_str, col_left_x, sec_y,
                               SCALE_BODY, COL_TEXT_SECONDARY, thickness=1, outline=2)
            (lw, _lh), _ = cv2.getTextSize(lbl_str, cv2.FONT_HERSHEY_SIMPLEX, SCALE_BODY, 1)
            draw_outlined_text(frame, pct_str, col_left_x + lw + 6, sec_y,
                               SCALE_BODY, pct_color, thickness=1, outline=2)
            _draw_bar(frame, col_left_x + _ix(pw * 0.18), sec_y - bar_h + 2,
                      bar_w - _ix(pw * 0.18), bar_h, pct, pct_color, bg_color=(28, 28, 28))
            sec_y += _ix(body_ph * 0.065)

        # Left: Gestures
        sec_y += _ix(body_ph * 0.02)
        draw_outlined_text(frame, "GESTURES", col_left_x, sec_y,
                           SCALE_CAPTION, COL_TEXT_DIM, thickness=1, outline=2)
        sec_y += _ix(body_ph * 0.07)
        freq = data.get("gesture_freq", {})
        for g in ("Rock", "Paper", "Scissors"):
            pct      = freq.get(g, 0)
            pct_color = get_gesture_color(g)
            pct_str  = f"{pct:.0%}"
            draw_outlined_text(frame, f"{g}:", col_left_x, sec_y,
                               SCALE_BODY, COL_TEXT_SECONDARY, thickness=1, outline=2)
            (lw2, _), _ = cv2.getTextSize(f"{g}:", cv2.FONT_HERSHEY_SIMPLEX, SCALE_BODY, 1)
            draw_outlined_text(frame, pct_str, col_left_x + lw2 + 6, sec_y,
                               SCALE_BODY, pct_color, thickness=1, outline=2)
            _draw_bar(frame, col_left_x + _ix(pw * 0.22), sec_y - bar_h + 2,
                      bar_w - _ix(pw * 0.22), bar_h, pct, pct_color, bg_color=(28, 28, 28))
            sec_y += _ix(body_ph * 0.065)

        # Right: After-outcome response
        sec_y2 = body_y1
        draw_outlined_text(frame, "AFTER OUTCOME", col_right_x, sec_y2,
                           SCALE_CAPTION, COL_TEXT_DIM, thickness=1, outline=2)
        sec_y2 += _ix(body_ph * 0.07)
        for outcome in ("win", "lose", "draw"):
            resp = data.get("outcome_response", {}).get(outcome, {})
            stay = resp.get("stay", 0)
            up   = resp.get("upgrade", 0)
            down = resp.get("downgrade", 0)
            line = f"{outcome.title()}: stay {stay:.0%} | up {up:.0%} | dn {down:.0%}"
            avail_w = x2 - col_right_x - _ix(pw * 0.04)
            sc = get_fit_scale(line, avail_w, base_scale=SCALE_BODY, thickness=1, min_scale=0.26)
            draw_outlined_text(frame, line, col_right_x, sec_y2,
                               sc, COL_TEXT_SECONDARY, thickness=1, outline=2)
            sec_y2 += _ix(body_ph * 0.065)

        # Traits
        traits_y = y2 - _ix(ph * 0.26)
        cv2.line(frame, (x1 + _ix(pw * 0.04), traits_y - _ix(ph * 0.01)),
                 (x2 - _ix(pw * 0.04), traits_y - _ix(ph * 0.01)), COL_BORDER_HAIR, 1)
        draw_outlined_text(frame, "PLAYER TRAITS", col_left_x, traits_y,
                           SCALE_CAPTION, COL_TEXT_DIM, thickness=1, outline=2)
        traits_y += _ix(ph * 0.055)
        for trait in traits[:3]:
            sc = get_fit_scale(trait, _ix(pw * 0.90), base_scale=SCALE_BODY, thickness=1, min_scale=0.26)
            draw_outlined_text(frame, trait, col_left_x, traits_y,
                               sc, COL_TEXT_SECONDARY, thickness=1, outline=2)
            traits_y += _ix(ph * 0.048)

        # 14-px square history chips
        rounds = stats_state.get("rounds", [])
        if rounds:
            recent   = rounds[-24:]
            n        = len(recent)
            chip     = 14
            chip_gap = 2
            total_chips_w = n * (chip + chip_gap) - chip_gap
            chip_x0 = (x1 + x2) // 2 - total_chips_w // 2
            chip_y  = y2 - _ix(ph * 0.06)
            _col_map = {"win": COL_GREEN, "lose": COL_RED, "draw": COL_TEXT_SECONDARY}
            for i, r in enumerate(recent):
                outcome = r.get("outcome", r.get("player_outcome", "draw"))
                col = _col_map.get(outcome, COL_TEXT_DIM)
                cx = chip_x0 + i * (chip + chip_gap)
                cv2.rectangle(frame, (cx, chip_y), (cx + chip, chip_y + chip), col, -1)

    # ====================================================================
    # HISTORY TAB
    # ====================================================================
    else:
        sessions = stats_state.get("sessions", [])
        body_ph  = y2 - body_y1

        if not sessions:
            draw_centered_text(frame, "No session history yet",
                               body_y1 + _ix(body_ph * 0.40),
                               SCALE_BODY, COL_TEXT_DIM, thickness=1, outline=2)
        else:
            # Header row
            hdr_y = body_y1 + _ix(body_ph * 0.04)
            for txt, xpct in [("DATE / TIME", 0.04), ("MODE", 0.30), ("SCORE", 0.47),
                               ("WIN%", 0.62), ("AVG RT", 0.76)]:
                draw_outlined_text(frame, txt, x1 + _ix(pw * xpct), hdr_y,
                                   SCALE_MICRO, COL_TEXT_DIM, thickness=1, outline=2)
            cv2.line(frame,
                     (x1 + _ix(pw * 0.03), hdr_y + _ix(body_ph * 0.045)),
                     (x2 - _ix(pw * 0.03), hdr_y + _ix(body_ph * 0.045)),
                     COL_BORDER_HAIR, 1)

            row_h = _ix(body_ph * 0.13)
            row_y = hdr_y + _ix(body_ph * 0.07)
            for sess in reversed(sessions):   # most recent first
                w_rate = sess.get("win_rate", 0)
                wins   = sess.get("wins", 0)
                losses = sess.get("losses", 0)
                draws  = sess.get("draws", 0)
                rt     = sess.get("avg_reaction_ms")
                mode   = sess.get("mode", "?")
                date   = sess.get("date", "?")

                if w_rate >= 0.5:
                    tint_col = COL_GREEN
                elif w_rate < 0.35:
                    tint_col = COL_RED
                else:
                    tint_col = None
                if tint_col:
                    ty1 = row_y - row_h + 2
                    ty2 = row_y + 2
                    if ty1 >= 0 and ty2 <= frame.shape[0]:
                        roi = frame[ty1:ty2, x1:x2]
                        if roi.size > 0:
                            ov = roi.copy()
                            cv2.rectangle(ov, (0, 0), (x2 - x1, ty2 - ty1), tint_col, -1)
                            cv2.addWeighted(ov, 0.06, roi, 0.94, 0, roi)

                dot_col = COL_GREEN if w_rate >= 0.5 else (COL_RED if w_rate < 0.35 else COL_TEXT_DIM)
                dot_x   = x1 + _ix(pw * 0.015)
                cv2.circle(frame, (dot_x, row_y - 4), 4, dot_col, -1)
                for txt, xpct in [
                    (date, 0.04),
                    (mode, 0.30),
                    (f"{wins}W {losses}L {draws}D", 0.47),
                    (f"{w_rate:.0%}", 0.62),
                    (f"{rt}ms" if rt else "n/a", 0.76),
                ]:
                    sc = get_fit_scale(txt, _ix(pw * 0.22),
                                       base_scale=SCALE_CAPTION, thickness=1, min_scale=0.24)
                    draw_outlined_text(frame, txt, x1 + _ix(pw * xpct), row_y,
                                       sc, COL_TEXT_PRIMARY, thickness=1, outline=2)
                cv2.line(frame, (x1 + _ix(pw * 0.03), row_y + 4),
                         (x2 - _ix(pw * 0.03), row_y + 4), COL_BORDER_HAIR, 1)
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

    step        = tut_state.get("step", {})
    step_idx    = tut_state.get("step_index", 0)
    total       = tut_state.get("total_steps", 6)
    detected    = tut_state.get("detected_gesture", "Unknown")
    conf        = tut_state.get("gesture_confidence", 0.0)
    voice_mode  = tut_state.get("voice_mode", False)

    step_id      = step.get("id", "")
    gesture_name = step.get("gesture_name", step.get("target_gesture", "Rock"))
    instruction  = step.get("instruction", "")
    description  = step.get("description", step.get("sub", ""))
    step_num     = step_idx + 1

    mode_label = "VOICE MODE" if voice_mode else "HOW TO PLAY"
    draw_top_bar(frame, mode_label, f"Step {step_num} of {total}  |  ESC to skip")

    # 3px progress bar + captions immediately below top bar
    prog       = step_idx / max(total - 1, 1) if total > 1 else 1.0
    bar_y      = _ix(h * 0.065)
    bar_x      = _ix(w * 0.02)
    bar_w_full = w - _ix(w * 0.04)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w_full, bar_y + 3), (28, 28, 28), -1)
    fill_x = bar_x + _ix(bar_w_full * prog)
    if fill_x > bar_x:
        cv2.rectangle(frame, (bar_x, bar_y), (fill_x, bar_y + 3), COL_ACCENT, -1)
    step_lbl = f"STEP {step_num:02d}  *  OF {total:02d}"
    draw_outlined_text(frame, step_lbl, bar_x, bar_y + 16,
                       SCALE_MICRO, COL_TEXT_DIM, thickness=1, outline=2)
    if gesture_name and step_id not in ("done", ""):
        show_lbl = f"SHOW {gesture_name.upper()}"
        (tw, _), _ = cv2.getTextSize(show_lbl, cv2.FONT_HERSHEY_SIMPLEX, SCALE_MICRO, 1)
        draw_outlined_text(frame, show_lbl, w - _ix(w * 0.02) - tw, bar_y + 16,
                           SCALE_MICRO, COL_ACCENT, thickness=1, outline=2)

    # Column bounds (y=18% to y=82%)
    body_y1 = _ix(h * 0.18)
    body_y2 = _ix(h * 0.82)
    lx1 = _ix(w * 0.04);  lx2 = _ix(w * 0.52)
    rx1 = _ix(w * 0.54);  rx2 = _ix(w * 0.92)

    # ── Left panel ────────────────────────────────────────────────────────
    draw_panel(frame, lx1, body_y1, lx2, body_y2)

    if step_id == "done":
        draw_outlined_text(frame, "YOU'RE READY",
                           _ix(w * 0.05), _ix(h * 0.32),
                           SCALE_BODY, COL_ACCENT, thickness=1, outline=2)
        draw_outlined_text(frame, "You know the basics.",
                           _ix(w * 0.05), _ix(h * 0.44),
                           SCALE_BODY, COL_TEXT_SECONDARY, thickness=1, outline=2)
        draw_outlined_text(frame, "Press ENTER to return to the menu",
                           _ix(w * 0.05), _ix(h * 0.58),
                           SCALE_CAPTION, COL_TEXT_DIM, thickness=1, outline=2)
    else:
        draw_outlined_text(frame, "NOW TRY",
                           _ix(w * 0.05), _ix(h * 0.22),
                           SCALE_MICRO, COL_TEXT_DIM, thickness=1, outline=2)
        font_d = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, gesture_name, (_ix(w * 0.05), _ix(h * 0.32)),
                    font_d, SCALE_DISPLAY_L, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, gesture_name, (_ix(w * 0.05), _ix(h * 0.32)),
                    font_d, SCALE_DISPLAY_L, COL_TEXT_PRIMARY, 2, cv2.LINE_AA)
        sc_instr = get_fit_scale(instruction, lx2 - _ix(w * 0.07),
                                 base_scale=SCALE_BODY, thickness=1, min_scale=0.28)
        draw_outlined_text(frame, instruction,
                           _ix(w * 0.05), _ix(h * 0.44),
                           sc_instr, COL_TEXT_SECONDARY, thickness=1, outline=2)
        if description:
            sc_desc = get_fit_scale(description, lx2 - _ix(w * 0.07),
                                    base_scale=SCALE_CAPTION, thickness=1, min_scale=0.24)
            draw_outlined_text(frame, description,
                               _ix(w * 0.05), _ix(h * 0.54),
                               sc_desc, COL_TEXT_DIM, thickness=1, outline=2)
        draw_outlined_text(frame, "Hold gesture to advance...",
                           _ix(w * 0.05), _ix(h * 0.64),
                           SCALE_CAPTION, COL_TEXT_DIM, thickness=1, outline=2)

    # ── Right panel — glyph panel (or empty on final step) ────────────────
    draw_panel(frame, rx1, body_y1, rx2, body_y2)

    if step_id != "done" and gesture_name:
        glyph_cx   = _ix(w * 0.73)
        glyph_cy   = _ix(h * 0.50)
        panel_half = 120
        draw_panel(frame,
                   glyph_cx - panel_half, glyph_cy - panel_half,
                   glyph_cx + panel_half, glyph_cy + panel_half,
                   fill=COL_PANEL_BG, alpha=0.70,
                   border=COL_BORDER_HAIR, border_thickness=1)
        draw_gesture_glyph(frame, gesture_name,
                           (glyph_cx - 70, glyph_cy - 70,
                            glyph_cx + 70, glyph_cy + 70),
                           COL_ACCENT)

    # ── Detection badge centred at y=83% ──────────────────────────────────
    badge_label = f"DETECTED  {detected.upper()}  {conf:.2f}"
    font_s = cv2.FONT_HERSHEY_SIMPLEX
    (tw, _), _ = cv2.getTextSize(badge_label, font_s, SCALE_CAPTION, 1)
    badge_x = (w - tw - 60) // 2
    draw_gesture_badge(frame, detected, conf, badge_x, _ix(h * 0.83))

    draw_bottom_bar(frame, "Hold gesture to advance  *  ESC to skip")


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

    # Landmark dots - clipped so they don't paint over the diagnostic info panel.
    # Info panel occupies roughly x: 2%-55%, y: 15%-76% of the frame.
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
      inactive    - grey dot at fingertip (shows tracking is working)
      warming_up  - white pulsing ring + teal progress arc (counting frames)
      active      - solid cyan circle + dwell arc filling as hover accumulates
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
        # No hand detected - just show the hint badge
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

    # -- Cursor circle ---------------------------------------------------- #
    if active:
        cv2.circle(frame, (px, py), 16, (255, 220, 0), -1)
        cv2.circle(frame, (px, py), 17, (0, 0, 0), 2)

        # Dwell arc: fills as hover time accumulates, shifts cyan > red
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
        # Hand detected but not yet pointing - grey dot so user can
        # see tracking is live and confirm coordinate mapping is correct
        cv2.circle(frame, (px, py), 8, (90, 90, 90), 1)

    # -- Status badge (bottom-left) --------------------------------------- #
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


# ============================================================
# LOGIN SCREEN
# ============================================================

def draw_login_screen(frame, login_text="", saved_name="", verified_players=None):
    """
    Login / name-entry screen shown on first launch or player switch.
    """
    import time as _time

    w, h = frame.shape[1], frame.shape[0]
    t    = _time.time()

    recent = list(verified_players or [])
    if saved_name and saved_name not in recent:
        recent.insert(0, saved_name)

    draw_top_bar(frame, "R P S   R O B O T", "ENTER to confirm")

    # ── Heading block centred at y=18% ───────────────────────────────────
    draw_outlined_text(frame, "NICE TO MEET YOU",
                       _ix(w * 0.50) - 120, _ix(h * 0.20),
                       SCALE_MICRO, COL_TEXT_DIM, thickness=1, outline=2)
    heading = "What should we call you?"
    font_d  = cv2.FONT_HERSHEY_DUPLEX
    (hw, _), _ = cv2.getTextSize(heading, font_d, SCALE_DISPLAY_L, 2)
    hx = (w - hw) // 2
    cv2.putText(frame, heading, (hx, _ix(h * 0.30)),
                font_d, SCALE_DISPLAY_L, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, heading, (hx, _ix(h * 0.30)),
                font_d, SCALE_DISPLAY_L, COL_TEXT_PRIMARY, 2, cv2.LINE_AA)
    caption = "Type your name and press ENTER"
    sc_cap  = get_fit_scale(caption, _ix(w * 0.60), base_scale=SCALE_CAPTION,
                            thickness=1, min_scale=0.24)
    draw_centered_text(frame, caption, _ix(h * 0.37),
                       sc_cap, COL_TEXT_DIM, thickness=1, outline=2)

    # ── Input rectangle: x=22.7%, y=43%, w=54.6%, h=70px ────────────────
    box_x1 = _ix(w * 0.227)
    box_y1 = _ix(h * 0.43)
    box_x2 = box_x1 + _ix(w * 0.546)
    box_y2 = box_y1 + 70
    draw_panel(frame, box_x1, box_y1, box_x2, box_y2,
               fill=COL_PANEL_BG, alpha=0.70, border=COL_ACCENT, border_thickness=1)

    txt       = login_text or ""
    font_d2   = cv2.FONT_HERSHEY_DUPLEX
    (tw2, th2), _ = cv2.getTextSize(txt or "A", font_d2, SCALE_HEADING, 2)
    ty        = box_y1 + (70 + th2) // 2
    tx        = box_x1 + 14
    if txt:
        cv2.putText(frame, txt, (tx, ty), font_d2, SCALE_HEADING,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, txt, (tx, ty), font_d2, SCALE_HEADING,
                    COL_TEXT_PRIMARY, 2, cv2.LINE_AA)
    if int(t * 2) % 2 == 0:
        (cw, _), _ = cv2.getTextSize(txt, font_d2, SCALE_HEADING, 2)
        cur_x = tx + cw + 2
        cv2.rectangle(frame, (cur_x, ty - th2 - 2), (cur_x + 2, ty + 4), COL_ACCENT, -1)

    # ── Char counter top-right of box ────────────────────────────────────
    char_count = len(txt)
    counter_str = f"{char_count}/20"
    counter_col = COL_AMBER if char_count >= 18 else COL_TEXT_DIM
    draw_outlined_text(frame, counter_str, box_x2 - _ix(w * 0.06), box_y1 - 6,
                       SCALE_MICRO, counter_col, thickness=1, outline=2)

    # ── Caption below input ───────────────────────────────────────────────
    draw_centered_text(frame, "A-Z  *  up to 20 characters  *  press ENTER when done",
                       _ix(h * 0.60), SCALE_MICRO, COL_TEXT_DIM, thickness=1, outline=2)

    # ── Recent players chips at y=68% ────────────────────────────────────
    if recent:
        draw_outlined_text(frame, "OR PICK A RECENT PLAYER",
                           _ix(w * 0.10), _ix(h * 0.68),
                           SCALE_MICRO, COL_TEXT_DIM, thickness=1, outline=2)
        chip_x = _ix(w * 0.10)
        chip_y = _ix(h * 0.73)
        for name in recent[:5]:
            is_sel = (name == saved_name)
            (nw, nh), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, SCALE_CAPTION, 1)
            chip_w = nw + 32
            chip_h = nh + 16
            draw_panel(frame, chip_x, chip_y, chip_x + chip_w, chip_y + chip_h,
                       fill=COL_PANEL_BG, alpha=0.85,
                       border=COL_ACCENT if is_sel else COL_BORDER_HAIR,
                       border_thickness=1)
            draw_outlined_text(frame, name, chip_x + 16, chip_y + chip_h - 8,
                               SCALE_CAPTION,
                               COL_ACCENT if is_sel else COL_TEXT_SECONDARY,
                               thickness=1, outline=2)
            chip_x += chip_w + 8
            if chip_x > _ix(w * 0.88):
                break

    draw_bottom_bar(frame, "Type name  *  ENTER Confirm  *  ESC Back")


# ============================================================
# HARDWARE TEST VIEW
# ============================================================

def draw_hardware_test_view(frame, diag_state):
    """
    Hardware test UI — left DEVICES panel, right COMMANDS panel.
    """
    w, h = frame.shape[1], frame.shape[0]

    pyserial_ok   = diag_state.get("pyserial_installed", False)
    connected     = diag_state.get("connected",          False)
    port_name     = diag_state.get("port",               None)
    last_tx       = diag_state.get("last_tx",            None)
    last_rx       = diag_state.get("last_rx",            None)
    last_tx_age   = diag_state.get("last_tx_age_ms",     None)
    available     = diag_state.get("available_ports",    [])
    sel_idx       = diag_state.get("selected_port_index", 0)
    status_msg    = diag_state.get("status_message",     "")

    draw_top_bar(frame, "HARDWARE TEST",
                 "[ ] Port  |  ENTER Connect  |  R/P/S Send  |  X Disconnect  |  ESC Back")

    if not pyserial_ok:
        draw_centered_text_in_rect(frame, "pyserial NOT INSTALLED",
            (0, _ix(h*0.30), w, _ix(h*0.42)),
            base_scale=0.70, color=COL_RED, thickness=1, outline=2)
        draw_centered_text_in_rect(frame, "Run:  pip install pyserial",
            (0, _ix(h*0.44), w, _ix(h*0.54)),
            base_scale=0.46, color=COL_ACCENT, thickness=1, outline=2)
        draw_centered_text_in_rect(frame, "Then restart the app.",
            (0, _ix(h*0.55), w, _ix(h*0.63)),
            base_scale=0.36, color=COL_TEXT_DIM, thickness=1, outline=1)
        draw_bottom_bar(frame, "ESC Back")
        return

    top_y = _ix(h * 0.105)
    bot_y = _ix(h * 0.945)
    pad   = _ix(w * 0.01)

    # ── LEFT: Devices panel (x 3.6%, w 31%) ──────────────────────────────
    lx1 = _ix(w * 0.036);  lx2 = lx1 + _ix(w * 0.31)
    draw_panel(frame, lx1, top_y, lx2, bot_y,
               fill=COL_PANEL_BG, alpha=0.88, border=COL_BORDER_HAIR, border_thickness=1)
    lph = bot_y - top_y
    lpw = lx2 - lx1

    title_h = 32
    draw_outlined_text(frame, "DEVICES",
                       lx1 + _ix(lpw * 0.05), top_y + _ix(title_h * 0.72),
                       SCALE_MICRO, COL_TEXT_DIM, thickness=1, outline=2)
    cv2.line(frame, (lx1, top_y + title_h), (lx2, top_y + title_h), COL_BORDER_HAIR, 1)

    row_h  = 68
    row_y  = top_y + title_h
    if not available:
        draw_outlined_text(frame, "(no devices found)",
                           lx1 + _ix(lpw * 0.05), row_y + 28,
                           SCALE_CAPTION, COL_TEXT_DIM, thickness=1, outline=2)
    else:
        for i, port in enumerate(available):
            ry1 = row_y + i * row_h
            ry2 = ry1 + row_h
            if ry2 > bot_y:
                break
            is_sel = (i == sel_idx)
            if is_sel:
                roi = frame[ry1:ry2, lx1:lx2]
                if roi.size > 0:
                    ov = roi.copy()
                    cv2.rectangle(ov, (0, 0), (lx2 - lx1, row_h), COL_ACCENT, -1)
                    cv2.addWeighted(ov, 0.06, roi, 0.94, 0, roi)
                cv2.rectangle(frame, (lx1, ry1 + 2), (lx1 + 2, ry2 - 2), COL_ACCENT, -1)
            row_pad = _ix(lpw * 0.05)
            draw_outlined_text(frame, port,
                               lx1 + row_pad, ry1 + 26,
                               SCALE_BODY, COL_TEXT_PRIMARY, thickness=1, outline=2)
            is_connected = connected and port == port_name
            detail = "Active connection" if is_connected else "Available"
            draw_outlined_text(frame, detail,
                               lx1 + row_pad, ry1 + 48,
                               SCALE_CAPTION, COL_TEXT_DIM, thickness=1, outline=2)
            dot_col  = COL_GREEN if is_connected else COL_AMBER
            dot_lbl  = "CONNECTED" if is_connected else "IDLE"
            dot_x    = lx2 - _ix(lpw * 0.12)
            dot_y_c  = ry1 + 30
            cv2.circle(frame, (dot_x, dot_y_c), 4, dot_col, -1)
            draw_outlined_text(frame, dot_lbl, dot_x + 8, dot_y_c + 4,
                               SCALE_MICRO, dot_col, thickness=1, outline=2)
            if i > 0:
                cv2.line(frame, (lx1 + 3, ry1), (lx2 - 3, ry1), COL_BORDER_HAIR, 1)

    # ── RIGHT: Commands panel (x 36.4%, w 60%) ───────────────────────────
    rx1 = _ix(w * 0.364);  rx2 = rx1 + _ix(w * 0.60)
    draw_panel(frame, rx1, top_y, rx2, bot_y,
               fill=COL_PANEL_BG, alpha=0.88, border=COL_BORDER_HAIR, border_thickness=1)
    rph = bot_y - top_y
    rpw = rx2 - rx1

    draw_outlined_text(frame, "COMMANDS",
                       rx1 + _ix(rpw * 0.03), top_y + _ix(title_h * 0.72),
                       SCALE_MICRO, COL_TEXT_DIM, thickness=1, outline=2)
    cv2.line(frame, (rx1, top_y + title_h), (rx2, top_y + title_h), COL_BORDER_HAIR, 1)

    commands = [
        ("R", "ROCK"),
        ("P", "PAPER"),
        ("S", "SCISSORS"),
        ("O", "OPEN"),
        ("C", "CLOSE"),
        ("T", "PING"),
    ]
    n_cols   = 2
    tile_w   = rpw // n_cols
    tile_h   = _ix(rph * 0.12)
    tile_y0  = top_y + title_h + 4

    for i, (key_char, cmd) in enumerate(commands):
        col_i = i % n_cols
        row_i = i // n_cols
        tx1   = rx1 + col_i * tile_w + 4
        tx2   = tx1 + tile_w - 8
        ty1   = tile_y0 + row_i * tile_h + 2
        ty2   = ty1 + tile_h - 4
        active = last_tx == f"CMD|{cmd}"
        draw_panel(frame, tx1, ty1, tx2, ty2,
                   fill=COL_PANEL_BG, alpha=0.85,
                   border=COL_ACCENT if active else COL_BORDER_HAIR, border_thickness=1)
        kx1 = tx1 + 6;  kx2 = kx1 + _ix(tile_h * 0.60)
        draw_panel(frame, kx1, ty1 + 6, kx2, ty2 - 6,
                   fill=COL_PANEL_BG, alpha=0.92, border=COL_ACCENT, border_thickness=1)
        draw_centered_text_in_rect(frame, key_char,
            (kx1, ty1 + 6, kx2, ty2 - 6),
            base_scale=SCALE_CAPTION, color=COL_ACCENT, thickness=1, outline=2)
        draw_outlined_text(frame, f"CMD|{cmd}",
                           kx2 + 8, ty1 + _ix(tile_h * 0.60),
                           SCALE_CAPTION, COL_TEXT_PRIMARY if active else COL_TEXT_SECONDARY,
                           thickness=1, outline=2)
        if active:
            draw_outlined_text(frame, "SENT",
                               tx2 - _ix(rpw * 0.08), ty1 + _ix(tile_h * 0.60),
                               SCALE_MICRO, COL_GREEN, thickness=1, outline=2)

    # Footer log at panel inner-bottom
    log_y = bot_y - _ix(rph * 0.12)
    cv2.line(frame, (rx1 + 4, log_y), (rx2 - 4, log_y), COL_BORDER_HAIR, 1)
    if last_tx:
        age_str = f"{last_tx_age}ms ago" if last_tx_age else ""
        ack_str = "ACK received" if last_rx else ""
        parts   = [p for p in ["LAST  *  sent", last_tx, age_str, ack_str] if p]
        log_txt = "  *  ".join(parts)
        sc_log  = get_fit_scale(log_txt, rpw - 12, base_scale=SCALE_MICRO,
                                thickness=1, min_scale=0.24)
        draw_outlined_text(frame, log_txt, rx1 + 6, log_y + _ix(rph * 0.06),
                           sc_log, COL_TEXT_SECONDARY, thickness=1, outline=2)
    elif status_msg:
        draw_outlined_text(frame, status_msg, rx1 + 6, log_y + _ix(rph * 0.06),
                           SCALE_MICRO, COL_TEXT_DIM, thickness=1, outline=2)

    draw_bottom_bar(frame,
        "[ ] Cycle ports  *  ENTER Connect  *  R/P/S Send  *  T Ping  *  X Disconnect  *  ESC Back")


# ============================================================
# PLAYER FEEDBACK / NOTES SCREEN
# ============================================================

def draw_notes_screen(frame, text_buffer, submitted=False, saved_path="", return_screen="MENU"):
    """
    Full-screen note-taking screen.
    Player types a suggestion/feedback and presses ENTER to submit.
    """
    import time as _time
    import math as _math

    w, h = frame.shape[1], frame.shape[0]
    t    = _time.monotonic()

    draw_top_bar(frame, "PLAYER FEEDBACK",
                 "Type your suggestion and press ENTER  |  ESC Cancel")

    if submitted:
        # Confirmation screen
        pulse = 0.85 + 0.15 * abs(_math.sin(t * _math.pi * 1.5))
        col   = tuple(min(255, int(c * pulse)) for c in COL_GREEN)
        draw_centered_text_in_rect(frame, "FEEDBACK SUBMITTED",
            (0, _ix(h*0.28), w, _ix(h*0.42)),
            base_scale=0.80, color=col, thickness=1, outline=2)
        draw_centered_text_in_rect(frame,
            "Thank you! Your suggestion has been saved.",
            (0, _ix(h*0.45), w, _ix(h*0.53)),
            base_scale=0.38, color=COL_TEXT_ACCENT, thickness=1, outline=2)
        if saved_path:
            fname = saved_path.split("/")[-1].split("\\")[-1]
            draw_centered_text_in_rect(frame, f"Saved as: {fname}",
                (0, _ix(h*0.54), w, _ix(h*0.61)),
                base_scale=0.30, color=COL_TEXT_DIM, thickness=1, outline=1)
        draw_centered_text_in_rect(frame, "Press any key to return to menu",
            (0, _ix(h*0.65), w, _ix(h*0.73)),
            base_scale=0.38, color=COL_TEXT_DIM, thickness=1, outline=2)
        draw_bottom_bar(frame, "Any key  -  return to menu")
        return

    # Instruction
    draw_centered_text_in_rect(frame,
        "Share a suggestion, bug report, or idea for the game:",
        (0, _ix(h*0.10), w, _ix(h*0.18)),
        base_scale=0.40, color=COL_TEXT_ACCENT, thickness=1, outline=2)

    # Text input box
    box_x1 = _ix(w * 0.06)
    box_x2 = _ix(w * 0.94)
    box_y1 = _ix(h * 0.20)
    box_y2 = _ix(h * 0.72)
    draw_panel(frame, box_x1, box_y1, box_x2, box_y2,
               fill=(6, 10, 24), alpha=0.92,
               border=COL_BORDER_HAIR, border_thickness=1)

    # Word-wrap the text buffer
    max_chars_per_line = 72
    words   = text_buffer.replace("\n", " \n ").split(" ")
    lines   = []
    current = ""
    for word in words:
        if word == "\n":
            lines.append(current)
            current = ""
        elif len(current) + len(word) + 1 <= max_chars_per_line:
            current = (current + " " + word).strip()
        else:
            lines.append(current)
            current = word
    lines.append(current)

    # Draw lines
    line_h   = _ix(h * 0.048)
    text_x   = box_x1 + _ix(w * 0.025)
    text_y   = box_y1 + _ix(h * 0.03)
    max_lines = int((box_y2 - box_y1 - _ix(h*0.06)) / line_h)

    visible = lines[-max_lines:] if len(lines) > max_lines else lines
    for i, line in enumerate(visible):
        draw_outlined_text(frame, line, text_x, text_y + i * line_h,
                           0.36, COL_TEXT, thickness=1, outline=2)

    # Blinking cursor on last line
    if abs(_math.sin(t * _math.pi * 1.5)) > 0.5:
        last_line = visible[-1] if visible else ""
        tw, _     = cv2.getTextSize(last_line, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
        cur_x     = text_x + tw[0] + 3
        cur_y_top = text_y + (len(visible) - 1) * line_h - _ix(h * 0.02)
        cur_y_bot = cur_y_top + _ix(h * 0.035)
        cv2.line(frame, (cur_x, cur_y_top), (cur_x, cur_y_bot), COL_CYAN, 2)

    # Character count
    char_count = len(text_buffer)
    max_chars  = 500
    cc_col     = COL_TEXT_DIM if char_count < max_chars * 0.8 else COL_YELLOW
    draw_outlined_text(frame, f"{char_count}/{max_chars}",
                       box_x2 - _ix(w*0.10), box_y2 - _ix(h*0.015),
                       0.30, cc_col, thickness=1, outline=1)

    # Hints
    draw_centered_text_in_rect(frame,
        "ENTER submit  |  BACKSPACE delete  |  ESC cancel",
        (0, _ix(h*0.74), w, _ix(h*0.81)),
        base_scale=0.34, color=COL_TEXT_DIM, thickness=1, outline=1)

    draw_bottom_bar(frame, "ENTER  -  Submit feedback  |  ESC  -  Cancel")


# ============================================================
# PRIVACY CONSENT SCREEN
# ============================================================

def draw_consent_screen(frame, selected=0):
    """
    First-run privacy consent screen.
    selected: 0 = Accept, 1 = Decline
    """
    w, h = frame.shape[1], frame.shape[0]

    draw_top_bar(frame, "BEFORE YOU PLAY", "W/S Navigate  *  ENTER Select")

    # ── Content panel ────────────────────────────────────────────────────
    px1 = _ix(w * 0.10);  px2 = _ix(w * 0.90)
    py1 = _ix(h * 0.12);  py2 = _ix(h * 0.72)
    pw  = px2 - px1
    ph  = py2 - py1
    draw_panel(frame, px1, py1, px2, py2,
               fill=COL_PANEL_BG, alpha=0.88, border=COL_BORDER_HAIR, border_thickness=1)

    lx = px1 + _ix(pw * 0.08)
    rx = px1 + _ix(pw * 0.28)
    row_gap = _ix(ph * 0.22)
    data_rows = [
        ("CAMERA",    "We use your camera to detect hand gestures only."),
        ("HAND DATA", "Hand tracking is local and not stored or sent."),
        ("NAMES",     "Your player name is saved on this device only."),
        ("NO ACCOUNT","No login, no tracking, no cloud required."),
    ]
    for i, (tag, body) in enumerate(data_rows):
        ry = py1 + _ix(ph * 0.10) + i * row_gap
        draw_outlined_text(frame, tag, lx, ry,
                           SCALE_CAPTION, COL_ACCENT, thickness=1, outline=2)
        sc_b = get_fit_scale(body, px2 - rx - _ix(pw * 0.04),
                             base_scale=SCALE_BODY, thickness=1, min_scale=0.28)
        draw_outlined_text(frame, body, rx, ry,
                           sc_b, COL_TEXT_SECONDARY, thickness=1, outline=2)
        if i < len(data_rows) - 1:
            cv2.line(frame, (lx, ry + _ix(ph * 0.14)), (px2 - _ix(pw * 0.04), ry + _ix(ph * 0.14)),
                     COL_BORDER_HAIR, 1)

    # ── Button cards at y=80-93% ─────────────────────────────────────────
    btn_y1  = _ix(h * 0.80);  btn_y2 = _ix(h * 0.93)
    btn_h   = btn_y2 - btn_y1
    btn_gap = _ix(w * 0.02)
    ac_x1   = _ix(w * 0.10);  ac_x2 = w // 2 - btn_gap
    dc_x1   = w // 2 + btn_gap; dc_x2 = _ix(w * 0.90)

    # Accept card
    acc_sel = (selected == 0)
    draw_panel(frame, ac_x1, btn_y1, ac_x2, btn_y2,
               fill=COL_PANEL_BG, alpha=0.85,
               border=COL_ACCENT if acc_sel else COL_BORDER_HAIR, border_thickness=1)
    if acc_sel:
        cv2.rectangle(frame, (ac_x1, btn_y1 + 2), (ac_x1 + 2, btn_y2 - 2), COL_ACCENT, -1)
    draw_centered_text_in_rect(frame, "Accept",
        (ac_x1, btn_y1, ac_x2, btn_y1 + _ix(btn_h * 0.52)),
        base_scale=SCALE_HEADING,
        color=COL_TEXT_PRIMARY if acc_sel else COL_TEXT_SECONDARY,
        thickness=1, outline=2)
    draw_centered_text_in_rect(frame, "I agree to camera use",
        (ac_x1, btn_y1 + _ix(btn_h * 0.55), ac_x2, btn_y2 - _ix(btn_h * 0.08)),
        base_scale=SCALE_CAPTION,
        color=COL_TEXT_DIM if acc_sel else COL_TEXT_DIM,
        thickness=1, outline=2)
    if acc_sel:
        hint_str = "ENTER CONFIRM"
        (hw, _), _ = cv2.getTextSize(hint_str, cv2.FONT_HERSHEY_SIMPLEX, SCALE_MICRO, 1)
        draw_outlined_text(frame, hint_str, ac_x2 - hw - _ix(pw * 0.04),
                           btn_y2 - _ix(btn_h * 0.14),
                           SCALE_MICRO, COL_ACCENT, thickness=1, outline=2)

    # No Thanks card
    dec_sel = (selected == 1)
    draw_panel(frame, dc_x1, btn_y1, dc_x2, btn_y2,
               fill=COL_PANEL_BG, alpha=0.85,
               border=COL_BORDER_HAIR, border_thickness=1)
    if dec_sel:
        cv2.rectangle(frame, (dc_x1, btn_y1 + 2), (dc_x1 + 2, btn_y2 - 2),
                      COL_TEXT_PRIMARY, -1)
    draw_centered_text_in_rect(frame, "No thanks",
        (dc_x1, btn_y1, dc_x2, btn_y1 + _ix(btn_h * 0.52)),
        base_scale=SCALE_HEADING,
        color=COL_TEXT_PRIMARY if dec_sel else COL_TEXT_SECONDARY,
        thickness=1, outline=2)
    draw_centered_text_in_rect(frame, "Local data only",
        (dc_x1, btn_y1 + _ix(btn_h * 0.55), dc_x2, btn_y2 - _ix(btn_h * 0.08)),
        base_scale=SCALE_CAPTION, color=COL_TEXT_DIM, thickness=1, outline=2)

    draw_bottom_bar(frame, "W/S Navigate  *  ENTER Select  *  ESC Back")


# ============================================================
# GESTURE CALIBRATION SCREEN
# ============================================================

def draw_calibration_view(frame, cal_state, hand_state=None):
    """
    Guided gesture calibration screen for new players.
    Walks through Rock, Paper, Scissors collection then auto-trains.
    """
    import math as _math
    import time as _time

    w, h = frame.shape[1], frame.shape[0]
    t    = _time.monotonic()

    phase         = cal_state.get("phase",          "INTRO")
    gesture       = cal_state.get("gesture",        "Rock")
    gesture_idx   = cal_state.get("gesture_idx",    0)
    gesture_count = cal_state.get("gesture_count",  3)
    samples_this  = cal_state.get("samples_this",   0)
    samples_need  = cal_state.get("samples_needed", 15)
    counts        = cal_state.get("counts",         {})
    instruction   = cal_state.get("instruction",    "")
    status_msg    = cal_state.get("status_msg",     "")
    variation     = cal_state.get("variation_hint", "")
    hand_visible  = cal_state.get("hand_visible",   False)
    accuracy      = cal_state.get("training_result",None)

    g_idx_label = f"{gesture_idx + 1} OF {gesture_count}"
    var_label   = variation if variation else ""
    top_right   = f"VARIATION {var_label}  |  ESC to skip" if var_label else "ESC to skip"
    draw_top_bar(frame, f"CALIBRATION  *  {g_idx_label}", top_right)

    # ── INTRO ────────────────────────────────────────────────────────────
    if phase == "INTRO":
        draw_centered_text_in_rect(frame, "QUICK SETUP REQUIRED",
            (0, _ix(h*0.10), w, _ix(h*0.20)),
            base_scale=0.65, color=COL_CYAN, thickness=1, outline=2)

        px1, px2 = _ix(w*0.08), _ix(w*0.92)
        py1, py2 = _ix(h*0.22), _ix(h*0.76)
        draw_panel(frame, px1, py1, px2, py2,
                   fill=(6,10,24), alpha=0.92,
                   border=(50,70,100), border_thickness=1)

        lines = [
            (0.08, "Your device needs to learn what your hand gestures look like.",
                   COL_TEXT_ACCENT, 0.36),
            (0.18, "This takes about 1 minute and only happens once.", COL_TEXT_DIM, 0.32),
            (0.30, "You will be asked to show:", COL_TEXT_ACCENT, 0.36),
            (0.40, "  Rock  -  20 samples", (100, 200, 120), 0.34),
            (0.48, "  Paper  -  20 samples", (100, 200, 120), 0.34),
            (0.56, "  Scissors  -  20 samples", (100, 200, 120), 0.34),
            (0.68, "Press SPACE each time to capture a frame.", COL_TEXT_ACCENT, 0.34),
            (0.78, "Make sure your hand is clearly visible.", COL_TEXT_DIM, 0.30),
        ]
        ph = py2 - py1
        for frac, text, col, scale in lines:
            ty = py1 + int(ph * frac)
            draw_outlined_text(frame, text, px1 + _ix(w*0.03), ty,
                               scale, col, thickness=1, outline=2)

        pulse = 0.6 + 0.4 * abs(_math.sin(t * _math.pi * 1.2))
        pc = tuple(min(255, int(c * pulse)) for c in COL_GREEN)
        draw_centered_text_in_rect(frame, "Press SPACE or ENTER to begin  |  ESC to skip for now",
            (0, _ix(h*0.80), w, _ix(h*0.90)),
            base_scale=0.42, color=pc, thickness=2, outline=3)
        draw_bottom_bar(frame, "SPACE / ENTER  -  begin  |  ESC  -  skip for now")

    # ── COLLECTING ───────────────────────────────────────────────────────
    elif phase == "COLLECTING":
        pct_done = min(1.0, samples_this / max(samples_need, 1))
        gesture_names_all = ["Rock", "Paper", "Scissors"]

        # Per-gesture progress strip at y=12%
        card_w  = _ix(w * 0.30)
        card_h  = 80
        card_y1 = _ix(h * 0.12)
        card_y2 = card_y1 + card_h
        for i, gn in enumerate(gesture_names_all):
            cx1 = _ix(w * (0.03 + i * 0.32))
            cx2 = cx1 + card_w
            done    = i < gesture_idx
            current = i == gesture_idx
            if done:
                card_border = COL_GREEN
                label_col   = COL_GREEN
            elif current:
                card_border = COL_ACCENT
                label_col   = COL_ACCENT
            else:
                card_border = COL_BORDER_HAIR
                label_col   = COL_TEXT_DIM
            draw_panel(frame, cx1, card_y1, cx2, card_y2,
                       fill=COL_PANEL_BG, alpha=0.85 if not current else 0.92,
                       border=card_border, border_thickness=1)
            if current:
                roi = frame[card_y1:card_y2, cx1:cx2]
                if roi.size > 0:
                    ov = roi.copy()
                    cv2.rectangle(ov, (0, 0), (cx2 - cx1, card_h), COL_ACCENT, -1)
                    cv2.addWeighted(ov, 0.08, roi, 0.92, 0, roi)
            glyph_sz = 24
            gcx = cx1 + card_w // 2
            gcy = card_y1 + 28
            draw_gesture_glyph(frame, gn,
                               (gcx - glyph_sz, gcy - glyph_sz,
                                gcx + glyph_sz, gcy + glyph_sz),
                               label_col)
            draw_centered_text_in_rect(frame, gn,
                (cx1, gcy + glyph_sz + 2, cx2, gcy + glyph_sz + 18),
                base_scale=SCALE_CAPTION, color=label_col, thickness=1, outline=2)
            if done:
                status_cap = "Done"
            elif current:
                status_cap = f"{samples_this} / {samples_need}"
            else:
                status_cap = "Waiting"
            draw_centered_text_in_rect(frame, status_cap,
                (cx1, gcy + glyph_sz + 18, cx2, card_y2 - 6),
                base_scale=SCALE_MICRO, color=label_col, thickness=1, outline=2)
            if current:
                bar_y = card_y2 - 5
                cv2.rectangle(frame, (cx1 + 2, bar_y), (cx2 - 2, bar_y + 3), (28, 28, 28), -1)
                fx = cx1 + 2 + _ix((cx2 - cx1 - 4) * pct_done)
                if fx > cx1 + 2:
                    cv2.rectangle(frame, (cx1 + 2, bar_y), (fx, bar_y + 3), COL_ACCENT, -1)

        # Centre block at y=38%
        draw_outlined_text(frame, "HOLD STILL",
                           _ix(w * 0.05), _ix(h * 0.38),
                           SCALE_MICRO, COL_TEXT_DIM, thickness=1, outline=2)
        font_d = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, gesture, (_ix(w * 0.05), _ix(h * 0.48)),
                    font_d, SCALE_DISPLAY_L, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, gesture, (_ix(w * 0.05), _ix(h * 0.48)),
                    font_d, SCALE_DISPLAY_L, COL_TEXT_PRIMARY, 2, cv2.LINE_AA)
        sc_instr = get_fit_scale(instruction, _ix(w * 0.85),
                                 base_scale=SCALE_BODY, thickness=1, min_scale=0.26)
        draw_outlined_text(frame, instruction,
                           _ix(w * 0.05), _ix(h * 0.58),
                           sc_instr, COL_TEXT_SECONDARY, thickness=1, outline=2)

        # Footer at y=83%
        if hand_state is not None:
            detected_g = getattr(hand_state, 'gesture', '') or ''
            g_conf = getattr(hand_state, 'confidence', 0.0) or 0.0
        else:
            detected_g = gesture if hand_visible else "Unknown"
            g_conf = 0.85 if hand_visible else 0.0
        draw_gesture_badge(frame, detected_g, g_conf, _ix(w * 0.04), _ix(h * 0.83))

        pill_lbl = "SPACE TO CAPTURE"
        (plw, plh), _ = cv2.getTextSize(pill_lbl, cv2.FONT_HERSHEY_SIMPLEX, SCALE_MICRO, 1)
        pill_pad_x = 14;  pill_pad_y = 4
        pill_x2 = _ix(w * 0.96)
        pill_x1 = pill_x2 - plw - pill_pad_x * 2
        pill_y1 = _ix(h * 0.83)
        pill_y2 = pill_y1 + plh + pill_pad_y * 2
        draw_panel(frame, pill_x1, pill_y1, pill_x2, pill_y2,
                   fill=COL_PANEL_BG, alpha=0.85, border=COL_ACCENT, border_thickness=1)
        draw_outlined_text(frame, pill_lbl, pill_x1 + pill_pad_x, pill_y2 - pill_pad_y,
                           SCALE_MICRO, COL_ACCENT, thickness=1, outline=2)

        draw_bottom_bar(frame,
            f"Capturing: {gesture}  *  SPACE capture  *  {samples_this}/{samples_need} done")

    # ── TRAINING ─────────────────────────────────────────────────────────
    elif phase == "TRAINING":
        draw_centered_text_in_rect(frame, "TRAINING MODEL...",
            (0, _ix(h*0.35), w, _ix(h*0.50)),
            base_scale=0.70, color=COL_YELLOW, thickness=1, outline=2)
        draw_centered_text_in_rect(frame,
            "Please wait  -  this takes a few seconds",
            (0, _ix(h*0.52), w, _ix(h*0.60)),
            base_scale=0.36, color=COL_TEXT_DIM, thickness=1, outline=2)

    # ── DONE ─────────────────────────────────────────────────────────────
    elif phase == "DONE":
        pulse4 = 0.85 + 0.15 * abs(_math.sin(t * _math.pi * 1.5))
        gc = tuple(min(255, int(c * pulse4)) for c in COL_GREEN)
        draw_centered_text_in_rect(frame, "CALIBRATION COMPLETE",
            (0, _ix(h*0.18), w, _ix(h*0.30)),
            base_scale=0.70, color=gc, thickness=1, outline=2)
        if accuracy is not None:
            draw_centered_text_in_rect(frame,
                f"Model accuracy: {accuracy:.0%}",
                (0, _ix(h*0.32), w, _ix(h*0.42)),
                base_scale=0.50, color=gc, thickness=2, outline=3)
        draw_centered_text_in_rect(frame,
            "Your gestures have been learned.",
            (0, _ix(h*0.44), w, _ix(h*0.52)),
            base_scale=0.36, color=COL_TEXT_ACCENT, thickness=1, outline=2)
        draw_centered_text_in_rect(frame,
            "You can recalibrate any time from Settings.",
            (0, _ix(h*0.53), w, _ix(h*0.60)),
            base_scale=0.30, color=COL_TEXT_DIM, thickness=1, outline=1)
        pc5 = tuple(min(255, int(c * pulse4)) for c in COL_GREEN)
        draw_centered_text_in_rect(frame, "Press SPACE or ENTER to start playing",
            (0, _ix(h*0.68), w, _ix(h*0.78)),
            base_scale=0.48, color=pc5, thickness=2, outline=3)
        draw_bottom_bar(frame, "SPACE / ENTER  -  start playing")

    # ── FAILED ───────────────────────────────────────────────────────────
    elif phase == "FAILED":
        draw_centered_text_in_rect(frame, "TRAINING FAILED",
            (0, _ix(h*0.25), w, _ix(h*0.38)),
            base_scale=0.70, color=(220,80,80), thickness=1, outline=2)
        draw_centered_text_in_rect(frame,
            "Not enough samples were collected.",
            (0, _ix(h*0.40), w, _ix(h*0.48)),
            base_scale=0.36, color=(180,120,120), thickness=1, outline=2)
        draw_centered_text_in_rect(frame,
            "Press ENTER to try again.",
            (0, _ix(h*0.56), w, _ix(h*0.64)),
            base_scale=0.44, color=COL_TEXT_DIM, thickness=1, outline=2)
        draw_bottom_bar(frame, "ENTER  -  try again")

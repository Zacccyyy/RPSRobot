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
                     update_label=""):
    layout = _menu_layout(frame)
    w, h = layout["w"], layout["h"]
    x1, y1, x2, y2 = layout["panel"]

    draw_panel(frame, 0, 0, w - 1, h - 1, fill=COL_BG_DARK, alpha=0.75,
               border=COL_BG_DARK, border_thickness=0)

    top_right = "UP/DOWN Navigate | Enter Select | N Feedback | ESC Back | Q Quit"
    draw_top_bar(frame, "RPS ROBOT", top_right)

    # Update banner — pulsing yellow strip above the menu panel
    if update_label:
        import time as _t, math as _m
        pulse  = 0.65 + 0.35 * abs(_m.sin(_t.monotonic() * _m.pi * 1.2))
        bc     = tuple(min(255, int(c * pulse)) for c in COL_YELLOW)
        ban_y1 = y1 - _ix(h * 0.060)
        ban_y2 = y1 - _ix(h * 0.008)
        draw_panel(frame, _ix(w*0.03), ban_y1, _ix(w*0.97), ban_y2,
                   fill=(18, 15, 0), alpha=0.90, border=bc, border_thickness=2)
        draw_centered_text_in_rect(frame, update_label,
            (_ix(w*0.04), ban_y1, _ix(w*0.96), ban_y2),
            base_scale=0.36, color=bc, thickness=1, outline=2)

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
      progress:      float 0-1
      progress_text: str  e.g. "random vs fair_play  4/10"
      results:       dict from run_simulation() or None
      error:         str or None
    """
    layout = _menu_layout(frame)
    w, h = layout["w"], layout["h"]
    x1, y1, x2, y2 = layout["panel"]

    draw_panel(frame, 0, 0, w - 1, h - 1, fill=COL_BG_DARK, alpha=0.75,
               border=COL_BG_DARK, border_thickness=0)
    draw_top_bar(frame, "SIMULATION", "Running...  ESC to go back when done")
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

    draw_panel(frame, 0, 0, w - 1, h - 1, fill=COL_BG_DARK, alpha=0.75,
               border=COL_BG_DARK, border_thickness=0)

    if text_edit:
        hint = "Type name | ENTER confirm | ESC cancel"
    elif nav_active:
        hint = "Move left [-]  |  center = select  |  [+] move right"
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

    # Voice model download hint - shown when voice_model item is selected
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
    Optional feature toggles - separate from program settings.
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
           "Move left [-]  |  center = select  |  [+] move right"
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
                    txt_col = (10, 10, 10) if is_on else (140, 140, 60)
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
    layout = _menu_layout(frame)
    w, h   = layout["w"], layout["h"]
    px1, py1, px2, py2 = layout["panel"]
    ph, pw = py2 - py1, px2 - px1

    draw_panel(frame, 0, 0, w - 1, h - 1, fill=COL_BG_DARK, alpha=0.75,
               border=COL_BG_DARK, border_thickness=0)

    hint = ("W/S Navigate  |  Enter Open  |  ESC Back" if not in_mode_list
            else "W/S Select Mode  |  Enter Start  |  ESC Back")
    draw_top_bar(frame, "GAME MODES", hint)

    # -- Left panel: categories --------------------------------------------
    lx1 = px1
    lx2 = px1 + _ix(pw * 0.42)
    draw_panel(frame, lx1, py1, lx2, py2, fill=COL_BG_PANEL, alpha=0.94,
               border=COL_CYAN, border_thickness=2)

    draw_centered_text_in_rect(frame, "CATEGORIES",
        (lx1, py1, lx2, py1 + _ix(ph * 0.10)),
        base_scale=0.52, color=COL_CYAN, thickness=2, outline=3)

    n_cats       = len(categories)
    VISIBLE_CATS = 6   # how many rows visible at once  -  matches original sizing
    cat_area_top = py1 + _ix(ph * 0.14)
    item_h       = _ix(ph * 0.78 / VISIBLE_CATS)   # fixed height regardless of total count
    bar_half     = min(_ix(h * 0.016), item_h // 2 - 3)

    # Scroll offset: keep selected item visible
    scroll_off = max(0, min(category_index - VISIBLE_CATS + 1,
                            n_cats - VISIBLE_CATS))
    scroll_off = max(0, scroll_off)

    for i, cat in enumerate(categories):
        vis_idx = i - scroll_off
        if vis_idx < 0 or vis_idx >= VISIBLE_CATS:
            continue
        cy      = cat_area_top + vis_idx * item_h + item_h // 2
        sel_cat = (i == category_index)
        fill    = (12, 30, 40) if sel_cat else COL_BG_PANEL
        border  = COL_CYAN if sel_cat else (30, 50, 60)
        draw_panel(frame, lx1 + _ix(w * 0.008), cy - bar_half,
                   lx2 - _ix(w * 0.008), cy + bar_half,
                   fill=fill, alpha=0.80, border=border, border_thickness=2 if sel_cat else 1)
        col    = COL_CYAN if sel_cat else COL_TEXT_DIM
        prefix = "> " if sel_cat else "  "
        draw_centered_text_in_rect(frame, f"{prefix}{cat['label']}",
            (lx1 + _ix(w * 0.012), cy - bar_half, lx2 - _ix(w * 0.012), cy + bar_half),
            base_scale=0.48, color=col, thickness=2, outline=2)

    # Scroll indicators
    if scroll_off > 0:
        draw_centered_text(frame, "^  more",
                           cat_area_top - _ix(h * 0.015), 0.28, COL_TEXT_DIM,
                           thickness=1, outline=1)
    if scroll_off + VISIBLE_CATS < n_cats:
        draw_centered_text(frame, "v  more",
                           cat_area_top + VISIBLE_CATS * item_h + _ix(h * 0.010),
                           0.28, COL_TEXT_DIM, thickness=1, outline=1)

    # -- Right panel: description OR mode list ----------------------------
    rx1 = lx2 + _ix(w * 0.012)
    rx2 = px2
    rpw = rx2 - rx1
    draw_panel(frame, rx1, py1, rx2, py2, fill=(8, 8, 22), alpha=0.94,
               border=COL_MAGENTA, border_thickness=2)

    sel_cat = categories[category_index]
    pad_x   = _ix(rpw * 0.07)

    if not in_mode_list:
        # -- Description panel --------------------------------------------
        title_y2 = py1 + _ix(ph * 0.13)
        draw_centered_text_in_rect(frame, sel_cat["label"],
            (rx1, py1, rx2, title_y2),
            base_scale=0.58, color=COL_MAGENTA, thickness=2, outline=3)

        sep_y = py1 + _ix(ph * 0.15)
        cv2.line(frame,
                 (rx1 + _ix(rpw * 0.05), sep_y),
                 (rx2 - _ix(rpw * 0.05), sep_y),
                 COL_MAGENTA, 1)

        # Word-wrap description -- break on "--" into separate lines too
        raw_desc  = sel_cat.get("desc", "")
        # Split on -- to create natural line breaks
        parts     = [p.strip() for p in raw_desc.replace("--", "\n").split("\n") if p.strip()]
        font_sc   = 0.50
        max_px    = rpw - 2 * pad_x
        lines     = []
        for part in parts:
            words = part.split()
            cur   = ""
            for word in words:
                test = f"{cur} {word}".strip()
                (tw, _), _ = cv2.getTextSize(test, cv2.FONT_HERSHEY_SIMPLEX, font_sc, 1)
                if tw <= max_px:
                    cur = test
                else:
                    if cur:
                        lines.append(cur)
                    cur = word
            if cur:
                lines.append(cur)
            lines.append("")   # blank line between parts

        # Remove trailing blank
        while lines and lines[-1] == "":
            lines.pop()

        dy   = py1 + _ix(ph * 0.20)
        dgap = _ix(ph * 0.095)
        for line in lines[:7]:
            if line == "":
                dy += _ix(ph * 0.030)   # extra gap between parts
                continue
            draw_outlined_text(frame, line, rx1 + pad_x, dy,
                               font_sc, COL_TEXT, thickness=1, outline=2)
            dy += dgap

        # Modes preview list - max 3 items to avoid overlap with hint
        sep2_y = py1 + _ix(ph * 0.60)
        cv2.line(frame,
                 (rx1 + _ix(rpw * 0.05), sep2_y),
                 (rx2 - _ix(rpw * 0.05), sep2_y),
                 (50, 30, 70), 1)
        draw_outlined_text(frame, "Modes:", rx1 + pad_x, sep2_y + _ix(ph * 0.05),
                           0.44, COL_TEXT_DIM, thickness=1, outline=1)
        my = sep2_y + _ix(ph * 0.12)
        for (ml, _) in sel_cat["modes"][:3]:
            draw_outlined_text(frame, f"  {ml}", rx1 + pad_x, my,
                               0.42, (130, 130, 180), thickness=1, outline=1)
            my += _ix(ph * 0.075)

        # Hint pinned to absolute panel bottom
        draw_outlined_text(frame, "Enter to open",
                           rx1 + pad_x, py2 - _ix(ph * 0.035),
                           0.40, COL_CYAN, thickness=1, outline=2)

    else:
        # -- Mode list ----------------------------------------------------
        draw_centered_text_in_rect(frame, "SELECT MODE",
            (rx1, py1, rx2, py1 + _ix(ph * 0.10)),
            base_scale=0.54, color=COL_MAGENTA, thickness=2, outline=3)

        modes    = sel_cat["modes"]
        n_modes  = len(modes)
        VISIBLE_MODES = 5   # always show at most 5 at full size
        m_area   = py1 + _ix(ph * 0.14)
        m_gap    = _ix(ph * 0.74 / VISIBLE_MODES)   # fixed height for 5 items
        m_half   = min(_ix(h * 0.032), m_gap // 2 - 4)

        # Scroll to keep selected mode visible
        m_scroll = max(0, min(mode_index - VISIBLE_MODES + 1,
                              n_modes - VISIBLE_MODES))
        m_scroll = max(0, m_scroll)

        for j, (ml, _) in enumerate(modes):
            vis_j = j - m_scroll
            if vis_j < 0 or vis_j >= VISIBLE_MODES:
                continue
            my    = m_area + vis_j * m_gap + m_gap // 2
            sel_m = (j == mode_index)
            fill  = (20, 8, 30) if sel_m else (8, 8, 22)
            border = COL_MAGENTA if sel_m else (40, 20, 50)
            draw_panel(frame, rx1 + _ix(w * 0.008), my - m_half,
                       rx2 - _ix(w * 0.008), my + m_half,
                       fill=fill, alpha=0.85, border=border,
                       border_thickness=2 if sel_m else 1)
            col    = COL_MAGENTA if sel_m else COL_TEXT
            prefix = "> " if sel_m else "  "
            draw_centered_text_in_rect(frame, f"{prefix}{ml}",
                (rx1 + _ix(w * 0.012), my - m_half, rx2 - _ix(w * 0.012), my + m_half),
                base_scale=0.52, color=col, thickness=2, outline=2)

        if m_scroll + VISIBLE_MODES < n_modes:
            draw_centered_text(frame, "v  more",
                               m_area + VISIBLE_MODES * m_gap + _ix(h * 0.010),
                               0.28, COL_TEXT_DIM, thickness=1, outline=1)

        draw_outlined_text(frame, "ESC to go back",
                           rx1 + pad_x, py2 - _ix(ph * 0.035),
                           0.40, COL_TEXT_DIM, thickness=1, outline=2)

    draw_bottom_bar(frame,
        "W/S Navigate  |  Enter Select  |  ESC Back  |  Q Quit")


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

    draw_panel(frame, 0, 0, w - 1, h - 1, fill=COL_BG_DARK, alpha=0.75,
               border=COL_BG_DARK, border_thickness=0)
    draw_top_bar(frame, "SIMULATIONS", "W/S Select  |  Enter Run  |  ESC Back")
    draw_panel(frame, px1, py1, px2, py2, fill=COL_BG_PANEL, alpha=0.94,
               border=COL_YELLOW, border_thickness=2)

    # Title and subtitle — bounded to the outer panel
    draw_centered_text_in_rect(frame, "SIMULATION LAB",
        (px1, py1, px2, py1 + _ix(ph * 0.10)),
        base_scale=0.80, color=COL_YELLOW, thickness=3, outline=4)
    draw_centered_text_in_rect(frame, "Run high-fidelity RPS strategy simulations",
        (px1, py1 + _ix(ph * 0.10), px2, py1 + _ix(ph * 0.18)),
        base_scale=0.38, color=COL_TEXT_DIM, thickness=1, outline=2)

    cv2.line(frame,
             (px1 + _ix(pw * 0.05), py1 + _ix(ph * 0.19)),
             (px2 - _ix(pw * 0.05), py1 + _ix(ph * 0.19)),
             COL_YELLOW, 1)

    # Left: sim list
    lx1 = px1 + _ix(pw * 0.02)
    lx2 = px1 + _ix(pw * 0.42)
    for i, entry in enumerate(SIM_ENTRIES):
        bar_y    = py1 + _ix(ph * 0.24) + i * _ix(ph * 0.18)
        bar_half = _ix(ph * 0.07)
        sel      = (i == selected_index)
        fill     = (16, 30, 12) if sel else COL_BG_PANEL
        border   = entry["color"] if sel else (40, 50, 40)
        draw_panel(frame, lx1, bar_y - bar_half, lx2, bar_y + bar_half,
                   fill=fill, alpha=0.88, border=border, border_thickness=2 if sel else 1)
        col    = entry["color"] if sel else COL_TEXT_DIM
        prefix = "> " if sel else "  "
        draw_centered_text_in_rect(frame, f"{prefix}{entry['label']}",
            (lx1, bar_y - bar_half, lx2, bar_y + bar_half),
            base_scale=0.48, color=col, thickness=2, outline=2)

    # Right: description panel
    rx1 = px1 + _ix(pw * 0.46)
    rx2 = px2 - _ix(pw * 0.02)
    ry1 = py1 + _ix(ph * 0.20)
    ry2 = py2 - _ix(ph * 0.08)
    rw  = rx2 - rx1
    rh  = ry2 - ry1
    sel_entry = SIM_ENTRIES[selected_index]
    draw_panel(frame, rx1, ry1, rx2, ry2, fill=(8, 16, 8), alpha=0.92,
               border=sel_entry["color"], border_thickness=2)

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

    # -- Mode filter strip - ALWAYS drawn so user can still navigate ------
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

    # -- Tab strip - ALWAYS drawn -----------------------------------------
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

    # ====================================================================
    # HISTORY TAB
    # ====================================================================
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
        # Voice mode status panel - show listening indicator and word heard
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
        # Physical mode - original status panel logic
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
        else "Your camera feed is live - try the gestures!  |  Say BACK to exit"
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
    Login screen  -  shown on first launch or when switching player.

    - If saved_name is set: shows "Continue as <name>" option
    - If verified_players exist: shows "Login via Fingerprint" option
    - Always shows text input for typing a new name
    """
    import math
    import time as _time

    w, h  = frame.shape[1], frame.shape[0]
    t     = _time.monotonic()
    cx, cy = w // 2, h // 2

    verified = verified_players or []

    # Dark background
    draw_panel(frame, 0, 0, w - 1, h - 1, fill=(4, 6, 14), alpha=0.92,
               border=(4, 6, 14), border_thickness=0)

    draw_top_bar(frame, "WELCOME", "Type your name and press ENTER")

    # Main panel
    px1, px2 = _ix(w * 0.20), _ix(w * 0.80)
    py1, py2 = _ix(h * 0.12), _ix(h * 0.88)
    pw, ph   = px2 - px1, py2 - py1
    draw_panel(frame, px1, py1, px2, py2,
               fill=(6, 10, 22), alpha=0.96,
               border=(60, 100, 140), border_thickness=2)

    # Title
    draw_centered_text_in_rect(frame, "RPS Gesture Recogniser",
        (px1, py1, px2, py1 + _ix(ph * 0.10)),
        base_scale=0.52, color=(180, 190, 200), thickness=1, outline=2)

    cv2.line(frame, (px1 + _ix(pw * 0.05), py1 + _ix(ph * 0.11)),
             (px2 - _ix(pw * 0.05), py1 + _ix(ph * 0.11)), (40, 55, 75), 1)

    section_y = py1 + _ix(ph * 0.14)

    # ── Continue as saved player ────────────────────────────────────────
    if saved_name:
        draw_panel(frame, px1 + _ix(pw * 0.05), section_y,
                   px2 - _ix(pw * 0.05), section_y + _ix(ph * 0.12),
                   fill=(8, 24, 12), alpha=0.90,
                   border=(60, 140, 80), border_thickness=2)
        draw_centered_text_in_rect(frame,
            f"Continue as  {saved_name}",
            (px1 + _ix(pw * 0.05), section_y,
             px2 - _ix(pw * 0.05), section_y + _ix(ph * 0.06)),
            base_scale=0.46, color=(160, 220, 160), thickness=1, outline=2)
        draw_centered_text_in_rect(frame,
            "Press ENTER with empty box to continue",
            (px1 + _ix(pw * 0.05), section_y + _ix(ph * 0.06),
             px2 - _ix(pw * 0.05), section_y + _ix(ph * 0.12)),
            base_scale=0.28, color=(100, 130, 100), thickness=1, outline=1)
        section_y += _ix(ph * 0.16)

    # ── Name input box ───────────────────────────────────────────────────
    draw_outlined_text(frame, "Enter your name:",
        px1 + _ix(pw * 0.07), section_y + _ix(ph * 0.02),
        0.34, (160, 170, 185), thickness=1, outline=1)
    section_y += _ix(ph * 0.07)

    box_x1 = px1 + _ix(pw * 0.07)
    box_x2 = px2 - _ix(pw * 0.07)
    box_y1 = section_y
    box_y2 = section_y + _ix(ph * 0.11)
    draw_panel(frame, box_x1, box_y1, box_x2, box_y2,
               fill=(10, 14, 28), alpha=0.95,
               border=(80, 120, 180), border_thickness=2)

    # Cursor blink
    cursor = "|" if int(t * 2) % 2 == 0 else ""
    display_text = (login_text or "") + cursor
    draw_centered_text_in_rect(frame, display_text or cursor,
        (box_x1, box_y1, box_x2, box_y2),
        base_scale=0.52, color=(220, 230, 255), thickness=1, outline=2)
    section_y += _ix(ph * 0.15)

    # ── Fingerprint login button ─────────────────────────────────────────
    if verified:
        cv2.line(frame,
                 (px1 + _ix(pw * 0.05), section_y),
                 (px2 - _ix(pw * 0.05), section_y),
                 (40, 55, 75), 1)
        section_y += _ix(ph * 0.04)

        draw_panel(frame, px1 + _ix(pw * 0.05), section_y,
                   px2 - _ix(pw * 0.05), section_y + _ix(ph * 0.12),
                   fill=(8, 16, 28), alpha=0.90,
                   border=(60, 100, 180), border_thickness=2)
        pulse = 0.7 + 0.3 * abs(math.sin(t * math.pi * 0.8))
        fp_col = tuple(min(255, int(c * pulse)) for c in (120, 180, 255))
        draw_centered_text_in_rect(frame, "Login via Fingerprint  (TAB)",
            (px1 + _ix(pw * 0.05), section_y,
             px2 - _ix(pw * 0.05), section_y + _ix(ph * 0.07)),
            base_scale=0.40, color=fp_col, thickness=1, outline=2)
        names_str = "  ".join(verified[:3])
        draw_centered_text_in_rect(frame,
            f"Verified: {names_str}",
            (px1 + _ix(pw * 0.05), section_y + _ix(ph * 0.07),
             px2 - _ix(pw * 0.05), section_y + _ix(ph * 0.12)),
            base_scale=0.26, color=(80, 110, 150), thickness=1, outline=1)
        section_y += _ix(ph * 0.16)

    # ── Helper tip ───────────────────────────────────────────────────────
    draw_centered_text_in_rect(frame,
        "New name = new profile created automatically",
        (px1, py2 - _ix(ph * 0.07), px2, py2),
        base_scale=0.28, color=(70, 85, 100), thickness=1, outline=1)

    draw_bottom_bar(frame, "Type name  |  ENTER confirm  |  TAB fingerprint login")


# ============================================================
# HARDWARE TEST VIEW
# ============================================================

def draw_hardware_test_view(frame, diag_state):
    """
    Dedicated full-screen hardware test UI for ESP32 serial communication.
    Shows port selection, connection status, command log, and key hints.
    """
    import time as _time
    import cv2 as _cv2

    w, h = frame.shape[1], frame.shape[0]

    pyserial_ok    = diag_state.get("pyserial_installed", False)
    connected      = diag_state.get("connected",          False)
    port_name      = diag_state.get("port",               None)
    last_tx        = diag_state.get("last_tx",            None)
    last_rx        = diag_state.get("last_rx",            None)
    available      = diag_state.get("available_ports",    [])
    selected_port  = diag_state.get("selected_port",      None)
    sel_idx        = diag_state.get("selected_port_index", 0)
    status_msg     = diag_state.get("status_message",     "")

    # Dark background
    draw_panel(frame, 0, 0, w - 1, h - 1, fill=(4, 6, 16), alpha=0.82,
               border=(30, 30, 50), border_thickness=0)

    draw_top_bar(frame, "HARDWARE TEST  -  ESP32 SERIAL",
                 "[ ] Select port  |  ENTER Connect  |  R/P/S Send  |  X Disconnect  |  ESC Back")

    # ── pyserial warning ─────────────────────────────────────────────────
    if not pyserial_ok:
        draw_centered_text_in_rect(frame, "pyserial NOT INSTALLED",
            (0, _ix(h*0.30), w, _ix(h*0.42)),
            base_scale=0.70, color=(220, 80, 80), thickness=2, outline=4)
        draw_centered_text_in_rect(frame,
            "Run:  pip install pyserial",
            (0, _ix(h*0.44), w, _ix(h*0.54)),
            base_scale=0.46, color=COL_YELLOW, thickness=1, outline=2)
        draw_centered_text_in_rect(frame,
            "Then restart the app.",
            (0, _ix(h*0.55), w, _ix(h*0.63)),
            base_scale=0.36, color=COL_TEXT_DIM, thickness=1, outline=1)
        draw_bottom_bar(frame, "ESC Back")
        return

    # ── Layout: left panel (ports + status) | right panel (commands) ────
    mid_x  = w // 2
    pad    = _ix(w * 0.02)
    top_y  = _ix(h * 0.11)
    bot_y  = _ix(h * 0.90)

    # ── LEFT: Connection panel ───────────────────────────────────────────
    lx1, lx2 = pad, mid_x - pad
    draw_panel(frame, lx1, top_y, lx2, bot_y,
               fill=(6, 8, 20), alpha=0.90,
               border=COL_CYAN if connected else (80, 80, 100),
               border_thickness=2)

    ly = top_y + _ix(h * 0.03)
    lh = _ix(h * 0.038)

    # Connection status header
    status_col = COL_GREEN if connected else (200, 80, 80)
    status_str = f"CONNECTED  -  {port_name}" if connected else "DISCONNECTED"
    draw_centered_text_in_rect(frame, status_str,
        (lx1, ly, lx2, ly + lh),
        base_scale=0.44, color=status_col, thickness=2, outline=3)
    ly += lh + _ix(h * 0.01)

    # Divider
    cv2.line(frame, (lx1 + pad, ly), (lx2 - pad, ly), (40, 50, 70), 1)
    ly += _ix(h * 0.015)

    # Port list
    draw_outlined_text(frame, "Available ports:", lx1 + pad, ly, 0.36,
                       COL_TEXT_DIM, thickness=1, outline=1)
    ly += _ix(h * 0.04)

    if not available:
        draw_outlined_text(frame, "  (none found  -  is ESP32 plugged in?)",
                           lx1 + pad, ly, 0.32, (180, 100, 60),
                           thickness=1, outline=1)
        ly += _ix(h * 0.04)
    else:
        for i, port in enumerate(available):
            is_sel = (i == sel_idx)
            bg_col = (20, 40, 60) if is_sel else (8, 10, 20)
            bdr    = COL_CYAN if is_sel else (40, 50, 70)
            draw_panel(frame,
                       lx1 + pad, ly - _ix(h*0.005),
                       lx2 - pad, ly + _ix(h*0.032),
                       fill=bg_col, alpha=0.90, border=bdr, border_thickness=1)
            prefix = "> " if is_sel else "  "
            col    = COL_CYAN if is_sel else COL_TEXT_DIM
            draw_outlined_text(frame, f"{prefix}{port}",
                               lx1 + _ix(w*0.03), ly + _ix(h*0.022),
                               0.38, col, thickness=1, outline=2)
            ly += _ix(h * 0.048)

    ly += _ix(h * 0.01)
    cv2.line(frame, (lx1 + pad, ly), (lx2 - pad, ly), (40, 50, 70), 1)
    ly += _ix(h * 0.015)

    # Last TX / RX
    tx_col = COL_GREEN if last_tx else COL_TEXT_DIM
    rx_col = COL_YELLOW if last_rx else COL_TEXT_DIM
    draw_outlined_text(frame, f"TX: {last_tx or '(none)'}",
                       lx1 + pad, ly, 0.34, tx_col, thickness=1, outline=1)
    ly += _ix(h * 0.04)
    draw_outlined_text(frame, f"RX: {last_rx or '(none)'}",
                       lx1 + pad, ly, 0.34, rx_col, thickness=1, outline=1)
    ly += _ix(h * 0.05)

    # Status message
    if status_msg:
        draw_centered_text_in_rect(frame, status_msg,
            (lx1, ly, lx2, ly + _ix(h*0.045)),
            base_scale=0.32, color=COL_TEXT_ACCENT, thickness=1, outline=1)

    # ── RIGHT: Command panel ─────────────────────────────────────────────
    rx1, rx2 = mid_x + pad, w - pad
    draw_panel(frame, rx1, top_y, rx2, bot_y,
               fill=(6, 8, 20), alpha=0.90,
               border=(80, 80, 100), border_thickness=1)

    ry = top_y + _ix(h * 0.03)
    rh = _ix(h * 0.038)

    draw_centered_text_in_rect(frame, "COMMANDS",
        (rx1, ry, rx2, ry + rh),
        base_scale=0.44, color=COL_TEXT_ACCENT, thickness=1, outline=2)
    ry += rh + _ix(h * 0.015)

    cv2.line(frame, (rx1 + pad, ry), (rx2 - pad, ry), (40, 50, 70), 1)
    ry += _ix(h * 0.02)

    commands = [
        ("R", "ROCK",     COL_CYAN),
        ("P", "PAPER",    COL_GREEN),
        ("S", "SCISSORS", COL_MAGENTA),
        ("O", "OPEN",     (180, 180, 80)),
        ("C", "CLOSE",    (120, 120, 180)),
        ("T", "PING",     COL_TEXT_DIM),
    ]

    cmd_h = _ix(h * 0.068)
    for key_char, cmd, col in commands:
        active = last_tx == f"CMD|{cmd}"
        bg = (15, 30, 15) if active else (8, 10, 20)
        bdr = COL_GREEN if active else (40, 50, 70)
        draw_panel(frame,
                   rx1 + pad, ry,
                   rx2 - pad, ry + cmd_h - 4,
                   fill=bg, alpha=0.92, border=bdr, border_thickness=1)
        # Key badge
        kx = rx1 + _ix(w*0.03)
        draw_panel(frame, kx, ry + 4, kx + _ix(w*0.04), ry + cmd_h - 8,
                   fill=(20, 35, 55), alpha=0.95, border=col, border_thickness=1)
        draw_centered_text_in_rect(frame, key_char,
            (kx, ry + 4, kx + _ix(w*0.04), ry + cmd_h - 8),
            base_scale=0.42, color=col, thickness=2, outline=2)
        # Command name
        draw_outlined_text(frame, f"CMD|{cmd}",
                           kx + _ix(w*0.05), ry + _ix(h*0.038),
                           0.38, col if active else COL_TEXT_DIM,
                           thickness=1, outline=2)
        if active:
            draw_outlined_text(frame, "SENT",
                               rx2 - _ix(w*0.09), ry + _ix(h*0.038),
                               0.30, COL_GREEN, thickness=1, outline=1)
        ry += cmd_h

    ry += _ix(h * 0.01)
    cv2.line(frame, (rx1 + pad, ry), (rx2 - pad, ry), (40, 50, 70), 1)
    ry += _ix(h * 0.02)

    # Connection instructions
    if not connected:
        hint_lines = [
            "1. Plug in ESP32 via USB",
            "2. Press [ ] to select port",
            "3. Press ENTER to connect",
            "4. Use R / P / S to test",
        ]
    else:
        hint_lines = [
            "Connected! Test with R / P / S",
            "T = PING (connection test)",
            "X = Disconnect",
        ]
    for line in hint_lines:
        col = COL_TEXT_ACCENT if connected else COL_TEXT_DIM
        draw_outlined_text(frame, line, rx1 + pad, ry, 0.32, col,
                           thickness=1, outline=1)
        ry += _ix(h * 0.038)

    draw_bottom_bar(frame,
        "[ ] Cycle ports  |  ENTER Connect  |  R Rock  P Paper  S Scissors  |  T Ping  |  X Disconnect  |  ESC Back")


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

    draw_panel(frame, 0, 0, w - 1, h - 1, fill=(4, 6, 16), alpha=0.80,
               border=(30, 30, 50), border_thickness=0)
    draw_top_bar(frame, "PLAYER FEEDBACK",
                 "Type your suggestion and press ENTER  |  ESC Cancel")

    if submitted:
        # Confirmation screen
        pulse = 0.85 + 0.15 * abs(_math.sin(t * _math.pi * 1.5))
        col   = tuple(min(255, int(c * pulse)) for c in COL_GREEN)
        draw_centered_text_in_rect(frame, "FEEDBACK SUBMITTED",
            (0, _ix(h*0.28), w, _ix(h*0.42)),
            base_scale=0.80, color=col, thickness=3, outline=5)
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
               border=COL_CYAN, border_thickness=2)

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
    Shown once before the player enters their name.
    selected: 0 = Accept, 1 = Decline
    """
    import math as _math
    import time as _time

    w, h = frame.shape[1], frame.shape[0]
    t    = _time.monotonic()

    # Full dark background
    draw_panel(frame, 0, 0, w - 1, h - 1, fill=(4, 6, 16), alpha=0.90,
               border=(4, 6, 16), border_thickness=0)

    draw_top_bar(frame, "RPS ROBOT", "Privacy Notice")

    # ── Title ────────────────────────────────────────────────────────────
    title_y1 = _ix(h * 0.08)
    title_y2 = _ix(h * 0.16)
    draw_centered_text_in_rect(frame, "BEFORE YOU PLAY",
        (0, title_y1, w, title_y2),
        base_scale=0.65, color=COL_CYAN, thickness=2, outline=4)

    # ── Content panel — occupies middle section ──────────────────────────
    px1 = _ix(w * 0.07)
    px2 = _ix(w * 0.93)
    py1 = _ix(h * 0.17)
    py2 = _ix(h * 0.70)
    pw  = px2 - px1
    ph  = py2 - py1

    draw_panel(frame, px1, py1, px2, py2,
               fill=(6, 10, 24), alpha=0.94,
               border=(50, 70, 100), border_thickness=1)

    # Content rows — fixed pixel positions, not cumulative offsets
    # This prevents any overflow regardless of line count
    pad  = _ix(w * 0.035)
    tx   = px1 + pad
    rows = [
        # (y_fraction_of_panel, text, color, scale)
        (0.06,  "To help improve RPS Robot, the app can optionally send:",
                (160, 190, 220), 0.36),

        (0.20,  "CRASH REPORTS  (automatic, if the app stops unexpectedly)",
                (100, 210, 120), 0.34),
        (0.28,  "   Includes: error message, OS version, Python version, app version.",
                (100, 130, 110), 0.28),
        (0.34,  "   Never includes: gameplay data, camera feed, or personal info.",
                (100, 130, 110), 0.28),

        (0.48,  "FEEDBACK  (only when you press ENTER to submit a note)",
                (100, 210, 120), 0.34),
        (0.56,  "   Includes: your player name, the message you typed, timestamp.",
                (100, 130, 110), 0.28),

        (0.70,  "NOTHING ELSE is ever sent. No camera. No location. No tracking.",
                (220, 200, 100), 0.32),

        (0.82,  "All data goes to a private developer Discord channel only.",
                (100, 110, 130), 0.27),
        (0.90,  "You can change this choice at any time in Settings > Privacy.",
                (100, 110, 130), 0.27),
    ]

    for frac, text, col, scale in rows:
        ty = py1 + int(ph * frac)
        draw_outlined_text(frame, text, tx, ty, scale, col,
                           thickness=1, outline=2)

    # ── Buttons ──────────────────────────────────────────────────────────
    # Fixed positions — never overlap the content panel
    btn_y1  = _ix(h * 0.73)
    btn_y2  = _ix(h * 0.89)
    btn_h   = btn_y2 - btn_y1
    btn_mid = w // 2
    btn_gap = _ix(w * 0.025)

    # -- Accept --
    ac_x1 = _ix(w * 0.07)
    ac_x2 = btn_mid - btn_gap

    if selected == 0:
        # Selected: dark background, bright cyan border, white text
        draw_panel(frame, ac_x1, btn_y1, ac_x2, btn_y2,
                   fill=(8, 28, 18), alpha=0.96,
                   border=(60, 220, 120), border_thickness=3)
        # Label
        draw_centered_text_in_rect(frame, "ACCEPT",
            (ac_x1, btn_y1, ac_x2, btn_y1 + int(btn_h * 0.55)),
            base_scale=0.54, color=(60, 230, 130), thickness=2, outline=3)
        # Subtitle
        draw_centered_text_in_rect(frame, "Send crash reports + feedback",
            (ac_x1, btn_y1 + int(btn_h * 0.58), ac_x2, btn_y2),
            base_scale=0.27, color=(60, 160, 90), thickness=1, outline=2)
    else:
        # Unselected: very dark, dim text
        draw_panel(frame, ac_x1, btn_y1, ac_x2, btn_y2,
                   fill=(6, 14, 10), alpha=0.90,
                   border=(30, 80, 50), border_thickness=1)
        draw_centered_text_in_rect(frame, "ACCEPT",
            (ac_x1, btn_y1, ac_x2, btn_y1 + int(btn_h * 0.55)),
            base_scale=0.54, color=(40, 120, 70), thickness=1, outline=2)
        draw_centered_text_in_rect(frame, "Send crash reports + feedback",
            (ac_x1, btn_y1 + int(btn_h * 0.58), ac_x2, btn_y2),
            base_scale=0.27, color=(30, 80, 50), thickness=1, outline=1)

    # -- No Thanks --
    dc_x1 = btn_mid + btn_gap
    dc_x2 = _ix(w * 0.93)

    if selected == 1:
        # Selected: dark background, red border, red text
        draw_panel(frame, dc_x1, btn_y1, dc_x2, btn_y2,
                   fill=(28, 8, 8), alpha=0.96,
                   border=(220, 80, 80), border_thickness=3)
        draw_centered_text_in_rect(frame, "NO THANKS",
            (dc_x1, btn_y1, dc_x2, btn_y1 + int(btn_h * 0.55)),
            base_scale=0.54, color=(230, 90, 90), thickness=2, outline=3)
        draw_centered_text_in_rect(frame, "Save locally only",
            (dc_x1, btn_y1 + int(btn_h * 0.58), dc_x2, btn_y2),
            base_scale=0.27, color=(160, 70, 70), thickness=1, outline=2)
    else:
        # Unselected: very dark, dim text
        draw_panel(frame, dc_x1, btn_y1, dc_x2, btn_y2,
                   fill=(14, 6, 6), alpha=0.90,
                   border=(80, 30, 30), border_thickness=1)
        draw_centered_text_in_rect(frame, "NO THANKS",
            (dc_x1, btn_y1, dc_x2, btn_y1 + int(btn_h * 0.55)),
            base_scale=0.54, color=(120, 50, 50), thickness=1, outline=2)
        draw_centered_text_in_rect(frame, "Save locally only",
            (dc_x1, btn_y1 + int(btn_h * 0.58), dc_x2, btn_y2),
            base_scale=0.27, color=(80, 35, 35), thickness=1, outline=1)

    # Selection indicator arrow above active button
    pulse = 0.6 + 0.4 * abs(_math.sin(t * _math.pi * 1.5))
    arr_col = tuple(min(255, int(c * pulse)) for c in
                    ((60, 220, 120) if selected == 0 else (220, 80, 80)))
    arr_cx = (ac_x1 + ac_x2) // 2 if selected == 0 else (dc_x1 + dc_x2) // 2
    arr_y  = btn_y1 - _ix(h * 0.015)
    cv2.fillPoly(frame, [__import__('numpy').array([
        [arr_cx,          arr_y],
        [arr_cx - _ix(w*0.015), arr_y - _ix(h*0.025)],
        [arr_cx + _ix(w*0.015), arr_y - _ix(h*0.025)],
    ], dtype=__import__('numpy').int32)], arr_col)

    draw_bottom_bar(frame,
        "LEFT / RIGHT  -  choose  |  ENTER  -  confirm  |  TAB  -  toggle")

"""
ui_modes.py -- Per-mode screen renderers: 2-player, reflex, bluff, simon, squid, rpsls.
"""
import cv2
import math
import time
import numpy as np
from ui_base import *
from ui_game import _draw_gesture_icon, _draw_last_round_replay  # noqa: F401

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

    # Draw player panels - show live tracker gesture during gameplay,
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
    1v1v1: Player 1 vs Player 2 vs AI - everyone for themselves.
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
    """AI Personality selector  -  left list, right detail card."""
    from fair_play_ai import PERSONALITIES, PERSONALITY_NAMES
    w, h = frame.shape[1], frame.shape[0]
    t    = time.monotonic()

    draw_panel(frame, 0, 0, w - 1, h - 1, fill=(6, 8, 18), alpha=0.95,
               border=COL_MAGENTA, border_thickness=2)
    draw_top_bar(frame, "AI PERSONALITIES", "W/S Select  |  Enter Confirm  |  ESC Back")

    # ── Per-personality metadata ──────────────────────────────────────────
    _META = {
        "Normal":           {"colour": (120,180,220), "icon": "~",
                             "flavour": "Balanced adaptive play. Watches and learns over time.",
                             "strength": "Well-rounded  -  hard to exploit any single weakness.",
                             "weakness": "No dominant strategy. Predictable mid-game.",
                             "tip": "After 5+ rounds you can start reading its patterns.",
                             "stats": [("Aggression",0.50),("Pattern Use",0.65),("Randomness",0.35),("Adaptability",0.60)]},
        "The Psychologist": {"colour": (180,80,255),  "icon": "P",
                             "flavour": "Exploits win-stay/lose-shift. Built on cognitive bias research.",
                             "strength": "Deadly against players who repeat after wins or shift after losses.",
                             "weakness": "Plays randomly in early rounds  -  easy before round 8.",
                             "tip": "Force yourself to break your own patterns. Randomise deliberately.",
                             "stats": [("Aggression",0.75),("Pattern Use",0.90),("Randomness",0.15),("Adaptability",0.85)]},
        "The Gambler":      {"colour": (60,200,120),  "icon": "G",
                             "flavour": "Reads patterns then randomly ignores them. High variance.",
                             "strength": "Unpredictable. 20% wild moves break any counter-strategy.",
                             "weakness": "Wild moves cost it winning rounds  -  patience beats it.",
                             "tip": "Stay consistent. Its own chaos will hand you wins.",
                             "stats": [("Aggression",0.60),("Pattern Use",0.45),("Randomness",0.75),("Adaptability",0.40)]},
        "The Mirror":       {"colour": (80,220,220),  "icon": "M",
                             "flavour": "Copies your most common gesture. Blatantly exploitable.",
                             "strength": "Wins hard against players with a dominant gesture bias.",
                             "weakness": "Locked to your modal move. One adaptation beats it.",
                             "tip": "Throw your least common gesture 3 times in a row.",
                             "stats": [("Aggression",0.30),("Pattern Use",0.95),("Randomness",0.05),("Adaptability",0.20)]},
        "The Ghost":        {"colour": (160,160,200), "icon": "G",
                             "flavour": "Plays your previous move back at you. One step behind.",
                             "strength": "Beats players who unconsciously repeat their last throw.",
                             "weakness": "Always one round behind. Trivially exploitable once known.",
                             "tip": "Throw Rock. It plays Rock next. Counter with Paper.",
                             "stats": [("Aggression",0.25),("Pattern Use",0.80),("Randomness",0.10),("Adaptability",0.15)]},
        "The Chaos Agent":  {"colour": (200,60,60),   "icon": "?",
                             "flavour": "Pure Nash equilibrium. 33/33/33 every single round.",
                             "strength": "Theoretically unbeatable. No patterns to exploit.",
                             "weakness": "Cannot adapt or exploit you either. Max draws.",
                             "tip": "You cannot beat it consistently. Accept the chaos.",
                             "stats": [("Aggression",0.33),("Pattern Use",0.00),("Randomness",1.00),("Adaptability",0.00)]},
        "The Hustler":      {"colour": (255,160,40),  "icon": "H",
                             "flavour": "Learns fast. Hits hard early, plays coy when winning.",
                             "strength": "Best pattern learning. Adapts within 3-4 rounds.",
                             "weakness": "Overconfident when ahead  -  variance spikes late.",
                             "tip": "Stay unpredictable in rounds 1-5 to disrupt its read.",
                             "stats": [("Aggression",0.90),("Pattern Use",0.85),("Randomness",0.20),("Adaptability",0.95)]},
    }

    # Layout constants
    list_x1 = _ix(w * 0.03)
    list_x2 = _ix(w * 0.39)
    desc_x1 = _ix(w * 0.42)
    desc_x2 = _ix(w * 0.97)
    y_start  = _ix(h * 0.13)
    n        = len(PERSONALITY_NAMES)
    item_h   = int((h * 0.86 - y_start) / n)   # even split, no overflow

    # ── Left: personality list ──────────────────────────────────────────────
    for i, name in enumerate(PERSONALITY_NAMES):
        p      = PERSONALITIES[name]
        meta   = _META.get(name, {})
        col    = meta.get("colour", COL_TEXT_ACCENT)
        iy1    = y_start + i * item_h
        iy2    = iy1 + item_h - 3
        is_sel = (name == selected_name)

        fill   = tuple(min(255, int(c * 0.15)) for c in col) if is_sel else (8, 8, 16)
        draw_panel(frame, list_x1, iy1, list_x2, iy2,
                   fill=fill, alpha=0.92,
                   border=col if is_sel else (40, 40, 60),
                   border_thickness=2 if is_sel else 1)

        # Colour dot
        dot_r = max(6, item_h // 5)
        dot_x = list_x1 + dot_r + 6
        dot_y = (iy1 + iy2) // 2
        cv2.circle(frame, (dot_x, dot_y), dot_r, col if is_sel else (50,50,60), -1)

        # Label  -  centred in remaining width
        label_scale = 0.40 if len(p["label"]) > 12 else 0.46
        draw_centered_text_in_rect(frame, p["label"],
            (list_x1 + dot_r*2 + 10, iy1, list_x2 - 4, iy2),
            base_scale=label_scale,
            color=col if is_sel else COL_TEXT_DIM,
            thickness=2 if is_sel else 1,
            outline=2 if is_sel else 1)

        # Arrow to right panel when selected
        if is_sel:
            pulse = 0.6 + 0.4 * abs(math.sin(t * math.pi * 2.0))
            ac = tuple(min(255, int(c * pulse)) for c in col)
            ax = desc_x1 - 8
            ay = dot_y
            pts = np.array([[ax, ay],
                             [ax-12, ay-8],
                             [ax-12, ay+8]], dtype=np.int32)
            cv2.fillPoly(frame, [pts], ac)

    # ── Right: detail panel ────────────────────────────────────────────────
    panel_y1 = y_start
    panel_y2 = y_start + n * item_h
    panel_h  = panel_y2 - panel_y1
    sel_meta = _META.get(selected_name, {})
    sel_col  = sel_meta.get("colour", COL_TEXT_ACCENT)
    draw_panel(frame, desc_x1, panel_y1, desc_x2, panel_y2,
               fill=(8, 5, 20), alpha=0.94, border=sel_col, border_thickness=2)

    desc_w = desc_x2 - desc_x1
    py = panel_y1 + _ix(h * 0.015)

    # Name header
    sel_label = PERSONALITIES.get(selected_name, {}).get("label", selected_name)
    draw_centered_text_in_rect(frame, sel_label,
        (desc_x1, py, desc_x2, py + _ix(h * 0.058)),
        base_scale=0.58, color=sel_col, thickness=2, outline=3)
    py += _ix(h * 0.065)

    # Flavour  -  centred within the right panel
    flavour = sel_meta.get("flavour", "")
    fl_scale = get_fit_scale(flavour, int(desc_w * 0.88), base_scale=0.32, thickness=1, min_scale=0.24)
    draw_centered_text_in_rect(frame, flavour,
        (desc_x1, py, desc_x2, py + _ix(h * 0.046)),
        base_scale=fl_scale, color=COL_TEXT_ACCENT, thickness=1, outline=1)
    py += _ix(h * 0.050)

    # ── Stat bars ──────────────────────────────────────────────────────────
    bx1    = desc_x1 + int(desc_w * 0.05)
    bx2    = desc_x2 - int(desc_w * 0.05)
    full_w = bx2 - bx1
    label_w = int(full_w * 0.38)  # left portion for text
    bar_x1  = bx1 + label_w + 4
    bar_x2  = bx2 - _ix(w * 0.030)   # leave room for % label without clipping
    bar_h   = max(6, _ix(h * 0.016))

    for stat_name, val in sel_meta.get("stats", []):
        # Label left-aligned
        draw_outlined_text(frame, stat_name, bx1, py + bar_h,
                           0.26, COL_TEXT_DIM, thickness=1, outline=0)
        # Bar background
        cv2.rectangle(frame, (bar_x1, py), (bar_x2, py + bar_h), (25, 25, 35), -1)
        # Bar fill
        fill_end = bar_x1 + int((bar_x2 - bar_x1) * val)
        bar_col  = sel_col if val >= 0.6 else (COL_YELLOW if val >= 0.3 else (80, 80, 100))
        if fill_end > bar_x1:
            cv2.rectangle(frame, (bar_x1, py), (fill_end, py + bar_h), bar_col, -1)
        # Bar border
        cv2.rectangle(frame, (bar_x1, py), (bar_x2, py + bar_h), (60, 60, 80), 1)
        # Percentage label  -  right-aligned within panel
        pct_txt = f"{int(val*100)}%"
        draw_outlined_text(frame, pct_txt, bar_x2 + _ix(w * 0.005), py + bar_h,
                           0.24, sel_col, thickness=1, outline=0)
        py += _ix(h * 0.050)

    py += _ix(h * 0.008)

    # ── Strength ───────────────────────────────────────────────────────────
    cv2.line(frame, (bx1, py), (bx2, py), (50, 50, 70), 1)
    py += _ix(h * 0.012)
    draw_outlined_text(frame, "STRENGTH:", bx1, py + _ix(h * 0.022),
                       0.28, COL_GREEN, thickness=1, outline=1)
    py += _ix(h * 0.030)
    strength = sel_meta.get("strength", "")
    s_scale = get_fit_scale(strength, int(desc_w * 0.88), base_scale=0.28, thickness=1, min_scale=0.22)
    draw_centered_text_in_rect(frame, strength,
        (desc_x1, py, desc_x2, py + _ix(h * 0.038)),
        base_scale=s_scale, color=COL_GREEN, thickness=1, outline=1)
    py += _ix(h * 0.042)

    # ── Weakness ───────────────────────────────────────────────────────────
    draw_outlined_text(frame, "WEAKNESS:", bx1, py + _ix(h * 0.022),
                       0.28, COL_RED, thickness=1, outline=1)
    py += _ix(h * 0.030)
    weakness = sel_meta.get("weakness", "")
    w_scale = get_fit_scale(weakness, int(desc_w * 0.88), base_scale=0.28, thickness=1, min_scale=0.22)
    draw_centered_text_in_rect(frame, weakness,
        (desc_x1, py, desc_x2, py + _ix(h * 0.038)),
        base_scale=w_scale, color=COL_RED, thickness=1, outline=1)
    py += _ix(h * 0.042)

    # ── How to beat it ─────────────────────────────────────────────────────
    draw_outlined_text(frame, "HOW TO BEAT IT:", bx1, py + _ix(h * 0.022),
                       0.28, COL_YELLOW, thickness=1, outline=1)
    py += _ix(h * 0.030)
    tip = sel_meta.get("tip", "")
    t_scale = get_fit_scale(tip, int(desc_w * 0.88), base_scale=0.28, thickness=1, min_scale=0.22)
    draw_centered_text_in_rect(frame, tip,
        (desc_x1, py, desc_x2, py + _ix(h * 0.038)),
        base_scale=t_scale, color=COL_YELLOW, thickness=1, outline=1)

    draw_bottom_bar(frame, "W/S Navigate  |  Enter to Select  |  ESC Back  |  Currently: " + selected_name)


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

    draw_top_bar(frame, "SPEED REFLEX  -  SOLO",
                 f"Match the gesture! Score: {score}  |  Q Quit")

    if cur_state == "INTRO":
        draw_centered_text(frame, "SPEED REFLEX", h // 2 - _ix(h * 0.10),
                           1.0, COL_YELLOW, thickness=3, outline=5)
        draw_centered_text(frame, "Match each gesture as fast as you can!",
                           h // 2 + _ix(h * 0.02), 0.44, COL_TEXT_ACCENT,
                           thickness=1, outline=2)
        draw_centered_text(frame, "30 seconds  -  No penalty for misses",
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
            res_txt = "HIT!" if last_res == "hit" else ("TIME!" if last_res == "timeout" else "MISS")
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

    draw_bottom_bar(frame, f"Score: {score}  |  Avg RT: {avg_rt}ms  |  Say RESTART or BACK  |  Q Quit")


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

    draw_top_bar(frame, "SPEED REFLEX  -  2 PLAYER",
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

    # -- Centre: AI declaration panel --------------------------------------
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

    # -- Result banner full-width ------------------------------------------
    if cur_state in ("ROUND_RESULT", "MATCH_RESULT"):
        b_col = get_result_banner_color(banner)
        draw_centered_text(frame, banner, pan_y2 + _ix(h * 0.10),
                           0.62, b_col, thickness=2, outline=3)

    # -- Score strip ------------------------------------------------------
    draw_centered_text(frame, score_text, pan_y2 + _ix(h * 0.19),
                       0.50, COL_TEXT_ACCENT, thickness=2, outline=3)

    # -- Research stat: bluff rate so far ---------------------------------
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
        # Showing the sequence - display each gesture big
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
        draw_centered_text(frame, "CORRECT!", cy_mid - _ix(h * 0.06),
                           0.90, COL_GREEN, thickness=2, outline=4)
        draw_centered_text(frame, f"Next: {seq_len + 1} gestures",
                           cy_mid + _ix(h * 0.07), 0.44, COL_TEXT_DIM,
                           thickness=1, outline=2)
    else:
        # Player input phase
        draw_status_chip(frame, f"YOUR TURN - Step {step_index + 1} / {seq_len}",
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
    draw_bottom_bar(frame, "Match each gesture within 2 seconds  |  Say BACK  |  Q Quit")


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

    draw_top_bar(frame, "SIMON SAYS  -  2 PLAYER CHAIN",
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
        draw_status_chip(frame, f"{active}  -  ADD A GESTURE", _ix(h * 0.10), act_col)
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
        draw_status_chip(frame, f"{active}  -  REPEAT STEP {step_index + 1}/{chain_len}",
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
    # Support both key names from different controller versions
    capture_pct  = game_state.get("dwell_pct",
                   game_state.get("capture_progress", 0.0))
    score        = game_state.get("score", 0)
    dots_done    = game_state.get("dots_collected", 0)
    survival     = game_state.get("survived_secs",
                   game_state.get("survival_time", 0.0))
    game_over    = game_state.get("eliminated",
                   game_state.get("game_over", False))
    light_left   = game_state.get("light_time_left", 0.0)
    state        = game_state.get("state", "PLAYING")
    t            = time.monotonic()

    # -- Background tint based on light -----------------------------------
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

    if state == "INTRO":
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

    if game_over or state == "GAME_OVER":
        draw_centered_text(frame, "ELIMINATED!", h // 2 - _ix(h * 0.14),
                           1.0, COL_RED, thickness=3, outline=5)
        go_text = game_state.get("game_over_reason",
                  game_state.get("game_over_text", ""))
        parts = go_text.split("|")
        for i, part in enumerate(parts[:3]):
            draw_centered_text(frame, part.strip(),
                               h // 2 + _ix(h * 0.02) + i * _ix(h * 0.09),
                               0.38, COL_TEXT_ACCENT, thickness=1, outline=2)
        draw_top_bar(frame, "SQUID GAME  -  GAME OVER", "Q Quit  |  ESC Menu")
        return

    # -- Light indicator ---------------------------------------------------
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

    # -- Dot ---------------------------------------------------------------
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
    cv2.circle(frame, (dx, dy), max(dot_r - 4, 2), dot_col, -1)

    # Capture progress ring
    if capture_pct > 0:
        angle = int(360 * capture_pct)
        cv2.ellipse(frame, (dx, dy), (dot_r + 10, dot_r + 10), -90,
                    0, angle, COL_GREEN, 4)

    # -- Finger cursor -----------------------------------------------------
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

    # -- Score strip -------------------------------------------------------
    score_y = _ix(h * 0.92)
    for txt, sx in [
        (f"Score: {score}", _ix(w * 0.05)),
        (f"Dots: {dots_done}", _ix(w * 0.35)),
        (f"Time: {survival:.0f}s", _ix(w * 0.65)),
    ]:
        draw_outlined_text(frame, txt, sx, score_y, 0.40,
                           COL_TEXT_ACCENT, thickness=1, outline=2)

    draw_bottom_bar(frame, "Move finger to dot on GREEN  |  FREEZE on RED  |  Say BACK or Q Quit")


# ============================================================
# SQUID GAME 2P: RED LIGHT GREEN LIGHT - TWO PLAYER RACE
# ============================================================

def draw_squid_game_2p_view(frame, game_state, p1_hand=None, p2_hand=None):
    """Two-player Red Light Green Light. P1=cyan left half, P2=magenta right half."""
    w, h = frame.shape[1], frame.shape[0]
    t    = time.monotonic()

    state      = game_state.get("state", "INTRO")
    light      = game_state.get("light", "GREEN")
    light_left = game_state.get("light_time_left", 0.0)
    winner     = game_state.get("winner", 0)
    loser      = game_state.get("loser", 0)
    reason     = game_state.get("game_over_reason", "")
    win_dots   = game_state.get("win_dots", 5)

    # Background tint
    if state == "GAME_OVER":
        bg = (0, 0, 20)
    elif light == "RED":
        pulse_bg = 0.15 + 0.08 * abs(math.sin(t * math.pi * 3))
        bg = (0, 0, int(40 * pulse_bg + 10))
    else:
        bg = (0, 20, 0)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), bg, -1)
    cv2.addWeighted(overlay, 0.40, frame, 0.60, 0, frame)

    play_y1 = _ix(h * 0.14)
    play_y2 = _ix(h * 0.90)
    play_h  = play_y2 - play_y1

    # ── INTRO ──────────────────────────────────────────────────────────────
    if state == "INTRO":
        draw_centered_text(frame, "RED LIGHT / GREEN LIGHT", h // 2 - _ix(h * 0.12),
                           0.68, COL_GREEN, thickness=2, outline=4)
        draw_centered_text(frame, "P1 (cyan) and P2 (magenta) race to collect dots",
                           h // 2 + _ix(h * 0.02), 0.38, COL_TEXT_ACCENT, thickness=1, outline=2)
        draw_centered_text(frame, f"First to {win_dots} dots wins  |  FREEZE on RED",
                           h // 2 + _ix(h * 0.10), 0.38, COL_TEXT_DIM, thickness=1, outline=2)
        draw_centered_text(frame, "P1 = Left hand index finger",
                           h // 2 + _ix(h * 0.18), 0.36, COL_CYAN, thickness=1, outline=2)
        draw_centered_text(frame, "P2 = Right hand index finger",
                           h // 2 + _ix(h * 0.25), 0.36, COL_MAGENTA, thickness=1, outline=2)
        draw_top_bar(frame, "RED LIGHT GREEN LIGHT  -  2 PLAYER", "Q Quit")
        return

    # ── GAME_OVER ───────────────────────────────────────────────────────────
    if state == "GAME_OVER":
        if winner == 0:
            draw_centered_text(frame, "BOTH ELIMINATED!", h // 2 - _ix(h * 0.10),
                               0.80, COL_RED, thickness=3, outline=5)
        else:
            w_col = COL_CYAN if winner == 1 else COL_MAGENTA
            l_col = COL_CYAN if loser  == 1 else COL_MAGENTA
            draw_centered_text(frame, f"P{winner} WINS!", h // 2 - _ix(h * 0.12),
                               0.90, w_col, thickness=3, outline=5)
            if loser:
                draw_centered_text(frame, f"P{loser} eliminated",
                                   h // 2 + _ix(h * 0.00), 0.50, l_col, thickness=2, outline=3)
        draw_centered_text(frame, reason, h // 2 + _ix(h * 0.12),
                           0.40, COL_TEXT_ACCENT, thickness=1, outline=2)
        draw_top_bar(frame, "RED LIGHT GREEN LIGHT  -  2P", "Q Quit  |  ESC Menu")
        return

    # ── PLAYING ─────────────────────────────────────────────────────────────

    # Light banner
    light_col = (0, 60, 0) if light == "GREEN" else (60, 0, 0)
    light_txt = "GREEN LIGHT" if light == "GREEN" else "RED LIGHT"
    light_fc  = COL_GREEN if light == "GREEN" else COL_RED
    pulse     = 0.85 + 0.15 * abs(math.sin(t * math.pi * (4 if light == "RED" else 1)))
    pc        = tuple(min(255, int(c * pulse)) for c in light_fc)
    draw_panel(frame, 0, 0, w, _ix(h * 0.12),
               fill=light_col, alpha=0.70, border=pc, border_thickness=3)
    draw_centered_text(frame, light_txt, _ix(h * 0.07), 0.80, pc, thickness=3, outline=5)

    # Light countdown bar
    pct_l = min(1.0, light_left / 5.0)
    bx1 = _ix(w * 0.05); bx2 = _ix(w * 0.95)
    by  = _ix(h * 0.12);  bh  = _ix(h * 0.012)
    cv2.rectangle(frame, (bx1, by), (bx2, by + bh), (30, 30, 30), -1)
    cv2.rectangle(frame, (bx1, by), (bx1 + int((bx2 - bx1) * pct_l), by + bh), pc, -1)

    # Centre divider
    mid_x = w // 2
    cv2.line(frame, (mid_x, play_y1), (mid_x, play_y2), (60, 60, 80), 1)
    draw_outlined_text(frame, "P1", _ix(w * 0.02), play_y1 + _ix(h * 0.04),
                       0.36, COL_CYAN, thickness=1, outline=1)
    draw_outlined_text(frame, "P2", _ix(w * 0.53), play_y1 + _ix(h * 0.04),
                       0.36, COL_MAGENTA, thickness=1, outline=1)

    # Draw dots and cursors for each player
    player_defs = [
        ("p1", COL_CYAN,    p1_hand, "p1_dot_x", "p1_dot_y",
         "p1_dwell_pct", "p1_flash", "p1_eliminated", "p1_dots"),
        ("p2", COL_MAGENTA, p2_hand, "p2_dot_x", "p2_dot_y",
         "p2_dwell_pct", "p2_flash", "p2_eliminated", "p2_dots"),
    ]

    for (pid, col, hand_st,
         dx_key, dy_key, dwell_key, flash_key, elim_key, dots_key) in player_defs:

        eliminated = game_state.get(elim_key, False)
        dot_x      = game_state.get(dx_key, 0.5)
        dot_y      = game_state.get(dy_key, 0.5)
        dwell_pct  = game_state.get(dwell_key, 0.0)
        flash      = game_state.get(flash_key, False)
        dots       = game_state.get(dots_key, 0)

        if eliminated:
            col = (80, 80, 80)

        dx    = int(dot_x * w)
        dy    = int(play_y1 + dot_y * play_h)
        dot_r = _ix(min(w, h) * 0.032)

        dot_pulse = 0.80 + 0.20 * abs(math.sin(t * math.pi * 1.5))
        dc = tuple(min(255, int(c * dot_pulse)) for c in col)
        cv2.circle(frame, (dx, dy), dot_r + 4, dc, 2)
        cv2.circle(frame, (dx, dy), dot_r, (10, 10, 20), -1)
        cv2.circle(frame, (dx, dy), dot_r, dc, 2)
        cv2.circle(frame, (dx, dy), max(dot_r - 5, 2), dc, -1)

        if dwell_pct > 0:
            sweep = int(360 * dwell_pct)
            cv2.ellipse(frame, (dx, dy), (dot_r + 10, dot_r + 10), -90, 0, sweep, col, 4)
        if flash:
            cv2.circle(frame, (dx, dy), dot_r + 16, col, 3)

        if hand_st and not eliminated:
            ix = hand_st.get("index_tip_x")
            iy = hand_st.get("index_tip_y")
            if ix is not None:
                fx = int(ix * w)
                fy = int(play_y1 + iy * play_h)
                cur_col = col if light == "GREEN" else COL_RED
                cv2.circle(frame, (fx, fy), _ix(w * 0.014), cur_col, -1)
                cv2.circle(frame, (fx, fy), _ix(w * 0.020), cur_col, 2)
                cv2.line(frame, (fx, fy), (dx, dy), tuple(c // 3 for c in col), 1)

    # Score strip
    score_y  = _ix(h * 0.92)
    p1_dots  = game_state.get("p1_dots", 0)
    p2_dots  = game_state.get("p2_dots", 0)
    p1_elim  = game_state.get("p1_eliminated", False)
    p2_elim  = game_state.get("p2_eliminated", False)
    p1_txt   = f"P1: {p1_dots}/{win_dots}" + (" [OUT]" if p1_elim else "")
    p2_txt   = f"P2: {p2_dots}/{win_dots}" + (" [OUT]" if p2_elim else "")
    draw_outlined_text(frame, p1_txt, _ix(w * 0.05), score_y, 0.48,
                       COL_CYAN if not p1_elim else (80, 80, 80), thickness=2, outline=2)
    draw_outlined_text(frame, p2_txt, _ix(w * 0.60), score_y, 0.48,
                       COL_MAGENTA if not p2_elim else (80, 80, 80), thickness=2, outline=2)

    draw_top_bar(frame, "RED LIGHT GREEN LIGHT  -  2 PLAYER",
                 f"First to {win_dots} dots  |  Q Quit")
    draw_bottom_bar(frame,
        "P1=Left finger  |  P2=Right finger  |  Guide to your dot  |  FREEZE on RED  |  Q Quit")

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
    "Rock": "[R]", "Paper": "[P]", "Scissors": "[S]",
    "Lizard": "[L]", "Spock": "[Sp]",
}

# The 10 win rules as compact strings
_RPSLS_RULES = [
    "Scissors cuts Paper",    "Paper covers Rock",
    "Rock crushes Lizard",    "Lizard poisons Spock",
    "Spock smashes Scissors", "Scissors decapitates Lizard",
    "Lizard eats Paper",      "Paper disproves Spock",
    "Spock vaporizes Rock",   "Rock crushes Scissors",
]


def _draw_rpsls_gesture_icon(frame, gesture, rect, show_name=True):
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
        # Snake/lizard head: oval with eyes
        axes_x = max(4, r)
        axes_y = max(3, _ix(r * 0.55))
        cv2.ellipse(frame, (cx, cy), (axes_x, axes_y), 0, 0, 180, col, 3)
        cv2.ellipse(frame, (cx, cy), (axes_x, axes_y), 0, 180, 360, col, 2)
        eye_r  = max(2, _ix(r * 0.12))
        eye_ox = max(2, _ix(r * 0.35))
        eye_oy = max(1, _ix(r * 0.20))
        cv2.circle(frame, (cx - eye_ox, cy - eye_oy), eye_r, col, -1)
        cv2.circle(frame, (cx + eye_ox, cy - eye_oy), eye_r, col, -1)
    elif gesture == "Spock":
        # Four fingers with palm arc — Vulcan salute
        finger_h = max(4, _ix(r * 0.85))
        offsets  = [-_ix(r*0.65), -_ix(r*0.22), _ix(r*0.22), _ix(r*0.65)]
        for i, fx in enumerate(offsets):
            top = cy - finger_h - (_ix(r * 0.15) if i in (0, 3) else 0)
            cv2.line(frame, (cx + fx, cy + max(1, _ix(r*0.3))),
                     (cx + fx, top), col, 3)
        palm_ax = max(4, _ix(r*0.65))
        palm_ay = max(3, _ix(r*0.30))
        cv2.ellipse(frame, (cx, cy + max(1, _ix(r*0.3))),
                    (palm_ax, palm_ay), 0, 0, 180, col, 3)
    else:
        cv2.circle(frame, (cx, cy), r, COL_TEXT_DIM, 1)

    # Name  -  only when caller hasn't already drawn it
    if show_name and gesture not in ("Unknown", ""):
        draw_centered_text_in_rect(frame, gesture,
            (x1, y2, x2, y2 + _ix(ph * 0.28)),
            base_scale=0.38, color=col, thickness=1, outline=2)


# ============================================================
# PREDICTION RACE
# ============================================================

def draw_prediction_race_view(frame, game_state, tracker_state=None):
    """Prediction Race: AI shows its live prediction. Player wins by NOT throwing it."""
    w, h = frame.shape[1], frame.shape[0]
    t    = time.monotonic()
    draw_panel(frame, 0, 0, w - 1, h - 1, fill=COL_BG_DARK, alpha=0.45,
               border=COL_BG_DARK, border_thickness=0)

    cur_state    = game_state.get("state", "WAITING_FOR_ROCK")
    beat_count   = game_state.get("beat_count", 0)
    player_score = game_state.get("player_score", 0)
    ai_score     = game_state.get("ai_score", 0)
    win_target   = game_state.get("win_target", 5)
    round_num    = game_state.get("round_number", 1)
    prediction   = game_state.get("ai_prediction", "")
    player_gest  = game_state.get("player_gesture", "")
    banner       = game_state.get("result_banner", "")
    insight      = game_state.get("last_insight", "")
    score_text   = game_state.get("score_text", "")
    cx, cy       = w // 2, h // 2

    draw_top_bar(frame, "PREDICTION RACE",
                 f"Round {round_num}  |  First to {win_target}  |  Q Quit")
    draw_bottom_bar(frame,
        "Throw anything EXCEPT what the AI predicts  |  Pump to start  |  Q Quit")

    draw_outlined_text(frame, score_text,
                       _ix(w * 0.04), _ix(h * 0.10), 0.44, COL_CYAN,
                       thickness=1, outline=2)

    for i in range(win_target):
        px = _ix(w * 0.04) + i * _ix(w * 0.028)
        pc = COL_CYAN if i < player_score else (40, 40, 60)
        cv2.circle(frame, (px, _ix(h * 0.145)), _ix(h * 0.012), pc, -1 if i < player_score else 2)
        ax = w - _ix(w * 0.04) - i * _ix(w * 0.028)
        ac = COL_RED if i < ai_score else (40, 40, 60)
        cv2.circle(frame, (ax, _ix(h * 0.145)), _ix(h * 0.012), ac, -1 if i < ai_score else 2)

    if cur_state == "MATCH_RESULT":
        player_won  = player_score >= win_target
        winner_txt  = "YOU WIN THE MATCH!" if player_won else "AI WINS THE MATCH!"
        winner_col  = COL_GREEN if player_won else COL_RED
        pulse = 0.7 + 0.3 * abs(math.sin(t * math.pi * 2.0))
        pc2   = tuple(min(255, int(c * pulse)) for c in winner_col)
        draw_centered_text(frame, winner_txt, cy - _ix(h * 0.18),
                           0.90, pc2, thickness=3, outline=5)
        draw_centered_text(frame, score_text,
                           cy - _ix(h * 0.04), 0.60, COL_YELLOW,
                           thickness=2, outline=3)
        msg     = "You successfully fooled the AI!" if player_won else "The AI read your patterns. Train harder!"
        msg_col = COL_GREEN if player_won else COL_RED
        draw_centered_text(frame, msg,
                           cy + _ix(h * 0.08), 0.40, msg_col,
                           thickness=1, outline=2)
        pulse2 = 0.5 + 0.5 * abs(math.sin(t * math.pi * 1.2))
        pcol2  = tuple(min(255, int(c * pulse2)) for c in COL_GREEN)
        draw_centered_text(frame, "Press ENTER to play again  |  ESC for menu",
                           cy + _ix(h * 0.20), 0.44, pcol2,
                           thickness=2, outline=3)
        return

    if prediction and cur_state in ("COUNTDOWN", "SHOOT_WINDOW", "ROUND_RESULT"):
        pan_w  = _ix(w * 0.44)
        px1    = (w - pan_w) // 2; px2 = px1 + pan_w
        pan_y1 = _ix(h * 0.18);   pan_y2 = _ix(h * 0.72)
        pred_col = get_gesture_color(prediction)
        draw_panel(frame, px1, pan_y1, px2, pan_y2,
                   fill=(6, 6, 18), alpha=0.88, border=pred_col, border_thickness=2)
        draw_status_chip(frame, "AI PREDICTS", pan_y1 + _ix(h * 0.02), COL_YELLOW)

        if cur_state == "ROUND_RESULT":
            half = (px2 - px1) // 2
            _draw_gesture_icon(frame, prediction,
                (px1+_ix(8), pan_y1+_ix(h*0.12), px1+half-_ix(4), pan_y1+_ix(h*0.55)))
            draw_centered_text_in_rect(frame, prediction,
                (px1, pan_y1+_ix(h*0.56), px1+half, pan_y1+_ix(h*0.67)),
                base_scale=0.46, color=pred_col, thickness=2, outline=2)
            if player_gest and player_gest != "Unknown":
                p_col = get_gesture_color(player_gest)
                _draw_gesture_icon(frame, player_gest,
                    (px1+half+_ix(4), pan_y1+_ix(h*0.12), px2-_ix(8), pan_y1+_ix(h*0.55)))
                draw_centered_text_in_rect(frame, player_gest,
                    (px1+half, pan_y1+_ix(h*0.56), px2, pan_y1+_ix(h*0.67)),
                    base_scale=0.46, color=p_col, thickness=2, outline=2)
            cv2.line(frame, (px1+half, pan_y1+_ix(h*0.08)),
                     (px1+half, pan_y1+_ix(h*0.68)), (80,80,100), 1)
            ban_col = COL_GREEN if banner == "FOOLED IT!" else COL_RED
            pulse2  = 0.7 + 0.3 * abs(math.sin(t * math.pi * 2.5))
            bc2     = tuple(min(255, int(c * pulse2)) for c in ban_col)
            draw_centered_text_in_rect(frame, banner,
                (px1+4, pan_y2-_ix(h*0.14), px2-4, pan_y2-_ix(h*0.02)),
                base_scale=0.72, color=bc2, thickness=3, outline=4)
            if insight:
                i_sc = get_fit_scale(insight, _ix(w*0.70), base_scale=0.30, thickness=1, min_scale=0.22)
                draw_centered_text(frame, insight,
                                   pan_y2+_ix(h*0.04), i_sc, COL_TEXT_DIM, thickness=1, outline=1)
        else:
            pulse3 = 0.75 + 0.25 * abs(math.sin(t * math.pi * 2.0))
            pc3    = tuple(min(255, int(c * pulse3)) for c in pred_col)
            _draw_gesture_icon(frame, prediction,
                (px1+_ix(pan_w*0.10), pan_y1+_ix(h*0.12),
                 px2-_ix(pan_w*0.10), pan_y1+_ix(h*0.58)))
            draw_centered_text_in_rect(frame, prediction,
                (px1, pan_y1+_ix(h*0.59), px2, pan_y1+_ix(h*0.72)),
                base_scale=0.66, color=pc3, thickness=2, outline=3)
            if cur_state == "SHOOT_WINDOW":
                draw_centered_text_in_rect(frame, "THROW NOW!",
                    (px1, pan_y2-_ix(h*0.14), px2, pan_y2-_ix(h*0.02)),
                    base_scale=0.68, color=COL_GREEN, thickness=3, outline=4)
            else:
                draw_centered_text_in_rect(frame, "Don't throw this!",
                    (px1, pan_y2-_ix(h*0.12), px2, pan_y2-_ix(h*0.02)),
                    base_scale=0.36, color=COL_TEXT_DIM, thickness=1, outline=2)
    elif cur_state == "WAITING_FOR_ROCK":
        draw_centered_text(frame, "MAKE A FIST TO START", cy-_ix(h*0.06),
                           0.68, COL_CYAN, thickness=2, outline=4)
        draw_centered_text(frame, "Beat the pump 4 times, then throw",
                           cy+_ix(h*0.05), 0.38, COL_TEXT_DIM, thickness=1, outline=2)
        draw_centered_text(frame, "Throw anything EXCEPT what the AI predicts to score",
                           cy+_ix(h*0.12), 0.34, COL_TEXT_ACCENT, thickness=1, outline=2)

    if cur_state in ("COUNTDOWN", "SHOOT_WINDOW"):
        dot_y  = h - _ix(h * 0.12)
        dot_sp = _ix(w * 0.06)
        dot_x0 = w // 2 - dot_sp * 2
        for bi in range(4):
            bx  = dot_x0 + bi * dot_sp
            act = bi < beat_count or (bi == 3 and cur_state == "SHOOT_WINDOW")
            col = COL_RED if (bi == 3 and cur_state == "SHOOT_WINDOW") else                   (COL_CYAN if act else (50, 50, 70))
            cv2.circle(frame, (bx, dot_y), _ix(w*0.012), col, -1 if act else 2)


# ============================================================
# GESTURE TRAINER / REHAB
# ============================================================

def draw_gesture_rehab_view(frame, game_state):
    """Gesture Trainer: guided hold exercises with accuracy tracking."""
    w, h = frame.shape[1], frame.shape[0]
    t    = time.monotonic()
    draw_panel(frame, 0, 0, w-1, h-1, fill=COL_BG_DARK, alpha=0.45,
               border=COL_BG_DARK, border_thickness=0)

    cur_state   = game_state.get("state", "INTRO")
    target      = game_state.get("target", "")
    held        = game_state.get("held_gesture", "")
    dwell_pct   = game_state.get("dwell_pct", 0.0)
    step        = game_state.get("step", 0)
    total_steps = game_state.get("total_steps", 9)
    completed   = game_state.get("completed", 0)
    missed      = game_state.get("missed", 0)
    accuracy    = game_state.get("accuracy", 1.0)
    HOLD_SECS   = game_state.get("hold_secs", 3)
    REPS_PER_GEST = game_state.get("reps_per_gest", 3)
    cx, cy      = w // 2, h // 2

    draw_top_bar(frame, "GESTURE TRAINER",
                 f"Step {step+1}/{total_steps}  |  Accuracy: {accuracy:.0%}  |  Q Quit")
    draw_bottom_bar(frame,
        "Press ENTER or say START to begin" if cur_state == "INTRO"
        else "Hold each gesture for 3 seconds  |  Say BACK to exit  |  Q Quit")

    if cur_state == "INTRO":
        px1, px2 = _ix(w * 0.04), _ix(w * 0.96)
        py1, py2 = _ix(h * 0.11), _ix(h * 0.91)
        pw2, ph2 = px2 - px1, py2 - py1
        draw_panel(frame, px1, py1, px2, py2,
                   fill=(6, 10, 22), alpha=0.95,
                   border=(60, 100, 120), border_thickness=2)

        # Title - softer white, not neon yellow
        draw_centered_text_in_rect(frame, "GESTURE TRAINER",
            (px1, py1, px2, py1 + _ix(ph2 * 0.10)),
            base_scale=0.62, color=(200, 210, 220), thickness=2, outline=3)

        cv2.line(frame,
                 (px1 + _ix(pw2 * 0.04), py1 + _ix(ph2 * 0.11)),
                 (px2 - _ix(pw2 * 0.04), py1 + _ix(ph2 * 0.11)),
                 (60, 80, 100), 1)

        col1_x = px1 + _ix(pw2 * 0.05)
        col2_x = px1 + _ix(pw2 * 0.53)
        gap    = _ix(ph2 * 0.065)   # single line spacing, no doubles
        txt_sc = 0.33               # larger body text
        lbl_sc = 0.36               # larger section headers

        # Soft section header colour - muted teal, not neon cyan
        HDR   = (100, 180, 190)
        BODY  = (180, 185, 195)   # soft grey-white, easy to read
        DIM   = (140, 148, 158)   # slightly dimmer for secondary info
        CITE  = (110, 130, 155)   # citation colour

        # Left column
        draw_outlined_text(frame, "WHY THIS WORKS",
            col1_x, py1 + _ix(ph2 * 0.15),
            lbl_sc, HDR, thickness=1, outline=1)

        left_lines = [
            "Repetitive targeted movement",
            "stimulates neuroplasticity - the",
            "brain's ability to rewire motor",
            "pathways after injury or disuse.",
            "",
            "Rock, Paper and Scissors each train",
            "distinct finger shapes, building",
            "flexion, extension, and independent",
            "finger control.",
            "",
            "Gesture-based exergames have shown",
            "improvements in dexterity and grip",
            "strength in clinical settings.",
            "(Brain Sciences, 2022)",
        ]
        ly = py1 + _ix(ph2 * 0.22)
        for line in left_lines:
            if line:
                col = CITE if line.startswith("(") else (BODY if "movement" in line or "ability" in line or "injury" in line else DIM)
                draw_outlined_text(frame, line, col1_x, ly,
                                   txt_sc, col, thickness=1, outline=1)
                ly += gap
            else:
                ly += _ix(ph2 * 0.022)

        # Vertical divider
        cv2.line(frame,
                 (px1 + _ix(pw2 * 0.50), py1 + _ix(ph2 * 0.13)),
                 (px1 + _ix(pw2 * 0.50), py2 - _ix(ph2 * 0.09)),
                 (40, 55, 70), 1)

        # Right column
        draw_outlined_text(frame, "THIS SESSION",
            col2_x, py1 + _ix(ph2 * 0.15),
            lbl_sc, HDR, thickness=1, outline=1)

        right_lines = [
            (f"{REPS_PER_GEST} reps each of Rock, Paper,", BODY),
            (f"Scissors in randomised order", BODY),
            (f"({REPS_PER_GEST * 3} gestures total).", BODY),
            ("", None),
            ("Hold each gesture steady for", DIM),
            (f"{HOLD_SECS:.0f} seconds to complete a rep.", DIM),
            ("", None),
            ("Tips for best form:", (160, 175, 130)),
            ("Keep your elbow relaxed", DIM),
            ("Extend all fingers for Paper", DIM),
            ("Curl into a tight fist for Rock", DIM),
            ("Spread two fingers for Scissors", DIM),
            ("", None),
            ("Your accuracy is shown at the end.", DIM),
        ]
        ry = py1 + _ix(ph2 * 0.22)
        for text, col in right_lines:
            if text:
                draw_outlined_text(frame, text, col2_x, ry,
                                   txt_sc, col, thickness=1, outline=1)
                ry += gap
            else:
                ry += _ix(ph2 * 0.022)

        return

    if cur_state == "COMPLETE":
        done_col = COL_GREEN if accuracy >= 0.8 else COL_YELLOW
        draw_centered_text(frame, "SESSION COMPLETE!", cy-_ix(h*0.16),
                           0.90, done_col, thickness=3, outline=5)
        draw_centered_text(frame, f"Accuracy: {accuracy:.0%}",
                           cy-_ix(h*0.04), 0.70, COL_YELLOW, thickness=2, outline=3)
        draw_centered_text(frame, f"Completed: {completed}   Missed: {missed}",
                           cy+_ix(h*0.08), 0.44, COL_TEXT_ACCENT, thickness=1, outline=2)
        msg = "Excellent form!" if accuracy >= 0.9 else ("Good work! Keep practising." if accuracy >= 0.7 else "Keep at it - accuracy improves with practice.")
        msg_col = COL_GREEN if accuracy >= 0.9 else (COL_YELLOW if accuracy >= 0.7 else COL_RED)
        draw_centered_text(frame, msg, cy+_ix(h*0.18), 0.40, msg_col, thickness=1, outline=2)
        return

    if cur_state == "REST":
        log = game_state.get("session_log", [])
        was_ok = log[-1]["success"] if log else True
        flash_col = COL_GREEN if was_ok else COL_RED
        pulse = 0.7 + 0.3 * abs(math.sin(t * math.pi * 3.0))
        pc = tuple(min(255, int(c * pulse)) for c in flash_col)
        draw_centered_text(frame, "Correct!" if was_ok else "Missed", cy, 0.80, pc, thickness=2, outline=4)
        return

    if cur_state == "EXERCISE" and target:
        target_col = get_gesture_color(target)
        is_holding = (held == target)
        draw_status_chip(frame, "SHOW THIS GESTURE", _ix(h*0.13), COL_CYAN)
        _draw_gesture_icon(frame, target,
            (_ix(w*0.28), _ix(h*0.20), _ix(w*0.72), _ix(h*0.62)))
        draw_centered_text_in_rect(frame, target,
            (0, _ix(h*0.64), w, _ix(h*0.73)),
            base_scale=0.72, color=target_col, thickness=2, outline=3)

        bx1 = _ix(w*0.15); bx2 = _ix(w*0.85)
        by  = _ix(h*0.78);  bh2 = _ix(h*0.022)
        cv2.rectangle(frame, (bx1, by), (bx2, by+bh2), (30,30,30), -1)
        if dwell_pct > 0:
            bar_col = COL_GREEN if is_holding else COL_YELLOW
            cv2.rectangle(frame, (bx1, by), (bx1+int((bx2-bx1)*dwell_pct), by+bh2), bar_col, -1)
        cv2.rectangle(frame, (bx1, by), (bx2, by+bh2), (80,80,100), 1)

        if is_holding and dwell_pct > 0:
            draw_centered_text(frame, "Hold it...", by+bh2+_ix(h*0.025), 0.40, COL_GREEN, thickness=1, outline=2)
        elif held and held != target:
            draw_centered_text(frame, f"Showing: {held}", by+bh2+_ix(h*0.025), 0.38, get_gesture_color(held), thickness=1, outline=2)
        else:
            draw_centered_text(frame, "Show the gesture above", by+bh2+_ix(h*0.025), 0.36, COL_TEXT_DIM, thickness=1, outline=2)

        dot_y  = _ix(h*0.88)
        dot_sp = min(_ix(w*0.06), _ix(w*0.85/total_steps))
        start_x = w//2 - (total_steps*dot_sp)//2
        log = game_state.get("session_log", [])
        for i in range(total_steps):
            dx = start_x + i*dot_sp
            if i < len(log):
                cv2.circle(frame, (dx, dot_y), _ix(w*0.010),
                           COL_GREEN if log[i]["success"] else COL_RED, -1)
            elif i == step:
                pulse2 = 0.6 + 0.4*abs(math.sin(t*math.pi*2))
                pc2    = tuple(min(255, int(c*pulse2)) for c in target_col)
                cv2.circle(frame, (dx, dot_y), _ix(w*0.012), pc2, 2)
            else:
                cv2.circle(frame, (dx, dot_y), _ix(w*0.010), (50,50,60), 1)




def draw_arcade_snake_view(frame, game_state, tracker_state=None):
    """Gesture Snake — Rock=straight, Scissors=left, Paper=right."""
    w, h = frame.shape[1], frame.shape[0]
    t    = time.monotonic()

    state         = game_state.get("state", "INTRO")
    snake         = game_state.get("snake", [])
    apple         = game_state.get("apple", (0, 0))
    score         = game_state.get("score", 0)
    high_score    = game_state.get("high_score", 0)
    session_score = game_state.get("session_score", 0)
    is_new_record = game_state.get("is_new_record", False)
    leaderboard   = game_state.get("leaderboard", [])
    grid_w        = game_state.get("grid_w", 20)
    grid_h        = game_state.get("grid_h", 15)
    last_gest     = game_state.get("last_gesture", "Unknown")
    cx, cy        = w // 2, h // 2

    draw_panel(frame, 0, 0, w - 1, h - 1, fill=COL_BG_DARK, alpha=0.55,
               border=COL_BG_DARK, border_thickness=0)
    draw_top_bar(frame, "GESTURE SNAKE",
                 f"Score: {score}  |  Best: {high_score}  |  Q Quit")
    draw_bottom_bar(frame,
        "Rock=Straight  |  Scissors=Left  |  Paper=Right  |  Gesture only  |  Q Quit")

    # ── INTRO ──────────────────────────────────────────────────────────────
    if state == "INTRO":
        draw_centered_text(frame, "GESTURE SNAKE",
                           cy - _ix(h * 0.28), 0.90, COL_GREEN,
                           thickness=3, outline=5)
        draw_centered_text(frame, "Rock=Straight   Scissors=Left   Paper=Right",
                           cy - _ix(h * 0.14), 0.38, COL_TEXT_ACCENT,
                           thickness=1, outline=2)

        # Leaderboard panel
        if leaderboard:
            lbx1 = _ix(w * 0.30); lbx2 = _ix(w * 0.70)
            lby1 = cy - _ix(h * 0.08); lby2 = cy + _ix(h * 0.28)
            draw_panel(frame, lbx1, lby1, lbx2, lby2,
                       fill=(6, 18, 8), alpha=0.88,
                       border=COL_GREEN, border_thickness=2)
            draw_centered_text_in_rect(frame, "HIGH SCORES",
                (lbx1, lby1, lbx2, lby1 + _ix(h * 0.06)),
                base_scale=0.44, color=COL_GREEN, thickness=2, outline=2)
            medal = ["1st", "2nd", "3rd", "4th", "5th"]
            for i, entry in enumerate(leaderboard[:5]):
                ey = lby1 + _ix(h * 0.08) + i * _ix(h * 0.058)
                col = COL_YELLOW if i == 0 else COL_TEXT_DIM
                draw_outlined_text(frame,
                    f"{medal[i]}  {entry['score']:>5}   {entry['date']}",
                    lbx1 + _ix(w * 0.04), ey + _ix(h * 0.022),
                    0.32, col, thickness=1, outline=1)
        else:
            draw_centered_text(frame, "No scores yet — be the first!",
                               cy + _ix(h * 0.02), 0.38, COL_TEXT_DIM,
                               thickness=1, outline=1)

        pulse = 0.5 + 0.5 * abs(math.sin(t * math.pi * 1.2))
        pc    = tuple(min(255, int(c * pulse)) for c in COL_GREEN)
        draw_centered_text(frame, "Make a FIST to start",
                           cy + _ix(h * 0.34), 0.52, pc,
                           thickness=2, outline=3)
        return

    # ── Grid helpers ───────────────────────────────────────────────────────
    grid_x1 = _ix(w * 0.05); grid_x2 = _ix(w * 0.95)
    grid_y1 = _ix(h * 0.13); grid_y2 = _ix(h * 0.88)
    cell_w  = (grid_x2 - grid_x1) // grid_w
    cell_h  = (grid_y2 - grid_y1) // grid_h

    def cell_to_px(gx, gy):
        return (grid_x1 + gx * cell_w + cell_w // 2,
                grid_y1 + gy * cell_h + cell_h // 2)

    # Grid lines
    for gx in range(grid_w + 1):
        cv2.line(frame, (grid_x1 + gx * cell_w, grid_y1),
                 (grid_x1 + gx * cell_w, grid_y2), (20, 25, 35), 1)
    for gy in range(grid_h + 1):
        cv2.line(frame, (grid_x1, grid_y1 + gy * cell_h),
                 (grid_x2, grid_y1 + gy * cell_h), (20, 25, 35), 1)

    # Apple
    ax, ay = cell_to_px(*apple)
    ar = max(3, cell_w // 2 - 2)
    ap = 0.80 + 0.20 * abs(math.sin(t * math.pi * 2))
    ac = tuple(min(255, int(c * ap)) for c in (60, 220, 60))
    cv2.circle(frame, (ax, ay), ar, ac, -1)
    cv2.circle(frame, (ax, ay), ar, (100, 255, 100), 1)

    # Snake
    for i, (gx, gy) in enumerate(snake):
        px, py = cell_to_px(gx, gy)
        r = max(2, cell_w // 2 - 1)
        if i == 0:
            cv2.rectangle(frame, (px-r, py-r), (px+r, py+r), COL_CYAN, -1)
            cv2.rectangle(frame, (px-r, py-r), (px+r, py+r), (200,255,255), 1)
        else:
            fade = max(0.35, 1.0 - i * 0.035)
            cv2.rectangle(frame,
                (px-r+1, py-r+1), (px+r-1, py+r-1),
                tuple(int(c * fade) for c in COL_CYAN), -1)

    # Gesture HUD
    display_gest = game_state.get("voted_gesture") or last_gest
    if display_gest in ("Rock", "Paper", "Scissors"):
        action = {"Rock": "STRAIGHT", "Paper": "RIGHT >", "Scissors": "< LEFT"}
        draw_outlined_text(frame,
            f"{display_gest}  {action[display_gest]}",
            _ix(w * 0.04), grid_y2 + _ix(h * 0.02),
            0.42, get_gesture_color(display_gest), thickness=1, outline=2)

    # ── GAME OVER ──────────────────────────────────────────────────────────
    if state == "GAME_OVER":
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # Title
        title_col = COL_YELLOW if is_new_record else COL_RED
        title_txt = "NEW RECORD!" if is_new_record else "GAME OVER"
        draw_centered_text(frame, title_txt,
                           cy - _ix(h * 0.28), 0.90, title_col,
                           thickness=3, outline=5)

        # Score
        draw_centered_text(frame, f"Score: {session_score}",
                           cy - _ix(h * 0.14), 0.68, COL_YELLOW,
                           thickness=2, outline=3)
        if is_new_record:
            draw_centered_text(frame, "Personal best!",
                               cy - _ix(h * 0.04), 0.44, COL_GREEN,
                               thickness=1, outline=2)
        else:
            draw_centered_text(frame, f"Best: {high_score}",
                               cy - _ix(h * 0.04), 0.44, COL_TEXT_DIM,
                               thickness=1, outline=2)

        # Leaderboard
        if leaderboard:
            lbx1 = _ix(w * 0.30); lbx2 = _ix(w * 0.70)
            lby1 = cy + _ix(h * 0.04); lby2 = cy + _ix(h * 0.38)
            draw_panel(frame, lbx1, lby1, lbx2, lby2,
                       fill=(6, 18, 8), alpha=0.88,
                       border=COL_GREEN, border_thickness=2)
            draw_centered_text_in_rect(frame, "TOP SCORES",
                (lbx1, lby1, lbx2, lby1 + _ix(h * 0.06)),
                base_scale=0.40, color=COL_GREEN, thickness=2, outline=2)
            medal = ["1st", "2nd", "3rd", "4th", "5th"]
            for i, entry in enumerate(leaderboard[:5]):
                ey = lby1 + _ix(h * 0.08) + i * _ix(h * 0.052)
                is_this = (entry["score"] == session_score and is_new_record and i == 0)
                col = COL_YELLOW if is_this else (COL_TEXT_ACCENT if i == 0 else COL_TEXT_DIM)
                draw_outlined_text(frame,
                    f"{medal[i]}  {entry['score']:>5}   {entry['date']}",
                    lbx1 + _ix(w * 0.04), ey + _ix(h * 0.020),
                    0.30, col, thickness=1, outline=1)

        pulse = 0.5 + 0.5 * abs(math.sin(t * math.pi * 1.2))
        pc    = tuple(min(255, int(c * pulse)) for c in COL_GREEN)
        draw_centered_text(frame, "Make a FIST to play again",
                           cy + _ix(h * 0.42), 0.48, pc,
                           thickness=2, outline=3)


def draw_rpsls_tutorial_screen(frame, step=0, hand_state=None):
    """
    RPSLS How-to-Play tutorial.  6 steps:
      0  - Welcome / overview of 5 gestures
      1  - Rock and Scissors gestures (live detection)
      2  - Paper and Lizard gestures (live detection)
      3  - Spock gesture (live detection)
      4  - Win / lose rules wheel
      5  - Gesture diagnostic (all 5, live feedback)
    """
    w, h = frame.shape[1], frame.shape[0]
    t    = time.monotonic()
    N_STEPS = 6

    draw_panel(frame, 0, 0, w - 1, h - 1, fill=(4, 6, 16), alpha=0.70,
               border=(4, 6, 16), border_thickness=0)
    draw_top_bar(frame, "RPSLS  HOW TO PLAY",
                 f"Step {step+1}/{N_STEPS}  |  W/D or Enter = Next  |  A/S = Back  |  ESC Quit")

    cx = w // 2
    lm = _menu_layout(frame)
    px1, py1, px2, py2 = lm["panel"]
    ph = py2 - py1
    pw = px2 - px1

    # Progress dots
    dot_y = py2 - _ix(ph * 0.04)
    dot_sp = _ix(w * 0.025)
    for i in range(N_STEPS):
        dx = cx + (i - N_STEPS // 2) * dot_sp + dot_sp // 2
        col = COL_CYAN if i <= step else (40, 40, 60)
        cv2.circle(frame, (dx, dot_y), _ix(w * 0.007), col, -1 if i <= step else 2)

    # ── Current step content ─────────────────────────────────────────────
    if step == 0:
        # Overview
        draw_centered_text(frame, "ROCK  PAPER  SCISSORS", py1 + _ix(ph * 0.06),
                           0.62, COL_TEXT, thickness=2, outline=3)
        draw_centered_text(frame, "LIZARD  SPOCK", py1 + _ix(ph * 0.14),
                           0.72, COL_YELLOW, thickness=2, outline=3)

        draw_centered_text(frame, "5 gestures. 10 outcomes. Classic RPS plus two more.",
                           py1 + _ix(ph * 0.24), 0.44, COL_TEXT_ACCENT, thickness=1, outline=2)
        draw_centered_text(frame, "Each gesture beats exactly two others",
                           py1 + _ix(ph * 0.31), 0.42, COL_TEXT_DIM, thickness=1, outline=2)
        draw_centered_text(frame, "and loses to exactly two others.",
                           py1 + _ix(ph * 0.37), 0.42, COL_TEXT_DIM, thickness=1, outline=2)

        # Row of all 5 icons
        gestures  = ["Rock", "Paper", "Scissors", "Lizard", "Spock"]
        icon_w    = _ix(pw * 0.16)
        icon_gap  = _ix(pw * 0.03)
        total_row = len(gestures) * (icon_w + icon_gap) - icon_gap
        start_x   = cx - total_row // 2
        iy1 = py1 + _ix(ph * 0.46)
        iy2 = iy1  + _ix(ph * 0.30)
        for i, g in enumerate(gestures):
            gx1 = start_x + i * (icon_w + icon_gap)
            gx2 = gx1 + icon_w
            col = _RPSLS_COLS.get(g, COL_TEXT_DIM)
            draw_panel(frame, gx1, iy1, gx2, iy2, fill=(8, 8, 20), alpha=0.85,
                       border=col, border_thickness=2)
            _draw_rpsls_gesture_icon(frame, g, (gx1 + 4, iy1 + 4, gx2 - 4, iy2 - _ix(ph * 0.06)))

        draw_centered_text(frame, "Press Enter or D for next step",
                           py2 - _ix(ph * 0.10), 0.38, COL_TEXT_DIM, thickness=1, outline=2)

    elif step in (1, 2, 3):
        # Gesture teach steps with live detection feedback.
        # Each gesture gets its own card; description lives INSIDE the card
        # below the icon; title drawn once in the card's colour.
        step_gestures = {
            1: [
                ("Rock",     "Closed fist.\nAll fingers curled in."),
                ("Scissors", "Index + middle up.\nRing + pinky curled.\nThumb position: either."),
            ],
            2: [
                ("Paper",    "Flat open palm.\nAll fingers extended."),
                ("Lizard",   "Sock-puppet mouth.\nFingers forward, thumb below.\nHold hand sideways."),
            ],
            3: [
                ("Spock",    "Vulcan salute.\n4 fingers extended.\nWide gap: middle + ring."),
            ],
        }
        pairs = step_gestures[step]
        n = len(pairs)

        detected = "Unknown"
        if hand_state is not None:
            detected = hand_state.get("raw_gesture", "Unknown")

        # ── Layout ────────────────────────────────────────────────────────
        # Cards: for 2 gestures use 46% width each with 4% gap between.
        # For 1 gesture (Spock) use 54% width centred.
        gap = _ix(pw * 0.04)
        if n == 2:
            card_w = _ix(pw * 0.46)
            total  = 2 * card_w + gap
        else:
            card_w = _ix(pw * 0.54)
            total  = card_w
        start_x = cx - total // 2

        # Card spans the full content area; icon takes top 52%, name strip 10%,
        # description takes remaining ~30% at bottom.
        card_y1 = py1 + _ix(ph * 0.04)
        card_y2 = py2 - _ix(ph * 0.10)   # leave room for progress dots
        card_h  = card_y2 - card_y1

        icon_zone_frac  = 0.52   # fraction of card height for icon
        name_zone_frac  = 0.10   # fraction for gesture name strip
        desc_zone_frac  = 0.30   # fraction for description lines
        # Positions within card (relative to card_y1)
        icon_y1 = card_y1 + _ix(card_h * 0.02)
        icon_y2 = card_y1 + _ix(card_h * icon_zone_frac)
        name_y1 = icon_y2
        name_y2 = name_y1 + _ix(card_h * name_zone_frac)
        desc_y1 = name_y2 + _ix(card_h * 0.01)

        for i, (g, desc) in enumerate(pairs):
            gx1 = start_x + i * (card_w + gap)
            gx2 = gx1 + card_w

            is_det = (detected == g)
            col = _RPSLS_COLS.get(g, COL_TEXT_DIM)
            border_col = COL_GREEN if is_det else col
            fill_col   = (6, 22, 6) if is_det else (8, 8, 20)

            # Card panel
            draw_panel(frame, gx1, card_y1, gx2, card_y2,
                       fill=fill_col, alpha=0.92,
                       border=border_col, border_thickness=3 if is_det else 2)

            # Icon  -  no built-in name (show_name=False)
            _draw_rpsls_gesture_icon(frame, g,
                (gx1 + _ix(card_w * 0.06), icon_y1,
                 gx2 - _ix(card_w * 0.06), icon_y2),
                show_name=False)

            # Gesture name strip
            name_col = COL_GREEN if is_det else col
            draw_centered_text_in_rect(frame, g,
                (gx1, name_y1, gx2, name_y2),
                base_scale=0.50, color=name_col, thickness=2, outline=2)

            # Separator line
            cv2.line(frame,
                     (gx1 + _ix(card_w * 0.05), name_y2 + 2),
                     (gx2 - _ix(card_w * 0.05), name_y2 + 2),
                     tuple(c // 2 for c in col), 1)

            # Description  -  one line per \n entry
            lines = desc.split("\n")
            line_h = _ix(card_h * 0.078)
            dy = desc_y1 + _ix(card_h * 0.015)
            for line in lines:
                draw_centered_text_in_rect(frame, line,
                    (gx1 + 4, dy, gx2 - 4, dy + line_h),
                    base_scale=0.34,
                    color=COL_TEXT_DIM if not is_det else (160, 220, 160),
                    thickness=1, outline=1)
                dy += line_h + _ix(card_h * 0.005)

        # Single status line above cards (not per-card)
        any_det = any(detected == g for g, _ in pairs)
        status  = f"DETECTED: {detected}!" if any_det else "Show a gesture to the camera"
        s_col   = COL_GREEN if any_det else COL_TEXT_DIM
        draw_centered_text(frame, status,
                           py1 + _ix(ph * 0.01),
                           0.40, s_col, thickness=1, outline=2)

    elif step == 4:
        # Rules wheel - who beats who
        draw_centered_text(frame, "WHO BEATS WHO", py1 + _ix(ph * 0.04),
                           0.68, COL_YELLOW, thickness=2, outline=3)

        rules_compact = [
            ("Scissors", "cuts",       "Paper"),
            ("Paper",    "covers",     "Rock"),
            ("Rock",     "crushes",    "Lizard"),
            ("Lizard",   "poisons",    "Spock"),
            ("Spock",    "smashes",    "Scissors"),
            ("Scissors", "decapitates","Lizard"),
            ("Lizard",   "eats",       "Paper"),
            ("Paper",    "disproves",  "Spock"),
            ("Spock",    "vaporizes",  "Rock"),
            ("Rock",     "crushes",    "Scissors"),
        ]

        # Two columns
        col_w    = _ix(pw * 0.46)
        left_x   = px1 + _ix(pw * 0.02)
        right_x  = cx + _ix(pw * 0.04)
        row_h    = _ix(ph * 0.074)
        top_y    = py1 + _ix(ph * 0.14)

        for idx, (winner, verb, loser) in enumerate(rules_compact):
            col_x  = left_x if idx < 5 else right_x
            ry     = top_y + (idx % 5) * row_h
            wcol   = _RPSLS_COLS.get(winner, COL_TEXT_DIM)
            lcol   = _RPSLS_COLS.get(loser,  COL_TEXT_DIM)
            line   = f"{winner} {verb} {loser}"
            # Winner in its colour, rest dimmed
            draw_outlined_text(frame, winner, col_x, ry, 0.42, wcol, thickness=1, outline=2)
            (ww, _), _ = cv2.getTextSize(winner, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            draw_outlined_text(frame, f" {verb} ", col_x + ww + 2, ry,
                               0.38, COL_TEXT_DIM, thickness=1, outline=1)
            (vw, _), _ = cv2.getTextSize(f" {verb} ", cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            draw_outlined_text(frame, loser, col_x + ww + vw + 4, ry,
                               0.42, lcol, thickness=1, outline=2)

    elif step == 5:
        # Live diagnostic - all 5 gestures
        draw_centered_text(frame, "GESTURE DIAGNOSTIC", py1 + _ix(ph * 0.04),
                           0.62, COL_CYAN, thickness=2, outline=3)
        draw_centered_text(frame, "Try all five gestures - watch the detector respond",
                           py1 + _ix(ph * 0.12), 0.40, COL_TEXT_DIM, thickness=1, outline=2)

        detected = hand_state.get("raw_gesture", "Unknown") if hand_state else "Unknown"

        gestures = ["Rock", "Paper", "Scissors", "Lizard", "Spock"]
        icon_w   = _ix(pw * 0.16)
        icon_gap = _ix(pw * 0.025)
        total_row = len(gestures) * (icon_w + icon_gap) - icon_gap
        start_x  = cx - total_row // 2
        iy1 = py1 + _ix(ph * 0.20)
        iy2 = iy1  + _ix(ph * 0.44)

        for i, g in enumerate(gestures):
            gx1 = start_x + i * (icon_w + icon_gap)
            gx2 = gx1 + icon_w
            is_det   = (detected == g)
            base_col = _RPSLS_COLS.get(g, COL_TEXT_DIM)
            if is_det:
                pulse = 0.7 + 0.3 * abs(math.sin(t * math.pi * 3))
                col   = tuple(min(255, int(c * pulse)) for c in base_col)
            else:
                col = tuple(c // 3 for c in base_col)
            border_col = base_col if is_det else tuple(c // 2 for c in base_col)
            fill_col   = (8, 25, 8) if is_det else (8, 8, 16)
            draw_panel(frame, gx1, iy1, gx2, iy2, fill=fill_col, alpha=0.90,
                       border=border_col, border_thickness=3 if is_det else 1)
            _draw_rpsls_gesture_icon(frame, g,
                (gx1 + 4, iy1 + 4, gx2 - 4, iy2 - _ix(ph * 0.04)))

        if detected in gestures:
            draw_centered_text(frame, f"Detected: {detected}",
                               iy2 + _ix(ph * 0.06), 0.56, _RPSLS_COLS.get(detected, COL_TEXT),
                               thickness=2, outline=3)
        else:
            draw_centered_text(frame, "No hand detected - show your hand",
                               iy2 + _ix(ph * 0.06), 0.42, COL_TEXT_DIM, thickness=1, outline=2)

        # Orientation tip
        draw_centered_text(frame, "Hold hand sideways to camera  |  Side mode active",
                           iy2 + _ix(ph * 0.15), 0.36, COL_YELLOW, thickness=1, outline=2)

        draw_centered_text(frame, "Press Enter to start playing!",
                           py2 - _ix(ph * 0.10), 0.44, COL_GREEN, thickness=1, outline=2)

    draw_bottom_bar(frame,
        "A/S = Back  |  D/Enter = Next  |  ESC = Back to menu")


def draw_rpsls_side_notice(frame, was_front_on=False, confirmed_gesture="Unknown",
                            ticked=None, dwell_pct=0.0):
    """
    RPSLS orientation notice screen.
    Shows all 5 gestures as cards. Player holds each for 1.5s to tick it off.
    Enter skips directly to the game.
    """
    if ticked is None:
        ticked = set()

    w, h = frame.shape[1], frame.shape[0]
    t    = time.monotonic()

    draw_panel(frame, 0, 0, w - 1, h - 1, fill=(4, 6, 16), alpha=0.72,
               border=(30, 30, 50), border_thickness=0)
    draw_top_bar(frame, "ROCK PAPER SCISSORS LIZARD SPOCK",
                 "Hold each gesture to tick it off  |  Enter to start  |  ESC Back")

    # Orientation warning if player was using front-on
    if was_front_on:
        draw_centered_text(frame, "SIDE-ON VIEW REQUIRED FOR RPSLS",
                           _ix(h * 0.12), 0.44, COL_YELLOW, thickness=1, outline=2)
        draw_centered_text(frame, "Turn your hand 90 degrees so it faces sideways",
                           _ix(h * 0.18), 0.36, COL_TEXT_DIM, thickness=1, outline=1)

    GESTURES_5 = ["Rock", "Paper", "Scissors", "Lizard", "Spock"]
    _RPSLS_GESTURE_COLS = {
        "Rock":     COL_CYAN,
        "Paper":    COL_GREEN,
        "Scissors": COL_MAGENTA,
        "Lizard":   (80, 200, 80),
        "Spock":    (255, 200, 0),
    }

    # 5 cards in a row
    n       = len(GESTURES_5)
    card_w  = _ix(w * 0.16)
    card_h  = _ix(h * 0.42)
    gap     = _ix(w * 0.02)
    total_w = n * card_w + (n - 1) * gap
    start_x = (w - total_w) // 2
    card_y  = _ix(h * 0.25)

    for i, gest in enumerate(GESTURES_5):
        cx1 = start_x + i * (card_w + gap)
        cx2 = cx1 + card_w
        cy2 = card_y + card_h

        is_ticked  = gest in ticked
        is_current = (confirmed_gesture == gest and not is_ticked)
        col        = _RPSLS_GESTURE_COLS.get(gest, COL_TEXT_ACCENT)

        if is_ticked:
            fill   = tuple(min(255, int(c * 0.30)) for c in col)
            border = col
            bthick = 3
        elif is_current:
            fill   = tuple(min(255, int(c * 0.12)) for c in col)
            border = col
            bthick = 2
        else:
            fill   = (8, 10, 20)
            border = (50, 50, 70)
            bthick = 1

        draw_panel(frame, cx1, card_y, cx2, cy2,
                   fill=fill, alpha=0.92, border=border, border_thickness=bthick)

        # Gesture icon
        icon_pad = _ix(card_w * 0.12)
        _draw_gesture_icon(frame, gest,
            (cx1 + icon_pad, card_y + _ix(card_h * 0.08),
             cx2 - icon_pad, card_y + _ix(card_h * 0.60)))

        # Gesture name
        draw_centered_text_in_rect(frame, gest,
            (cx1, card_y + _ix(card_h * 0.62), cx2, card_y + _ix(card_h * 0.78)),
            base_scale=0.38, color=col if (is_ticked or is_current) else COL_TEXT_DIM,
            thickness=1, outline=1)

        # Tick or dwell bar
        bar_y  = card_y + _ix(card_h * 0.82)
        bar_x1 = cx1 + _ix(card_w * 0.08)
        bar_x2 = cx2 - _ix(card_w * 0.08)
        bar_h2 = _ix(h * 0.018)
        if is_ticked:
            cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + bar_h2), col, -1)
            draw_centered_text_in_rect(frame, "DONE",
                (cx1, bar_y - _ix(2), cx2, bar_y + bar_h2 + _ix(2)),
                base_scale=0.32, color=(0, 0, 0), thickness=1, outline=0)
        elif is_current and dwell_pct > 0:
            cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + bar_h2), (30, 30, 40), -1)
            fill_x = bar_x1 + int((bar_x2 - bar_x1) * dwell_pct)
            cv2.rectangle(frame, (bar_x1, bar_y), (fill_x, bar_y + bar_h2), col, -1)
            cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + bar_h2), col, 1)
        else:
            cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + bar_h2), (30, 30, 40), -1)
            cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + bar_h2), (50, 50, 60), 1)

    # Progress text
    n_done = len(ticked)
    prog_col = COL_GREEN if n_done == 5 else COL_TEXT_DIM
    draw_centered_text(frame, f"{n_done} / 5 gestures practiced",
                       card_y + card_h + _ix(h * 0.04), 0.40, prog_col,
                       thickness=1, outline=2)

    pulse = 0.5 + 0.5 * abs(math.sin(t * math.pi * 1.2))
    pcol  = tuple(min(255, int(c * pulse)) for c in COL_GREEN)
    draw_centered_text(frame, "Press ENTER when ready to play",
                       card_y + card_h + _ix(h * 0.12), 0.44, pcol,
                       thickness=2, outline=3)

    draw_bottom_bar(frame, "Hold each gesture 1.5s to tick off  |  Enter to start  |  ESC Back")


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

    draw_top_bar(frame, "RPSLS  -  vs AI",
                 f"{round_txt}  |  First to {win_target}  |  Q Quit")

    # -- Panel geometry ----------------------------------------------------
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

    # -- Live gesture display (player) -------------------------------------
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

    # -- AI panel ----------------------------------------------------------
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

    # -- Centre panel ------------------------------------------------------
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

    # -- Scrolling rule strip at bottom ------------------------------------
    strip_y = h - _ix(h*0.050)
    # Slowly scroll rules
    scroll_idx = int(t * 0.5) % len(_RPSLS_RULES)
    rule_txt = _RPSLS_RULES[scroll_idx] + "  |  " + _RPSLS_RULES[(scroll_idx+1) % len(_RPSLS_RULES)]
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

        # Pipeline: raw > stable > confirmed
        raw  = hs.get("raw_gesture", "-")
        stab = ts.get("stable_gesture",   "-")
        conf = ts.get("confirmed_gesture","-")
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

        _row("Wrist Y:",    f"{wrist_y:.3f}" if wrist_y else "-")
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
        reason = hs.get("reason_text", "-")
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





# ============================================================
# HAND GEOMETRY ENROLL VIEW
# ============================================================

# ── Load hand outline PNG (rendered from Hand2011.svg) ──────────────────────
import os as _os
_HAND_IMG_CACHE = {}   # cache: scale_key -> (bgr, alpha_norm)

def _load_hand_img():
    """
    Load hand_outline.png. If it has no alpha (e.g. exported from Preview),
    treat light pixels as transparent and keep only the dark hand lines.
    """
    import numpy as _np
    try:
        _here = _os.path.dirname(_os.path.abspath(__file__))
        path  = _os.path.join(_here, "hand_outline.png")
        from PIL import Image as _PILImage
        pil   = _PILImage.open(path).convert("RGBA")
        arr   = _np.array(pil, dtype=_np.float32)

        r, g, b, a = arr[:,:,0], arr[:,:,1], arr[:,:,2], arr[:,:,3]

        # Check if the image already has meaningful alpha
        has_alpha = (a < 200).sum() > (arr.shape[0] * arr.shape[1] * 0.05)

        if not has_alpha:
            # Flat white background — derive alpha from darkness.
            # The hand is black/dark lines on white — keep dark, remove light.
            lightness = (r + g + b) / 3.0
            # Alpha = how dark the pixel is (0=white→transparent, 255=black→opaque)
            # Soft curve: anything below lightness 230 starts becoming visible
            new_alpha = _np.clip((230.0 - lightness) / 180.0, 0.0, 1.0) * 255.0
            arr[:,:,3] = new_alpha
            print(f"[HandScan] Derived alpha from darkness. "
                  f"Opaque pixels: {int((new_alpha > 64).sum())}")
        else:
            print(f"[HandScan] Using existing alpha channel.")

        arr = arr.clip(0, 255).astype(_np.uint8)
        bgr   = arr[:, :, :3][:, :, ::-1].copy()   # RGB→BGR
        alpha = arr[:, :, 3].astype(_np.float32) / 255.0
        return bgr, alpha
    except Exception as e:
        print(f"[HandScan] Could not load hand_outline.png: {e}")
        return None, None

_HAND_IMG_RAW = None
_HAND_IMG_LOADED = False


def _draw_hand_silhouette(frame, cx, cy, scale, col, alpha=0.35, thickness=2, mirror=False):
    """
    Composite the hand_outline.png onto the frame, tinted with `col`.
    cx, cy  = pixel position of the palm centre target
    scale   = pixel height of the rendered hand
    col     = BGR tint colour
    alpha   = overall opacity (0..1)
    mirror  = True for left hand (flips image horizontally)
    """
    global _HAND_IMG_RAW, _HAND_IMG_LOADED
    import numpy as _np

    if not _HAND_IMG_LOADED:
        _HAND_IMG_RAW = _load_hand_img()
        _HAND_IMG_LOADED = True

    src_bgr, src_alpha = _HAND_IMG_RAW if _HAND_IMG_RAW[0] is not None else (None, None)

    if src_bgr is None:
        hw = int(scale * 0.45)
        cv2.rectangle(frame, (cx - hw, cy - scale), (cx + hw, cy + int(scale*0.2)),
                      col, thickness)
        return

    import cv2 as _cv2

    # Resize to target scale (height = scale)
    src_h, src_w = src_bgr.shape[:2]
    tgt_h = int(scale)
    tgt_w = int(src_w * tgt_h / src_h)

    cache_key = (tgt_h, tgt_w, mirror)
    if cache_key not in _HAND_IMG_CACHE:
        r_bgr   = _cv2.resize(src_bgr,   (tgt_w, tgt_h), interpolation=_cv2.INTER_AREA)
        r_alpha = _cv2.resize(src_alpha, (tgt_w, tgt_h), interpolation=_cv2.INTER_AREA)
        if mirror:
            r_bgr   = _np.fliplr(r_bgr).copy()
            r_alpha = _np.fliplr(r_alpha).copy()
        _HAND_IMG_CACHE[cache_key] = (r_bgr, r_alpha)
    r_bgr, r_alpha = _HAND_IMG_CACHE[cache_key]

    # Tint: paint the target colour wherever the hand lines are (alpha = darkness)
    # The source image has black lines — we want those to show as `col`
    tinted = _np.zeros_like(r_bgr)
    for i, c in enumerate(col[:3]):
        tinted[:, :, i] = c

    # Anchor: the palm centre of the SVG hand sits at ~65% of image height.
    # Map that point to cy so the hand is centred at the target position.
    palm_base_y = int(tgt_h * 0.65)
    paste_x = cx - tgt_w // 2
    paste_y = cy - palm_base_y

    # Clamp to frame bounds
    fh, fw = frame.shape[:2]
    src_x0 = max(0, -paste_x);      src_y0 = max(0, -paste_y)
    dst_x0 = max(0,  paste_x);      dst_y0 = max(0,  paste_y)
    dst_x1 = min(fw, paste_x + tgt_w); dst_y1 = min(fh, paste_y + tgt_h)
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)

    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return

    roi       = frame[dst_y0:dst_y1, dst_x0:dst_x1].astype(_np.float32)
    hand_crop = tinted[src_y0:src_y1, src_x0:src_x1].astype(_np.float32)
    a_crop    = r_alpha[src_y0:src_y1, src_x0:src_x1] * alpha

    a3 = a_crop[:, :, _np.newaxis]
    blended = roi * (1.0 - a3) + hand_crop * a3
    frame[dst_y0:dst_y1, dst_x0:dst_x1] = blended.clip(0, 255).astype(_np.uint8)


def draw_hand_enroll_view(frame, game_state, hand_state=None):
    """
    Hand geometry enrollment screen.
    Shows a hand silhouette that animates to a new position/size each round.
    Player holds their open hand inside it; system auto-captures geometry.
    """
    import math as _math
    import time as _time

    w, h = frame.shape[1], frame.shape[0]
    t    = _time.monotonic()

    fp_phase      = game_state.get("fp_phase",           "INTRO")
    fp_round      = game_state.get("fp_round",           0)
    fp_rounds_tgt  = game_state.get("fp_rounds_target",   20)
    fp_samples_needed = game_state.get("fp_samples_needed", 10)
    fp_capture    = game_state.get("fp_capture_pct",     0.0)
    fp_stab       = game_state.get("fp_stability_pct",   0.0)
    fp_rest       = game_state.get("fp_rest_pct",        0.0)
    fp_name       = game_state.get("fp_player_name",     "?")
    fp_samples    = game_state.get("fp_samples",         0)
    fp_hint       = game_state.get("fp_next_hint",       "")
    fp_quality    = game_state.get("fp_quality_reason",  "")
    fp_hand_side  = game_state.get("fp_hand_side",       "Unknown")
    hand_visible  = game_state.get("hand_visible",       False)
    hand_open     = game_state.get("hand_open",          False)
    sil_cx_n      = game_state.get("sil_cx_norm",        0.50)
    sil_cy_n      = game_state.get("sil_cy_norm",        0.54)
    sil_sf        = game_state.get("sil_scale_factor",   1.00)

    # Dark overlay
    draw_panel(frame, 0, 0, w - 1, h - 1, fill=(4, 6, 16), alpha=0.70,
               border=(30, 30, 50), border_thickness=0)
    draw_top_bar(frame, "HAND SCAN - ENROLLMENT",
                 f"Enrolling: {fp_name}  |  ESC Cancel")

    # ── RECOGNIZED / INSUFFICIENT result overlay ────────────────────────
    if fp_phase in ("RECOGNIZED", "INSUFFICIENT"):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

        recog_name = game_state.get("recog_name", None)
        recog_conf = game_state.get("recog_conf", 0.0)

        if fp_phase == "RECOGNIZED":
            draw_centered_text_in_rect(frame, "HAND SCAN COMPLETE",
                (0, _ix(h*0.18), w, _ix(h*0.30)),
                base_scale=0.80, color=(80, 220, 100), thickness=3, outline=5)
            draw_centered_text_in_rect(frame,
                f"{fp_name}'s hand geometry saved  ({fp_samples} samples)",
                (0, _ix(h*0.32), w, _ix(h*0.40)),
                base_scale=0.36, color=(160, 200, 160), thickness=1, outline=2)
            # Recognition test result
            draw_centered_text_in_rect(frame, "RECOGNITION TEST",
                (0, _ix(h*0.43), w, _ix(h*0.51)),
                base_scale=0.44, color=COL_TEXT_ACCENT, thickness=1, outline=2)
            if recog_name:
                name_col = COL_GREEN if recog_name == fp_name else COL_YELLOW
                draw_centered_text_in_rect(frame, recog_name,
                    (0, _ix(h*0.52), w, _ix(h*0.63)),
                    base_scale=0.90, color=name_col, thickness=3, outline=5)
                draw_centered_text_in_rect(frame, f"Confidence: {recog_conf:.0%}",
                    (0, _ix(h*0.64), w, _ix(h*0.71)),
                    base_scale=0.36, color=(120, 160, 120), thickness=1, outline=1)
        else:
            samples_got   = fp_samples
            samples_need  = fp_samples_needed
            samples_short = max(0, samples_need - samples_got)
            extra_rounds  = samples_short  # roughly one round per sample

            draw_centered_text_in_rect(frame, "MORE DATA NEEDED",
                (0, _ix(h*0.18), w, _ix(h*0.30)),
                base_scale=0.80, color=(220, 160, 40), thickness=3, outline=5)

            # Samples captured vs needed
            draw_centered_text_in_rect(frame,
                f"Captured {samples_got} of {samples_need} required samples  "
                f"({samples_short} more needed, approx {extra_rounds} more rounds)",
                (0, _ix(h*0.32), w, _ix(h*0.40)),
                base_scale=0.34, color=(200, 175, 110), thickness=1, outline=2)

            # Hand detected
            hand_label = fp_hand_side if fp_hand_side != "Unknown" else "hand (detecting...)"
            draw_centered_text_in_rect(frame,
                f"Detected: {hand_label} hand",
                (0, _ix(h*0.41), w, _ix(h*0.48)),
                base_scale=0.34, color=(140, 180, 210), thickness=1, outline=1)

            draw_centered_text_in_rect(frame,
                "Keep your hand flat, fingers spread, palm facing the camera.",
                (0, _ix(h*0.50), w, _ix(h*0.57)),
                base_scale=0.32, color=(170, 150, 110), thickness=1, outline=1)
            draw_centered_text_in_rect(frame,
                "Make sure your full hand stays inside the outline each round.",
                (0, _ix(h*0.58), w, _ix(h*0.65)),
                base_scale=0.32, color=(140, 125, 90), thickness=1, outline=1)

        pulse = 0.5 + 0.5 * abs(_math.sin(t * _math.pi * 1.0))
        pc = tuple(min(255, int(c * pulse)) for c in (180, 220, 180))
        enter_label = "Press ENTER to return to Settings" if fp_phase == "RECOGNIZED" else "Press ENTER to try again"
        draw_centered_text_in_rect(frame, enter_label,
            (0, _ix(h*0.74), w, _ix(h*0.82)),
            base_scale=0.42, color=pc, thickness=1, outline=2)
        bottom = "ENTER - return to Settings  |  ESC cancel" if fp_phase == "RECOGNIZED" else "ENTER - try again  |  ESC cancel"
        draw_bottom_bar(frame, bottom)
        return

    # ── TRAINING brief flash ─────────────────────────────────────────────
    if fp_phase == "TRAINING":
        draw_centered_text_in_rect(frame, "BUILDING HAND MODEL...",
            (0, _ix(h*0.44), w, _ix(h*0.56)),
            base_scale=0.60, color=COL_YELLOW, thickness=2, outline=4)
        draw_bottom_bar(frame, "Please wait...")
        return

    # ── Silhouette — position and size driven by game_state ─────────────
    sil_cx    = int(sil_cx_n * w)
    sil_cy    = int(sil_cy_n * h)
    base_scale = _ix(h * 0.75)
    sil_scale  = int(base_scale * sil_sf)

    if fp_phase == "CAPTURING":
        pulse   = 0.7 + 0.3 * abs(_math.sin(t * _math.pi * 3.0))
        sil_col = tuple(min(255, int(c * pulse)) for c in COL_GREEN)
        sil_alpha = 0.90
    elif fp_phase == "ALIGN" and hand_visible and hand_open:
        sil_col   = COL_YELLOW
        sil_alpha = 0.90
    elif fp_phase == "REST":
        sil_col   = (80, 100, 140)
        sil_alpha = 0.70
    else:
        sil_col   = COL_CYAN
        sil_alpha = 0.85

    _draw_hand_silhouette(frame, sil_cx, sil_cy, sil_scale,
                          sil_col, alpha=sil_alpha, thickness=2,
                          mirror=(fp_hand_side == "Left"))
    instruct_y1 = _ix(h * 0.08)
    instruct_y2 = _ix(h * 0.20)

    if fp_phase == "INTRO":
        draw_centered_text_in_rect(frame,
            "Place your open hand inside the outline",
            (0, instruct_y1, w, instruct_y2),
            base_scale=0.48, color=COL_CYAN, thickness=2, outline=3)

    elif fp_phase == "ALIGN":
        if not hand_visible:
            msg, col = "Show your open hand to the camera", COL_TEXT_DIM
        elif not hand_open:
            msg, col = "Open your hand flat - spread your fingers", COL_YELLOW
        else:
            msg, col = "Hold still inside the outline...", COL_YELLOW
        draw_centered_text_in_rect(frame, msg,
            (0, instruct_y1, w, instruct_y2),
            base_scale=0.48, color=col, thickness=2, outline=3)

    elif fp_phase == "CAPTURING":
        if fp_quality:
            # Quality gate rejecting frames — show specific reason
            draw_centered_text_in_rect(frame, fp_quality,
                (0, instruct_y1, w, instruct_y2),
                base_scale=0.48, color=COL_YELLOW, thickness=2, outline=3)
        else:
            draw_centered_text_in_rect(frame, "Scanning - hold perfectly still",
                (0, instruct_y1, w, instruct_y2),
                base_scale=0.48, color=COL_GREEN, thickness=2, outline=3)

    elif fp_phase == "REST":
        # Show the specific repositioning instruction
        if fp_hint:
            draw_centered_text_in_rect(frame, fp_hint,
                (0, instruct_y1, w, instruct_y2),
                base_scale=0.44, color=COL_CYAN, thickness=2, outline=3)
        # Show "follow the outline" sub-hint
        draw_centered_text_in_rect(frame,
            "Follow the outline as it moves",
            (0, _ix(h*0.20), w, _ix(h*0.28)),
            base_scale=0.36, color=COL_TEXT_DIM, thickness=1, outline=2)

    elif fp_phase == "VERIFYING":
        draw_centered_text_in_rect(frame,
            "Verifying your hand profile - keep scanning",
            (0, instruct_y1, w, instruct_y2),
            base_scale=0.44, color=COL_YELLOW, thickness=2, outline=3)

    # ── Progress panel ───────────────────────────────────────────────────
    pan_x1 = _ix(w * 0.10)
    pan_x2 = w - _ix(w * 0.10)
    pan_y1 = _ix(h * 0.82)
    pan_y2 = _ix(h * 0.95)
    draw_panel(frame, pan_x1, pan_y1, pan_x2, pan_y2,
               fill=(6, 8, 18), alpha=0.88, border=COL_CYAN, border_thickness=1)

    bar_x1 = pan_x1 + _ix(w * 0.02)
    bar_x2 = pan_x2 - _ix(w * 0.02)
    bar_w  = bar_x2 - bar_x1
    bar_h  = _ix(h * 0.018)
    mid_y  = (pan_y1 + pan_y2) // 2
    bar_y  = mid_y + _ix(h * 0.010)

    if fp_phase == "TRAINING":
        pct, bar_col = 1.0, COL_YELLOW
        label   = "Building hand model..."
        l_col   = COL_YELLOW
    elif fp_phase == "CAPTURING":
        good_frames = game_state.get("fp_good_frames", 0)
        cap_target  = game_state.get("fp_capture_target", 40)
        pct, bar_col = fp_capture, COL_GREEN
        label   = f"Scanning round {fp_round + 1} of {fp_rounds_tgt}  -  {good_frames}/{cap_target} frames"
        l_col   = COL_GREEN
    elif fp_phase == "REST":
        pct, bar_col = fp_rest, COL_CYAN
        label   = f"Round {fp_round} complete - follow the outline"
        l_col   = COL_CYAN
    else:
        pct     = fp_stab
        bar_col = COL_YELLOW if fp_stab > 0.5 else (80, 80, 100)
        label   = f"Stability  -  {fp_round}/{fp_rounds_tgt} rounds done  -  hold still to scan"
        l_col   = COL_TEXT_ACCENT

    cv2.rectangle(frame, (bar_x1, bar_y - bar_h), (bar_x2, bar_y), (25, 30, 45), -1)
    fill_x = bar_x1 + int(bar_w * pct)
    if fill_x > bar_x1:
        cv2.rectangle(frame, (bar_x1, bar_y - bar_h), (fill_x, bar_y), bar_col, -1)
    cv2.rectangle(frame, (bar_x1, bar_y - bar_h), (bar_x2, bar_y), (50, 65, 90), 1)
    draw_centered_text_in_rect(frame, label,
        (pan_x1, pan_y1, pan_x2, mid_y),
        base_scale=0.34, color=l_col, thickness=1, outline=1)

    # Round pips — above the panel, sized to fit up to 20 rounds
    pip_y = pan_y1 - _ix(h * 0.025)
    pip_spacing = max(_ix(w * 0.013), (w - _ix(w * 0.20)) // max(fp_rounds_tgt, 1))
    pip_total_w = fp_rounds_tgt * pip_spacing
    pip_start   = (w - pip_total_w) // 2
    for ri in range(fp_rounds_tgt):
        px   = pip_start + ri * pip_spacing
        done = ri < fp_round
        cur  = (ri == fp_round and fp_phase == "CAPTURING")
        pc   = COL_GREEN if done else (COL_YELLOW if cur else (40, 40, 60))
        r    = 4 if done else 3
        cv2.circle(frame, (px, pip_y), r, pc, -1 if (done or cur) else 1)

    draw_bottom_bar(frame, "Open hand - align inside outline - hold still  |  ESC Cancel")

    # ── Live data strip (top-right corner) ──────────────────────────────
    info_x2 = w - _ix(w * 0.02)
    info_y  = _ix(h * 0.10)
    line_h  = _ix(h * 0.038)
    def _info_line(text, col, y):
        tw, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        tx = info_x2 - tw[0] - 4
        cv2.putText(frame, text, (tx + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(frame, text, (tx, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, col, 1, cv2.LINE_AA)

    hand_label = fp_hand_side if fp_hand_side != "Unknown" else "detecting..."
    _info_line(f"Hand: {hand_label}", COL_CYAN, info_y)
    _info_line(f"Samples: {fp_samples}/{fp_rounds_tgt}", COL_GREEN if fp_samples > 0 else COL_TEXT_DIM, info_y + line_h)
    if fp_phase == "CAPTURING":
        good_frames    = game_state.get("fp_good_frames",    0)
        capture_target = game_state.get("fp_capture_target", CAPTURE_FRAMES if 'CAPTURE_FRAMES' in dir() else 40)
        if fp_quality:
            _info_line(f"! {fp_quality}", COL_YELLOW, info_y + line_h * 2)
        else:
            _info_line(f"Frames: {good_frames}/{capture_target}", (80, 200, 80), info_y + line_h * 2)
    elif fp_quality:
        _info_line(f"! {fp_quality}", COL_YELLOW, info_y + line_h * 2)

# ============================================================
# HAND GEOMETRY LOGIN VIEW
# ============================================================

def draw_hand_login_view(frame, game_state, hand_state=None):
    """
    Simple hand scan login screen.
    Hold your hand up — it scans once and identifies you immediately.
    No rounds, no silhouette movement, no data collection UX.
    """
    import math as _math
    import time as _time

    w, h = frame.shape[1], frame.shape[0]
    t    = _time.monotonic()

    fp_phase     = game_state.get("fp_phase",          "WAITING")
    scan_pct     = game_state.get("scan_pct",           0.0)
    quality      = game_state.get("fp_quality_reason",  "")
    hand_side    = game_state.get("fp_hand_side",       "Unknown")
    has_profiles = game_state.get("has_profiles",       True)
    result       = game_state.get("login_result",       None)
    conf         = game_state.get("login_confidence",   0.0)
    failed       = game_state.get("login_failed",       False)
    hand_visible = game_state.get("hand_visible",       False)
    hand_open    = game_state.get("hand_open",          False)

    draw_panel(frame, 0, 0, w - 1, h - 1, fill=(4, 6, 16), alpha=0.65,
               border=(30, 30, 50), border_thickness=0)
    draw_top_bar(frame, "HAND SCAN - LOGIN",
                 "Hold up your open hand  |  ESC Back")

    # ── No profiles ──────────────────────────────────────────────────────
    if not has_profiles:
        draw_centered_text_in_rect(frame, "NO HAND PROFILES ENROLLED",
            (0, _ix(h*0.35), w, _ix(h*0.50)),
            base_scale=0.60, color=COL_YELLOW, thickness=2, outline=4)
        draw_centered_text_in_rect(frame,
            "Go to Settings - Enroll Hand Scan first.",
            (0, _ix(h*0.52), w, _ix(h*0.62)),
            base_scale=0.40, color=COL_TEXT_DIM, thickness=1, outline=2)
        draw_bottom_bar(frame, "ESC Back")
        return

    # ── Login failed ─────────────────────────────────────────────────────
    if failed:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)
        draw_centered_text_in_rect(frame, "HAND NOT RECOGNISED",
            (0, _ix(h*0.22), w, _ix(h*0.36)),
            base_scale=0.80, color=(220, 80, 80), thickness=3, outline=5)
        draw_centered_text_in_rect(frame,
            "Your hand did not match any enrolled profile.",
            (0, _ix(h*0.38), w, _ix(h*0.47)),
            base_scale=0.36, color=(190, 150, 150), thickness=1, outline=2)
        draw_centered_text_in_rect(frame,
            "Make sure you use the same hand you enrolled with, palm facing the camera.",
            (0, _ix(h*0.48), w, _ix(h*0.57)),
            base_scale=0.32, color=(160, 120, 120), thickness=1, outline=1)
        pulse = 0.5 + 0.5 * abs(_math.sin(t * _math.pi * 1.0))
        pc = tuple(min(255, int(c * pulse)) for c in (200, 160, 160))
        draw_centered_text_in_rect(frame, "ENTER to try again  |  ESC to type name",
            (0, _ix(h*0.64), w, _ix(h*0.73)),
            base_scale=0.44, color=pc, thickness=1, outline=2)
        draw_bottom_bar(frame, "ENTER try again  |  ESC type name instead")
        return

    # ── Identity confirmed ───────────────────────────────────────────────
    if result:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)
        draw_centered_text_in_rect(frame, "IDENTITY CONFIRMED",
            (0, _ix(h*0.18), w, _ix(h*0.30)),
            base_scale=0.70, color=(80, 220, 100), thickness=3, outline=5)
        draw_centered_text_in_rect(frame, result,
            (0, _ix(h*0.32), w, _ix(h*0.54)),
            base_scale=1.20, color=(220, 235, 220), thickness=4, outline=6)
        draw_centered_text_in_rect(frame, f"Confidence: {conf:.0%}",
            (0, _ix(h*0.55), w, _ix(h*0.63)),
            base_scale=0.38, color=(130, 180, 130), thickness=1, outline=2)
        pulse = 0.5 + 0.5 * abs(_math.sin(t * _math.pi * 1.0))
        pc = tuple(min(255, int(c * pulse)) for c in (160, 220, 160))
        draw_centered_text_in_rect(frame, "Press ENTER to log in",
            (0, _ix(h*0.66), w, _ix(h*0.74)),
            base_scale=0.48, color=pc, thickness=2, outline=3)
        draw_centered_text_in_rect(frame, "ESC to cancel",
            (0, _ix(h*0.75), w, _ix(h*0.81)),
            base_scale=0.30, color=(90, 110, 90), thickness=1, outline=1)
        draw_bottom_bar(frame, "ENTER to log in  |  ESC cancel")
        return

    # ── Scanning ─────────────────────────────────────────────────────────
    # Central instruction
    instruct_y1 = _ix(h * 0.10)
    instruct_y2 = _ix(h * 0.22)

    if fp_phase == "WAITING":
        if not hand_visible:
            msg, col = "Show your open hand to the camera", COL_TEXT_DIM
        elif not hand_open:
            msg, col = "Open your hand flat - spread your fingers", COL_YELLOW
        else:
            msg, col = "Hold still...", COL_YELLOW
        draw_centered_text_in_rect(frame, msg,
            (0, instruct_y1, w, instruct_y2),
            base_scale=0.50, color=col, thickness=2, outline=3)
    elif fp_phase == "SCANNING":
        if quality:
            draw_centered_text_in_rect(frame, quality,
                (0, instruct_y1, w, instruct_y2),
                base_scale=0.50, color=COL_YELLOW, thickness=2, outline=3)
        else:
            draw_centered_text_in_rect(frame, "Scanning - hold perfectly still",
                (0, instruct_y1, w, instruct_y2),
                base_scale=0.50, color=COL_GREEN, thickness=2, outline=3)

    # Big hand icon / progress ring in centre of screen
    cx = w // 2
    cy = _ix(h * 0.52)
    radius = _ix(h * 0.18)

    if fp_phase == "SCANNING" and scan_pct > 0:
        # Progress ring — fills as frames accumulate
        # Background circle
        cv2.circle(frame, (cx, cy), radius, (30, 40, 60), 4)
        # Arc fill
        import math as _math2
        start_angle = -90
        end_angle   = start_angle + int(360 * scan_pct)
        axes = (radius, radius)
        arc_col = COL_GREEN if not quality else COL_YELLOW
        cv2.ellipse(frame, (cx, cy), axes, 0, start_angle, end_angle, arc_col, 4)
        # Percentage in centre
        pct_str = f"{int(scan_pct * 100)}%"
        tw, th_ = cv2.getTextSize(pct_str, cv2.FONT_HERSHEY_SIMPLEX, 0.90, 2)
        cv2.putText(frame, pct_str,
                    (cx - tw[0]//2 + 1, cy + tw[1]//2 + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, pct_str,
                    (cx - tw[0]//2, cy + tw[1]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.90, arc_col, 2, cv2.LINE_AA)
    else:
        # Idle — simple hand icon hint
        cv2.circle(frame, (cx, cy), radius, (30, 40, 60), 2)
        draw_centered_text_in_rect(frame, "✋" if False else "HAND",
            (cx - radius, cy - _ix(h*0.04), cx + radius, cy + _ix(h*0.04)),
            base_scale=0.44, color=(50, 60, 80), thickness=1, outline=1)

    # Hand side indicator
    if hand_side != "Unknown":
        draw_centered_text_in_rect(frame, f"{hand_side} hand detected",
            (0, _ix(h*0.73), w, _ix(h*0.80)),
            base_scale=0.34, color=(80, 120, 160), thickness=1, outline=1)

    draw_bottom_bar(frame, "Hold your open hand steady to scan  |  ESC Back")

_FEATURE_LABELS = [
    "Index/Middle ratio",
    "Middle/Ring ratio",
    "Ring/Pinky ratio",
    "Thumb spread",
    "Index spread",
    "Pinky spread",
    "Palm aspect",
    "Index curl",
    "Middle curl",
    "Ring curl",
    "Knuckle span",
    "Index angle",
]

def draw_hand_diag_view(frame, diag_state, hand_state=None):
    """
    Live hand identity diagnostic screen.
    Shows who the system thinks is holding their hand up, in real time.
    Displays per-feature z-scores so you can see exactly which measurements
    match or deviate from the enrolled profile.
    """
    import math as _math
    import time as _time

    w, h = frame.shape[1], frame.shape[0]
    t    = _time.monotonic()

    pred_name       = diag_state.get("pred_name",       None)
    pred_conf       = diag_state.get("pred_conf",       0.0)
    hand_side       = diag_state.get("hand_side",       "Unknown")
    hand_visible    = diag_state.get("hand_visible",    False)
    quality_reason  = diag_state.get("quality_reason",  "")
    enrolled_names  = diag_state.get("enrolled_names",  [])
    has_profiles    = diag_state.get("has_profiles",    False)
    feature_zscores = diag_state.get("feature_zscores", [])
    buf_pct         = diag_state.get("buf_pct",         0.0)
    clf_mode        = diag_state.get("clf_mode",        "none")

    # Dark overlay
    draw_panel(frame, 0, 0, w - 1, h - 1, fill=(4, 6, 16), alpha=0.65,
               border=(30, 30, 50), border_thickness=0)

    enrolled_str = ", ".join(enrolled_names) if enrolled_names else "none"
    draw_top_bar(frame, "HAND SCAN - DIAGNOSTIC",
                 f"Enrolled: {enrolled_str}  |  Mode: {clf_mode}  |  ESC Back")

    # ── No profiles enrolled ─────────────────────────────────────────────
    if not has_profiles:
        draw_centered_text_in_rect(frame, "NO HAND PROFILES ENROLLED",
            (0, _ix(h*0.35), w, _ix(h*0.50)),
            base_scale=0.60, color=COL_YELLOW, thickness=2, outline=4)
        draw_centered_text_in_rect(frame,
            "Go to Settings - Enroll Hand Scan first.",
            (0, _ix(h*0.52), w, _ix(h*0.60)),
            base_scale=0.40, color=COL_TEXT_DIM, thickness=1, outline=2)
        draw_bottom_bar(frame, "ESC Back")
        return

    # ── Identity panel (large, centre-top) ──────────────────────────────
    id_pan_x1 = _ix(w * 0.08)
    id_pan_x2 = _ix(w * 0.92)
    id_pan_y1 = _ix(h * 0.10)
    id_pan_y2 = _ix(h * 0.42)
    draw_panel(frame, id_pan_x1, id_pan_y1, id_pan_x2, id_pan_y2,
               fill=(6, 8, 20), alpha=0.90, border=COL_CYAN, border_thickness=2)

    if not hand_visible or quality_reason:
        # No usable hand data
        msg = quality_reason if quality_reason else "No hand detected"
        draw_centered_text_in_rect(frame, msg,
            (id_pan_x1, id_pan_y1, id_pan_x2, id_pan_y2),
            base_scale=0.46, color=COL_TEXT_DIM, thickness=1, outline=2)

    elif pred_name is None and pred_conf < 0.05:
        # Still accumulating first window
        draw_centered_text_in_rect(frame, "Reading hand...",
            (id_pan_x1, id_pan_y1, id_pan_x2, id_pan_y2),
            base_scale=0.46, color=COL_TEXT_DIM, thickness=1, outline=2)
        # Buffer fill bar
        bx1 = id_pan_x1 + _ix(w*0.06)
        bx2 = id_pan_x2 - _ix(w*0.06)
        bh  = _ix(h * 0.015)
        by  = id_pan_y2 - _ix(h * 0.04)
        cv2.rectangle(frame, (bx1, by - bh), (bx2, by), (25, 30, 45), -1)
        fx = bx1 + int((bx2 - bx1) * buf_pct)
        if fx > bx1:
            cv2.rectangle(frame, (bx1, by - bh), (fx, by), COL_CYAN, -1)
        cv2.rectangle(frame, (bx1, by - bh), (bx2, by), (50, 65, 90), 1)

    elif pred_name is None:
        # Distance check failed — not a known person
        draw_centered_text_in_rect(frame, "UNKNOWN",
            (id_pan_x1, id_pan_y1 + _ix((id_pan_y2-id_pan_y1)*0.08),
             id_pan_x2, id_pan_y1 + _ix((id_pan_y2-id_pan_y1)*0.58)),
            base_scale=1.20, color=(200, 130, 50), thickness=4, outline=6)
        draw_centered_text_in_rect(frame,
            "Hand geometry does not match any enrolled profile",
            (id_pan_x1, id_pan_y1 + _ix((id_pan_y2-id_pan_y1)*0.62),
             id_pan_x2, id_pan_y1 + _ix((id_pan_y2-id_pan_y1)*0.82)),
            base_scale=0.34, color=(160, 110, 60), thickness=1, outline=2)

    else:
        # Recognised!
        pulse = 0.85 + 0.15 * abs(_math.sin(t * _math.pi * 1.5))
        name_col = tuple(min(255, int(c * pulse)) for c in COL_GREEN)
        draw_centered_text_in_rect(frame, pred_name,
            (id_pan_x1, id_pan_y1 + _ix((id_pan_y2-id_pan_y1)*0.06),
             id_pan_x2, id_pan_y1 + _ix((id_pan_y2-id_pan_y1)*0.56)),
            base_scale=1.10, color=name_col, thickness=3, outline=5)

        # Confidence bar
        conf_col = COL_GREEN if pred_conf > 0.70 else (COL_YELLOW if pred_conf > 0.40 else (200, 80, 60))
        bx1 = id_pan_x1 + _ix(w*0.06)
        bx2 = id_pan_x2 - _ix(w*0.06)
        bh  = _ix(h * 0.025)
        by  = id_pan_y2 - _ix(h * 0.05)
        cv2.rectangle(frame, (bx1, by - bh), (bx2, by), (25, 30, 45), -1)
        fx = bx1 + int((bx2 - bx1) * pred_conf)
        if fx > bx1:
            cv2.rectangle(frame, (bx1, by - bh), (fx, by), conf_col, -1)
        cv2.rectangle(frame, (bx1, by - bh), (bx2, by), (50, 65, 90), 1)
        draw_centered_text_in_rect(frame, f"Confidence: {pred_conf:.0%}",
            (id_pan_x1, id_pan_y1 + _ix((id_pan_y2-id_pan_y1)*0.60),
             id_pan_x2, id_pan_y1 + _ix((id_pan_y2-id_pan_y1)*0.78)),
            base_scale=0.38, color=conf_col, thickness=1, outline=2)

    # ── Info strip ───────────────────────────────────────────────────────
    info_y1 = _ix(h * 0.44)
    info_y2 = _ix(h * 0.52)
    draw_panel(frame, _ix(w*0.08), info_y1, _ix(w*0.92), info_y2,
               fill=(6, 8, 20), alpha=0.85, border=(40, 60, 80), border_thickness=1)

    hand_label = hand_side if hand_side != "Unknown" else "detecting..."
    info_txt = (f"Hand: {hand_label}   |   "
                f"Enrolled profiles: {len(enrolled_names)}   |   "
                f"Classifier: {clf_mode}")
    draw_centered_text_in_rect(frame, info_txt,
        (_ix(w*0.08), info_y1, _ix(w*0.92), info_y2),
        base_scale=0.34, color=COL_TEXT_ACCENT, thickness=1, outline=1)

    # ── Per-feature z-score panel ────────────────────────────────────────
    if feature_zscores and len(feature_zscores) == 12:
        feat_pan_x1 = _ix(w * 0.08)
        feat_pan_x2 = _ix(w * 0.92)
        feat_pan_y1 = _ix(h * 0.54)
        feat_pan_y2 = _ix(h * 0.93)
        draw_panel(frame, feat_pan_x1, feat_pan_y1, feat_pan_x2, feat_pan_y2,
                   fill=(4, 6, 16), alpha=0.88, border=(40, 60, 80), border_thickness=1)

        draw_centered_text_in_rect(frame, "FEATURE Z-SCORES  (how many std-devs from your enrolled profile)",
            (feat_pan_x1, feat_pan_y1, feat_pan_x2, feat_pan_y1 + _ix(h*0.04)),
            base_scale=0.30, color=COL_TEXT_DIM, thickness=1, outline=1)

        # Draw 12 mini bars in two columns of 6
        bar_area_y1 = feat_pan_y1 + _ix(h * 0.045)
        bar_area_y2 = feat_pan_y2 - _ix(h * 0.01)
        bar_h_total = bar_area_y2 - bar_area_y1
        row_h       = bar_h_total // 6
        col_w       = (feat_pan_x2 - feat_pan_x1) // 2
        bar_max_z   = 5.0   # z-score of 5 = full bar width

        for i, z in enumerate(feature_zscores):
            col   = i // 6
            row   = i % 6
            rx1   = feat_pan_x1 + col * col_w + _ix(w * 0.01)
            rx2   = feat_pan_x1 + (col + 1) * col_w - _ix(w * 0.01)
            ry1   = bar_area_y1 + row * row_h + 2
            ry2   = ry1 + row_h - 4
            mid_y = (ry1 + ry2) // 2

            # Label
            label = _FEATURE_LABELS[i] if i < len(_FEATURE_LABELS) else f"F{i}"
            lw, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.30, 1)
            label_w = lw[0] + 4
            cv2.putText(frame, label, (rx1 + 2, mid_y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (80, 90, 110), 1, cv2.LINE_AA)

            # Bar
            bar_x1 = rx1 + label_w + 4
            bar_x2 = rx2 - _ix(w * 0.025)
            bar_bh  = max(4, row_h - 8)
            bar_y   = mid_y - bar_bh // 2

            cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + bar_bh),
                          (20, 25, 40), -1)
            fill_w = int((bar_x2 - bar_x1) * min(1.0, z / bar_max_z))
            if fill_w > 0:
                # Green = close match, yellow = moderate, red = far
                if z < 1.5:
                    bar_col = (40, 200, 60)
                elif z < 3.0:
                    bar_col = (40, 200, 200)
                else:
                    bar_col = (60, 80, 220)
                cv2.rectangle(frame, (bar_x1, bar_y),
                              (bar_x1 + fill_w, bar_y + bar_bh), bar_col, -1)
            cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + bar_bh),
                          (40, 50, 70), 1)

            # Z value text
            z_str = f"{z:.1f}"
            cv2.putText(frame, z_str,
                        (bar_x2 + 2, mid_y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                        (100, 120, 140), 1, cv2.LINE_AA)

    elif not feature_zscores:
        draw_centered_text_in_rect(frame,
            "Hold your open hand up to see feature match scores",
            (_ix(w*0.08), _ix(h*0.56), _ix(w*0.92), _ix(h*0.68)),
            base_scale=0.38, color=COL_TEXT_DIM, thickness=1, outline=2)
        if clf_mode == "svm":
            draw_centered_text_in_rect(frame,
                "(Feature scores only available in single-person distance mode)",
                (_ix(w*0.08), _ix(h*0.68), _ix(w*0.92), _ix(h*0.76)),
                base_scale=0.30, color=(60, 70, 90), thickness=1, outline=1)

    draw_bottom_bar(frame, "Hold your open hand up to the camera  |  ESC Back")

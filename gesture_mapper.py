"""
gesture_mapper.py
=================
Converts MediaPipe hand landmarks into gestures.

two modes:
  five_gesture_mode=False (default)  ->  Rock, Paper, Scissors only
  five_gesture_mode=True  (RPSLS)    ->  Rock, Paper, Scissors, Spock, Lizard

Keeping Spock/Lizard detection out of standard RPS modes prevents false
positives (Paper being read as Lizard, etc.) that made RPS janky.
"""

import math


def _dist(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def _angle3(a, b, c):
    """Angle at vertex B in degrees (2-D, rotation invariant)."""
    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)
    denom = (math.sqrt(ba[0]**2 + ba[1]**2) *
             math.sqrt(bc[0]**2 + bc[1]**2))
    if denom < 1e-8:
        return 180.0
    return math.degrees(math.acos(max(-1.0, min(1.0,
                                                (ba[0]*bc[0] + ba[1]*bc[1]) / denom))))


def _pip(lm, mcp, pip, dip):
    return _angle3(lm[mcp], lm[pip], lm[dip])


def _palm_scale(lm):
    return max(_dist(lm[0], lm[5]), 1e-6)


def classify_rps_gesture(count_result, hand_landmarks=None, five_gesture_mode=False):
    """
    Returns {"gesture": str, "command": str, "reason": str}.

    five_gesture_mode=False  →  Rock / Paper / Scissors only (standard RPS)
    five_gesture_mode=True   →  adds Spock / Lizard detection (RPSLS only)

    When hand_landmarks is provided the geometry path bypasses finger_counter's
    ambiguous_pose gate and classifies directly from PIP joint angles.
    """
    if not count_result and hand_landmarks is None:
        return {"gesture": "Unknown", "command": "CMD_UNKNOWN", "reason": "no_result"}

    # Geometry path: bypass ambiguous_pose when landmarks are present
    if hand_landmarks is None:
        if not count_result or count_result["count_text"] == "Unknown" \
                or count_result["states"] is None:
            return {"gesture": "Unknown", "command": "CMD_UNKNOWN",
                    "reason": (count_result or {}).get("reason", "no_result")}

    # ── Geometry path ─────────────────────────────────────────────────────
    if hand_landmarks is not None:
        lm    = hand_landmarks.landmark
        scale = _palm_scale(lm)

        p_idx = _pip(lm, 5,  6,  7)
        p_mid = _pip(lm, 9,  10, 11)
        p_rng = _pip(lm, 13, 14, 15)
        p_pnk = _pip(lm, 17, 18, 19)
        avg   = (p_idx + p_mid + p_rng + p_pnk) / 4.0

        idx_ext = p_idx >= 150
        mid_ext = p_mid >= 150
        rng_ext = p_rng >= 150
        pnk_ext = p_pnk >= 150

        thumb_tip  = lm[4]
        index_tip  = lm[8]
        middle_tip = lm[12]
        ring_tip   = lm[16]
        pinky_tip  = lm[20]
        wrist      = lm[0]

        # ── SPOCK + PAPER (all 4 fingers extended) ────────────────────────
        if idx_ext and mid_ext and rng_ext and pnk_ext:
            if five_gesture_mode:
                # Only check Vulcan split in RPSLS mode
                gap_im = _dist(index_tip,  middle_tip)
                gap_mr = _dist(middle_tip, ring_tip)
                gap_rp = _dist(ring_tip,   pinky_tip)
                avg_nb = (gap_im + gap_rp) / 2.0
                ratio  = gap_mr / max(avg_nb, 1e-6)
                if ratio >= 1.4:
                    return {"gesture": "Spock", "command": "CMD_SPOCK",
                            "reason": f"vulcan ratio={ratio:.2f}"}
            # No Vulcan split (or RPS mode) → Paper
            return {"gesture": "Paper", "command": "CMD_PAPER",
                    "reason": "all_extended"}

        # ── LIZARD (RPSLS only) ───────────────────────────────────────────
        if five_gesture_mode:
            scissors_pattern = idx_ext and mid_ext and not rng_ext and not pnk_ext
            if not scissors_pattern and avg > 100:
                f_avg_y = (index_tip.y + middle_tip.y + ring_tip.y + pinky_tip.y) / 4.0
                thumb_below_fingers = thumb_tip.y > f_avg_y
                lizard_pip_range    = avg > 130
                if thumb_below_fingers and lizard_pip_range:
                    return {"gesture": "Lizard", "command": "CMD_LIZARD",
                            "reason": f"sock-puppet rise={(wrist.y-f_avg_y)/_palm_scale(lm):.2f} pip={avg:.0f}"}

        # ── ROCK ──────────────────────────────────────────────────────────
        if avg < 120 and not idx_ext and not mid_ext:
            return {"gesture": "Rock", "command": "CMD_ROCK",
                    "reason": f"fist pip={avg:.0f}"}

        # ── SCISSORS ──────────────────────────────────────────────────────
        if idx_ext and mid_ext and not rng_ext and not pnk_ext:
            return {"gesture": "Scissors", "command": "CMD_SCISSORS",
                    "reason": "index_middle_only"}

        return {"gesture": "Unknown", "command": "CMD_UNKNOWN",
                "reason": f"no_match pip={avg:.0f}"}

    # ── Binary-state fallback (no landmarks) ─────────────────────────────
    if not count_result or count_result["states"] is None:
        return {"gesture": "Unknown", "command": "CMD_UNKNOWN",
                "reason": "no_landmarks_no_states"}
    st = count_result["states"]
    if not any(st.values()):
        return {"gesture": "Rock",     "command": "CMD_ROCK",     "reason": "all_down"}
    if st["index"] and st["middle"] and st["ring"] and st["pinky"]:
        return {"gesture": "Paper",    "command": "CMD_PAPER",    "reason": "four_up"}
    if st["index"] and st["middle"] and not st["ring"] and not st["pinky"]:
        return {"gesture": "Scissors", "command": "CMD_SCISSORS", "reason": "index_middle"}
    return {"gesture": "Unknown", "command": "CMD_UNKNOWN", "reason": "no_match_no_lm"}

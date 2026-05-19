"""
gesture_mapper.py
=================
Converts a MediaPipe hand landmark object (optionally paired with a
finger-counter result) into a named RPS / RPSLS gesture.

What it does:
  - Uses PIP joint angles (the knuckle angles) and fingertip geometry
    to classify Rock, Paper, Scissors — and optionally Spock and Lizard.
  - The geometry path (using hand_landmarks) is the primary one and runs
    whenever landmarks are available.
  - A binary-state fallback (using the finger-count result only) exists
    for cases where landmarks are not passed in.

Why two modes?
  - five_gesture_mode=False (standard RPS): Spock/Lizard detection is
    intentionally disabled.  Enabling it caused false positives — Paper
    was sometimes read as Lizard, making basic RPS unreliable.
  - five_gesture_mode=True  (RPSLS mode): all five gestures active.

Where it fits:
  - Called by hand_landmarks.py once per frame, after finger_counter.py.
  - Returns {"gesture": str, "command": str, "reason": str}.
"""

import math


# =============================================================================
# Geometry helpers
# =============================================================================

def _dist(a, b):
    """2-D Euclidean distance between two MediaPipe landmark objects."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def _angle3(a, b, c):
    """
    Angle at vertex B in degrees (2-D, rotation invariant).

    Builds vectors BA and BC, then uses the dot-product formula.
    Clamps the cosine to [-1, 1] before acos to avoid domain errors from
    floating-point rounding.  Returns 180° if either vector is near-zero.
    """
    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)
    denom = (math.sqrt(ba[0]**2 + ba[1]**2) *
             math.sqrt(bc[0]**2 + bc[1]**2))
    if denom < 1e-8:
        return 180.0
    cos_val = (ba[0]*bc[0] + ba[1]*bc[1]) / denom
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_val))))


def _pip(lm, mcp, pip, dip):
    """
    Return the angle at the PIP (proximal interphalangeal) joint for one
    finger, given the landmark array and the MCP/PIP/DIP landmark indices.
    A straight finger has an angle near 180°; a curled finger is much lower.
    """
    return _angle3(lm[mcp], lm[pip], lm[dip])


def _palm_scale(lm):
    """
    Return the wrist-to-index-MCP distance as a scale reference.
    Used to make Spock's Vulcan-split ratio check scale-invariant.
    Floor at 1e-6 to prevent division by zero.
    """
    return max(_dist(lm[0], lm[5]), 1e-6)


# =============================================================================
# Main classifier
# =============================================================================

def classify_rps_gesture(count_result, hand_landmarks=None, five_gesture_mode=False):
    """
    Classify the current hand pose as a gesture name.

    Parameters:
        count_result       — dict from finger_counter.count_hand_fingers()
        hand_landmarks     — MediaPipe NormalizedLandmarkList (optional but
                             strongly recommended; enables the geometry path)
        five_gesture_mode  — set True for RPSLS; adds Spock and Lizard

    Returns:
        {"gesture": str, "command": str, "reason": str}

        gesture  — "Rock" | "Paper" | "Scissors" | "Spock" | "Lizard" | "Unknown"
        command  — "CMD_ROCK" etc. (ready for downstream command dispatch)
        reason   — short debug string explaining why this gesture was chosen
    """
    # Nothing to work with at all
    if not count_result and hand_landmarks is None:
        return {"gesture": "Unknown", "command": "CMD_UNKNOWN", "reason": "no_result"}

    # If no landmarks are provided, validate the count result before proceeding
    if hand_landmarks is None:
        if (not count_result
                or count_result["count_text"] == "Unknown"
                or count_result["states"] is None):
            return {
                "gesture": "Unknown",
                "command": "CMD_UNKNOWN",
                "reason":  (count_result or {}).get("reason", "no_result"),
            }

    # ── Geometry path (preferred) ─────────────────────────────────────────────
    # This runs whenever hand_landmarks is available.  It measures PIP joint
    # angles directly rather than relying on the finger-counter booleans, which
    # makes it more robust to borderline hand positions.
    if hand_landmarks is not None:
        lm    = hand_landmarks.landmark
        scale = _palm_scale(lm)

        # Measure the PIP angle for each of the four main fingers.
        # Near 180° = straight/extended; much lower = curled/closed.
        p_idx = _pip(lm, 5,  6,  7)   # index finger
        p_mid = _pip(lm, 9,  10, 11)  # middle finger
        p_rng = _pip(lm, 13, 14, 15)  # ring finger
        p_pnk = _pip(lm, 17, 18, 19)  # pinky finger
        avg   = (p_idx + p_mid + p_rng + p_pnk) / 4.0

        # Boolean: is each finger extended? (threshold: ≥150°)
        idx_ext = p_idx >= 150
        mid_ext = p_mid >= 150
        rng_ext = p_rng >= 150
        pnk_ext = p_pnk >= 150

        # Convenience references for fingertip positions
        thumb_tip  = lm[4]
        index_tip  = lm[8]
        middle_tip = lm[12]
        ring_tip   = lm[16]
        pinky_tip  = lm[20]
        wrist      = lm[0]

        # ── Paper / Spock (all four fingers extended) ─────────────────────
        if idx_ext and mid_ext and rng_ext and pnk_ext:
            if five_gesture_mode:
                # Check for the Vulcan "V" split between middle and ring fingers.
                # In the Spock sign, the gap between middle and ring is notably
                # wider than the adjacent gaps (index-middle and ring-pinky).
                gap_im = _dist(index_tip,  middle_tip)
                gap_mr = _dist(middle_tip, ring_tip)
                gap_rp = _dist(ring_tip,   pinky_tip)
                avg_nb = (gap_im + gap_rp) / 2.0      # average "normal" gap
                ratio  = gap_mr / max(avg_nb, 1e-6)   # how much wider is the split?
                if ratio >= 1.4:
                    # The middle-ring gap is 40%+ wider than normal — it's Spock
                    return {"gesture": "Spock", "command": "CMD_SPOCK",
                            "reason": f"vulcan ratio={ratio:.2f}"}
            # All fingers extended but no Vulcan split (or RPS mode) → Paper
            return {"gesture": "Paper", "command": "CMD_PAPER",
                    "reason": "all_extended"}

        # ── Lizard (RPSLS only) ───────────────────────────────────────────
        # Lizard looks like a flat "sock puppet" — fingers partly extended
        # but pitched forward, with the thumb below the finger plane.
        if five_gesture_mode:
            scissors_pattern = idx_ext and mid_ext and not rng_ext and not pnk_ext
            # Only test for Lizard if it doesn't look like Scissors already
            if not scissors_pattern and avg > 100:
                f_avg_y            = (index_tip.y + middle_tip.y + ring_tip.y + pinky_tip.y) / 4.0
                thumb_below_fingers = thumb_tip.y > f_avg_y
                lizard_pip_range    = avg > 130
                if thumb_below_fingers and lizard_pip_range:
                    return {
                        "gesture": "Lizard",
                        "command": "CMD_LIZARD",
                        "reason":  (
                            f"sock-puppet rise={(wrist.y - f_avg_y) / _palm_scale(lm):.2f}"
                            f" pip={avg:.0f}"
                        ),
                    }

        # ── Rock (fist) ───────────────────────────────────────────────────
        # All fingers curled: average PIP angle well below 120° and the two
        # most diagnostic fingers (index and middle) not extended.
        if avg < 120 and not idx_ext and not mid_ext:
            return {"gesture": "Rock", "command": "CMD_ROCK",
                    "reason": f"fist pip={avg:.0f}"}

        # ── Scissors ──────────────────────────────────────────────────────
        # Index and middle extended, ring and pinky down.
        if idx_ext and mid_ext and not rng_ext and not pnk_ext:
            return {"gesture": "Scissors", "command": "CMD_SCISSORS",
                    "reason": "index_middle_only"}

        # Didn't match any known pattern
        return {"gesture": "Unknown", "command": "CMD_UNKNOWN",
                "reason": f"no_match pip={avg:.0f}"}

    # ── Binary-state fallback (no landmarks) ─────────────────────────────────
    # This path runs only when hand_landmarks was not provided.  It uses the
    # simple True/False extended-state dict from finger_counter instead.
    if not count_result or count_result["states"] is None:
        return {"gesture": "Unknown", "command": "CMD_UNKNOWN",
                "reason": "no_landmarks_no_states"}

    st = count_result["states"]

    # All fingers down = Rock
    if not any(st.values()):
        return {"gesture": "Rock",     "command": "CMD_ROCK",     "reason": "all_down"}

    # All four main fingers up = Paper
    if st["index"] and st["middle"] and st["ring"] and st["pinky"]:
        return {"gesture": "Paper",    "command": "CMD_PAPER",    "reason": "four_up"}

    # Index and middle up, ring and pinky down = Scissors
    if st["index"] and st["middle"] and not st["ring"] and not st["pinky"]:
        return {"gesture": "Scissors", "command": "CMD_SCISSORS", "reason": "index_middle"}

    # Anything else is ambiguous in the no-landmarks fallback
    return {"gesture": "Unknown", "command": "CMD_UNKNOWN", "reason": "no_match_no_lm"}

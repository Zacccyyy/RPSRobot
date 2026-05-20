"""
gesture_mapper.py
=================
Converts a MediaPipe hand landmark object into a named RPS / RPSLS gesture.

How it works:
  - Measures the PIP (middle knuckle) joint angle for each finger.
    Near 180 degrees = finger is straight. Much lower = finger is curled.
  - Uses those angles to classify the hand as Rock, Paper, Scissors,
    and optionally Spock or Lizard (five_gesture_mode).
  - Falls back to a simpler boolean check (extended/not extended per finger)
    if no landmark object is provided.

Why two modes?
  - five_gesture_mode=False (standard RPS): Spock and Lizard are disabled.
    Enabling them caused false positives -- Paper was sometimes read as
    Lizard, breaking basic RPS reliability.
  - five_gesture_mode=True (RPSLS mode): all five gestures active.

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
    Angle at vertex B (in degrees) formed by points A-B-C, in 2D.

    Builds vectors BA and BC from the landmarks, then uses the dot-product
    formula to find the angle between them. Clamps the cosine to [-1, 1]
    before calling acos to guard against floating-point rounding errors.
    Returns 180 degrees if either vector has zero length (degenerate case).
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
    Return the PIP joint angle for one finger.

    PIP = proximal interphalangeal joint (the middle knuckle).
    Straight finger ≈ 180 degrees. Curled finger = much lower.
    lm is the full landmark array; mcp, pip, dip are landmark indices.
    """
    return _angle3(lm[mcp], lm[pip], lm[dip])


def _palm_scale(lm):
    """
    Distance from wrist (lm[0]) to index MCP knuckle (lm[5]).
    Used as a scale reference so ratios work regardless of hand distance.
    Floor at 1e-6 to prevent division by zero.
    """
    return max(_dist(lm[0], lm[5]), 1e-6)


# =============================================================================
# Main classifier
# =============================================================================

def classify_rps_gesture(count_result, hand_landmarks=None, five_gesture_mode=False):
    """
    Classify the current hand pose as a named gesture.

    Parameters:
        count_result      -- dict from finger_counter.count_hand_fingers()
        hand_landmarks    -- MediaPipe NormalizedLandmarkList (optional but
                             strongly recommended; enables the geometry path)
        five_gesture_mode -- set True for RPSLS; adds Spock and Lizard detection

    Returns:
        {"gesture": str, "command": str, "reason": str}

        gesture  -- "Rock" | "Paper" | "Scissors" | "Spock" | "Lizard" | "Unknown"
        command  -- "CMD_ROCK" etc. (ready for downstream dispatch)
        reason   -- short debug string explaining the classification decision
    """
    # If we have absolutely nothing to work with, return Unknown immediately
    if not count_result and hand_landmarks is None:
        return {"gesture": "Unknown", "command": "CMD_UNKNOWN", "reason": "no_result"}

    # If there are no landmarks, validate that the count result is usable
    # before we try the fallback path
    if hand_landmarks is None:
        if (not count_result
                or count_result["count_text"] == "Unknown"
                or count_result["states"] is None):
            return {
                "gesture": "Unknown",
                "command": "CMD_UNKNOWN",
                "reason":  (count_result or {}).get("reason", "no_result"),
            }

    # ── Geometry path (primary) ───────────────────────────────────────────────
    # This runs whenever hand_landmarks is available. Measuring actual joint
    # angles is more robust than relying on the boolean finger-counter alone.
    if hand_landmarks is not None:
        lm = hand_landmarks.landmark

        # Measure the PIP angle for each of the four main fingers.
        # Landmark IDs: (MCP, PIP, DIP) for each finger.
        p_idx = _pip(lm, 5,  6,  7)   # index finger
        p_mid = _pip(lm, 9,  10, 11)  # middle finger
        p_rng = _pip(lm, 13, 14, 15)  # ring finger
        p_pnk = _pip(lm, 17, 18, 19)  # pinky finger
        avg   = (p_idx + p_mid + p_rng + p_pnk) / 4.0

        # Boolean: is each finger extended? Threshold is 150 degrees.
        idx_ext = p_idx >= 150
        mid_ext = p_mid >= 150
        rng_ext = p_rng >= 150
        pnk_ext = p_pnk >= 150

        # Shorthand references to fingertip and wrist positions
        thumb_tip  = lm[4]
        index_tip  = lm[8]
        middle_tip = lm[12]
        ring_tip   = lm[16]
        pinky_tip  = lm[20]
        wrist      = lm[0]

        # ── Paper / Spock: all four fingers extended ───────────────────────
        if idx_ext and mid_ext and rng_ext and pnk_ext:
            if five_gesture_mode:
                # In Spock (Vulcan salute), the middle-ring gap is noticeably
                # wider than the index-middle and ring-pinky gaps.
                # We check if that gap is at least 40% wider than the average of the others.
                gap_im = _dist(index_tip,  middle_tip)
                gap_mr = _dist(middle_tip, ring_tip)
                gap_rp = _dist(ring_tip,   pinky_tip)
                avg_nb = (gap_im + gap_rp) / 2.0       # "normal" adjacent gap
                ratio  = gap_mr / max(avg_nb, 1e-6)    # how much wider is the split?
                if ratio >= 1.4:
                    return {"gesture": "Spock", "command": "CMD_SPOCK",
                            "reason": f"vulcan ratio={ratio:.2f}"}
            # All fingers extended, no Vulcan split (or in RPS-only mode) = Paper
            return {"gesture": "Paper", "command": "CMD_PAPER", "reason": "all_extended"}

        # ── Lizard (RPSLS only) ────────────────────────────────────────────
        # Lizard looks like a flat "sock puppet" -- fingers partly extended,
        # thumb sitting below the finger plane. Only check if it doesn't
        # already look like Scissors.
        if five_gesture_mode:
            scissors_pattern = idx_ext and mid_ext and not rng_ext and not pnk_ext
            if not scissors_pattern and avg > 100:
                f_avg_y             = (index_tip.y + middle_tip.y + ring_tip.y + pinky_tip.y) / 4.0
                thumb_below_fingers = thumb_tip.y > f_avg_y   # in image coords, larger Y = lower
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

        # ── Rock (fist): all fingers curled ───────────────────────────────
        # Average PIP angle well below 120, and the two most diagnostic
        # fingers (index and middle) must not be extended.
        if avg < 120 and not idx_ext and not mid_ext:
            return {"gesture": "Rock", "command": "CMD_ROCK",
                    "reason": f"fist pip={avg:.0f}"}

        # ── Scissors: index and middle extended, ring and pinky down ──────
        if idx_ext and mid_ext and not rng_ext and not pnk_ext:
            return {"gesture": "Scissors", "command": "CMD_SCISSORS",
                    "reason": "index_middle_only"}

        # Didn't match any known pattern
        return {"gesture": "Unknown", "command": "CMD_UNKNOWN",
                "reason": f"no_match pip={avg:.0f}"}

    # ── Binary-state fallback (no landmarks provided) ─────────────────────────
    # Uses the simple True/False extended-state dict from finger_counter.
    # Less accurate than the geometry path but works as a last resort.
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

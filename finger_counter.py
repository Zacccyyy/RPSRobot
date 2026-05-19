# =============================================================================
# finger_counter.py
# -----------------
# Determines which fingers are currently extended (up) from a MediaPipe
# hand landmark object, and returns a structured count result.
#
# What it does:
#   - Measures PIP and DIP joint angles and tip-to-wrist distances.
#   - Applies separate rules for the thumb (different biomechanics) and
#     for each of the four remaining fingers.
#   - Flags fingers whose state is borderline ("ambiguous") — if three or
#     more fingers are ambiguous the whole result is marked uncertain.
#   - Returns a dict consumed by gesture_mapper.py and hand_landmarks.py.
#
# Where it fits:
#   - Called by process_hand_frame() / process_two_hands_frame() in
#     hand_landmarks.py once per frame per detected hand.
#   - Its output feeds directly into classify_rps_gesture() in gesture_mapper.py.
# =============================================================================

import math


# MediaPipe fingertip landmark indices (do not change — wired into the model)
THUMB_TIP  = 4
INDEX_TIP  = 8
MIDDLE_TIP = 12
RING_TIP   = 16
PINKY_TIP  = 20


# =============================================================================
# Low-level geometry helpers
# =============================================================================

def _distance(a, b):
    """
    3-D Euclidean distance between two MediaPipe landmark objects.
    Falls back to 0.0 for the Z axis if the landmark doesn't carry it
    (some MediaPipe models omit Z on older devices).
    """
    az = getattr(a, "z", 0.0)
    bz = getattr(b, "z", 0.0)
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (az - bz) ** 2)


def _distance_to_point(a, p):
    """
    3-D distance between a MediaPipe landmark and a plain (x, y, z) tuple.
    Used to measure how far the thumb tip is from the palm centre.
    """
    az = getattr(a, "z", 0.0)
    return math.sqrt((a.x - p[0]) ** 2 + (a.y - p[1]) ** 2 + (az - p[2]) ** 2)


def _angle(a, b, c):
    """
    Return the angle at vertex B (in degrees) formed by points A-B-C,
    using 3-D landmark coordinates.

    This is the standard joint-angle calculation: build vectors BA and BC,
    then use the dot-product formula to find the angle between them.
    Returns 0.0 if either vector has zero length (degenerate case).
    """
    # Vectors from B to A, and from B to C
    ab = (a.x - b.x, a.y - b.y, getattr(a, "z", 0.0) - getattr(b, "z", 0.0))
    cb = (c.x - b.x, c.y - b.y, getattr(c, "z", 0.0) - getattr(b, "z", 0.0))

    dot    = ab[0]*cb[0] + ab[1]*cb[1] + ab[2]*cb[2]
    mag_ab = math.sqrt(ab[0]**2 + ab[1]**2 + ab[2]**2)
    mag_cb = math.sqrt(cb[0]**2 + cb[1]**2 + cb[2]**2)

    if mag_ab == 0 or mag_cb == 0:
        return 0.0

    # Clamp to [-1, 1] to guard against floating-point rounding errors before acos
    cos_angle = max(-1.0, min(1.0, dot / (mag_ab * mag_cb)))
    return math.degrees(math.acos(cos_angle))


def _palm_size(lm):
    """
    Estimate the apparent hand size as the average of three wrist-to-MCP
    distances (index, middle, pinky knuckles).

    This gives us a scale-invariant reference length so that the extension
    thresholds below work regardless of how far the hand is from the camera.
    Returns at least 1e-6 to prevent division-by-zero in callers.
    """
    wrist      = lm[0]
    index_mcp  = lm[5]
    middle_mcp = lm[9]
    pinky_mcp  = lm[17]

    d1 = _distance(wrist, index_mcp)
    d2 = _distance(wrist, middle_mcp)
    d3 = _distance(wrist, pinky_mcp)

    return max((d1 + d2 + d3) / 3.0, 1e-6)


def _palm_center(lm):
    """
    Approximate the palm centre as the average of the wrist and the three
    main MCP (knuckle) joints.  Used as a reference point for the thumb.
    Returns an (x, y, z) tuple.
    """
    pts = [lm[0], lm[5], lm[9], lm[17]]
    x   = sum(p.x for p in pts) / len(pts)
    y   = sum(p.y for p in pts) / len(pts)
    z   = sum(getattr(p, "z", 0.0) for p in pts) / len(pts)
    return (x, y, z)


# =============================================================================
# Per-finger extension rules
# =============================================================================

def _is_finger_extended(lm, tip_id, dip_id, pip_id, mcp_id, palm_scale):
    """
    Decide whether one of the four non-thumb fingers (index/middle/ring/pinky)
    is extended (pointing away from the palm).

    Three independent conditions must ALL be true for a finger to be "up":
      1. straight_enough  — both joint angles (PIP and DIP) are close to 180°,
                            meaning the finger is not curled
      2. long_enough      — the tip-to-MCP distance is more than 55% of palm
                            scale, ruling out fingers folded sideways
      3. farther_than_pip — the fingertip is further from the wrist than the
                            PIP joint plus a small margin, ensuring the finger
                            is pointing away and not looping back

    If any condition is only *marginally* false, the finger is flagged as
    "ambiguous" — the caller can use this to reject uncertain overall counts.

    Returns:
        (is_up: bool, ambiguous: bool)
    """
    wrist = lm[0]
    tip   = lm[tip_id]
    dip   = lm[dip_id]
    pip   = lm[pip_id]
    mcp   = lm[mcp_id]

    pip_angle = _angle(mcp, pip, dip)    # angle at the knuckle joint
    dip_angle = _angle(pip, dip, tip)    # angle at the middle joint

    tip_wrist = _distance(tip, wrist)
    pip_wrist = _distance(pip, wrist)
    mcp_tip   = _distance(mcp, tip)

    # The three extension conditions
    straight_enough  = pip_angle > 160 and dip_angle > 150
    long_enough      = mcp_tip > 0.55 * palm_scale
    farther_than_pip = tip_wrist > pip_wrist + 0.18 * palm_scale

    is_up = straight_enough and long_enough and farther_than_pip

    # "Ambiguous" means we're right on the boundary — could go either way
    ambiguous = (
        150 <= pip_angle <= 160 or
        140 <= dip_angle <= 150 or
        abs(tip_wrist - (pip_wrist + 0.18 * palm_scale)) < 0.04 * palm_scale
    )

    return is_up, ambiguous


def _is_thumb_extended(lm, palm_scale):
    """
    Decide whether the thumb is extended (sticking outward from the hand).

    The thumb has a different joint structure and sits at the side of the
    hand, so it gets its own set of rules:
      1. straight_enough      — MCP and IP joint angles are fairly open
      2. away_from_palm       — thumb tip is clearly further from palm centre
                                than the IP joint
      3. separated_from_hand  — tip-to-index-MCP distance is large enough that
                                the thumb isn't tucked under the fingers

    Returns:
        (is_up: bool, ambiguous: bool)
    """
    thumb_cmc = lm[1]
    thumb_mcp = lm[2]
    thumb_ip  = lm[3]
    thumb_tip = lm[4]
    index_mcp = lm[5]

    palm_center = _palm_center(lm)

    mcp_angle = _angle(thumb_cmc, thumb_mcp, thumb_ip)
    ip_angle  = _angle(thumb_mcp, thumb_ip, thumb_tip)

    tip_palm  = _distance_to_point(thumb_tip, palm_center)
    ip_palm   = _distance_to_point(thumb_ip,  palm_center)
    tip_index = _distance(thumb_tip, index_mcp)

    straight_enough    = mcp_angle > 135 and ip_angle > 145
    away_from_palm     = tip_palm > ip_palm + 0.10 * palm_scale
    separated_from_hand = tip_index > 0.35 * palm_scale

    is_up = straight_enough and away_from_palm and separated_from_hand

    # Borderline cases
    ambiguous = (
        125 <= mcp_angle <= 135 or
        135 <= ip_angle  <= 145 or
        abs(tip_palm - (ip_palm + 0.10 * palm_scale)) < 0.03 * palm_scale
    )

    return is_up, ambiguous


# =============================================================================
# Public API
# =============================================================================

def count_hand_fingers(
    hand_landmarks,
    hand_label,
    target_hand="Auto",
    handedness_score=1.0,
    handedness_threshold=0.80,
):
    """
    Count how many fingers are currently extended on the given hand.

    Parameters:
        hand_landmarks       — MediaPipe NormalizedLandmarkList
        hand_label           — "Left" or "Right" (from MediaPipe handedness)
        target_hand          — "Left", "Right", or "Auto"
                               "Auto" accepts any hand regardless of label/score
        handedness_score     — MediaPipe's confidence that hand_label is correct
        handedness_threshold — minimum handedness_score required when not in Auto

    Returns a dict with:
        count        — int number of extended fingers, or None if uncertain
        count_text   — str(count) or "Unknown"
        states       — {"thumb": bool, "index": bool, ...} per finger
        up_fingers   — list of finger name strings that are extended
        tip_ids_up   — list of MediaPipe tip landmark IDs for extended fingers
                       (used by the diagnostic overlay to draw red dots)
        reason       — "ok" on success, or a short reason string on failure
        ambiguous    — how many fingers were in an ambiguous (borderline) state
    """
    # No landmarks at all — return a clean "nothing to work with" result
    if hand_landmarks is None:
        return {
            "count":       None,
            "count_text":  "Unknown",
            "states":      None,
            "up_fingers":  [],
            "tip_ids_up":  [],
            "reason":      "no_hand",
            "ambiguous":   0,
        }

    auto_mode = target_hand == "Auto"

    # In non-Auto mode: reject if the detected hand is not the one we want
    if not auto_mode and hand_label != target_hand:
        return {
            "count":       None,
            "count_text":  "Unknown",
            "states":      None,
            "up_fingers":  [],
            "tip_ids_up":  [],
            "reason":      "wrong_hand_mode",
            "ambiguous":   0,
        }

    # In non-Auto mode: reject if MediaPipe isn't confident which hand this is
    if not auto_mode and handedness_score < handedness_threshold:
        return {
            "count":       None,
            "count_text":  "Unknown",
            "states":      None,
            "up_fingers":  [],
            "tip_ids_up":  [],
            "reason":      "low_handedness_confidence",
            "ambiguous":   0,
        }

    lm         = hand_landmarks.landmark
    palm_scale = _palm_size(lm)

    # Check each finger — landmark IDs: (tip, dip, pip, mcp) per finger
    thumb_up,  thumb_amb  = _is_thumb_extended(lm, palm_scale)
    index_up,  index_amb  = _is_finger_extended(lm, 8,  7,  6,  5,  palm_scale)
    middle_up, middle_amb = _is_finger_extended(lm, 12, 11, 10, 9,  palm_scale)
    ring_up,   ring_amb   = _is_finger_extended(lm, 16, 15, 14, 13, palm_scale)
    pinky_up,  pinky_amb  = _is_finger_extended(lm, 20, 19, 18, 17, palm_scale)

    # Collect bool state per finger for downstream use
    states = {
        "thumb":  thumb_up,
        "index":  index_up,
        "middle": middle_up,
        "ring":   ring_up,
        "pinky":  pinky_up,
    }

    # Count how many fingers were on the borderline
    ambiguous_count = sum([thumb_amb, index_amb, middle_amb, ring_amb, pinky_amb])

    # Build the lists of extended finger names and their tip IDs
    up_fingers = []
    tip_ids_up = []

    if thumb_up:
        up_fingers.append("thumb");  tip_ids_up.append(THUMB_TIP)
    if index_up:
        up_fingers.append("index");  tip_ids_up.append(INDEX_TIP)
    if middle_up:
        up_fingers.append("middle"); tip_ids_up.append(MIDDLE_TIP)
    if ring_up:
        up_fingers.append("ring");   tip_ids_up.append(RING_TIP)
    if pinky_up:
        up_fingers.append("pinky");  tip_ids_up.append(PINKY_TIP)

    # If too many fingers are borderline, the whole result is unreliable
    if ambiguous_count >= 3:
        return {
            "count":       None,
            "count_text":  "Unknown",
            "states":      states,
            "up_fingers":  up_fingers,
            "tip_ids_up":  tip_ids_up,
            "reason":      "ambiguous_pose",
            "ambiguous":   ambiguous_count,
        }

    # Everything looks good — return the count
    count = len(up_fingers)
    return {
        "count":       count,
        "count_text":  str(count),
        "states":      states,
        "up_fingers":  up_fingers,
        "tip_ids_up":  tip_ids_up,
        "reason":      "ok",
        "ambiguous":   ambiguous_count,
    }

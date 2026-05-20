# =============================================================================
# finger_counter.py
# -----------------
# Determines which fingers are currently extended (pointing up/out) from a
# MediaPipe hand landmark object, and returns a structured result.
#
# How it works:
#   For each finger it measures three things:
#     1. PIP and DIP joint angles (are the joints straight?)
#     2. Tip-to-MCP distance (is the finger long enough to be extended?)
#     3. Tip-to-wrist vs PIP-to-wrist (is the tip actually further out?)
#   If all three pass, the finger is "up". If any are borderline, the finger
#   is flagged "ambiguous". Three or more ambiguous fingers = whole result
#   is marked uncertain.
#
# The thumb gets its own logic because it has different biomechanics and
# sits at the side of the hand rather than pointing up.
#
# Where it fits:
#   Called by process_hand_frame() in hand_landmarks.py once per frame.
#   Its output feeds directly into classify_rps_gesture() in gesture_mapper.py.
# =============================================================================

import math


# MediaPipe fingertip landmark indices -- these are fixed by the model, don't change them
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
    Uses the Z axis if it's available (some older devices omit it).
    """
    az = getattr(a, "z", 0.0)
    bz = getattr(b, "z", 0.0)
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (az - bz) ** 2)


def _distance_to_point(a, p):
    """
    3-D distance from a MediaPipe landmark to a plain (x, y, z) tuple.
    Used to measure how far the thumb tip is from the palm centre.
    """
    az = getattr(a, "z", 0.0)
    return math.sqrt((a.x - p[0]) ** 2 + (a.y - p[1]) ** 2 + (az - p[2]) ** 2)


def _angle(a, b, c):
    """
    Return the angle at vertex B (in degrees) formed by points A-B-C,
    using 3-D landmark coordinates.

    Builds vectors BA and BC, then uses the dot-product formula.
    Returns 0 degrees if either vector has zero length (degenerate case).
    """
    # Build vectors from B to A, and from B to C
    ab = (a.x - b.x, a.y - b.y, getattr(a, "z", 0.0) - getattr(b, "z", 0.0))
    cb = (c.x - b.x, c.y - b.y, getattr(c, "z", 0.0) - getattr(b, "z", 0.0))

    dot    = ab[0]*cb[0] + ab[1]*cb[1] + ab[2]*cb[2]
    mag_ab = math.sqrt(ab[0]**2 + ab[1]**2 + ab[2]**2)
    mag_cb = math.sqrt(cb[0]**2 + cb[1]**2 + cb[2]**2)

    if mag_ab == 0 or mag_cb == 0:
        return 0.0

    # Clamp to [-1, 1] before acos to guard against floating-point rounding
    cos_angle = max(-1.0, min(1.0, dot / (mag_ab * mag_cb)))
    return math.degrees(math.acos(cos_angle))


def _palm_size(lm):
    """
    Estimate the hand's apparent size as the average of three wrist-to-knuckle
    distances (index, middle, and pinky MCP joints).

    This gives a scale-invariant reference length so that the extension
    thresholds work correctly regardless of how far the hand is from the camera.
    Returns at least 1e-6 to prevent division by zero in callers.
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
    Approximate the palm centre as the average position of the wrist and
    the three main knuckle (MCP) joints. Returns an (x, y, z) tuple.
    Used as a reference point for the thumb extension check.
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
    Decide whether one non-thumb finger (index/middle/ring/pinky) is extended.

    Three conditions must ALL be true for a finger to count as "up":
      1. straight_enough  -- both joint angles (PIP and DIP) are close to 180,
                             meaning the finger is not curled
      2. long_enough      -- the tip-to-MCP distance is more than 55% of palm
                             scale, ruling out fingers folded sideways
      3. farther_than_pip -- the tip is further from the wrist than the PIP
                             joint plus a small margin (finger points outward)

    "Ambiguous" means we're right on the boundary of one of these checks
    and it could go either way. The caller uses this to reject uncertain results.

    Returns:
        (is_up: bool, ambiguous: bool)
    """
    wrist = lm[0]
    tip   = lm[tip_id]
    dip   = lm[dip_id]
    pip   = lm[pip_id]
    mcp   = lm[mcp_id]

    pip_angle = _angle(mcp, pip, dip)   # angle at the PIP (middle) joint
    dip_angle = _angle(pip, dip, tip)   # angle at the DIP (outer) joint

    tip_wrist = _distance(tip, wrist)
    pip_wrist = _distance(pip, wrist)
    mcp_tip   = _distance(mcp, tip)

    # All three must pass
    straight_enough  = pip_angle > 160 and dip_angle > 150
    long_enough      = mcp_tip > 0.55 * palm_scale
    farther_than_pip = tip_wrist > pip_wrist + 0.18 * palm_scale

    is_up = straight_enough and long_enough and farther_than_pip

    # Borderline: very close to a threshold in any of the three checks
    ambiguous = (
        150 <= pip_angle <= 160 or
        140 <= dip_angle <= 150 or
        abs(tip_wrist - (pip_wrist + 0.18 * palm_scale)) < 0.04 * palm_scale
    )

    return is_up, ambiguous


def _is_thumb_extended(lm, palm_scale):
    """
    Decide whether the thumb is extended (sticking outward from the hand).

    The thumb has different biomechanics and sits on the side of the hand,
    so it gets its own checks:
      1. straight_enough       -- MCP and IP joint angles are fairly open
      2. away_from_palm        -- thumb tip is clearly further from the palm
                                  centre than the IP joint
      3. separated_from_hand   -- tip-to-index-MCP distance is large enough
                                  to rule out a tucked-under thumb

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

    straight_enough     = mcp_angle > 135 and ip_angle > 145
    away_from_palm      = tip_palm > ip_palm + 0.10 * palm_scale
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
        hand_landmarks       -- MediaPipe NormalizedLandmarkList
        hand_label           -- "Left" or "Right" (from MediaPipe handedness)
        target_hand          -- "Left", "Right", or "Auto"
                                "Auto" accepts any hand regardless of label/score
        handedness_score     -- MediaPipe's confidence that hand_label is correct
        handedness_threshold -- minimum confidence required when not in Auto mode

    Returns a dict with:
        count        -- int number of extended fingers, or None if uncertain
        count_text   -- str(count) or "Unknown"
        states       -- {"thumb": bool, "index": bool, ...} per finger
        up_fingers   -- list of finger name strings that are extended
        tip_ids_up   -- list of MediaPipe tip landmark IDs for extended fingers
                        (used by the diagnostic overlay to draw red dots)
        reason       -- "ok" on success, or a short reason string on failure
        ambiguous    -- how many fingers were in a borderline state
    """
    # No landmarks at all -- return a clean "nothing to work with" result
    if hand_landmarks is None:
        return {
            "count":      None,
            "count_text": "Unknown",
            "states":     None,
            "up_fingers": [],
            "tip_ids_up": [],
            "reason":     "no_hand",
            "ambiguous":  0,
        }

    auto_mode = target_hand == "Auto"

    # In non-Auto mode: reject if this is the wrong hand
    if not auto_mode and hand_label != target_hand:
        return {
            "count":      None,
            "count_text": "Unknown",
            "states":     None,
            "up_fingers": [],
            "tip_ids_up": [],
            "reason":     "wrong_hand_mode",
            "ambiguous":  0,
        }

    # In non-Auto mode: reject if MediaPipe isn't confident about which hand this is
    if not auto_mode and handedness_score < handedness_threshold:
        return {
            "count":      None,
            "count_text": "Unknown",
            "states":     None,
            "up_fingers": [],
            "tip_ids_up": [],
            "reason":     "low_handedness_confidence",
            "ambiguous":  0,
        }

    lm         = hand_landmarks.landmark
    palm_scale = _palm_size(lm)

    # Check each finger -- landmark IDs are (tip, dip, pip, mcp) per finger
    thumb_up,  thumb_amb  = _is_thumb_extended(lm, palm_scale)
    index_up,  index_amb  = _is_finger_extended(lm, 8,  7,  6,  5,  palm_scale)
    middle_up, middle_amb = _is_finger_extended(lm, 12, 11, 10, 9,  palm_scale)
    ring_up,   ring_amb   = _is_finger_extended(lm, 16, 15, 14, 13, palm_scale)
    pinky_up,  pinky_amb  = _is_finger_extended(lm, 20, 19, 18, 17, palm_scale)

    # Build the per-finger True/False state dict for downstream use
    states = {
        "thumb":  thumb_up,
        "index":  index_up,
        "middle": middle_up,
        "ring":   ring_up,
        "pinky":  pinky_up,
    }

    # Count how many fingers were right on the threshold (borderline)
    ambiguous_count = sum([thumb_amb, index_amb, middle_amb, ring_amb, pinky_amb])

    # Build the list of extended finger names and their tip landmark IDs
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

    # If too many fingers are borderline, the whole result is unreliable --
    # return Unknown so downstream code doesn't act on a shaky reading
    if ambiguous_count >= 3:
        return {
            "count":      None,
            "count_text": "Unknown",
            "states":     states,
            "up_fingers": up_fingers,
            "tip_ids_up": tip_ids_up,
            "reason":     "ambiguous_pose",
            "ambiguous":  ambiguous_count,
        }

    # All good -- return the count
    count = len(up_fingers)
    return {
        "count":      count,
        "count_text": str(count),
        "states":     states,
        "up_fingers": up_fingers,
        "tip_ids_up": tip_ids_up,
        "reason":     "ok",
        "ambiguous":  ambiguous_count,
    }

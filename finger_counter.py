import math


# Fingertip landmark IDs
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20


def _distance(a, b):
    """3D Euclidean distance between two landmarks."""
    az = getattr(a, "z", 0.0)
    bz = getattr(b, "z", 0.0)
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (az - bz) ** 2)


def _distance_to_point(a, p):
    """3D Euclidean distance between a landmark and an (x, y, z) tuple."""
    az = getattr(a, "z", 0.0)
    return math.sqrt((a.x - p[0]) ** 2 + (a.y - p[1]) ** 2 + (az - p[2]) ** 2)


def _angle(a, b, c):
    """
    Returns angle ABC in degrees using 3D landmark coordinates.
    """
    ab = (
        a.x - b.x,
        a.y - b.y,
        getattr(a, "z", 0.0) - getattr(b, "z", 0.0)
    )
    cb = (
        c.x - b.x,
        c.y - b.y,
        getattr(c, "z", 0.0) - getattr(b, "z", 0.0)
    )

    dot = ab[0] * cb[0] + ab[1] * cb[1] + ab[2] * cb[2]
    mag_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2 + ab[2] ** 2)
    mag_cb = math.sqrt(cb[0] ** 2 + cb[1] ** 2 + cb[2] ** 2)

    if mag_ab == 0 or mag_cb == 0:
        return 0.0

    cos_angle = dot / (mag_ab * mag_cb)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def _palm_size(lm):
    """
    Estimates hand scale so thresholds adapt to different hand sizes/distances.
    """
    wrist = lm[0]
    index_mcp = lm[5]
    middle_mcp = lm[9]
    pinky_mcp = lm[17]

    d1 = _distance(wrist, index_mcp)
    d2 = _distance(wrist, middle_mcp)
    d3 = _distance(wrist, pinky_mcp)

    return max((d1 + d2 + d3) / 3.0, 1e-6)


def _palm_center(lm):
    """
    Approximate palm center from wrist + main MCP joints.
    """
    pts = [lm[0], lm[5], lm[9], lm[17]]
    x = sum(p.x for p in pts) / len(pts)
    y = sum(p.y for p in pts) / len(pts)
    z = sum(getattr(p, "z", 0.0) for p in pts) / len(pts)
    return (x, y, z)


def _is_finger_extended(lm, tip_id, dip_id, pip_id, mcp_id, palm_scale):
    """
    Rotation-tolerant rule for index/middle/ring/pinky.
    """
    wrist = lm[0]
    tip = lm[tip_id]
    dip = lm[dip_id]
    pip = lm[pip_id]
    mcp = lm[mcp_id]

    pip_angle = _angle(mcp, pip, dip)
    dip_angle = _angle(pip, dip, tip)

    tip_wrist = _distance(tip, wrist)
    pip_wrist = _distance(pip, wrist)
    mcp_tip = _distance(mcp, tip)

    straight_enough = pip_angle > 160 and dip_angle > 150
    long_enough = mcp_tip > 0.55 * palm_scale
    farther_than_pip = tip_wrist > pip_wrist + 0.18 * palm_scale

    is_up = straight_enough and long_enough and farther_than_pip

    ambiguous = (
        150 <= pip_angle <= 160 or
        140 <= dip_angle <= 150 or
        abs(tip_wrist - (pip_wrist + 0.18 * palm_scale)) < 0.04 * palm_scale
    )

    return is_up, ambiguous


def _is_thumb_extended(lm, palm_scale):
    """
    Rotation-tolerant thumb rule.
    """
    thumb_cmc = lm[1]
    thumb_mcp = lm[2]
    thumb_ip = lm[3]
    thumb_tip = lm[4]
    index_mcp = lm[5]

    palm_center = _palm_center(lm)

    mcp_angle = _angle(thumb_cmc, thumb_mcp, thumb_ip)
    ip_angle = _angle(thumb_mcp, thumb_ip, thumb_tip)

    tip_palm = _distance_to_point(thumb_tip, palm_center)
    ip_palm = _distance_to_point(thumb_ip, palm_center)
    tip_index = _distance(thumb_tip, index_mcp)

    straight_enough = mcp_angle > 135 and ip_angle > 145
    away_from_palm = tip_palm > ip_palm + 0.10 * palm_scale
    separated_from_hand = tip_index > 0.35 * palm_scale

    is_up = straight_enough and away_from_palm and separated_from_hand

    ambiguous = (
        125 <= mcp_angle <= 135 or
        135 <= ip_angle <= 145 or
        abs(tip_palm - (ip_palm + 0.10 * palm_scale)) < 0.03 * palm_scale
    )

    return is_up, ambiguous


def count_hand_fingers(
    hand_landmarks,
    hand_label,
    target_hand="Auto",
    handedness_score=1.0,
    handedness_threshold=0.80
):
    """
    General hand counter for Right, Left, or Auto mode.

    Auto mode:
    - accepts either hand
    - does not reject low handedness confidence
    """
    if hand_landmarks is None:
        return {
            "count": None,
            "count_text": "Unknown",
            "states": None,
            "up_fingers": [],
            "tip_ids_up": [],
            "reason": "no_hand",
            "ambiguous": 0
        }

    auto_mode = target_hand == "Auto"

    if (not auto_mode) and hand_label != target_hand:
        return {
            "count": None,
            "count_text": "Unknown",
            "states": None,
            "up_fingers": [],
            "tip_ids_up": [],
            "reason": "wrong_hand_mode",
            "ambiguous": 0
        }

    if (not auto_mode) and handedness_score < handedness_threshold:
        return {
            "count": None,
            "count_text": "Unknown",
            "states": None,
            "up_fingers": [],
            "tip_ids_up": [],
            "reason": "low_handedness_confidence",
            "ambiguous": 0
        }

    lm = hand_landmarks.landmark
    palm_scale = _palm_size(lm)

    thumb_up, thumb_amb = _is_thumb_extended(lm, palm_scale)
    index_up, index_amb = _is_finger_extended(lm, 8, 7, 6, 5, palm_scale)
    middle_up, middle_amb = _is_finger_extended(lm, 12, 11, 10, 9, palm_scale)
    ring_up, ring_amb = _is_finger_extended(lm, 16, 15, 14, 13, palm_scale)
    pinky_up, pinky_amb = _is_finger_extended(lm, 20, 19, 18, 17, palm_scale)

    states = {
        "thumb": thumb_up,
        "index": index_up,
        "middle": middle_up,
        "ring": ring_up,
        "pinky": pinky_up
    }

    ambiguous_count = sum([
        thumb_amb,
        index_amb,
        middle_amb,
        ring_amb,
        pinky_amb
    ])

    up_fingers = []
    tip_ids_up = []

    if thumb_up:
        up_fingers.append("thumb")
        tip_ids_up.append(THUMB_TIP)
    if index_up:
        up_fingers.append("index")
        tip_ids_up.append(INDEX_TIP)
    if middle_up:
        up_fingers.append("middle")
        tip_ids_up.append(MIDDLE_TIP)
    if ring_up:
        up_fingers.append("ring")
        tip_ids_up.append(RING_TIP)
    if pinky_up:
        up_fingers.append("pinky")
        tip_ids_up.append(PINKY_TIP)

    if ambiguous_count >= 3:
        return {
            "count": None,
            "count_text": "Unknown",
            "states": states,
            "up_fingers": up_fingers,
            "tip_ids_up": tip_ids_up,
            "reason": "ambiguous_pose",
            "ambiguous": ambiguous_count
        }

    count = len(up_fingers)

    return {
        "count": count,
        "count_text": str(count),
        "states": states,
        "up_fingers": up_fingers,
        "tip_ids_up": tip_ids_up,
        "reason": "ok",
        "ambiguous": ambiguous_count
    }
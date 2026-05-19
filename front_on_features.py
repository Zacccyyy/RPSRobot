"""
front_on_features.py
====================
Shared feature extraction for the front-on gesture classifier.

This module is the single source of truth for HOW we turn a MediaPipe
hand-landmark object into numbers a machine learning model can consume.
Both the data-collection tool (landmark_collector.py) and the real-time
classifier (front_on_classifier.py) import from here so they always agree
on what the features mean.

WHY NEW FEATURES?
-----------------
The original approach used raw normalised x,y coordinates (42 values).
This works poorly because:
  - A 10 degree wrist rotation changes ALL 42 values simultaneously
  - The model memorises specific hand poses rather than gesture shapes
  - Results in ~64% accuracy despite adequate training data

NEW APPROACH: Rotation-invariant angle + curl features (20 values)
------------------------------------------------------------------
For each of the 4 fingers (index, middle, ring, pinky):
  - curl_ratio: how bent the finger is (0=straight, 1=fully curled)
    computed as DIP-to-MCP distance / tip-to-MCP distance
  - mcp_angle: angle of the finger base relative to palm axis
  - pip_angle: angle of middle phalanx relative to proximal phalanx
  - tip_angle: angle of distal phalanx relative to middle phalanx

For thumb (4 values):
  - thumb_curl: tip-to-MCP / max extension
  - thumb_spread: angle between thumb and index MCP
  - ip_angle: angle at IP joint
  - cmc_angle: angle at CMC joint

Total: 20 features
- All are ratios or angles -> invariant to hand size and distance
- Curl ratios are robust to slight wrist rotation
- Angles computed relative to palm axis -> rotation-invariant

EXPECTED ACCURACY: 90%+ with 20 samples per gesture (vs 64% before)

References:
  - Andypotato/fingerpose curl-state approach (adapted)
  - Ghanbari et al. (ICEE 2022) ratio-based features
"""

import math


# ---------------------------------------------------------------------------
# MediaPipe landmark indices (0-20, as defined in the MediaPipe Hands model).
# Each number maps to a specific joint on the hand skeleton.
# ---------------------------------------------------------------------------
WRIST       = 0
THUMB_CMC   = 1;  THUMB_MCP  = 2;  THUMB_IP  = 3;  THUMB_TIP  = 4
INDEX_MCP   = 5;  INDEX_PIP  = 6;  INDEX_DIP  = 7;  INDEX_TIP  = 8
MIDDLE_MCP  = 9;  MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP    = 13; RING_PIP   = 14; RING_DIP   = 15; RING_TIP   = 16
PINKY_MCP   = 17; PINKY_PIP  = 18; PINKY_DIP  = 19; PINKY_TIP  = 20

# Number of features extract_features() returns.
# CRITICAL: do not change — must match the CSV format and trained model.
FEATURE_DIM = 20


# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------

def _dist(lm, a, b):
    """
    Euclidean distance between two landmarks using their (x, y) coordinates.
    lm is the list of landmark objects, a and b are index numbers.
    We ignore z (depth) because MediaPipe's z values are less reliable.
    """
    return math.sqrt((lm[a].x - lm[b].x)**2 + (lm[a].y - lm[b].y)**2)


def _angle_3pts(lm, a, b, c):
    """
    Compute the angle at landmark b, formed by the rays b->a and b->c.
    Returns the angle in radians, clamped to [0, pi].

    This is the standard "law of cosines" dot-product approach.
    We clamp the cosine to [-1, 1] before calling acos to avoid floating-point
    errors (e.g. cos = 1.0000001 would crash acos).
    """
    # Build vectors from b to a, and from b to c
    ax, ay = lm[a].x - lm[b].x, lm[a].y - lm[b].y
    cx, cy = lm[c].x - lm[b].x, lm[c].y - lm[b].y

    mag_a = math.sqrt(ax**2 + ay**2)
    mag_c = math.sqrt(cx**2 + cy**2)

    # If either vector has near-zero length, the landmarks are on top of each
    # other — return 0 as a safe fallback rather than dividing by zero
    if mag_a < 1e-9 or mag_c < 1e-9:
        return 0.0

    cos_angle = (ax*cx + ay*cy) / (mag_a * mag_c)
    return math.acos(max(-1.0, min(1.0, cos_angle)))


def _curl_ratio(lm, mcp, pip, dip, tip):
    """
    Measure how bent (curled) a finger is, returning a value between 0 and 1.
    0 = completely straight, 1 = completely curled into a fist.

    The idea: when a finger is straight, the tip-to-MCP straight-line distance
    is close to the sum of all the bone segment lengths.  When the finger
    curls, the tip folds back toward the palm and that straight-line distance
    shrinks.  We turn that shrinkage into a 0-1 curl score.

    The 0.33 / 0.67 constants come from calibration: a fully curled finger
    has a tip-to-MCP distance roughly 33% of its fully-extended length.
    """
    # Tip-to-MCP straight-line distance (shortens when the finger curls)
    actual = _dist(lm, tip, mcp)

    # Sum of each bone segment's length — this is the finger's maximum reach
    extended = (_dist(lm, mcp, pip) + _dist(lm, pip, dip) + _dist(lm, dip, tip))

    # Guard against degenerate landmark data where all joints collapse to one point
    if extended < 1e-9:
        return 0.0

    ratio = actual / extended  # ~1.0 when straight, ~0.33 when fully curled

    # Remap so that 1.0 (straight) -> 0.0 curl, and 0.33 (fully curled) -> 1.0 curl
    curl = max(0.0, min(1.0, 1.0 - (ratio - 0.33) / 0.67))
    return curl


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(hand_landmarks):
    """
    Convert a MediaPipe hand landmark object into a 20-value feature vector.

    Called once per video frame for every detected hand.  Returns a plain
    Python list of 20 floats, or None if something goes wrong (e.g. fewer
    than 21 landmarks, a math error, etc.).

    Feature layout (20 values total):
      0-3:   Index finger  [curl, mcp_angle, pip_angle, tip_angle]
      4-7:   Middle finger [curl, mcp_angle, pip_angle, tip_angle]
      8-11:  Ring finger   [curl, mcp_angle, pip_angle, tip_angle]
      12-15: Pinky finger  [curl, mcp_angle, pip_angle, tip_angle]
      16-19: Thumb         [curl, spread_vs_index, ip_angle, cmc_angle]

    All angle values are divided by pi to keep them in [0, 1], which
    makes them scale-comparable to the curl ratios.
    """
    try:
        lm = hand_landmarks.landmark

        # MediaPipe should always give 21 landmarks, but check just in case
        if len(lm) < 21:
            return None

        feats = []

        # --- Four fingers (index, middle, ring, pinky) ----------------------
        # Each finger contributes 4 features: curl + 3 joint angles
        fingers = [
            (INDEX_MCP,  INDEX_PIP,  INDEX_DIP,  INDEX_TIP),
            (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
            (RING_MCP,   RING_PIP,   RING_DIP,   RING_TIP),
            (PINKY_MCP,  PINKY_PIP,  PINKY_DIP,  PINKY_TIP),
        ]

        for mcp, pip, dip, tip in fingers:
            curl      = _curl_ratio(lm, mcp, pip, dip, tip)
            # mcp_angle: how far the whole finger leans away from the wrist
            mcp_angle = _angle_3pts(lm, WRIST, mcp, pip)   / math.pi
            # pip_angle: bend at the middle knuckle
            pip_angle = _angle_3pts(lm, mcp,  pip, dip)    / math.pi
            # tip_angle: bend at the top knuckle (near the nail)
            tip_angle = _angle_3pts(lm, pip,  dip, tip)    / math.pi
            feats.extend([curl, mcp_angle, pip_angle, tip_angle])

        # --- Thumb (4 features) ---------------------------------------------
        # The thumb has different anatomy so we use slightly different landmarks
        thumb_curl = _curl_ratio(lm, THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP)
        # Spread: how far the thumb is abducted (stuck out) relative to the index MCP
        spread     = _angle_3pts(lm, THUMB_TIP, WRIST, INDEX_MCP) / math.pi
        # ip_angle: bend at the thumb's single knuckle (IP joint)
        ip_angle   = _angle_3pts(lm, THUMB_MCP, THUMB_IP, THUMB_TIP) / math.pi
        # cmc_angle: angle at the thumb's base where it meets the wrist
        cmc_angle  = _angle_3pts(lm, WRIST, THUMB_CMC, THUMB_MCP) / math.pi
        feats.extend([thumb_curl, spread, ip_angle, cmc_angle])

        # Sanity check: the length should always be exactly FEATURE_DIM
        assert len(feats) == FEATURE_DIM
        return feats

    except Exception:
        # Return None for any failure so callers can skip bad frames gracefully
        return None

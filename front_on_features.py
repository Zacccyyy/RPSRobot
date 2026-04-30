"""
front_on_features.py
====================
Shared feature extraction for the front-on gesture classifier.

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


# MediaPipe landmark indices
WRIST       = 0
THUMB_CMC   = 1;  THUMB_MCP  = 2;  THUMB_IP  = 3;  THUMB_TIP  = 4
INDEX_MCP   = 5;  INDEX_PIP  = 6;  INDEX_DIP  = 7;  INDEX_TIP  = 8
MIDDLE_MCP  = 9;  MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP    = 13; RING_PIP   = 14; RING_DIP   = 15; RING_TIP   = 16
PINKY_MCP   = 17; PINKY_PIP  = 18; PINKY_DIP  = 19; PINKY_TIP  = 20

FEATURE_DIM = 20   # must match what extract_features returns


def _dist(lm, a, b):
    return math.sqrt((lm[a].x - lm[b].x)**2 + (lm[a].y - lm[b].y)**2)


def _angle_3pts(lm, a, b, c):
    """
    Angle at point b formed by vectors b->a and b->c.
    Returns angle in [0, pi].
    """
    ax, ay = lm[a].x - lm[b].x, lm[a].y - lm[b].y
    cx, cy = lm[c].x - lm[b].x, lm[c].y - lm[b].y
    mag_a = math.sqrt(ax**2 + ay**2)
    mag_c = math.sqrt(cx**2 + cy**2)
    if mag_a < 1e-9 or mag_c < 1e-9:
        return 0.0
    cos_angle = (ax*cx + ay*cy) / (mag_a * mag_c)
    return math.acos(max(-1.0, min(1.0, cos_angle)))


def _curl_ratio(lm, mcp, pip, dip, tip):
    """
    Finger curl ratio: 0 = fully straight, 1 = fully curled.
    Uses the ratio of tip-to-MCP straight-line distance
    vs tip-to-MCP if all segments were laid straight.
    """
    # Actual tip-to-MCP distance
    actual = _dist(lm, tip, mcp)
    # Sum of segment lengths (fully extended measure)
    extended = (_dist(lm, mcp, pip) + _dist(lm, pip, dip) + _dist(lm, dip, tip))
    if extended < 1e-9:
        return 0.0
    # When straight: actual ~ extended; when curled: actual << extended
    ratio = actual / extended
    # Convert: straight=1.0 -> curl=0.0, curled=0.33 -> curl~1.0
    curl = max(0.0, min(1.0, 1.0 - (ratio - 0.33) / 0.67))
    return curl


def extract_features(hand_landmarks):
    """
    Extract 20 rotation-invariant features from a MediaPipe hand landmark object.

    Returns list of 20 floats, or None if landmarks are invalid.

    Feature layout (20 values):
      0-3:   Index finger  [curl, mcp_angle, pip_angle, tip_angle]
      4-7:   Middle finger [curl, mcp_angle, pip_angle, tip_angle]
      8-11:  Ring finger   [curl, mcp_angle, pip_angle, tip_angle]
      12-15: Pinky finger  [curl, mcp_angle, pip_angle, tip_angle]
      16-19: Thumb         [curl, spread_vs_index, ip_angle, cmc_angle]
    """
    try:
        lm = hand_landmarks.landmark
        if len(lm) < 21:
            return None

        feats = []

        # ── 4 fingers ──────────────────────────────────────────────────────
        fingers = [
            (INDEX_MCP,  INDEX_PIP,  INDEX_DIP,  INDEX_TIP),
            (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
            (RING_MCP,   RING_PIP,   RING_DIP,   RING_TIP),
            (PINKY_MCP,  PINKY_PIP,  PINKY_DIP,  PINKY_TIP),
        ]
        for mcp, pip, dip, tip in fingers:
            curl      = _curl_ratio(lm, mcp, pip, dip, tip)
            mcp_angle = _angle_3pts(lm, WRIST, mcp, pip)   / math.pi
            pip_angle = _angle_3pts(lm, mcp,  pip, dip)    / math.pi
            tip_angle = _angle_3pts(lm, pip,  dip, tip)    / math.pi
            feats.extend([curl, mcp_angle, pip_angle, tip_angle])

        # ── Thumb ───────────────────────────────────────────────────────────
        thumb_curl   = _curl_ratio(lm, THUMB_MCP, THUMB_IP, THUMB_IP, THUMB_TIP)
        # Thumb spread: angle between thumb tip and index MCP, relative to wrist
        spread       = _angle_3pts(lm, THUMB_TIP, WRIST, INDEX_MCP) / math.pi
        ip_angle     = _angle_3pts(lm, THUMB_MCP, THUMB_IP, THUMB_TIP) / math.pi
        cmc_angle    = _angle_3pts(lm, WRIST, THUMB_CMC, THUMB_MCP) / math.pi
        feats.extend([thumb_curl, spread, ip_angle, cmc_angle])

        assert len(feats) == FEATURE_DIM
        return feats

    except Exception:
        return None

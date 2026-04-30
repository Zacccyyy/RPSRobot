"""
Front-On Gesture Classifier — Hybrid ML + Curl Analysis

Two layers working together:

    Layer 1: ML MODEL (trained on your hand data)
        - Analyses normalised landmark positions
        - Best for static poses with high accuracy
        - Can lag during fast transitions (pump→shoot)

    Layer 2: CURL ANALYSIS (real-time angle measurement)
        - Measures PIP and DIP joint angles per finger
        - Classifies each finger as: NoCurl / HalfCurl / FullCurl
        - Very fast to respond to finger movement
        - Rotation-invariant (uses angles, not positions)

    Combination logic:
        - If ML is confident (>70%), trust ML
        - If ML is uncertain AND curl gives a clear signal, trust curl
        - If both agree, extra confidence
        - Diagnostic shows both signals for debugging

Model path: ~/Desktop/CapStone/front_on_gesture_model.pkl
"""

import math
from front_on_features import extract_features as _extract_features
from pathlib import Path

# Lazy-loaded model cache.
_cached_model = None
_cached_labels = None
_model_checked = False

# ============================================================
# CURL ANALYSIS
# ============================================================

# Landmark IDs for each finger's joint chain.
# (name, mcp_id, pip_id, dip_id, tip_id)
FINGER_JOINTS = [
    ("index",  5,  6,  7,  8),
    ("middle", 9,  10, 11, 12),
    ("ring",   13, 14, 15, 16),
    ("pinky",  17, 18, 19, 20),
]

# Curl thresholds (angle at PIP joint).
# These are the angles between MCP→PIP→DIP.
# Straight finger ≈ 170-180°, fully curled ≈ 40-80°.
CURL_NO = 150       # above this = NoCurl (finger is straight)
CURL_HALF = 110     # above this but below CURL_NO = HalfCurl
                    # below CURL_HALF = FullCurl


def _angle_3pt(a, b, c):
    """
    Angle ABC in degrees using 2D coordinates.
    B is the vertex (joint being measured).
    """
    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)

    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)

    if mag_ba < 1e-8 or mag_bc < 1e-8:
        return 180.0

    cos_angle = dot / (mag_ba * mag_bc)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def _finger_curl(lm, mcp_id, pip_id, dip_id, tip_id):
    """
    Measure curl state of a single finger.
    Returns (curl_label, pip_angle, dip_angle).
    """
    pip_angle = _angle_3pt(lm[mcp_id], lm[pip_id], lm[dip_id])
    dip_angle = _angle_3pt(lm[pip_id], lm[dip_id], lm[tip_id])

    # Use the more bent of the two joints to classify curl.
    min_angle = min(pip_angle, dip_angle)

    if min_angle >= CURL_NO:
        curl = "NoCurl"
    elif min_angle >= CURL_HALF:
        curl = "HalfCurl"
    else:
        curl = "FullCurl"

    return curl, pip_angle, dip_angle


def _curl_classify(hand_landmarks):
    """
    Classify gesture purely from curl analysis.
    Returns (gesture, confidence, debug_str).

    confidence: 0.0 to 1.0 indicating how clear the curl signal is.
    """
    lm = hand_landmarks.landmark

    curls = {}
    angles = {}
    for name, mcp, pip, dip, tip in FINGER_JOINTS:
        curl, pip_a, dip_a = _finger_curl(lm, mcp, pip, dip, tip)
        curls[name] = curl
        angles[name] = (pip_a, dip_a)

    no_curl_count = sum(1 for c in curls.values() if c == "NoCurl")
    full_curl_count = sum(1 for c in curls.values() if c == "FullCurl")
    half_curl_count = sum(1 for c in curls.values() if c == "HalfCurl")

    # Debug string.
    dbg_parts = []
    for name in ("index", "middle", "ring", "pinky"):
        c = curls[name]
        pip_a, dip_a = angles[name]
        short = {"NoCurl": "O", "HalfCurl": "H", "FullCurl": "C"}[c]
        dbg_parts.append(f"{name[0]}:{short}{min(pip_a, dip_a):.0f}")
    dbg = " ".join(dbg_parts)

    # Classification.
    if full_curl_count == 4:
        return "Rock", 0.95, dbg

    if no_curl_count >= 3:
        # Could be Paper or Scissors — check ring+pinky specifically.
        if curls["ring"] == "NoCurl" and curls["pinky"] == "NoCurl":
            return "Paper", 0.85, dbg
        elif curls["index"] == "NoCurl" and curls["middle"] == "NoCurl":
            # Index+middle open but ring+pinky not fully open.
            if curls["ring"] in ("FullCurl", "HalfCurl") or curls["pinky"] in ("FullCurl", "HalfCurl"):
                return "Scissors", 0.80, dbg
            else:
                return "Paper", 0.75, dbg
        else:
            return "Paper", 0.70, dbg

    if full_curl_count >= 3:
        return "Rock", 0.80, dbg

    # Scissors: index+middle open, ring+pinky curled.
    if (curls["index"] == "NoCurl" and curls["middle"] == "NoCurl"
            and curls["ring"] in ("FullCurl", "HalfCurl")
            and curls["pinky"] in ("FullCurl", "HalfCurl")):
        return "Scissors", 0.90, dbg

    # Partial signals.
    if no_curl_count == 2 and curls["index"] == "NoCurl" and curls["middle"] == "NoCurl":
        return "Scissors", 0.65, dbg

    if no_curl_count >= 2 and full_curl_count <= 1:
        return "Paper", 0.50, dbg

    if full_curl_count >= 2 and no_curl_count <= 1:
        return "Rock", 0.55, dbg

    return "Unknown", 0.0, dbg


# ============================================================
# ML MODEL
# ============================================================

def _load_model_once():
    """Load the trained model once, cache the result."""
    global _cached_model, _cached_labels, _model_checked

    if _model_checked:
        return _cached_model, _cached_labels

    _model_checked = True

    model_path = Path.home() / "Desktop" / "CapStone" / "front_on_gesture_model.pkl"

    if not model_path.exists():
        print("[FrontOn] No trained model found. Use Diagnostic mode to collect data.")
        return None, None

    try:
        import pickle
        with open(model_path, "rb") as f:
            data = pickle.load(f)

        _cached_model = data["model"]
        _cached_labels = data["int_to_label"]
        n = data.get("n_samples", "?")
        acc = data.get("accuracy")
        acc_str = f"{acc:.0%}" if acc else "unknown"
        print(f"[FrontOn] Loaded ML model ({n} samples, {acc_str} accuracy)")
        return _cached_model, _cached_labels

    except Exception as exc:
        print(f"[FrontOn] Failed to load model: {exc}")
        return None, None


def _normalise_landmarks(hand_landmarks):
    """Normalise 21 landmarks: translate to wrist, scale by palm size."""
    lm = hand_landmarks.landmark
    wrist_x = lm[0].x
    wrist_y = lm[0].y

    def _d(a, b):
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    d1 = _d(lm[0], lm[5])
    d2 = _d(lm[0], lm[9])
    d3 = _d(lm[0], lm[17])
    palm_scale = max((d1 + d2 + d3) / 3.0, 1e-6)

    row = []
    for i in range(21):
        row.append((lm[i].x - wrist_x) / palm_scale)
        row.append((lm[i].y - wrist_y) / palm_scale)
    return row


def _ml_classify(hand_landmarks):
    """
    Classify using the trained ML model.
    Returns (gesture, confidence, prob_str) or (None, 0, "").
    """
    model, int_to_label = _load_model_once()
    if model is None:
        return None, 0.0, "no_model"

    features = _normalise_landmarks(hand_landmarks)
    if features is None:
        return None, 0.0, "normalise_failed"

    try:
        import numpy as np
        X = np.array([features], dtype=np.float32)

        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]

        gesture = int_to_label.get(prediction, "Unknown")
        confidence = float(probabilities[prediction])

        sorted_probs = sorted(
            [(int_to_label[i], p) for i, p in enumerate(probabilities)],
            key=lambda x: x[1],
            reverse=True
        )
        prob_str = " ".join(f"{n[0]}:{p:.0%}" for n, p in sorted_probs)

        return gesture, confidence, prob_str

    except Exception:
        return None, 0.0, "ml_error"


# ============================================================
# HYBRID CLASSIFIER
# ============================================================

# Threshold: above this, ML is trusted alone.
ML_CONFIDENCE_THRESHOLD = 0.70


def classify_front_on(hand_landmarks):
    """
    Classify RPS gesture using hybrid ML + curl analysis.
    """
    # Run both classifiers.
    curl_gesture, curl_conf, curl_dbg = _curl_classify(hand_landmarks)
    ml_gesture, ml_conf, ml_dbg = _ml_classify(hand_landmarks)

    cmd_map = {"Rock": "CMD_ROCK", "Paper": "CMD_PAPER",
               "Scissors": "CMD_SCISSORS"}

    # Decision logic.
    if ml_gesture is not None and ml_conf >= ML_CONFIDENCE_THRESHOLD:
        if ml_gesture == curl_gesture:
            # Both agree — highest confidence.
            gesture = ml_gesture
            reason = f"agree ml={ml_dbg} curl={curl_dbg}"
        else:
            # ML is confident but curl disagrees.
            # Trust ML unless curl has very high confidence on a different answer.
            if curl_conf >= 0.85 and ml_conf < 0.85:
                gesture = curl_gesture
                reason = f"curl_override ml={ml_dbg} curl={curl_dbg}"
            else:
                gesture = ml_gesture
                reason = f"ml_wins ml={ml_dbg} curl={curl_dbg}"
    elif ml_gesture is not None:
        # ML has low confidence.
        if curl_conf >= 0.60:
            # Curl has a reasonable signal — use it.
            gesture = curl_gesture
            reason = f"curl_leads ml={ml_dbg} curl={curl_dbg}"
        else:
            # Both uncertain — use ML's best guess.
            gesture = ml_gesture
            reason = f"ml_weak ml={ml_dbg} curl={curl_dbg}"
    else:
        # No ML model — curl only.
        if curl_conf >= 0.50:
            gesture = curl_gesture
            reason = f"curl_only {curl_dbg}"
        else:
            gesture = "Unknown"
            reason = f"uncertain {curl_dbg}"

    cmd = cmd_map.get(gesture, "CMD_UNKNOWN")

    return {
        "gesture": gesture,
        "command": cmd,
        "reason": reason,
    }


def reload_model():
    """Force reload the model (called after retraining)."""
    global _cached_model, _cached_labels, _model_checked
    _cached_model = None
    _cached_labels = None
    _model_checked = False
    return _load_model_once()
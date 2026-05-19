"""
front_on_classifier.py
======================
Real-time RPS gesture classifier used when the player faces the camera
front-on (palm toward lens).

HOW IT WORKS — two layers that vote:

    Layer 1: ML MODEL (trained on your hand data)
        - Analyses normalised landmark positions
        - Best for static poses with high accuracy
        - Can lag during fast transitions (pump -> shoot)

    Layer 2: CURL ANALYSIS (real-time angle measurement)
        - Measures PIP and DIP joint angles per finger
        - Classifies each finger as: NoCurl / HalfCurl / FullCurl
        - Very fast to respond to finger movement
        - Rotation-invariant (uses angles, not positions)

    Combination logic:
        - If ML is confident (>70%), trust ML
        - If ML is uncertain AND curl gives a clear signal, trust curl
        - If both agree, extra confidence
        - The "reason" field in the return value shows which path was taken
          so you can debug misclassifications

Model path: ~/Desktop/CapStone/front_on_gesture_model.pkl
"""

import math
from pathlib import Path
from front_on_features import extract_features as _extract_features


# ---------------------------------------------------------------------------
# Model cache — we load the model file once and keep it in memory.
# Using module-level globals avoids the overhead of re-reading the pickle
# on every frame.
# ---------------------------------------------------------------------------
_cached_model   = None
_cached_labels  = None
_model_checked  = False   # set to True after the first load attempt


# ---------------------------------------------------------------------------
# CURL ANALYSIS
# ---------------------------------------------------------------------------

# Landmark IDs for each finger's joint chain: (name, mcp, pip, dip, tip).
# These numbers are MediaPipe's fixed indices for each hand joint.
FINGER_JOINTS = [
    ("index",  5,  6,  7,  8),
    ("middle", 9,  10, 11, 12),
    ("ring",   13, 14, 15, 16),
    ("pinky",  17, 18, 19, 20),
]

# Curl classification thresholds (degrees at the PIP joint).
# A straight finger reads ~170-180 degrees; fully curled reads ~40-80 degrees.
CURL_NO   = 150    # above this -> NoCurl (finger is straight)
CURL_HALF = 110    # between CURL_HALF and CURL_NO -> HalfCurl
                   # below CURL_HALF -> FullCurl


def _angle_3pt(a, b, c):
    """
    Angle ABC in degrees, where B is the vertex (the joint we are measuring).
    Uses 2D (x, y) coordinates from landmark objects.

    Returns 180.0 (straight) if either vector is degenerate — this is a safe
    default that won't accidentally classify a straight finger as curled.
    """
    # Build 2D vectors from the vertex B outward to A and C
    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)

    dot    = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)

    # Avoid division by zero when two landmarks happen to coincide
    if mag_ba < 1e-8 or mag_bc < 1e-8:
        return 180.0

    # Clamp before acos to handle floating-point overshoot (e.g. 1.0000001)
    cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


def _finger_curl(lm, mcp_id, pip_id, dip_id, tip_id):
    """
    Determine how curled a single finger is.

    We measure the angle at both the PIP (middle knuckle) and DIP (top knuckle)
    joints, then use whichever is more bent.  This catches fingers that curl
    primarily at one joint versus the other.

    Returns:
        (curl_label, pip_angle, dip_angle)
        curl_label is one of "NoCurl", "HalfCurl", "FullCurl"
    """
    pip_angle = _angle_3pt(lm[mcp_id], lm[pip_id], lm[dip_id])
    dip_angle = _angle_3pt(lm[pip_id], lm[dip_id], lm[tip_id])

    # Use the tighter (smaller) angle as the classification criterion,
    # since one very bent joint is enough to call the finger curled
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
    Classify the RPS gesture using only finger-curl analysis (no ML model).

    This is fast and reliable for clear gestures, but less accurate for
    borderline poses (e.g. a fist with one finger slightly extended).

    Returns:
        (gesture, confidence, debug_str)
        gesture:    "Rock", "Paper", "Scissors", or "Unknown"
        confidence: 0.0 to 1.0 — how certain we are
        debug_str:  one-letter curl code + angle per finger, for logging
    """
    lm = hand_landmarks.landmark

    # Measure each finger and store results in dicts keyed by finger name
    curls  = {}
    angles = {}
    for name, mcp, pip, dip, tip in FINGER_JOINTS:
        curl, pip_a, dip_a = _finger_curl(lm, mcp, pip, dip, tip)
        curls[name]  = curl
        angles[name] = (pip_a, dip_a)

    # Count how many fingers are in each curl state
    no_curl_count   = sum(1 for c in curls.values() if c == "NoCurl")
    full_curl_count = sum(1 for c in curls.values() if c == "FullCurl")

    # Build a compact debug string like "i:O170 m:C45 r:C40 p:C38"
    # O=open, H=half, C=curled
    short_code = {"NoCurl": "O", "HalfCurl": "H", "FullCurl": "C"}
    dbg_parts = []
    for name in ("index", "middle", "ring", "pinky"):
        c      = curls[name]
        pip_a, dip_a = angles[name]
        dbg_parts.append(f"{name[0]}:{short_code[c]}{min(pip_a, dip_a):.0f}")
    dbg = " ".join(dbg_parts)

    # --- Classification rules (ordered from most to least specific) ---------

    # Rock: all four fingers fully curled into a fist
    if full_curl_count == 4:
        return "Rock", 0.95, dbg

    # Paper (strong): three or more fingers clearly open
    if no_curl_count >= 3:
        if curls["ring"] == "NoCurl" and curls["pinky"] == "NoCurl":
            # Ring and pinky open -> almost certainly Paper
            return "Paper", 0.85, dbg
        if curls["index"] == "NoCurl" and curls["middle"] == "NoCurl":
            # Index and middle open — check if ring/pinky are folded for Scissors
            if curls["ring"] in ("FullCurl", "HalfCurl") or curls["pinky"] in ("FullCurl", "HalfCurl"):
                return "Scissors", 0.80, dbg
            else:
                # All four fingers pretty much open -> Paper
                return "Paper", 0.75, dbg
        # At least 3 open but not the clear patterns above -> probably Paper
        return "Paper", 0.70, dbg

    # Rock (weaker): three or more fingers curled (but not all four)
    if full_curl_count >= 3:
        return "Rock", 0.80, dbg

    # Scissors (strong): index and middle open, ring and pinky folded
    if (curls["index"] == "NoCurl" and curls["middle"] == "NoCurl"
            and curls["ring"]  in ("FullCurl", "HalfCurl")
            and curls["pinky"] in ("FullCurl", "HalfCurl")):
        return "Scissors", 0.90, dbg

    # Scissors (weaker): only index and middle open, everything else unclear
    if no_curl_count == 2 and curls["index"] == "NoCurl" and curls["middle"] == "NoCurl":
        return "Scissors", 0.65, dbg

    # Ambiguous Paper: two or more open, few fully curled
    if no_curl_count >= 2 and full_curl_count <= 1:
        return "Paper", 0.50, dbg

    # Ambiguous Rock: two or more curled, few open
    if full_curl_count >= 2 and no_curl_count <= 1:
        return "Rock", 0.55, dbg

    # Could not match any pattern
    return "Unknown", 0.0, dbg


# ---------------------------------------------------------------------------
# ML MODEL
# ---------------------------------------------------------------------------

def _load_model_once():
    """
    Load the trained ML model from disk on the first call, then cache it.

    We use a module-level flag (_model_checked) so we only attempt the file
    read once per session — even if the file is missing, we won't keep
    printing the warning every frame.

    Returns:
        (model, int_to_label) or (None, None) if no model is available.
    """
    global _cached_model, _cached_labels, _model_checked

    # Return the cached result if we have already attempted a load
    if _model_checked:
        return _cached_model, _cached_labels

    _model_checked = True  # mark that we have tried, regardless of outcome

    model_path = Path.home() / "Desktop" / "CapStone" / "front_on_gesture_model.pkl"

    if not model_path.exists():
        print("[FrontOn] No trained model found. Use Diagnostic mode to collect data.")
        return None, None

    try:
        import pickle
        with open(model_path, "rb") as f:
            data = pickle.load(f)

        _cached_model  = data["model"]
        _cached_labels = data["int_to_label"]

        # Log a friendly summary so it's obvious which model was picked up
        n       = data.get("n_samples", "?")
        acc     = data.get("accuracy")
        acc_str = f"{acc:.0%}" if acc else "unknown"
        print(f"[FrontOn] Loaded ML model ({n} samples, {acc_str} accuracy)")

        return _cached_model, _cached_labels

    except Exception as exc:
        print(f"[FrontOn] Failed to load model: {exc}")
        return None, None


def _normalise_landmarks(hand_landmarks):
    """
    Translate all 21 landmarks to be relative to the wrist, then scale them
    so that the average wrist-to-MCP distance equals 1.0.

    This makes the feature vector invariant to where the hand is in the frame
    and how far it is from the camera.  The resulting 42 floats (x, y per
    landmark) are what the ML model was trained on.
    """
    lm = hand_landmarks.landmark

    # Use landmark 0 (wrist) as the origin for translation
    wrist_x = lm[0].x
    wrist_y = lm[0].y

    # Helper for 2D distance between two landmark objects
    def _d(a, b):
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    # Palm scale = average distance from wrist to the three base knuckles
    # (index=5, middle=9, pinky=17).  This is more stable than a single bone.
    d1 = _d(lm[0], lm[5])
    d2 = _d(lm[0], lm[9])
    d3 = _d(lm[0], lm[17])
    palm_scale = max((d1 + d2 + d3) / 3.0, 1e-6)  # clamp to avoid div-by-zero

    # Build the 42-value feature row: [(x0, y0), (x1, y1), ... (x20, y20)]
    row = []
    for i in range(21):
        row.append((lm[i].x - wrist_x) / palm_scale)
        row.append((lm[i].y - wrist_y) / palm_scale)

    return row


def _ml_classify(hand_landmarks):
    """
    Run the trained ML model on the current hand landmarks.

    Returns:
        (gesture, confidence, prob_str)
        gesture:    predicted class name, or None if the model is unavailable
        confidence: probability assigned to the top class (0.0 - 1.0)
        prob_str:   compact string like "R:82% P:11% S:7%" for debugging
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

        prediction    = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]

        gesture    = int_to_label.get(prediction, "Unknown")
        class_idx  = list(model.classes_).index(prediction)
        confidence = float(probabilities[class_idx])

        # Sort probabilities highest-first for the debug string
        sorted_probs = sorted(
            [(int_to_label[i], p) for i, p in enumerate(probabilities)],
            key=lambda x: x[1],
            reverse=True,
        )
        prob_str = " ".join(f"{n[0]}:{p:.0%}" for n, p in sorted_probs)

        return gesture, confidence, prob_str

    except Exception:
        return None, 0.0, "ml_error"


# ---------------------------------------------------------------------------
# HYBRID CLASSIFIER
# ---------------------------------------------------------------------------

# If the ML model's top probability is above this threshold, we trust it alone
# (unless curl strongly disagrees — see the logic below).
ML_CONFIDENCE_THRESHOLD = 0.70


def classify_front_on(hand_landmarks):
    """
    Classify an RPS gesture by combining ML and curl-analysis votes.

    This is the main entry point used by the game.  It always returns a dict
    with at least: gesture, command, reason.

    Decision priority:
        1. If ML is confident (>=70%): use ML, unless curl is very confident
           on a different answer (>=85% curl vs <85% ML) — then trust curl.
        2. If ML is uncertain (<70%): use curl if curl has >=60% confidence,
           otherwise fall back to ML's best guess.
        3. If there is no model at all: use curl if >=50%, else Unknown.
    """
    # Run both classifiers every frame regardless of which we end up using,
    # so the debug output always shows both signals
    curl_gesture, curl_conf, curl_dbg = _curl_classify(hand_landmarks)
    ml_gesture,   ml_conf,   ml_dbg   = _ml_classify(hand_landmarks)

    # Map gesture names to the command strings expected by the game engine
    cmd_map = {
        "Rock":     "CMD_ROCK",
        "Paper":    "CMD_PAPER",
        "Scissors": "CMD_SCISSORS",
    }

    # --- Decision tree ------------------------------------------------------
    if ml_gesture is not None and ml_conf >= ML_CONFIDENCE_THRESHOLD:
        if ml_gesture == curl_gesture:
            # Both classifiers agree — highest possible confidence
            gesture = ml_gesture
            reason  = f"agree ml={ml_dbg} curl={curl_dbg}"
        else:
            # ML is confident but curl disagrees.  Prefer ML unless curl is
            # very sure about a different answer AND ML isn't especially high.
            if curl_conf >= 0.85 and ml_conf < 0.85:
                gesture = curl_gesture
                reason  = f"curl_override ml={ml_dbg} curl={curl_dbg}"
            else:
                gesture = ml_gesture
                reason  = f"ml_wins ml={ml_dbg} curl={curl_dbg}"

    elif ml_gesture is not None:
        # ML is uncertain — let curl have a say
        if curl_conf >= 0.60:
            gesture = curl_gesture
            reason  = f"curl_leads ml={ml_dbg} curl={curl_dbg}"
        else:
            # Both are uncertain — go with ML's best guess anyway
            gesture = ml_gesture
            reason  = f"ml_weak ml={ml_dbg} curl={curl_dbg}"

    else:
        # No ML model loaded — curl is our only option
        if curl_conf >= 0.50:
            gesture = curl_gesture
            reason  = f"curl_only {curl_dbg}"
        else:
            gesture  = "Unknown"
            reason   = f"uncertain {curl_dbg}"

    cmd = cmd_map.get(gesture, "CMD_UNKNOWN")

    return {
        "gesture": gesture,
        "command": cmd,
        "reason":  reason,
    }


def reload_model():
    """
    Force the classifier to discard its cached model and reload from disk.
    Call this after front_on_trainer.py finishes retraining so the game
    immediately picks up the new model without needing a restart.
    """
    global _cached_model, _cached_labels, _model_checked
    _cached_model   = None
    _cached_labels  = None
    _model_checked  = False  # reset the "already tried" flag so load runs again
    return _load_model_once()

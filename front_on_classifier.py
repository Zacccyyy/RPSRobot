"""
front_on_classifier.py
======================
Real-time RPS gesture classifier for when the player faces the camera
front-on (palm toward the lens).

HOW IT WORKS — two classifiers that vote:

    Layer 1: ML MODEL (trained on collected hand data)
        - Analyses rotation-invariant angle + curl features
        - Highly accurate for static poses it was trained on
        - Can lag during fast transitions (pump -> shoot)

    Layer 2: CURL ANALYSIS (rule-based, no training needed)
        - Measures PIP and DIP joint angles per finger in real time
        - Classifies each finger as: NoCurl / HalfCurl / FullCurl
        - Very fast and responsive to finger movement
        - Rotation-invariant (uses angles, not raw positions)

    Decision logic:
        - If ML confidence >= 70%: trust ML
            - Unless curl is very confident (>=85%) on a different answer
              AND ML is below 85% — then trust curl (override)
        - If ML confidence < 70%: let curl take over if curl >= 60% confident
            - Otherwise fall back to ML's best guess
        - If no model is loaded: use curl alone if >= 50% confident

    The "reason" field in the return dict shows which path fired,
    making it easy to debug misclassifications.

Model file: ~/Desktop/CapStone/front_on_gesture_model.pkl
"""

import math
import pickle
from pathlib import Path

from front_on_features import extract_features as _extract_features


# ---------------------------------------------------------------------------
# Model cache — loaded once on first use, then kept in memory.
# This avoids re-reading the pickle file on every frame.
# ---------------------------------------------------------------------------
_cached_model  = None
_cached_labels = None
_model_checked = False   # True after the first load attempt (even if it failed)


# ---------------------------------------------------------------------------
# CURL ANALYSIS — rule-based finger-curl classifier
# ---------------------------------------------------------------------------

# Landmark IDs for each finger: (name, mcp, pip, dip, tip).
# These are MediaPipe's fixed joint indices.
FINGER_JOINTS = [
    ("index",  5,  6,  7,  8),
    ("middle", 9,  10, 11, 12),
    ("ring",   13, 14, 15, 16),
    ("pinky",  17, 18, 19, 20),
]

# Curl state thresholds (degrees at the PIP joint).
# A fully extended finger reads ~170-180 degrees; fully curled reads ~40-80 degrees.
CURL_NO   = 150   # above this -> NoCurl (finger is straight)
CURL_HALF = 110   # between CURL_HALF and CURL_NO -> HalfCurl; below -> FullCurl


def _angle_3pt(a, b, c):
    """
    Compute the angle at landmark b (the vertex) formed by rays b->a and b->c.
    Uses 2D (x, y) coordinates from landmark objects. Returns degrees.

    Returns 180.0 (straight) if either vector is degenerate — this is a safe
    default that won't accidentally classify a straight finger as curled.
    """
    # 2D vectors from the vertex b outward to a and c
    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)

    dot    = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)

    # Avoid division by zero when two landmarks happen to coincide
    if mag_ba < 1e-8 or mag_bc < 1e-8:
        return 180.0

    # Clamp before acos to handle floating-point overshoot (e.g. cos = 1.0000001)
    cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


def _finger_curl(lm, mcp_id, pip_id, dip_id, tip_id):
    """
    Determine how curled a single finger is by measuring its two bending joints.

    We measure the angle at both the PIP (middle knuckle) and DIP (top knuckle),
    then use whichever is more bent (smaller angle). This catches fingers that
    curl primarily at one joint rather than the other.

    Returns:
        (curl_label, pip_angle, dip_angle)
        curl_label is one of "NoCurl", "HalfCurl", or "FullCurl".
    """
    pip_angle = _angle_3pt(lm[mcp_id], lm[pip_id], lm[dip_id])
    dip_angle = _angle_3pt(lm[pip_id], lm[dip_id], lm[tip_id])

    # One very bent joint is enough to call the finger curled, so use the min
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
    borderline poses (e.g. a fist with one finger slightly raised).

    Returns:
        (gesture, confidence, debug_str)
        gesture:    "Rock", "Paper", "Scissors", or "Unknown"
        confidence: 0.0 to 1.0 — how certain we are
        debug_str:  compact per-finger state, e.g. "i:O170 m:C45 r:C40 p:C38"
    """
    lm = hand_landmarks.landmark

    # Measure the curl state of each finger
    curls  = {}
    angles = {}
    for name, mcp, pip, dip, tip in FINGER_JOINTS:
        curl, pip_a, dip_a = _finger_curl(lm, mcp, pip, dip, tip)
        curls[name]  = curl
        angles[name] = (pip_a, dip_a)

    # Count fingers in each state — useful for the classification rules below
    no_curl_count   = sum(1 for c in curls.values() if c == "NoCurl")
    full_curl_count = sum(1 for c in curls.values() if c == "FullCurl")

    # Build a compact debug string so misclassifications are easy to diagnose
    short_code = {"NoCurl": "O", "HalfCurl": "H", "FullCurl": "C"}
    dbg_parts  = []
    for name in ("index", "middle", "ring", "pinky"):
        pip_a, dip_a = angles[name]
        code = short_code[curls[name]]
        dbg_parts.append(f"{name[0]}:{code}{min(pip_a, dip_a):.0f}")
    dbg = " ".join(dbg_parts)

    # --- Classification rules, ordered from most to least specific -----------

    # Rock: all four fingers fully curled (closed fist)
    if full_curl_count == 4:
        return "Rock", 0.95, dbg

    # Paper (strong): three or more fingers clearly open
    if no_curl_count >= 3:
        if curls["ring"] == "NoCurl" and curls["pinky"] == "NoCurl":
            # Ring and pinky both open -> almost certainly Paper (not Scissors)
            return "Paper", 0.85, dbg
        if curls["index"] == "NoCurl" and curls["middle"] == "NoCurl":
            # Index and middle open — check if ring/pinky are folded for Scissors
            if curls["ring"] in ("FullCurl", "HalfCurl") or curls["pinky"] in ("FullCurl", "HalfCurl"):
                return "Scissors", 0.80, dbg
            else:
                # All four fingers roughly open -> Paper
                return "Paper", 0.75, dbg
        # At least 3 open but not the clear patterns above -> probably Paper
        return "Paper", 0.70, dbg

    # Rock (weaker): three or more fingers curled but not all four
    if full_curl_count >= 3:
        return "Rock", 0.80, dbg

    # Scissors (strong): exactly index and middle open, ring and pinky folded
    if (curls["index"]  == "NoCurl"
            and curls["middle"] == "NoCurl"
            and curls["ring"]   in ("FullCurl", "HalfCurl")
            and curls["pinky"]  in ("FullCurl", "HalfCurl")):
        return "Scissors", 0.90, dbg

    # Scissors (weaker): only two fingers open and they are index + middle
    if no_curl_count == 2 and curls["index"] == "NoCurl" and curls["middle"] == "NoCurl":
        return "Scissors", 0.65, dbg

    # Ambiguous Paper: two or more fingers open, very few fully curled
    if no_curl_count >= 2 and full_curl_count <= 1:
        return "Paper", 0.50, dbg

    # Ambiguous Rock: two or more fingers curled, very few open
    if full_curl_count >= 2 and no_curl_count <= 1:
        return "Rock", 0.55, dbg

    # Could not match any known pattern
    return "Unknown", 0.0, dbg


# ---------------------------------------------------------------------------
# ML MODEL — loads and runs the trained sklearn/MLP classifier
# ---------------------------------------------------------------------------

def _load_model_once():
    """
    Load the trained ML model from disk on the first call, then cache it.

    We use a module-level flag (_model_checked) so we only attempt the file
    read once per session. If the file is missing, we print a message once
    and return (None, None) for every subsequent call without spamming.

    Returns:
        (model, int_to_label) or (None, None) if no model is available.
    """
    global _cached_model, _cached_labels, _model_checked

    # If we have already tried loading (success or failure), return the cached result
    if _model_checked:
        return _cached_model, _cached_labels

    _model_checked = True   # mark that we've tried, regardless of outcome

    model_path = Path.home() / "Desktop" / "CapStone" / "front_on_gesture_model.pkl"

    if not model_path.exists():
        print("[FrontOn] No trained model found. Use Diagnostic mode to collect data.")
        return None, None

    try:
        with open(model_path, "rb") as f:
            data = pickle.load(f)

        _cached_model  = data["model"]
        _cached_labels = data["int_to_label"]

        # Print a summary so it's obvious the model was picked up successfully
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
    Translate all 21 landmarks to be relative to the wrist, then scale so
    the average wrist-to-MCP distance equals 1.0.

    This makes the feature vector invariant to where the hand sits in the frame
    and how far it is from the camera. The result is 42 floats (x, y per landmark)
    in the format the ML model was trained on.
    """
    lm = hand_landmarks.landmark

    # Wrist is landmark 0 — use it as the translation origin
    wrist_x = lm[0].x
    wrist_y = lm[0].y

    # Helper: distance between two landmark objects
    def _d(a, b):
        return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

    # Palm scale = average distance from wrist to three base knuckles (index=5,
    # middle=9, pinky=17). Averaging three bones is more stable than using one.
    d1 = _d(lm[0], lm[5])
    d2 = _d(lm[0], lm[9])
    d3 = _d(lm[0], lm[17])
    palm_scale = max((d1 + d2 + d3) / 3.0, 1e-6)   # clamp to avoid div-by-zero

    # Build 42 values: normalised (x, y) for each of the 21 landmarks
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
        confidence: probability of the top class (0.0 - 1.0)
        prob_str:   compact string like "R:82% P:11% S:7%" for the debug log
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
# HYBRID CLASSIFIER — combines ML and curl votes
# ---------------------------------------------------------------------------

# Above this confidence threshold, we trust ML alone (unless curl strongly overrides)
ML_CONFIDENCE_THRESHOLD = 0.70


def classify_front_on(hand_landmarks):
    """
    Classify an RPS gesture by combining ML and curl-analysis votes.

    This is the main entry point used by the game. It always returns a dict
    with at least: gesture, command, reason.

    Decision priority (see module docstring for full explanation):
        1. ML confident (>=70%): use ML, unless curl is very confident (>=85%)
           on a different answer AND ML is below 85% — then curl overrides.
        2. ML uncertain (<70%): use curl if curl >= 60% confident,
           otherwise fall back to ML's best guess anyway.
        3. No model loaded: use curl if >= 50% confident, else return Unknown.
    """
    # Run both classifiers on every frame so the debug output always shows both
    curl_gesture, curl_conf, curl_dbg = _curl_classify(hand_landmarks)
    ml_gesture,   ml_conf,   ml_dbg   = _ml_classify(hand_landmarks)

    # Map gesture names to the command strings the game engine expects
    cmd_map = {
        "Rock":     "CMD_ROCK",
        "Paper":    "CMD_PAPER",
        "Scissors": "CMD_SCISSORS",
    }

    # --- Decision tree -------------------------------------------------------
    if ml_gesture is not None and ml_conf >= ML_CONFIDENCE_THRESHOLD:
        if ml_gesture == curl_gesture:
            # Both classifiers agree — highest confidence situation
            gesture = ml_gesture
            reason  = f"agree ml={ml_dbg} curl={curl_dbg}"
        else:
            # ML is confident but curl disagrees. Prefer ML unless curl is
            # very sure about something different and ML isn't at its peak.
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
            # Both are uncertain — use ML's best guess as the least-bad option
            gesture = ml_gesture
            reason  = f"ml_weak ml={ml_dbg} curl={curl_dbg}"

    else:
        # No ML model loaded — curl is our only option
        if curl_conf >= 0.50:
            gesture = curl_gesture
            reason  = f"curl_only {curl_dbg}"
        else:
            gesture = "Unknown"
            reason  = f"uncertain {curl_dbg}"

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

    # Reset all three globals so _load_model_once() runs a fresh file read
    _cached_model  = None
    _cached_labels = None
    _model_checked = False

    return _load_model_once()

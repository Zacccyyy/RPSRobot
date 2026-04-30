import cv2
import math
import mediapipe as mp

from finger_counter import count_hand_fingers
from gesture_mapper import classify_rps_gesture
from front_on_classifier import classify_front_on

# MediaPipe 0.10.x compatibility
# Some versions expose mp.solutions, others require direct import
try:
    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    # Newer MediaPipe versions - import solutions directly
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import drawing_utils as mp_drawing


# ── Kalman filter for wrist Y smoothing ───────────────────────────────────────
class KalmanWrist1D:
    """
    1-D Kalman filter for wrist Y position.
    State: [position, velocity]
    Replaces the EMA (α=0.35) used previously.

    Advantages over EMA:
      • Handles missing frames gracefully — predicts forward using velocity
      • Separates process noise (smooth motion) from measurement noise (jitter)
      • Recovers faster after occlusion because it knows the hand was moving

    Tuning:
      process_noise  — how much we trust the motion model.  Higher = follows
                       measured position more aggressively (acts like lower α).
      measurement_noise — how noisy the raw MediaPipe wrist landmark is.
                          Higher = smoother but more lag.
    """

    def __init__(self, process_noise: float = 2e-3, measurement_noise: float = 4e-3):
        self._x   = 0.5    # estimated position
        self._v   = 0.0    # estimated velocity (in normalised units per frame)
        self._p   = 1.0    # position variance
        self._pv  = 0.0    # cross-covariance (pos × vel)
        self._vv  = 1.0    # velocity variance
        self._qp  = process_noise
        self._r   = measurement_noise
        self._initialized = False

    def update(self, measurement):
        # type: (float | None) -> float  -- written for Python 3.9 compat
        """
        Pass the raw wrist Y (0–1) or None if hand not detected.
        Returns the smoothed estimate.
        """
        if not self._initialized:
            if measurement is None:
                return 0.5
            self._x          = measurement
            self._initialized = True
            return self._x

        # ── Predict ──────────────────────────────────────────────────────
        x_pred = self._x + self._v
        p_pred = self._p + self._pv * 2 + self._vv + self._qp
        pv_pred = self._pv + self._vv
        vv_pred = self._vv + self._qp * 0.1

        if measurement is None:
            # No observation — keep prediction, widen uncertainty
            self._x  = x_pred
            self._v  = self._v * 0.92    # friction: velocity decays
            self._p  = p_pred * 1.05
            self._pv = pv_pred
            self._vv = vv_pred * 1.05
            return float(self._x)

        # ── Update ───────────────────────────────────────────────────────
        innov = measurement - x_pred
        s     = p_pred + self._r
        kg_x  = p_pred  / s   # Kalman gain for position
        kg_v  = pv_pred / s   # Kalman gain for velocity

        self._x  = x_pred  + kg_x * innov
        self._v  = self._v + kg_v * innov
        self._p  = (1 - kg_x) * p_pred
        self._pv = (1 - kg_x) * pv_pred
        self._vv = vv_pred - kg_v * pv_pred

        return float(self._x)

    def reset(self):
        self._initialized = False
        self._x = 0.5; self._v = 0.0
        self._p = 1.0; self._pv = 0.0; self._vv = 1.0


def create_kalman_wrist_state() -> dict:
    """
    Creates a wrist state dict using a KalmanWrist1D filter.
    Drop-in replacement for the old {"wrist_y": None} EMA dict.
    The dict stores the filter object so it persists across frames.
    """
    return {"kalman": KalmanWrist1D(), "wrist_y": None}


def create_hands_detector():
    """
    Main game detector.
    max_num_hands=2 is required for two-player modes.
    model_complexity=0 is the fastest model.
    Slightly higher confidence thresholds (0.6) hold tracking locks
    better between frames and reduce jitter on fast pump movements.
    """
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )


def create_nav_detector():
    """
    Lightweight single-hand detector for gesture nav on menu screens.
    Lower confidence thresholds are fine here -- menu nav is forgiving.
    """
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )


def _palm_scale(landmarks):
    """Estimate hand size from wrist-to-MCP distances. Larger = closer to camera."""
    lm = landmarks.landmark
    def _d(a, b):
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
    return (_d(lm[0], lm[5]) + _d(lm[0], lm[9]) + _d(lm[0], lm[17])) / 3.0


def _select_closest_hand(results):
    """
    When multiple hands are detected, select the one closest to the
    camera (largest palm scale in the frame).

    Returns (hand_landmarks, hand_label, handedness_score) for the
    chosen hand, or (None, "Unknown", 0.0) if no hands found.
    """
    if not results.multi_hand_landmarks:
        return None, "Unknown", 0.0

    if len(results.multi_hand_landmarks) == 1:
        hand_landmarks = results.multi_hand_landmarks[0]
        hand_label = "Unknown"
        handedness_score = 0.0
        if results.multi_handedness:
            h = results.multi_handedness[0].classification[0]
            hand_label = h.label
            handedness_score = h.score
        return hand_landmarks, hand_label, handedness_score

    # Multiple hands: pick the largest (closest to camera).
    best_idx = 0
    best_scale = 0.0

    for i, landmarks in enumerate(results.multi_hand_landmarks):
        scale = _palm_scale(landmarks)
        if scale > best_scale:
            best_scale = scale
            best_idx = i

    hand_landmarks = results.multi_hand_landmarks[best_idx]
    hand_label = "Unknown"
    handedness_score = 0.0

    if results.multi_handedness and best_idx < len(results.multi_handedness):
        h = results.multi_handedness[best_idx].classification[0]
        hand_label = h.label
        handedness_score = h.score

    return hand_landmarks, hand_label, handedness_score


def process_hand_frame(
    frame,
    hands,
    target_hand="Auto",
    display_mode="Game",
    handedness_threshold=0.80,
    hand_orientation="Side",
    _ema_state=None,
    five_gesture_mode=False,   # True only for RPSLS
):
    """
    Processes one camera frame and returns:
    - the mirrored frame (with optional diagnostic landmarks drawn)
    - a dictionary of hand / gesture / status values for the game loop
    - the RGB frame (reusable for emotion tracking — avoids double conversion)

    _ema_state: {"wrist_y": float | None} — shared across calls for EMA smoothing.
    """
    frame = cv2.flip(frame, 1)

    # Lighting quality check — warn if hand region is too dark
    brightness = float(frame.mean())
    poor_lighting = brightness < 55  # empirically tuned threshold

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True
    # rgb is returned so callers can reuse it (e.g. for emotion tracking)
    # without doing a second BGR→RGB conversion on the same frame.

    num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

    hand_state = {
        "count_text": "Unknown",
        "raw_gesture": "Unknown",
        "status_text": "No hand detected",
        "reason_text": "no_hand",
        "ambiguous_count": 0,
        "wrist_y": None,        # Kalman-smoothed (for display)
        "raw_wrist_y": None,    # unsmoothed landmark Y (for pump detection)
        "_landmarks": None,
        "hands_detected": num_hands,
        # Gesture nav fields
        "up_fingers":  [],
        "index_tip_x": None,
        "index_tip_y": None,
        # Quality feedback
        "poor_lighting": poor_lighting,
        "hand_too_far":  False,
        "palm_scale":    0.0,
    }

    hand_landmarks, hand_label, handedness_score = _select_closest_hand(results)

    if hand_landmarks is not None:
        tip_ids_for_debug = []
        closest_tag = f" [{num_hands} hands]" if num_hands > 1 else ""

        # Palm scale — proxy for hand distance from camera.
        # Smaller = farther away; values < 0.10 typically mean the hand is
        # too far for reliable landmark accuracy.
        palm_sc = _palm_scale(hand_landmarks)
        hand_state["palm_scale"]   = palm_sc
        hand_state["hand_too_far"] = palm_sc < 0.09

        # Always import these — used by both Front-On and Side-View paths
        from finger_counter import count_hand_fingers
        from gesture_mapper import classify_rps_gesture

        if hand_orientation == "Front":
            # ----- FRONT-ON PATH -----
            count_result = count_hand_fingers(
                hand_landmarks=hand_landmarks,
                hand_label=hand_label,
                target_hand=target_hand,
                handedness_score=handedness_score,
                handedness_threshold=handedness_threshold,
            )
            geo_result = classify_rps_gesture(
                count_result, hand_landmarks=hand_landmarks,
                five_gesture_mode=five_gesture_mode)

            if geo_result["gesture"] in ("Spock", "Lizard"):
                gesture = geo_result["gesture"]
                reason  = f"geo_priority: {geo_result['reason']}"
            else:
                front_result = classify_front_on(hand_landmarks)
                gesture = front_result["gesture"]
                reason  = front_result["reason"]

            hand_state["raw_gesture"]     = gesture
            hand_state["reason_text"]     = reason
            hand_state["count_text"]      = gesture[:1] if gesture != "Unknown" else "?"
            hand_state["ambiguous_count"] = 0
            hand_state["status_text"]     = (
                f"Seen: {hand_label} ({handedness_score:.2f}) | front{closest_tag}"
            )

        else:
            # ----- SIDE-VIEW PATH -----
            count_result = count_hand_fingers(
                hand_landmarks=hand_landmarks,
                hand_label=hand_label,
                target_hand=target_hand,
                handedness_score=handedness_score,
                handedness_threshold=handedness_threshold
            )

            gesture_result = classify_rps_gesture(
                count_result, hand_landmarks=hand_landmarks,
                five_gesture_mode=five_gesture_mode)

            hand_state["count_text"]      = count_result["count_text"]
            hand_state["raw_gesture"]     = gesture_result["gesture"]
            hand_state["ambiguous_count"] = count_result["ambiguous"]

            reason_text = count_result["reason"]
            if reason_text == "ok":
                reason_text = gesture_result["reason"]

            hand_state["reason_text"] = reason_text
            hand_state["status_text"] = (
                f"Seen: {hand_label} ({handedness_score:.2f}){closest_tag}"
            )

            tip_ids_for_debug = count_result["tip_ids_up"]
            hand_state["up_fingers"] = count_result.get("up_fingers", [])

        # Kalman filter smoothing on wrist_y.
        # raw_wrist_y is always the direct MediaPipe landmark — used for pump detection.
        # wrist_y is Kalman-smoothed — used for display only.
        raw_wrist_y = hand_landmarks.landmark[0].y
        hand_state["raw_wrist_y"] = raw_wrist_y
        if _ema_state is not None:
            kf = _ema_state.get("kalman")
            if kf is not None:
                smoothed = kf.update(raw_wrist_y)
                _ema_state["wrist_y"] = smoothed
                hand_state["wrist_y"] = smoothed
            else:
                # Legacy EMA path (old {"wrist_y": None} dicts)
                prev = _ema_state.get("wrist_y")
                smoothed = raw_wrist_y if prev is None else 0.35 * raw_wrist_y + 0.65 * prev
                _ema_state["wrist_y"] = smoothed
                hand_state["wrist_y"] = smoothed
        else:
            hand_state["wrist_y"] = raw_wrist_y

        hand_state["_landmarks"] = hand_landmarks

        # Always populate gesture-nav fields regardless of orientation path.
        lm = hand_landmarks.landmark
        hand_state["index_tip_x"] = lm[8].x
        hand_state["index_tip_y"] = lm[8].y

        if display_mode == "Diagnostic":
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            for tip_id in tip_ids_for_debug:
                lm = hand_landmarks.landmark[tip_id]
                h, w, _ = frame.shape
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)

    return frame, hand_state, rgb

def process_two_hands_frame(
    frame,
    hands,
    hand_orientation="Side",
    handedness_threshold=0.80,
    ema_states=None,
    five_gesture_mode=False,   # True only for RPSLS
):
    """
    Process a frame expecting up to 2 hands simultaneously.

    Returns:
        frame  — mirrored, annotated
        p1     — hand_state dict for the LEFT-most detected hand (Player 1)
        p2     — hand_state dict for the RIGHT-most detected hand (Player 2)
        rgb    — RGB frame for emotion tracking reuse

    Hands are sorted by wrist X-position: lowest X (left of frame) → P1,
    highest X (right of frame) → P2.  Missing hands get a default state
    with raw_gesture="Unknown".
    """
    frame = cv2.flip(frame, 1)

    brightness = float(frame.mean())
    poor_lighting = brightness < 55

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True

    def _empty_state():
        return {
            "count_text":     "?",
            "raw_gesture":    "Unknown",
            "status_text":    "No hand",
            "reason_text":    "no_hand",
            "ambiguous_count": 0,
            "wrist_y":        None,
            "_landmarks":     None,
            "hands_detected": 0,
            "up_fingers":     [],
            "index_tip_x":    None,
            "index_tip_y":    None,
            "poor_lighting":  poor_lighting,
            "hand_too_far":   False,
            "palm_scale":     0.0,
        }

    def _extract_state(lm_obj, label, score, ema_state):
        """Build a hand_state dict from a MediaPipe landmark object."""
        state = _empty_state()
        state["hands_detected"] = len(results.multi_hand_landmarks) \
            if results.multi_hand_landmarks else 0

        palm_sc = _palm_scale(lm_obj)
        state["palm_scale"]   = palm_sc
        state["hand_too_far"] = palm_sc < 0.09

        # Imports hoisted — both branches need these
        from front_on_classifier import classify_front_on
        from gesture_mapper import classify_rps_gesture
        from finger_counter import count_hand_fingers

        if hand_orientation == "Front":
            # Geometry first — Spock/Lizard skip ML entirely (only in five_gesture_mode)
            cr_geo = count_hand_fingers(
                hand_landmarks=lm_obj,
                hand_label=label,
                target_hand="Auto",
                handedness_score=score,
                handedness_threshold=handedness_threshold,
            )
            geo = classify_rps_gesture(cr_geo, hand_landmarks=lm_obj,
                                       five_gesture_mode=five_gesture_mode)
            if geo["gesture"] in ("Spock", "Lizard"):
                gesture = geo["gesture"]
                reason  = f"geo_priority: {geo['reason']}"
            else:
                fr = classify_front_on(lm_obj)
                gesture = fr["gesture"]
                reason  = fr["reason"]
            state["raw_gesture"]     = gesture
            state["reason_text"]     = reason
            state["count_text"]      = gesture[:1] if gesture != "Unknown" else "?"
            state["ambiguous_count"] = 0
            state["status_text"]     = f"{label} ({score:.2f}) | front"
        else:
            cr = count_hand_fingers(
                hand_landmarks=lm_obj,
                hand_label=label,
                target_hand="Auto",
                handedness_score=score,
                handedness_threshold=handedness_threshold,
            )
            gr = classify_rps_gesture(cr, hand_landmarks=lm_obj,
                                      five_gesture_mode=five_gesture_mode)
            state["count_text"]      = cr["count_text"]
            state["raw_gesture"]     = gr["gesture"]
            state["ambiguous_count"] = cr["ambiguous"]
            reason = cr["reason"]
            if reason == "ok":
                reason = gr["reason"]
            state["reason_text"]  = reason
            state["status_text"]  = f"{label} ({score:.2f})"
            state["up_fingers"]   = cr.get("up_fingers", [])

        # EMA wrist smoothing — keep smoothed for display, raw for pump detection
        raw_y = lm_obj.landmark[0].y
        state["raw_wrist_y"] = raw_y
        if ema_state is not None:
            prev = ema_state.get("wrist_y")
            smoothed = raw_y if prev is None else 0.35 * raw_y + 0.65 * prev
            ema_state["wrist_y"] = smoothed
            state["wrist_y"] = smoothed
        else:
            state["wrist_y"] = raw_y

        state["_landmarks"]   = lm_obj
        lm = lm_obj.landmark
        state["index_tip_x"] = lm[8].x
        state["index_tip_y"] = lm[8].y
        return state

    if ema_states is None:
        ema_states = [{}, {}]

    # ── Collect all detected hands ──────────────────────────────────────
    detected = []
    if results.multi_hand_landmarks:
        for idx, lm_obj in enumerate(results.multi_hand_landmarks):
            label = "Unknown"
            score = 0.0
            if results.multi_handedness and idx < len(results.multi_handedness):
                h = results.multi_handedness[idx].classification[0]
                label = h.label
                score = h.score
            # Use wrist X for sorting (after mirror-flip, MediaPipe "Right" is on left of screen)
            wrist_x = lm_obj.landmark[0].x
            detected.append((wrist_x, lm_obj, label, score))

    # Sort: smallest X = leftmost = Player 1 column
    detected.sort(key=lambda t: t[0])

    p1 = _extract_state(*detected[0][1:], ema_states[0]) if len(detected) >= 1 else _empty_state()
    p2 = _extract_state(*detected[1][1:], ema_states[1]) if len(detected) >= 2 else _empty_state()

    return frame, p1, p2, rgb

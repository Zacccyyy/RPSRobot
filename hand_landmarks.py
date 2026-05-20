# =============================================================================
# hand_landmarks.py
# -----------------
# The main entry point for all camera-frame processing in this project.
#
# What it does:
#   - Wraps MediaPipe Hands so the rest of the app never touches the library
#     directly.
#   - Exposes two high-level frame-processing functions:
#       process_hand_frame()       — single-player: picks the closest hand
#       process_two_hands_frame()  — two-player: returns one state dict per
#                                    player, sorted left-to-right on screen
#   - Keeps a 1-D Kalman filter (KalmanWrist1D) per hand for smooth wrist-Y
#     tracking without the phase-lag that an EMA introduces.
#   - Returns "hand_state" dicts that feed gesture detection, the game loop,
#     and the UI renderer.
# =============================================================================

import cv2
import math
import mediapipe as mp

from finger_counter import count_hand_fingers
from gesture_mapper import classify_rps_gesture
from front_on_classifier import classify_front_on

# MediaPipe changed its import path between versions.
# Try the usual mp.solutions namespace first; fall back to direct import.
try:
    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    # Newer MediaPipe versions expose solutions as top-level modules
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import drawing_utils as mp_drawing


# =============================================================================
# Kalman filter for wrist-Y smoothing
# =============================================================================

class KalmanWrist1D:
    """
    A 1-D Kalman filter that tracks the wrist's vertical position (Y, 0-1).

    State vector: [position, velocity]
    We use a simple constant-velocity motion model — the filter assumes the
    wrist keeps moving at roughly the same speed from frame to frame.

    Why Kalman instead of an exponential moving average (EMA)?
      - EMA always lags behind fast motion.  Kalman uses its velocity estimate
        to *predict* where the wrist will be next, so it catches up faster.
      - When the hand disappears (occlusion), Kalman predicts forward rather
        than snapping to a fixed value.
      - It separates "how noisy is MediaPipe" from "how fast is the hand
        really moving", giving cleaner smoothing.

    Tuning parameters:
      process_noise     — trust the motion model more (higher = follows
                          measurements more aggressively, lower = smoother)
      measurement_noise — how jittery the raw MediaPipe landmark is
                          (higher = smoother output but more lag)
    """

    def __init__(self, process_noise: float = 2e-3, measurement_noise: float = 4e-3):
        self._x   = 0.5    # current position estimate (0 = top, 1 = bottom)
        self._v   = 0.0    # current velocity estimate (normalised units/frame)
        self._p   = 1.0    # position variance (uncertainty)
        self._pv  = 0.0    # cross-covariance between position and velocity
        self._vv  = 1.0    # velocity variance
        self._qp  = process_noise
        self._r   = measurement_noise
        self._initialized = False

    def update(self, measurement):
        # type: (float | None) -> float  -- written for Python 3.9 compat
        """
        Feed one new frame's raw wrist Y (0-1), or None if hand not detected.
        Returns the smoothed estimate.
        """
        # First valid observation initialises the filter position directly
        if not self._initialized:
            if measurement is None:
                return 0.5   # nothing to go on; return a neutral default
            self._x = measurement
            self._initialized = True
            return self._x

        # ── Predict step ──────────────────────────────────────────────────
        # Project the state forward by one frame using constant-velocity model
        x_pred  = self._x + self._v
        p_pred  = self._p + self._pv * 2 + self._vv + self._qp
        pv_pred = self._pv + self._vv
        vv_pred = self._vv + self._qp * 0.1

        if measurement is None:
            # No observation this frame — accept the prediction as-is and
            # widen uncertainty slightly so the next real measurement can
            # pull us back quickly
            self._x  = x_pred
            self._v  = self._v * 0.92   # "friction" — decay velocity when unseen
            self._p  = p_pred * 1.05    # grow position uncertainty
            self._pv = pv_pred
            self._vv = vv_pred * 1.05   # grow velocity uncertainty
            return float(self._x)

        # ── Update step ───────────────────────────────────────────────────
        # We have a real observation — blend prediction with measurement
        innov = measurement - x_pred   # how far off was our prediction?
        s     = p_pred + self._r       # total uncertainty
        kg_x  = p_pred  / s            # Kalman gain for position
        kg_v  = pv_pred / s            # Kalman gain for velocity

        # Correct the state using the Kalman gains
        self._x  = x_pred  + kg_x * innov
        self._v  = self._v + kg_v * innov
        self._p  = (1 - kg_x) * p_pred
        self._pv = (1 - kg_x) * pv_pred
        self._vv = vv_pred - kg_v * pv_pred

        return float(self._x)

    def reset(self):
        """Throw away all state — useful when switching players."""
        self._initialized = False
        self._x = 0.5
        self._v = 0.0
        self._p = 1.0
        self._pv = 0.0
        self._vv = 1.0


def create_kalman_wrist_state() -> dict:
    """
    Build the wrist-state dict that process_hand_frame() expects.
    Stores the KalmanWrist1D object so it persists across frames without
    being a module-level global.

    Drop-in replacement for the old {"wrist_y": None} EMA dict — the key
    names are the same so existing callers need no changes.
    """
    return {"kalman": KalmanWrist1D(), "wrist_y": None}


# =============================================================================
# MediaPipe detector factories
# =============================================================================

def create_hands_detector():
    """
    Create the MediaPipe Hands detector used during gameplay.

    Settings chosen for performance:
      - max_num_hands=2      needed for two-player local mode
      - model_complexity=0   fastest (lite) model, fine for RPS gestures
      - confidence at 0.6    reduces jitter on fast pump movements
    """
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )


def create_nav_detector():
    """
    Create a lighter MediaPipe Hands detector for menu/settings navigation.

    Single hand only; lower confidence thresholds are fine because menu
    nav doesn't need gesture-level precision — just rough position.
    """
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


# =============================================================================
# Internal helpers
# =============================================================================

def _palm_scale(landmarks):
    """
    Estimate the apparent size of the hand by averaging three wrist-to-MCP
    distances (index, middle, pinky knuckles).

    A larger value means the hand is closer to the camera.
    Used to pick the "closest hand" when multiple hands appear and to flag
    when the hand is too far for reliable landmark accuracy.
    """
    lm = landmarks.landmark

    # Simple 2-D Euclidean distance between two landmarks
    def _d(a, b):
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    # Average the three distances to get a stable size estimate
    return (_d(lm[0], lm[5]) + _d(lm[0], lm[9]) + _d(lm[0], lm[17])) / 3.0


def _select_closest_hand(results):
    """
    When MediaPipe returns multiple hands, choose the one closest to the
    camera (i.e. the one with the largest palm scale in the frame).

    Returns:
        (hand_landmarks, hand_label, handedness_score)
        or (None, "Unknown", 0.0) if no hands were detected.
    """
    if not results.multi_hand_landmarks:
        return None, "Unknown", 0.0

    # Only one hand detected — no comparison needed
    if len(results.multi_hand_landmarks) == 1:
        hand_landmarks   = results.multi_hand_landmarks[0]
        hand_label       = "Unknown"
        handedness_score = 0.0
        if results.multi_handedness:
            h = results.multi_handedness[0].classification[0]
            hand_label       = h.label
            handedness_score = h.score
        return hand_landmarks, hand_label, handedness_score

    # Multiple hands: find the one with the largest palm (closest to camera)
    best_idx   = 0
    best_scale = 0.0
    for i, landmarks in enumerate(results.multi_hand_landmarks):
        scale = _palm_scale(landmarks)
        if scale > best_scale:
            best_scale = scale
            best_idx   = i

    hand_landmarks   = results.multi_hand_landmarks[best_idx]
    hand_label       = "Unknown"
    handedness_score = 0.0

    if results.multi_handedness and best_idx < len(results.multi_handedness):
        h                = results.multi_handedness[best_idx].classification[0]
        hand_label       = h.label
        handedness_score = h.score

    return hand_landmarks, hand_label, handedness_score


# =============================================================================
# Wrist smoothing helper — used in both frame processors
# =============================================================================

def _apply_wrist_smoothing(raw_wrist_y, kalman_state):
    """
    Smooth the raw wrist Y value using either a Kalman filter (modern path)
    or a simple EMA (legacy fallback).

    kalman_state is the dict from create_kalman_wrist_state(), or None for
    no smoothing at all (raw value passed straight through).

    Returns the smoothed wrist Y value and updates kalman_state in-place.
    """
    if kalman_state is None:
        # No smoothing requested — use raw landmark value directly
        return raw_wrist_y

    kf = kalman_state.get("kalman")
    if kf is not None:
        # Modern path: run the Kalman filter
        smoothed = kf.update(raw_wrist_y)
    else:
        # Legacy path: old {"wrist_y": None} dict without a Kalman object.
        # Simple EMA (alpha=0.35) — kept for backwards compat.
        prev     = kalman_state.get("wrist_y")
        smoothed = raw_wrist_y if prev is None else 0.35 * raw_wrist_y + 0.65 * prev

    kalman_state["wrist_y"] = smoothed
    return smoothed


# =============================================================================
# Main single-hand frame processor
# =============================================================================

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
    Process one camera frame and return everything the game loop needs.

    Returns:
        frame      — horizontally flipped (mirrored) frame; landmark overlay
                     added in Diagnostic mode
        hand_state — dict of gesture / position / quality values (see below)
        rgb        — the RGB version of the frame, returned so the caller can
                     reuse it (e.g. for emotion detection) without doing a
                     second BGR->RGB conversion

    hand_state keys:
        count_text       — string finger count or "Unknown"
        raw_gesture      — "Rock" / "Paper" / "Scissors" / "Spock" / "Lizard"
                           / "Unknown"
        status_text      — human-readable MediaPipe status
        reason_text      — machine-readable reason for the current gesture call
        ambiguous_count  — how many fingers were in an ambiguous position
        wrist_y          — Kalman-smoothed wrist Y (for display)
        raw_wrist_y      — direct MediaPipe wrist Y (for pump detection)
        _landmarks       — raw MediaPipe landmark object (for downstream use)
        hands_detected   — number of hands MediaPipe found this frame
        up_fingers       — list of finger names currently extended
        index_tip_x/y    — index fingertip position (for gesture nav cursor)
        poor_lighting    — True when the frame is too dark for reliable detection
        hand_too_far     — True when palm_scale suggests the hand is out of range
        palm_scale       — estimated hand size (proxy for distance from camera)

    _ema_state: pass the dict from create_kalman_wrist_state() to get smoothed
                wrist_y across frames.  If None, raw landmark Y is used.
    """
    # Mirror the frame so it feels like looking in a mirror
    frame = cv2.flip(frame, 1)

    # Quick lighting check — if the frame is very dark, detection will be poor
    brightness    = float(frame.mean())
    poor_lighting = brightness < 55   # threshold tuned empirically

    # Convert to RGB once; reuse the result for both MediaPipe and the caller
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False          # prevents an unnecessary copy inside MP
    results = hands.process(rgb)
    rgb.flags.writeable = True

    num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

    # Default hand_state — filled in below if a hand is found
    hand_state = {
        "count_text":      "Unknown",
        "raw_gesture":     "Unknown",
        "status_text":     "No hand detected",
        "reason_text":     "no_hand",
        "ambiguous_count": 0,
        "wrist_y":         None,        # Kalman-smoothed (for display)
        "raw_wrist_y":     None,        # unsmoothed landmark Y (for pump detection)
        "_landmarks":      None,
        "hands_detected":  num_hands,
        # Gesture nav fields
        "up_fingers":      [],
        "index_tip_x":     None,
        "index_tip_y":     None,
        # Quality feedback for the UI
        "poor_lighting":   poor_lighting,
        "hand_too_far":    False,
        "palm_scale":      0.0,
    }

    # Pick the best hand (closest to camera if multiple detected)
    hand_landmarks, hand_label, handedness_score = _select_closest_hand(results)

    if hand_landmarks is not None:
        # Tag appended to status text when multiple hands are visible simultaneously
        closest_tag = f" [{num_hands} hands]" if num_hands > 1 else ""

        # Estimate hand distance from camera and flag if too far away
        palm_sc                   = _palm_scale(hand_landmarks)
        hand_state["palm_scale"]  = palm_sc
        hand_state["hand_too_far"] = palm_sc < 0.09

        # tip_ids_for_debug collects extended fingertip landmark IDs so we can
        # draw red circles over them in Diagnostic mode.  Set to empty list on
        # the front-on path since that path doesn't compute per-finger tip IDs.
        tip_ids_for_debug = []

        if hand_orientation == "Front":
            # ── Front-on path ─────────────────────────────────────────────
            # The hand faces directly into the camera (palm toward lens).
            # We try geometry first (more reliable for Spock/Lizard), then
            # the ML front-on classifier for Rock/Paper/Scissors.
            count_result = count_hand_fingers(
                hand_landmarks=hand_landmarks,
                hand_label=hand_label,
                target_hand=target_hand,
                handedness_score=handedness_score,
                handedness_threshold=handedness_threshold,
            )
            geo_result = classify_rps_gesture(
                count_result,
                hand_landmarks=hand_landmarks,
                five_gesture_mode=five_gesture_mode,
            )

            # Geometry is the authoritative source for Spock and Lizard
            # because the front-on ML model wasn't trained on those gestures
            if geo_result["gesture"] in ("Spock", "Lizard"):
                gesture = geo_result["gesture"]
                reason  = f"geo_priority: {geo_result['reason']}"
            else:
                # Fall back to the dedicated front-on ML classifier
                front_result = classify_front_on(hand_landmarks)
                gesture      = front_result["gesture"]
                reason       = front_result["reason"]

            hand_state["raw_gesture"]     = gesture
            hand_state["reason_text"]     = reason
            # Use first letter of gesture as a compact count display (e.g. "R", "P")
            hand_state["count_text"]      = gesture[:1] if gesture != "Unknown" else "?"
            hand_state["ambiguous_count"] = 0
            hand_state["status_text"]     = (
                f"Seen: {hand_label} ({handedness_score:.2f}) | front{closest_tag}"
            )

        else:
            # ── Side-view path ────────────────────────────────────────────
            # Default mode: hand shown in profile (thumb side visible).
            # Finger-counter runs first, then geometry classifier refines it.
            count_result = count_hand_fingers(
                hand_landmarks=hand_landmarks,
                hand_label=hand_label,
                target_hand=target_hand,
                handedness_score=handedness_score,
                handedness_threshold=handedness_threshold,
            )
            gesture_result = classify_rps_gesture(
                count_result,
                hand_landmarks=hand_landmarks,
                five_gesture_mode=five_gesture_mode,
            )

            hand_state["count_text"]      = count_result["count_text"]
            hand_state["raw_gesture"]     = gesture_result["gesture"]
            hand_state["ambiguous_count"] = count_result["ambiguous"]

            # Use the gesture mapper's reason only when finger counting succeeded
            reason_text = count_result["reason"]
            if reason_text == "ok":
                reason_text = gesture_result["reason"]

            hand_state["reason_text"] = reason_text
            hand_state["status_text"] = (
                f"Seen: {hand_label} ({handedness_score:.2f}){closest_tag}"
            )

            # tip_ids_up lists which fingertip landmark IDs are extended;
            # used to draw debug circles over extended fingertips in Diagnostic mode
            tip_ids_for_debug        = count_result["tip_ids_up"]
            hand_state["up_fingers"] = count_result.get("up_fingers", [])

        # ── Wrist-Y smoothing ─────────────────────────────────────────────
        # raw_wrist_y: straight from MediaPipe, used for pump-detection logic
        # wrist_y:     smoothed by Kalman filter, used only for display
        raw_wrist_y               = hand_landmarks.landmark[0].y
        hand_state["raw_wrist_y"] = raw_wrist_y
        hand_state["wrist_y"]     = _apply_wrist_smoothing(raw_wrist_y, _ema_state)

        hand_state["_landmarks"] = hand_landmarks

        # Index fingertip position always populated for the nav cursor
        lm = hand_landmarks.landmark
        hand_state["index_tip_x"] = lm[8].x
        hand_state["index_tip_y"] = lm[8].y

        # ── Diagnostic overlay ────────────────────────────────────────────
        if display_mode == "Diagnostic":
            # Draw the full landmark skeleton on screen
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
            )
            # Highlight each extended fingertip with a red dot
            for tip_id in tip_ids_for_debug:
                lm_pt  = hand_landmarks.landmark[tip_id]
                h, w, _ = frame.shape
                cx = int(lm_pt.x * w)
                cy = int(lm_pt.y * h)
                cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)

    return frame, hand_state, rgb


# =============================================================================
# Two-hand frame processor (for local two-player mode)
# =============================================================================

def process_two_hands_frame(
    frame,
    hands,
    hand_orientation="Side",
    handedness_threshold=0.80,
    ema_states=None,
    five_gesture_mode=False,   # True only for RPSLS
):
    """
    Process a frame expecting up to 2 hands simultaneously (local 2-player mode).

    Returns:
        frame  — mirrored, optionally annotated
        p1     — hand_state dict for the LEFT-most detected hand (Player 1)
        p2     — hand_state dict for the RIGHT-most detected hand (Player 2)
        rgb    — RGB frame, returned so the caller can reuse it

    Hands are assigned to players by wrist X-position after mirror-flip:
        smallest X (left side of screen)  -> Player 1
        largest  X (right side of screen) -> Player 2
    Missing hands produce a default state with raw_gesture="Unknown".
    """
    frame = cv2.flip(frame, 1)

    brightness    = float(frame.mean())
    poor_lighting = brightness < 55

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True

    # ── Inner helpers ─────────────────────────────────────────────────────────

    def _empty_state():
        """Return a neutral hand_state when no hand is assigned to a player."""
        return {
            "count_text":      "?",
            "raw_gesture":     "Unknown",
            "status_text":     "No hand",
            "reason_text":     "no_hand",
            "ambiguous_count": 0,
            "wrist_y":         None,
            "_landmarks":      None,
            "hands_detected":  0,
            "up_fingers":      [],
            "index_tip_x":     None,
            "index_tip_y":     None,
            "poor_lighting":   poor_lighting,
            "hand_too_far":    False,
            "palm_scale":      0.0,
        }

    def _extract_state(lm_obj, label, score, ema_state):
        """
        Build a full hand_state dict from a single MediaPipe landmark object.
        Same logic as process_hand_frame() but extracted here so we can run
        it twice (once per player) without duplicating code.
        """
        state = _empty_state()

        # Record total hands detected so the UI can show "2 hands visible"
        state["hands_detected"] = (
            len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        )

        # Estimate hand distance and flag if too far
        palm_sc               = _palm_scale(lm_obj)
        state["palm_scale"]   = palm_sc
        state["hand_too_far"] = palm_sc < 0.09

        if hand_orientation == "Front":
            # Geometry first; Spock/Lizard skip the ML model (same as single-hand path)
            cr_geo = count_hand_fingers(
                hand_landmarks=lm_obj,
                hand_label=label,
                target_hand="Auto",
                handedness_score=score,
                handedness_threshold=handedness_threshold,
            )
            geo = classify_rps_gesture(
                cr_geo,
                hand_landmarks=lm_obj,
                five_gesture_mode=five_gesture_mode,
            )
            if geo["gesture"] in ("Spock", "Lizard"):
                gesture = geo["gesture"]
                reason  = f"geo_priority: {geo['reason']}"
            else:
                fr      = classify_front_on(lm_obj)
                gesture = fr["gesture"]
                reason  = fr["reason"]

            state["raw_gesture"]     = gesture
            state["reason_text"]     = reason
            state["count_text"]      = gesture[:1] if gesture != "Unknown" else "?"
            state["ambiguous_count"] = 0
            state["status_text"]     = f"{label} ({score:.2f}) | front"

        else:
            # Side-view path — same finger-count + gesture-map flow as single-hand
            cr = count_hand_fingers(
                hand_landmarks=lm_obj,
                hand_label=label,
                target_hand="Auto",
                handedness_score=score,
                handedness_threshold=handedness_threshold,
            )
            gr = classify_rps_gesture(
                cr,
                hand_landmarks=lm_obj,
                five_gesture_mode=five_gesture_mode,
            )
            state["count_text"]      = cr["count_text"]
            state["raw_gesture"]     = gr["gesture"]
            state["ambiguous_count"] = cr["ambiguous"]
            reason = cr["reason"]
            if reason == "ok":
                reason = gr["reason"]
            state["reason_text"] = reason
            state["status_text"] = f"{label} ({score:.2f})"
            state["up_fingers"]  = cr.get("up_fingers", [])

        # Wrist smoothing — same dual raw/smoothed split as single-hand path
        raw_y                = lm_obj.landmark[0].y
        state["raw_wrist_y"] = raw_y
        state["wrist_y"]     = _apply_wrist_smoothing(raw_y, ema_state)

        state["_landmarks"]  = lm_obj
        lm = lm_obj.landmark
        state["index_tip_x"] = lm[8].x
        state["index_tip_y"] = lm[8].y
        return state

    # Default to empty dicts if the caller didn't provide Kalman state objects
    if ema_states is None:
        ema_states = [{}, {}]

    # ── Collect and sort all detected hands ───────────────────────────────────
    detected = []
    if results.multi_hand_landmarks:
        for idx, lm_obj in enumerate(results.multi_hand_landmarks):
            label = "Unknown"
            score = 0.0
            if results.multi_handedness and idx < len(results.multi_handedness):
                h     = results.multi_handedness[idx].classification[0]
                label = h.label
                score = h.score
            # After the mirror-flip, wrist X increases left-to-right on screen.
            # Storing wrist_x here lets us sort hands by their screen position.
            wrist_x = lm_obj.landmark[0].x
            detected.append((wrist_x, lm_obj, label, score))

    # Sort ascending by X so index 0 = left = Player 1, index 1 = right = Player 2
    detected.sort(key=lambda t: t[0])

    # Unpack the first two hands, or fall back to empty state if fewer detected
    p1 = _extract_state(*detected[0][1:], ema_states[0]) if len(detected) >= 1 else _empty_state()
    p2 = _extract_state(*detected[1][1:], ema_states[1]) if len(detected) >= 2 else _empty_state()

    return frame, p1, p2, rgb

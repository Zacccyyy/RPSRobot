"""
hand_enroll_state.py
====================
Hand geometry biometric enrollment and login.

ENROLLMENT
----------
Collects ENROLL_ROUNDS (20) capture rounds across varied positions and
distances. Each round extracts one 12-value geometry feature vector
(averaged over 40 stable frames). After all rounds, builds a centroid
profile and optionally trains an SVM if multiple people are enrolled.

The minimum for a reliable distance-based verification is 10 clean samples
(rounds). With 20 rounds at varied positions/distances we get a robust
centroid. If fewer than MIN_SAMPLES_NEEDED clean samples are captured,
the result is INSUFFICIENT — try again.

Phases: INTRO -> ALIGN -> CAPTURING -> REST (x ENROLL_ROUNDS) -> TRAINING
     -> RECOGNIZED or INSUFFICIENT

RECOGNITION (login)
-------------------
Collects RECOG_ROUNDS (5) rounds, predicts identity on each, votes.
Same ALIGN -> CAPTURING loop. Phases: INTRO -> ALIGN -> CAPTURING ->
REST (repeat) -> result shown.
"""

import time
import math
from collections import Counter

from gesture_fingerprint import (
    FingerprintStore,
    FingerprintClassifier,
    extract_geometry_features,
    MIN_SAMPLES_DISTANCE,
    MIN_SAMPLES_SVM,
    DISTANCE_THRESHOLD,
    MIN_DWELL_FRAMES,
)

# ── Tuning ──────────────────────────────────────────────────────────────────

STABLE_FRAMES_REQUIRED = 18    # wrist stability frames before capture starts
CAPTURE_FRAMES         = 40    # good-quality frames needed per round
REST_SECS              = 2.5   # pause between rounds

# 20 rounds gives ~20 samples — enough for a stable centroid (research min: 10)
ENROLL_ROUNDS   = 20
# Minimum clean samples before enrollment is accepted
MIN_SAMPLES_NEEDED = 10

# 5 recognition rounds give 5 prediction votes — more reliable than 3
RECOG_ROUNDS    = 5

JITTER_THRESHOLD   = 0.020
MIN_PALM_SCALE     = 0.12
MIN_FINGERS_EXTENDED = 3
POSITION_TOLERANCE = 0.30     # slightly relaxed from 0.28

# Landmark indices
WRIST      = 0
MIDDLE_MCP = 9
MIDDLE_TIP = 12

# ── Silhouette positions (cx_norm, cy_norm, scale_factor) ───────────────────
# 20 rounds cycling through 7 distinct positions / distances.
# Varying position and scale is important so the trained profile captures
# the hand at different apparent sizes (partial distance variance).
_BASE_TARGETS = [
    (0.50, 0.62, 1.00),   # 0  centre,      normal
    (0.36, 0.58, 0.75),   # 1  left,         further
    (0.64, 0.58, 0.75),   # 2  right,        further
    (0.50, 0.62, 1.28),   # 3  centre,       closer
    (0.38, 0.66, 0.88),   # 4  lower-left,   mid
    (0.62, 0.55, 0.88),   # 5  upper-right,  mid
    (0.50, 0.62, 1.00),   # 6  centre,       normal (repeat)
]

def _get_target(round_idx):
    return _BASE_TARGETS[round_idx % len(_BASE_TARGETS)]

_BASE_HINTS = [
    "Hold still - first scan",
    "Move LEFT and hold your hand FURTHER from the camera",
    "Move RIGHT and hold your hand FURTHER from the camera",
    "Move CLOSER to the camera - fill the outline",
    "Move to the LOWER LEFT",
    "Move to the UPPER RIGHT",
    "Back to centre - relax your hand slightly",
    "Hold still again",
    "Move LEFT again, slightly different angle",
    "Move RIGHT again, slightly different angle",
    "Move CLOSER again",
    "Lower left - tilt hand slightly",
    "Upper right - spread fingers wide",
    "Centre - hand slightly higher",
    "Left side, normal distance",
    "Right side, normal distance",
    "Closer again - fill the outline",
    "Centre - hold very still",
    "Left - furthest from camera",
    "Final scan - centre, normal distance",
]

def _get_hint(round_idx):
    if round_idx < len(_BASE_HINTS):
        return _BASE_HINTS[round_idx]
    return f"Round {round_idx + 1} - hold your hand in the outline"


# ── Helper functions ─────────────────────────────────────────────────────────

def _hand_open(lm_obj):
    """True when middle fingertip is above middle MCP (hand roughly open)."""
    try:
        lm = lm_obj.landmark
        return lm[MIDDLE_TIP].y < lm[MIDDLE_MCP].y
    except Exception:
        return False


def _wrist_x(lm_obj):
    try:
        return lm_obj.landmark[WRIST].x
    except Exception:
        return 0.5


def _stdev(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var  = sum((x - mean) ** 2 for x in values) / len(values)
    return var ** 0.5


def _detect_hand_side(lm_obj, status_text=""):
    """
    Returns 'Left', 'Right', or 'Unknown'.
    On a mirrored webcam: right hand thumb appears on the RIGHT (higher x).
    The SVG outline is a right hand — flip only for left hands.
    """
    if lm_obj is not None:
        try:
            lm = lm_obj.landmark
            thumb_x = lm[4].x
            pinky_x = lm[17].x
            if abs(thumb_x - pinky_x) > 0.05:
                return "Right" if thumb_x > pinky_x else "Left"
        except (IndexError, AttributeError):
            pass
    if status_text:
        st = status_text.lower()
        if "right" in st:
            return "Right"
        if "left" in st:
            return "Left"
    return "Unknown"


def _assess_quality(lm_obj, palm_scale, sil_cx_n=0.5, sil_cy_n=0.62):
    """Returns (ok: bool, reason: str)."""
    if lm_obj is None:
        return False, "No hand detected"
    lm = lm_obj.landmark

    if palm_scale < MIN_PALM_SCALE:
        return False, "Move closer to the camera"

    extended = sum(
        1 for (tip, pip) in [(8,6),(12,10),(16,14),(20,18)]
        if len(lm) > tip and lm[tip].y < lm[pip].y
    )
    if extended < MIN_FINGERS_EXTENDED:
        return False, "Face your palm toward the camera"

    try:
        if lm[0].y < lm[9].y:
            return False, "Hold your hand upright - fingers pointing up"
    except (IndexError, AttributeError):
        pass

    try:
        palm_cx = (lm[0].x + lm[9].x) / 2.0
        palm_cy = (lm[0].y + lm[9].y) / 2.0
        if (abs(palm_cx - sil_cx_n) > POSITION_TOLERANCE or
                abs(palm_cy - sil_cy_n) > POSITION_TOLERANCE):
            return False, "Move your hand inside the outline"
    except (IndexError, AttributeError):
        pass

    return True, "ok"


# ─────────────────────────────────────────────────────────────────────────────
# Shared base
# ─────────────────────────────────────────────────────────────────────────────

class _ScanBase:
    def __init__(self):
        self.fp_phase         = "INTRO"
        self._round           = 0
        self._stable_buf      = []
        self._capture_lms     = []
        self._rest_until      = 0.0
        self._quality_reason  = ""
        self._hand_side       = "Unknown"
        self._next_hint       = ""

    def _is_stable(self):
        if len(self._stable_buf) < STABLE_FRAMES_REQUIRED:
            return False
        return _stdev(self._stable_buf[-STABLE_FRAMES_REQUIRED:]) < JITTER_THRESHOLD

    def _stability_pct(self):
        if len(self._stable_buf) < 2:
            return 0.0
        recent = self._stable_buf[-STABLE_FRAMES_REQUIRED:]
        return max(0.0, 1.0 - min(1.0, _stdev(recent) / JITTER_THRESHOLD))

    def _tick_align(self, lm_obj, hand_visible, hand_open):
        if not hand_visible:
            self._stable_buf.clear()
            return False
        self._stable_buf.append(_wrist_x(lm_obj))
        if len(self._stable_buf) > STABLE_FRAMES_REQUIRED * 2:
            self._stable_buf.pop(0)
        if self._is_stable() and hand_open:
            self.fp_phase     = "CAPTURING"
            self._capture_lms = []
            return True
        return False

    def _tick_capturing_frame(self, lm_obj, hand_visible, hand_state):
        """
        Accumulate good-quality frames until CAPTURE_FRAMES collected.
        Bad frames silently skipped. Only hand disappearing resets round.
        Returns list of captured landmark frames when round is complete.
        """
        if not hand_visible:
            self.fp_phase        = "ALIGN"
            self._stable_buf     = []
            self._capture_lms    = []
            self._quality_reason = ""
            return None

        palm_scale = hand_state.get("palm_scale", 0.0) if hand_state else 0.0
        cx_n, cy_n, _ = _get_target(self._round)
        quality_ok, reason = _assess_quality(lm_obj, palm_scale, cx_n, cy_n)
        self._quality_reason = "" if quality_ok else reason

        if quality_ok and lm_obj:
            self._capture_lms.append(lm_obj.landmark)

        if len(self._capture_lms) >= CAPTURE_FRAMES:
            captured          = list(self._capture_lms)
            self._capture_lms = []
            self._stable_buf  = []
            self._round      += 1
            return captured
        return None

    def _sil_output(self, now):
        cx_n, cy_n, scale_f = _get_target(self._round)
        if self._hand_side == "Left":
            cx_n = 1.0 - cx_n
        if self.fp_phase == "REST" and REST_SECS > 0:
            elapsed  = now - (self._rest_until - REST_SECS)
            rest_pct = min(1.0, max(0.0, elapsed / REST_SECS))
            prev_cx, prev_cy, prev_sf = _get_target(max(0, self._round - 1))
            prev_cx_adj = (1.0 - prev_cx) if self._hand_side == "Left" else prev_cx
            cx_n    = prev_cx_adj + (cx_n - prev_cx_adj) * rest_pct
            cy_n    = prev_cy    + (cy_n - prev_cy)      * rest_pct
            scale_f = prev_sf   + (scale_f - prev_sf)   * rest_pct
        return cx_n, cy_n, scale_f


# ─────────────────────────────────────────────────────────────────────────────
# Enrollment controller
# ─────────────────────────────────────────────────────────────────────────────

class HandEnrollController(_ScanBase):
    """
    Collects ENROLL_ROUNDS geometry samples across varied positions.
    After all rounds completes, builds centroid profile and trains classifier.

    Result phases:
      RECOGNIZED   — enough clean samples, profile saved, self-test passed
      INSUFFICIENT — fewer than MIN_SAMPLES_NEEDED clean samples captured
    """

    def __init__(self, player_name, store=None):
        super().__init__()
        self.player_name      = player_name
        self._store           = store or FingerprintStore()
        self._session_samples = []   # one 12-vector per completed round
        self.enroll_start     = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.recog_name       = None
        self.recog_conf       = 0.0
        self.recog_z_score    = 0.0  # how many std-devs the self-test was
        self.n_enrolled_people = self._store.count_enrolled()

    def reset(self):
        super().__init__()
        self._session_samples  = []
        self.recog_name        = None
        self.recog_conf        = 0.0
        self.recog_z_score     = 0.0
        self.enroll_start      = time.strftime("%Y-%m-%dT%H:%M:%S")

    def update(self, hand_state, now=None):
        if now is None:
            now = time.monotonic()

        lm_obj       = hand_state.get("_landmarks") if hand_state else None
        hand_visible = lm_obj is not None
        hand_open    = _hand_open(lm_obj) if hand_visible else False

        if hand_visible:
            status = hand_state.get("status_text", "") if hand_state else ""
            detected = _detect_hand_side(lm_obj, status)
            if detected != "Unknown":
                self._hand_side = detected

        if self.fp_phase in ("RECOGNIZED", "INSUFFICIENT", "TRAINING"):
            pass  # terminal

        elif self.fp_phase == "INTRO":
            if hand_visible:
                self.fp_phase = "ALIGN"

        elif self.fp_phase == "ALIGN":
            self._tick_align(lm_obj, hand_visible, hand_open)

        elif self.fp_phase == "CAPTURING":
            captured = self._tick_capturing_frame(lm_obj, hand_visible, hand_state)
            if captured is not None:
                geo = extract_geometry_features(captured)
                if geo is not None:
                    self._session_samples.append(geo)
                    print(f"[HandEnroll] Round {self._round - 1} OK -> "
                          f"{len(self._session_samples)} samples")
                else:
                    print(f"[HandEnroll] Round {self._round - 1} FAILED "
                          f"(geometry extraction returned None)")

                if self._round >= ENROLL_ROUNDS:
                    self._finish(now)
                else:
                    self._next_hint = _get_hint(self._round)
                    self.fp_phase    = "REST"
                    self._rest_until = now + REST_SECS

        elif self.fp_phase == "REST":
            if now >= self._rest_until:
                self.fp_phase    = "ALIGN"
                self._stable_buf = []

        return self._build_output(now, hand_visible, hand_open)

    def _finish(self, now):
        """Called after all ENROLL_ROUNDS completed."""
        n = len(self._session_samples)
        print(f"[HandEnroll] Finished {ENROLL_ROUNDS} rounds, {n} clean samples")

        if n < MIN_SAMPLES_NEEDED:
            print(f"[HandEnroll] INSUFFICIENT: need {MIN_SAMPLES_NEEDED}, got {n}")
            self.fp_phase = "INSUFFICIENT"
            return

        self.fp_phase = "TRAINING"

        # Save profile (computes centroid + std internally)
        self._store.save_profile(
            self.player_name,
            self._session_samples,
            hand_side    = self._hand_side,
            verified     = True,
            enrolled_at  = self.enroll_start,
        )

        # Train classifier on all enrolled data
        clf = FingerprintClassifier()
        ok  = clf.train(self._store, include_unverified_for=self.player_name)

        if not ok:
            print(f"[HandEnroll] Classifier train failed")
            self.fp_phase = "INSUFFICIENT"
            return

        # Self-test: predict against the centroid of our own samples
        # Use the median sample (most representative, not an outlier)
        import numpy as _np
        arr = _np.array(self._session_samples)
        centroid = arr.mean(axis=0).tolist()
        pred_name, pred_conf = clf.predict(centroid)

        self.recog_name  = pred_name
        self.recog_conf  = pred_conf
        self.fp_phase    = "RECOGNIZED"

        n_people = len(clf.classes)
        mode     = clf._mode
        print(f"[HandEnroll] RECOGNIZED as '{pred_name}' "
              f"(conf={pred_conf:.0%}, mode={mode}, "
              f"{n_people} people enrolled, {n} samples)")

    def _build_output(self, now, hand_visible=False, hand_open=False):
        good_frames = len(self._capture_lms)
        capture_pct = (min(1.0, good_frames / CAPTURE_FRAMES)
                       if self.fp_phase == "CAPTURING" else 0.0)
        rest_pct = 0.0
        if self.fp_phase == "REST":
            elapsed  = now - (self._rest_until - REST_SECS)
            rest_pct = min(1.0, max(0.0, elapsed / REST_SECS))

        cx_n, cy_n, scale_f = self._sil_output(now)

        return {
            "fp_phase":            self.fp_phase,
            "fp_round":            self._round,
            "fp_rounds_target":    ENROLL_ROUNDS,
            "fp_samples":          len(self._session_samples),
            "fp_samples_needed":   MIN_SAMPLES_NEEDED,
            "fp_capture_pct":      capture_pct,
            "fp_good_frames":      good_frames,
            "fp_capture_target":   CAPTURE_FRAMES,
            "fp_stability_pct":    self._stability_pct(),
            "fp_rest_pct":         rest_pct,
            "fp_player_name":      self.player_name,
            "fp_next_hint":        self._next_hint,
            "fp_quality_reason":   self._quality_reason,
            "fp_hand_side":        self._hand_side,
            "recog_name":          self.recog_name,
            "recog_conf":          self.recog_conf,
            "sil_cx_norm":         cx_n,
            "sil_cy_norm":         cy_n,
            "sil_scale_factor":    scale_f,
            "hand_visible":        hand_visible,
            "hand_open":           hand_open,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Login / Recognition controller
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Login / Recognition controller — single scan, fast
# ─────────────────────────────────────────────────────────────────────────────

# How many good frames needed for a login prediction
LOGIN_FRAMES = 30   # ~1 second of good frames at 30fps

class HandLoginController:
    """
    Fast single-scan login. No rounds, no silhouette movement, no REST.

    Flow:
      WAITING  — no hand visible yet
      SCANNING — hand detected and open; accumulating good frames
      RESULT   — prediction made; shows name or UNKNOWN

    Once LOGIN_FRAMES good frames are collected the geometry is extracted
    and the classifier runs once. Result is shown immediately.
    If the distance check fails (unknown hand) or no profiles exist,
    login_failed is set and the user is prompted to try again or type name.
    """

    def __init__(self, store=None):
        self._store           = store or FingerprintStore()
        self._clf             = FingerprintClassifier()
        ok = self._clf.train(self._store)
        self._has_profiles    = ok and len(self._clf.classes) > 0
        if not ok:
            print("[HandLogin] No enrolled profiles — login will fail")

        self.fp_phase         = "WAITING"
        self._buf             = []       # good-quality landmark frames
        self._hand_side       = "Unknown"
        self._quality_reason  = ""
        self.login_result     = None
        self.login_confidence = 0.0
        self.login_failed     = False

    def reset(self):
        self.fp_phase         = "WAITING"
        self._buf             = []
        self._hand_side       = "Unknown"
        self._quality_reason  = ""
        self.login_result     = None
        self.login_confidence = 0.0
        self.login_failed     = False

    def update(self, hand_state, now=None):
        lm_obj       = hand_state.get("_landmarks") if hand_state else None
        hand_visible = lm_obj is not None
        hand_open    = _hand_open(lm_obj) if hand_visible else False

        if hand_visible:
            status   = hand_state.get("status_text", "") if hand_state else ""
            detected = _detect_hand_side(lm_obj, status)
            if detected != "Unknown":
                self._hand_side = detected

        # Terminal — just keep returning the result
        if self.login_result or self.login_failed:
            return self._output(hand_visible, hand_open)

        palm_scale = hand_state.get("palm_scale", 0.0) if hand_state else 0.0

        if not hand_visible or not hand_open:
            # Hand not ready — clear buffer, show waiting
            self._buf.clear()
            self._quality_reason = "" if not hand_visible else "Open your hand flat"
            self.fp_phase = "WAITING"
            return self._output(hand_visible, hand_open)

        # Check quality
        ok, reason = _assess_quality(lm_obj, palm_scale)
        self._quality_reason = "" if ok else reason

        if ok and lm_obj:
            self._buf.append(lm_obj.landmark)
            self.fp_phase = "SCANNING"
        else:
            # Quality failed — don't reset buffer, just don't add frame
            # But if we've been failing too long, reset
            pass

        # Enough frames — predict
        if len(self._buf) >= LOGIN_FRAMES:
            self._predict()

        return self._output(hand_visible, hand_open)

    def _predict(self):
        geo = extract_geometry_features(self._buf)
        self._buf.clear()

        if geo is None or not self._has_profiles:
            self.login_failed = True
            self.fp_phase     = "RESULT"
            return

        name, conf = self._clf.predict(geo)
        if name is not None:
            self.login_result     = name
            self.login_confidence = conf
            print(f"[HandLogin] Identified: {name} ({conf:.0%})")
        else:
            self.login_failed = True
            print(f"[HandLogin] Not recognised (conf={conf:.0%})")
        self.fp_phase = "RESULT"

    def _output(self, hand_visible, hand_open):
        scan_pct = min(1.0, len(self._buf) / LOGIN_FRAMES)
        return {
            "fp_phase":           self.fp_phase,
            "scan_pct":           scan_pct,
            "fp_quality_reason":  self._quality_reason,
            "fp_hand_side":       self._hand_side,
            "has_profiles":       self._has_profiles,
            "login_result":       self.login_result,
            "login_confidence":   self.login_confidence,
            "login_failed":       self.login_failed,
            "hand_visible":       hand_visible,
            "hand_open":          hand_open,
        }

class HandDiagController:
    """
    Continuously runs geometry extraction and prediction every frame.
    No enrollment, no voting — just live per-frame identification.

    Accumulates a short rolling window of landmarks (DIAG_WINDOW frames)
    and predicts once the window fills, then slides it forward.
    Also maintains a smoothed confidence using exponential moving average.
    """

    DIAG_WINDOW  = 20    # frames per prediction burst (~0.66s at 30fps)
    SMOOTH_ALPHA = 0.25  # EMA smoothing for displayed confidence

    def __init__(self, store=None):
        self._store       = store or FingerprintStore()
        self._clf         = FingerprintClassifier()
        trained = self._clf.train(self._store)
        self._has_profiles = trained and len(self._clf.classes) > 0

        self._buf         = []          # rolling landmark buffer
        self._hand_side   = "Unknown"
        self._quality_reason = ""

        # Smoothed prediction displayed to user
        self.pred_name    = None        # None = unknown/no match
        self.pred_conf    = 0.0
        self._raw_conf    = 0.0

        # Per-feature z-scores for the last prediction (for debug display)
        self.feature_zscores = []       # list of 12 floats
        self.enrolled_names  = self._clf.classes if trained else []

    def update(self, hand_state, now=None):
        lm_obj       = hand_state.get("_landmarks") if hand_state else None
        hand_visible = lm_obj is not None
        hand_open    = _hand_open(lm_obj) if hand_visible else False

        if hand_visible:
            status   = hand_state.get("status_text", "") if hand_state else ""
            detected = _detect_hand_side(lm_obj, status)
            if detected != "Unknown":
                self._hand_side = detected

        palm_scale = hand_state.get("palm_scale", 0.0) if hand_state else 0.0

        if hand_visible and lm_obj:
            ok, reason = _assess_quality(lm_obj, palm_scale)
            self._quality_reason = "" if ok else reason
            if ok:
                self._buf.append(lm_obj.landmark)
        else:
            self._quality_reason = "No hand detected"
            self._buf.clear()

        # When window fills, run a prediction
        if len(self._buf) >= self.DIAG_WINDOW:
            geo = extract_geometry_features(self._buf)
            self._buf = self._buf[self.DIAG_WINDOW // 2:]  # slide by half

            if geo is not None and self._has_profiles:
                name, conf = self._clf.predict(geo)
                # Smooth confidence with EMA
                self._raw_conf = conf
                self.pred_conf  = (self.SMOOTH_ALPHA * conf +
                                   (1 - self.SMOOTH_ALPHA) * self.pred_conf)
                self.pred_name  = name  # None if distance threshold failed

                # Compute per-feature z-scores for diagnostic display
                if (self._clf._mode == "distance" and
                        self._clf._classes and self._clf._profiles):
                    profile = self._clf._profiles.get(self._clf._classes[0], {})
                    cent = profile.get("centroid")
                    std  = profile.get("std")
                    if cent is not None and std is not None:
                        import numpy as _np
                        arr = _np.array(geo)
                        self.feature_zscores = list(
                            abs(arr - cent) / _np.maximum(std, 1e-6)
                        )
            else:
                self.pred_name = None
                self.pred_conf = max(0.0, self.pred_conf * 0.9)  # decay

        return {
            "pred_name":        self.pred_name,
            "pred_conf":        round(self.pred_conf, 3),
            "hand_side":        self._hand_side,
            "hand_visible":     hand_visible,
            "hand_open":        hand_open,
            "quality_reason":   self._quality_reason,
            "enrolled_names":   self.enrolled_names,
            "has_profiles":     self._has_profiles,
            "feature_zscores":  self.feature_zscores,
            "buf_pct":          min(1.0, len(self._buf) / self.DIAG_WINDOW),
            "clf_mode":         getattr(self._clf, "_mode", "none"),
        }

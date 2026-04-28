"""
Emotion Tracker — Facial Expression Detection via MediaPipe FaceMesh  (v4)

KEY CHANGE vs v3: personal baseline calibration.

All brow and eye signals are now scored as DEVIATIONS from the player's own
resting neutral face, measured during the first BASELINE_FRAMES frames of
detection. This means the system adapts to each person's face shape rather
than requiring every face to match a hardcoded absolute value.

Why this matters:
  - A person with naturally low-set brows will never clear an absolute
    "brow_height < 0.035" threshold just by frowning because their neutral
    brow_height is already below that value.
  - With a personal baseline, the same person only needs to move their brows
    DOWNWARD from THEIR own resting position — which is always detectable.

Detected states:
    Happy      — mouth corners raised + mouth widened
    Surprised  — mouth open OR dramatic brow raise above baseline
    Frustrated — brow DROPPED below baseline AND brow PINCHED inward
    Neutral    — none of the above

Calibration:
    The first BASELINE_FRAMES (60) frames with a detected face are used to
    build a baseline. During calibration the tracker returns Neutral and
    sets calibrated=False in the state dict. Progress is visible in the
    debug overlay. Baseline resets if the face is lost for > RESET_SECONDS.
"""

import mediapipe as mp
from collections import deque, Counter

mp_face_mesh = mp.solutions.face_mesh

# ============================================================
# CALIBRATION SETTINGS
# ============================================================
BASELINE_FRAMES  = 60    # frames to average for neutral baseline (~2 s at 30fps)
RESET_SECONDS    = 3.0   # face must be absent this long before baseline resets

# ============================================================
# LANDMARK INDICES
# ============================================================
MOUTH_LEFT        = 61
MOUTH_RIGHT       = 291
MOUTH_OPEN_TOP    = 13
MOUTH_OPEN_BOTTOM = 14

LEFT_EYE_TOP      = 159
LEFT_EYE_BOTTOM   = 145
RIGHT_EYE_TOP     = 386
RIGHT_EYE_BOTTOM  = 374

LEFT_BROW_INNER   = 107
LEFT_BROW_MID     = 105
LEFT_BROW_OUTER   = 70
RIGHT_BROW_INNER  = 336
RIGHT_BROW_MID    = 334
RIGHT_BROW_OUTER  = 300

FOREHEAD    = 10
CHIN        = 152
LEFT_CHEEK  = 234
RIGHT_CHEEK = 454
NOSE_TIP    = 1

DEBUG_POINT_GROUPS = {
    "mouth":  (MOUTH_LEFT, MOUTH_RIGHT, MOUTH_OPEN_TOP, MOUTH_OPEN_BOTTOM),
    "eyes":   (LEFT_EYE_TOP, LEFT_EYE_BOTTOM, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM),
    "brows":  (LEFT_BROW_INNER, LEFT_BROW_MID, LEFT_BROW_OUTER,
               RIGHT_BROW_INNER, RIGHT_BROW_MID, RIGHT_BROW_OUTER),
    "anchor": (FOREHEAD, CHIN, LEFT_CHEEK, RIGHT_CHEEK, NOSE_TIP),
}

DEBUG_COLORS = {
    "mouth":  (80,  220, 80),
    "eyes":   (255, 200,  0),
    "brows":  (0,   200, 255),
    "anchor": (180, 180, 180),
}

# ============================================================
# GEOMETRY HELPERS
# ============================================================

def _dist(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

def _face_height(lm):
    return max(_dist(lm[FOREHEAD], lm[CHIN]), 1e-6)

def _face_width(lm):
    return max(_dist(lm[LEFT_CHEEK], lm[RIGHT_CHEEK]), 1e-6)


# ============================================================
# RAW METRICS  (absolute, normalised by face size)
# ============================================================

def _compute_metrics(lm):
    """
    Compute normalised geometry values. These are the raw absolute values
    used both for baseline construction and for smile scoring (which is
    universal enough not to need personalisation).
    """
    face_h = _face_height(lm)
    face_w = _face_width(lm)

    mouth_width    = _dist(lm[MOUTH_LEFT], lm[MOUTH_RIGHT])
    mouth_open     = _dist(lm[MOUTH_OPEN_TOP], lm[MOUTH_OPEN_BOTTOM])
    mouth_center_y = (lm[MOUTH_OPEN_TOP].y + lm[MOUTH_OPEN_BOTTOM].y) / 2
    corner_avg_y   = (lm[MOUTH_LEFT].y + lm[MOUTH_RIGHT].y) / 2

    left_brow_h  = (lm[LEFT_EYE_TOP].y  - lm[LEFT_BROW_MID].y) / face_h
    right_brow_h = (lm[RIGHT_EYE_TOP].y - lm[RIGHT_BROW_MID].y) / face_h
    brow_height  = (left_brow_h + right_brow_h) / 2

    brow_gap   = _dist(lm[LEFT_BROW_INNER], lm[RIGHT_BROW_INNER])
    brow_pinch = 1.0 - (brow_gap / face_w)

    left_eye_open  = _dist(lm[LEFT_EYE_TOP],  lm[LEFT_EYE_BOTTOM])  / face_h
    right_eye_open = _dist(lm[RIGHT_EYE_TOP], lm[RIGHT_EYE_BOTTOM]) / face_h
    eye_open = (left_eye_open + right_eye_open) / 2

    return {
        "face_h":        face_h,
        "face_w":        face_w,
        "mouth_width_r": mouth_width / face_w,
        "mouth_open_r":  mouth_open  / face_h,
        "corner_rise":   (mouth_center_y - corner_avg_y) / face_h,
        "brow_height":   brow_height,
        "brow_pinch":    brow_pinch,
        "eye_open":      eye_open,
    }


# ============================================================
# EMOTION SCORING  (deviation-based for brow signals)
# ============================================================

def _smile_score(m):
    """
    Smile: mouth corners raised + mouth widened.
    Universal enough to use absolute thresholds — smiling looks the same
    across faces regardless of resting position.
    """
    score = 0.0

    if m["mouth_width_r"] > 0.40:
        score += min((m["mouth_width_r"] - 0.40) / 0.12, 0.45)

    if m["corner_rise"] > 0.022:
        score += min((m["corner_rise"] - 0.022) / 0.040, 0.45)

    # Small brow-lift bonus from personal baseline
    if m.get("brow_raise_delta", 0) > 0.008:
        score += min(m["brow_raise_delta"] / 0.030, 0.10)

    return min(score, 1.0)


def _surprise_score(m):
    """
    Surprised: two independent paths, either can fire:

    Path A — Classic surprise: mouth open + brow raised above baseline.
      Requires mouth_open_r > 0.040 as a soft gate (not hard — allows
      combining with Path B).

    Path B — Brow-only surprise: dramatic raise above personal baseline
      even without an open mouth. This handles the "eyes wide, brows up"
      expression that doesn't always open the mouth.
      Requires brow_raise_delta > RAISE_STRONG (more extreme than a Happy
      bonus lift) + eye widening above baseline.

    The two paths contribute independently; the total is capped at 1.0.
    """
    brow_raise = m.get("brow_raise_delta", 0.0)
    eye_raise  = m.get("eye_open_delta",   0.0)

    # --- Path A: mouth-open surprise ---
    path_a = 0.0
    if m["mouth_open_r"] > 0.040:
        path_a += min((m["mouth_open_r"] - 0.040) / 0.08, 0.50)
        if brow_raise > 0.010:
            path_a += min((brow_raise - 0.010) / 0.030, 0.30)
        if eye_raise > 0.003:
            path_a += min((eye_raise - 0.003) / 0.015, 0.20)

    # --- Path B: brow-only surprise ---
    # A dramatic raise above personal baseline fires Surprised even without
    # an open mouth. Eye widening adds a small bonus but is NOT required —
    # removing the eye gate prevents the score from bleeding out on frames
    # where brow raise is held but eye delta dips below threshold.
    path_b = 0.0
    if brow_raise > 0.022:
        path_b += min((brow_raise - 0.022) / 0.025, 0.70)
        if eye_raise > 0.003:
            path_b += min((eye_raise - 0.003) / 0.020, 0.30)

    return min(path_a + path_b, 1.0)


def _frustration_score(m):
    """
    Frustrated: brow DROPPED below baseline AND brow PINCHED beyond baseline.

    Both signals are now relative to the player's personal neutral:
      brow_drop_delta  — how much lower the brows are vs resting
      brow_pinch_delta — how much more inward the brows are vs resting

    Open mouth (> 0.040) hard-suppresses Frustrated.
    Both signals must be present; single-signal contribution capped at 0.20.
    """
    if m["mouth_open_r"] > 0.040:
        return 0.0

    brow_drop  = m.get("brow_drop_delta",  0.0)   # positive = brows dropped
    brow_pinch = m.get("brow_pinch_delta", 0.0)   # positive = more pinched

    # Even small deviations count — the key is they're relative to THIS face
    drop_signal  = min(brow_drop  / 0.012, 1.0) if brow_drop  > 0.002 else 0.0
    pinch_signal = min(brow_pinch / 0.030, 1.0) if brow_pinch > 0.007 else 0.0

    both_present = drop_signal > 0.0 and pinch_signal > 0.0
    brow_score   = (drop_signal + pinch_signal) / 2 * 0.75

    if not both_present:
        brow_score = min(brow_score, 0.20)

    # Lip compression bonus
    lip_comp  = 1.0 - min(m["mouth_open_r"] / 0.025, 1.0)
    lip_score = min((lip_comp - 0.55) / 0.30, 1.0) * 0.25 if lip_comp > 0.55 else 0.0

    return min(brow_score + lip_score, 1.0)


# ============================================================
# CLASSIFIER
# ============================================================

def _classify_emotion(smile, surprise, frustration):
    SMILE_THRESH       = 0.38
    SURPRISE_THRESH    = 0.40
    FRUSTRATION_THRESH = 0.36

    candidates = []
    if smile       >= SMILE_THRESH:       candidates.append(("Happy",      smile))
    if surprise    >= SURPRISE_THRESH:    candidates.append(("Surprised",  surprise))
    if frustration >= FRUSTRATION_THRESH: candidates.append(("Frustrated", frustration))

    if not candidates:
        return "Neutral", max(1.0 - smile - surprise - frustration, 0.1)

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0]


# ============================================================
# TRACKER
# ============================================================

class EmotionTracker:
    """
    Per-frame emotion detection with personal baseline calibration,
    deviation-based brow scoring, hysteresis, and temporal smoothing.

    Calibration phase:
        The first BASELINE_FRAMES frames with a detected face are averaged
        to build the neutral baseline for brow_height, brow_pinch, and
        eye_open. Until calibration is complete, all non-smile emotions
        return Neutral (no false positives during warmup).

    Hysteresis:
        HYSTERESIS_ENTER — extra score margin required to leave Neutral.
        HYSTERESIS_EXIT  — score must drop below this to return to Neutral.
    """

    HYSTERESIS_ENTER = 0.10
    HYSTERESIS_EXIT  = 0.22

    def __init__(self, history_size=10):
        self._face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,  # False = ~2x faster; brow/mouth geometry doesn't need iris refinement
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.raw_history    = deque(maxlen=history_size)
        self.raw_emotion    = "Unknown"
        self.stable_emotion = "Neutral"
        self.confidence     = 0.0
        self.scores         = {"smile": 0.0, "surprise": 0.0, "frustration": 0.0}
        self.face_detected  = False

        # Calibration state
        self._cal_samples        = []          # list of raw metric dicts
        self._baseline           = None        # averaged neutral baseline
        self._cal_frame_count    = 0
        self._last_face_time     = None        # monotonic time of last face frame
        self.calibrated          = False
        self.calibration_progress = 0          # 0–100 int

        # Hysteresis state
        self._locked_emotion = "Neutral"
        self._locked_score   = 0.0

        self._debug_landmarks = None

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def update(self, rgb_frame):
        import time
        now = time.monotonic()

        results = self._face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            self.face_detected    = False
            self.raw_emotion      = "Unknown"
            self._debug_landmarks = None

            # Reset baseline if face absent long enough
            if self._last_face_time is not None:
                if (now - self._last_face_time) > RESET_SECONDS:
                    self._reset_calibration()

            return self._build_state()

        self.face_detected    = True
        self._last_face_time  = now
        lm                    = results.multi_face_landmarks[0].landmark
        self._debug_landmarks = lm

        m = _compute_metrics(lm)

        # --- Calibration phase ---
        if not self.calibrated:
            self._cal_samples.append({
                "brow_height": m["brow_height"],
                "brow_pinch":  m["brow_pinch"],
                "eye_open":    m["eye_open"],
                "mouth_open_r": m["mouth_open_r"],
            })
            self._cal_frame_count += 1
            self.calibration_progress = int(
                self._cal_frame_count / BASELINE_FRAMES * 100
            )

            if self._cal_frame_count >= BASELINE_FRAMES:
                self._build_baseline()

            # During calibration: only smile can fire (it's universal)
            smile = _smile_score(m)
            self.scores = {"smile": round(smile, 3), "surprise": 0.0, "frustration": 0.0}
            self.raw_emotion    = "Happy" if smile >= 0.38 else "Neutral"
            self.stable_emotion = self.raw_emotion
            self.confidence     = round(smile, 3) if smile >= 0.38 else 0.1
            return self._build_state()

        # --- Post-calibration: inject deviation deltas into metrics ---
        m = self._inject_deltas(m)

        smile       = _smile_score(m)
        surprise    = _surprise_score(m)
        frustration = _frustration_score(m)

        self.scores = {
            "smile":       round(smile, 3),
            "surprise":    round(surprise, 3),
            "frustration": round(frustration, 3),
        }

        raw_label, raw_conf = _classify_emotion(smile, surprise, frustration)
        self.raw_emotion    = raw_label
        self.confidence     = round(raw_conf, 3)

        # --- Hysteresis ---
        score_map = {
            "Happy":      smile,
            "Surprised":  surprise,
            "Frustrated": frustration,
            "Neutral":    1.0 - max(smile, surprise, frustration),
        }

        if raw_label != self._locked_emotion:
            ENTRY = {
                "Happy":      0.38,
                "Surprised":  0.40,
                "Frustrated": 0.36,
            }.get(raw_label, 0.0)
            if raw_conf >= ENTRY + self.HYSTERESIS_ENTER:
                self._locked_emotion = raw_label
                self._locked_score   = raw_conf
        else:
            current_score = score_map.get(self._locked_emotion, 0.0)
            if current_score < self.HYSTERESIS_EXIT and self._locked_emotion != "Neutral":
                self._locked_emotion = "Neutral"
                self._locked_score   = 0.0

        self.raw_history.append(self._locked_emotion)
        counts              = Counter(self.raw_history)
        self.stable_emotion = counts.most_common(1)[0][0]

        return self._build_state()

    def get_round_snapshot(self):
        return {
            "emotion":            self.stable_emotion,
            "emotion_raw":        self.raw_emotion,
            "emotion_confidence": self.confidence,
            "smile_score":        self.scores["smile"],
            "surprise_score":     self.scores["surprise"],
            "frustration_score":  self.scores["frustration"],
        }

    def get_debug_overlay(self, frame_w, frame_h):
        if self._debug_landmarks is None:
            return None

        lm     = self._debug_landmarks
        points = {}
        for group, indices in DEBUG_POINT_GROUPS.items():
            pts = []
            for idx in indices:
                px = int(lm[idx].x * frame_w)
                py = int(lm[idx].y * frame_h)
                pts.append((px, py))
            points[group] = pts

        return {
            "points":              points,
            "scores":              dict(self.scores),
            "emotion":             self.stable_emotion,
            "confidence":          self.confidence,
            "calibrated":          self.calibrated,
            "calibration_progress": self.calibration_progress,
            "baseline":            dict(self._baseline) if self._baseline else None,
        }

    def reset(self):
        self.raw_history.clear()
        self.raw_emotion      = "Unknown"
        self.stable_emotion   = "Neutral"
        self.confidence       = 0.0
        self.scores           = {"smile": 0.0, "surprise": 0.0, "frustration": 0.0}
        self._locked_emotion  = "Neutral"
        self._locked_score    = 0.0
        self._debug_landmarks = None
        self._reset_calibration()

    def close(self):
        self._face_mesh.close()

    # ----------------------------------------------------------
    # Internal
    # ----------------------------------------------------------

    def _reset_calibration(self):
        self._cal_samples         = []
        self._cal_frame_count     = 0
        self._baseline            = None
        self.calibrated           = False
        self.calibration_progress = 0

    def _build_baseline(self):
        n = len(self._cal_samples)
        self._baseline = {
            "brow_height":  sum(s["brow_height"]  for s in self._cal_samples) / n,
            "brow_pinch":   sum(s["brow_pinch"]   for s in self._cal_samples) / n,
            "eye_open":     sum(s["eye_open"]      for s in self._cal_samples) / n,
            "mouth_open_r": sum(s["mouth_open_r"]  for s in self._cal_samples) / n,
        }
        self.calibrated           = True
        self.calibration_progress = 100
        self._cal_samples         = []   # free memory
        print(
            f"[Emotion] Baseline calibrated: "
            f"brow_h={self._baseline['brow_height']:.4f}  "
            f"brow_pinch={self._baseline['brow_pinch']:.4f}  "
            f"eye={self._baseline['eye_open']:.4f}"
        )

    def _inject_deltas(self, m):
        """
        Add deviation-from-baseline keys to the metrics dict.

        brow_drop_delta   — how much lower brows are vs baseline (positive = dropped)
        brow_raise_delta  — how much higher brows are vs baseline (positive = raised)
        brow_pinch_delta  — how much more pinched vs baseline (positive = more pinched)
        eye_open_delta    — how much wider eyes are vs baseline (positive = wider)
        """
        b = self._baseline
        out = dict(m)

        brow_diff = m["brow_height"] - b["brow_height"]
        out["brow_drop_delta"]  = max(-brow_diff, 0.0)   # positive when dropped
        out["brow_raise_delta"] = max( brow_diff, 0.0)   # positive when raised

        out["brow_pinch_delta"] = max(m["brow_pinch"] - b["brow_pinch"], 0.0)
        out["eye_open_delta"]   = max(m["eye_open"]   - b["eye_open"],   0.0)

        return out

    def _build_state(self):
        return {
            "raw_emotion":             self.raw_emotion,
            "stable_emotion":          self.stable_emotion,
            "confidence":              self.confidence,
            "scores":                  dict(self.scores),
            "face_detected":           self.face_detected,
            "calibrated":              self.calibrated,
            "calibration_progress":    self.calibration_progress,
        }

# emotion_tracker.py
# -------------------
# Real-time facial expression detection using MediaPipe FaceMesh.
#
# KEY DESIGN CHOICE: personal baseline calibration.
#
# All brow and eye signals are measured as DEVIATIONS from the player's own
# resting neutral face, which is estimated during the first BASELINE_FRAMES
# frames of detection. This means the system adapts to each person's face
# shape instead of requiring every face to match hardcoded absolute values.
#
# Why that matters:
#   A person with naturally low-set brows will never trigger a
#   "brow_height < 0.035" absolute threshold just by frowning, because their
#   neutral brow position is already below that value. With a personal
#   baseline, the same person only needs to move their brows DOWNWARD from
#   THEIR OWN resting position — which is always detectable.
#
# Detected states:
#   Happy      — mouth corners raised + mouth widened
#   Surprised  — mouth open OR dramatic brow raise above baseline
#   Frustrated — brow dropped below baseline AND brow pinched inward
#   Neutral    — none of the above thresholds met
#
# Calibration:
#   The first BASELINE_FRAMES (60) frames with a detected face are averaged
#   to build the baseline. During calibration the tracker returns Neutral
#   (except for smiles, which are universal) and sets calibrated=False.
#   Calibration progress (0-100) is available for a debug overlay.
#   The baseline resets if the face is absent for more than RESET_SECONDS.

import mediapipe as mp
from collections import deque, Counter

# MediaPipe FaceMesh solution handle — this is the module-level object we use
# to create FaceMesh instances.
mp_face_mesh = mp.solutions.face_mesh

# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

# Number of frames to average when building the neutral baseline (~2 s at 30 fps).
BASELINE_FRAMES = 60

# If the face disappears for this many seconds, reset the calibration entirely
# so the next person gets a fresh baseline.
RESET_SECONDS = 3.0

# ─────────────────────────────────────────────────────────────────────────────
# LANDMARK INDICES  (MediaPipe FaceMesh 468-point model)
# ─────────────────────────────────────────────────────────────────────────────
# Each number is the index of a specific point on the face mesh. MediaPipe
# always returns 468 landmarks in a fixed order, so these indices are stable.

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

# Anchor points used to normalise measurements by face size.
FOREHEAD    = 10
CHIN        = 152
LEFT_CHEEK  = 234
RIGHT_CHEEK = 454
NOSE_TIP    = 1

# Groups of landmark indices used when drawing the debug overlay.
DEBUG_POINT_GROUPS = {
    "mouth":  (MOUTH_LEFT, MOUTH_RIGHT, MOUTH_OPEN_TOP, MOUTH_OPEN_BOTTOM),
    "eyes":   (LEFT_EYE_TOP, LEFT_EYE_BOTTOM, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM),
    "brows":  (LEFT_BROW_INNER, LEFT_BROW_MID, LEFT_BROW_OUTER,
               RIGHT_BROW_INNER, RIGHT_BROW_MID, RIGHT_BROW_OUTER),
    "anchor": (FOREHEAD, CHIN, LEFT_CHEEK, RIGHT_CHEEK, NOSE_TIP),
}

# Colours (BGR) used when drawing each landmark group on the debug overlay.
DEBUG_COLORS = {
    "mouth":  (80,  220,  80),
    "eyes":   (255, 200,   0),
    "brows":  (0,   200, 255),
    "anchor": (180, 180, 180),
}

# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _dist(a, b):
    """Euclidean distance between two landmarks (in normalised 0-1 coordinates)."""
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def _face_height(lm):
    """Vertical distance from forehead to chin — used to normalise measurements."""
    return max(_dist(lm[FOREHEAD], lm[CHIN]), 1e-6)


def _face_width(lm):
    """Horizontal distance between cheeks — used to normalise brow gap."""
    return max(_dist(lm[LEFT_CHEEK], lm[RIGHT_CHEEK]), 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# RAW METRICS  (absolute values, normalised by face size)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(lm):
    """
    Extract normalised geometry values from a set of face landmarks.

    These are the raw absolute values used both for:
      - Building the neutral baseline during calibration.
      - Computing the smile score (which is universal enough to work
        without personalisation).

    All distances are divided by face height or face width so the values
    are roughly scale-independent (same person, different distance from camera).

    Returns a dict of named float metrics.
    """
    face_h = _face_height(lm)
    face_w = _face_width(lm)

    # Mouth geometry — how wide and how open the mouth is.
    mouth_width    = _dist(lm[MOUTH_LEFT], lm[MOUTH_RIGHT])
    mouth_open     = _dist(lm[MOUTH_OPEN_TOP], lm[MOUTH_OPEN_BOTTOM])
    mouth_center_y = (lm[MOUTH_OPEN_TOP].y + lm[MOUTH_OPEN_BOTTOM].y) / 2
    corner_avg_y   = (lm[MOUTH_LEFT].y + lm[MOUTH_RIGHT].y) / 2

    # Brow height = distance between the eye top and the brow midpoint,
    # normalised by face height. Higher value = brows further from eyes = raised.
    left_brow_h  = (lm[LEFT_EYE_TOP].y  - lm[LEFT_BROW_MID].y) / face_h
    right_brow_h = (lm[RIGHT_EYE_TOP].y - lm[RIGHT_BROW_MID].y) / face_h
    brow_height  = (left_brow_h + right_brow_h) / 2

    # Brow pinch = how close the inner brow corners are relative to face width.
    # A value close to 1.0 means very pinched / furrowed brows.
    brow_gap   = _dist(lm[LEFT_BROW_INNER], lm[RIGHT_BROW_INNER])
    brow_pinch = 1.0 - (brow_gap / face_w)

    # Eye openness = vertical opening of each eye, normalised by face height.
    left_eye_open  = _dist(lm[LEFT_EYE_TOP],  lm[LEFT_EYE_BOTTOM])  / face_h
    right_eye_open = _dist(lm[RIGHT_EYE_TOP], lm[RIGHT_EYE_BOTTOM]) / face_h
    eye_open = (left_eye_open + right_eye_open) / 2

    return {
        "face_h":        face_h,
        "face_w":        face_w,
        "mouth_width_r": mouth_width / face_w,   # mouth width as fraction of face width
        "mouth_open_r":  mouth_open  / face_h,   # mouth gap as fraction of face height
        "corner_rise":   (mouth_center_y - corner_avg_y) / face_h,  # positive = corners raised (smile)
        "brow_height":   brow_height,
        "brow_pinch":    brow_pinch,
        "eye_open":      eye_open,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EMOTION SCORING  (deviation-based for brow signals)
# ─────────────────────────────────────────────────────────────────────────────

def _smile_score(m):
    """
    Score how strongly the face is smiling (0.0 to 1.0).

    Two independent signals both contribute:
      1. Mouth width — smiling stretches the mouth sideways.
      2. Corner rise — smiling lifts the corners relative to the mouth centre.

    These are absolute thresholds rather than relative-to-baseline because
    smiling looks the same across virtually all face shapes.

    A small brow-raise bonus is added if the baseline deltas are available
    (post-calibration), since a genuine happy expression often lifts the brows.
    """
    score = 0.0

    # Mouth width contribution — the wider the mouth, the more it looks like a smile.
    if m["mouth_width_r"] > 0.40:
        score += min((m["mouth_width_r"] - 0.40) / 0.12, 0.45)

    # Corner lift contribution — lifted corners are the most reliable smile signal.
    if m["corner_rise"] > 0.022:
        score += min((m["corner_rise"] - 0.022) / 0.040, 0.45)

    # Small brow-raise bonus (only available post-calibration, so may be absent).
    if m.get("brow_raise_delta", 0) > 0.008:
        score += min(m["brow_raise_delta"] / 0.030, 0.10)

    return min(score, 1.0)


def _surprise_score(m):
    """
    Score how strongly the face looks surprised (0.0 to 1.0).

    Two independent paths can both contribute:

    Path A — Classic surprise: mouth open + brow raised above baseline.
        Requires mouth_open_r > 0.040 as a soft gate.

    Path B — Brow-only surprise: dramatic brow raise even without an open
        mouth. This handles the "eyes wide, brows up" expression that
        doesn't always open the mouth.

    The two paths contribute independently and their sum is capped at 1.0.
    """
    brow_raise = m.get("brow_raise_delta", 0.0)
    eye_raise  = m.get("eye_open_delta",   0.0)

    # Path A: open-mouth surprise — mouth open is the primary signal.
    path_a = 0.0
    if m["mouth_open_r"] > 0.040:
        path_a += min((m["mouth_open_r"] - 0.040) / 0.08, 0.50)
        if brow_raise > 0.010:
            path_a += min((brow_raise - 0.010) / 0.030, 0.30)
        if eye_raise > 0.003:
            path_a += min((eye_raise - 0.003) / 0.015, 0.20)

    # Path B: brow-only surprise.
    # We don't require eye widening as a hard gate — eye delta can dip below
    # threshold even when the brow raise is held, which would falsely kill
    # the score. Eye widening is a bonus here, not a requirement.
    path_b = 0.0
    if brow_raise > 0.022:
        path_b += min((brow_raise - 0.022) / 0.025, 0.70)
        if eye_raise > 0.003:
            path_b += min((eye_raise - 0.003) / 0.020, 0.30)

    return min(path_a + path_b, 1.0)


def _frustration_score(m):
    """
    Score how strongly the face looks frustrated (0.0 to 1.0).

    Frustration requires BOTH of these relative-to-baseline signals:
        brow_drop_delta  — brows dropped below the player's neutral resting height
        brow_pinch_delta — brows more pinched inward than at rest

    An open mouth hard-suppresses the frustration score entirely, because
    open-mouth expressions are already claimed by Surprised and Happy.

    A lip-compression bonus (very closed, tight mouth) adds a small extra
    signal — this is common in "gritting teeth" frustration.

    If only one of the two brow signals is present, the score is capped at
    0.20 to avoid false positives from slight natural asymmetry.
    """
    # Open mouth means it can't be frustration.
    if m["mouth_open_r"] > 0.040:
        return 0.0

    brow_drop  = m.get("brow_drop_delta",  0.0)   # positive = brows lower than baseline
    brow_pinch = m.get("brow_pinch_delta", 0.0)   # positive = brows more pinched than baseline

    # Scale each signal from 0.0 to 1.0 relative to its activation range.
    drop_signal  = min(brow_drop  / 0.012, 1.0) if brow_drop  > 0.002 else 0.0
    pinch_signal = min(brow_pinch / 0.030, 1.0) if brow_pinch > 0.007 else 0.0

    both_present = drop_signal > 0.0 and pinch_signal > 0.0
    brow_score   = (drop_signal + pinch_signal) / 2 * 0.75

    # If only one signal is present, cap contribution to avoid false positives.
    if not both_present:
        brow_score = min(brow_score, 0.20)

    # Lip compression: mouth almost fully closed suggests a tight/gritted expression.
    lip_comp  = 1.0 - min(m["mouth_open_r"] / 0.025, 1.0)
    lip_score = min((lip_comp - 0.55) / 0.30, 1.0) * 0.25 if lip_comp > 0.55 else 0.0

    return min(brow_score + lip_score, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def _classify_emotion(smile, surprise, frustration):
    """
    Map the three scores to a single emotion label and a confidence value.

    Each emotion has a minimum score threshold before it can be declared.
    If multiple emotions are above their thresholds, the one with the
    highest score wins. If none reach their threshold, return Neutral.
    """
    SMILE_THRESH       = 0.38
    SURPRISE_THRESH    = 0.40
    FRUSTRATION_THRESH = 0.36

    # Collect all emotions that are above their detection threshold.
    candidates = []
    if smile       >= SMILE_THRESH:       candidates.append(("Happy",      smile))
    if surprise    >= SURPRISE_THRESH:    candidates.append(("Surprised",  surprise))
    if frustration >= FRUSTRATION_THRESH: candidates.append(("Frustrated", frustration))

    if not candidates:
        # No emotion detected — confidence increases as all scores decrease.
        return "Neutral", max(1.0 - smile - surprise - frustration, 0.1)

    # Pick the highest-scoring candidate.
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0]


# ─────────────────────────────────────────────────────────────────────────────
# TRACKER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class EmotionTracker:
    """
    Per-frame emotion detection with personal baseline calibration,
    deviation-based brow scoring, hysteresis, and temporal smoothing.

    Calibration phase:
        The first BASELINE_FRAMES frames with a detected face are averaged
        to build the neutral baseline for brow_height, brow_pinch, and
        eye_open. Until calibration completes, non-smile emotions always
        return Neutral to avoid false positives during warmup.

    Hysteresis:
        Once an emotion is detected, a higher entry margin is required to
        switch TO that state (HYSTERESIS_ENTER), and the score must drop
        below a lower exit margin before switching AWAY from it
        (HYSTERESIS_EXIT). This prevents flickering between states.

    Temporal smoothing:
        The last `history_size` frames' locked emotion labels are kept in a
        rolling window. The most common label in the window becomes the
        reported stable_emotion.
    """

    # Extra score margin required on top of the threshold to enter a new state.
    HYSTERESIS_ENTER = 0.10
    # Score must fall below this before leaving an active non-Neutral state.
    HYSTERESIS_EXIT  = 0.22

    def __init__(self, history_size=10):
        """
        Set up the tracker with a fresh calibration state and empty history.

        history_size controls how many recent frames to use for the stable
        emotion vote — larger = more smoothing, but slower to react to changes.
        """
        # Set up MediaPipe FaceMesh.
        # refine_landmarks=False is ~2x faster — brow/mouth geometry doesn't
        # need the extra iris refinement that the full model provides.
        self._face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Rolling window of recent locked emotion labels for smoothing.
        self.raw_history    = deque(maxlen=history_size)
        self.raw_emotion    = "Unknown"
        self.stable_emotion = "Neutral"
        self.confidence     = 0.0
        self.scores         = {"smile": 0.0, "surprise": 0.0, "frustration": 0.0}
        self.face_detected  = False

        # Calibration state — tracking how many frames we've collected.
        self._cal_samples         = []    # list of raw metric dicts from the baseline window
        self._baseline            = None  # averaged baseline dict (set after calibration completes)
        self._cal_frame_count     = 0
        self._last_face_time      = None  # monotonic timestamp of the last frame with a face
        self.calibrated           = False
        self.calibration_progress = 0     # int 0-100, shown in the debug overlay

        # Hysteresis state — which emotion we're currently "locked into".
        self._locked_emotion = "Neutral"
        self._locked_score   = 0.0

        # Stored landmarks for the debug overlay (None when no face is detected).
        self._debug_landmarks = None

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, rgb_frame):
        """
        Process one camera frame and update all emotion state.

        Call this once per frame in the main loop before reading any
        emotion properties. Returns the current state dict (same as
        _build_state()).
        """
        import time
        now = time.monotonic()

        # Run MediaPipe face detection on this frame.
        results = self._face_mesh.process(rgb_frame)

        # --- No face detected ---
        if not results.multi_face_landmarks:
            self.face_detected    = False
            self.raw_emotion      = "Unknown"
            self._debug_landmarks = None

            # If the face has been gone long enough, discard the calibration
            # so the next person gets a fresh baseline.
            if self._last_face_time is not None:
                if (now - self._last_face_time) > RESET_SECONDS:
                    self._reset_calibration()

            return self._build_state()

        # --- Face detected ---
        self.face_detected    = True
        self._last_face_time  = now
        lm                    = results.multi_face_landmarks[0].landmark
        self._debug_landmarks = lm  # save for the debug overlay

        m = _compute_metrics(lm)

        # --- Calibration phase (before baseline is ready) ---
        if not self.calibrated:
            # Collect this frame's metrics for averaging into the baseline.
            self._cal_samples.append({
                "brow_height":  m["brow_height"],
                "brow_pinch":   m["brow_pinch"],
                "eye_open":     m["eye_open"],
                "mouth_open_r": m["mouth_open_r"],
            })
            self._cal_frame_count     += 1
            self.calibration_progress  = int(self._cal_frame_count / BASELINE_FRAMES * 100)

            # Once we have enough frames, build the baseline.
            if self._cal_frame_count >= BASELINE_FRAMES:
                self._build_baseline()

            # During calibration only smile can fire — it's universal and
            # doesn't need a personal baseline to be reliable.
            smile = _smile_score(m)
            self.scores         = {"smile": round(smile, 3), "surprise": 0.0, "frustration": 0.0}
            self.raw_emotion    = "Happy" if smile >= 0.38 else "Neutral"
            self.stable_emotion = self.raw_emotion
            self.confidence     = round(smile, 3) if smile >= 0.38 else 0.1
            return self._build_state()

        # --- Post-calibration: inject deviation deltas into the metrics ---
        # This adds brow_drop_delta, brow_raise_delta, etc. relative to the
        # player's personal neutral baseline.
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

        # --- Hysteresis: prevent rapid switching between emotion states ---
        # Map each emotion to its current score so we can check it easily.
        score_map = {
            "Happy":      smile,
            "Surprised":  surprise,
            "Frustrated": frustration,
            "Neutral":    1.0 - max(smile, surprise, frustration),
        }

        if raw_label != self._locked_emotion:
            # We might be switching to a new emotion — require the threshold
            # plus an extra HYSTERESIS_ENTER margin before committing.
            ENTRY = {
                "Happy":      0.38,
                "Surprised":  0.40,
                "Frustrated": 0.36,
            }.get(raw_label, 0.0)
            if raw_conf >= ENTRY + self.HYSTERESIS_ENTER:
                self._locked_emotion = raw_label
                self._locked_score   = raw_conf
        else:
            # We're already in this emotion — check if we should drop back to Neutral.
            current_score = score_map.get(self._locked_emotion, 0.0)
            if current_score < self.HYSTERESIS_EXIT and self._locked_emotion != "Neutral":
                self._locked_emotion = "Neutral"
                self._locked_score   = 0.0

        # Push the current locked label into the rolling history window.
        self.raw_history.append(self._locked_emotion)

        # The stable emotion is the most common label in the recent history window.
        counts = Counter(self.raw_history)
        top    = counts.most_common(1)
        self.stable_emotion = top[0][0] if top else "Neutral"

        return self._build_state()

    def get_round_snapshot(self):
        """
        Return a flat dict with all emotion values at the moment a round resolves.
        This is what gets stored in the player profile alongside each round.
        """
        return {
            "emotion":            self.stable_emotion,
            "emotion_raw":        self.raw_emotion,
            "emotion_confidence": self.confidence,
            "smile_score":        self.scores["smile"],
            "surprise_score":     self.scores["surprise"],
            "frustration_score":  self.scores["frustration"],
        }

    def get_debug_overlay(self, frame_w, frame_h):
        """
        Build a debug overlay payload for the renderer.

        Converts normalised landmark coordinates (0.0-1.0) to pixel
        coordinates using the given frame dimensions.

        Returns None if no face is currently detected.
        """
        if self._debug_landmarks is None:
            return None

        lm = self._debug_landmarks

        # Convert each debug group's landmarks from 0-1 normalised coords to pixels.
        points = {}
        for group, indices in DEBUG_POINT_GROUPS.items():
            pts = []
            for idx in indices:
                px = int(lm[idx].x * frame_w)
                py = int(lm[idx].y * frame_h)
                pts.append((px, py))
            points[group] = pts

        return {
            "points":               points,
            "scores":               dict(self.scores),
            "emotion":              self.stable_emotion,
            "confidence":           self.confidence,
            "calibrated":           self.calibrated,
            "calibration_progress": self.calibration_progress,
            "baseline":             dict(self._baseline) if self._baseline else None,
        }

    def reset(self):
        """
        Clear all transient state (history, locked emotion, etc.) and
        reset the calibration so the next person gets a fresh baseline.
        """
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
        """Release the MediaPipe FaceMesh resources. Call when exiting."""
        self._face_mesh.close()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _reset_calibration(self):
        """Discard any accumulated calibration data and start fresh."""
        self._cal_samples         = []
        self._cal_frame_count     = 0
        self._baseline            = None
        self.calibrated           = False
        self.calibration_progress = 0

    def _build_baseline(self):
        """
        Average the accumulated calibration samples into a single baseline dict.

        After this runs, _inject_deltas() can compute meaningful deviations.
        The sample list is cleared to free memory.
        """
        n = len(self._cal_samples)
        # Average each metric across all collected frames.
        self._baseline = {
            "brow_height":  sum(s["brow_height"]  for s in self._cal_samples) / n,
            "brow_pinch":   sum(s["brow_pinch"]   for s in self._cal_samples) / n,
            "eye_open":     sum(s["eye_open"]      for s in self._cal_samples) / n,
            "mouth_open_r": sum(s["mouth_open_r"]  for s in self._cal_samples) / n,
        }
        self.calibrated           = True
        self.calibration_progress = 100
        self._cal_samples         = []  # free memory — no longer needed
        print(
            f"[Emotion] Baseline calibrated: "
            f"brow_h={self._baseline['brow_height']:.4f}  "
            f"brow_pinch={self._baseline['brow_pinch']:.4f}  "
            f"eye={self._baseline['eye_open']:.4f}"
        )

    def _inject_deltas(self, m):
        """
        Add deviation-from-baseline keys to the raw metrics dict.

        Added keys:
            brow_drop_delta   — how much lower brows are vs baseline (positive = dropped)
            brow_raise_delta  — how much higher brows are vs baseline (positive = raised)
            brow_pinch_delta  — how much more pinched vs baseline (positive = more pinched)
            eye_open_delta    — how much wider eyes are vs baseline (positive = wider)

        These replace absolute brow/eye measurements for all post-calibration
        emotion scoring, making the system personal to each face.
        """
        b   = self._baseline
        out = dict(m)  # copy so we don't mutate the original

        # brow_height increases when brows raise (further from eyes in normalised coords).
        # So: positive diff = raised, negative diff = dropped.
        brow_diff              = m["brow_height"] - b["brow_height"]
        out["brow_drop_delta"]  = max(-brow_diff, 0.0)  # positive when brows are lower than baseline
        out["brow_raise_delta"] = max( brow_diff, 0.0)  # positive when brows are higher than baseline

        # How much more pinched and how much more open vs the person's own neutral.
        out["brow_pinch_delta"] = max(m["brow_pinch"] - b["brow_pinch"], 0.0)
        out["eye_open_delta"]   = max(m["eye_open"]   - b["eye_open"],   0.0)

        return out

    def _build_state(self):
        """
        Assemble the current tracker state into a flat dict for callers.

        This is the standard return value from update() and is also used
        internally whenever we need to return early (e.g. no face detected).
        """
        return {
            "raw_emotion":          self.raw_emotion,
            "stable_emotion":       self.stable_emotion,
            "confidence":           self.confidence,
            "scores":               dict(self.scores),
            "face_detected":        self.face_detected,
            "calibrated":           self.calibrated,
            "calibration_progress": self.calibration_progress,
        }

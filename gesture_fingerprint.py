"""
gesture_fingerprint.py
======================
Hand geometry biometric identification.

FEATURE EXTRACTION
------------------
12 scale-invariant ratio features computed from MediaPipe 2D landmarks.
All features are ratios — normalised by palm width or other hand lengths —
so they are theoretically independent of hand size and camera distance.

Features 0-2:  Finger length ratios (index/middle, middle/ring, ring/pinky)
Features 3-5:  Spread ratios (thumb, index, pinky tip positions / palm width)
Feature  6:    Palm aspect ratio (height / width)
Features 7-9:  Curl ratios (tip-to-MCP / tip-to-wrist per finger)
Feature  10:   Knuckle spacing (index MCP to pinky MCP / palm width)
Feature  11:   Index angle relative to palm axis

Research basis: Ghanbari et al. (ICEE 2022) achieved 98.7% accuracy on
hand geometry ID using MediaPipe landmark ratio features with SVM.
Their FPL method extracts phalanges length ratios — equivalent approach.

IDENTIFICATION STRATEGY (chosen based on enrolment count)
----------------------------------------------------------
1 person enrolled  →  Distance verification
   Compare new sample to stored centroid via normalised Euclidean distance.
   Threshold at 3 standard deviations. No "other class" needed.
   Returns (name, confidence) where confidence = 1 - normalised_distance.

2+ people enrolled →  SVM discrimination
   Standard RBF-kernel SVM trained on all enrolled samples.
   Returns (predicted_name, probability).

Why this matters: an SVM is a discriminative classifier — it learns the
boundary between classes. With only one class it cannot draw a boundary
and will always return that class with high confidence (meaningless).
Distance-based verification is the correct approach for single-user setups.

SAMPLE REQUIREMENTS (from research)
------------------------------------
Ghanbari et al. used multiple images per session.  A practical minimum
for reliable distance-based verification is 10 samples (rounds), giving
a stable centroid. For SVM with 2 users, 15+ samples per person.
We use ENROLL_ROUNDS = 20 in hand_enroll_state.py to be safe.

STORAGE
-------
~/Desktop/CapStone/fingerprints/<name>_fp.json
{
  "player_name": "Zac",
  "samples": [[f0,f1,...,f11], ...],   # one 12-vector per round
  "centroid": [f0,...,f11],            # mean across all samples
  "std":      [s0,...,s11],            # std across all samples (per feature)
  "n_samples": 20,
  "hand_side": "Right",
  "verified": true,
  "enrolled_at": "2026-04-13T..."
}
"""

import json
import math
import time
from pathlib import Path

import numpy as np

FINGERPRINT_DIR = Path.home() / "Desktop" / "CapStone" / "fingerprints"

# Minimum samples before we attempt verification (distance-based)
MIN_SAMPLES_DISTANCE  = 10   # need enough for a stable centroid
# Minimum samples per person before SVM is worth using
MIN_SAMPLES_SVM       = 15
# Distance threshold: how many std-devs away before we reject identity
# 2.5 is a loose threshold — tighten to 2.0 for stricter security
DISTANCE_THRESHOLD    = 2.5
# Minimum frames per captured round for feature extraction
MIN_DWELL_FRAMES      = 8

# MediaPipe landmark indices
WRIST      = 0
THUMB_TIP  = 4
INDEX_MCP  = 5
INDEX_TIP  = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_MCP   = 13
RING_TIP   = 16
PINKY_MCP  = 17
PINKY_TIP  = 20


def _dist(lm, a, b):
    return math.sqrt((lm[a].x - lm[b].x)**2 + (lm[a].y - lm[b].y)**2)


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_geometry_features(landmark_frames):
    """
    Extract 12 scale-invariant geometry features from a list of MediaPipe
    landmark objects (one per frame).

    Each frame contributes one feature vector; we average across all valid
    frames for stability.  Frames where palm_w is near-zero are skipped.

    Returns list[12] of floats, or None if insufficient valid frames.
    """
    if not landmark_frames or len(landmark_frames) < MIN_DWELL_FRAMES:
        return None

    feat_sum = [0.0] * 12
    valid    = 0

    for lm in landmark_frames:
        try:
            palm_w = _dist(lm, INDEX_MCP, PINKY_MCP)
            if palm_w < 1e-6:
                continue
            palm_h   = _dist(lm, WRIST, MIDDLE_MCP)
            idx_len  = _dist(lm, INDEX_TIP,  WRIST)
            mid_len  = _dist(lm, MIDDLE_TIP, WRIST)
            rng_len  = _dist(lm, RING_TIP,   WRIST)
            pky_len  = _dist(lm, PINKY_TIP,  WRIST)

            if any(v < 1e-6 for v in [mid_len, rng_len, pky_len, idx_len]):
                continue

            idx_curl = _dist(lm, INDEX_TIP,  INDEX_MCP)
            mid_curl = _dist(lm, MIDDLE_TIP, MIDDLE_MCP)
            rng_curl = _dist(lm, RING_TIP,   RING_MCP)

            # Index angle vs palm axis
            pdx = lm[MIDDLE_MCP].x - lm[WRIST].x
            pdy = lm[MIDDLE_MCP].y - lm[WRIST].y
            idx_dx = lm[INDEX_TIP].x - lm[INDEX_MCP].x
            idx_dy = lm[INDEX_TIP].y - lm[INDEX_MCP].y
            pmag = math.sqrt(pdx**2 + pdy**2)
            imag = math.sqrt(idx_dx**2 + idx_dy**2)
            if pmag > 1e-6 and imag > 1e-6:
                cos_a     = (pdx*idx_dx + pdy*idx_dy) / (pmag * imag)
                idx_angle = math.acos(max(-1.0, min(1.0, cos_a)))
            else:
                idx_angle = 0.0

            f = [
                idx_len / mid_len,                        # 0 index/middle ratio
                mid_len / rng_len,                        # 1 middle/ring ratio
                rng_len / pky_len,                        # 2 ring/pinky ratio
                _dist(lm, THUMB_TIP, INDEX_MCP) / palm_w, # 3 thumb spread
                _dist(lm, INDEX_TIP, MIDDLE_MCP) / palm_w,# 4 index spread
                _dist(lm, PINKY_TIP, RING_MCP) / palm_w,  # 5 pinky spread
                palm_h / palm_w,                          # 6 palm aspect
                idx_curl / idx_len,                       # 7 index curl
                mid_curl / mid_len,                       # 8 middle curl
                rng_curl / rng_len,                       # 9 ring curl
                _dist(lm, INDEX_MCP, PINKY_MCP) / palm_w, # 10 knuckle span
                idx_angle / math.pi,                      # 11 index angle
            ]
            for i, v in enumerate(f):
                feat_sum[i] += v
            valid += 1

        except (AttributeError, IndexError, ZeroDivisionError):
            continue

    if valid < MIN_DWELL_FRAMES // 2:
        return None

    return [v / valid for v in feat_sum]


# ─────────────────────────────────────────────────────────────────────────────
# Storage
# ─────────────────────────────────────────────────────────────────────────────

class FingerprintStore:

    def __init__(self):
        FINGERPRINT_DIR.mkdir(parents=True, exist_ok=True)

    def _path(self, name):
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
        return FINGERPRINT_DIR / f"{safe.lower()}_fp.json"

    def save_profile(self, name, samples, hand_side="Unknown",
                     verified=True, enrolled_at=None):
        """
        Save samples plus derived centroid and std.
        centroid and std are used for distance-based verification.
        """
        arr = np.array(samples, dtype=np.float64)
        centroid = arr.mean(axis=0).tolist()
        std      = arr.std(axis=0).tolist()
        # Replace zero stds with a small floor to avoid division by zero
        std = [max(s, 1e-6) for s in std]

        data = {
            "player_name":    name,
            "samples":        samples,
            "centroid":       centroid,
            "std":            std,
            "n_samples":      len(samples),
            "hand_side":      hand_side,
            "verified":       verified,
            "enrolled_at":    enrolled_at or time.strftime("%Y-%m-%dT%H:%M:%S"),
            "feature_version": "v3_geometry_only",
        }
        try:
            self._path(name).write_text(json.dumps(data, indent=2))
            print(f"[FingerprintStore] Saved {len(samples)} samples for {name} "
                  f"-> {self._path(name).name}")
        except Exception as e:
            print(f"[FingerprintStore] Save error: {e}")

    # Keep old name for compatibility
    def save_samples(self, name, samples, verified=False, enrolled_at=None):
        self.save_profile(name, samples, verified=verified,
                          enrolled_at=enrolled_at)

    def load_profile(self, name):
        p = self._path(name)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    # Keep old name for compatibility
    def load_samples(self, name):
        return self.load_profile(name)

    def list_all_enrolled(self):
        result = []
        for p in FINGERPRINT_DIR.glob("*_fp.json"):
            try:
                d = json.loads(p.read_text())
                result.append((d["player_name"],
                                d.get("n_samples", 0),
                                d.get("verified", False)))
            except Exception:
                pass
        return result

    def list_verified(self):
        return [n for n, _, v in self.list_all_enrolled() if v]

    def mark_verified(self, name):
        d = self.load_profile(name)
        if d:
            d["verified"] = True
            try:
                self._path(name).write_text(json.dumps(d, indent=2))
            except Exception as e:
                print(f"[FingerprintStore] Verify error: {e}")

    def delete(self, name):
        p = self._path(name)
        if p.exists():
            p.unlink()

    def count_enrolled(self):
        return len(self.list_all_enrolled())


# ─────────────────────────────────────────────────────────────────────────────
# Identification / Verification
# ─────────────────────────────────────────────────────────────────────────────

class FingerprintClassifier:
    """
    Unified classifier that switches strategy based on how many
    people are enrolled:

    1 person  → Distance verification against stored centroid.
                Returns (name, confidence) where confidence measures
                how close the sample is to the enrolled centroid.

    2+ people → SVM discrimination between enrolled people.
                Returns (predicted_name, probability).
    """

    def __init__(self):
        self._store      = None
        self._mode       = None    # "distance" | "svm"
        self._svm        = None
        self._scaler     = None
        self._profiles   = {}      # name -> {centroid, std} for distance mode
        self._classes    = []
        self._trained    = False

    @property
    def trained(self):
        return self._trained

    @property
    def classes(self):
        return list(self._classes)

    def train(self, store, include_unverified_for=None):
        """
        Load all enrolled profiles and choose the right strategy.
        Returns True if ready to predict.
        """
        self._store = store
        self._profiles = {}
        self._trained  = False

        enrolled = store.list_all_enrolled()

        for name, n_samples, verified in enrolled:
            if not verified and name != include_unverified_for:
                continue
            profile = store.load_profile(name)
            if not profile or not profile.get("samples"):
                continue
            samples = [s for s in profile["samples"] if len(s) == 12]
            if len(samples) < 2:
                continue

            arr = np.array(samples, dtype=np.float64)
            centroid = arr.mean(axis=0)
            std      = np.maximum(arr.std(axis=0), 1e-6)
            self._profiles[name] = {
                "centroid": centroid,
                "std":      std,
                "n":        len(samples),
            }

        n_people = len(self._profiles)
        if n_people == 0:
            return False

        self._classes = sorted(self._profiles.keys())

        if n_people == 1:
            # Distance verification — only one person to check against
            self._mode = "distance"
            self._trained = True
            name = self._classes[0]
            n = self._profiles[name]["n"]
            print(f"[Classifier] Distance mode: 1 person ({name}, {n} samples)")
            return True

        # 2+ people — use SVM
        X, y = [], []
        for name, info in self._profiles.items():
            profile = store.load_profile(name)
            for s in profile["samples"]:
                if len(s) == 12:
                    X.append(s)
                    y.append(name)

        if len(X) < 4:
            return False

        try:
            from sklearn.svm import SVC
            from sklearn.preprocessing import StandardScaler

            X_arr = np.array(X, dtype=np.float32)
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X_arr)
            self._svm = SVC(
                kernel="rbf", C=5.0, gamma="scale",
                probability=True, class_weight="balanced"
            )
            self._svm.fit(X_scaled, y)
            self._mode    = "svm"
            self._trained = True
            print(f"[Classifier] SVM mode: {n_people} people, {len(X)} samples")
            return True
        except Exception as e:
            print(f"[Classifier] SVM train error: {e}")
            # Fall back to distance on error
            self._mode    = "distance"
            self._trained = True
            return True

    def predict(self, features):
        """
        Returns (name: str | None, confidence: float 0..1).
        confidence meaning:
          distance mode: 1.0 = perfect match, 0.0 = very far away
          svm mode:      classifier probability for predicted class
        """
        if not self._trained or not features or len(features) != 12:
            return None, 0.0

        feat = np.array(features, dtype=np.float64)

        if self._mode == "distance":
            name = self._classes[0]
            info = self._profiles[name]
            cent = info["centroid"]
            std  = info["std"]
            # Normalised Euclidean (Mahalanobis with diagonal covariance)
            z_scores  = np.abs((feat - cent) / std)
            mean_z    = float(z_scores.mean())
            # Confidence: 1.0 when z=0 (perfect), 0.0 when z >= threshold
            confidence = max(0.0, 1.0 - mean_z / DISTANCE_THRESHOLD)
            if mean_z <= DISTANCE_THRESHOLD:
                return name, confidence
            else:
                return None, confidence  # failed distance check

        elif self._mode == "svm":
            try:
                X = self._scaler.transform([features])
                probs = self._svm.predict_proba(X)[0]
                idx   = int(np.argmax(probs))
                name  = self._svm.classes_[idx]
                conf  = float(probs[idx])
                return name, conf
            except Exception:
                return None, 0.0

        return None, 0.0

    def verify(self, name, features):
        """
        Binary check: does this feature vector match the named person?
        Returns (match: bool, confidence: float).
        Used during enrollment self-test and login verification.
        """
        pred_name, conf = self.predict(features)
        if pred_name is None:
            return False, conf
        if self._mode == "distance":
            # Distance mode: any above-threshold match counts as this person
            return True, conf
        else:
            return (pred_name == name), conf


# ── Compatibility stubs (for legacy squid_fingerprint_state.py) ──────────────

def extract_movement_features(trajectory):
    """Legacy stub — movement features no longer used. Returns None."""
    return None


def combine_features(geometry, movement):
    """Legacy stub — geometry-only features now. Returns geometry as-is."""
    return geometry


MIN_SAMPLES_FOR_TRAINING = MIN_SAMPLES_DISTANCE   # legacy alias
VERIFY_WINDOW    = 20
VERIFY_THRESHOLD = 0.80

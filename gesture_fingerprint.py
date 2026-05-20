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
1 person enrolled  ->  Distance verification
   Compare new sample to stored centroid via normalised Euclidean distance.
   Threshold at 3 standard deviations. No "other class" needed.
   Returns (name, confidence) where confidence = 1 - normalised_distance.

2+ people enrolled ->  SVM discrimination
   Standard RBF-kernel SVM trained on all enrolled samples.
   Returns (predicted_name, probability).

Why this matters: an SVM is a discriminative classifier — it learns the
boundary between classes. With only one class it cannot draw a boundary
and will always return that class with high confidence (meaningless).
Distance-based verification is the correct approach for single-user setups.

SAMPLE REQUIREMENTS (from research)
------------------------------------
Ghanbari et al. used multiple images per session. A practical minimum
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

# Try to import the shared path config; fall back to a sensible default.
try:
    from capstone_paths import CAPSTONE_DIR
    FINGERPRINT_DIR = CAPSTONE_DIR / "fingerprints"
except ImportError:
    FINGERPRINT_DIR = Path.home() / "Desktop" / "CapStone" / "fingerprints"

# Minimum samples needed before we can compute a stable centroid for verification.
MIN_SAMPLES_DISTANCE = 10
# Minimum samples per person before SVM training is worthwhile.
MIN_SAMPLES_SVM      = 15
# How many standard deviations away a sample can be before we reject it as "not this person".
# 2.5 is fairly lenient — lower to 2.0 for stricter matching.
DISTANCE_THRESHOLD   = 2.5
# Minimum frames held inside the dot before we extract features for a round.
MIN_DWELL_FRAMES     = 8

# MediaPipe landmark indices — named so the feature code is readable.
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
    """Euclidean distance between two MediaPipe landmarks (2D, using x and y only)."""
    return math.sqrt((lm[a].x - lm[b].x) ** 2 + (lm[a].y - lm[b].y) ** 2)


# =============================================================================
# Feature extraction
# =============================================================================

def extract_geometry_features(landmark_frames):
    """
    Extract 12 scale-invariant geometry features from a list of MediaPipe
    landmark objects (one per captured frame).

    We compute a feature vector for every frame and average them together —
    this smooths out jitter and gives a more stable reading than a single frame.
    Frames where the palm width is near-zero (hand edge-on to camera) are skipped.

    Returns a list of 12 floats, or None if there weren't enough valid frames.
    """
    # Need at least MIN_DWELL_FRAMES frames to compute a reliable average.
    if not landmark_frames or len(landmark_frames) < MIN_DWELL_FRAMES:
        return None

    feat_sum = [0.0] * 12  # running total for each of the 12 features
    valid    = 0            # count of frames that passed quality checks

    for lm in landmark_frames:
        try:
            # Palm width is the normalisation denominator — skip if hand is edge-on.
            palm_w = _dist(lm, INDEX_MCP, PINKY_MCP)
            if palm_w < 1e-6:
                continue

            palm_h  = _dist(lm, WRIST,      MIDDLE_MCP)
            idx_len = _dist(lm, INDEX_TIP,  WRIST)
            mid_len = _dist(lm, MIDDLE_TIP, WRIST)
            rng_len = _dist(lm, RING_TIP,   WRIST)
            pky_len = _dist(lm, PINKY_TIP,  WRIST)

            # Skip frames where any finger length is degenerate (hand partially visible).
            if any(v < 1e-6 for v in [idx_len, mid_len, rng_len, pky_len]):
                continue

            # Curl = distance from fingertip to its MCP knuckle (small = finger is curled).
            idx_curl = _dist(lm, INDEX_TIP,  INDEX_MCP)
            mid_curl = _dist(lm, MIDDLE_TIP, MIDDLE_MCP)
            rng_curl = _dist(lm, RING_TIP,   RING_MCP)

            # Index finger angle relative to the palm axis (wrist -> middle MCP).
            # We compute the cosine of the angle between those two vectors.
            pdx   = lm[MIDDLE_MCP].x - lm[WRIST].x
            pdy   = lm[MIDDLE_MCP].y - lm[WRIST].y
            idx_dx = lm[INDEX_TIP].x - lm[INDEX_MCP].x
            idx_dy = lm[INDEX_TIP].y - lm[INDEX_MCP].y
            pmag   = math.sqrt(pdx**2 + pdy**2)
            imag   = math.sqrt(idx_dx**2 + idx_dy**2)

            if pmag > 1e-6 and imag > 1e-6:
                cos_a     = (pdx * idx_dx + pdy * idx_dy) / (pmag * imag)
                idx_angle = math.acos(max(-1.0, min(1.0, cos_a)))  # clamp for safety
            else:
                idx_angle = 0.0

            # Build the 12-element feature vector for this frame.
            f = [
                idx_len / mid_len,                          # 0: index/middle length ratio
                mid_len / rng_len,                          # 1: middle/ring length ratio
                rng_len / pky_len,                          # 2: ring/pinky length ratio
                _dist(lm, THUMB_TIP,  INDEX_MCP)  / palm_w, # 3: thumb spread
                _dist(lm, INDEX_TIP,  MIDDLE_MCP) / palm_w, # 4: index spread
                _dist(lm, PINKY_TIP,  RING_MCP)   / palm_w, # 5: pinky spread
                palm_h / palm_w,                            # 6: palm aspect ratio
                idx_curl / idx_len,                         # 7: index curl ratio
                mid_curl / mid_len,                         # 8: middle curl ratio
                rng_curl / rng_len,                         # 9: ring curl ratio
                _dist(lm, INDEX_MCP, PINKY_MCP)   / palm_h, # 10: knuckle span/height ratio
                idx_angle / math.pi,                        # 11: index angle (normalised 0-1)
            ]

            # Accumulate into the running total.
            for i, v in enumerate(f):
                feat_sum[i] += v
            valid += 1

        except (AttributeError, IndexError, ZeroDivisionError):
            # Skip malformed landmark objects rather than crashing.
            continue

    # Require at least half the frames to be valid.
    if valid < MIN_DWELL_FRAMES // 2:
        return None

    # Return the per-feature average across all valid frames.
    return [v / valid for v in feat_sum]


# =============================================================================
# Storage
# =============================================================================

class FingerprintStore:
    """
    Loads and saves fingerprint profiles as JSON files.
    Each profile stores the raw samples plus the computed centroid and std
    needed for distance-based verification.
    Files live in FINGERPRINT_DIR as <name>_fp.json.
    """

    def __init__(self):
        # Make sure the fingerprints directory exists.
        FINGERPRINT_DIR.mkdir(parents=True, exist_ok=True)

    def _path(self, name):
        """Build a safe file path for a given player name (strips special characters)."""
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
        return FINGERPRINT_DIR / f"{safe.lower()}_fp.json"

    def save_profile(self, name, samples, hand_side="Unknown",
                     verified=True, enrolled_at=None):
        """
        Persist a player's fingerprint samples to disk.
        Also computes and stores the centroid (mean) and std across all samples —
        these are used later for distance-based identity verification.
        """
        arr      = np.array(samples, dtype=np.float64)
        centroid = arr.mean(axis=0).tolist()
        std      = arr.std(axis=0).tolist()
        # Replace any zero std values with a tiny floor to avoid divide-by-zero later.
        std = [max(s, 1e-6) for s in std]

        data = {
            "player_name":     name,
            "samples":         samples,
            "centroid":        centroid,
            "std":             std,
            "n_samples":       len(samples),
            "hand_side":       hand_side,
            "verified":        verified,
            "enrolled_at":     enrolled_at or time.strftime("%Y-%m-%dT%H:%M:%S"),
            "feature_version": "v3_geometry_only",
        }
        try:
            self._path(name).write_text(json.dumps(data, indent=2))
            print(f"[FingerprintStore] Saved {len(samples)} samples for {name} "
                  f"-> {self._path(name).name}")
        except Exception as e:
            print(f"[FingerprintStore] Save error: {e}")

    # Kept for backwards compatibility with older callers.
    def save_samples(self, name, samples, verified=False, enrolled_at=None):
        self.save_profile(name, samples, verified=verified, enrolled_at=enrolled_at)

    def load_profile(self, name):
        """Load and return a player's profile dict, or None if it doesn't exist."""
        p = self._path(name)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    # Kept for backwards compatibility.
    def load_samples(self, name):
        return self.load_profile(name)

    def list_all_enrolled(self):
        """
        Scan the fingerprints directory and return a list of
        (player_name, n_samples, verified) tuples for every profile found.
        """
        result = []
        for p in FINGERPRINT_DIR.glob("*_fp.json"):
            try:
                d = json.loads(p.read_text())
                result.append((
                    d["player_name"],
                    d.get("n_samples", 0),
                    d.get("verified", False),
                ))
            except Exception:
                pass  # skip corrupted files
        return result

    def list_verified(self):
        """Return just the names of players whose profiles are marked verified."""
        return [name for name, _, verified in self.list_all_enrolled() if verified]

    def mark_verified(self, name):
        """Set the 'verified' flag to True on a player's stored profile."""
        d = self.load_profile(name)
        if d:
            d["verified"] = True
            try:
                self._path(name).write_text(json.dumps(d, indent=2))
            except Exception as e:
                print(f"[FingerprintStore] Verify error: {e}")

    def delete(self, name):
        """Delete a player's fingerprint file from disk."""
        p = self._path(name)
        if p.exists():
            p.unlink()

    def count_enrolled(self):
        """Return the total number of enrolled profiles (verified or not)."""
        return len(self.list_all_enrolled())


# =============================================================================
# Identification / Verification
# =============================================================================

class FingerprintClassifier:
    """
    Switches between two identification strategies based on how many people
    are enrolled:

    1 person  -> Distance verification: compare the new sample's feature vector
                 to the enrolled person's stored centroid using normalised
                 Euclidean distance. Returns (name, confidence) where confidence
                 is 1.0 for a perfect match and 0.0 when at the rejection threshold.

    2+ people -> SVM discrimination: train a classifier on all enrolled samples
                 and let it pick the most likely person. Returns (name, probability).
    """

    def __init__(self):
        self._store    = None
        self._mode     = None      # "distance" or "svm"
        self._svm      = None
        self._scaler   = None
        self._profiles = {}        # name -> {centroid, std, n} used in distance mode
        self._classes  = []        # sorted list of enrolled names
        self._trained  = False

    @property
    def trained(self):
        return self._trained

    @property
    def classes(self):
        return list(self._classes)

    def train(self, store, include_unverified_for=None):
        """
        Load all enrolled profiles from the store and pick the right strategy.
        `include_unverified_for` lets us include an in-progress enrollment (the
        person being enrolled right now, who isn't verified yet).
        Returns True if the classifier is ready to predict.
        """
        self._store    = store
        self._profiles = {}
        self._trained  = False

        enrolled = store.list_all_enrolled()

        # Build the profiles dict, skipping unverified players unless they're
        # the one currently being enrolled.
        for name, n_samples, verified in enrolled:
            if not verified and name != include_unverified_for:
                continue

            profile = store.load_profile(name)
            if not profile or not profile.get("samples"):
                continue

            # Only keep samples with the right feature length (guards against old data).
            samples = [s for s in profile["samples"] if len(s) == 12]
            if len(samples) < 2:
                continue

            arr      = np.array(samples, dtype=np.float64)
            centroid = arr.mean(axis=0)
            std      = np.maximum(arr.std(axis=0), 1e-6)

            self._profiles[name] = {
                "centroid": centroid,
                "std":      std,
                "n":        len(samples),
            }

        n_people = len(self._profiles)
        if n_people == 0:
            return False  # nothing to train on

        self._classes = sorted(self._profiles.keys())

        if n_people == 1:
            # Only one person — use distance verification (SVM can't draw a boundary
            # between classes when there's only one class).
            self._mode    = "distance"
            self._trained = True
            name = self._classes[0]
            print(f"[Classifier] Distance mode: 1 person ({name}, "
                  f"{self._profiles[name]['n']} samples)")
            return True

        # Two or more people — train an SVM to discriminate between them.
        X, y = [], []
        for name, info in self._profiles.items():
            profile = store.load_profile(name)
            for s in profile["samples"]:
                if len(s) == 12:
                    X.append(s)
                    y.append(name)

        # Need at least 4 total samples to do anything meaningful.
        if len(X) < 4:
            return False

        try:
            from sklearn.svm import SVC
            from sklearn.preprocessing import StandardScaler

            # Standardise features so the SVM isn't skewed by scale differences.
            X_arr         = np.array(X, dtype=np.float32)
            self._scaler  = StandardScaler()
            X_scaled      = self._scaler.fit_transform(X_arr)

            # RBF kernel SVM with balanced class weights handles unequal sample counts.
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
            # If SVM fails (e.g. sklearn not installed), fall back to distance mode.
            self._mode    = "distance"
            self._trained = True
            return True

    def predict(self, features):
        """
        Identify the player from a 12-element feature vector.

        Returns (name, confidence):
            name       -- predicted player name, or None if identity rejected
            confidence -- 0.0 to 1.0; meaning depends on mode:
                          distance mode: how close the sample is to the centroid
                          SVM mode:      classifier probability for the predicted class
        """
        if not self._trained or not features or len(features) != 12:
            return None, 0.0

        feat = np.array(features, dtype=np.float64)

        if self._mode == "distance":
            name = self._classes[0]
            info = self._profiles[name]
            # Normalised Euclidean distance (like Mahalanobis with diagonal covariance).
            # Each dimension is scaled by its std, so all features contribute equally.
            z_scores   = np.abs((feat - info["centroid"]) / info["std"])
            mean_z     = float(z_scores.mean())
            # Map the z-score to a 0-1 confidence: 0.0 at the threshold, 1.0 at perfect.
            confidence = max(0.0, 1.0 - mean_z / DISTANCE_THRESHOLD)

            if mean_z <= DISTANCE_THRESHOLD:
                return name, confidence   # close enough — accept
            else:
                return None, confidence   # too far — reject

        elif self._mode == "svm":
            try:
                X     = self._scaler.transform([features])
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
        Binary check: does this feature vector belong to the named person?
        Returns (match: bool, confidence: float).
        Used during enrollment self-test and login verification.
        """
        pred_name, conf = self.predict(features)
        if pred_name is None:
            return False, conf
        if self._mode == "distance":
            # In distance mode any passing result is considered this person
            # (there's only one enrolled person to compare against).
            return True, conf
        else:
            return (pred_name == name), conf


# =============================================================================
# Compatibility stubs (kept so older code doesn't break)
# =============================================================================

def extract_movement_features(trajectory):
    """Legacy stub — movement features are no longer used. Returns None."""
    return None


def combine_features(geometry, movement):
    """Legacy stub — the system is geometry-only now. Returns geometry as-is."""
    return geometry


# Legacy aliases so imports in other files still resolve.
MIN_SAMPLES_FOR_TRAINING = MIN_SAMPLES_DISTANCE
VERIFY_WINDOW            = 20
VERIFY_THRESHOLD         = 0.80

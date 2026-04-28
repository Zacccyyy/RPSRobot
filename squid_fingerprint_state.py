"""
squid_fingerprint_state.py
==========================
Fingerprint enrollment and verification built on Squid Game.

Data collection strategy:
  PRIMARY  — Geometry features sampled from dwell frames (finger held on dot).
             Sampling happens every frame inside the dot once dwell begins,
             giving ~30 clean landmark snapshots per capture at 30fps / 1s dwell.
  SECONDARY — Movement features from the approach trajectory only (transit
              phase before the finger enters the dot radius). Collection
              stops the moment the finger enters the dot.

This means:
  - The bulk of fingerprint data comes from the still, controlled dwell phase
  - Movement data is a bonus signal from the transit, not the primary evidence
  - Increasing dwell to 1s gives ~30 geometry frames per dot → rich signal

States:
  COLLECTING  — playing normally, accumulating samples
  TRAINING    — training the SVM (brief)
  VERIFYING   — continuing to play, each capture tests prediction accuracy
  VERIFIED    — 80%+ accuracy confirmed, fingerprint saved
  FAILED      — accuracy never reached threshold after extended play
"""

import time
from collections import deque

from squid_game_state import (
    SquidGameController, _landmark_pos,
    DOT_RADIUS_NORM, CAPTURE_DWELL_SECS,
)
from gesture_fingerprint import (
    FingerprintStore, FingerprintClassifier,
    extract_geometry_features,
    extract_movement_features,
    combine_features,
    MIN_SAMPLES_FOR_TRAINING,
    VERIFY_WINDOW, VERIFY_THRESHOLD,
    MIN_DWELL_FRAMES,
)


class SquidFingerprintController(SquidGameController):
    """
    Squid Game with silent dual-channel fingerprint enrollment.

    Dwell phase:  geometry landmarks sampled every frame while inside dot
    Transit phase: (x,y) positions collected while approaching dot
    On capture:   combine both → one 32-element feature vector → store
    """

    def __init__(self, player_name, store=None):
        super().__init__()
        self.player_name      = player_name
        self._store           = store or FingerprintStore()
        self._clf             = FingerprintClassifier()

        # Transit trajectory: (x,y) tip positions before dot entry
        self._transit_traj    = []
        self._in_dot          = False   # True once finger is inside dot radius

        # Dwell landmark collection: raw landmark objects per frame inside dot
        self._dwell_landmarks = []

        # Session samples (combined feature vectors)
        self._session_samples = []
        existing = self._store.load_samples(player_name)
        if existing and existing.get("samples"):
            # Only reuse v2 (32-element) vectors
            self._session_samples = [
                s for s in existing["samples"] if len(s) == 32
            ]

        # Phase tracking
        self.fp_phase          = "COLLECTING"
        self.verify_results    = []
        self.verify_total      = 0
        self.verify_accuracy   = 0.0
        self.enroll_start      = time.strftime("%Y-%m-%dT%H:%M:%S")

    def reset(self):
        super().reset()
        self._transit_traj    = []
        self._dwell_landmarks = []
        self._in_dot          = False

    def _on_capture(self, dwell_lms, transit_traj):
        """
        Called when a dot is captured. Extract both feature channels,
        combine and store.
        """
        geo = extract_geometry_features(dwell_lms)
        mov = extract_movement_features(transit_traj) if len(transit_traj) >= 5 else None
        vec = combine_features(geo, mov)

        if vec is None:
            return

        if self.fp_phase == "COLLECTING":
            self._session_samples.append(vec)
            self._store.save_samples(
                self.player_name,
                self._session_samples,
                verified=False,
                enrolled_at=self.enroll_start,
            )
            n = len(self._session_samples)
            if geo:
                print(f"[Fingerprint] Sample {n}: geometry OK "
                      f"({len(dwell_lms)} dwell frames), "
                      f"movement {'OK' if mov else 'skipped'}")
            if n >= MIN_SAMPLES_FOR_TRAINING:
                self._train()

        elif self.fp_phase == "VERIFYING":
            predicted, conf = self._clf.predict(vec)
            correct = (predicted == self.player_name and conf >= 0.55)
            self.verify_results.append(correct)
            self.verify_total += 1
            window = self.verify_results[-VERIFY_WINDOW:]
            self.verify_accuracy = sum(window) / len(window)
            print(f"[Fingerprint] Verify {self.verify_total}: "
                  f"pred={predicted} conf={conf:.0%} "
                  f"acc={self.verify_accuracy:.0%}")
            if (len(window) >= VERIFY_WINDOW
                    and self.verify_accuracy >= VERIFY_THRESHOLD):
                self.fp_phase = "VERIFIED"
                self._store.mark_verified(self.player_name)
                print(f"[Fingerprint] {self.player_name} VERIFIED "
                      f"({self.verify_accuracy:.0%})")
            elif self.verify_total >= VERIFY_WINDOW * 3 \
                    and self.verify_accuracy < 0.50:
                self.fp_phase = "FAILED"

    def _train(self):
        self.fp_phase = "TRAINING"
        ok = self._clf.train(self._store, include_unverified_for=self.player_name)
        self.fp_phase = "VERIFYING" if ok else "COLLECTING"

    def update(self, hand_state, now=None):
        if now is None:
            now = time.monotonic()

        tip = _landmark_pos(hand_state)
        lm_obj = hand_state.get("_landmarks") if hand_state else None

        # ── Track transit vs dwell ─────────────────────────────────────
        if tip is not None:
            dist = self._dist_to_dot(tip[0], tip[1])
            inside = dist <= DOT_RADIUS_NORM

            if not self._in_dot and not inside:
                # Still in transit — collect (x,y) positions
                self._transit_traj.append(tip)

            elif not self._in_dot and inside:
                # Just entered the dot — stop transit collection
                self._in_dot = True
                self._dwell_landmarks = []
                # Keep transit trajectory for movement features

            if inside and lm_obj is not None:
                # Inside dot — collect landmark objects for geometry
                self._dwell_landmarks.append(lm_obj.landmark)

            if not inside and self._in_dot:
                # Left the dot without capture (e.g. drifted out)
                self._in_dot = False
                self._transit_traj.clear()
                self._dwell_landmarks.clear()
        else:
            # Hand not visible — reset both buffers
            self._transit_traj.clear()
            self._dwell_landmarks.clear()
            self._in_dot = False

        prev_dots = self.dots_collected
        result    = super().update(hand_state=hand_state, now=now)

        # ── Dot just captured ──────────────────────────────────────────
        if self.dots_collected > prev_dots:
            self._on_capture(
                dwell_lms   = list(self._dwell_landmarks),
                transit_traj= list(self._transit_traj),
            )
            # Reset for next dot
            self._transit_traj.clear()
            self._dwell_landmarks.clear()
            self._in_dot = False

        # Inject fingerprint state into output dict
        result["fp_phase"]           = self.fp_phase
        result["fp_samples"]         = len(self._session_samples)
        result["fp_target"]          = MIN_SAMPLES_FOR_TRAINING
        result["fp_verify_total"]    = self.verify_total
        result["fp_verify_accuracy"] = self.verify_accuracy
        result["fp_verify_target"]   = VERIFY_WINDOW
        result["fp_player_name"]     = self.player_name
        result["fp_classes"]         = self._clf.classes
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Login via Fingerprint: short session, predict identity
# ─────────────────────────────────────────────────────────────────────────────

class SquidFingerprintLoginController(SquidGameController):
    """
    Short Squid Game session for fingerprint login.
    Collects dual-channel features from each capture, votes on identity.
    Commits after MIN_LOGIN_DOTS captures with >= VERIFY_THRESHOLD majority.
    """

    MIN_LOGIN_DOTS = 8

    def __init__(self, store=None):
        super().__init__()
        self._store        = store or FingerprintStore()
        self._clf          = FingerprintClassifier()
        self._transit_traj = []
        self._dwell_landmarks = []
        self._in_dot       = False
        self._predictions  = []   # list of (name, confidence)
        self.login_result  = None
        self.login_confidence = 0.0
        self._clf.train(self._store)   # train on all verified fingerprints

    def reset(self):
        super().reset()
        self._transit_traj    = []
        self._dwell_landmarks = []
        self._in_dot          = False
        self._predictions     = []
        self.login_result     = None
        self.login_confidence = 0.0

    def update(self, hand_state, now=None):
        if now is None:
            now = __import__("time").monotonic()

        tip    = _landmark_pos(hand_state)
        lm_obj = hand_state.get("_landmarks") if hand_state else None

        if tip is not None:
            dist   = self._dist_to_dot(tip[0], tip[1])
            inside = dist <= DOT_RADIUS_NORM
            if not self._in_dot and not inside:
                self._transit_traj.append(tip)
            elif not self._in_dot and inside:
                self._in_dot = True
                self._dwell_landmarks = []
            if inside and lm_obj is not None:
                self._dwell_landmarks.append(lm_obj.landmark)
            if not inside and self._in_dot:
                self._in_dot = False
                self._transit_traj.clear()
                self._dwell_landmarks.clear()
        else:
            self._transit_traj.clear()
            self._dwell_landmarks.clear()
            self._in_dot = False

        prev_dots = self.dots_collected
        result    = super().update(hand_state=hand_state, now=now)

        if self.dots_collected > prev_dots:
            geo = extract_geometry_features(list(self._dwell_landmarks))
            mov = extract_movement_features(list(self._transit_traj)) \
                  if len(self._transit_traj) >= 5 else None
            vec = combine_features(geo, mov)
            if vec is not None:
                name, conf = self._clf.predict(vec)
                if name:
                    self._predictions.append((name, conf))
            self._transit_traj.clear()
            self._dwell_landmarks.clear()
            self._in_dot = False

        # Commit after enough captures
        if len(self._predictions) >= self.MIN_LOGIN_DOTS \
                and self.login_result is None:
            from collections import Counter
            votes   = Counter(n for n, _ in self._predictions)
            top, n  = votes.most_common(1)[0]
            ratio   = n / len(self._predictions)
            top_confs = [c for nm, c in self._predictions if nm == top]
            avg_conf  = sum(top_confs) / max(len(top_confs), 1)
            if ratio >= VERIFY_THRESHOLD:
                self.login_result     = top
                self.login_confidence = avg_conf

        result["fp_phase"]         = "LOGIN"
        result["fp_predictions"]   = len(self._predictions)
        result["fp_target"]        = self.MIN_LOGIN_DOTS
        result["login_result"]     = self.login_result
        result["login_confidence"] = self.login_confidence
        return result

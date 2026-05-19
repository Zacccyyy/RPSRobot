"""
squid_fingerprint_state.py
==========================
Fingerprint enrollment and verification built on top of Squid Game.

The game collects biometric data silently while the player plays normally.
Two channels of data are captured per dot:

  PRIMARY (geometry)  -- Hand landmark coordinates sampled every frame while
                         the finger is held inside the dot (dwell phase).
                         At ~30fps with a 1s dwell this gives ~30 clean
                         snapshots of the hand's shape.

  SECONDARY (movement) -- (x, y) tip positions collected during the approach
                          trajectory (transit phase), i.e. before the finger
                          enters the dot radius.  Collection stops the moment
                          the finger enters the dot.

Both channels are combined into a single 32-element feature vector per dot
capture and stored/used for ML classification.

The controller subclasses SquidGameController and layers the enrollment
logic on top — the player just plays the game and data is collected silently.

States (fp_phase, separate from the game state machine):
  COLLECTING  -- accumulating feature vectors until MIN_SAMPLES_FOR_TRAINING
  TRAINING    -- background thread is training the SVM classifier
  VERIFYING   -- enough data collected; testing prediction accuracy on new dots
  VERIFIED    -- 80%+ accuracy confirmed, fingerprint saved
  FAILED      -- accuracy never reached threshold after extended play
"""

import time
import threading
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

# Minimum confidence for a single-dot prediction to count as "correct".
# Intentionally lower than VERIFY_THRESHOLD (which is an accuracy ratio
# across many dots) so individual shaky captures don't block verification.
_MIN_PRED_CONFIDENCE = 0.55


class SquidFingerprintController(SquidGameController):
    """
    Squid Game with silent dual-channel fingerprint enrollment.

    Dwell phase:   geometry landmarks sampled every frame while inside dot
    Transit phase: (x,y) positions collected while approaching the dot
    On capture:    combine both channels -> one 32-element feature vector -> store
    """

    def __init__(self, player_name, store=None):
        super().__init__()
        self.player_name      = player_name
        self._store           = store or FingerprintStore()
        self._clf             = FingerprintClassifier()

        # Transit trajectory: raw (x,y) tip positions collected before dot entry
        self._transit_traj    = []
        self._in_dot          = False   # True once finger has entered the dot radius

        # Dwell landmark collection: raw MediaPipe landmark lists, one per frame inside dot
        self._dwell_landmarks = []

        # Combined feature vectors collected so far in this session
        self._session_samples = []

        # Load any previously collected (v2, 32-element) samples for this player
        existing = self._store.load_samples(player_name)
        if existing and existing.get("samples"):
            # Only reuse v2 (32-element) vectors; discard old format data
            self._session_samples = [
                s for s in existing["samples"] if len(s) == 32
            ]

        # Fingerprint phase (separate from the game state machine)
        self.fp_phase          = "COLLECTING"
        self.verify_results    = []    # list of True/False per verification dot
        self.verify_total      = 0     # total dots used for verification
        self.verify_accuracy   = 0.0   # rolling accuracy in the last VERIFY_WINDOW dots
        self.enroll_start      = time.strftime("%Y-%m-%dT%H:%M:%S")

    def reset(self):
        """Reset game state and clear the per-dot collection buffers."""
        super().reset()
        self._transit_traj    = []
        self._dwell_landmarks = []
        self._in_dot          = False

    def _on_capture(self, dwell_lms, transit_traj):
        """
        Called each time the player captures a dot.

        Extracts geometry features from the dwell landmarks and movement
        features from the approach trajectory, combines them into one feature
        vector, and either stores it (COLLECTING) or tests it (VERIFYING).

        Args:
            dwell_lms:    list of landmark lists (one per frame) from the dwell phase
            transit_traj: list of (x, y) positions from the approach phase
        """
        geo = extract_geometry_features(dwell_lms)
        # Only extract movement features if we have at least 5 transit positions
        mov = extract_movement_features(transit_traj) if len(transit_traj) >= 5 else None
        vec = combine_features(geo, mov)

        # If neither channel produced usable data, skip this capture
        if vec is None:
            return

        if self.fp_phase == "COLLECTING":
            # Store the sample and persist to disk
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

            # Start training once we have enough samples
            if n >= MIN_SAMPLES_FOR_TRAINING:
                self._train()

        elif self.fp_phase == "VERIFYING":
            # Use the classifier to predict whose fingerprint this is
            predicted, conf = self._clf.predict(vec)
            correct = (predicted == self.player_name and conf >= _MIN_PRED_CONFIDENCE)
            self.verify_results.append(correct)
            self.verify_total += 1

            # Rolling accuracy over the last VERIFY_WINDOW dots
            window = self.verify_results[-VERIFY_WINDOW:]
            self.verify_accuracy = sum(window) / len(window)

            print(f"[Fingerprint] Verify {self.verify_total}: "
                  f"pred={predicted} conf={conf:.0%} "
                  f"acc={self.verify_accuracy:.0%}")

            # Mark as VERIFIED if accuracy exceeds threshold over a full window
            if (len(window) >= VERIFY_WINDOW
                    and self.verify_accuracy >= VERIFY_THRESHOLD):
                self.fp_phase = "VERIFIED"
                self._store.mark_verified(self.player_name)
                print(f"[Fingerprint] {self.player_name} VERIFIED "
                      f"({self.verify_accuracy:.0%})")

            # Give up if we've tried 3 full windows and accuracy is still below 50%
            elif self.verify_total >= VERIFY_WINDOW * 3 \
                    and self.verify_accuracy < 0.50:
                self.fp_phase = "FAILED"

    def _train(self):
        """
        Kick off background training if not already in progress.
        Uses a daemon thread so it doesn't block the game loop.
        """
        if getattr(self, "_training_in_progress", False):
            return
        self._training_in_progress = True
        self.fp_phase = "TRAINING"
        threading.Thread(target=self._do_train, daemon=True).start()

    def _do_train(self):
        """Background thread: train the SVM on all stored samples."""
        try:
            ok = self._clf.train(self._store, include_unverified_for=self.player_name)
            # Move to VERIFYING on success, back to COLLECTING if training failed
            self.fp_phase = "VERIFYING" if ok else "COLLECTING"
        finally:
            self._training_in_progress = False

    def update(self, hand_state, now=None):
        """
        Frame update.  Layers fingerprint data collection on top of the base
        Squid Game update.

        The collection pipeline works like this each frame:
          1. Determine if tip is inside or outside the dot
          2. OUTSIDE: accumulate transit trajectory (approach positions)
          3. INSIDE:  accumulate dwell landmarks (shape snapshots)
          4. After base update, check if a new dot was captured
          5. If captured, call _on_capture() then reset the buffers
        """
        if now is None:
            now = time.monotonic()

        tip    = _landmark_pos(hand_state)
        lm_obj = hand_state.get("_landmarks") if hand_state else None

        # ── Track transit vs dwell phases ──────────────────────────────────
        if tip is not None:
            dist   = self._dist_to_dot(tip[0], tip[1])
            inside = dist <= DOT_RADIUS_NORM

            if not self._in_dot and not inside:
                # Still approaching — record position for movement features
                self._transit_traj.append(tip)

            elif not self._in_dot and inside:
                # Just entered the dot — switch from transit to dwell collection
                self._in_dot          = True
                self._dwell_landmarks = []
                # Keep _transit_traj intact; it'll be consumed on capture

            if inside and lm_obj is not None:
                # Inside dot — record landmark snapshot for geometry features
                self._dwell_landmarks.append(lm_obj.landmark)

            if not inside and self._in_dot:
                # Left the dot without capturing (drifted out) — discard buffers
                self._in_dot = False
                self._transit_traj.clear()
                self._dwell_landmarks.clear()
        else:
            # Hand not visible — clear everything so stale data doesn't carry over
            self._transit_traj.clear()
            self._dwell_landmarks.clear()
            self._in_dot = False

        # Run the base game update; capture happens inside here
        prev_dots = self.dots_collected
        result    = super().update(hand_state=hand_state, now=now)

        # ── A new dot was captured this frame ──────────────────────────────
        if self.dots_collected > prev_dots:
            # Pass copies so buffers can be cleared immediately after
            self._on_capture(
                dwell_lms    = list(self._dwell_landmarks),
                transit_traj = list(self._transit_traj),
            )
            # Reset for the next dot
            self._transit_traj.clear()
            self._dwell_landmarks.clear()
            self._in_dot = False

        # Inject fingerprint status into the output dict the UI reads
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

    Collects dual-channel features from each dot capture and votes on identity.
    After MIN_LOGIN_DOTS captures, the most-voted name with >= VERIFY_THRESHOLD
    vote share is accepted as the logged-in player.
    """

    # How many dot captures we need before committing to a login decision
    MIN_LOGIN_DOTS = 8

    def __init__(self, store=None):
        super().__init__()
        self._store           = store or FingerprintStore()
        self._clf             = FingerprintClassifier()
        self._transit_traj    = []
        self._dwell_landmarks = []
        self._in_dot          = False
        self._predictions     = []    # accumulated (name, confidence) pairs
        self.login_result     = None  # set to the predicted player name on commit
        self.login_confidence = 0.0
        # Train immediately on all verified fingerprints so we're ready to predict
        self._clf.train(self._store)

    def reset(self):
        """Reset game state and clear all login buffers."""
        super().reset()
        self._transit_traj    = []
        self._dwell_landmarks = []
        self._in_dot          = False
        self._predictions     = []
        self.login_result     = None
        self.login_confidence = 0.0

    def update(self, hand_state, now=None):
        """
        Frame update.  Same dual-channel collection as SquidFingerprintController
        but uses predictions for identity voting rather than accuracy testing.

        After MIN_LOGIN_DOTS captures the top-voted name is committed as the
        login result if its vote share meets VERIFY_THRESHOLD.
        """
        if now is None:
            now = __import__("time").monotonic()

        tip    = _landmark_pos(hand_state)
        lm_obj = hand_state.get("_landmarks") if hand_state else None

        # ── Same transit/dwell tracking as the enrollment controller ──────
        if tip is not None:
            dist   = self._dist_to_dot(tip[0], tip[1])
            inside = dist <= DOT_RADIUS_NORM
            if not self._in_dot and not inside:
                self._transit_traj.append(tip)
            elif not self._in_dot and inside:
                # Entering the dot — stop transit, start dwell
                self._in_dot          = True
                self._dwell_landmarks = []
            if inside and lm_obj is not None:
                self._dwell_landmarks.append(lm_obj.landmark)
            if not inside and self._in_dot:
                # Left dot without capture — reset buffers
                self._in_dot = False
                self._transit_traj.clear()
                self._dwell_landmarks.clear()
        else:
            # Hand lost — clear buffers
            self._transit_traj.clear()
            self._dwell_landmarks.clear()
            self._in_dot = False

        # Run base game update
        prev_dots = self.dots_collected
        result    = super().update(hand_state=hand_state, now=now)

        # ── Dot captured: extract features and record a prediction ────────
        if self.dots_collected > prev_dots:
            geo = extract_geometry_features(list(self._dwell_landmarks))
            mov = extract_movement_features(list(self._transit_traj)) \
                  if len(self._transit_traj) >= 5 else None
            vec = combine_features(geo, mov)
            if vec is not None:
                name, conf = self._clf.predict(vec)
                if name:
                    self._predictions.append((name, conf))
            # Reset buffers for next dot
            self._transit_traj.clear()
            self._dwell_landmarks.clear()
            self._in_dot = False

        # ── Commit login decision after enough captures ───────────────────
        if len(self._predictions) >= self.MIN_LOGIN_DOTS \
                and self.login_result is None:
            from collections import Counter
            # Count votes for each name
            votes = Counter(n for n, _ in self._predictions)
            top, n = votes.most_common(1)[0]
            ratio  = n / len(self._predictions)
            # Average confidence of the top-voted name's captures
            top_confs = [c for nm, c in self._predictions if nm == top]
            avg_conf  = sum(top_confs) / max(len(top_confs), 1)
            # Only commit if the vote share is convincing enough
            if ratio >= VERIFY_THRESHOLD:
                self.login_result     = top
                self.login_confidence = avg_conf

        # Inject login status into the output dict
        result["fp_phase"]         = "LOGIN"
        result["fp_predictions"]   = len(self._predictions)
        result["fp_target"]        = self.MIN_LOGIN_DOTS
        result["login_result"]     = self.login_result
        result["login_confidence"] = self.login_confidence
        return result

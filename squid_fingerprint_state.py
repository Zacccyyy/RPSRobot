# squid_fingerprint_state.py
# ---------------------------
# Fingerprint enrollment and login built on top of the Squid Game mini-game.
#
# The player plays the dot-touching game normally; the biometric system
# collects hand shape data silently in the background. No extra UI is needed.
#
# Two data channels are captured per dot touch:
#
#   DWELL (geometry)   -- MediaPipe landmark snapshots taken every frame
#                         while the fingertip is held inside the dot.
#                         At ~30fps this gives ~30 clean shape samples per dot.
#
#   TRANSIT (movement) -- (x, y) tip positions collected during the approach
#                         to the dot (before the fingertip enters the radius).
#                         Stops the moment the finger enters the dot.
#
# Both channels are combined into a single feature vector per dot and handed
# to gesture_fingerprint.py for storage or classification.
#
# Fingerprint phases (stored in fp_phase, separate from the game's state machine):
#   COLLECTING  -- gathering feature vectors until we hit MIN_SAMPLES_FOR_TRAINING
#   TRAINING    -- background thread is fitting the classifier
#   VERIFYING   -- testing accuracy by predicting on new dot captures
#   VERIFIED    -- accuracy confirmed at 80%+, profile saved as verified
#   FAILED      -- accuracy never reached threshold after extended play

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

# Minimum classifier confidence for a single prediction to count as "correct"
# during the verification phase. Set lower than VERIFY_THRESHOLD so a slightly
# shaky capture doesn't unfairly block the player from being verified.
_MIN_PRED_CONFIDENCE = 0.55


# =============================================================================
# Enrollment controller
# =============================================================================

class SquidFingerprintController(SquidGameController):
    """
    Squid Game with silent dual-channel fingerprint enrollment layered on top.

    The player just plays the game. This controller intercepts each dot capture,
    extracts geometry (dwell) and movement (transit) features, and either stores
    them (COLLECTING phase) or uses them to test accuracy (VERIFYING phase).
    """

    def __init__(self, player_name, store=None):
        super().__init__()
        self.player_name = player_name
        self._store      = store or FingerprintStore()
        self._clf        = FingerprintClassifier()

        # Approach trajectory: (x, y) tip positions before the finger enters the dot.
        self._transit_traj    = []
        self._in_dot          = False   # True once the fingertip has entered the dot

        # Per-dot landmark buffer: one MediaPipe landmark list per frame inside the dot.
        self._dwell_landmarks = []

        # All feature vectors collected so far this session.
        self._session_samples = []

        # Load any previously saved samples for this player so enrollment
        # can pick up where it left off across multiple sessions.
        existing = self._store.load_samples(player_name)
        if existing and existing.get("samples"):
            # Only keep 32-element vectors — older format data is incompatible.
            self._session_samples = [
                s for s in existing["samples"] if len(s) == 32
            ]

        # Fingerprint phase (independent of the Squid Game's own state machine).
        self.fp_phase        = "COLLECTING"
        self.verify_results  = []    # per-dot True/False accuracy log
        self.verify_total    = 0     # total dots used in verification so far
        self.verify_accuracy = 0.0   # rolling accuracy over the last VERIFY_WINDOW dots
        self.enroll_start    = time.strftime("%Y-%m-%dT%H:%M:%S")

    def reset(self):
        """Reset the game state and clear the per-dot collection buffers."""
        super().reset()
        self._transit_traj    = []
        self._dwell_landmarks = []
        self._in_dot          = False

    def _on_capture(self, dwell_lms, transit_traj):
        """
        Called once per dot capture with the buffered dwell landmarks and transit
        trajectory. Extracts features from both channels, combines them, and
        either stores the vector (COLLECTING) or tests it for accuracy (VERIFYING).

        Args:
            dwell_lms:    list of landmark snapshots from the dwell phase
            transit_traj: list of (x, y) positions from the approach phase
        """
        # Extract the geometry (hand shape) features from the dwell frames.
        geo = extract_geometry_features(dwell_lms)

        # Only attempt movement features if we have enough transit points.
        mov = extract_movement_features(transit_traj) if len(transit_traj) >= 5 else None

        # Combine the two channels into one vector (movement is a no-op stub for now).
        vec = combine_features(geo, mov)

        # Nothing useful was extracted — skip this dot.
        if vec is None:
            return

        if self.fp_phase == "COLLECTING":
            # Add the new sample and immediately persist to disk.
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

            # Once we have enough samples, kick off training in the background.
            if n >= MIN_SAMPLES_FOR_TRAINING:
                self._train()

        elif self.fp_phase == "VERIFYING":
            # Ask the trained classifier whose fingerprint this looks like.
            predicted, conf = self._clf.predict(vec)
            correct = (predicted == self.player_name and conf >= _MIN_PRED_CONFIDENCE)
            self.verify_results.append(correct)
            self.verify_total += 1

            # Rolling accuracy: only look at the most recent VERIFY_WINDOW dots.
            window = self.verify_results[-VERIFY_WINDOW:]
            self.verify_accuracy = sum(window) / len(window)

            print(f"[Fingerprint] Verify {self.verify_total}: "
                  f"pred={predicted} conf={conf:.0%} "
                  f"acc={self.verify_accuracy:.0%}")

            # If accuracy is high enough across a full window, mark as verified.
            if len(window) >= VERIFY_WINDOW and self.verify_accuracy >= VERIFY_THRESHOLD:
                self.fp_phase = "VERIFIED"
                self._store.mark_verified(self.player_name)
                print(f"[Fingerprint] {self.player_name} VERIFIED "
                      f"({self.verify_accuracy:.0%})")

            # Give up if accuracy is still below 50% after 3 full windows.
            elif (self.verify_total >= VERIFY_WINDOW * 3
                  and self.verify_accuracy < 0.50):
                self.fp_phase = "FAILED"

    def _train(self):
        """
        Start the classifier training on a background thread.
        Using a daemon thread means it won't block the game loop or prevent exit.
        Guards against starting a second training run if one is already in progress.
        """
        if getattr(self, "_training_in_progress", False):
            return
        self._training_in_progress = True
        self.fp_phase = "TRAINING"
        threading.Thread(target=self._do_train, daemon=True).start()

    def _do_train(self):
        """
        Background thread body: fits the classifier on all stored samples.
        Transitions to VERIFYING on success, or back to COLLECTING if it fails.
        """
        try:
            ok = self._clf.train(self._store, include_unverified_for=self.player_name)
            self.fp_phase = "VERIFYING" if ok else "COLLECTING"
        finally:
            self._training_in_progress = False

    def update(self, hand_state, now=None):
        """
        Per-frame update. Runs the fingerprint collection pipeline on top of
        the base Squid Game update.

        Each frame:
          1. Check whether the fingertip is inside or outside the dot.
          2. Outside: record (x, y) position into the transit trajectory.
          3. Inside:  record landmark snapshot into the dwell buffer.
          4. Run the base game update (which triggers a capture when dwell time is met).
          5. If a dot was captured, call _on_capture() then clear the buffers.
        """
        if now is None:
            now = time.monotonic()

        tip    = _landmark_pos(hand_state)
        lm_obj = hand_state.get("_landmarks") if hand_state else None

        if tip is not None:
            dist   = self._dist_to_dot(tip[0], tip[1])
            inside = dist <= DOT_RADIUS_NORM

            if not self._in_dot and not inside:
                # Approaching the dot — accumulate transit positions.
                self._transit_traj.append(tip)

            elif not self._in_dot and inside:
                # Just entered the dot — switch from transit to dwell mode.
                self._in_dot          = True
                self._dwell_landmarks = []
                # Transit trajectory is kept; it's consumed when the dot is captured.

            if inside and lm_obj is not None:
                # Inside the dot — record a landmark snapshot this frame.
                self._dwell_landmarks.append(lm_obj.landmark)

            if not inside and self._in_dot:
                # Drifted out of the dot without capturing — discard stale data.
                self._in_dot = False
                self._transit_traj.clear()
                self._dwell_landmarks.clear()

        else:
            # Hand is no longer visible — clear everything so old data doesn't bleed
            # into the next capture when the hand reappears.
            self._transit_traj.clear()
            self._dwell_landmarks.clear()
            self._in_dot = False

        # Let the base game run its own logic (dwell timer, capture detection, etc.).
        prev_dots = self.dots_collected
        result    = super().update(hand_state=hand_state, now=now)

        # If a new dot was captured this frame, extract features and reset buffers.
        if self.dots_collected > prev_dots:
            # Pass copies so we can safely clear the buffers right after.
            self._on_capture(
                dwell_lms    = list(self._dwell_landmarks),
                transit_traj = list(self._transit_traj),
            )
            self._transit_traj.clear()
            self._dwell_landmarks.clear()
            self._in_dot = False

        # Inject fingerprint status fields into the result dict so the UI can display them.
        result["fp_phase"]           = self.fp_phase
        result["fp_samples"]         = len(self._session_samples)
        result["fp_target"]          = MIN_SAMPLES_FOR_TRAINING
        result["fp_verify_total"]    = self.verify_total
        result["fp_verify_accuracy"] = self.verify_accuracy
        result["fp_verify_target"]   = VERIFY_WINDOW
        result["fp_player_name"]     = self.player_name
        result["fp_classes"]         = self._clf.classes
        return result


# =============================================================================
# Login controller
# =============================================================================

class SquidFingerprintLoginController(SquidGameController):
    """
    Short Squid Game session used for biometric login.

    Collects dual-channel features from each dot capture and builds up a tally
    of predicted identities. After MIN_LOGIN_DOTS captures, the most-voted name
    is accepted as the logged-in player if its vote share meets VERIFY_THRESHOLD.
    """

    # How many dot captures we need before we're confident enough to commit.
    MIN_LOGIN_DOTS = 8

    def __init__(self, store=None):
        super().__init__()
        self._store           = store or FingerprintStore()
        self._clf             = FingerprintClassifier()
        self._transit_traj    = []
        self._dwell_landmarks = []
        self._in_dot          = False
        self._predictions     = []    # list of (name, confidence) pairs from each dot
        self.login_result     = None  # set to the winning player name once decided
        self.login_confidence = 0.0

        # Train immediately on all verified profiles so the classifier is ready.
        self._clf.train(self._store)

    def reset(self):
        """Reset the game and clear all login state."""
        super().reset()
        self._transit_traj    = []
        self._dwell_landmarks = []
        self._in_dot          = False
        self._predictions     = []
        self.login_result     = None
        self.login_confidence = 0.0

    def update(self, hand_state, now=None):
        """
        Per-frame update for the login session.

        Uses the same dual-channel collection logic as the enrollment controller.
        On each dot capture, the classifier makes a prediction and we record
        the (name, confidence) pair. Once we have MIN_LOGIN_DOTS predictions
        we tally the votes and commit to a login result.
        """
        if now is None:
            now = __import__("time").monotonic()

        tip    = _landmark_pos(hand_state)
        lm_obj = hand_state.get("_landmarks") if hand_state else None

        # Same transit/dwell tracking logic as the enrollment controller.
        if tip is not None:
            dist   = self._dist_to_dot(tip[0], tip[1])
            inside = dist <= DOT_RADIUS_NORM

            if not self._in_dot and not inside:
                self._transit_traj.append(tip)
            elif not self._in_dot and inside:
                # Entering the dot — stop transit collection, start dwell.
                self._in_dot          = True
                self._dwell_landmarks = []

            if inside and lm_obj is not None:
                self._dwell_landmarks.append(lm_obj.landmark)

            if not inside and self._in_dot:
                # Left the dot without capturing — discard stale buffers.
                self._in_dot = False
                self._transit_traj.clear()
                self._dwell_landmarks.clear()

        else:
            # Hand disappeared — clear everything.
            self._transit_traj.clear()
            self._dwell_landmarks.clear()
            self._in_dot = False

        # Run the base game update.
        prev_dots = self.dots_collected
        result    = super().update(hand_state=hand_state, now=now)

        # On each new dot capture, extract features and record a prediction.
        if self.dots_collected > prev_dots:
            geo = extract_geometry_features(list(self._dwell_landmarks))
            mov = (extract_movement_features(list(self._transit_traj))
                   if len(self._transit_traj) >= 5 else None)
            vec = combine_features(geo, mov)

            if vec is not None:
                name, conf = self._clf.predict(vec)
                if name:
                    self._predictions.append((name, conf))

            # Clear buffers for the next dot.
            self._transit_traj.clear()
            self._dwell_landmarks.clear()
            self._in_dot = False

        # Once we have enough predictions, decide who's logging in.
        if len(self._predictions) >= self.MIN_LOGIN_DOTS and self.login_result is None:
            from collections import Counter

            # Count how many times each name was predicted.
            votes    = Counter(name for name, _ in self._predictions)
            top, n   = votes.most_common(1)[0]
            ratio    = n / len(self._predictions)

            # Average confidence across all captures that voted for the top name.
            top_confs = [conf for name, conf in self._predictions if name == top]
            avg_conf  = sum(top_confs) / max(len(top_confs), 1)

            # Only accept the login if the vote share meets the threshold.
            if ratio >= VERIFY_THRESHOLD:
                self.login_result     = top
                self.login_confidence = avg_conf

        # Add login status to the result dict for the UI to read.
        result["fp_phase"]         = "LOGIN"
        result["fp_predictions"]   = len(self._predictions)
        result["fp_target"]        = self.MIN_LOGIN_DOTS
        result["login_result"]     = self.login_result
        result["login_confidence"] = self.login_confidence
        return result

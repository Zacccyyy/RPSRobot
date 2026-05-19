"""
ml_model.py
===========
Machine-learning model wrapper and game-compatible AI for predicting player
moves in the RPS Challenge mode.

WHERE IT FITS:
    ml_feature_extractor.py  -->  feature vectors
            |
         ml_model.py  (this file — trains, infers, saves, loads)
            |
    challenge_mode_state.py  (uses MLPredictionAI as the robot brain)

TWO CLASSES:
    RPSModel        — thin scikit-learn wrapper: train / predict / save / load
    MLPredictionAI  — drop-in replacement for ChallengeAI / FairPlayAI
                      uses RPSModel to predict and then counter the player

USAGE (training):
    model = RPSModel()
    model.train(X, y)
    model.save("rps_model.pkl")
    print(model.evaluate(X_test, y_test))

USAGE (in-game):
    ai = MLPredictionAI(model_path="rps_model.pkl")
    move = ai.choose_robot_move(history, streak, round_number)
"""

import os
import random
import pickle

import numpy as np

from ml_feature_extractor import (
    extract_features,
    get_feature_names,
    GESTURE_INDEX,
)

# All valid gesture strings (used for random fallback and validation)
VALID_GESTURES = ("Rock", "Paper", "Scissors")

# Reverse mapping: integer label -> gesture string
INDEX_TO_GESTURE = {v: k for k, v in GESTURE_INDEX.items()}

# What move beats each gesture (used to pick the counter-move)
COUNTER_MOVE = {
    "Rock":     "Paper",
    "Paper":    "Scissors",
    "Scissors": "Rock",
}


# ---------------------------------------------------------------------------
# RPSModel — scikit-learn wrapper
# ---------------------------------------------------------------------------

class RPSModel:
    """
    Thin wrapper around a scikit-learn classifier.

    Keeping sklearn behind this class means we can swap the underlying
    algorithm (logistic regression, random forest, etc.) without touching
    any game code.

    Default model: LogisticRegression — fast, explainable, and hard to overfit
    on small datasets.  RandomForest is available as an alternative.
    """

    def __init__(self, model=None, lookback=3):
        # lookback is stored so save/load can round-trip the feature dimension
        self.lookback   = lookback
        self.model      = model      # the underlying sklearn estimator
        self.is_trained = False
        self.classes_   = None       # list of integer class labels seen during training

    def train(self, X, y, model_type="logistic"):
        """
        Fit the model on feature matrix X and label vector y.

        Parameters:
            X          — list or array of feature vectors
            y          — list or array of integer labels (0=Rock, 1=Paper, 2=Scissors)
            model_type — "logistic" (default) or "forest"

        Returns self so you can chain: model.train(...).save(...)
        """
        from sklearn.linear_model       import LogisticRegression
        from sklearn.ensemble           import RandomForestClassifier

        # Convert to numpy so sklearn is happy regardless of whether X/y came
        # in as plain Python lists or numpy arrays
        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y, dtype=np.int32)

        if model_type == "forest":
            # Random forest: generally higher accuracy on larger datasets,
            # but can overfit when we only have a few dozen samples
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                n_jobs=-1,    # use all CPU cores to speed up training
            )
        else:
            # Logistic regression: simpler, less likely to overfit,
            # and the coefficients are interpretable via get_feature_importance()
            self.model = LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                random_state=42,
            )

        self.model.fit(X_arr, y_arr)
        self.is_trained = True
        self.classes_   = list(self.model.classes_)

        # Print a training summary so it's clear what was learned
        sample_count = len(y)
        class_counts = {
            INDEX_TO_GESTURE.get(c, c): int(np.sum(y_arr == c))
            for c in sorted(set(y_arr))
        }
        print(f"[RPSModel] Trained {model_type} on {sample_count} samples.")
        print(f"[RPSModel] Class distribution: {class_counts}")

        return self

    def predict_proba(self, features):
        """
        Given a single feature vector, return a dict mapping each gesture
        to the model's estimated probability that the player will throw it.

        Falls back to a uniform 1/3 distribution if the model is not trained,
        so callers never have to handle a None return value.
        """
        if not self.is_trained or self.model is None:
            # Untrained — return equal probabilities (no information)
            return {"Rock": 0.333, "Paper": 0.333, "Scissors": 0.333}

        # sklearn expects a 2D array even for a single sample
        X     = np.array([features], dtype=np.float32)
        proba = self.model.predict_proba(X)[0]

        # Build a dict keyed by gesture name
        result = {}
        for i, cls in enumerate(self.classes_):
            gesture = INDEX_TO_GESTURE.get(cls, str(cls))
            result[gesture] = float(proba[i])

        # Ensure all three gestures are present (sklearn only includes
        # classes it saw during training, so a missing gesture -> 0.0)
        for g in VALID_GESTURES:
            if g not in result:
                result[g] = 0.0

        return result

    def predict(self, features):
        """
        Return the single most-likely gesture string.
        Just picks the key with the highest value from predict_proba().
        """
        proba = self.predict_proba(features)
        return max(proba, key=proba.get)

    def evaluate(self, X, y):
        """
        Evaluate the model on a held-out test set.

        Returns a dict with:
            "accuracy"  — overall fraction correct
            "samples"   — number of test samples
            "report"    — per-class precision/recall/f1 from sklearn
        """
        if not self.is_trained:
            return {"error": "Model not trained"}

        from sklearn.metrics import accuracy_score, classification_report

        X_arr  = np.array(X, dtype=np.float32)
        y_arr  = np.array(y, dtype=np.int32)
        y_pred = self.model.predict(X_arr)

        accuracy = accuracy_score(y_arr, y_pred)

        # Build target_names from the union of true and predicted labels
        # so the report covers every class that appeared in either set
        target_names = [
            INDEX_TO_GESTURE.get(c, str(c))
            for c in sorted(set(y_arr) | set(y_pred))
        ]

        report = classification_report(
            y_arr,
            y_pred,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )

        return {
            "accuracy": round(accuracy, 4),
            "samples":  len(y),
            "report":   report,
        }

    def get_feature_importance(self):
        """
        Return a list of (feature_name, importance_score) pairs, sorted
        highest-first.  Useful for understanding which history signals the
        model relies on most.

        Works for:
            - RandomForest: uses built-in feature_importances_
            - LogisticRegression: averages absolute coefficient magnitudes
              across all classes

        Returns None if the model does not support importance scores.
        """
        if not self.is_trained:
            return None

        names = get_feature_names(self.lookback)

        if hasattr(self.model, "feature_importances_"):
            # Random forest provides a direct importance array
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            # Logistic regression has one coefficient row per class;
            # average absolute values across classes to get a single ranking
            importances = np.mean(np.abs(self.model.coef_), axis=0)
        else:
            return None

        # Sort (name, score) pairs by score descending
        paired = sorted(
            zip(names, importances),
            key=lambda pair: pair[1],
            reverse=True,
        )

        return paired

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, path):
        """
        Pickle the trained model and its metadata to disk.
        The 'lookback' value is saved so we can reconstruct the right
        feature-vector shape on load.
        """
        data = {
            "model":      self.model,
            "lookback":   self.lookback,
            "is_trained": self.is_trained,
            "classes_":   self.classes_,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        print(f"[RPSModel] Saved to {path}")

    @classmethod
    def load(cls, path):
        """
        Load a previously saved RPSModel from disk.
        Returns an untrained empty RPSModel if the file is missing.
        """
        if not os.path.exists(path):
            print(f"[RPSModel] File not found: {path}")
            return cls()  # return a blank (untrained) instance as a safe fallback

        with open(path, "rb") as f:
            data = pickle.load(f)

        # Reconstruct the instance from the saved dict fields
        instance            = cls(model=data["model"], lookback=data.get("lookback", 3))
        instance.is_trained = data.get("is_trained", True)
        instance.classes_   = data.get("classes_")

        print(f"[RPSModel] Loaded from {path}")
        return instance


# ---------------------------------------------------------------------------
# MLPredictionAI — game-compatible AI class
# ---------------------------------------------------------------------------

class MLPredictionAI:
    """
    Drop-in replacement for ChallengeAI or FairPlayAI.

    Uses a trained RPSModel to predict what gesture the player is likely to
    throw next, then returns the move that beats it.

    Matches the interface:
        choose_robot_move(history, streak, round_number)
        self.last_prediction   — dict with debug info from the last call
        reset()                — called between games
    """

    def __init__(self, model_path=None, model=None, lookback=3):
        self.lookback        = lookback
        self.last_prediction = None  # stores metadata from the most recent call

        # Accept a pre-built RPSModel object, a path to load from, or neither
        # (in which case we create an untrained model that falls back to random)
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = RPSModel.load(model_path)
        else:
            self.model = RPSModel(lookback=lookback)  # untrained — will use random

    def reset(self):
        """Clear state between games."""
        self.last_prediction = None

    def choose_robot_move(self, history, streak=0, round_number=1):
        """
        Decide the robot's next move based on the player's history.

        On the first round (or when history is empty), fall back to a random
        choice because there is nothing to predict from yet.

        Sets self.last_prediction with debug info regardless of outcome.

        Returns:
            str — one of "Rock", "Paper", "Scissors"
        """
        # First round or no history — nothing to predict, pick randomly
        if round_number <= 1 or not history:
            self.last_prediction = {
                "top_predicted_move":  None,
                "used_predicted_move": None,
                "effective_skill":     None,
                "ml_probabilities":    None,
            }
            return random.choice(VALID_GESTURES)

        # Build features from the most recently completed round
        current_index  = len(history) - 1
        last_round     = history[-1]
        reaction_time  = last_round.get("reaction_time_ms")  # may be None in old data

        features = extract_features(
            history=history,
            current_index=current_index,
            lookback=self.lookback,
            reaction_time_ms=reaction_time,
        )

        # If we can't build features (malformed history) or the model isn't
        # trained yet, fall back to random rather than crashing
        if features is None or not self.model.is_trained:
            self.last_prediction = {
                "top_predicted_move":  None,
                "used_predicted_move": None,
                "effective_skill":     None,
                "ml_probabilities":    None,
            }
            return random.choice(VALID_GESTURES)

        # Get per-gesture probabilities and pick the most likely player move
        proba          = self.model.predict_proba(features)
        predicted_move = max(proba, key=proba.get)

        # Store metadata so the game UI / research log can show how confident
        # the AI was and which prediction it acted on
        self.last_prediction = {
            "top_predicted_move":  predicted_move,
            "used_predicted_move": predicted_move,
            "effective_skill":     proba.get(predicted_move, 0.0),
            "ml_probabilities":    proba,
        }

        # Return the move that beats the predicted player move
        return COUNTER_MOVE[predicted_move]

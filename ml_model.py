"""
ML Model for RPS Player Prediction

Two classes:
    RPSModel       — wraps scikit-learn: train, predict, save, load, evaluate
    MLPredictionAI — game-compatible AI matching ChallengeAI's interface

Usage (training):
    model = RPSModel()
    model.train(X, y)
    model.save("rps_model.pkl")
    print(model.evaluate(X_test, y_test))

Usage (in-game):
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

VALID_GESTURES = ("Rock", "Paper", "Scissors")
INDEX_TO_GESTURE = {v: k for k, v in GESTURE_INDEX.items()}

COUNTER_MOVE = {
    "Rock": "Paper",
    "Paper": "Scissors",
    "Scissors": "Rock",
}


class RPSModel:
    """
    Thin wrapper around a scikit-learn classifier.

    Default model: LogisticRegression (fast, explainable).
    Can swap in RandomForest or any sklearn classifier.
    """

    def __init__(self, model=None, lookback=3):
        self.lookback = lookback
        self.model = model
        self.is_trained = False
        self.classes_ = None

    def train(self, X, y, model_type="logistic"):
        """
        Train on feature vectors X and labels y.

        model_type:
            "logistic"  — LogisticRegression (default, fast, explainable)
            "forest"    — RandomForestClassifier (more complex, may overfit)

        Returns self for chaining.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier

        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y, dtype=np.int32)

        if model_type == "forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                n_jobs=-1,
            )
        else:
            self.model = LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                random_state=42,
            )

        self.model.fit(X_arr, y_arr)
        self.is_trained = True
        self.classes_ = list(self.model.classes_)

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
        Given a single feature vector, returns {gesture: probability}.

        Returns uniform distribution if the model is not trained.
        """
        if not self.is_trained or self.model is None:
            return {"Rock": 0.333, "Paper": 0.333, "Scissors": 0.333}

        X = np.array([features], dtype=np.float32)
        proba = self.model.predict_proba(X)[0]

        result = {}
        for i, cls in enumerate(self.classes_):
            gesture = INDEX_TO_GESTURE.get(cls, str(cls))
            result[gesture] = float(proba[i])

        # Fill in any missing gestures with 0.
        for g in VALID_GESTURES:
            if g not in result:
                result[g] = 0.0

        return result

    def predict(self, features):
        """
        Returns the single most likely gesture.
        """
        proba = self.predict_proba(features)
        return max(proba, key=proba.get)

    def evaluate(self, X, y):
        """
        Returns a dict with accuracy and per-class metrics.
        """
        if not self.is_trained:
            return {"error": "Model not trained"}

        from sklearn.metrics import accuracy_score, classification_report

        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y, dtype=np.int32)

        y_pred = self.model.predict(X_arr)
        accuracy = accuracy_score(y_arr, y_pred)

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
            "samples": len(y),
            "report": report,
        }

    def get_feature_importance(self):
        """
        Returns feature importances if the model supports them.

        - LogisticRegression: coefficient magnitudes (averaged across classes)
        - RandomForest: feature_importances_
        """
        if not self.is_trained:
            return None

        names = get_feature_names(self.lookback)

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            # Average absolute coefficient across classes.
            importances = np.mean(np.abs(self.model.coef_), axis=0)
        else:
            return None

        paired = sorted(
            zip(names, importances),
            key=lambda pair: pair[1],
            reverse=True,
        )

        return paired

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path):
        """Save the trained model to disk."""
        data = {
            "model": self.model,
            "lookback": self.lookback,
            "is_trained": self.is_trained,
            "classes_": self.classes_,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        print(f"[RPSModel] Saved to {path}")

    @classmethod
    def load(cls, path):
        """Load a trained model from disk."""
        if not os.path.exists(path):
            print(f"[RPSModel] File not found: {path}")
            return cls()

        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls(
            model=data["model"],
            lookback=data.get("lookback", 3),
        )
        instance.is_trained = data.get("is_trained", True)
        instance.classes_ = data.get("classes_")

        print(f"[RPSModel] Loaded from {path}")
        return instance


# ======================================================================
# Game-compatible AI class
# ======================================================================

class MLPredictionAI:
    """
    Drop-in replacement for ChallengeAI / FairPlayAI.

    Uses a trained RPSModel to predict the player's next move,
    then plays the counter.

    Matches the interface:
        choose_robot_move(history, streak, round_number)
        self.last_prediction
        reset()
    """

    def __init__(self, model_path=None, model=None, lookback=3):
        self.lookback = lookback
        self.last_prediction = None

        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = RPSModel.load(model_path)
        else:
            self.model = RPSModel(lookback=lookback)

    def reset(self):
        self.last_prediction = None

    def choose_robot_move(self, history, streak=0, round_number=1):
        """
        Predict the player's next move and return the counter.

        Falls back to random if the model is not trained or there
        is not enough history.
        """
        if round_number <= 1 or not history:
            self.last_prediction = {
                "top_predicted_move": None,
                "used_predicted_move": None,
                "effective_skill": None,
                "ml_probabilities": None,
            }
            return random.choice(VALID_GESTURES)

        # Build features from the most recent round.
        current_index = len(history) - 1

        # Try to get reaction time from the last round if available.
        last_round = history[-1]
        reaction_time = last_round.get("reaction_time_ms")

        features = extract_features(
            history=history,
            current_index=current_index,
            lookback=self.lookback,
            reaction_time_ms=reaction_time,
        )

        if features is None or not self.model.is_trained:
            self.last_prediction = {
                "top_predicted_move": None,
                "used_predicted_move": None,
                "effective_skill": None,
                "ml_probabilities": None,
            }
            return random.choice(VALID_GESTURES)

        # Get probability distribution over gestures.
        proba = self.model.predict_proba(features)
        predicted_move = max(proba, key=proba.get)

        self.last_prediction = {
            "top_predicted_move": predicted_move,
            "used_predicted_move": predicted_move,
            "effective_skill": proba.get(predicted_move, 0.0),
            "ml_probabilities": proba,
        }

        return COUNTER_MOVE[predicted_move]

"""
Front-On Gesture Training Script

Reads landmark data from front_on_training_data.csv
Trains a small classifier and saves it as front_on_gesture_model.pkl

Run from terminal:
    cd ~/rps_hand_counter
    python front_on_trainer.py

Or press 'T' in Diagnostic mode (when collection mode is active)
to train without leaving the app.
"""

import os
import csv
import pickle
import numpy as np
from pathlib import Path


MODEL_DIR = Path.home() / "Desktop" / "CapStone"
CSV_PATH = MODEL_DIR / "front_on_training_data.csv"
MODEL_PATH = MODEL_DIR / "front_on_gesture_model.pkl"

LABEL_TO_INT = {"Rock": 0, "Scissors": 1, "Paper": 2}
INT_TO_LABEL = {0: "Rock", 1: "Scissors", 2: "Paper"}


def load_data():
    """Load CSV into X (features) and y (labels) arrays."""
    if not CSV_PATH.exists():
        print(f"[Trainer] CSV not found: {CSV_PATH}")
        return None, None

    X_rows = []
    y_rows = []

    with open(CSV_PATH, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)

        for row in reader:
            if not row or row[0] not in LABEL_TO_INT:
                continue

            label = LABEL_TO_INT[row[0]]
            features = [float(v) for v in row[1:]]

            if len(features) != 42:
                continue

            X_rows.append(features)
            y_rows.append(label)

    if not X_rows:
        print("[Trainer] No valid samples found in CSV.")
        return None, None

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int32)

    return X, y


def train_and_save():
    """Train classifier and save to disk. Returns accuracy or None."""
    X, y = load_data()
    if X is None:
        return None

    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import cross_val_score

    # Count per class.
    unique, counts = np.unique(y, return_counts=True)
    print(f"[Trainer] Dataset: {len(X)} samples")
    for u, c in zip(unique, counts):
        print(f"  {INT_TO_LABEL[u]}: {c}")

    min_count = min(counts)
    if min_count < 10:
        print(f"[Trainer] WARNING: Need at least 10 samples per gesture. "
              f"Smallest class has {min_count}.")
        if min_count < 3:
            print("[Trainer] Not enough data to train. Collect more samples.")
            return None

    # Small MLP — fast to train, plenty for 3-class on 42 features.
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
    )

    # Cross-validation if enough data.
    if len(X) >= 30:
        n_folds = min(5, min_count)
        scores = cross_val_score(model, X, y, cv=n_folds, scoring="accuracy")
        accuracy = scores.mean()
        print(f"[Trainer] Cross-val accuracy: {accuracy:.1%} "
              f"(±{scores.std():.1%}, {n_folds}-fold)")
    else:
        accuracy = None
        print("[Trainer] Too few samples for cross-validation, training on all data.")

    # Train final model on ALL data.
    model.fit(X, y)

    # Save.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model": model,
            "int_to_label": INT_TO_LABEL,
            "label_to_int": LABEL_TO_INT,
            "n_samples": len(X),
            "accuracy": accuracy,
        }, f)

    print(f"[Trainer] Model saved to {MODEL_PATH}")
    print(f"[Trainer] Total samples: {len(X)}")
    if accuracy is not None:
        print(f"[Trainer] Estimated accuracy: {accuracy:.1%}")

    return accuracy


def load_model():
    """Load trained model from disk. Returns (model, int_to_label) or (None, None)."""
    if not MODEL_PATH.exists():
        return None, None

    try:
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        return data["model"], data["int_to_label"]
    except Exception as exc:
        print(f"[Trainer] Failed to load model: {exc}")
        return None, None


if __name__ == "__main__":
    result = train_and_save()
    if result is not None:
        print(f"\nDone! Estimated accuracy: {result:.1%}")
        print(f"Model saved to: {MODEL_PATH}")
    else:
        print("\nTraining failed. Check that you have enough samples.")
        print(f"CSV location: {CSV_PATH}")

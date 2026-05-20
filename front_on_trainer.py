"""
front_on_trainer.py
===================
Trains the front-on gesture classifier from collected CSV data and saves
the result as a pickle file that front_on_classifier.py loads at runtime.

WHERE IT FITS:
    landmark_collector.py  -->  front_on_training_data.csv
                                       |
                              front_on_trainer.py  (this file)
                                       |
                          front_on_gesture_model.pkl
                                       |
                           front_on_classifier.py  (uses at runtime)

HOW TO RUN:
    Option A — from the terminal:
        cd ~/rps_hand_counter
        python front_on_trainer.py

    Option B — press 'T' inside Diagnostic mode (when collection is active)
        to train without leaving the app.
"""

import csv
import pickle
from pathlib import Path

import numpy as np

# FEATURE_DIM must match front_on_features.FEATURE_DIM (currently 20).
# We duplicate it here so this script can run standalone without importing
# the rest of the project.
FEATURE_DIM = 20

# --- Resolve the output directory -------------------------------------------
# If the project ships capstone_paths.py, use its CAPSTONE_DIR constant.
# Otherwise, fall back to ~/Desktop/CapStone on macOS or ~/CapStone elsewhere.
try:
    from capstone_paths import CAPSTONE_DIR as MODEL_DIR
except ImportError:
    import sys as _sys
    MODEL_DIR = (
        Path.home() / "Desktop" / "CapStone"
        if _sys.platform == "darwin"
        else Path.home() / "CapStone"
    )

CSV_PATH   = MODEL_DIR / "front_on_training_data.csv"
MODEL_PATH = MODEL_DIR / "front_on_gesture_model.pkl"

# Integer label encoding used in the CSV and the saved model.
# Keep the order consistent — changing it would break a trained model.
LABEL_TO_INT = {"Rock": 0, "Scissors": 1, "Paper": 2}
INT_TO_LABEL = {0: "Rock",  1: "Scissors", 2: "Paper"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """
    Read the training CSV and return (X, y) as numpy arrays.

    The CSV has one row per sample. The first column is the gesture label
    (Rock / Scissors / Paper), and the remaining FEATURE_DIM columns are
    the numeric features produced by front_on_features.extract_features().

    Returns (None, None) if the file is missing or contains no valid rows.
    """
    if not CSV_PATH.exists():
        print(f"[Trainer] CSV not found: {CSV_PATH}")
        return None, None

    X_rows = []
    y_rows = []

    with open(CSV_PATH, "r") as f:
        reader = csv.reader(f)
        next(reader, None)   # skip the header row

        for row in reader:
            # Skip blank rows and rows with unrecognised labels
            if not row or row[0] not in LABEL_TO_INT:
                continue

            label    = LABEL_TO_INT[row[0]]
            features = [float(v) for v in row[1:]]

            # Skip rows with the wrong feature count — they were recorded with
            # an older version of the feature extractor and are incompatible
            if len(features) != FEATURE_DIM:
                continue

            X_rows.append(features)
            y_rows.append(label)

    if not X_rows:
        print("[Trainer] No valid samples found in CSV.")
        return None, None

    # Convert to numpy arrays for sklearn compatibility
    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int32)

    return X, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_save():
    """
    Load the CSV, train an MLP classifier, run cross-validation, and save
    the resulting model to disk.

    We use a small two-layer MLP (64 -> 32 nodes) because:
    - It's fast to train on the small dataset collected in Diagnostic mode
    - It handles the non-linear decision boundaries between gestures well enough
    - It can be retrained in-app without noticeable lag

    Returns:
        float — cross-validation accuracy (0.0 to 1.0), or None if training failed.
    """
    X, y = load_data()
    if X is None:
        return None

    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import cross_val_score

    # Print a per-class breakdown so we can see if any gesture needs more samples
    unique, counts = np.unique(y, return_counts=True)
    print(f"[Trainer] Dataset: {len(X)} samples")
    for u, c in zip(unique, counts):
        print(f"  {INT_TO_LABEL[u]}: {c}")

    min_count = min(counts)

    # Require at least 5 samples per gesture — fewer than that and the model
    # has almost nothing to learn for the underrepresented class
    if min_count < 5:
        print(
            f"[Trainer] Not enough data to train. Need at least 5 per gesture, "
            f"smallest class has {min_count}."
        )
        return None

    # Enable early stopping only when the dataset is large enough to spare
    # a 15% validation split — we need at least 60 samples so each split is useful
    use_early_stopping = len(X) >= 60

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),   # two hidden layers, enough for 3 output classes
        activation="relu",
        max_iter=1000,
        random_state=42,               # fixed seed for reproducible results
        early_stopping=use_early_stopping,
        validation_fraction=0.15 if use_early_stopping else 0.0,
    )

    # Cross-validation gives a more honest accuracy estimate than a single train/test
    # split, because it tests on every part of the data at least once
    if len(X) >= 30:
        # Use at most 5 folds, but never more folds than samples in the smallest class
        n_folds  = min(5, min_count)
        scores   = cross_val_score(model, X, y, cv=n_folds, scoring="accuracy")
        accuracy = scores.mean()
        print(
            f"[Trainer] Cross-val accuracy: {accuracy:.1%} "
            f"(+/-{scores.std():.1%}, {n_folds}-fold)"
        )
    else:
        # Too few samples for meaningful cross-validation — report None
        accuracy = None
        print("[Trainer] Too few samples for cross-validation, training on all data.")

    # Train the final model on ALL available data so we get maximum coverage
    model.fit(X, y)

    # Create the output directory if it doesn't exist yet
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save everything the classifier needs to load and run predictions
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model":        model,
            "int_to_label": INT_TO_LABEL,
            "label_to_int": LABEL_TO_INT,
            "n_samples":    len(X),
            "accuracy":     accuracy,
        }, f)

    print(f"[Trainer] Model saved to {MODEL_PATH}")
    print(f"[Trainer] Total samples: {len(X)}")
    if accuracy is not None:
        print(f"[Trainer] Estimated accuracy: {accuracy:.1%}")

    return accuracy


# ---------------------------------------------------------------------------
# Loading helper (convenience function used by front_on_classifier.py)
# ---------------------------------------------------------------------------

def load_model():
    """
    Load the saved model from disk.

    Returns:
        (model, int_to_label) on success, or (None, None) if the file is
        missing or corrupt.
    """
    if not MODEL_PATH.exists():
        return None, None

    try:
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        return data["model"], data["int_to_label"]
    except Exception as exc:
        print(f"[Trainer] Failed to load model: {exc}")
        return None, None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = train_and_save()
    if result is not None:
        print(f"\nDone! Estimated accuracy: {result:.1%}")
        print(f"Model saved to: {MODEL_PATH}")
    else:
        print("\nTraining failed. Check that you have enough samples.")
        print(f"CSV location: {CSV_PATH}")

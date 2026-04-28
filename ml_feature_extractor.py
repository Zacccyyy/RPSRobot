"""
ML Feature Extractor

Converts a list of round history dicts into a numeric feature vector
that scikit-learn models can consume.

Each round dict looks like:
    {
        "round_number": 3,
        "player_gesture": "Rock",
        "robot_gesture": "Paper",
        "player_outcome": "lose",
    }

Feature groups:
    1. Last N player gestures  (one-hot, 3 values each)
    2. Last N outcomes          (one-hot, 3 values each)
    3. Last response type       (one-hot: stay / upgrade / downgrade)
    4. Session gesture frequencies (3 floats, normalised 0–1)
    5. Current streak           (1 float, normalised)
    6. Reaction time            (1 float, normalised, optional)
"""

GESTURE_INDEX = {"Rock": 0, "Paper": 1, "Scissors": 2}
OUTCOME_INDEX = {"win": 0, "lose": 1, "draw": 2}

UPGRADE_MOVE = {
    "Rock": "Paper",
    "Paper": "Scissors",
    "Scissors": "Rock",
}

DOWNGRADE_MOVE = {
    "Rock": "Scissors",
    "Paper": "Rock",
    "Scissors": "Paper",
}


def _one_hot_gesture(gesture):
    """Returns [R, P, S] as 0/1."""
    vec = [0.0, 0.0, 0.0]
    idx = GESTURE_INDEX.get(gesture)
    if idx is not None:
        vec[idx] = 1.0
    return vec


def _one_hot_outcome(outcome):
    """Returns [win, lose, draw] as 0/1."""
    vec = [0.0, 0.0, 0.0]
    idx = OUTCOME_INDEX.get(outcome)
    if idx is not None:
        vec[idx] = 1.0
    return vec


def _get_response_type(previous_gesture, current_gesture):
    """Classify how the player changed between two consecutive throws."""
    if current_gesture == previous_gesture:
        return "stay"
    if UPGRADE_MOVE.get(previous_gesture) == current_gesture:
        return "upgrade"
    if DOWNGRADE_MOVE.get(previous_gesture) == current_gesture:
        return "downgrade"
    return "unknown"


def _one_hot_response_type(response_type):
    """Returns [stay, upgrade, downgrade] as 0/1."""
    mapping = {"stay": 0, "upgrade": 1, "downgrade": 2}
    vec = [0.0, 0.0, 0.0]
    idx = mapping.get(response_type)
    if idx is not None:
        vec[idx] = 1.0
    return vec


def _gesture_frequencies(history):
    """
    Returns normalised gesture frequencies [R_frac, P_frac, S_frac]
    from the full history so far.
    """
    counts = [0, 0, 0]
    for record in history:
        idx = GESTURE_INDEX.get(record["player_gesture"])
        if idx is not None:
            counts[idx] += 1

    total = sum(counts)
    if total == 0:
        return [0.333, 0.333, 0.333]

    return [c / total for c in counts]


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def get_feature_names(lookback=3):
    """
    Returns the ordered list of feature names for a given lookback.
    Useful for inspecting model weights or debugging.
    """
    names = []

    for i in range(lookback):
        step = lookback - i
        names.extend([
            f"prev{step}_rock",
            f"prev{step}_paper",
            f"prev{step}_scissors",
        ])

    for i in range(lookback):
        step = lookback - i
        names.extend([
            f"outcome{step}_win",
            f"outcome{step}_lose",
            f"outcome{step}_draw",
        ])

    names.extend([
        "response_stay",
        "response_upgrade",
        "response_downgrade",
    ])

    names.extend([
        "freq_rock",
        "freq_paper",
        "freq_scissors",
    ])

    names.append("streak_norm")
    names.append("reaction_time_norm")

    return names


def extract_features(history, current_index, lookback=3, reaction_time_ms=None):
    """
    Build a feature vector for predicting the player's NEXT move
    after the round at current_index.

    Parameters:
        history         list of round dicts (full session)
        current_index   index of the most recent completed round
        lookback        how many past rounds to encode (default 3)
        reaction_time_ms  optional reaction time for the current round

    Returns:
        list of floats (the feature vector), or None if not enough data.
    """
    if current_index < 0 or current_index >= len(history):
        return None

    features = []

    # --- 1. Last N player gestures (one-hot) ---
    for i in range(lookback):
        idx = current_index - (lookback - 1 - i)
        if idx >= 0:
            features.extend(_one_hot_gesture(history[idx]["player_gesture"]))
        else:
            features.extend([0.0, 0.0, 0.0])

    # --- 2. Last N outcomes (one-hot) ---
    for i in range(lookback):
        idx = current_index - (lookback - 1 - i)
        if idx >= 0:
            features.extend(_one_hot_outcome(history[idx]["player_outcome"]))
        else:
            features.extend([0.0, 0.0, 0.0])

    # --- 3. Response type of most recent transition ---
    if current_index >= 1:
        prev_gesture = history[current_index - 1]["player_gesture"]
        curr_gesture = history[current_index]["player_gesture"]
        response = _get_response_type(prev_gesture, curr_gesture)
    else:
        response = "unknown"

    features.extend(_one_hot_response_type(response))

    # --- 4. Session gesture frequencies so far ---
    freq_history = history[: current_index + 1]
    features.extend(_gesture_frequencies(freq_history))

    # --- 5. Streak (normalised: divide by 20 to keep roughly 0–1) ---
    streak = 0
    for j in range(current_index, -1, -1):
        if history[j]["player_outcome"] == "win":
            streak += 1
        else:
            break
    features.append(min(streak / 20.0, 1.0))

    # --- 6. Reaction time (normalised: divide by 500ms) ---
    if reaction_time_ms is not None:
        features.append(min(reaction_time_ms / 500.0, 1.0))
    else:
        features.append(0.5)  # neutral default

    return features


def build_training_set(rounds_by_run, lookback=3):
    """
    Given a dict of {run_id: [list of round dicts]}, build X and y
    arrays ready for scikit-learn.

    Each sample predicts what the player threw on round N+1
    given features from rounds up to N.

    Returns:
        X  list of feature vectors
        y  list of target labels (0=Rock, 1=Paper, 2=Scissors)
    """
    X = []
    y = []

    for run_id, rounds in rounds_by_run.items():
        for i in range(len(rounds) - 1):
            features = extract_features(
                history=rounds,
                current_index=i,
                lookback=lookback,
                reaction_time_ms=rounds[i].get("reaction_time_ms"),
            )

            if features is None:
                continue

            next_gesture = rounds[i + 1]["player_gesture"]
            target = GESTURE_INDEX.get(next_gesture)

            if target is None:
                continue

            X.append(features)
            y.append(target)

    return X, y

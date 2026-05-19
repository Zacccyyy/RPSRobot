"""
ml_feature_extractor.py
=======================
Converts a player's round history into numeric feature vectors that
scikit-learn models can consume for predicting the player's NEXT move.

WHERE IT FITS:
    rps_game_state.py  (writes round dicts into history list)
            |
    ml_feature_extractor.py  (this file — turns history into numbers)
            |
    ml_model.py  (RPSModel / MLPredictionAI — trains and infers)

WHAT A ROUND DICT LOOKS LIKE:
    {
        "round_number":   3,
        "player_gesture": "Rock",
        "robot_gesture":  "Paper",
        "player_outcome": "lose",        # "win" | "lose" | "draw"
        "reaction_time_ms": 412,         # optional
    }

FEATURE GROUPS (for default lookback=3, 21 values total + 1 optional = 22):
    1. Last N player gestures        — one-hot, 3 values each  (9 total)
    2. Last N outcomes               — one-hot, 3 values each  (9 total)
    3. Most recent response type     — one-hot, 3 values        (3 total)
    4. Session gesture frequencies   — 3 normalised floats      (3 total)
    5. Current win streak            — 1 float, normalised      (1 total)
    6. Reaction time                 — 1 float, normalised      (1 total)
"""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Encode gestures as integers for sklearn's numeric labels
GESTURE_INDEX = {"Rock": 0, "Paper": 1, "Scissors": 2}

# Encode round outcomes as integers
OUTCOME_INDEX = {"win": 0, "lose": 1, "draw": 2}

# Given your last gesture, what would be an "upgrade" (beats the thing that
# beats you)?  e.g. if you just played Rock and lost to Paper, upgrade = Scissors
UPGRADE_MOVE = {
    "Rock":     "Paper",
    "Paper":    "Scissors",
    "Scissors": "Rock",
}

# What would be a "downgrade" (the move your last move beats)?
DOWNGRADE_MOVE = {
    "Rock":     "Scissors",
    "Paper":    "Rock",
    "Scissors": "Paper",
}


# ---------------------------------------------------------------------------
# Internal one-hot helpers
# ---------------------------------------------------------------------------

def _one_hot_gesture(gesture):
    """
    Turn a gesture string into a 3-element binary vector [Rock, Paper, Scissors].
    e.g. "Paper" -> [0.0, 1.0, 0.0]
    Returns all zeros for an unrecognised gesture (treated as missing data).
    """
    vec = [0.0, 0.0, 0.0]
    idx = GESTURE_INDEX.get(gesture)
    if idx is not None:
        vec[idx] = 1.0
    return vec


def _one_hot_outcome(outcome):
    """
    Turn an outcome string into [win, lose, draw].
    e.g. "lose" -> [0.0, 1.0, 0.0]
    """
    vec = [0.0, 0.0, 0.0]
    idx = OUTCOME_INDEX.get(outcome)
    if idx is not None:
        vec[idx] = 1.0
    return vec


def _get_response_type(previous_gesture, current_gesture):
    """
    Classify how the player changed their gesture between two consecutive throws.

    "stay"      — played the same thing again
    "upgrade"   — switched to the move that beats the thing that beats them
    "downgrade" — switched to the move that their last gesture beats
    "unknown"   — unrecognised gesture string

    This captures a common human tendency to shift in a predictable direction
    after winning or losing (e.g. "I lost with Rock, so I'll try Paper").
    """
    if current_gesture == previous_gesture:
        return "stay"
    if UPGRADE_MOVE.get(previous_gesture) == current_gesture:
        return "upgrade"
    if DOWNGRADE_MOVE.get(previous_gesture) == current_gesture:
        return "downgrade"
    return "unknown"


def _one_hot_response_type(response_type):
    """
    Encode response type as [stay, upgrade, downgrade].
    "unknown" maps to all zeros.
    """
    mapping = {"stay": 0, "upgrade": 1, "downgrade": 2}
    vec = [0.0, 0.0, 0.0]
    idx = mapping.get(response_type)
    if idx is not None:
        vec[idx] = 1.0
    return vec


def _gesture_frequencies(history):
    """
    Compute what fraction of ALL rounds so far the player has used each gesture.

    Returns [rock_frac, paper_frac, scissors_frac], normalised so they sum to 1.
    If there is no history yet, return equal thirds (no information = assume uniform).
    """
    counts = [0, 0, 0]
    for record in history:
        idx = GESTURE_INDEX.get(record["player_gesture"])
        if idx is not None:
            counts[idx] += 1

    total = sum(counts)
    if total == 0:
        return [0.333, 0.333, 0.333]  # no data — assume uniform distribution

    return [c / total for c in counts]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_feature_names(lookback=3):
    """
    Return an ordered list of human-readable feature names matching the
    vector produced by extract_features() for the given lookback window.

    Useful for debugging (e.g. printing which features the model relies on
    most heavily via get_feature_importance()).
    """
    names = []

    # Last N gestures, oldest first (prev3, prev2, prev1)
    for i in range(lookback):
        step = lookback - i
        names.extend([
            f"prev{step}_rock",
            f"prev{step}_paper",
            f"prev{step}_scissors",
        ])

    # Last N outcomes, oldest first
    for i in range(lookback):
        step = lookback - i
        names.extend([
            f"outcome{step}_win",
            f"outcome{step}_lose",
            f"outcome{step}_draw",
        ])

    # How the player transitioned on the most recent throw
    names.extend(["response_stay", "response_upgrade", "response_downgrade"])

    # Overall session frequencies — tells the model "this player mostly plays Rock"
    names.extend(["freq_rock", "freq_paper", "freq_scissors"])

    # Streak and reaction time
    names.append("streak_norm")
    names.append("reaction_time_norm")

    return names


def extract_features(history, current_index, lookback=3, reaction_time_ms=None):
    """
    Build a feature vector describing the game state up to round current_index.
    The vector is designed to help a classifier predict what gesture the player
    will throw NEXT (i.e. on round current_index + 1).

    Parameters:
        history          — list of all round dicts for this session
        current_index    — index of the most recently completed round (0-based)
        lookback         — how many past rounds to encode (default 3)
        reaction_time_ms — optional: how long the player took to throw this round

    Returns:
        list of floats (the feature vector), or None if current_index is invalid.
    """
    # Validate index before touching the list
    if current_index < 0 or current_index >= len(history):
        return None

    features = []

    # --- 1. Last N player gestures (one-hot each) ----------------------------
    # We encode the window [current_index - (lookback-1)  ...  current_index].
    # Rounds before the session started are filled with zeros (no information).
    for i in range(lookback):
        idx = current_index - (lookback - 1 - i)  # oldest to newest
        if idx >= 0:
            features.extend(_one_hot_gesture(history[idx]["player_gesture"]))
        else:
            features.extend([0.0, 0.0, 0.0])  # pre-session padding

    # --- 2. Last N outcomes (one-hot each) -----------------------------------
    for i in range(lookback):
        idx = current_index - (lookback - 1 - i)
        if idx >= 0:
            features.extend(_one_hot_outcome(history[idx]["player_outcome"]))
        else:
            features.extend([0.0, 0.0, 0.0])

    # --- 3. Response type of the most recent transition ----------------------
    # What did the player do differently compared to the round before?
    if current_index >= 1:
        prev_gesture = history[current_index - 1]["player_gesture"]
        curr_gesture = history[current_index]["player_gesture"]
        response = _get_response_type(prev_gesture, curr_gesture)
    else:
        response = "unknown"  # first round — no transition to measure

    features.extend(_one_hot_response_type(response))

    # --- 4. Session-wide gesture frequencies ---------------------------------
    # Slice the history up to (and including) the current round
    freq_history = history[: current_index + 1]
    features.extend(_gesture_frequencies(freq_history))

    # --- 5. Current win streak -----------------------------------------------
    # Count how many consecutive wins the player has going into this round.
    # Divide by 20 to roughly normalise to [0, 1] (cap at 1.0 if longer).
    streak = 0
    for j in range(current_index, -1, -1):
        if history[j]["player_outcome"] == "win":
            streak += 1
        else:
            break  # streak is broken; stop counting
    features.append(min(streak / 20.0, 1.0))

    # --- 6. Reaction time ----------------------------------------------------
    # Divide by 500 ms to normalise: a "typical" fast throw is ~200-400 ms,
    # so this puts most values in [0.4, 0.8].
    # We use 0.0 as a sentinel for "not recorded" — the model learns this.
    if reaction_time_ms is not None:
        features.append(min(reaction_time_ms / 500.0, 1.0))
    else:
        features.append(0.0)

    return features


def build_training_set(rounds_by_run, lookback=3):
    """
    Convert a full multi-session history dict into (X, y) arrays for training.

    For each run, we treat every consecutive (round_N, round_N+1) pair as a
    training sample: the features come from round N, and the label is what
    the player actually threw on round N+1.

    Parameters:
        rounds_by_run — dict of {run_id: [list of round dicts]}
        lookback      — passed through to extract_features()

    Returns:
        X — list of feature vectors
        y — list of integer labels (0=Rock, 1=Paper, 2=Scissors)
    """
    X = []
    y = []

    for run_id, rounds in rounds_by_run.items():
        # We need at least two rounds to form one (features, label) pair
        for i in range(len(rounds) - 1):
            features = extract_features(
                history=rounds,
                current_index=i,
                lookback=lookback,
                reaction_time_ms=rounds[i].get("reaction_time_ms"),
            )

            if features is None:
                continue  # skip if feature extraction failed for this round

            # The target label is what the player threw on the NEXT round
            next_gesture = rounds[i + 1]["player_gesture"]
            target = GESTURE_INDEX.get(next_gesture)

            if target is None:
                continue  # skip unrecognised gesture strings

            X.append(features)
            y.append(target)

    return X, y

# ============================================================
# fair_play_ai.py
#
# The core AI "brain" used by Fair Play mode.
#
# This file defines FairPlayAI, which watches how the human player
# has been throwing and tries to predict their next move, then
# returns the gesture that beats it.
#
# How the prediction works (six stacked layers):
#   1. Population priors   - baseline psychology (humans tend to stay after winning, etc.)
#   2. Outcome response    - how THIS player specifically behaves after win/lose/draw
#   3. Transition memory   - what they tend to throw after a given gesture
#   4. Outcome-next move   - weaker layer: after a given outcome, what gesture appears?
#   5. Frequency fallback  - just lean toward whatever they throw most overall
#   6. Markov tables       - combines outcome + gesture into a single richer table
#
# Each layer's contribution is tuned by Thompson Sampling (a Bayesian
# method that learns which layers are actually accurate for this player).
#
# Personality presets let AI characters weight the layers differently.
# Difficulty presets control base skill, skill ceiling, and grace period.
#
# Where this fits:
#   fair_play_state.py creates a FairPlayAI and calls choose_robot_move()
#   once per round after beat 3. challenge_ai.py subclasses FairPlayAI,
#   so changes here affect Challenge mode too.
# ============================================================

import random
from collections import defaultdict


# The three legal gestures — used throughout for validation.
VALID_GESTURES = ("Rock", "Paper", "Scissors")

# What beats what: the key BEATS the value.
# Used to pick the counter-move once we've predicted what the player will throw.
COUNTER_MOVE = {
    "Rock":     "Paper",
    "Paper":    "Scissors",
    "Scissors": "Rock",
}

# What gesture comes "next" in the Rock -> Paper -> Scissors cycle.
# Used to classify when a player "upgrades" (steps forward in the cycle).
UPGRADE_MOVE = {
    "Rock":     "Paper",
    "Paper":    "Scissors",
    "Scissors": "Rock",
}

# What gesture comes "before" in the cycle — the opposite direction to UPGRADE_MOVE.
# Used to classify when a player "downgrades" (steps backward in the cycle).
DOWNGRADE_MOVE = {
    "Rock":     "Scissors",
    "Paper":    "Rock",
    "Scissors": "Paper",
}

# Difficulty presets: starting skill, max skill, and how many early "grace" rounds
# the AI plays near-randomly so the game feels fair at the start.
# DO NOT change these — they are calibrated.
_DIFFICULTY_PRESETS = {
    "Easy":   {"base_skill": 0.40, "max_skill": 0.55, "grace_rounds": 20},
    "Normal": {"base_skill": 0.66, "max_skill": 0.76, "grace_rounds": 10},
    "Hard":   {"base_skill": 0.80, "max_skill": 0.92, "grace_rounds":  0},
}

# ── AI Personality presets ──────────────────────────────────────────────────
# Each personality tweaks how the AI predicts and how it "misses" on purpose.
#
# "layer_bias" is a dict that multiplies each prediction layer's contribution.
#   e.g. {"outcome": 2.5} means the outcome layer counts 2.5x as much.
#
# "miss_mode" controls what the AI throws when the skill roll says "don't play optimally":
#   "random"  — pick uniformly at random from all gestures
#   "second"  — pick the second-best prediction (still somewhat smart)
#   "chaos"   — pure 33/33/33 Nash equilibrium (statistically unbeatable)
#   "modal"   — always throw the player's most-common gesture
#   "delayed" — throw back whatever the player threw last round (The Ghost)
PERSONALITIES = {
    "Normal": {
        "label":       "Normal",
        "desc":        "Balanced adaptive play. The default.",
        "skill_mult":  1.0,
        "grace_mult":  1.0,
        "layer_bias":  {},        # no layer overrides
        "miss_mode":   "random",
        "bluff_rate":  0.0,
    },
    "The Psychologist": {
        "label":       "The Psychologist",
        "desc":        "Exploits win-stay/lose-shift biases. Doubles down on outcome layers.",
        "skill_mult":  1.05,
        "grace_mult":  0.8,
        "layer_bias":  {"outcome": 2.5, "transition": 0.6, "frequency": 0.3},
        "miss_mode":   "second",
        "bluff_rate":  0.0,
    },
    "The Gambler": {
        "label":       "The Gambler",
        "desc":        "High variance. Occasionally ignores all patterns and plays wild.",
        "skill_mult":  0.9,
        "grace_mult":  1.2,
        "layer_bias":  {"outcome": 0.8, "transition": 0.8},
        "miss_mode":   "random",
        "bluff_rate":  0.0,
        "wild_rate":   0.20,      # 20% chance of a fully random throw each round
    },
    "The Mirror": {
        "label":       "The Mirror",
        "desc":        "Copies your most common gesture. Blatantly exploitable if you adapt.",
        "skill_mult":  0.6,
        "grace_mult":  2.0,
        "layer_bias":  {"frequency": 4.0, "outcome": 0.2, "transition": 0.2},
        "miss_mode":   "modal",
        "bluff_rate":  0.0,
    },
    "The Ghost": {
        # INTENTIONAL: this AI echoes the player's last throw — it does NOT counter it.
        # Setting use_delayed=True triggers a special early-return in choose_robot_move().
        "label":       "The Ghost",
        "desc":        "Plays your previous move back at you. One step behind, always.",
        "skill_mult":  0.55,
        "grace_mult":  2.0,
        "layer_bias":  {},
        "miss_mode":   "delayed",
        "bluff_rate":  0.0,
        "use_delayed": True,
    },
    "The Chaos Agent": {
        "label":       "The Chaos Agent",
        "desc":        "Pure Nash equilibrium. Unbeatable in theory, unreadable in practice.",
        "skill_mult":  0.0,       # skill_mult=0 means it always "misses" (i.e. goes to miss_mode)
        "grace_mult":  0.0,
        "layer_bias":  {},
        "miss_mode":   "chaos",
        "bluff_rate":  0.0,
    },
    "The Hustler": {
        "label":       "The Hustler",
        "desc":        "Hard-reads your patterns early, then plays dumb when it's winning.",
        "skill_mult":  1.12,
        "grace_mult":  0.5,       # short grace period — it learns fast
        "layer_bias":  {"transition": 2.0, "markov": 2.0, "outcome": 1.2},
        "miss_mode":   "second",
        "bluff_rate":  0.0,
    },
}

# Flat list of personality names — used by UI menus to iterate options.
PERSONALITY_NAMES = list(PERSONALITIES.keys())


def _detect_opponent_type(history):
    """
    Classify the human player's playstyle based on their round history.

    We look for a handful of well-known patterns:
      - rock_heavy / paper_heavy / scissors_heavy: one gesture used > 50% of the time
      - win_stay: player repeats the same move after winning more than 60% of the time
      - cycler:   player follows Rock -> Paper -> Scissors in sequence > 65% of the time
      - random:   no pattern found (the default fallback)

    Returns a string label. Needs at least 8 rounds before it tries — earlier than
    that the sample is too small to be meaningful.
    """
    # Not enough history yet — call it random and move on.
    if len(history) < 8:
        return "random"

    # Pull out just the gesture names, skipping any "Unknown" or bad entries.
    gestures = [r["player_gesture"] for r in history if r["player_gesture"] in VALID_GESTURES]
    if not gestures:
        return "random"

    total = len(gestures)

    # Check if any single gesture makes up more than half of all throws.
    freq = {g: gestures.count(g) / total for g in VALID_GESTURES}
    for g, f in freq.items():
        if f > 0.50:
            return f"{g.lower()}_heavy"

    # Win-stay check: after a win, did they throw the same gesture again?
    wins = [r for r in history if r.get("player_outcome") == "win"]
    win_stay_count = sum(
        1 for i, r in enumerate(history[:-1])
        if r.get("player_outcome") == "win"
        and history[i + 1]["player_gesture"] == r["player_gesture"]
    )
    if wins and win_stay_count / max(len(wins), 1) > 0.60:
        return "win_stay"

    # Cycler check: look at the last 9 throws for Rock -> Paper -> Scissors progressions.
    cycle = ("Rock", "Paper", "Scissors")
    recent = gestures[-9:]
    cycle_hits = sum(
        1 for i in range(len(recent) - 1)
        if recent[i] in cycle
        and recent[i + 1] == cycle[(cycle.index(recent[i]) + 1) % 3]
    )
    if len(recent) > 1 and cycle_hits / (len(recent) - 1) > 0.65:
        return "cycler"

    return "random"


def _markov_move_scores(history, move_scores):
    """
    Layer 6: WIN / LOSE / DRAW Markov transition tables.

    The insight: players behave differently depending on whether they just won,
    lost, or drew. This builds three separate tables — one per outcome — that
    track what gesture the player throws NEXT given their last gesture and last outcome.

    Example: if the player lost while throwing Rock, this layer might notice
    they usually switch to Paper next time.

    Recency weighting means more recent rounds count more (same 1.28 base as elsewhere).
    The result is added directly into move_scores (the caller passes the dict by reference).
    """
    # Need at least a couple of rounds to find any transitions at all.
    if len(history) < 3:
        return

    # Three tables: each maps (gesture_before -> gesture_after -> accumulated_weight).
    tables = {
        "win":  defaultdict(lambda: defaultdict(float)),
        "lose": defaultdict(lambda: defaultdict(float)),
        "draw": defaultdict(lambda: defaultdict(float)),
    }

    # Walk through every consecutive pair of rounds and record the weighted transition.
    for i in range(len(history) - 1):
        outcome = history[i].get("player_outcome", "")
        g_from  = history[i].get("player_gesture", "")
        g_to    = history[i + 1].get("player_gesture", "")

        # Skip anything that isn't a recognised outcome or gesture.
        if outcome in tables and g_from in VALID_GESTURES and g_to in VALID_GESTURES:
            # Older rounds get smaller weight: 1.28^-distance decays smoothly.
            distance = (len(history) - 2) - i
            weight   = 1.0 / (1.28 ** distance)
            tables[outcome][g_from][g_to] += weight

    # Now look at the most recent round and score what the player will likely do next.
    last    = history[-1]
    outcome = last.get("player_outcome", "")
    g_from  = last.get("player_gesture", "")

    if outcome in tables and g_from in tables[outcome]:
        row   = tables[outcome][g_from]
        total = sum(row.values())
        if total > 0:
            # Normalise to probabilities and scale by 1.8 (calibrated contribution weight).
            for g_to, w in row.items():
                move_scores[g_to] += 1.8 * (w / total)


class FairPlayAI:
    """
    Fair Play AI v3

    Predicts the player's next move using six stacked layers (described at
    the top of this file), then returns the gesture that beats that prediction.

    Thompson Sampling (a Bayesian bandit algorithm) dynamically adjusts how
    much the AI trusts each prediction layer based on its real accuracy so far.

    Personality presets change how layers are weighted and how the AI "misses"
    on purpose to keep the difficulty fair. Difficulty presets set the base
    skill, max skill ceiling, and length of the initial grace period.
    """

    def __init__(self, base_skill=0.66, max_skill=0.76, difficulty="Normal",
                 personality="Normal"):
        # Load the difficulty preset (Easy / Normal / Hard) for skill and grace settings.
        preset = _DIFFICULTY_PRESETS.get(difficulty, _DIFFICULTY_PRESETS["Normal"])
        self.base_skill   = preset["base_skill"]
        self.max_skill    = preset["max_skill"]
        self.grace_rounds = preset["grace_rounds"]
        self.difficulty   = difficulty
        self.last_prediction = None
        self.set_personality(personality)
        self.reset()

    def set_personality(self, name):
        """
        Apply a personality preset by name (e.g. "The Psychologist").

        Unpacks all the relevant fields into private attributes with an underscore
        prefix so they're quick to access in the prediction methods.
        """
        p = PERSONALITIES.get(name, PERSONALITIES["Normal"])
        self.personality    = name
        self._p_skill_mult  = p.get("skill_mult",  1.0)
        self._p_grace_mult  = p.get("grace_mult",  1.0)
        self._p_layer_bias  = p.get("layer_bias",  {})
        self._p_miss_mode   = p.get("miss_mode",   "random")
        self._p_wild_rate   = p.get("wild_rate",   0.0)
        self._p_use_delayed = p.get("use_delayed", False)

    def set_difficulty(self, difficulty):
        """Swap the difficulty preset mid-session without clearing any learned history."""
        preset = _DIFFICULTY_PRESETS.get(difficulty, _DIFFICULTY_PRESETS["Normal"])
        self.base_skill   = preset["base_skill"]
        self.max_skill    = preset["max_skill"]
        self.grace_rounds = preset["grace_rounds"]
        self.difficulty   = difficulty

    def reset(self):
        """
        Wipe everything learned from the current match.
        Called at match start and when the player restarts mid-game.
        """
        self.last_prediction     = None
        self._consecutive_wins   = 0
        self._consecutive_losses = 0

        # Thompson Sampling bandit: one Beta(alpha, beta) distribution per layer.
        # Starting at (1, 1) is a flat prior — the AI begins with no bias toward any layer.
        # Alpha grows when the layer predicts correctly; beta grows when it's wrong.
        self._bandit = {
            "outcome":    [1.0, 1.0],   # [alpha = successes, beta = failures]
            "transition": [1.0, 1.0],
            "opp_next":   [1.0, 1.0],
            "frequency":  [1.0, 1.0],
            "markov":     [1.0, 1.0],
        }
        # Saves each layer's contribution weight from the previous round so
        # update_bandit() knows which layers to reward or penalise.
        self._last_layer_contributions = {}

    def update_bandit(self, ai_predicted_gesture, actual_player_gesture):
        """
        Update the Thompson Sampling bandit after each resolved round.

        ai_predicted_gesture  — what the AI thought the player would throw
        actual_player_gesture — what the player actually threw

        If the prediction was correct, layers that contributed most get their
        alpha (success count) bumped up. If wrong, their beta (failure count)
        is bumped at half rate — we don't punish mistakes too harshly.

        Values are capped at 50.0 so no single layer becomes permanently dominant.
        """
        if not self._last_layer_contributions:
            return

        correct = (ai_predicted_gesture == actual_player_gesture)

        # Only update layers that meaningfully influenced the decision (contribution > 0.1).
        for layer_name, contribution in self._last_layer_contributions.items():
            if layer_name not in self._bandit:
                continue
            if contribution > 0.1:
                if correct:
                    # Reward the layer — it helped make the right call.
                    self._bandit[layer_name][0] = min(50.0,
                        self._bandit[layer_name][0] + contribution)
                else:
                    # Penalise at half rate — soft punishment so the AI stays flexible.
                    self._bandit[layer_name][1] = min(50.0,
                        self._bandit[layer_name][1] + contribution * 0.5)

    def _thompson_sample(self, layer_name):
        """
        Draw a weight for the given layer using an approximation of Thompson Sampling.

        True Thompson Sampling draws from a Beta distribution using scipy, but
        we approximate it cheaply: take the mean (alpha / (alpha + beta)) and add
        a small Gaussian noise term that shrinks as more data accumulates (more data
        = more confidence = less noise).

        Returns a multiplier in [0.1, 2.5] — higher means the AI trusts this layer more.
        """
        a, b = self._bandit.get(layer_name, [1.0, 1.0])
        mean  = a / (a + b)
        # Noise gets smaller as (a + b) grows — the AI becomes more decisive over time.
        noise = random.gauss(0, 0.08 / (a + b) ** 0.5)
        # Map mean into [0.1, 2.5] and clamp to that range.
        return max(0.1, min(2.5, mean * 2.0 + noise))

    def _blank_move_scores(self):
        """Return a score dict with each gesture starting at 1.0 — a uniform prior."""
        return {"Rock": 1.0, "Paper": 1.0, "Scissors": 1.0}

    def _blank_response_scores(self):
        """Return a score dict for the three response types: stay, upgrade, downgrade."""
        return {"stay": 0.0, "upgrade": 0.0, "downgrade": 0.0}

    def _recency_weight(self, distance_from_latest):
        """
        Compute how much weight an older round should carry.

        distance_from_latest = 0 means the most recent round (full weight = 1.0).
        Each step further back multiplies by 1/1.28, so:
          1 round ago  ≈ 0.78
          2 rounds ago ≈ 0.61
          3 rounds ago ≈ 0.48
        """
        return 1.0 / (1.28 ** distance_from_latest)

    def _get_response_type(self, previous_move, next_move):
        """
        Classify the transition from one gesture to the next:
          "stay"      — same gesture repeated
          "upgrade"   — moved forward in the Rock->Paper->Scissors cycle
          "downgrade" — moved backward in the cycle
        """
        if next_move == previous_move:
            return "stay"
        if UPGRADE_MOVE[previous_move] == next_move:
            return "upgrade"
        return "downgrade"

    def _apply_population_priors(self, last_outcome, response_scores):
        """
        Layer 1: soft baseline priors from well-documented human psychology.

        These are mild so the player's own history quickly overrides them.
        Think of it as the AI's opening assumption before it has seen enough data.

          After a win:  people tend to stay with what worked (~win-stay bias).
          After a loss: people tend to shift (upgrade or downgrade).
          After a draw: fairly neutral, slight preference for shifting.
          First round:  completely neutral — no prior information.
        """
        if last_outcome == "win":
            response_scores["stay"]      += 1.10
            response_scores["upgrade"]   += 0.90
            response_scores["downgrade"] += 0.90
        elif last_outcome == "lose":
            response_scores["stay"]      += 0.65
            response_scores["upgrade"]   += 1.15
            response_scores["downgrade"] += 1.15
        elif last_outcome == "draw":
            response_scores["stay"]      += 0.95
            response_scores["upgrade"]   += 1.00
            response_scores["downgrade"] += 1.00
        else:
            # First round or unknown — completely neutral.
            response_scores["stay"]      += 1.00
            response_scores["upgrade"]   += 1.00
            response_scores["downgrade"] += 1.00

    def _score_outcome_conditioned_responses(
        self, history, last_move, last_outcome, response_scores
    ):
        """
        Layer 2: learn how THIS player specifically responds to each outcome.

        We search history for every round that ended with the same outcome as now,
        then look at what the player did NEXT (stay / upgrade / downgrade).

        Recent examples count more (recency weighting). If the starting gesture
        also matches the current one, that example gets a 25% bonus — it's
        a closer match, so more relevant.
        """
        if len(history) < 2:
            return

        for i in range(len(history) - 1):
            prev      = history[i]
            next_round = history[i + 1]

            # Only care about rounds where the player was in the same situation (same outcome).
            if prev["player_outcome"] != last_outcome:
                continue

            response = self._get_response_type(prev["player_gesture"], next_round["player_gesture"])
            distance = (len(history) - 2) - i
            weight   = 2.2 * self._recency_weight(distance)

            # Extra relevance bonus if the starting gesture also matches.
            if prev["player_gesture"] == last_move:
                weight *= 1.25

            response_scores[response] += weight

    def _convert_response_scores_to_move_scores(self, last_move, response_scores, move_scores):
        """
        Translate stay / upgrade / downgrade probabilities into per-gesture scores.

        "stay"      -> the same gesture the player just threw
        "upgrade"   -> the next gesture up in the cycle from their last throw
        "downgrade" -> the gesture below in the cycle
        """
        move_scores[last_move]                  += response_scores["stay"]
        move_scores[UPGRADE_MOVE[last_move]]    += response_scores["upgrade"]
        move_scores[DOWNGRADE_MOVE[last_move]]  += response_scores["downgrade"]

    def _score_exact_transition_memory(
        self, history, last_move, last_outcome, move_scores
    ):
        """
        Layer 3: raw move-to-move memory, regardless of what outcome happened.

        If the player just threw Rock, scan history for every time they threw Rock
        and see what they threw next. Rounds where the outcome also matches
        get a 35% bonus — more context = more relevant example.
        """
        if len(history) < 2:
            return

        for i in range(len(history) - 1):
            prev      = history[i]
            next_round = history[i + 1]

            # Only look at rounds where the player started with the same gesture.
            if prev["player_gesture"] != last_move:
                continue

            distance = (len(history) - 2) - i
            weight   = 1.35 * self._recency_weight(distance)

            # Bonus when both the gesture AND the outcome match — doubly relevant.
            if prev["player_outcome"] == last_outcome:
                weight *= 1.35

            move_scores[next_round["player_gesture"]] += weight

    def _score_outcome_next_move_patterns(self, history, last_outcome, move_scores):
        """
        Layer 4: a broader, weaker helper layer.

        After a given outcome, what actual gesture tends to appear next —
        regardless of what was thrown in the previous round?
        This is noisier than layers 2 and 3 (it ignores the current gesture),
        so it carries a lower base weight of 0.70.
        """
        if len(history) < 2:
            return

        for i in range(len(history) - 1):
            prev      = history[i]
            next_round = history[i + 1]

            if prev["player_outcome"] != last_outcome:
                continue

            distance = (len(history) - 2) - i
            weight   = 0.70 * self._recency_weight(distance)
            move_scores[next_round["player_gesture"]] += weight

    def _score_overall_frequency(self, history, move_scores):
        """
        Layer 5: overall gesture frequency fallback.

        When outcome-specific patterns are weak (e.g. only a few rounds played),
        at least lean toward the gestures the player throws most often overall.
        Weighted by recency so the AI doesn't get stuck on an old favourite.
        Low base weight (0.30) so it doesn't overshadow smarter layers.
        """
        # Iterate from newest to oldest so the most recent round gets distance = 0.
        for idx, record in enumerate(reversed(history)):
            move_scores[record["player_gesture"]] += 0.30 * self._recency_weight(idx)

    def _predict_player_scores(self, history):
        """
        Run all six prediction layers and return a score dict for each gesture.

        Higher score means the AI thinks the player is more likely to throw that gesture.
        The caller then picks COUNTER_MOVE[highest-scored gesture] as the robot's throw.

        Thompson Sampling weights scale each layer's contribution based on past accuracy.
        Personality layer_bias multipliers are applied on top of those weights.
        """
        scores = self._blank_move_scores()

        if not history:
            return scores

        last_round   = history[-1]
        last_move    = last_round["player_gesture"]
        last_outcome = last_round["player_outcome"]

        response_scores = self._blank_response_scores()

        # Sample a weight for each layer from its Beta distribution.
        # Layers that have been more accurate will have higher alpha/beta ratios
        # and thus tend to produce larger weights.
        w_outcome    = self._thompson_sample("outcome")
        w_transition = self._thompson_sample("transition")
        w_opp_next   = self._thompson_sample("opp_next")
        w_frequency  = self._thompson_sample("frequency")
        w_markov     = self._thompson_sample("markov")

        # Apply personality-specific biases on top of the bandit weights.
        # e.g. The Psychologist multiplies outcome by 2.5 and halves transition.
        pb = self._p_layer_bias
        w_outcome    *= pb.get("outcome",    1.0)
        w_transition *= pb.get("transition", 1.0)
        w_frequency  *= pb.get("frequency",  1.0)
        w_markov     *= pb.get("markov",     1.0)

        # --- Layer 1: population priors (no bandit weight — always applied as-is) ---
        self._apply_population_priors(last_outcome, response_scores)

        # --- Layer 2: outcome-conditioned response learning ---
        # Compute into a separate dict, then merge scaled by w_outcome.
        rs2 = self._blank_response_scores()
        self._score_outcome_conditioned_responses(
            history=history, last_move=last_move,
            last_outcome=last_outcome, response_scores=rs2)
        for k in rs2:
            response_scores[k] += rs2[k] * w_outcome

        # Convert the combined response scores into per-gesture move scores.
        self._convert_response_scores_to_move_scores(
            last_move=last_move, response_scores=response_scores, move_scores=scores)

        # --- Layer 3: exact transition memory ---
        # Snapshot scores before and after so we can isolate this layer's delta,
        # then re-scale that delta by the bandit weight.
        pre3 = dict(scores)
        self._score_exact_transition_memory(
            history=history, last_move=last_move,
            last_outcome=last_outcome, move_scores=scores)
        for g in VALID_GESTURES:
            delta = scores[g] - pre3[g]
            scores[g] = pre3[g] + delta * w_transition

        # --- Layer 4: outcome -> next-move tendencies ---
        pre4 = dict(scores)
        self._score_outcome_next_move_patterns(
            history=history, last_outcome=last_outcome, move_scores=scores)
        for g in VALID_GESTURES:
            delta = scores[g] - pre4[g]
            scores[g] = pre4[g] + delta * w_opp_next

        # --- Layer 5: overall frequency fallback ---
        pre5 = dict(scores)
        self._score_overall_frequency(history, scores)
        for g in VALID_GESTURES:
            delta = scores[g] - pre5[g]
            scores[g] = pre5[g] + delta * w_frequency

        # --- Layer 6: Markov WIN/LOSE/DRAW tables ---
        pre6 = dict(scores)
        _markov_move_scores(history, scores)
        for g in VALID_GESTURES:
            delta = scores[g] - pre6[g]
            scores[g] = pre6[g] + delta * w_markov

        # --- Opponent-type bias (runs after bandit layers, not tuned by Thompson Sampling) ---
        # Adds a deterministic boost based on the detected playstyle pattern.
        opp_type = _detect_opponent_type(history)
        if opp_type.endswith("_heavy"):
            # Player is over-using one gesture — score that gesture higher.
            heavy_gesture = opp_type.replace("_heavy", "").capitalize()
            scores[heavy_gesture] += 1.5
        elif opp_type == "cycler":
            # Predict the next step in the Rock -> Paper -> Scissors cycle.
            cycle = ("Rock", "Paper", "Scissors")
            if last_move in cycle:
                predicted_next = cycle[(cycle.index(last_move) + 1) % 3]
                scores[predicted_next] += 2.0
        elif opp_type == "win_stay" and last_outcome == "win":
            # Win-stay player just won — they'll very likely repeat the same gesture.
            scores[last_move] += 2.0

        # Save layer weights so update_bandit() can reward/penalise them next round.
        self._last_layer_contributions = {
            "outcome":    w_outcome,
            "transition": w_transition,
            "opp_next":   w_opp_next,
            "frequency":  w_frequency,
            "markov":     w_markov,
        }

        return scores

    def _weighted_choice(self, score_dict):
        """
        Pick a gesture randomly, weighted by its score.

        Every gesture gets at least 0.001 weight so we never completely rule
        one out. Falls back to uniform random if the loop somehow doesn't
        pick anything (shouldn't happen in practice).
        """
        total   = sum(max(v, 0.001) for v in score_dict.values())
        pick    = random.uniform(0, total)
        current = 0.0

        for move, score in score_dict.items():
            current += max(score, 0.001)
            if current >= pick:
                return move

        # Safety net — should never reach this line.
        return random.choice(VALID_GESTURES)

    def choose_robot_move(self, history, round_number):
        """
        Main entry point: decide what gesture the robot should throw this round.

        Decision flow:
          1. First round / no history          -> pure random (nothing to predict yet)
          2. Grace period                       -> near-random with gradually rising skill
          3. Chaos Agent personality            -> pure 33/33/33 Nash equilibrium
          4. Ghost personality (use_delayed)    -> echo the player's last gesture
          5. Gambler wild roll                  -> fully random on some rounds
          6. Standard path                      -> run all six prediction layers,
                                                   apply a skill roll, use miss mode
                                                   if the roll fails, return the counter-move

        Returns the gesture string the robot should throw ("Rock", "Paper", or "Scissors").
        """
        # Scale the grace period length by the personality's grace multiplier.
        eff_grace = int(self.grace_rounds * self._p_grace_mult)

        # Round 1 or no history — no data to analyse, just throw randomly.
        if round_number <= 1 or not history:
            self.last_prediction = {
                "top_predicted_move":  None,
                "used_predicted_move": None,
                "effective_skill":     None,
                "opponent_type":       "unknown",
            }
            return random.choice(VALID_GESTURES)

        # During the grace period, skill ramps up gradually from a floor of 0.30.
        # This gives the human time to warm up before the AI starts reading them.
        if round_number <= eff_grace:
            grace_skill = max(0.30, self.base_skill * (round_number / max(eff_grace, 1)))
            move = random.choice(VALID_GESTURES)
            self.last_prediction = {
                "top_predicted_move":  None,
                "used_predicted_move": move,
                "effective_skill":     grace_skill,
                "opponent_type":       "grace_period",
                "personality":         self.personality,
            }
            return move

        # Chaos Agent: skip all prediction — always play pure Nash equilibrium.
        if self._p_miss_mode == "chaos":
            move = random.choice(VALID_GESTURES)
            self.last_prediction = {
                "top_predicted_move":  None,
                "used_predicted_move": move,
                "effective_skill":     0.333,
                "opponent_type":       "nash",
                "personality":         self.personality,
            }
            return move

        # Ghost: throw back the player's last gesture (mirror, NOT the counter of it).
        # INTENTIONAL: move = last_player_g, not COUNTER_MOVE[last_player_g].
        if self._p_use_delayed and history:
            last_player_g = history[-1].get("player_gesture")
            if last_player_g in VALID_GESTURES:
                self.last_prediction = {
                    "top_predicted_move":  last_player_g,
                    "used_predicted_move": last_player_g,
                    "effective_skill":     0.55,
                    "opponent_type":       "ghost",
                    "personality":         self.personality,
                }
                return last_player_g

        # Gambler: occasionally ignore all strategy and throw randomly.
        if self._p_wild_rate > 0 and random.random() < self._p_wild_rate:
            move = random.choice(VALID_GESTURES)
            self.last_prediction = {
                "top_predicted_move":  None,
                "used_predicted_move": move,
                "effective_skill":     0.33,
                "opponent_type":       "wild",
                "personality":         self.personality,
            }
            return move

        # --- Standard prediction path ---

        scores = self._predict_player_scores(history)

        # Sort gestures by prediction score, highest first.
        ranked      = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_move   = ranked[0][0]
        best_score  = ranked[0][1]
        second_move  = ranked[1][0] if len(ranked) > 1 else random.choice(VALID_GESTURES)
        second_score = ranked[1][1] if len(ranked) > 1 else best_score

        # Skill grows as history accumulates, capped at max_skill, then scaled by personality.
        effective_skill = min(
            self.max_skill,
            self.base_skill + 0.02 * len(history)
        ) * self._p_skill_mult

        # Rubber-band adjustment: ease off if the AI is on a long win streak,
        # push harder if the player is on one. Keeps the game engaging.
        if self._consecutive_wins >= 5:
            # AI winning too much — dial back skill a little.
            effective_skill = max(0.30, effective_skill - 0.04 * min(self._consecutive_wins - 4, 4))
        elif self._consecutive_losses >= 5:
            # Player winning a lot — AI pushes harder.
            effective_skill = min(self.max_skill + 0.06,
                                  effective_skill + 0.03 * min(self._consecutive_losses - 4, 4))

        # Update the consecutive win/loss counters for next round's rubber-band check.
        last_outcome = history[-1].get("player_outcome", "")
        if last_outcome == "win":
            self._consecutive_wins  += 1
            self._consecutive_losses = 0
        elif last_outcome == "lose":
            self._consecutive_losses += 1
            self._consecutive_wins   = 0
        else:
            # Draw or unknown — reset both streaks.
            self._consecutive_wins   = 0
            self._consecutive_losses = 0

        # If the top two scores are very close, the AI is genuinely uncertain —
        # reduce skill slightly to reflect that uncertainty.
        if (best_score - second_score) < 0.40:
            effective_skill -= 0.08

        # Never let skill drop below the minimum floor.
        effective_skill = max(0.35, effective_skill)

        # The "skill roll": if random() falls below effective_skill, play optimally.
        # Otherwise "miss" — the AI deliberately plays sub-optimally to stay fair.
        if random.random() < effective_skill:
            predicted_player_move = best_move
        else:
            # Miss mode determines HOW the AI plays when it's not going optimal.
            if self._p_miss_mode == "second":
                # Pick the second-best prediction — still fairly informed.
                predicted_player_move = second_move
            elif self._p_miss_mode == "modal" and history:
                # Pick the player's most-thrown gesture overall.
                gestures = [r["player_gesture"] for r in history if r["player_gesture"] in VALID_GESTURES]
                predicted_player_move = max(set(gestures), key=gestures.count) if gestures else best_move
            else:
                # Default: pick randomly, still weighted by the prediction scores.
                predicted_player_move = self._weighted_choice(scores)

        opp_type = _detect_opponent_type(history)
        self.last_prediction = {
            "top_predicted_move":  best_move,
            "used_predicted_move": predicted_player_move,
            "effective_skill":     round(effective_skill, 4),
            "opponent_type":       opp_type,
            "personality":         self.personality,
        }

        # Return the gesture that BEATS the predicted player move.
        return COUNTER_MOVE[predicted_player_move]

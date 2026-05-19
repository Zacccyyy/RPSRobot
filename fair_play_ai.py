# ============================================================
# fair_play_ai.py
#
# The core AI "brain" used by Fair Play mode.
#
# This module contains:
#   - FairPlayAI: a multi-layer, adaptive prediction engine
#     that tries to read the human player's patterns and pick
#     the counter-move.
#   - Support utilities: outcome lookup tables, difficulty presets,
#     personality configs, opponent-type detection, and a
#     recency-weighted Markov layer.
#
# Where it fits:
#   fair_play_state.py creates a FairPlayAI instance and calls
#   choose_robot_move() once per round after beat 3.
#   challenge_ai.py subclasses FairPlayAI, so changes here
#   affect Challenge mode too.
# ============================================================

import random
from collections import defaultdict


# The three legal gestures — used throughout for validation.
VALID_GESTURES = ("Rock", "Paper", "Scissors")

# What beats what: key beats value (used to pick the winning counter-move).
COUNTER_MOVE = {
    "Rock": "Paper",
    "Paper": "Scissors",
    "Scissors": "Rock",
}

# What gesture "upgrades" to next in the Rock→Paper→Scissors order.
# Used when tracking whether the player escalated after a round.
UPGRADE_MOVE = {
    "Rock": "Paper",
    "Paper": "Scissors",
    "Scissors": "Rock",
}

# The reverse — what gesture "downgrades" to the previous in cycle.
# Paired with UPGRADE_MOVE so every transition is fully described.
DOWNGRADE_MOVE = {
    "Rock": "Scissors",
    "Paper": "Rock",
    "Scissors": "Paper",
}

# Difficulty presets: how aggressive the AI starts, how high it can grow,
# and how many grace rounds it plays near-randomly at the start.
# DO NOT change these values — they are calibrated.
_DIFFICULTY_PRESETS = {
    "Easy":   {"base_skill": 0.40, "max_skill": 0.55, "grace_rounds": 20},
    "Normal": {"base_skill": 0.66, "max_skill": 0.76, "grace_rounds": 10},
    "Hard":   {"base_skill": 0.80, "max_skill": 0.92, "grace_rounds":  0},
}

# ── AI Personality presets ─────────────────────────────────────────────────────
# Each personality tweaks how the AI predicts, how quickly it learns, and how
# it "misses" on purpose to stay fair.
#
# "layer_bias" multiplies each prediction layer's contribution.
# "miss_mode" controls what the AI throws when it decides NOT to play optimally:
#   "random"  — pick uniformly from all gestures (default)
#   "second"  — pick second-best prediction (still smart)
#   "chaos"   — pure 33/33/33 Nash equilibrium (statistically unbeatable)
#   "modal"   — always copy the player's most-common gesture
#   "delayed" — echo the player's last gesture (one round behind)
PERSONALITIES = {
    "Normal": {
        "label": "Normal",
        "desc": "Balanced adaptive play. The default.",
        "skill_mult":    1.0,
        "grace_mult":    1.0,
        "layer_bias":    {},   # no overrides
        "miss_mode":     "random",
        "bluff_rate":    0.0,  # no extra bluffing
    },
    "The Psychologist": {
        "label": "The Psychologist",
        "desc": "Exploits win-stay/lose-shift biases. Doubles down on outcome layers.",
        "skill_mult":    1.05,
        "grace_mult":    0.8,
        "layer_bias":    {"outcome": 2.5, "transition": 0.6, "frequency": 0.3},
        "miss_mode":     "second",
        "bluff_rate":    0.0,
    },
    "The Gambler": {
        "label": "The Gambler",
        "desc": "High variance. Occasionally ignores all patterns and plays wild.",
        "skill_mult":    0.9,
        "grace_mult":    1.2,
        "layer_bias":    {"outcome": 0.8, "transition": 0.8},
        "miss_mode":     "random",
        "bluff_rate":    0.0,
        "wild_rate":     0.20,  # 20% chance of fully random move
    },
    "The Mirror": {
        "label": "The Mirror",
        "desc": "Copies your most common gesture. Blatantly exploitable if you adapt.",
        "skill_mult":    0.6,
        "grace_mult":    2.0,
        "layer_bias":    {"frequency": 4.0, "outcome": 0.2, "transition": 0.2},
        "miss_mode":     "modal",
        "bluff_rate":    0.0,
    },
    "The Ghost": {
        # INTENTIONAL: move = last_player_g is an echo, NOT a counter.
        # The Ghost copies your last throw — it does not beat it.
        "label": "The Ghost",
        "desc": "Plays your previous move back at you. One step behind, always.",
        "skill_mult":    0.55,
        "grace_mult":    2.0,
        "layer_bias":    {},
        "miss_mode":     "delayed",
        "bluff_rate":    0.0,
        "use_delayed":   True,
    },
    "The Chaos Agent": {
        "label": "The Chaos Agent",
        "desc": "Pure Nash equilibrium. Unbeatable in theory, unreadable in practice.",
        "skill_mult":    0.0,   # always 'misses' intentionally
        "grace_mult":    0.0,
        "layer_bias":    {},
        "miss_mode":     "chaos",
        "bluff_rate":    0.0,
    },
    "The Hustler": {
        "label": "The Hustler",
        "desc": "Hard-reads your patterns early, then plays dumb when it's winning.",
        "skill_mult":    1.12,
        "grace_mult":    0.5,   # learns fast
        "layer_bias":    {"transition": 2.0, "markov": 2.0, "outcome": 1.2},
        "miss_mode":     "second",
        "bluff_rate":    0.0,
    },
}

# Flat list of personality names — used by UI menus to iterate options.
PERSONALITY_NAMES = list(PERSONALITIES.keys())


def _detect_opponent_type(history):
    """
    Classify the human player's playstyle from their round history.

    We look at the last N rounds and check for recognisable patterns:
      - rock_heavy / paper_heavy / scissors_heavy: over 50% use of one gesture
      - win_stay: player repeats the same move after winning more than 60% of the time
      - cycler: player follows Rock→Paper→Scissors in order more than 65% of the time
      - random: no pattern found (default fallback)

    Returns a string label that the main AI uses to add a bias score.
    Needs at least 8 rounds of history before it bothers trying.
    """
    if len(history) < 8:
        # Not enough data — just call it random.
        return "random"

    # Pull out only the valid gesture names (skip any "Unknown" entries).
    gestures = [r["player_gesture"] for r in history if r["player_gesture"] in VALID_GESTURES]
    if not gestures:
        return "random"

    total = len(gestures)
    # Calculate what fraction of throws was each gesture.
    freq = {g: gestures.count(g) / total for g in VALID_GESTURES}

    # If any single gesture is used more than half the time, flag it as heavy.
    for g, f in freq.items():
        if f > 0.50:
            return f"{g.lower()}_heavy"

    # Win-stay check: count how often the player repeated after a win.
    wins = [r for r in history if r.get("player_outcome") == "win"]
    win_stay_count = 0
    for i, r in enumerate(history[:-1]):
        if r.get("player_outcome") == "win":
            # If the very next round uses the same gesture, that's a "stay".
            if history[i + 1]["player_gesture"] == r["player_gesture"]:
                win_stay_count += 1
    if wins and win_stay_count / max(len(wins), 1) > 0.60:
        return "win_stay"

    # Cycler check: look at the last 9 rounds for R→P→S progressions.
    cycle = ("Rock", "Paper", "Scissors")
    cycle_hits = 0
    recent = gestures[-9:]
    for i in range(len(recent) - 1):
        # What gesture would come next in the cycle after recent[i]?
        expected_next = cycle[(cycle.index(recent[i]) + 1) % 3] if recent[i] in cycle else None
        if expected_next and recent[i + 1] == expected_next:
            cycle_hits += 1
    if len(recent) > 1 and cycle_hits / (len(recent) - 1) > 0.65:
        return "cycler"

    return "random"


def _markov_move_scores(history, move_scores):
    """
    Layer 6: WIN/LOSE/TIE Markov transition tables.

    The idea: after a win, players tend to behave differently than after a loss.
    This builds three separate tables — one for each outcome — that track
    what gesture the player tends to throw NEXT given their last gesture AND
    their last outcome.

    For example: if the player lost while throwing Rock, this layer might
    notice they usually switch to Paper next.

    Recency weighting means recent transitions count more than old ones
    (controlled by the 1.28 base, same formula used elsewhere).

    Inspired by iamvigneshwars/rock-paper-scissors-ai.
    Adds directly into move_scores (no return value needed).
    """
    if len(history) < 3:
        # Need at least a couple of transitions to be meaningful.
        return

    # Three tables: win/lose/draw, each mapping gesture→gesture→weight.
    tables = {
        "win":  defaultdict(lambda: defaultdict(float)),
        "lose": defaultdict(lambda: defaultdict(float)),
        "draw": defaultdict(lambda: defaultdict(float)),
    }

    # Walk through history pairs and record weighted transitions.
    for i in range(len(history) - 1):
        outcome = history[i].get("player_outcome", "")
        g_from  = history[i].get("player_gesture", "")
        g_to    = history[i + 1].get("player_gesture", "")
        if outcome in tables and g_from in VALID_GESTURES and g_to in VALID_GESTURES:
            # Older rounds get a smaller weight — the formula is the same
            # as _recency_weight() in the main class (1.28 ^ -distance).
            distance = (len(history) - 2) - i
            weight = 1.0 / (1.28 ** distance)
            tables[outcome][g_from][g_to] += weight

    # Now look at the most recent round and score what the player will likely do next.
    last = history[-1]
    outcome = last.get("player_outcome", "")
    g_from  = last.get("player_gesture", "")

    if outcome in tables and g_from in tables[outcome]:
        row = tables[outcome][g_from]
        total = sum(row.values())
        if total > 0:
            # Normalise to probabilities, then scale by 1.8 (calibrated contribution weight).
            for g_to, w in row.items():
                move_scores[g_to] += 1.8 * (w / total)


class FairPlayAI:
    """
    Fair Play AI v3

    Predicts the player's next move using six stacked layers, then
    returns the move that beats the prediction.

    Strategy layers (applied in order):
      1. Soft population priors — baseline human psychology tendencies
      2. Outcome-conditioned response learning — stay/upgrade/downgrade after win/lose/draw
      3. Exact transition memory — move-to-move tendencies regardless of outcome
      4. Outcome→next-move patterns — a weaker version of layer 2
      5. Overall frequency fallback — which gestures appear most overall
      6. WIN/LOSE/TIE Markov tables — combines outcome AND transition into one table

    Each layer's contribution is weighted by Thompson Sampling (a Bayesian
    bandit that learns which layers are actually accurate for this player).

    Personality presets (see PERSONALITIES above) let each AI character
    bias these layers differently — The Psychologist leans on outcome layers,
    The Gambler adds random wild throws, etc.

    Difficulty presets control base skill, max skill ceiling, and grace period.
    """

    def __init__(self, base_skill=0.66, max_skill=0.76, difficulty="Normal",
                 personality="Normal"):
        # Load the preset for the chosen difficulty (Easy/Normal/Hard).
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
        Apply a personality preset by name.
        Caches all relevant personality fields as underscore-prefixed attributes
        so the prediction methods can access them quickly.
        """
        p = PERSONALITIES.get(name, PERSONALITIES["Normal"])
        self.personality      = name
        self._p_skill_mult    = p.get("skill_mult", 1.0)
        self._p_grace_mult    = p.get("grace_mult", 1.0)
        self._p_layer_bias    = p.get("layer_bias", {})
        self._p_miss_mode     = p.get("miss_mode", "random")
        self._p_wild_rate     = p.get("wild_rate", 0.0)
        self._p_use_delayed   = p.get("use_delayed", False)

    def set_difficulty(self, difficulty):
        """Swap difficulty mid-session without resetting learned history."""
        preset = _DIFFICULTY_PRESETS.get(difficulty, _DIFFICULTY_PRESETS["Normal"])
        self.base_skill   = preset["base_skill"]
        self.max_skill    = preset["max_skill"]
        self.grace_rounds = preset["grace_rounds"]
        self.difficulty   = difficulty

    def reset(self):
        """
        Clear everything learned from the current match.
        Called at match start and when the player restarts.
        """
        self.last_prediction = None
        self._consecutive_wins   = 0
        self._consecutive_losses = 0

        # Thompson Sampling bandit: one Beta(alpha, beta) distribution per layer.
        # Starting at (1, 1) = flat/non-informative prior — the AI starts unbiased.
        # Alpha grows when the layer predicts correctly; beta when it's wrong.
        self._bandit = {
            "outcome":    [1.0, 1.0],   # [alpha_successes, beta_failures]
            "transition": [1.0, 1.0],
            "opp_next":   [1.0, 1.0],
            "frequency":  [1.0, 1.0],
            "markov":     [1.0, 1.0],
        }
        # Stores each layer's contribution weight from the last round so
        # update_bandit() can reward or penalise the right layers.
        self._last_layer_contributions = {}

    def update_bandit(self, ai_predicted_gesture, actual_player_gesture):
        """
        Call after each resolved round to update the Thompson Sampling bandit.

        ai_predicted_gesture: what gesture the AI thought the player would throw
        actual_player_gesture: what the player actually threw

        If the prediction was correct, the layers that contributed most get their
        alpha (success count) bumped. If wrong, their beta (failure count) is bumped
        (at half weight so mistakes don't punish too harshly).

        Values are capped at 50.0 to prevent any layer from becoming permanently dominant.
        """
        if not self._last_layer_contributions:
            return

        correct = (ai_predicted_gesture == actual_player_gesture)

        for layer_name, contribution in self._last_layer_contributions.items():
            if layer_name not in self._bandit:
                continue
            # Only update layers that meaningfully influenced the decision.
            if contribution > 0.1:
                if correct:
                    self._bandit[layer_name][0] = min(50.0,
                        self._bandit[layer_name][0] + contribution)
                else:
                    self._bandit[layer_name][1] = min(50.0,
                        self._bandit[layer_name][1] + contribution * 0.5)

    def _thompson_sample(self, layer_name):
        """
        Draw a sample from Beta(alpha, beta) for the given layer.

        In true Thompson Sampling you'd draw from scipy.stats.beta, but
        here we approximate it cheaply: mean plus a small Gaussian noise term
        that shrinks as the layer accumulates more data (more data = more confident).

        Returns a multiplier in [0.1, 2.5] — higher means the AI trusts this layer more.
        """
        a, b = self._bandit.get(layer_name, [1.0, 1.0])
        mean = a / (a + b)
        # Noise scale shrinks as total observations grow — the AI becomes more decisive.
        noise = random.gauss(0, 0.08 / (a + b) ** 0.5)
        # Scale mean up to [0.1, 2.5] range and clamp.
        return max(0.1, min(2.5, mean * 2.0 + noise))

    def _blank_move_scores(self):
        """Return a fresh score dict with each gesture starting at 1.0 (uniform prior)."""
        return {
            "Rock": 1.0,
            "Paper": 1.0,
            "Scissors": 1.0,
        }

    def _blank_response_scores(self):
        """Return a fresh response-type score dict (stay / upgrade / downgrade)."""
        return {
            "stay": 0.0,
            "upgrade": 0.0,
            "downgrade": 0.0,
        }

    def _recency_weight(self, distance_from_latest):
        """
        Decay weight for older rounds.
        distance_from_latest = 0 → most recent pattern (weight = 1.0).
        Each step back multiplies by 1/1.28, so two rounds ago ≈ 0.78, three ≈ 0.61.
        """
        return 1.0 / (1.28 ** distance_from_latest)

    def _get_response_type(self, previous_move, next_move):
        """
        Given two consecutive gestures, classify the transition:
          "stay"      — same gesture repeated
          "upgrade"   — moved up the cycle (Rock→Paper, Paper→Scissors, Scissors→Rock)
          "downgrade" — moved down the cycle (the other direction)
        """
        if next_move == previous_move:
            return "stay"
        if UPGRADE_MOVE[previous_move] == next_move:
            return "upgrade"
        return "downgrade"

    def _apply_population_priors(self, last_outcome, response_scores):
        """
        Layer 1: soft baseline priors based on well-documented human tendencies.

        These are intentionally mild so the player's own behaviour
        quickly overrides them. Think of it as the AI's starting assumption
        before it has seen enough data.

        After a win: people tend to stay with what worked.
        After a loss: people tend to shift (upgrade or downgrade).
        After a draw: fairly neutral.
        """
        if last_outcome == "lose":
            response_scores["stay"] += 0.65
            response_scores["upgrade"] += 1.15
            response_scores["downgrade"] += 1.15

        elif last_outcome == "win":
            response_scores["stay"] += 1.10
            response_scores["upgrade"] += 0.90
            response_scores["downgrade"] += 0.90

        elif last_outcome == "draw":
            response_scores["stay"] += 0.95
            response_scores["upgrade"] += 1.00
            response_scores["downgrade"] += 1.00

        else:
            # First round / unknown outcome — completely neutral.
            response_scores["stay"] += 1.00
            response_scores["upgrade"] += 1.00
            response_scores["downgrade"] += 1.00

    def _score_outcome_conditioned_responses(
        self,
        history,
        last_move,
        last_outcome,
        response_scores
    ):
        """
        Layer 2: learn how THIS player responds after a given outcome.

        We search history for every round that ended with the same outcome
        as the current one, then look at what the player did NEXT.

        Example: if the player just lost, find past rounds where they also
        lost and check whether they stayed, upgraded, or downgraded.
        Recent examples are weighted more heavily; examples where the
        starting gesture also matches get a 25% bonus.
        """
        if len(history) < 2:
            return

        for i in range(len(history) - 1):
            previous_round = history[i]
            next_round = history[i + 1]

            # Only look at rounds with the same outcome as the current situation.
            if previous_round["player_outcome"] != last_outcome:
                continue

            observed_response = self._get_response_type(
                previous_round["player_gesture"],
                next_round["player_gesture"]
            )

            distance = (len(history) - 2) - i
            weight = 2.2 * self._recency_weight(distance)

            # If the historical starting gesture also matches the current one,
            # this example is especially relevant — boost its weight.
            if previous_round["player_gesture"] == last_move:
                weight *= 1.25

            response_scores[observed_response] += weight

    def _convert_response_scores_to_move_scores(
        self,
        last_move,
        response_scores,
        move_scores
    ):
        """
        Translate stay/upgrade/downgrade probabilities into per-gesture scores.
        Each response type maps directly to a target gesture from the player's last move.
        """
        move_scores[last_move] += response_scores["stay"]
        move_scores[UPGRADE_MOVE[last_move]] += response_scores["upgrade"]
        move_scores[DOWNGRADE_MOVE[last_move]] += response_scores["downgrade"]

    def _score_exact_transition_memory(
        self,
        history,
        last_move,
        last_outcome,
        move_scores
    ):
        """
        Layer 3: raw move-to-move memory, ignoring outcome.

        If the player just threw Rock, scan history for every time they
        previously threw Rock and see what they threw next.
        Rounds where the outcome also matches get a 35% bonus — more context
        = more relevant example.
        """
        if len(history) < 2:
            return

        for i in range(len(history) - 1):
            previous_round = history[i]
            next_round = history[i + 1]

            # Only look at rounds where the player started with the same gesture.
            if previous_round["player_gesture"] != last_move:
                continue

            distance = (len(history) - 2) - i
            weight = 1.35 * self._recency_weight(distance)

            # Bonus when both the starting move AND the outcome match.
            if previous_round["player_outcome"] == last_outcome:
                weight *= 1.35

            move_scores[next_round["player_gesture"]] += weight

    def _score_outcome_next_move_patterns(self, history, last_outcome, move_scores):
        """
        Layer 4: a weaker helper layer.

        After a given outcome, what actual gesture tends to appear next —
        regardless of what was thrown in the previous round?
        This is broader and noisier than layers 2 and 3, hence the lower base weight (0.70).
        """
        if len(history) < 2:
            return

        for i in range(len(history) - 1):
            previous_round = history[i]
            next_round = history[i + 1]

            if previous_round["player_outcome"] != last_outcome:
                continue

            distance = (len(history) - 2) - i
            weight = 0.70 * self._recency_weight(distance)
            move_scores[next_round["player_gesture"]] += weight

    def _score_overall_frequency(self, history, move_scores):
        """
        Layer 5: overall gesture frequency fallback.

        If outcome-specific patterns are weak (e.g. the player just started),
        at least lean toward the gestures they throw most often.
        Weighted by recency so the AI doesn't get stuck on an old favourite.
        Low base weight (0.30) so it doesn't drown out the smarter layers.
        """
        # Iterate from newest to oldest so distance = 0 for the latest round.
        for idx, record in enumerate(reversed(history)):
            move = record["player_gesture"]
            move_scores[move] += 0.30 * self._recency_weight(idx)

    def _predict_player_scores(self, history):
        """
        Run all six prediction layers and return a score dict for each gesture.

        Higher score = AI thinks the player is more likely to throw that gesture.
        The caller then picks the gesture that beats the highest-scored prediction.

        Thompson Sampling weights adjust each layer's contribution dynamically
        based on how accurate that layer has been for this specific player.
        Personality layer_bias multipliers are applied on top.
        """
        scores = self._blank_move_scores()

        if not history:
            return scores

        last_round   = history[-1]
        last_move    = last_round["player_gesture"]
        last_outcome = last_round["player_outcome"]

        response_scores = self._blank_response_scores()

        # Sample a weight for each layer from its Beta distribution.
        # Higher alpha/beta ratio = layer has been more accurate = higher weight.
        w_outcome    = self._thompson_sample("outcome")
        w_transition = self._thompson_sample("transition")
        w_opp_next   = self._thompson_sample("opp_next")
        w_frequency  = self._thompson_sample("frequency")
        w_markov     = self._thompson_sample("markov")

        # Apply personality biases on top of the learned bandit weights.
        # E.g. The Psychologist doubles the outcome layer, halves transition.
        pb = self._p_layer_bias
        w_outcome    *= pb.get("outcome",    1.0)
        w_transition *= pb.get("transition", 1.0)
        w_frequency  *= pb.get("frequency",  1.0)
        w_markov     *= pb.get("markov",     1.0)

        # Layer 1: population priors — always applied, no bandit weight.
        self._apply_population_priors(last_outcome, response_scores)

        # Layer 2: outcome-conditioned response learning.
        # Compute into a separate dict, then add to response_scores scaled by w_outcome.
        rs2 = self._blank_response_scores()
        self._score_outcome_conditioned_responses(
            history=history, last_move=last_move,
            last_outcome=last_outcome, response_scores=rs2)
        for k in rs2:
            response_scores[k] += rs2[k] * w_outcome

        # Convert response scores (stay/upgrade/downgrade) into per-gesture scores.
        self._convert_response_scores_to_move_scores(
            last_move=last_move, response_scores=response_scores,
            move_scores=scores)

        # Layer 3: exact transition memory, scaled by bandit weight.
        # We snapshot scores before and after to isolate this layer's contribution.
        pre3 = {g: scores[g] for g in VALID_GESTURES}
        self._score_exact_transition_memory(
            history=history, last_move=last_move,
            last_outcome=last_outcome, move_scores=scores)
        contrib3 = sum(max(0, scores[g] - pre3[g]) for g in VALID_GESTURES)
        # Re-scale: undo the raw contribution and apply the bandit weight.
        if contrib3 > 0:
            for g in VALID_GESTURES:
                delta = scores[g] - pre3[g]
                scores[g] = pre3[g] + delta * w_transition

        # Layer 4: outcome→next-move tendencies, scaled by bandit weight.
        pre4 = {g: scores[g] for g in VALID_GESTURES}
        self._score_outcome_next_move_patterns(
            history=history, last_outcome=last_outcome, move_scores=scores)
        # Always apply — the `if True` in the original was intentional scaffolding.
        for g in VALID_GESTURES:
            delta = scores[g] - pre4[g]
            scores[g] = pre4[g] + delta * w_opp_next

        # Layer 5: overall frequency fallback, scaled by bandit weight.
        pre5 = {g: scores[g] for g in VALID_GESTURES}
        self._score_overall_frequency(history, scores)
        for g in VALID_GESTURES:
            delta = scores[g] - pre5[g]
            scores[g] = pre5[g] + delta * w_frequency

        # Layer 6: Markov WIN/LOSE/TIE tables, scaled by bandit weight.
        pre6 = {g: scores[g] for g in VALID_GESTURES}
        _markov_move_scores(history, scores)
        for g in VALID_GESTURES:
            delta = scores[g] - pre6[g]
            scores[g] = pre6[g] + delta * w_markov

        # Opponent-type bias: deterministic boost based on detected playstyle.
        # This runs after the bandit layers so it's not tuned by Thompson Sampling.
        opp_type = _detect_opponent_type(history)
        if opp_type.endswith("_heavy"):
            # Player is over-using one gesture — score it higher.
            heavy_gesture = opp_type.replace("_heavy", "").capitalize()
            scores[heavy_gesture] += 1.5
        elif opp_type == "cycler":
            # Predict the next step in the Rock→Paper→Scissors cycle.
            cycle = ("Rock", "Paper", "Scissors")
            if last_move in cycle:
                predicted_next = cycle[(cycle.index(last_move) + 1) % 3]
                scores[predicted_next] += 2.0
        elif opp_type == "win_stay" and last_outcome == "win":
            # Win-stay player just won — they'll likely throw the same again.
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

        Ensures every gesture has at least a tiny chance (min 0.001) so we
        never completely rule out a gesture. Falls back to uniform random if
        the loop somehow doesn't pick anything (shouldn't happen in practice).
        """
        total = sum(max(v, 0.001) for v in score_dict.values())
        pick = random.uniform(0, total)
        current = 0.0

        for move, score in score_dict.items():
            current += max(score, 0.001)
            if current >= pick:
                return move

        # Safety fallback — should never reach here.
        return random.choice(VALID_GESTURES)

    def choose_robot_move(self, history, round_number):
        """
        Main entry point: decide what the robot should throw.

        Flow:
          1. First round / no history → pure random (nothing to predict yet).
          2. Grace period → near-random with gradually rising skill.
          3. Chaos Agent → pure 33/33/33 Nash equilibrium.
          4. Ghost → echo the player's last gesture (NOT counter it).
          5. Gambler wild roll → fully random on some rounds.
          6. Standard path → run all prediction layers, apply skill roll,
             pick miss mode if the roll fails, return COUNTER_MOVE[prediction].

        Returns the gesture string the robot should throw ("Rock"/"Paper"/"Scissors").
        """
        # Scale the grace period by the personality's grace multiplier.
        eff_grace = int(self.grace_rounds * self._p_grace_mult)

        # Round 1 or empty history — no data to work with, just pick randomly.
        if round_number <= 1 or not history:
            self.last_prediction = {
                "top_predicted_move": None,
                "used_predicted_move": None,
                "effective_skill": None,
                "opponent_type": "unknown",
            }
            return random.choice(VALID_GESTURES)

        # During the grace period, skill ramps up gradually from 0.30.
        if round_number <= eff_grace:
            grace_skill = max(0.30, self.base_skill * (round_number / max(eff_grace, 1)))
            move = random.choice(VALID_GESTURES)
            self.last_prediction = {
                "top_predicted_move": None,
                "used_predicted_move": move,
                "effective_skill": grace_skill,
                "opponent_type": "grace_period",
                "personality": self.personality,
            }
            return move

        # ── Chaos Agent: pure Nash equilibrium — always random. ──
        if self._p_miss_mode == "chaos":
            move = random.choice(VALID_GESTURES)
            self.last_prediction = {
                "top_predicted_move": None, "used_predicted_move": move,
                "effective_skill": 0.333, "opponent_type": "nash",
                "personality": self.personality,
            }
            return move

        # ── Ghost: echo the player's last gesture (mirror, NOT counter). ──
        # INTENTIONAL: move = last_player_g, not COUNTER_MOVE[last_player_g].
        if self._p_use_delayed and history:
            last_player_g = history[-1].get("player_gesture")
            if last_player_g in VALID_GESTURES:
                move = last_player_g
                self.last_prediction = {
                    "top_predicted_move": last_player_g, "used_predicted_move": last_player_g,
                    "effective_skill": 0.55, "opponent_type": "ghost",
                    "personality": self.personality,
                }
                return move

        # ── Gambler wild roll: randomly ignore all strategy. ──
        if self._p_wild_rate > 0 and random.random() < self._p_wild_rate:
            move = random.choice(VALID_GESTURES)
            self.last_prediction = {
                "top_predicted_move": None, "used_predicted_move": move,
                "effective_skill": 0.33, "opponent_type": "wild",
                "personality": self.personality,
            }
            return move

        # ── Standard prediction path ──
        scores = self._predict_player_scores(history)

        # Rank gestures by their prediction score, highest first.
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_move    = ranked[0][0]
        best_score   = ranked[0][1]
        second_score = ranked[1][1] if len(ranked) > 1 else best_score
        second_move  = ranked[1][0] if len(ranked) > 1 else random.choice(VALID_GESTURES)

        # Skill grows with history length, capped at max_skill, modified by personality.
        effective_skill = min(
            self.max_skill,
            self.base_skill + 0.02 * len(history)
        ) * self._p_skill_mult

        # Rubber-band adjustment: if the AI is on a long win streak, ease off a little
        # so the game stays engaging. If the player is on a streak, tighten up.
        wins_   = getattr(self, "_consecutive_wins",   0)
        losses_ = getattr(self, "_consecutive_losses", 0)
        if wins_ >= 5:
            # AI has been winning too much — dial back skill slightly.
            effective_skill = max(0.30, effective_skill - 0.04 * min(wins_ - 4, 4))
        elif losses_ >= 5:
            # Player has been winning — AI pushes a bit harder.
            effective_skill = min(self.max_skill + 0.06, effective_skill + 0.03 * min(losses_ - 4, 4))

        # Update the consecutive win/loss counters for next round's rubber-band check.
        if history:
            last_outcome = history[-1].get("player_outcome", "")
            if last_outcome == "win":
                self._consecutive_wins   = getattr(self, "_consecutive_wins", 0) + 1
                self._consecutive_losses = 0
            elif last_outcome == "lose":
                self._consecutive_losses = getattr(self, "_consecutive_losses", 0) + 1
                self._consecutive_wins   = 0
            else:
                self._consecutive_wins   = 0
                self._consecutive_losses = 0

        # If the top two scores are very close, the AI is uncertain — reduce skill a bit.
        if (best_score - second_score) < 0.40:
            effective_skill -= 0.08

        # Never let skill drop below the minimum floor.
        effective_skill = max(0.35, effective_skill)

        # The "skill roll": if random() < effective_skill, the AI plays optimally.
        # Otherwise it "misses" according to the personality's miss_mode.
        if random.random() < effective_skill:
            predicted_player_move = best_move
        else:
            # Miss mode — how the AI deliberately plays sub-optimally.
            if self._p_miss_mode == "second":
                # Pick the second-best prediction — still fairly smart.
                predicted_player_move = second_move
            elif self._p_miss_mode == "modal" and history:
                # Pick the player's overall most common gesture.
                gestures = [r["player_gesture"] for r in history if r["player_gesture"] in VALID_GESTURES]
                predicted_player_move = max(set(gestures), key=gestures.count) if gestures else best_move
            else:
                # Default: weighted random pick from all scores.
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

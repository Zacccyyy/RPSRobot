import random
from collections import defaultdict


VALID_GESTURES = ("Rock", "Paper", "Scissors")

COUNTER_MOVE = {
    "Rock": "Paper",
    "Paper": "Scissors",
    "Scissors": "Rock",
}

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

# Skill presets per difficulty
_DIFFICULTY_PRESETS = {
    "Easy":   {"base_skill": 0.40, "max_skill": 0.55, "grace_rounds": 20},
    "Normal": {"base_skill": 0.66, "max_skill": 0.76, "grace_rounds": 10},
    "Hard":   {"base_skill": 0.80, "max_skill": 0.92, "grace_rounds":  0},
}

# ── AI Personality presets ─────────────────────────────────────────────────────
# Each personality overrides layer weights and/or skill in choose_robot_move.
# "layer_bias" keys correspond to prediction layer multipliers.
# "miss_mode" controls how the AI chooses when it deliberately misses:
#   "random"  — pick uniformly from all gestures (default)
#   "second"  — pick second-best prediction
#   "chaos"   — true 33/33/33 Nash equilibrium play (never exploitable)
#   "modal"   — always play the player's most common gesture (blatant tell)
#   "delayed" — use the player's last gesture (one round behind)
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

PERSONALITY_NAMES = list(PERSONALITIES.keys())


def _detect_opponent_type(history):
    """
    Classify the opponent's playstyle from their round history.
    Returns a label used to bias the AI's prediction layer.

    Types:
        rock_heavy   — plays Rock > 50% of the time
        paper_heavy  — plays Paper > 50%
        scissors_heavy — plays Scissors > 50%
        cycler       — follows a regular R→P→S pattern
        win_stay     — repeats winning moves
        random       — no detectable pattern (default)
    """
    if len(history) < 8:
        return "random"

    gestures = [r["player_gesture"] for r in history if r["player_gesture"] in VALID_GESTURES]
    if not gestures:
        return "random"

    total = len(gestures)
    freq = {g: gestures.count(g) / total for g in VALID_GESTURES}

    # Heavy bias toward one gesture
    for g, f in freq.items():
        if f > 0.50:
            return f"{g.lower()}_heavy"

    # Win-stay detection: how often does the player repeat after a win?
    wins = [r for r in history if r.get("player_outcome") == "win"]
    win_stay_count = 0
    for i, r in enumerate(history[:-1]):
        if r.get("player_outcome") == "win":
            if history[i + 1]["player_gesture"] == r["player_gesture"]:
                win_stay_count += 1
    if wins and win_stay_count / max(len(wins), 1) > 0.60:
        return "win_stay"

    # Cycler detection: check R→P→S progression in last 9 rounds
    cycle = ("Rock", "Paper", "Scissors")
    cycle_hits = 0
    recent = gestures[-9:]
    for i in range(len(recent) - 1):
        expected_next = cycle[(cycle.index(recent[i]) + 1) % 3] if recent[i] in cycle else None
        if expected_next and recent[i + 1] == expected_next:
            cycle_hits += 1
    if len(recent) > 1 and cycle_hits / (len(recent) - 1) > 0.65:
        return "cycler"

    return "random"


def _markov_move_scores(history, move_scores):
    """
    Layer 6: WIN/LOSE/TIE Markov transition tables.
    For each outcome (win/lose/draw), track gesture→gesture transitions
    and score the most likely next gesture given last outcome + last gesture.
    Inspired by iamvigneshwars/rock-paper-scissors-ai.
    """
    if len(history) < 3:
        return

    tables = {
        "win":  defaultdict(lambda: defaultdict(float)),
        "lose": defaultdict(lambda: defaultdict(float)),
        "draw": defaultdict(lambda: defaultdict(float)),
    }

    for i in range(len(history) - 1):
        outcome = history[i].get("player_outcome", "")
        g_from  = history[i].get("player_gesture", "")
        g_to    = history[i + 1].get("player_gesture", "")
        if outcome in tables and g_from in VALID_GESTURES and g_to in VALID_GESTURES:
            # Recency-weighted
            distance = (len(history) - 2) - i
            weight = 1.0 / (1.28 ** distance)
            tables[outcome][g_from][g_to] += weight

    last = history[-1]
    outcome = last.get("player_outcome", "")
    g_from  = last.get("player_gesture", "")

    if outcome in tables and g_from in tables[outcome]:
        row = tables[outcome][g_from]
        total = sum(row.values())
        if total > 0:
            for g_to, w in row.items():
                move_scores[g_to] += 1.8 * (w / total)


class FairPlayAI:
    """
    Fair Play AI v3

    Strategy layers (in order of application):
    1. Soft population priors (human psychology baselines)
    2. Outcome-conditioned response learning (stay/upgrade/downgrade)
    3. Exact transition memory (move-to-move)
    4. Weaker outcome→next-move tendencies
    5. Tiny overall frequency fallback
    6. WIN/LOSE/TIE Markov transition tables (new)

    Difficulty presets: Easy / Normal / Hard
    Grace period: first N rounds play near-random to give new players a chance
    Opponent type detection: adapts to rock-heavy, cycler, win-stay players
    """

    def __init__(self, base_skill=0.66, max_skill=0.76, difficulty="Normal",
                 personality="Normal"):
        preset = _DIFFICULTY_PRESETS.get(difficulty, _DIFFICULTY_PRESETS["Normal"])
        self.base_skill   = preset["base_skill"]
        self.max_skill    = preset["max_skill"]
        self.grace_rounds = preset["grace_rounds"]
        self.difficulty   = difficulty
        self.last_prediction = None
        self.set_personality(personality)

    def set_personality(self, name):
        p = PERSONALITIES.get(name, PERSONALITIES["Normal"])
        self.personality      = name
        self._p_skill_mult    = p.get("skill_mult", 1.0)
        self._p_grace_mult    = p.get("grace_mult", 1.0)
        self._p_layer_bias    = p.get("layer_bias", {})
        self._p_miss_mode     = p.get("miss_mode", "random")
        self._p_wild_rate     = p.get("wild_rate", 0.0)
        self._p_use_delayed   = p.get("use_delayed", False)

    def set_difficulty(self, difficulty):
        preset = _DIFFICULTY_PRESETS.get(difficulty, _DIFFICULTY_PRESETS["Normal"])
        self.base_skill   = preset["base_skill"]
        self.max_skill    = preset["max_skill"]
        self.grace_rounds = preset["grace_rounds"]
        self.difficulty   = difficulty

    def reset(self):
        self.last_prediction = None
        self._consecutive_wins   = 0
        self._consecutive_losses = 0
        # Thompson Sampling bandit: Beta(alpha, beta) per layer
        # Each layer starts with a non-informative prior (1, 1)
        self._bandit = {
            "outcome":    [1.0, 1.0],   # [alpha_successes, beta_failures]
            "transition": [1.0, 1.0],
            "opp_next":   [1.0, 1.0],
            "frequency":  [1.0, 1.0],
            "markov":     [1.0, 1.0],
        }
        self._last_layer_contributions = {}  # which layers were top scorers

    def update_bandit(self, ai_predicted_gesture, actual_player_gesture):
        """
        Call after each resolved round to update layer reward estimates.
        ai_predicted_gesture: what the AI predicted the player would throw
        actual_player_gesture: what the player actually threw
        correct = AI prediction was right
        """
        if not self._last_layer_contributions:
            return
        correct = (ai_predicted_gesture == actual_player_gesture)
        for layer_name, contribution in self._last_layer_contributions.items():
            if layer_name not in self._bandit:
                continue
            # Only update layers that had meaningful contribution (> threshold)
            if contribution > 0.1:
                if correct:
                    self._bandit[layer_name][0] = min(50.0,
                        self._bandit[layer_name][0] + contribution)
                else:
                    self._bandit[layer_name][1] = min(50.0,
                        self._bandit[layer_name][1] + contribution * 0.5)

    def _thompson_sample(self, layer_name):
        """Sample from Beta(alpha, beta) for a given layer — Thompson Sampling."""
        a, b = self._bandit.get(layer_name, [1.0, 1.0])
        # Simple beta sample: use mean + noise for performance
        mean = a / (a + b)
        # Add small exploration noise
        noise = random.gauss(0, 0.08 / (a + b) ** 0.5)
        return max(0.1, min(2.5, mean * 2.0 + noise))  # scale to [0.1, 2.5]

    def _blank_move_scores(self):
        return {
            "Rock": 1.0,
            "Paper": 1.0,
            "Scissors": 1.0,
        }

    def _blank_response_scores(self):
        return {
            "stay": 0.0,
            "upgrade": 0.0,
            "downgrade": 0.0,
        }

    def _recency_weight(self, distance_from_latest):
        """
        More recent rounds matter more.
        distance_from_latest = 0 means most recent usable pattern.
        """
        return 1.0 / (1.28 ** distance_from_latest)

    def _get_response_type(self, previous_move, next_move):
        if next_move == previous_move:
            return "stay"

        if UPGRADE_MOVE[previous_move] == next_move:
            return "upgrade"

        return "downgrade"

    def _apply_population_priors(self, last_outcome, response_scores):
        """
        Soft baseline tendencies.
        These are intentionally mild so the player's own behaviour
        can quickly take over.
        """
        if last_outcome == "lose":
            # After a loss, people often shift rather than repeat.
            response_scores["stay"] += 0.65
            response_scores["upgrade"] += 1.15
            response_scores["downgrade"] += 1.15

        elif last_outcome == "win":
            # After a win, some tendency to stay.
            response_scores["stay"] += 1.10
            response_scores["upgrade"] += 0.90
            response_scores["downgrade"] += 0.90

        elif last_outcome == "draw":
            # Draw is treated more neutrally.
            response_scores["stay"] += 0.95
            response_scores["upgrade"] += 1.00
            response_scores["downgrade"] += 1.00

        else:
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
        Learn how this player tends to respond after a given outcome.

        Example:
        if the last round was a player loss, look through older examples
        where the player also lost, then see whether they stayed,
        upgraded, or downgraded on the following round.
        """
        if len(history) < 2:
            return

        for i in range(len(history) - 1):
            previous_round = history[i]
            next_round = history[i + 1]

            if previous_round["player_outcome"] != last_outcome:
                continue

            observed_response = self._get_response_type(
                previous_round["player_gesture"],
                next_round["player_gesture"]
            )

            distance = (len(history) - 2) - i
            weight = 2.2 * self._recency_weight(distance)

            # Extra relevance if the old example started from the same move
            # the player just used now.
            if previous_round["player_gesture"] == last_move:
                weight *= 1.25

            response_scores[observed_response] += weight

    def _convert_response_scores_to_move_scores(
        self,
        last_move,
        response_scores,
        move_scores
    ):
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
        Secondary layer:
        learn exact move-to-move tendencies.

        Example:
        if the player just used Rock, and older data shows they often go
        Rock -> Paper next, give Paper extra score.
        """
        if len(history) < 2:
            return

        for i in range(len(history) - 1):
            previous_round = history[i]
            next_round = history[i + 1]

            if previous_round["player_gesture"] != last_move:
                continue

            distance = (len(history) - 2) - i
            weight = 1.35 * self._recency_weight(distance)

            # Slight bonus when both the move and the outcome match
            # the current situation.
            if previous_round["player_outcome"] == last_outcome:
                weight *= 1.35

            move_scores[next_round["player_gesture"]] += weight

    def _score_outcome_next_move_patterns(self, history, last_outcome, move_scores):
        """
        Small helper layer:
        after a given outcome, what actual move tends to appear next,
        regardless of the starting move?
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
        Very light fallback memory so the AI still has some direction
        when outcome-specific patterns are weak.
        """
        reversed_history = list(reversed(history))

        for idx, record in enumerate(reversed_history):
            move = record["player_gesture"]
            move_scores[move] += 0.30 * self._recency_weight(idx)

    def _predict_player_scores(self, history):
        scores = self._blank_move_scores()

        if not history:
            return scores

        last_round   = history[-1]
        last_move    = last_round["player_gesture"]
        last_outcome = last_round["player_outcome"]

        response_scores = self._blank_response_scores()

        # ── Thompson Sampling weights for each layer ──────────────────────
        w_outcome    = self._thompson_sample("outcome")
        w_transition = self._thompson_sample("transition")
        w_opp_next   = self._thompson_sample("opp_next")
        w_frequency  = self._thompson_sample("frequency")
        w_markov     = self._thompson_sample("markov")

        # Apply personality layer biases on top of bandit weights
        pb = self._p_layer_bias
        w_outcome    *= pb.get("outcome",    1.0)
        w_transition *= pb.get("transition", 1.0)
        w_frequency  *= pb.get("frequency",  1.0)
        w_markov     *= pb.get("markov",     1.0)

        # Layer 1: soft baseline priors (no bandit — always applies)
        self._apply_population_priors(last_outcome, response_scores)

        # Layer 2: outcome-conditioned response learning
        rs2 = self._blank_response_scores()
        self._score_outcome_conditioned_responses(
            history=history, last_move=last_move,
            last_outcome=last_outcome, response_scores=rs2)
        for k in rs2:
            response_scores[k] += rs2[k] * w_outcome

        self._convert_response_scores_to_move_scores(
            last_move=last_move, response_scores=response_scores,
            move_scores=scores)

        # Layer 3: exact transition memory
        pre3 = {g: scores[g] for g in VALID_GESTURES}
        self._score_exact_transition_memory(
            history=history, last_move=last_move,
            last_outcome=last_outcome, move_scores=scores)
        contrib3 = sum(max(0, scores[g] - pre3[g]) for g in VALID_GESTURES)
        # Re-apply with bandit weight (undo raw contribution and scale)
        if contrib3 > 0:
            for g in VALID_GESTURES:
                delta = scores[g] - pre3[g]
                scores[g] = pre3[g] + delta * w_transition

        # Layer 4: outcome→next-move tendencies
        pre4 = {g: scores[g] for g in VALID_GESTURES}
        self._score_outcome_next_move_patterns(
            history=history, last_outcome=last_outcome, move_scores=scores)
        if True:
            for g in VALID_GESTURES:
                delta = scores[g] - pre4[g]
                scores[g] = pre4[g] + delta * w_opp_next

        # Layer 5: frequency fallback
        pre5 = {g: scores[g] for g in VALID_GESTURES}
        self._score_overall_frequency(history, scores)
        for g in VALID_GESTURES:
            delta = scores[g] - pre5[g]
            scores[g] = pre5[g] + delta * w_frequency

        # Layer 6: Markov WIN/LOSE/TIE tables
        pre6 = {g: scores[g] for g in VALID_GESTURES}
        _markov_move_scores(history, scores)
        for g in VALID_GESTURES:
            delta = scores[g] - pre6[g]
            scores[g] = pre6[g] + delta * w_markov

        # Opponent-type bias (deterministic, no bandit)
        opp_type = _detect_opponent_type(history)
        if opp_type.endswith("_heavy"):
            heavy_gesture = opp_type.replace("_heavy", "").capitalize()
            scores[heavy_gesture] += 1.5
        elif opp_type == "cycler":
            cycle = ("Rock", "Paper", "Scissors")
            if last_move in cycle:
                predicted_next = cycle[(cycle.index(last_move) + 1) % 3]
                scores[predicted_next] += 2.0
        elif opp_type == "win_stay" and last_outcome == "win":
            scores[last_move] += 2.0

        # Track layer contributions for bandit update next round
        self._last_layer_contributions = {
            "outcome":    w_outcome,
            "transition": w_transition,
            "opp_next":   w_opp_next,
            "frequency":  w_frequency,
            "markov":     w_markov,
        }

        return scores

    def _weighted_choice(self, score_dict):
        total = sum(max(v, 0.001) for v in score_dict.values())
        pick = random.uniform(0, total)
        current = 0.0

        for move, score in score_dict.items():
            current += max(score, 0.001)
            if current >= pick:
                return move

        return random.choice(VALID_GESTURES)

    def choose_robot_move(self, history, round_number):
        # Grace period: play near-random for first N rounds
        eff_grace = int(self.grace_rounds * self._p_grace_mult)
        if round_number <= 1 or not history:
            self.last_prediction = {
                "top_predicted_move": None,
                "used_predicted_move": None,
                "effective_skill": None,
                "opponent_type": "unknown",
            }
            return random.choice(VALID_GESTURES)

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

        # ── Chaos Agent: pure Nash ──
        if self._p_miss_mode == "chaos":
            move = random.choice(VALID_GESTURES)
            self.last_prediction = {
                "top_predicted_move": None, "used_predicted_move": move,
                "effective_skill": 0.333, "opponent_type": "nash",
                "personality": self.personality,
            }
            return move

        # ── Ghost: play player's last gesture ──
        if self._p_use_delayed and history:
            last_player_g = history[-1].get("player_gesture")
            if last_player_g in VALID_GESTURES:
                move = COUNTER_MOVE[last_player_g]
                self.last_prediction = {
                    "top_predicted_move": last_player_g, "used_predicted_move": last_player_g,
                    "effective_skill": 0.55, "opponent_type": "ghost",
                    "personality": self.personality,
                }
                return move

        # ── Gambler wild roll ──
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

        # Apply personality layer biases
        bias = self._p_layer_bias
        if "frequency" in bias:
            # Recompute with boosted frequency weight — simple rescale via post-hoc
            pass  # layer biases are applied inside _predict_player_scores via self._p_layer_bias

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_move    = ranked[0][0]
        best_score   = ranked[0][1]
        second_score = ranked[1][1] if len(ranked) > 1 else best_score
        second_move  = ranked[1][0] if len(ranked) > 1 else random.choice(VALID_GESTURES)

        effective_skill = min(
            self.max_skill,
            self.base_skill + 0.02 * len(history)
        ) * self._p_skill_mult

        wins_   = getattr(self, "_consecutive_wins",   0)
        losses_ = getattr(self, "_consecutive_losses", 0)
        if wins_ >= 5:
            effective_skill = max(0.30, effective_skill - 0.04 * min(wins_ - 4, 4))
        elif losses_ >= 5:
            effective_skill = min(self.max_skill + 0.06, effective_skill + 0.03 * min(losses_ - 4, 4))

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

        if (best_score - second_score) < 0.40:
            effective_skill -= 0.08

        effective_skill = max(0.35, effective_skill)

        if random.random() < effective_skill:
            predicted_player_move = best_move
        else:
            # Miss mode — personality controls how the AI "misses"
            if self._p_miss_mode == "second":
                predicted_player_move = second_move
            elif self._p_miss_mode == "modal" and history:
                gestures = [r["player_gesture"] for r in history if r["player_gesture"] in VALID_GESTURES]
                predicted_player_move = max(set(gestures), key=gestures.count) if gestures else best_move
            else:
                predicted_player_move = self._weighted_choice(scores)

        opp_type = _detect_opponent_type(history)
        self.last_prediction = {
            "top_predicted_move":  best_move,
            "used_predicted_move": predicted_player_move,
            "effective_skill":     round(effective_skill, 4),
            "opponent_type":       opp_type,
            "personality":         self.personality,
        }

        return COUNTER_MOVE[predicted_player_move]
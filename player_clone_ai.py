"""
Player Clone AI

Plays AS a recorded player by reproducing their statistical patterns.
Uses four layers of decision-making, weighted by specificity:

    1. Outcome + gesture transition (most specific):
       "After losing with Rock, this player throws Paper 60% of the time"

    2. Gesture transition:
       "After Rock, this player usually goes to Paper"

    3. Outcome response type:
       "After a loss, this player upgrades 55% of the time"

    4. Overall frequency fallback:
       "This player throws Rock 45% of the time"

The clone combines these layers with recency-weighted sampling
from the actual round history, so it captures timing patterns
as well as aggregate statistics.

Interface matches FairPlayAI so it drops into FairPlayController.
"""

import random


GESTURES = ("Rock", "Paper", "Scissors")
UPGRADE = {"Rock": "Paper", "Paper": "Scissors", "Scissors": "Rock"}
DOWNGRADE = {"Rock": "Scissors", "Paper": "Rock", "Scissors": "Paper"}


class PlayerCloneAI:
    """
    AI that plays AS a specific recorded player.

    Unlike FairPlayAI (which predicts then counters), this AI
    predicts what the cloned player would throw and THROWS that move.
    """

    def __init__(self, pattern_tables, accuracy=0.85):
        """
        pattern_tables: dict from PlayerProfileStore.build_pattern_tables()
        accuracy: how often to use the most specific prediction vs random
        """
        self.tables = pattern_tables
        self.accuracy = accuracy
        self.player_name = pattern_tables.get("player_name", "Unknown")
        self.round_count = pattern_tables.get("round_count", 0)

    def reset(self):
        pass

    def choose_robot_move(self, history, round_number=1, **kwargs):
        """
        Choose what the cloned player would throw.

        history: list of round dicts with player_gesture, player_outcome
                 (Note: in clone mode, "player" is the human opponent,
                  and the robot IS the clone. So we look at the clone's
                  own history from the robot's perspective.)
        """
        if not history or round_number <= 1:
            return self._sample_from_frequency()

        # The clone's previous move is the robot_gesture from history.
        # The clone's previous outcome is the inverse of player_outcome.
        last_round = history[-1]
        clone_last_move = last_round.get("robot_gesture", None)
        player_outcome = last_round.get("player_outcome", "draw")

        # Invert outcome: if player won, clone lost.
        clone_outcome = {
            "win": "lose",
            "lose": "win",
            "draw": "draw",
        }.get(player_outcome, "draw")

        if clone_last_move not in GESTURES:
            return self._sample_from_frequency()

        # Decide whether to use pattern-based prediction or add noise.
        if random.random() > self.accuracy:
            return self._sample_from_frequency()

        # Layer 1: Outcome + gesture → next gesture (most specific).
        ot = self.tables.get("outcome_transition", {})
        if clone_outcome in ot and clone_last_move in ot[clone_outcome]:
            probs = ot[clone_outcome][clone_last_move]
            if self._has_data(probs):
                return self._weighted_sample(probs)

        # Layer 2: Gesture transition.
        trans = self.tables.get("transition", {})
        if clone_last_move in trans:
            probs = trans[clone_last_move]
            if self._has_data(probs):
                return self._weighted_sample(probs)

        # Layer 3: Outcome response type → derive move.
        or_table = self.tables.get("outcome_response", {})
        if clone_outcome in or_table:
            response_probs = or_table[clone_outcome]
            if self._has_data(response_probs):
                response = self._weighted_sample(response_probs)
                if response == "stay":
                    return clone_last_move
                elif response == "upgrade":
                    return UPGRADE[clone_last_move]
                else:
                    return DOWNGRADE[clone_last_move]

        # Layer 4: Overall frequency fallback.
        return self._sample_from_frequency()

    def _sample_from_frequency(self):
        freq = self.tables.get("gesture_freq", {})
        if not freq or not self._has_data(freq):
            return random.choice(GESTURES)
        return self._weighted_sample(freq)

    def _weighted_sample(self, prob_dict):
        """Sample from a probability dictionary."""
        items = list(prob_dict.items())
        weights = [max(v, 0.001) for _, v in items]
        total = sum(weights)
        pick = random.uniform(0, total)
        current = 0.0
        for item, w in zip(items, weights):
            current += w
            if current >= pick:
                return item[0]
        return items[-1][0]

    def _has_data(self, prob_dict):
        """Check if the probability dict has any real data (not all zeros)."""
        return sum(prob_dict.values()) > 0.01

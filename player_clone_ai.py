"""
player_clone_ai.py
==================
An AI that imitates a specific recorded human player.

Most AI opponents in this game try to BEAT the player.  This one is different:
it tries to PLAY LIKE the player, reproducing their statistical tendencies so
the opponent can practice against a "ghost" version of themselves or someone
else whose data was collected.

How it works — four layers of decision-making (most specific first):
  1. Outcome + last gesture  → "After losing with Rock, this player plays Paper 60% of the time"
  2. Last gesture only       → "After Rock, this player usually moves to Paper"
  3. Outcome only (response type) → "After a loss, this player upgrades 55% of the time"
  4. Overall gesture frequency    → "This player throws Rock 45% of the time"

Each layer falls through to the next if there is not enough data for it.
A configurable `accuracy` parameter adds random noise so the clone is not
perfectly predictable, which makes the game feel more natural.

The pattern_tables dict comes from PlayerProfileStore.build_pattern_tables()
(see player_profile_store.py).  The class interface mirrors FairPlayAI so it
drops into FairPlayController without changes.
"""

import random

# The three legal gestures in standard RPS.
GESTURES = ("Rock", "Paper", "Scissors")

# What gesture "beats" the key.  Used to compute "upgrade" transitions.
UPGRADE   = {"Rock": "Paper",    "Paper": "Scissors", "Scissors": "Rock"}
# What gesture "loses to" the key.  Used to compute "downgrade" transitions.
DOWNGRADE = {"Rock": "Scissors", "Paper": "Rock",     "Scissors": "Paper"}


class PlayerCloneAI:
    """
    AI that plays AS a specific recorded player.

    Unlike FairPlayAI (which predicts the player's move and counters it), this
    AI predicts what the cloned player would throw and then THROWS that same
    move — the goal is imitation, not victory.
    """

    def __init__(self, pattern_tables, accuracy=0.85):
        """
        Build the clone from pre-computed pattern tables.

        Parameters
        ----------
        pattern_tables : dict
            Output of PlayerProfileStore.build_pattern_tables().  Expected keys:
              "player_name"       — display name of the cloned player
              "round_count"       — how many rounds of data the clone is based on
              "gesture_freq"      — {gesture: probability} overall frequency
              "transition"        — {gesture: {gesture: probability}} move-to-move matrix
              "outcome_response"  — {outcome: {response_type: probability}}
              "outcome_transition"— {outcome: {gesture: {gesture: probability}}}
        accuracy : float (0–1)
            Fraction of moves made using the statistical pattern rather than a
            completely random gesture.  0.85 means 85% pattern-based, 15% noise.
        """
        self.tables      = pattern_tables
        self.accuracy    = accuracy
        self.player_name = pattern_tables.get("player_name", "Unknown")
        self.round_count = pattern_tables.get("round_count", 0)

    def reset(self):
        """Reset the clone between games.  No per-game state to clear right now."""
        pass

    def choose_robot_move(self, history, round_number=1, **kwargs):
        """
        Decide what the cloned player would throw this round.

        In Clone mode the robot IS the clone, so we look at the robot's own
        previous gesture and the inverse of the player's outcome to understand
        what the clone "experienced" last round.

        Parameters
        ----------
        history      : list of round dicts, each containing at minimum:
                         "robot_gesture"  — what the clone (robot) threw last round
                         "player_outcome" — "win" / "lose" / "draw" from the human's POV
        round_number : int — used to skip the lookup on round 1 (no prior history)
        **kwargs     : ignored (keeps the interface compatible with FairPlayAI)

        Returns
        -------
        str : one of "Rock", "Paper", "Scissors"
        """
        # On the very first round there is no prior history, so fall back to
        # the clone's overall gesture frequency.
        if not history or round_number <= 1:
            return self._sample_from_frequency()

        last_round     = history[-1]
        clone_last_move = last_round.get("robot_gesture", None)  # what the clone threw
        player_outcome  = last_round.get("player_outcome", "draw")  # from human's perspective

        # Invert the player's outcome to get the clone's outcome.
        # If the human won, the clone lost — and vice versa.
        clone_outcome = {
            "win":  "lose",
            "lose": "win",
            "draw": "draw",
        }.get(player_outcome, "draw")

        # If the last move was unrecognised (e.g. a glitch), use the frequency fallback.
        if clone_last_move not in GESTURES:
            return self._sample_from_frequency()

        # Occasionally play a completely random gesture to prevent the clone
        # from being perfectly predictable ("accuracy noise").
        if random.random() > self.accuracy:
            return self._sample_from_frequency()

        # ---------------------------------------------------------------
        # Layer 1: outcome + last gesture → next gesture (most specific)
        # ---------------------------------------------------------------
        # Example: "After losing with Rock, cloned player throws Paper 60%"
        ot = self.tables.get("outcome_transition", {})
        if clone_outcome in ot and clone_last_move in ot[clone_outcome]:
            probs = ot[clone_outcome][clone_last_move]
            if self._has_data(probs):
                return self._weighted_sample(probs)

        # ---------------------------------------------------------------
        # Layer 2: last gesture only → next gesture
        # ---------------------------------------------------------------
        # Example: "After Rock, this player usually throws Paper"
        trans = self.tables.get("transition", {})
        if clone_last_move in trans:
            probs = trans[clone_last_move]
            if self._has_data(probs):
                return self._weighted_sample(probs)

        # ---------------------------------------------------------------
        # Layer 3: outcome → response type → derive move
        # ---------------------------------------------------------------
        # Example: "After a loss, this player upgrades 55% of the time"
        # We sample a response type (stay/upgrade/downgrade) then apply it
        # to the clone's last move to get an actual gesture.
        or_table = self.tables.get("outcome_response", {})
        if clone_outcome in or_table:
            response_probs = or_table[clone_outcome]
            if self._has_data(response_probs):
                response = self._weighted_sample(response_probs)
                if response == "stay":
                    return clone_last_move          # play the same gesture again
                elif response == "upgrade":
                    return UPGRADE[clone_last_move]  # move up the RPS cycle
                else:
                    return DOWNGRADE[clone_last_move]  # move down the RPS cycle

        # ---------------------------------------------------------------
        # Layer 4: overall gesture frequency fallback
        # ---------------------------------------------------------------
        # None of the more specific layers had data — just use how often the
        # player throws each gesture overall.
        return self._sample_from_frequency()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_from_frequency(self):
        """
        Sample a gesture weighted by the clone's overall gesture frequency.

        Falls back to a uniform random choice if no frequency data is available.
        """
        freq = self.tables.get("gesture_freq", {})
        if not freq or not self._has_data(freq):
            return random.choice(GESTURES)  # no data at all — pure random
        return self._weighted_sample(freq)

    def _weighted_sample(self, prob_dict):
        """
        Draw one key from a {key: weight} dict using weighted random sampling.

        Weights are floored at 0.001 so zero-probability items can't cause a
        divide-by-zero or an infinite loop, and they can still occasionally
        appear (which adds natural noise to the clone's behaviour).
        """
        items   = list(prob_dict.items())
        weights = [max(v, 0.001) for _, v in items]
        total   = sum(weights)
        pick    = random.uniform(0, total)

        # Walk through the items, accumulating weight until we reach the pick point.
        current = 0.0
        for item, w in zip(items, weights):
            current += w
            if current >= pick:
                return item[0]  # item is a (key, value) tuple; return just the key

        # Floating-point edge case: pick == total exactly.  Return the last item.
        return items[-1][0]

    def _has_data(self, prob_dict):
        """
        Return True if the probability dict has any meaningful data.

        A total weight below 0.01 means all values are effectively zero, so
        sampling from it would produce garbage — better to fall through to the
        next layer.
        """
        return sum(prob_dict.values()) > 0.01

import random

from fair_play_ai import FairPlayAI, VALID_GESTURES, COUNTER_MOVE


# Emotion-based confidence adjustments.
# Positive = AI trusts its prediction more (exploits harder).
# Negative = AI plays more cautiously (harder for player to read).
#
# Rationale (grounded in behavioural research):
#   Frustrated: Dyson et al. (2016) showed negative outcomes evoke cyclic
#       irrational decisions. Frustrated players repeat losing patterns.
#   Happy: Players in flow are confident and less predictable.
#   Surprised: Disoriented players fall back to default tendencies.
EMOTION_SKILL_MODIFIER = {
    "Frustrated": 0.06,
    "Happy": -0.04,
    "Surprised": 0.04,
    "Neutral": 0.00,
    "Unknown": 0.00,
}


class ChallengeAI(FairPlayAI):
    """
    Challenge Mode AI

    Uses the same research-backed prediction model as Fair Play AI v2,
    but ramps its confidence as the player's win streak increases.

    Emotion-Aware Extension:
    When an emotion snapshot is available, the AI adjusts its effective
    confidence. Against frustrated players it exploits more aggressively;
    against happy/confident players it plays more cautiously.

    Design goals:
    - starts beatable
    - becomes noticeably stronger over repeated wins
    - adapts exploitation level based on player emotional state
    - keeps the same lightweight, modular architecture
    """

    def __init__(
        self,
        base_skill=0.68,
        max_skill=0.92,
        ramp_per_win=0.035
    ):
        super().__init__(base_skill=base_skill, max_skill=max_skill)
        self.ramp_per_win = ramp_per_win
        self.emotion_snapshot = None

    def reset(self):
        super().reset()                  # initialises _bandit, last_prediction, etc.
        self.emotion_snapshot = None

    def set_emotion(self, snapshot):
        """Receive the latest emotion snapshot from the tracker."""
        self.emotion_snapshot = snapshot

    def _get_emotion_modifier(self):
        """
        Returns (modifier_value, emotion_label) based on the current
        emotion snapshot. Used to adjust effective_skill.
        """
        if not self.emotion_snapshot:
            return 0.0, "none"

        emotion = self.emotion_snapshot.get("emotion", "Unknown")
        confidence = self.emotion_snapshot.get("emotion_confidence", 0.0)

        base_mod = EMOTION_SKILL_MODIFIER.get(emotion, 0.0)

        # Scale modifier by detection confidence so weak reads have less impact.
        scaled_mod = base_mod * min(confidence, 1.0)

        return round(scaled_mod, 4), emotion

    def _confidence_penalty(self, best_score, second_score, streak):
        """
        If the prediction margin is weak, reduce confidence a little.
        As streak rises, the AI trusts itself more.
        """
        margin = best_score - second_score

        if margin >= 1.00:
            return 0.00
        if margin >= 0.55:
            return 0.02 if streak < 4 else 0.00
        if margin >= 0.25:
            return 0.07 if streak < 4 else 0.03
        return 0.12 if streak < 4 else 0.05

    def choose_robot_move(self, history, streak, round_number=1):
        if round_number <= 1 or not history:
            self.last_prediction = {
                "top_predicted_move": None,
                "used_predicted_move": None,
                "effective_skill": None,
                "emotion_modifier": 0.0,
                "emotion_detected": "none",
            }
            return random.choice(VALID_GESTURES)

        scores = self._predict_player_scores(history)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        best_move = ranked[0][0]
        best_score = ranked[0][1]
        second_score = ranked[1][1] if len(ranked) > 1 else best_score

        effective_skill = min(
            self.max_skill,
            self.base_skill + self.ramp_per_win * max(streak, 0)
        )

        effective_skill -= self._confidence_penalty(
            best_score=best_score,
            second_score=second_score,
            streak=streak
        )

        # --- Emotion-aware adjustment ---
        emotion_mod, emotion_label = self._get_emotion_modifier()
        effective_skill += emotion_mod

        effective_skill = max(0.64, effective_skill)

        if random.random() < effective_skill:
            predicted_player_move = best_move
        else:
            # Early streaks: still fairly human/beatable
            if streak < 3:
                predicted_player_move = self._weighted_choice(scores)

            # Mid streaks: mistakes come from the top two guesses only
            elif streak < 6 and len(ranked) > 1:
                top_two = {
                    ranked[0][0]: ranked[0][1],
                    ranked[1][0]: ranked[1][1],
                }
                predicted_player_move = self._weighted_choice(top_two)

            # High streaks: even its "misses" are usually near-optimal
            elif len(ranked) > 1:
                predicted_player_move = ranked[1][0]
            else:
                predicted_player_move = best_move

        self.last_prediction = {
            "top_predicted_move": best_move,
            "used_predicted_move": predicted_player_move,
            "effective_skill": round(effective_skill, 4),
            "emotion_modifier": emotion_mod,
            "emotion_detected": emotion_label,
        }

        return COUNTER_MOVE[predicted_player_move]
# ============================================================
# challenge_ai.py
#
# AI for Challenge Mode — an escalating variant of Fair Play AI.
#
# What this file does:
#   Extends FairPlayAI with two extra behaviours:
#     1. Streak ramping: the AI gets harder as the player's
#        win streak grows (effective_skill goes up each win).
#     2. Emotion awareness: if a webcam emotion tracker is
#        running, the AI adjusts its confidence based on
#        whether the player looks frustrated, happy, or surprised.
#
# Where it fits in the project:
#   challenge_mode_state.py creates a ChallengeAI instance and
#   calls choose_robot_move(history, streak, round_number).
#   The extra `streak` argument is the main difference from the
#   base FairPlayAI.choose_robot_move(history, round_number).
# ============================================================

import random

from fair_play_ai import FairPlayAI, VALID_GESTURES, COUNTER_MOVE


# How much each detected emotion adjusts the AI's effective skill.
# Positive = exploit harder (player is easier to read).
# Negative = back off a little (player is harder to exploit).
#
# Research basis:
#   Frustrated: frustrated players fall into irrational, cyclic
#       decisions — they're easier to predict. (Dyson et al. 2016)
#   Happy: players in flow state are more unpredictable.
#   Surprised: disoriented players briefly revert to default patterns.
#
# DO NOT change these values — they are calibrated against the research.
EMOTION_SKILL_MODIFIER = {
    "Frustrated": 0.06,
    "Happy":     -0.04,
    "Surprised":  0.04,
    "Neutral":    0.00,
    "Unknown":    0.00,
}


class ChallengeAI(FairPlayAI):
    """
    Challenge Mode AI — inherits the full prediction stack from FairPlayAI.

    Key differences from the base class:
      - choose_robot_move() takes a `streak` argument and ramps skill with it.
      - Emotion snapshots adjust effective_skill based on detected mood.
      - The "miss" fallback is smarter at high streaks (the AI barely lets go).
      - base_skill and max_skill start higher because Challenge mode is hard.
    """

    def __init__(self, base_skill=0.68, max_skill=0.92, ramp_per_win=0.035):
        # Initialise the parent FairPlayAI with our higher skill bounds.
        super().__init__(base_skill=base_skill, max_skill=max_skill)

        # How much effective_skill grows per win in the current streak.
        # DO NOT change — calibrated value.
        self.ramp_per_win = ramp_per_win

        # Latest emotion snapshot from the webcam tracker (can be None).
        self.emotion_snapshot = None

    def reset(self):
        """
        Reset all learned state and the emotion snapshot.
        Calls parent reset() first to clear the bandit, last_prediction, etc.
        """
        super().reset()
        self.emotion_snapshot = None

    def set_emotion(self, snapshot):
        """
        Store the latest emotion snapshot from the tracker.

        snapshot is a dict with keys like:
          "emotion"             — dominant emotion label string
          "emotion_confidence"  — float 0.0–1.0 detection confidence
          "smile_score"         — raw smile detector value
          "surprise_score"      — raw surprise detector value
          "frustration_score"   — raw frustration detector value
        Can be None if no face is detected.
        """
        self.emotion_snapshot = snapshot

    def _get_emotion_modifier(self):
        """
        Compute the emotion-based skill adjustment for this round.

        The modifier is scaled by detection confidence — a faint detection
        barely moves the needle, a confident one applies the full modifier.

        Returns (modifier_float, emotion_label_string).
        Returns (0.0, "none") if there is no snapshot.
        """
        if not self.emotion_snapshot:
            return 0.0, "none"

        emotion    = self.emotion_snapshot.get("emotion", "Unknown")
        confidence = self.emotion_snapshot.get("emotion_confidence", 0.0)

        base_mod = EMOTION_SKILL_MODIFIER.get(emotion, 0.0)

        # Clamp confidence to 1.0 so over-confident detections don't over-adjust.
        scaled_mod = base_mod * min(confidence, 1.0)

        return round(scaled_mod, 4), emotion

    def _confidence_penalty(self, best_score, second_score, streak):
        """
        Penalise the AI when the top two prediction scores are very close,
        meaning it isn't sure which gesture the player will throw.

        At high streaks the AI trusts itself more, so the penalty shrinks.

        Penalty tiers by margin (best minus second-best score):
          >= 1.00 → no penalty (clear winner)
          >= 0.55 → tiny penalty, waived at streak >= 4
          >= 0.25 → moderate penalty, halved at streak >= 4
          <  0.25 → significant penalty, halved at streak >= 4
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
        """
        Decide the robot's move for Challenge Mode.

        Differences from FairPlayAI.choose_robot_move():
          - `streak` is the player's current consecutive-win count.
          - effective_skill ramps up with the streak (streak * ramp_per_win).
          - The "miss" fallback gets smarter as the streak rises:
              streak 0–2 → weighted random from all gestures (beatable)
              streak 3–5 → weighted choice from only the top TWO predictions
              streak 6+  → always the second-best prediction (near-optimal miss)
          - Emotion modifier is applied on top of everything else.

        Returns the gesture string the robot should throw.
        """
        # Round 1 or no history — nothing to predict yet, just pick randomly.
        if round_number <= 1 or not history:
            self.last_prediction = {
                "top_predicted_move":  None,
                "used_predicted_move": None,
                "effective_skill":     None,
                "emotion_modifier":    0.0,
                "emotion_detected":    "none",
            }
            return random.choice(VALID_GESTURES)

        # Run all six prediction layers from FairPlayAI and sort by score.
        scores = self._predict_player_scores(history)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        best_move    = ranked[0][0]
        best_score   = ranked[0][1]
        second_score = ranked[1][1] if len(ranked) > 1 else best_score

        # Base skill grows with the streak — longer run means harder AI.
        effective_skill = min(
            self.max_skill,
            self.base_skill + self.ramp_per_win * max(streak, 0)
        )

        # Subtract a penalty if the prediction margin is too thin.
        effective_skill -= self._confidence_penalty(best_score, second_score, streak)

        # Apply the emotion-based adjustment on top.
        emotion_mod, emotion_label = self._get_emotion_modifier()
        effective_skill += emotion_mod

        # Floor: never drop below 0.64 regardless of penalty or emotion.
        effective_skill = max(0.64, effective_skill)

        # Skill roll: if it passes, the AI plays optimally against best prediction.
        if random.random() < effective_skill:
            predicted_player_move = best_move
        else:
            # The AI is "missing" — how smart the miss is depends on streak.
            if streak < 3:
                # Early streaks: weighted random so the AI is genuinely beatable.
                predicted_player_move = self._weighted_choice(scores)

            elif streak < 6 and len(ranked) > 1:
                # Mid streaks: only choose between the top two candidates.
                top_two = {ranked[0][0]: ranked[0][1], ranked[1][0]: ranked[1][1]}
                predicted_player_move = self._weighted_choice(top_two)

            elif len(ranked) > 1:
                # High streaks: even the miss is near-optimal — use second-best.
                predicted_player_move = ranked[1][0]
            else:
                # Edge case: only one candidate (shouldn't happen with 3 gestures).
                predicted_player_move = best_move

        # Store the prediction details for the UI/debug overlay.
        self.last_prediction = {
            "top_predicted_move":  best_move,
            "used_predicted_move": predicted_player_move,
            "effective_skill":     round(effective_skill, 4),
            "emotion_modifier":    emotion_mod,
            "emotion_detected":    emotion_label,
        }

        # Return the gesture that BEATS the predicted player move.
        return COUNTER_MOVE[predicted_player_move]

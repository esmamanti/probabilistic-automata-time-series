from __future__ import annotations

import math


class ProbabilityEngine:
    """Convert transition counts into probabilities and score state paths."""

    def __init__(self, smoothing: bool = True, epsilon: float = 1e-4):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.smoothing = smoothing
        self.epsilon = epsilon

    def build_transition_probabilities(
        self,
        transition_counts: dict[int, dict[int, int]],
        state_count: int,
    ) -> dict[int, dict[int, float]]:
        probabilities: dict[int, dict[int, float]] = {}

        for source_state, targets in transition_counts.items():
            total = sum(targets.values())
            if total <= 0:
                probabilities[source_state] = {}
                continue

            probabilities[source_state] = {}
            vocabulary_size = state_count if self.smoothing else 0
            denominator = total + (self.epsilon * vocabulary_size)
            for target_state, count in targets.items():
                numerator = count + (self.epsilon if self.smoothing else 0.0)
                probabilities[source_state][target_state] = numerator / denominator

        return probabilities

    def transition_probability(
        self,
        source_state: int,
        target_state: int,
        transition_probabilities: dict[int, dict[int, float]],
    ) -> float:
        targets = transition_probabilities.get(source_state, {})
        if target_state in targets:
            return targets[target_state]
        return self.epsilon if self.smoothing else 0.0

    def path_probability(
        self,
        states: list[int],
        transition_probabilities: dict[int, dict[int, float]],
    ) -> float:
        if len(states) <= 1:
            return 1.0

        probability = 1.0
        for source_state, target_state in zip(states, states[1:]):
            probability *= self.transition_probability(source_state, target_state, transition_probabilities)
        return probability

    def average_log_probability(
        self,
        states: list[int],
        transition_probabilities: dict[int, dict[int, float]],
    ) -> float:
        if len(states) <= 1:
            return 0.0

        log_probabilities = []
        for source_state, target_state in zip(states, states[1:]):
            probability = max(
                self.transition_probability(source_state, target_state, transition_probabilities),
                self.epsilon,
            )
            log_probabilities.append(math.log(probability))

        return sum(log_probabilities) / len(log_probabilities)

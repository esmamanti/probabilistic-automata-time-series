from __future__ import annotations

import math

from models.automata.unseen_handler import PatternResolution


class ExplainabilityEngine:
    """Create step-level explanations for automata decisions."""

    def build(
        self,
        resolutions: list[PatternResolution],
        states: list[int],
        transition_probabilities: dict[int, dict[int, float]],
        anomaly_threshold: float,
        epsilon: float,
    ) -> list[dict[str, object]]:
        explanations: list[dict[str, object]] = []
        cumulative_probability = 1.0
        log_probability_sum = 0.0
        transition_count = 0

        for time_step, resolution in enumerate(resolutions):
            previous_state = states[time_step - 1] if time_step > 0 else None
            current_state = states[time_step]

            if previous_state is None:
                probability = 1.0
            else:
                probability = transition_probabilities.get(previous_state, {}).get(current_state, 0.0)

            cumulative_probability *= probability
            if previous_state is not None:
                log_probability_sum += math.log(max(probability, epsilon))
                transition_count += 1

            average_log_probability = log_probability_sum / transition_count if transition_count > 0 else 0.0
            if resolution.status == "unseen":
                decision_reason = "unseen_pattern"
                is_anomaly = True
            elif probability < anomaly_threshold:
                decision_reason = "low_transition_probability"
                is_anomaly = True
            else:
                decision_reason = "expected_transition"
                is_anomaly = False

            explanations.append(
                {
                    "time_step": time_step,
                    "state": current_state,
                    "previous_state": previous_state,
                    "pattern": resolution.original_pattern,
                    "status": resolution.status,
                    "mapped_to": resolution.mapped_pattern,
                    "distance": resolution.distance,
                    "transition_probability": float(probability),
                    "probability": float(probability),
                    "path_probability": float(cumulative_probability),
                    "average_log_probability": float(average_log_probability),
                    "confidence_score": float(cumulative_probability),
                    "decision_reason": decision_reason,
                    "decision": "anomaly" if is_anomaly else "normal",
                }
            )

        return explanations

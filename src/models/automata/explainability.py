from __future__ import annotations

from models.automata.unseen_handler import PatternResolution


class ExplainabilityEngine:
    """Create step-level explanations for automata decisions."""

    def build(
        self,
        resolutions: list[PatternResolution],
        states: list[int],
        transition_probabilities: dict[int, dict[int, float]],
        anomaly_threshold: float,
    ) -> list[dict[str, object]]:
        explanations: list[dict[str, object]] = []

        for time_step, resolution in enumerate(resolutions):
            previous_state = states[time_step - 1] if time_step > 0 else None
            current_state = states[time_step]

            if previous_state is None:
                probability = 1.0
            else:
                probability = transition_probabilities.get(previous_state, {}).get(current_state, 0.0)

            is_anomaly = resolution.status == "unseen" or probability < anomaly_threshold
            explanations.append(
                {
                    "time_step": time_step,
                    "state": resolution.original_pattern,
                    "pattern": resolution.original_pattern,
                    "status": resolution.status,
                    "mapped_to": resolution.mapped_pattern,
                    "probability": float(probability),
                    "confidence_score": float(resolution.confidence_score),
                    "decision": "anomaly" if is_anomaly else "normal",
                }
            )

        return explanations

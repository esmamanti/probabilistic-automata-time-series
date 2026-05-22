from __future__ import annotations

from collections import Counter, defaultdict


class TransitionMatrixBuilder:
    """Count state transitions from an ordered state sequence."""

    def build(self, states: list[int]) -> dict[int, dict[int, int]]:
        transition_counts: dict[int, dict[int, int]] = defaultdict(dict)
        pair_counts = Counter(zip(states, states[1:]))

        for (source_state, target_state), count in pair_counts.items():
            transition_counts[source_state][target_state] = count

        return {state: dict(targets) for state, targets in transition_counts.items()}

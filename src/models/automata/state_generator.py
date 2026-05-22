from __future__ import annotations


class StateGenerator:
    """Create deterministic state ids from observed symbolic patterns."""

    def __init__(self):
        self.pattern_to_state: dict[str, int] = {}
        self.state_to_pattern: dict[int, str] = {}

    def fit(self, patterns: list[str]) -> "StateGenerator":
        unique_patterns = sorted(set(patterns))
        self.pattern_to_state = {pattern: index for index, pattern in enumerate(unique_patterns)}
        self.state_to_pattern = {index: pattern for pattern, index in self.pattern_to_state.items()}
        return self

    def transform(self, patterns: list[str]) -> list[int]:
        if not self.pattern_to_state:
            raise RuntimeError("StateGenerator must be fitted before transform")
        return [self.pattern_to_state[pattern] for pattern in patterns if pattern in self.pattern_to_state]

    def fit_transform(self, patterns: list[str]) -> list[int]:
        return self.fit(patterns).transform(patterns)

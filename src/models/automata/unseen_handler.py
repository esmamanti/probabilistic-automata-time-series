from __future__ import annotations

from dataclasses import dataclass

from models.automata.levenshtein import levenshtein_distance


@dataclass(frozen=True)
class PatternResolution:
    original_pattern: str
    status: str
    mapped_pattern: str
    distance: int
    confidence_score: float


class UnseenPatternHandler:
    """Map unseen symbolic patterns to the closest seen pattern."""

    def resolve(self, pattern: str, known_patterns: list[str]) -> PatternResolution:
        if not known_patterns:
            raise ValueError("known_patterns must contain at least one pattern")

        if pattern in known_patterns:
            return PatternResolution(
                original_pattern=pattern,
                status="seen",
                mapped_pattern=pattern,
                distance=0,
                confidence_score=1.0,
            )

        ranked_patterns = sorted(
            ((candidate, levenshtein_distance(pattern, candidate)) for candidate in known_patterns),
            key=lambda item: (item[1], item[0]),
        )
        mapped_pattern, distance = ranked_patterns[0]
        normalizer = max(len(pattern), len(mapped_pattern), 1)
        confidence_score = max(0.0, 1.0 - (distance / normalizer))
        return PatternResolution(
            original_pattern=pattern,
            status="unseen",
            mapped_pattern=mapped_pattern,
            distance=distance,
            confidence_score=confidence_score,
        )

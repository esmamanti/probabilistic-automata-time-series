from __future__ import annotations


def levenshtein_distance(source: str, target: str) -> int:
    """Compute the edit distance between two symbolic patterns."""
    if source == target:
        return 0
    if not source:
        return len(target)
    if not target:
        return len(source)

    previous_row = list(range(len(target) + 1))
    for source_index, source_char in enumerate(source, start=1):
        current_row = [source_index]
        for target_index, target_char in enumerate(target, start=1):
            insertion_cost = current_row[target_index - 1] + 1
            deletion_cost = previous_row[target_index] + 1
            substitution_cost = previous_row[target_index - 1] + (source_char != target_char)
            current_row.append(min(insertion_cost, deletion_cost, substitution_cost))
        previous_row = current_row

    return previous_row[-1]


def find_nearest_pattern(pattern: str, vocabulary: set[str] | list[str] | tuple[str, ...]) -> dict[str, object]:
    """Resolve a pattern against a known SAX vocabulary."""
    known_patterns = sorted(set(vocabulary))
    if not known_patterns:
        raise ValueError("vocabulary must contain at least one pattern")

    if pattern in known_patterns:
        return {
            "status": "known",
            "mapped_to": pattern,
            "distance": 0,
        }

    mapped_pattern, distance = min(
        ((candidate, levenshtein_distance(pattern, candidate)) for candidate in known_patterns),
        key=lambda item: (item[1], item[0]),
    )
    return {
        "status": "unseen",
        "mapped_to": mapped_pattern,
        "distance": distance,
    }

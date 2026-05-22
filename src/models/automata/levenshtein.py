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

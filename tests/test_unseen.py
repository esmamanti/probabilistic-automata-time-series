import pytest

from models.automata.unseen_handler import PatternResolution, UnseenPatternHandler


def test_unseen_handler_returns_seen_when_pattern_known():
    handler = UnseenPatternHandler()

    resolution = handler.resolve("abcd", ["zzzz", "abcd", "abce"])

    assert resolution.status == "seen"
    assert resolution.mapped_pattern == "abcd"
    assert resolution.distance == 0


def test_unseen_handler_returns_unseen_and_maps_to_nearest_pattern():
    handler = UnseenPatternHandler()

    resolution = handler.resolve("abcf", ["abce", "zzzz", "bbbb"])

    assert resolution.status == "unseen"
    assert resolution.mapped_pattern == "abce"
    assert resolution.distance == 1


def test_unseen_handler_raises_value_error_when_known_patterns_empty():
    handler = UnseenPatternHandler()

    with pytest.raises(ValueError, match="known_patterns must contain at least one pattern"):
        handler.resolve("abcd", [])


def test_pattern_resolution_dataclass_sets_all_fields_correctly():
    resolution = PatternResolution(
        original_pattern="abcf",
        status="unseen",
        mapped_pattern="abce",
        distance=1,
    )

    assert resolution.original_pattern == "abcf"
    assert resolution.status == "unseen"
    assert resolution.mapped_pattern == "abce"
    assert resolution.distance == 1


def test_unseen_handler_selects_pattern_with_smallest_levenshtein_distance():
    handler = UnseenPatternHandler()

    resolution = handler.resolve("abc", ["axxx", "abx", "a", "zzzz"])

    assert resolution.status == "unseen"
    assert resolution.mapped_pattern == "abx"
    assert resolution.distance == 1

from __future__ import annotations

import pytest

from models.automata.unseen_handler import PatternResolution, UnseenPatternHandler


def test_resolve_returns_seen_when_pattern_exists():
    handler = UnseenPatternHandler()

    resolution = handler.resolve("abcd", ["zzzz", "abcd", "abce"])

    assert resolution == PatternResolution(
        original_pattern="abcd",
        status="seen",
        mapped_pattern="abcd",
        distance=0,
    )


def test_resolve_maps_unseen_pattern_to_nearest_candidate():
    handler = UnseenPatternHandler()

    resolution = handler.resolve("abcf", ["abce", "bbbb", "zzzz"])

    assert resolution.status == "unseen"
    assert resolution.mapped_pattern == "abce"
    assert resolution.distance == 1


def test_resolve_uses_lexicographic_tiebreak_when_distances_match():
    handler = UnseenPatternHandler()

    resolution = handler.resolve("abc", ["abd", "abb"])

    assert resolution.status == "unseen"
    assert resolution.mapped_pattern == "abb"
    assert resolution.distance == 1


def test_resolve_raises_for_empty_known_pattern_list():
    handler = UnseenPatternHandler()

    with pytest.raises(ValueError, match="known_patterns must contain at least one pattern"):
        handler.resolve("abcd", [])


def test_pattern_resolution_preserves_all_fields():
    resolution = PatternResolution(
        original_pattern="adc",
        status="unseen",
        mapped_pattern="abc",
        distance=1,
    )

    assert resolution.original_pattern == "adc"
    assert resolution.status == "unseen"
    assert resolution.mapped_pattern == "abc"
    assert resolution.distance == 1

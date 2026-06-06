import pytest

from src.models.automata.levenshtein import find_nearest_pattern, levenshtein_distance


class TestLevenshteinDistance:
    def test_exact_match(self):
        sax_vocabulary = {"aab", "abc", "bcc"}

        result = find_nearest_pattern("aab", sax_vocabulary)

        assert result["status"] == "known"
        assert result["mapped_to"] == "aab"
        assert result["distance"] == 0

    def test_unseen_one_edit(self):
        sax_vocabulary = {"aab", "abc", "bcc"}

        result = find_nearest_pattern("aac", sax_vocabulary)

        assert result["status"] == "unseen"
        assert result["mapped_to"] == "aab"
        assert result["distance"] == 1

    def test_unseen_multiple_candidates(self):
        sax_vocabulary = {"aab", "bbc", "ccc"}

        result = find_nearest_pattern("abb", sax_vocabulary)

        assert result["status"] == "unseen"
        assert result["distance"] <= 2
        assert result["mapped_to"] in sax_vocabulary

    def test_empty_vocabulary(self):
        with pytest.raises(ValueError, match="vocabulary must contain at least one pattern"):
            find_nearest_pattern("abc", set())

    def test_symmetry(self):
        assert levenshtein_distance("abc", "bca") == levenshtein_distance("bca", "abc")

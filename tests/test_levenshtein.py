from models.automata.levenshtein import levenshtein_distance


def test_levenshtein_distance_matches_expected_edits():
    assert levenshtein_distance("abc", "abc") == 0
    assert levenshtein_distance("abc", "adc") == 1
    assert levenshtein_distance("abc", "ab") == 1
    assert levenshtein_distance("", "abc") == 3

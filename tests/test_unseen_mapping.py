from models.automata.unseen_handler import UnseenPatternHandler


def test_unseen_mapping_uses_nearest_levenshtein_match():
    handler = UnseenPatternHandler()

    resolution = handler.resolve("abcf", ["abce", "zzzz", "bbbb"])

    assert resolution.original_pattern == "abcf"
    assert resolution.mapped_pattern == "abce"
    assert resolution.distance == 1
    assert resolution.status == "unseen"

import numpy as np

from models.deep_learning.trainer import compute_pos_weight_from_targets


def test_pos_weight_matches_negative_positive_ratio():
    targets = np.asarray([0, 0, 0, 1, 1], dtype=int)

    pos_weight = compute_pos_weight_from_targets(targets)

    assert pos_weight == 3 / 2


def test_pos_weight_does_not_fail_when_positive_count_is_zero():
    targets = np.asarray([0, 0, 0, 0], dtype=int)

    pos_weight = compute_pos_weight_from_targets(targets)

    assert pos_weight == 1.0

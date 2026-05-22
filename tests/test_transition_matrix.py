from models.automata.transition_matrix import TransitionMatrixBuilder


def test_transition_matrix_counts_repeated_transitions():
    builder = TransitionMatrixBuilder()

    counts = builder.build([0, 1, 0, 1, 2])

    assert counts == {
        0: {1: 2},
        1: {0: 1, 2: 1},
    }

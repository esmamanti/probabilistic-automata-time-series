from experiments.run_parameter_analysis import transition_density


def test_transition_density_uses_observed_edges_over_state_square():
    transition_counts = {
        0: {0: 2, 1: 1},
        1: {1: 3},
    }

    density = transition_density(transition_counts, state_count=3)

    assert density == 3 / 9


def test_transition_density_returns_zero_for_empty_state_space():
    assert transition_density({}, state_count=0) == 0.0

import math

from models.automata.probability_engine import ProbabilityEngine


def test_transition_probability_uses_smoothing_for_unseen_edges():
    engine = ProbabilityEngine(smoothing=True, epsilon=1e-3)
    probabilities = {0: {1: 0.75}}

    assert math.isclose(engine.transition_probability(0, 1, probabilities), 0.75)
    assert math.isclose(engine.transition_probability(0, 2, probabilities), 1e-3)


def test_path_probability_multiplies_step_probabilities():
    engine = ProbabilityEngine(smoothing=False, epsilon=1e-3)
    probabilities = {0: {1: 0.5}, 1: {2: 0.25}}

    assert math.isclose(engine.path_probability([0, 1, 2], probabilities), 0.125)

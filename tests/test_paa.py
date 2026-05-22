import numpy as np

from models.automata.paa import PAATransformer


def test_paa_computes_piecewise_means():
    transformer = PAATransformer(window_size=2)

    result = transformer.transform([1.0, 3.0, 5.0, 7.0, 9.0])

    assert np.allclose(result, np.array([2.0, 6.0, 9.0]))

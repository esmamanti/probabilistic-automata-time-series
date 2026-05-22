from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.automata.paa import PAATransformer
from models.automata.probability_engine import ProbabilityEngine
from models.automata.sax import SAXTransformer
from models.automata.sliding_window import SlidingWindow
from models.automata.state_generator import StateGenerator
from models.automata.transition_matrix import TransitionMatrixBuilder


@dataclass
class AutomataArtifacts:
    paa_values: np.ndarray
    symbols: list[str]
    patterns: list[str]
    states: list[int]


class ProbabilisticAutomataModel:
    """End-to-end symbolic automata pipeline for 1D time-series."""

    def __init__(
        self,
        paa_window_size: int,
        alphabet_size: int,
        pattern_window_size: int,
        stride: int = 1,
        smoothing: bool = True,
        epsilon: float = 1e-4,
    ):
        self.paa = PAATransformer(window_size=paa_window_size)
        self.sax = SAXTransformer(alphabet_size=alphabet_size)
        self.sliding_window = SlidingWindow(size=pattern_window_size, stride=stride)
        self.state_generator = StateGenerator()
        self.transition_builder = TransitionMatrixBuilder()
        self.probability_engine = ProbabilityEngine(smoothing=smoothing, epsilon=epsilon)

        self.transition_counts_: dict[int, dict[int, int]] | None = None
        self.transition_probabilities_: dict[int, dict[int, float]] | None = None

    def _series_to_artifacts(self, series: np.ndarray | list[float]) -> AutomataArtifacts:
        paa_values = self.paa.transform(series)
        symbols = self.sax.transform(paa_values)
        patterns = self.sliding_window.transform(symbols)
        if not patterns:
            raise ValueError("Not enough symbolic values to create automata patterns")

        if self.state_generator.pattern_to_state:
            states = self.state_generator.transform(patterns)
        else:
            states = self.state_generator.fit_transform(patterns)

        return AutomataArtifacts(
            paa_values=paa_values,
            symbols=symbols,
            patterns=patterns,
            states=states,
        )

    def fit(self, series: np.ndarray | list[float]) -> "ProbabilisticAutomataModel":
        artifacts = self._series_to_artifacts(series)
        self.transition_counts_ = self.transition_builder.build(artifacts.states)
        self.transition_probabilities_ = self.probability_engine.build_transition_probabilities(
            self.transition_counts_,
            state_count=len(self.state_generator.pattern_to_state),
        )
        return self

    def transform(self, series: np.ndarray | list[float]) -> AutomataArtifacts:
        if self.transition_probabilities_ is None:
            raise RuntimeError("Model must be fitted before transform")
        return self._series_to_artifacts(series)

    def score_sequence(self, series: np.ndarray | list[float]) -> dict[str, object]:
        if self.transition_probabilities_ is None:
            raise RuntimeError("Model must be fitted before scoring")

        artifacts = self.transform(series)
        path_probability = self.probability_engine.path_probability(
            artifacts.states,
            self.transition_probabilities_,
        )
        average_log_probability = self.probability_engine.average_log_probability(
            artifacts.states,
            self.transition_probabilities_,
        )

        return {
            "path_probability": path_probability,
            "average_log_probability": average_log_probability,
            "patterns": artifacts.patterns,
            "states": artifacts.states,
        }

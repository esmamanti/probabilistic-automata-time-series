from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from evaluation.metrics import build_curve_frame, validate_aligned_binary_predictions, validate_binary_targets


def save_figure(figure: plt.Figure, output_path: str | Path, *, dpi: int = 150) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def plot_confusion_matrix(
    y_true,
    y_pred,
    *,
    title: str = "Confusion Matrix",
    normalize: bool = False,
) -> plt.Figure:
    true_array, pred_array = validate_aligned_binary_predictions(y_true, y_pred)
    matrix = confusion_matrix(true_array, pred_array, labels=[0, 1])
    heatmap_values = matrix.astype(float)
    annotation_values = matrix
    fmt = "d"
    if normalize:
        row_sums = heatmap_values.sum(axis=1, keepdims=True)
        heatmap_values = heatmap_values / row_sums.clip(min=1.0)
        annotation_values = heatmap_values
        fmt = ".2f"

    figure, axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        heatmap_values,
        annot=annotation_values,
        fmt=fmt,
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    figure.tight_layout()
    return figure


def plot_roc_curve(
    y_true,
    y_score,
    *,
    title: str = "ROC Curve",
) -> plt.Figure:
    validate_binary_targets(y_true)
    curve_df = build_curve_frame("roc", y_true, y_score)

    figure, axis = plt.subplots(figsize=(5, 4))
    axis.plot(curve_df["false_positive_rate"], curve_df["true_positive_rate"], label="ROC")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    axis.set_title(title)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.legend()
    figure.tight_layout()
    return figure


def plot_precision_recall_curve(
    y_true,
    y_score,
    *,
    title: str = "Precision-Recall Curve",
) -> plt.Figure:
    validate_binary_targets(y_true)
    curve_df = build_curve_frame("precision_recall", y_true, y_score)

    figure, axis = plt.subplots(figsize=(5, 4))
    axis.plot(curve_df["recall"], curve_df["precision"], label="PR")
    axis.set_title(title)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.legend()
    figure.tight_layout()
    return figure


def plot_metric_bars(
    metrics_df: pd.DataFrame,
    *,
    x: str,
    y: str,
    hue: str | None = None,
    title: str | None = None,
) -> plt.Figure:
    if x not in metrics_df.columns or y not in metrics_df.columns:
        raise KeyError(f"Required columns '{x}' and/or '{y}' are missing from metrics frame")

    figure, axis = plt.subplots(figsize=(7, 4))
    sns.barplot(data=metrics_df, x=x, y=y, hue=hue, ax=axis)
    axis.set_title(title or f"{y} by {x}")
    axis.set_xlabel(x)
    axis.set_ylabel(y)
    if hue is not None:
        axis.legend(title=hue)
    figure.tight_layout()
    return figure


def plot_parameter_sensitivity(
    metrics_df: pd.DataFrame,
    *,
    x: str,
    y: str,
    hue: str | None = None,
    style: str | None = None,
    title: str = "Parameter Sensitivity",
) -> plt.Figure:
    if x not in metrics_df.columns or y not in metrics_df.columns:
        raise KeyError(f"Required columns '{x}' and/or '{y}' are missing from metrics frame")

    figure, axis = plt.subplots(figsize=(7, 4))
    sns.lineplot(data=metrics_df, x=x, y=y, hue=hue, style=style, marker="o", ax=axis)
    axis.set_title(title)
    axis.set_xlabel(x)
    axis.set_ylabel(y)
    figure.tight_layout()
    return figure


def plot_transition_probability_heatmap(
    transition_probabilities: dict[int, dict[int, float]],
    *,
    state_labels: dict[int, str] | None = None,
    title: str = "Transition Probability Heatmap",
) -> plt.Figure:
    state_ids = sorted(
        {
            *transition_probabilities.keys(),
            *(target_state for targets in transition_probabilities.values() for target_state in targets.keys()),
        }
    )
    if not state_ids:
        raise ValueError("transition_probabilities must contain at least one state")

    matrix = np.zeros((len(state_ids), len(state_ids)), dtype=float)
    state_index = {state_id: index for index, state_id in enumerate(state_ids)}

    for source_state, targets in transition_probabilities.items():
        for target_state, probability in targets.items():
            matrix[state_index[source_state], state_index[target_state]] = float(probability)

    labels = [state_labels.get(state_id, str(state_id)) if state_labels else str(state_id) for state_id in state_ids]
    figure, axis = plt.subplots(figsize=(max(6, len(labels) * 0.5), max(5, len(labels) * 0.45)))
    sns.heatmap(
        matrix,
        cmap="mako",
        annot=len(labels) <= 20,
        fmt=".2f",
        xticklabels=labels,
        yticklabels=labels,
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel("To state")
    axis.set_ylabel("From state")
    figure.tight_layout()
    return figure


def plot_automata_state_diagram(
    transition_probabilities: dict[int, dict[int, float]],
    *,
    state_labels: dict[int, str] | None = None,
    title: str = "Automata State Diagram",
    probability_threshold: float = 0.0,
) -> plt.Figure:
    graph = nx.DiGraph()
    for source_state, targets in transition_probabilities.items():
        graph.add_node(source_state)
        for target_state, probability in targets.items():
            if float(probability) >= probability_threshold:
                graph.add_edge(source_state, target_state, weight=float(probability))
                graph.add_node(target_state)

    if graph.number_of_nodes() == 0:
        raise ValueError("No graph edges remain after applying probability_threshold")

    positions = nx.spring_layout(graph, seed=42)
    edge_weights = [graph.edges[edge]["weight"] for edge in graph.edges]
    scaled_widths = [1.0 + (weight * 4.0) for weight in edge_weights]
    node_labels = {node: state_labels.get(node, str(node)) if state_labels else str(node) for node in graph.nodes}
    edge_labels = {(u, v): f"{data['weight']:.2f}" for u, v, data in graph.edges(data=True)}

    figure, axis = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_nodes(graph, positions, node_size=1400, node_color="#d7eaf3", edgecolors="#23577a", ax=axis)
    nx.draw_networkx_labels(graph, positions, labels=node_labels, font_size=8, ax=axis)
    nx.draw_networkx_edges(
        graph,
        positions,
        width=scaled_widths,
        edge_color=edge_weights,
        edge_cmap=plt.cm.Blues,
        arrows=True,
        arrowsize=18,
        ax=axis,
    )
    nx.draw_networkx_edge_labels(graph, positions, edge_labels=edge_labels, font_size=7, ax=axis, rotate=False)
    axis.set_title(title)
    axis.axis("off")
    figure.tight_layout()
    return figure

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from evaluation.metrics import build_curve_frame, validate_aligned_binary_predictions, validate_binary_targets

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - exercised indirectly in environments without seaborn
    sns = None


def save_figure(figure: plt.Figure, output_path: str | Path, *, dpi: int = 150) -> Path:
    path = Path(output_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        return path
    except OSError:
        fallback_dir = Path.cwd() / "results" / "figures" / "_fallback_outputs"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        fallback_path = fallback_dir / path.name
        figure.savefig(fallback_path, dpi=dpi, bbox_inches="tight")
        return fallback_path


def _draw_heatmap(
    axis: plt.Axes,
    values: np.ndarray,
    *,
    annotations: np.ndarray | None = None,
    fmt: str = ".2f",
    cmap: str = "viridis",
    cbar: bool = True,
    xticklabels: list[str] | None = None,
    yticklabels: list[str] | None = None,
) -> None:
    if sns is not None:
        sns.heatmap(
            values,
            annot=annotations,
            fmt=fmt,
            cmap=cmap,
            cbar=cbar,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            ax=axis,
        )
        return

    available_colormaps = set(plt.colormaps())
    resolved_cmap = cmap if cmap in available_colormaps else "viridis"
    image = axis.imshow(values, cmap=resolved_cmap, aspect="auto")
    if cbar:
        axis.figure.colorbar(image, ax=axis)

    if xticklabels is not None:
        axis.set_xticks(np.arange(len(xticklabels)))
        axis.set_xticklabels(xticklabels)
    if yticklabels is not None:
        axis.set_yticks(np.arange(len(yticklabels)))
        axis.set_yticklabels(yticklabels)

    if annotations is not None:
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                axis.text(
                    column_index,
                    row_index,
                    format(annotations[row_index, column_index], fmt),
                    ha="center",
                    va="center",
                    color="black",
                )


def _draw_barplot(
    axis: plt.Axes,
    metrics_df: pd.DataFrame,
    *,
    x: str,
    y: str,
    hue: str | None,
) -> None:
    if sns is not None:
        sns.barplot(data=metrics_df, x=x, y=y, hue=hue, ax=axis)
        return

    if hue is None:
        categories = metrics_df[x].astype(str).tolist()
        values = metrics_df[y].astype(float).tolist()
        axis.bar(categories, values)
        return

    pivoted = metrics_df.pivot_table(index=x, columns=hue, values=y, aggfunc="mean")
    categories = pivoted.index.astype(str).tolist()
    hue_values = pivoted.columns.astype(str).tolist()
    positions = np.arange(len(categories))
    width = 0.8 / max(1, len(hue_values))
    for hue_index, hue_value in enumerate(hue_values):
        offsets = positions + ((hue_index - (len(hue_values) - 1) / 2.0) * width)
        axis.bar(offsets, pivoted[hue_value].to_numpy(dtype=float), width=width, label=hue_value)
    axis.set_xticks(positions)
    axis.set_xticklabels(categories)


def _draw_lineplot(
    axis: plt.Axes,
    metrics_df: pd.DataFrame,
    *,
    x: str,
    y: str,
    hue: str | None,
) -> None:
    if sns is not None:
        sns.lineplot(data=metrics_df, x=x, y=y, hue=hue, marker="o", ax=axis)
        return

    if hue is None:
        ordered = metrics_df.sort_values(x, kind="stable")
        axis.plot(ordered[x], ordered[y], marker="o")
        return

    for hue_value, group_df in metrics_df.groupby(hue, dropna=False):
        ordered = group_df.sort_values(x, kind="stable")
        axis.plot(ordered[x], ordered[y], marker="o", label=str(hue_value))


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
    _draw_heatmap(
        axis,
        heatmap_values,
        annotations=annotation_values,
        fmt=fmt,
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
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
    _draw_barplot(axis, metrics_df, x=x, y=y, hue=hue)
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
    _draw_lineplot(axis, metrics_df, x=x, y=y, hue=hue)
    axis.set_title(title)
    axis.set_xlabel(x)
    axis.set_ylabel(y)
    figure.tight_layout()
    return figure


def plot_metric_heatmap(
    metrics_df: pd.DataFrame,
    *,
    index: str,
    columns: str,
    values: str,
    title: str,
    cmap: str = "viridis",
    fmt: str = ".2f",
) -> plt.Figure:
    required_columns = {index, columns, values}
    missing_columns = required_columns.difference(metrics_df.columns)
    if missing_columns:
        raise KeyError(f"Required columns are missing from metrics frame: {sorted(missing_columns)}")

    pivoted = metrics_df.pivot_table(index=index, columns=columns, values=values, aggfunc="mean")
    if pivoted.empty:
        raise ValueError("metrics_df does not contain any values to plot")

    figure, axis = plt.subplots(figsize=(6, 5))
    _draw_heatmap(
        axis,
        pivoted.to_numpy(dtype=float),
        annotations=pivoted.to_numpy(dtype=float),
        fmt=fmt,
        cmap=cmap,
        xticklabels=[str(label) for label in pivoted.columns.tolist()],
        yticklabels=[str(label) for label in pivoted.index.tolist()],
    )
    axis.set_title(title)
    axis.set_xlabel(columns)
    axis.set_ylabel(index)
    figure.tight_layout()
    return figure


def plot_histogram_by_label(
    metrics_df: pd.DataFrame,
    *,
    value_column: str,
    label_column: str,
    title: str,
    bins: int = 20,
) -> plt.Figure:
    required_columns = {value_column, label_column}
    missing_columns = required_columns.difference(metrics_df.columns)
    if missing_columns:
        raise KeyError(f"Required columns are missing from metrics frame: {sorted(missing_columns)}")

    figure, axis = plt.subplots(figsize=(7, 4))
    grouped = metrics_df[[value_column, label_column]].dropna(subset=[value_column]).groupby(label_column, dropna=False)
    plotted = False
    for label_value, group_df in grouped:
        axis.hist(
            group_df[value_column].astype(float).to_numpy(),
            bins=bins,
            alpha=0.55,
            label=str(label_value),
        )
        plotted = True

    if not plotted:
        raise ValueError("metrics_df does not contain any values to plot")

    axis.set_title(title)
    axis.set_xlabel(value_column)
    axis.set_ylabel("Count")
    axis.legend(title=label_column)
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
    _draw_heatmap(
        axis,
        matrix,
        cmap="mako",
        annotations=matrix if len(labels) <= 20 else None,
        fmt=".2f",
        xticklabels=labels,
        yticklabels=labels,
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

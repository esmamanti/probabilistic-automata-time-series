from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"
RESULTS_RUNTIME = PROJECT_ROOT / "results" / "runtime"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_PATH = REPORTS_DIR / "ek_tablo_raporu.md"

MODEL_ORDER = ["LSTM", "GRU", "CNN", "AUTOMATA"]
MODEL_LABELS = {
    "LSTM": "LSTM",
    "GRU": "GRU",
    "CNN": "1D-CNN",
    "AUTOMATA": "Automata",
}
GRID_VALUES = [3, 4, 5, 6]
PREFERRED_VERSION_ORDER = [
    "tuned_threshold_weighted_loss",
    "tuned_threshold",
    "baseline_0.5_threshold",
]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def read_runtime_frame() -> pd.DataFrame:
    deep_runtime_path = RESULTS_TABLES / "deep_learning_runtime_summary.csv"
    if deep_runtime_path.exists():
        return read_csv(deep_runtime_path)
    runtime_comparison_path = RESULTS_RUNTIME / "runtime_comparison.csv"
    if runtime_comparison_path.exists():
        return read_csv(runtime_comparison_path)
    raise FileNotFoundError("Neither deep_learning_runtime_summary.csv nor runtime_comparison.csv was found.")


def to_float(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value)


def format_decimal(value: object, digits: int = 4) -> str:
    numeric = to_float(value)
    if pd.isna(numeric):
        return "N/A"
    return f"{numeric:.{digits}f}"


def format_pm(mean_value: object, std_value: object, digits: int = 4) -> str:
    mean_numeric = to_float(mean_value)
    std_numeric = to_float(std_value)
    if pd.isna(mean_numeric) or pd.isna(std_numeric):
        return "N/A"
    return f"{mean_numeric:.{digits}f} +- {std_numeric:.{digits}f}"


def normalize_model_name(model_name: object) -> str:
    return str(model_name).strip().upper()


def choose_preferred_version(frame: pd.DataFrame) -> pd.DataFrame:
    if "version" not in frame.columns:
        return frame.copy()

    available_versions = {str(value) for value in frame["version"].dropna().unique()}
    for candidate in PREFERRED_VERSION_ORDER:
        if candidate in available_versions:
            return frame[frame["version"] == candidate].copy()
    if not available_versions:
        return frame.copy()
    first_version = sorted(available_versions)[0]
    return frame[frame["version"] == first_version].copy()


def build_performance_lookup(deep_summary: pd.DataFrame, automata_summary: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float]]:
    lookup: dict[tuple[str, str], tuple[float, float]] = {}

    deep_frame = choose_preferred_version(deep_summary)
    deep_frame = deep_frame.assign(
        dataset=deep_frame["dataset"].astype(str).str.upper(),
        model=deep_frame["model"].map(normalize_model_name),
    )
    deep_aggregated = (
        deep_frame.groupby(["dataset", "model"], dropna=False)[["f1_score_mean", "f1_score_std"]]
        .mean()
        .reset_index()
    )
    for row in deep_aggregated.itertuples(index=False):
        lookup[(row.dataset, row.model)] = (float(row.f1_score_mean), float(row.f1_score_std))

    automata_frame = automata_summary.assign(
        dataset=automata_summary["dataset"].astype(str).str.upper(),
        model=automata_summary["model"].map(normalize_model_name),
    )
    automata_aggregated = (
        automata_frame.groupby(["dataset", "model"], dropna=False)[["f1_score_mean", "f1_score_std"]]
        .mean()
        .reset_index()
    )
    for row in automata_aggregated.itertuples(index=False):
        lookup[(row.dataset, row.model)] = (float(row.f1_score_mean), float(row.f1_score_std))

    return lookup


def build_noise_summary(noise_metrics: pd.DataFrame, unseen_metrics: pd.DataFrame | None) -> pd.DataFrame:
    frame = noise_metrics.assign(
        model=noise_metrics["model"].map(normalize_model_name),
        scenario=noise_metrics["scenario"].astype(str).str.lower(),
        noise_level=pd.to_numeric(noise_metrics["noise_level"], errors="coerce"),
        f1_score=pd.to_numeric(noise_metrics["f1_score"], errors="coerce"),
    )

    original = (
        frame[frame["scenario"] == "original"]
        .groupby("model", dropna=False)["f1_score"]
        .mean()
        .rename("original_f1")
    )
    noisy = (
        frame[(frame["scenario"] == "noise") & (frame["noise_level"] > 0)]
        .groupby("model", dropna=False)["f1_score"]
        .mean()
        .rename("noisy_f1")
    )

    summary = pd.concat([original, noisy], axis=1).reset_index()
    summary["f1_change"] = summary["noisy_f1"] - summary["original_f1"]

    if unseen_metrics is None or unseen_metrics.empty:
        summary["unseen_detection_rate"] = pd.NA
        summary["unseen_mapping_accuracy"] = pd.NA
        return summary

    unseen_frame = unseen_metrics.assign(
        model=unseen_metrics["model"].map(normalize_model_name),
        recall=pd.to_numeric(unseen_metrics["recall"], errors="coerce"),
        accuracy=pd.to_numeric(unseen_metrics["accuracy"], errors="coerce"),
    )
    unseen_summary = (
        unseen_frame.groupby("model", dropna=False)[["recall", "accuracy"]]
        .mean()
        .reset_index()
        .rename(columns={"recall": "unseen_detection_rate", "accuracy": "unseen_mapping_accuracy"})
    )
    return summary.merge(unseen_summary, on="model", how="left")


def build_parameter_tables(parameter_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = parameter_metrics.assign(
        dataset=parameter_metrics["dataset"].astype(str).str.upper(),
        window_size=pd.to_numeric(parameter_metrics["window_size"], errors="coerce"),
        alphabet_size=pd.to_numeric(parameter_metrics["alphabet_size"], errors="coerce"),
        f1_score=pd.to_numeric(parameter_metrics["f1_score"], errors="coerce"),
        state_count=pd.to_numeric(parameter_metrics["state_count"], errors="coerce"),
        transition_count=pd.to_numeric(parameter_metrics["transition_count"], errors="coerce"),
    )
    if "transition_density" in frame.columns:
        frame["transition_density"] = pd.to_numeric(frame["transition_density"], errors="coerce")
    else:
        frame["transition_density"] = frame["transition_count"] / (frame["state_count"] ** 2)

    batadal = frame[frame["dataset"] == "BATADAL"].copy()
    aggregated = (
        batadal.groupby(["window_size", "alphabet_size"], dropna=False)[["f1_score", "state_count", "transition_density"]]
        .mean()
        .reset_index()
    )
    f1_pivot = aggregated.pivot(index="window_size", columns="alphabet_size", values="f1_score")
    state_pivot = aggregated.pivot(index="window_size", columns="alphabet_size", values="state_count")
    density_pivot = aggregated.pivot(index="window_size", columns="alphabet_size", values="transition_density")
    return (
        f1_pivot.reindex(index=GRID_VALUES, columns=GRID_VALUES),
        state_pivot.reindex(index=GRID_VALUES, columns=GRID_VALUES),
        density_pivot.reindex(index=GRID_VALUES, columns=GRID_VALUES),
    )


def build_runtime_summary(deep_runtime: pd.DataFrame, automata_runtime: pd.DataFrame) -> pd.DataFrame:
    deep_frame = deep_runtime.assign(
        dataset=deep_runtime["dataset"].astype(str).str.upper(),
        model=deep_runtime["model"].map(normalize_model_name),
    )
    automata_frame = automata_runtime.assign(
        dataset=automata_runtime["dataset"].astype(str).str.upper(),
        model=automata_runtime["model"].map(normalize_model_name),
    )
    combined = pd.concat([deep_frame, automata_frame], ignore_index=True, sort=False)
    return (
        combined.groupby(["dataset", "model"], dropna=False)[["training_time_seconds_mean", "inference_time_seconds_mean"]]
        .mean()
        .reset_index()
    )


def markdown_grid_table(pivot_frame: pd.DataFrame, digits: int = 4) -> list[str]:
    lines = [
        "| Window Size \\ Alphabet Size | 3 | 4 | 5 | 6 |",
        "|-----------------------------|---|---|---|---|",
    ]
    for window_size in GRID_VALUES:
        cells = [str(window_size)]
        for alphabet_size in GRID_VALUES:
            cells.append(format_decimal(pivot_frame.loc[window_size, alphabet_size], digits=digits))
        lines.append(f"| {' | '.join(cells)} |")
    return lines


def build_report_markdown(
    performance_lookup: dict[tuple[str, str], tuple[float, float]],
    noise_summary: pd.DataFrame,
    f1_table: pd.DataFrame,
    state_table: pd.DataFrame,
    density_table: pd.DataFrame,
    runtime_summary: pd.DataFrame,
) -> str:
    lines: list[str] = [
        "# Ek Tablo Raporu",
        "",
        "Bu belge destekleyici tablolari tutar. Final sonuclarin ana kaynagi `reports/final_report.md` dosyasidir.",
        "",
        "## Tablo 1 - Model Performansi ve Stabilitesi",
        "",
        "| Model | SKAB (F1 +- std) | BATADAL (F1 +- std) |",
        "|-------|------------------|---------------------|",
    ]

    for model_name in MODEL_ORDER:
        skab_mean, skab_std = performance_lookup.get(("SKAB", model_name), (float("nan"), float("nan")))
        batadal_mean, batadal_std = performance_lookup.get(("BATADAL", model_name), (float("nan"), float("nan")))
        lines.append(f"| {MODEL_LABELS[model_name]} | {format_pm(skab_mean, skab_std)} | {format_pm(batadal_mean, batadal_std)} |")

    lines.extend(
        [
            "",
            "## Tablo 2 - Gurultu Etkisi ve Unseen Senaryo Analizi",
            "",
            "| Model | Orijinal F1 | Gurultulu F1 | F1 Degisimi | Unseen Det. Rate | Unseen Map. Acc. |",
            "|-------|-------------|--------------|-------------|------------------|------------------|",
        ]
    )

    noise_lookup = {normalize_model_name(row["model"]): row for row in noise_summary.to_dict(orient="records")}
    for model_name in MODEL_ORDER:
        row = noise_lookup.get(model_name, {})
        lines.append(
            f"| {MODEL_LABELS[model_name]} | {format_decimal(row.get('original_f1'))} | "
            f"{format_decimal(row.get('noisy_f1'))} | {format_decimal(row.get('f1_change'))} | "
            f"{format_decimal(row.get('unseen_detection_rate'))} | {format_decimal(row.get('unseen_mapping_accuracy'))} |"
        )

    lines.extend(
        [
            "",
            "## Tablo 4a - Automata F1-score Matrisi (BATADAL)",
            "",
            *markdown_grid_table(f1_table),
            "",
            "## Tablo 4b - Automata State Sayisi Matrisi (BATADAL)",
            "",
            *markdown_grid_table(state_table),
            "",
            "## Tablo 4c - Automata Gecis Yogunlugu Matrisi (BATADAL)",
            "",
            *markdown_grid_table(density_table),
            "",
            "## Tablo 5 - Egitim ve Cikarim Sureleri",
            "",
            "| Model | SKAB Egitim (sn) | SKAB Inference (sn) | BATADAL Egitim (sn) | BATADAL Inference (sn) |",
            "|-------|------------------|---------------------|---------------------|------------------------|",
        ]
    )

    runtime_lookup = {
        (str(row["dataset"]).upper(), normalize_model_name(row["model"])): row
        for row in runtime_summary.to_dict(orient="records")
    }
    for model_name in MODEL_ORDER:
        skab_row = runtime_lookup.get(("SKAB", model_name), {})
        batadal_row = runtime_lookup.get(("BATADAL", model_name), {})
        lines.append(
            f"| {MODEL_LABELS[model_name]} | "
            f"{format_decimal(skab_row.get('training_time_seconds_mean'))} | "
            f"{format_decimal(skab_row.get('inference_time_seconds_mean'))} | "
            f"{format_decimal(batadal_row.get('training_time_seconds_mean'))} | "
            f"{format_decimal(batadal_row.get('inference_time_seconds_mean'))} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    deep_learning_summary = read_csv(RESULTS_TABLES / "deep_learning_metrics_summary.csv")
    automata_summary = read_csv(RESULTS_TABLES / "automata_metrics_summary.csv")
    noise_metrics = read_csv(RESULTS_TABLES / "noise_experiment_metrics.csv")
    parameter_metrics = read_csv(RESULTS_TABLES / "parameter_analysis_metrics.csv")
    deep_runtime = read_runtime_frame()
    automata_runtime = read_csv(RESULTS_TABLES / "automata_runtime_summary.csv")
    unseen_metrics_path = RESULTS_TABLES / "unseen_metrics.csv"
    unseen_metrics = read_csv(unseen_metrics_path) if unseen_metrics_path.exists() else None

    performance_lookup = build_performance_lookup(deep_learning_summary, automata_summary)
    noise_summary = build_noise_summary(noise_metrics, unseen_metrics)
    f1_table, state_table, density_table = build_parameter_tables(parameter_metrics)
    runtime_summary = build_runtime_summary(deep_runtime, automata_runtime)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        build_report_markdown(
            performance_lookup=performance_lookup,
            noise_summary=noise_summary,
            f1_table=f1_table,
            state_table=state_table,
            density_table=density_table,
            runtime_summary=runtime_summary,
        ),
        encoding="utf-8",
    )
    print(f"Markdown report created at {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

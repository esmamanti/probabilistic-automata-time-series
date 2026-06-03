from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from uuid import uuid4

import pandas as pd

from experiments import generate_figures, run_all
from models.automata.automata_model import ProbabilisticAutomataModel
from models.automata.unseen_handler import UnseenPatternHandler
from utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_saved_deep_learning_artifacts_match_requirement_contract():
    metrics_df = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "deep_learning_metrics.csv", low_memory=False)
    predictions_df = pd.read_csv(PROJECT_ROOT / "results" / "explanations" / "deep_learning_predictions.csv", low_memory=False)

    assert {"CNN", "GRU", "LSTM"}.issubset(set(metrics_df["model"].dropna().astype(str)))
    assert set(metrics_df["seed"].dropna().astype(int)) == {7, 42, 123, 999, 2026}
    assert set(predictions_df.loc[predictions_df["dataset"] == "SKAB", "split"].dropna().astype(str)) == {
        "fold_0",
        "fold_1",
        "fold_2",
        "fold_3",
        "fold_4",
    }
    assert set(predictions_df.loc[predictions_df["dataset"] == "BATADAL", "split"].dropna().astype(str)) == {"test"}


def test_saved_noise_and_unseen_artifacts_cover_required_scenarios():
    noise_df = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "noise_experiment_metrics.csv")
    unseen_df = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "unseen_metrics.csv")

    assert {"original", "noise"}.issubset(set(noise_df["scenario"].dropna().astype(str)))
    assert {"SKAB", "BATADAL"}.issubset(set(noise_df["dataset"].dropna().astype(str)))
    assert {"DEEP", "AUTOMATA"}.issubset(set(noise_df["family"].dropna().astype(str)))
    assert {"SKAB", "BATADAL"}.issubset(set(unseen_df["dataset"].dropna().astype(str)))
    assert (PROJECT_ROOT / "results" / "explanations" / "unseen_summary.json").stat().st_size > 0
    assert (PROJECT_ROOT / "results" / "explanations" / "noise_experiment_summary.json").stat().st_size > 0


def test_parameter_analysis_grid_and_baseline_setting_are_complete():
    parameter_df = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "parameter_analysis_metrics.csv")
    models_config = load_config(PROJECT_ROOT / "configs" / "models.yaml")
    expected_grid = set(product(range(3, 7), range(3, 7)))

    for dataset_name in ("SKAB", "BATADAL"):
        observed_grid = set(
            zip(
                parameter_df.loc[parameter_df["dataset"] == dataset_name, "window_size"].astype(int),
                parameter_df.loc[parameter_df["dataset"] == dataset_name, "alphabet_size"].astype(int),
            )
        )
        assert observed_grid == expected_grid

    assert int(models_config["automata"]["paa"]["window_size"]) == 4
    assert int(models_config["automata"]["sliding_window"]["size"]) == 4
    assert int(models_config["automata"]["sax"]["alphabet_size"]) == 3


def test_unseen_handler_maps_unseen_pattern_to_expected_nearest_pattern():
    handler = UnseenPatternHandler()

    resolution = handler.resolve("adc", ["abc", "bbb", "ddd"])

    assert resolution.status == "unseen"
    assert resolution.mapped_pattern == "abc"
    assert resolution.distance == 1


def test_explanation_records_are_internally_consistent():
    model = ProbabilisticAutomataModel(
        paa_window_size=1,
        alphabet_size=3,
        pattern_window_size=2,
        anomaly_threshold=0.05,
        smoothing=True,
        epsilon=1e-3,
    )
    series = [0.0, 0.2, 0.4, 0.6, 0.8]

    model.fit(series)
    result = model.score_sequence(series)
    explanations = result["explanations"]

    assert explanations
    assert explanations[0]["path_probability"] == explanations[0]["confidence_score"]
    for index, explanation in enumerate(explanations):
        assert explanation["confidence_score"] == explanation["path_probability"]
        assert explanation["decision"] in {"normal", "anomaly"}
        if index == 0:
            assert explanation["previous_state"] is None
        else:
            previous = explanations[index - 1]
            expected_path_probability = previous["path_probability"] * explanation["transition_probability"]
            assert abs(explanation["path_probability"] - expected_path_probability) < 1e-12


def test_generate_figures_acceptance_writes_required_pngs(monkeypatch):
    tmp_path = PROJECT_ROOT / f".test-figures-{uuid4().hex}"
    results_dir = tmp_path / "results"
    explanations_dir = results_dir / "explanations"
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    explanations_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    predictions_df = pd.DataFrame(
        [
            {"dataset": "SKAB", "model": "CNN", "split": "fold_0", "seed": 42, "row_index": 0, "true_label": 0, "predicted_label": 0, "predicted_probability": 0.1},
            {"dataset": "SKAB", "model": "CNN", "split": "fold_0", "seed": 42, "row_index": 1, "true_label": 1, "predicted_label": 1, "predicted_probability": 0.9},
            {"dataset": "SKAB", "model": "CNN", "split": "fold_0", "seed": 42, "row_index": 2, "true_label": 0, "predicted_label": 0, "predicted_probability": 0.2},
            {"dataset": "SKAB", "model": "CNN", "split": "fold_0", "seed": 42, "row_index": 3, "true_label": 1, "predicted_label": 1, "predicted_probability": 0.8},
        ]
    )
    parameter_df = pd.DataFrame(
        [
            {"dataset": "SKAB", "window_size": 3, "alphabet_size": 3, "f1_score": 0.10},
            {"dataset": "SKAB", "window_size": 4, "alphabet_size": 3, "f1_score": 0.20},
            {"dataset": "SKAB", "window_size": 5, "alphabet_size": 4, "f1_score": 0.25},
            {"dataset": "SKAB", "window_size": 6, "alphabet_size": 5, "f1_score": 0.30},
        ]
    )
    predictions_df.to_csv(explanations_dir / "deep_learning_predictions.csv", index=False)
    parameter_df.to_csv(tables_dir / "parameter_analysis_metrics.csv", index=False)

    config = {
        "project": {"random_seeds": [42]},
        "paths": {
            "figures": "results/figures",
        }
    }
    models_config = {"automata": {}}

    class StubStateGenerator:
        state_to_pattern = {0: "aaaa", 1: "aaab"}

    class StubModel:
        transition_probabilities_ = {0: {0: 0.3, 1: 0.7}, 1: {1: 1.0}}
        state_generator = StubStateGenerator()

    monkeypatch.setattr(generate_figures, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(generate_figures, "load_config", lambda path: config if "config.yaml" in str(path) else models_config)
    monkeypatch.setattr(generate_figures, "build_automata_artifacts_for_skab", lambda config, models_config: StubModel())

    generate_figures.main()

    expected_files = {
        "confusion_matrix_best_model.png",
        "roc_curve_best_model.png",
        "precision_recall_curve_best_model.png",
        "automata_state_diagram_skab.png",
        "transition_probability_heatmap_skab.png",
        "parameter_sensitivity_skab.png",
    }
    produced_files = {path.name for path in figures_dir.iterdir() if path.is_file()}
    assert expected_files.issubset(produced_files)
    assert all((figures_dir / file_name).stat().st_size > 0 for file_name in expected_files)


def test_run_all_acceptance_creates_non_empty_required_outputs(monkeypatch):
    tmp_path = PROJECT_ROOT / f".test-run-all-{uuid4().hex}"
    config = {
        "paths": {
            "tables": "results/tables",
            "explanations": "results/explanations",
            "figures": "results/figures",
            "logs": "results/logs",
        }
    }
    experiments_config = {
        "experiments": {
            "original_data": {"enabled": True},
            "noisy_data": {"enabled": True},
            "unseen_data": {"enabled": True},
            "parameter_analysis": {"enabled": True},
        },
        "plots": {
            "confusion_matrix": True,
            "roc_curve": True,
            "precision_recall_curve": True,
            "automata_state_diagram": True,
            "transition_heatmap": True,
            "parameter_sensitivity_plot": True,
        },
    }

    def fake_load_config(path):
        return experiments_config if "experiments.yaml" in str(path) else config

    def write_file(relative_path: str, content: str) -> None:
        target_path = tmp_path / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content, encoding="utf-8")

    monkeypatch.setattr(run_all, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(run_all, "load_config", fake_load_config)
    monkeypatch.setattr(run_all.project_main, "main", lambda: write_file("results/logs/project.log", "started"))
    monkeypatch.setattr(run_all.run_automata, "main", lambda: write_file("results/explanations/automata_summary.json", json.dumps({"ok": True})))
    monkeypatch.setattr(run_all.run_deep_models, "main", lambda: write_file("results/tables/deep_learning_metrics.csv", "dataset,model\nSKAB,CNN\n"))
    monkeypatch.setattr(run_all.run_noise_experiment, "main", lambda: write_file("results/tables/noise_experiment_metrics.csv", "dataset,scenario\nSKAB,noise\n"))
    monkeypatch.setattr(run_all.run_unseen_experiment, "main", lambda: write_file("results/explanations/unseen_summary.json", json.dumps({"ok": True})))
    monkeypatch.setattr(run_all.run_parameter_analysis, "main", lambda: write_file("results/tables/parameter_analysis_metrics.csv", "dataset,window_size,alphabet_size\nSKAB,4,3\n"))
    monkeypatch.setattr(run_all.generate_figures, "main", lambda: write_file("results/figures/confusion_matrix_best_model.png", "png"))

    run_all.main()

    required_paths = [
        tmp_path / "results" / "logs" / "project.log",
        tmp_path / "results" / "explanations" / "automata_summary.json",
        tmp_path / "results" / "tables" / "deep_learning_metrics.csv",
        tmp_path / "results" / "tables" / "noise_experiment_metrics.csv",
        tmp_path / "results" / "explanations" / "unseen_summary.json",
        tmp_path / "results" / "tables" / "parameter_analysis_metrics.csv",
        tmp_path / "results" / "figures" / "confusion_matrix_best_model.png",
    ]
    assert all(path.exists() for path in required_paths)
    assert all(path.stat().st_size > 0 for path in required_paths)

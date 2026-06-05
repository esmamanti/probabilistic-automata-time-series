from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import main as project_main
from experiments import generate_figures, run_automata, run_deep_models, run_noise_experiment, run_parameter_analysis, run_unseen_experiment
from experiments import run_cross_dataset_experiment, run_explainability_export, run_runtime_analysis
from utils.config import load_config


STAGE_OUTPUTS = {
    "preprocessing": ["results/logs/project.log"],
    "training": [
        "results/tables/automata_skab_metrics.csv",
        "results/tables/automata_batadal_metrics.csv",
        "results/tables/automata_metrics_summary.csv",
        "results/tables/automata_runtime_metrics.csv",
        "results/tables/automata_runtime_summary.csv",
        "results/explanations/automata_summary.json",
        "results/tables/deep_learning_metrics.csv",
        "results/tables/deep_learning_metrics_summary.csv",
        "results/tables/deep_learning_runtime_metrics.csv",
        "results/tables/deep_learning_runtime_summary.csv",
        "results/explanations/deep_learning_predictions.csv",
        "results/explanations/deep_learning_summary.json",
        "results/thresholds/threshold_tuning_results.csv",
        "results/thresholds/probability_distribution.csv",
        "results/improvements/deep_learning_before_after.csv",
    ],
    "noise": [
        "results/tables/noise_experiment_metrics.csv",
        "results/explanations/noise_experiment_summary.json",
    ],
    "unseen": [
        "results/tables/unseen_metrics.csv",
        "results/explanations/unseen_summary.json",
    ],
    "cross_dataset": [
        "results/cross_dataset/cross_dataset_results.csv",
        "results/cross_dataset/cross_dataset_matrix.png",
    ],
    "parameter_analysis": [
        "results/tables/parameter_analysis_metrics.csv",
        "results/explanations/parameter_analysis_summary.json",
        "results/automata_analysis/state_transition_analysis.csv",
    ],
    "explainability": [
        "results/explanations/automata_explanations.csv",
        "results/explanations/automata_explanations.json",
        "results/explanations/confidence_histogram.png",
        "results/explanations/counterfactual_explanations.json",
    ],
    "runtime_analysis": [
        "results/runtime/runtime_comparison.csv",
        "results/runtime/runtime_comparison.png",
    ],
    "figures": [
        "results/figures/confusion_matrix_best_model.png",
        "results/figures/roc_curve_best_model.png",
        "results/figures/precision_recall_curve_best_model.png",
        "results/figures/automata_state_diagram_skab.png",
        "results/figures/transition_probability_heatmap_skab.png",
        "results/figures/parameter_sensitivity_skab.png",
    ],
}


def _path_to_label(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _collect_existing_outputs(config: dict) -> dict[str, list[Path]]:
    collected: dict[str, list[Path]] = {}
    for key in ("tables", "explanations", "figures", "logs"):
        directory = PROJECT_ROOT / config["paths"][key]
        files = sorted(path for path in directory.rglob("*") if path.is_file()) if directory.exists() else []
        collected[key] = files
    return collected


def _print_output_summary(collected_outputs: dict[str, list[Path]]) -> None:
    print()
    print("=== Generated Outputs ===")
    for category, files in collected_outputs.items():
        print(f"{category}: {len(files)} file(s)")
        for path in files:
            print(f"  - {_path_to_label(path)}")


def _ensure_logs_dir(config: dict) -> Path:
    logs_dir = PROJECT_ROOT / config["paths"]["logs"]
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def _progress_path(config: dict) -> Path:
    return _ensure_logs_dir(config) / "pipeline_progress.json"


def _stage_paths(stage_name: str) -> list[Path]:
    return [PROJECT_ROOT / relative_path for relative_path in STAGE_OUTPUTS.get(stage_name, [])]


def _stage_is_complete(stage_name: str) -> bool:
    expected_paths = _stage_paths(stage_name)
    return bool(expected_paths) and all(path.exists() and path.stat().st_size > 0 for path in expected_paths)


def _write_progress(config: dict, stage_name: str, status: str) -> None:
    progress_file = _progress_path(config)
    payload: dict[str, object]
    if progress_file.exists():
        with progress_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = {"stages": {}}

    payload.setdefault("stages", {})
    payload["stages"][stage_name] = {
        "status": status,
        "outputs": [str(path.relative_to(PROJECT_ROOT)) for path in _stage_paths(stage_name)],
    }
    with progress_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _run_stage(
    *,
    config: dict,
    stage_name: str,
    label: str,
    runner,
    resume_existing: bool,
) -> None:
    print()
    if resume_existing and _stage_is_complete(stage_name):
        print(f"=== Skipping {label} (existing outputs detected) ===")
        _write_progress(config, stage_name, "skipped_existing")
        return

    print(f"=== Running {label} ===")
    _write_progress(config, stage_name, "running")
    runner()
    _write_progress(config, stage_name, "completed")


def _run_original_training_and_test() -> None:
    run_automata.main()
    run_deep_models.main()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full experiment pipeline.")
    parser.add_argument(
        "--resume",
        dest="resume_existing",
        action="store_true",
        help="Resume and skip stages whose outputs already exist.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume_existing",
        action="store_false",
        help="Run all stages from scratch without skipping existing outputs.",
    )
    parser.set_defaults(resume_existing=None)
    return parser.parse_known_args()[0]


def main() -> None:
    args = _parse_args()
    config = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    experiments_config = load_config(PROJECT_ROOT / "configs" / "experiments.yaml")
    configured_resume_existing = bool(experiments_config.get("output", {}).get("resume_existing", False))
    resume_existing = configured_resume_existing if args.resume_existing is None else bool(args.resume_existing)

    print("=== Full Pipeline Started ===")
    print(f"Resume existing outputs: {resume_existing}")
    _run_stage(
        config=config,
        stage_name="preprocessing",
        label="Preprocessing",
        runner=project_main.main,
        resume_existing=resume_existing,
    )

    enabled_experiments = experiments_config.get("experiments", {})
    plots_config = experiments_config.get("plots", {})

    if enabled_experiments.get("original_data", {}).get("enabled", True):
        _run_stage(
            config=config,
            stage_name="training",
            label="Model Training and Original Test",
            runner=_run_original_training_and_test,
            resume_existing=resume_existing,
        )

    if enabled_experiments.get("noisy_data", {}).get("enabled", True):
        _run_stage(
            config=config,
            stage_name="noise",
            label="Noise Experiments",
            runner=run_noise_experiment.main,
            resume_existing=resume_existing,
        )

    if enabled_experiments.get("unseen_data", {}).get("enabled", True):
        _run_stage(
            config=config,
            stage_name="unseen",
            label="Unseen Experiments",
            runner=run_unseen_experiment.main,
            resume_existing=resume_existing,
        )

    if bool(config.get("cross_dataset", {}).get("enabled", False)):
        _run_stage(
            config=config,
            stage_name="cross_dataset",
            label="Cross-Dataset Test",
            runner=run_cross_dataset_experiment.main,
            resume_existing=resume_existing,
        )

    if enabled_experiments.get("parameter_analysis", {}).get("enabled", True):
        _run_stage(
            config=config,
            stage_name="parameter_analysis",
            label="Parameter Analysis",
            runner=run_parameter_analysis.main,
            resume_existing=resume_existing,
        )

    if bool(config.get("explainability", {}).get("save_json", False) or config.get("explainability", {}).get("save_csv", False)):
        _run_stage(
            config=config,
            stage_name="explainability",
            label="Explainability Export",
            runner=run_explainability_export.main,
            resume_existing=resume_existing,
        )

    if bool(config.get("runtime_analysis", {}).get("enabled", False)):
        _run_stage(
            config=config,
            stage_name="runtime_analysis",
            label="Runtime Analysis",
            runner=run_runtime_analysis.main,
            resume_existing=resume_existing,
        )

    if any(bool(enabled) for enabled in plots_config.values()):
        _run_stage(
            config=config,
            stage_name="figures",
            label="Figures",
            runner=generate_figures.main,
            resume_existing=resume_existing,
        )

    _print_output_summary(_collect_existing_outputs(config))
    print("All experiments completed successfully. Results saved under results/.")


if __name__ == "__main__":
    main()

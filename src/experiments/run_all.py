from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import main as project_main
from experiments import generate_figures, run_automata, run_deep_models, run_noise_experiment, run_parameter_analysis, run_unseen_experiment
from utils.config import load_config


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


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    experiments_config = load_config(PROJECT_ROOT / "configs" / "experiments.yaml")

    print("=== Full Pipeline Started ===")
    project_main.main()

    enabled_experiments = experiments_config.get("experiments", {})
    plots_config = experiments_config.get("plots", {})

    if enabled_experiments.get("original_data", {}).get("enabled", True):
        print()
        print("=== Running Automata Baseline Experiments ===")
        run_automata.main()
        print()
        print("=== Running Deep Learning Baseline Experiments ===")
        run_deep_models.main()

    if enabled_experiments.get("noisy_data", {}).get("enabled", True):
        print()
        print("=== Running Noise Experiments ===")
        run_noise_experiment.main()

    if enabled_experiments.get("unseen_data", {}).get("enabled", True):
        print()
        print("=== Running Unseen Experiments ===")
        run_unseen_experiment.main()

    if enabled_experiments.get("parameter_analysis", {}).get("enabled", True):
        print()
        print("=== Running Parameter Analysis ===")
        run_parameter_analysis.main()

    if any(bool(enabled) for enabled in plots_config.values()):
        print()
        print("=== Generating Figures ===")
        generate_figures.main()

    _print_output_summary(_collect_existing_outputs(config))


if __name__ == "__main__":
    main()

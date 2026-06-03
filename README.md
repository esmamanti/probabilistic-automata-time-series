# probabilistic-automata-time-series

This repository compares two families for time-series anomaly detection:

- Probabilistic automata with symbolic pattern transitions
- Deep sequential models: `LSTM`, `GRU`, `CNN`

The project is aligned to the Yazilim Gelistirme 2 requirements for:

- SKAB and BATADAL datasets
- leakage-safe preprocessing and splitting
- noise and unseen experiments
- automata parameter analysis
- statistical testing
- probabilistic explainability

## Required Dataset Usage

| Dataset | Target column | Required usage in this repo |
| --- | --- | --- |
| SKAB | `anomaly` | `valve1` and `valve2` are concatenated; `source_group` and `source_file` are added for traceability |
| BATADAL | `ATT_FLAG` | `BATADAL_dataset04.csv` is used; labels are remapped from `-999 -> 0` and `1 -> 1` |

Important requirement note:

- The BATADAL label column is explicitly `ATT_FLAG`.
- SKAB model inputs exclude `datetime`, `changepoint`, `source_group`, and `source_file`.
- BATADAL keeps time order and uses a strict `60/20/20` train/validation/test split.

## Architecture

- Central configuration: [configs/config.yaml](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/configs/config.yaml)
- Deep model registry: [configs/models.yaml](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/configs/models.yaml)
- Experiment toggles: [configs/experiments.yaml](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/configs/experiments.yaml)
- Full delivery pipeline: [src/experiments/run_all.py](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/src/experiments/run_all.py)

Key implementation points:

- preprocessing is fit only on train data
- PCA is reduced to one component before automata modeling
- SKAB uses group-aware fold evaluation
- BATADAL uses time-ordered evaluation
- automata unseen patterns are resolved with Levenshtein distance
- experiment context is stored in CSV/JSON outputs

## Produced Artifacts

Main outputs are written under:

- [results/tables](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/tables)
- [results/explanations](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/explanations)
- [results/figures](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/figures)

Important alignment notes with the requirements:

- `results/tables/deep_learning_metrics.csv` already includes `LSTM`, `GRU`, and `CNN`.
- `results/explanations/deep_learning_predictions.csv` stores SKAB fold splits as `fold_0` ... `fold_4`.
- `results/explanations/automata_explanation_example.json` provides a fixed explainability JSON example.
- runtime outputs are produced as `deep_learning_runtime_metrics.csv`, `deep_learning_runtime_summary.csv`, `automata_runtime_metrics.csv`, and `automata_runtime_summary.csv`.

## Model Comparison

The current deep-learning aggregate file is:

- [results/tables/deep_learning_metrics_summary.csv](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/tables/deep_learning_metrics_summary.csv)

Representative observations from the current artifacts:

- On SKAB, deep models outperform automata on F1.
- On SKAB, `CNN` is currently the strongest deep model on mean F1 in the saved aggregate outputs.
- On BATADAL, deep models preserve high accuracy but show weak anomaly recall/F1 because of class imbalance.
- Automata is weaker on raw predictive performance but stronger on interpretability and unseen-pattern traceability.

## Noise Effect

The noise experiment output is:

- [results/tables/noise_experiment_metrics.csv](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/tables/noise_experiment_metrics.csv)

Current qualitative reading:

- SKAB deep models are fairly stable under Gaussian noise.
- Automata changes more visibly under noise, but remains analyzable through transition behavior.
- BATADAL remains the harder dataset for anomaly recall.

## Unseen Behavior

The unseen outputs are:

- [results/tables/unseen_metrics.csv](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/tables/unseen_metrics.csv)
- [results/explanations/unseen_explanations.csv](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/explanations/unseen_explanations.csv)

Current qualitative reading:

- BATADAL produces a higher unseen-pattern ratio than SKAB.
- Unseen patterns are mapped to the nearest known pattern with Levenshtein distance.
- The automata pipeline keeps a trace of status, mapped pattern, distance, transition probability, path probability, and final decision.

## Parameter Effects

The parameter-analysis output is:

- [results/tables/parameter_analysis_metrics.csv](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/tables/parameter_analysis_metrics.csv)

Current qualitative reading:

- increasing `window_size` tends to expand the state space
- larger symbolic granularity can increase unseen patterns
- F1, state count, and transition density show a clear interpretability/performance trade-off

## Runtime Outputs

The runtime artifacts are produced during the experiment scripts:

- [results/tables/deep_learning_runtime_metrics.csv](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/tables/deep_learning_runtime_metrics.csv)
- [results/tables/deep_learning_runtime_summary.csv](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/tables/deep_learning_runtime_summary.csv)
- [results/tables/automata_runtime_metrics.csv](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/tables/automata_runtime_metrics.csv)
- [results/tables/automata_runtime_summary.csv](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/tables/automata_runtime_summary.csv)

These files capture per-run and aggregated:

- training time in seconds
- inference time in seconds
- evaluated split
- test example count

## Explainability Schema

The automata explanation pipeline reports:

- current state
- previous state
- observed pattern
- seen/unseen status
- mapped pattern for unseen cases
- transition probability
- path probability
- confidence score
- decision reason
- final decision

Example JSON output:

- [results/explanations/automata_explanation_example.json](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/explanations/automata_explanation_example.json)

Implementation and schema validation:

- [src/models/automata/explainability.py](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/src/models/automata/explainability.py)
- [tests/test_automata.py](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/tests/test_automata.py)

## Run

Single full pipeline command:

```bash
python src/experiments/run_all.py
```

Individual commands:

```bash
python src/main.py
python src/experiments/run_automata.py
python src/experiments/run_deep_models.py
python src/experiments/run_noise_experiment.py
python src/experiments/run_unseen_experiment.py
python src/experiments/run_parameter_analysis.py
python src/experiments/generate_figures.py
```

## Verification

Tests can be run from the project root with:

```bash
pytest -q
```

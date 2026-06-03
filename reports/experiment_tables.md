# Experiment Tables

## 1. Deep Learning and Automata Comparison

Primary artifact files:

- [results/tables/deep_learning_metrics_summary.csv](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/tables/deep_learning_metrics_summary.csv)
- [results/tables/model_comparison_metrics_summary.csv](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/tables/model_comparison_metrics_summary.csv)

Interpretation notes:

- SKAB fold-based outputs exist in the saved artifacts.
- `CNN`, `GRU`, and `LSTM` are all present in the current deep-learning outputs.
- BATADAL remains the harder dataset for anomaly recall.

## 2. Noise Effect

Primary artifact file:

- [results/tables/noise_experiment_metrics.csv](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/tables/noise_experiment_metrics.csv)

Key takeaway:

- deep models on SKAB are relatively stable under Gaussian noise
- automata changes are more visible but also more interpretable

## 3. Unseen Pattern Analysis

Primary artifact files:

- [results/tables/unseen_metrics.csv](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/tables/unseen_metrics.csv)
- [results/explanations/unseen_explanations.csv](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/explanations/unseen_explanations.csv)

Key takeaway:

- BATADAL produces more unseen patterns than SKAB in the current saved results
- unseen cases are mapped with Levenshtein nearest-pattern resolution

## 4. Automata Parameter Sensitivity

Primary artifact file:

- [results/tables/parameter_analysis_metrics.csv](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/tables/parameter_analysis_metrics.csv)

Key takeaway:

- state count and transition density change substantially with `window_size` and `alphabet_size`
- the best F1 setting is not always the simplest symbolic structure

## 5. Explainability Example

Primary artifact files:

- [results/explanations/automata_skab_explanations.csv](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/explanations/automata_skab_explanations.csv)
- [results/explanations/automata_explanation_example.json](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/explanations/automata_explanation_example.json)

Required fields covered:

- state and previous state
- pattern and mapped pattern
- seen/unseen status
- transition probability
- path probability
- confidence score
- decision reason
- final decision

## 6. Runtime Comparison

Primary artifact files:

- [results/tables/deep_learning_runtime_metrics.csv](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/tables/deep_learning_runtime_metrics.csv)
- [results/tables/deep_learning_runtime_summary.csv](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/tables/deep_learning_runtime_summary.csv)
- [results/tables/automata_runtime_metrics.csv](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/tables/automata_runtime_metrics.csv)
- [results/tables/automata_runtime_summary.csv](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/tables/automata_runtime_summary.csv)

Recorded fields:

- training time in seconds
- inference time in seconds
- split
- test example count

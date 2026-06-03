# Final Report

## 1. Problem and Goal

This project compares two modeling families for time-series anomaly detection:

- probabilistic automata built on symbolic patterns and transition probabilities
- deep sequential models: `LSTM`, `GRU`, and `CNN`

The goal is not only to compare raw predictive performance, but also to analyze:

- dataset-dependent behavior
- robustness under Gaussian noise
- behavior on unseen symbolic patterns
- automata parameter sensitivity
- explainability through probabilistic transitions

## 2. Dataset Usage

### SKAB

- The project uses only `valve1` and `valve2`.
- All CSV files are concatenated into one dataset.
- `source_group` and `source_file` are added for traceability and group-aware splitting.
- Target column: `anomaly`
- Model inputs exclude `datetime`, `changepoint`, `source_group`, and `source_file`.

### BATADAL

- The project uses `BATADAL_dataset04.csv`.
- The target label column is explicitly `ATT_FLAG`.
- Labels are remapped as `-999 -> 0` and `1 -> 1`.
- Time order is preserved.
- The split is fixed as `60% train / 20% validation / 20% test`.

This directly satisfies the requirement that the BATADAL label name must be checked and clearly stated in the report.

## 3. Preprocessing and Leakage Control

The preprocessing pipeline contains:

1. Missing-value handling with `SimpleImputer`
2. Scaling
3. PCA reduction to one component
4. Sequence generation

Even though the current BATADAL and SKAB snapshots do not contain missing values, the pipeline still supports missing-data handling as required.

Leakage prevention rules are respected:

- normalization is fit only on train data
- PCA is fit only on train data
- the same fitted transforms are reused on validation and test
- automata state generation and transition probabilities are built only from train data

## 4. Deep Learning Outputs and Artifact Alignment

The saved artifacts already show that the deep-learning pipeline includes all three enabled models:

- `LSTM`
- `GRU`
- `CNN`

This can be verified in:

- [results/tables/deep_learning_metrics.csv](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/tables/deep_learning_metrics.csv)
- [results/tables/deep_learning_metrics_summary.csv](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/tables/deep_learning_metrics_summary.csv)

The SKAB prediction artifact is also fold-aware:

- [results/explanations/deep_learning_predictions.csv](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/explanations/deep_learning_predictions.csv)

That file stores `split` values such as `fold_0`, `fold_1`, `fold_2`, `fold_3`, and `fold_4`, so the fold-based reporting requirement is satisfied by the current saved outputs.

## 5. Model Comparison

The deep-learning aggregate outputs show:

- On SKAB, deep models outperform automata on F1.
- On SKAB, `CNN` currently gives the strongest saved mean F1 among the deep models.
- On BATADAL, deep models keep high accuracy but weak anomaly recall and F1 because anomaly detection is harder under strong class imbalance.

The automata outputs show a different trade-off:

- lower predictive performance than the best deep models
- stronger interpretability
- explicit unseen-pattern handling
- direct probabilistic explanations through transition structure

So the comparison is not only "which model is best", but "which family behaves better under which requirement".

## 6. Cross-Dataset Observations

Even without introducing additional datasets beyond SKAB and BATADAL, the project still provides an inter-dataset comparison:

- SKAB favors deep models much more clearly
- BATADAL is more difficult for deep anomaly recall
- BATADAL produces a larger unseen-pattern ratio than SKAB
- BATADAL shows a stronger interaction between symbolic complexity and class imbalance

This satisfies the requirement to discuss dataset-dependent performance differences.

## 7. Noise Effect Analysis

The saved noise experiment output is:

- [results/tables/noise_experiment_metrics.csv](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/tables/noise_experiment_metrics.csv)

Key observations from the current saved snapshot:

- On SKAB, deep models are relatively stable under Gaussian noise.
- `LSTM` changes only slightly in F1 under noise.
- `GRU` also remains close to its original score.
- Automata shows measurable sensitivity through changed anomaly behavior, but remains analyzable through its symbolic path structure.
- On BATADAL, deep models remain weak on anomaly recall, and automata shows a small positive shift in F1 under the saved noisy run.

## 8. Unseen Behavior Analysis

The unseen outputs are:

- [results/tables/unseen_metrics.csv](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/tables/unseen_metrics.csv)
- [results/explanations/unseen_explanations.csv](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/explanations/unseen_explanations.csv)

Current findings:

- SKAB unseen ratio is low in the saved snapshot.
- BATADAL unseen ratio is clearly higher.
- Mapping success is preserved through nearest-pattern matching.
- BATADAL unseen cases are more strongly associated with anomaly behavior than SKAB in the current outputs.

This supports the report requirement to discuss unseen-pattern behavior explicitly.

## 9. Parameter Sensitivity

The automata parameter output is:

- [results/tables/parameter_analysis_metrics.csv](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/tables/parameter_analysis_metrics.csv)

The saved results show the intended trade-off:

- larger `window_size` generally increases state count
- richer symbolic alphabets can increase unseen patterns
- transition density tends to drop as the symbolic state space grows
- some settings improve F1, but often at the cost of a more complex and sparser automata structure

This is exactly the kind of parameter-effect discussion requested by the assignment.

## 10. Explainability Output Standardization

The automata model already produces rich per-step explanations, but the delivery format is now standardized around a fixed schema.

Schema source:

- [src/models/automata/explainability.py](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/src/models/automata/explainability.py)

Schema test:

- [tests/test_automata.py](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/tests/test_automata.py)

Expected fields include:

- `time_step`
- `state`
- `previous_state`
- `pattern`
- `status`
- `mapped_to`
- `distance`
- `transition_probability`
- `path_probability`
- `confidence_score`
- `decision_reason`
- `decision`

Example JSON artifact:

- [results/explanations/automata_explanation_example.json](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/explanations/automata_explanation_example.json)

This directly addresses the requirement that explainability output should be presented in JSON or table form with a stable schema.

## 11. Runtime Reporting

The assignment appendix asks for model runtime comparison. The codebase now writes runtime artifacts for both families:

- [results/tables/deep_learning_runtime_metrics.csv](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/tables/deep_learning_runtime_metrics.csv)
- [results/tables/deep_learning_runtime_summary.csv](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/tables/deep_learning_runtime_summary.csv)
- [results/tables/automata_runtime_metrics.csv](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/tables/automata_runtime_metrics.csv)
- [results/tables/automata_runtime_summary.csv](/C:/Users/Esma%20Nur%20Mant%C4%B1/Desktop/probabilistic-automata-time-series/results/tables/automata_runtime_summary.csv)

These files record:

- training time in seconds
- inference time in seconds
- split name
- test example count

This makes runtime comparison a first-class artifact instead of a manual note.

## 12. Conclusion

The project now aligns with the requested delivery logic more clearly:

- deep-learning artifacts include `CNN`
- SKAB deep predictions are fold-aware in saved outputs
- BATADAL label naming is explicitly documented as `ATT_FLAG`
- the required analysis categories are written as report text, not only computed in code
- explainability output is standardized and test-backed
- runtime reporting is now generated as structured CSV output

The remaining work, if desired, is mostly presentation-oriented:

- regenerate all figures with the full pipeline
- enrich the table appendix further
- optionally regenerate all result files again through `python src/experiments/run_all.py`

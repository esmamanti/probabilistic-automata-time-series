# probabilistic-automata-time-series

Bu repo, "From Black-Box to Explainability: Probabilistic Automata for Time Series Analysis" projesi icin zaman serisi anomali tespiti deneylerini uretir. Karsilastirilan iki ana model ailesi:

- Probabilistic automata tabanli sembolik model
- Deep learning modelleri: `LSTM`, `GRU`, `CNN`

## Dataset Summary

Kullanilan veri kumeleri:

- `SKAB`: sadece `valve1` ve `valve2`
- `BATADAL`: sadece `BATADAL_dataset04.csv` (Training Dataset 2)

Bu repoda `SWAT` veya `WADI` kullanilmaz.

## Model Comparison

Merkezi dosyalar:

- [configs/config.yaml](configs/config.yaml)
- [configs/models.yaml](configs/models.yaml)
- [configs/experiments.yaml](configs/experiments.yaml)
- [src/experiments/run_all.py](src/experiments/run_all.py)

Baslica ciktilar:

- [results/tables/deep_learning_metrics.csv](results/tables/deep_learning_metrics.csv)
- [results/tables/automata_metrics_summary.csv](results/tables/automata_metrics_summary.csv)
- [results/tables/model_comparison_metrics_summary.csv](results/tables/model_comparison_metrics_summary.csv)

## Cross-Dataset

Train/test yonleri:

- `SKAB -> BATADAL`
- `BATADAL -> SKAB`

Ciktilar:

- [results/cross_dataset/cross_dataset_results.csv](results/cross_dataset/cross_dataset_results.csv)
- [results/cross_dataset/cross_dataset_summary.csv](results/cross_dataset/cross_dataset_summary.csv)
- [results/cross_dataset/cross_dataset_matrix.png](results/cross_dataset/cross_dataset_matrix.png)

## Noise Robustness

Gaussian noise seviyeleri:

- `0.05`
- `0.10`
- `0.20`

Ciktilar:

- [results/noise/noise_robustness_results.csv](results/noise/noise_robustness_results.csv)
- [results/noise/noise_robustness_plot.png](results/noise/noise_robustness_plot.png)
- [results/tables/noise_experiment_metrics.csv](results/tables/noise_experiment_metrics.csv)

## Unseen Pattern

Levenshtein mapping ile unseen pattern analizi:

- [results/unseen/unseen_pattern_details.csv](results/unseen/unseen_pattern_details.csv)
- [results/unseen/unseen_distance_accuracy.csv](results/unseen/unseen_distance_accuracy.csv)
- [results/unseen/unseen_distance_accuracy.png](results/unseen/unseen_distance_accuracy.png)
- [results/explanations/unseen_summary.json](results/explanations/unseen_summary.json)

## State/Transition Analysis

Grid:

- `window_size: [3, 4, 5, 6]`
- `alphabet_size: [3, 4, 5, 6]`

Ciktilar:

- [results/automata_analysis/state_transition_analysis.csv](results/automata_analysis/state_transition_analysis.csv)
- [results/automata_analysis/state_count_vs_window.png](results/automata_analysis/state_count_vs_window.png)
- [results/automata_analysis/transition_density_vs_window.png](results/automata_analysis/transition_density_vs_window.png)
- [results/automata_analysis/f1_vs_window_alphabet.png](results/automata_analysis/f1_vs_window_alphabet.png)

## Confidence Score

Automata explanation export:

- [results/explanations/automata_explanations.csv](results/explanations/automata_explanations.csv)
- [results/explanations/automata_explanations.json](results/explanations/automata_explanations.json)
- [results/explanations/confidence_histogram.png](results/explanations/confidence_histogram.png)

Temel alanlar:

- `dataset`
- `time_step`
- `state`
- `pattern`
- `status`
- `mapped_to`
- `path_probability`
- `confidence_score`
- `decision`
- `true_label`

## Counterfactual

Unseen anomaly pattern'lar icin nearest seen pattern tabanli counterfactual export:

- [results/explanations/counterfactual_explanations.json](results/explanations/counterfactual_explanations.json)

## Runtime

Runtime karsilastirma ciktilari:

- [results/runtime/runtime_comparison.csv](results/runtime/runtime_comparison.csv)
- [results/runtime/runtime_comparison.png](results/runtime/runtime_comparison.png)

## Class Imbalance Problem in BATADAL

BATADAL veri setinde anomaly oranı düşük olduğu için deep learning modelleri yüksek accuracy üretmesine rağmen anomaly sınıfını kaçırma eğilimi gösterebilir.

## Why Accuracy is Misleading

BATADAL gibi dengesiz veri setlerinde sadece accuracy metriğine bakmak yeterli değildir. Bu nedenle recall ve F1-score öncelikli değerlendirilir.

## Threshold Calibration

Deep learning modelleri için validation set üzerinde `0.01` ile `0.99` arasında threshold taraması yapılır ve en iyi F1-score üreten threshold testte kullanılır.

İlgili çıktılar:

- [results/thresholds/threshold_tuning_results.csv](results/thresholds/threshold_tuning_results.csv)
- [results/thresholds/probability_distribution.csv](results/thresholds/probability_distribution.csv)
- [results/thresholds/probability_distribution.png](results/thresholds/probability_distribution.png)

## Weighted BCE Loss

Deep learning eğitiminde class imbalance azaltmak için `pos_weight = negative_count / positive_count` yaklaşımıyla weighted BCE loss uygulanabilir.

## Before/After Deep Learning Results

BATADAL veri setinde anomaly oranı düşük olduğu için deep learning modelleri yüksek accuracy üretmesine rağmen anomaly sınıfını kaçırmıştır. Bu nedenle validation tabanlı threshold tuning ve weighted BCE loss uygulanmıştır. Bu iyileştirme ile model seçiminde accuracy yerine recall ve F1-score daha öncelikli değerlendirilmiştir.

İlgili çıktılar:

- [results/improvements/deep_learning_before_after.csv](results/improvements/deep_learning_before_after.csv)
- [results/improvements/deep_learning_before_after.png](results/improvements/deep_learning_before_after.png)

## Statistical Tests

Istatistiksel degerlendirme:

- Wilcoxon
- McNemar

Ilgili ozetler `results/tables/` altinda saklanir.

## Visualizations

Ana gorseller:

- [results/figures/confusion_matrix_best_model.png](results/figures/confusion_matrix_best_model.png)
- [results/figures/roc_curve_best_model.png](results/figures/roc_curve_best_model.png)
- [results/figures/precision_recall_curve_best_model.png](results/figures/precision_recall_curve_best_model.png)
- [results/figures/automata_state_diagram_skab.png](results/figures/automata_state_diagram_skab.png)
- [results/figures/transition_probability_heatmap_skab.png](results/figures/transition_probability_heatmap_skab.png)
- [results/figures/parameter_sensitivity_skab.png](results/figures/parameter_sensitivity_skab.png)

## Run Order

Tum pipeline:

```bash
python src/experiments/run_all.py --no-resume
```

Asama bazli:

```bash
python src/main.py
python src/experiments/run_automata.py
python src/experiments/run_deep_models.py
python src/experiments/run_noise_experiment.py
python src/experiments/run_unseen_experiment.py
python src/experiments/run_cross_dataset_experiment.py
python src/experiments/run_parameter_analysis.py
python src/experiments/run_explainability_export.py
python src/experiments/run_runtime_analysis.py
python src/experiments/generate_figures.py
```

`run_all.py` su sirayla calisir:

1. preprocessing
2. model training and original test
3. noise test
4. unseen test
5. cross-dataset test
6. automata parameter analysis
7. explainability export
8. runtime analysis
9. visualization generation

Son satir:

`All experiments completed successfully. Results saved under results/.`

## Validation

```bash
pytest -q
```

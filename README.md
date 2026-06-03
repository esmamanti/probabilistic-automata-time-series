# probabilistic-automata-time-series

Bu repo, `yazlab2_v3.pdf` isterlerine uygun olarak zaman serisi anomali tespiti icin iki model ailesini karsilastirir:

- Olasiliksal otomata tabanli sembolik model
- Derin ogrenme modelleri: `LSTM`, `GRU`, `CNN`

## Kapsam

- Veri kumeleri: `SKAB` ve `BATADAL`
- Senaryolar: `original`, `noise`, `unseen`
- Parametre analizi: `window_size = 3..6`, `alphabet_size = 3..6`
- Istatistiksel analiz: Wilcoxon ve McNemar
- Olasiliksal aciklanabilirlik: state, pattern, transition probability, path probability, confidence

## Veri Kumesi Kullanimi

| Dataset | Hedef kolon | Bu repodaki kullanim |
| --- | --- | --- |
| SKAB | `anomaly` | Sadece `valve1` ve `valve2` birlestirilir; `source_group` ve `source_file` takip icin eklenir |
| BATADAL | `ATT_FLAG` | Sadece `BATADAL_dataset04.csv` kullanilir; etiketler `-999 -> 0`, `1 -> 1` olarak map edilir |

Notlar:

- BATADAL hedef etiketi acik olarak `ATT_FLAG` kolonudur.
- SKAB model girdisine `datetime`, `changepoint`, `source_group`, `source_file` alinmaz.
- BATADAL zaman sirasi korunarak `%60 / %20 / %20` train/validation/test ayrimi ile calisir.

## Mimari

- Merkezi config: [configs/config.yaml](configs/config.yaml)
- Model config: [configs/models.yaml](configs/models.yaml)
- Deney config: [configs/experiments.yaml](configs/experiments.yaml)
- Tum pipeline orkestrasyonu: [src/experiments/run_all.py](src/experiments/run_all.py)

Temel noktalar:

- preprocessing sadece train verisi uzerinde fit edilir
- PCA, automata icin tek boyuta (`PC1`) indirger
- SKAB grup bazli fold degerlendirmesi kullanir
- BATADAL zaman sirali degerlendirme kullanir
- unseen pattern'ler Levenshtein ile en yakin bilinen pattern'e map edilir
- deney baglami CSV ve JSON artifact'lerine yazilir

## Uretilen Artifact'ler

Ana ciktilar:

- [results/tables](results/tables)
- [results/explanations](results/explanations)
- [results/figures](results/figures)
- [results/logs](results/logs)

Isterlerle hizali kritik dosyalar:

- Deep metrics: [results/tables/deep_learning_metrics.csv](results/tables/deep_learning_metrics.csv)
- Deep summary: [results/tables/deep_learning_metrics_summary.csv](results/tables/deep_learning_metrics_summary.csv)
- Model comparison: [results/tables/model_comparison_metrics_summary.csv](results/tables/model_comparison_metrics_summary.csv)
- Noise metrics: [results/tables/noise_experiment_metrics.csv](results/tables/noise_experiment_metrics.csv)
- Unseen metrics: [results/tables/unseen_metrics.csv](results/tables/unseen_metrics.csv)
- Parameter analysis: [results/tables/parameter_analysis_metrics.csv](results/tables/parameter_analysis_metrics.csv)
- Deep predictions: [results/explanations/deep_learning_predictions.csv](results/explanations/deep_learning_predictions.csv)
- Automata explanations: [results/explanations/automata_skab_explanations.csv](results/explanations/automata_skab_explanations.csv)
- Explanation JSON example: [results/explanations/automata_explanation_example.json](results/explanations/automata_explanation_example.json)
- Pipeline progress: [results/logs/pipeline_progress.json](results/logs/pipeline_progress.json)

## Son Durum Ozeti

- `results/tables/deep_learning_metrics.csv` artik `LSTM`, `GRU` ve `CNN` sonuclarini icerir.
- `results/explanations/deep_learning_predictions.csv` icinde SKAB split'leri `fold_0` ... `fold_4` olarak kaydedilir.
- BATADAL split'i `test` olarak tutulur.
- Aciklanabilirlik ciktilari tablo + JSON ornegi formatinda mevcuttur.

## Beklenen Gorseller

- [results/figures/confusion_matrix_best_model.png](results/figures/confusion_matrix_best_model.png)
- [results/figures/roc_curve_best_model.png](results/figures/roc_curve_best_model.png)
- [results/figures/precision_recall_curve_best_model.png](results/figures/precision_recall_curve_best_model.png)
- [results/figures/automata_state_diagram_skab.png](results/figures/automata_state_diagram_skab.png)
- [results/figures/transition_probability_heatmap_skab.png](results/figures/transition_probability_heatmap_skab.png)
- [results/figures/parameter_sensitivity_skab.png](results/figures/parameter_sensitivity_skab.png)

## Aciklanabilirlik Semasi

Automata explanation kayitlari su alanlari raporlar:

- `state`
- `previous_state`
- `pattern`
- `status`
- `mapped_to`
- `distance`
- `transition_probability`
- `path_probability`
- `average_log_probability`
- `confidence_score`
- `decision_reason`
- `decision`

Ornek JSON:

- [results/explanations/automata_explanation_example.json](results/explanations/automata_explanation_example.json)

## Calistirma

Tum pipeline:

```bash
python src/experiments/run_all.py
```

Asama bazli komutlar:

```bash
python src/main.py
python src/experiments/run_automata.py
python src/experiments/run_deep_models.py
python src/experiments/run_noise_experiment.py
python src/experiments/run_unseen_experiment.py
python src/experiments/run_parameter_analysis.py
python src/experiments/generate_figures.py
```

`run_all.py` asama bazli progress kaydi tutar ve mevcut artifact'ler varsa ilgili asamayi atlayabilir. Bu bilgi [results/logs/pipeline_progress.json](results/logs/pipeline_progress.json) dosyasina yazilir.

## Dogrulama

```bash
pytest -q
```

Acceptance test kapsaminda sunlar dogrulanir:

- Deep artifact'lerde `CNN`, `GRU`, `LSTM` varligi
- SKAB fold split kayitlari
- Noise ve unseen senaryolari
- Parametre grid tamligi
- Levenshtein unseen mapping davranisi
- Explanation ic tutarliligi
- Gorsel ve output artifact uretimi

# Probabilistic Automata ile Zaman Serisi Anomali Tespiti

## Veri Setleri

Bu raporda kullanılan veri setleri:

- `SKAB`
- `BATADAL`

Not: EK şablonunda geçen farklı veri seti adları bu proje için geçerli değildir. Bu çalışma yalnızca `SKAB` ve `BATADAL` üzerinde yürütülmüştür.
EK PDF şablonunda bulunan `WADI` sütunu bu proje için kullanılmamaktadır; raporda kaldırılmalı veya `N/A` olarak işaretlenmelidir.

## Adım 1 — Temel Model Eğitimi (Tablo 1)

Bu adımda `LSTM`, `GRU`, `1D-CNN` ve `Probabilistic Automata` modelleri `SKAB` ve `BATADAL` veri setleri üzerinde karşılaştırılmıştır. Tablo 1 değerleri mevcut deney çıktılarından (`results/tables/model_comparison_metrics_summary.csv`) derlenmiştir; bu nedenle bu aşamada tam eğitimi yeniden çalıştırmak zorunlu değildir.

### Tablo 1 — Model Performansı

| Model | SKAB (F1 ± std) | BATADAL (F1 ± std) |
|-------|-----------------|--------------------|
| LSTM | 0.4942 ± 0.0539 | 0.1297 ± 0.0412 |
| GRU | 0.4979 ± 0.0468 | 0.1766 ± 0.0509 |
| 1D-CNN | 0.4965 ± 0.0774 | 0.0737 ± 0.0466 |
| Automata | 0.5022 ± 0.0934 | 0.3053 ± 0.0000 |

### Değerlendirme Notu

- `SKAB` sonuçları 5-fold grup tabanlı değerlendirme ortalamasıdır.
- `BATADAL` sonuçları 5 seed ortalamasıdır: `[42, 123, 2026, 7, 999]`.
- Bu adım için veri seti isimleri düzeltilmiş ve tablo yalnızca `SKAB` ile `BATADAL` için hazırlanmıştır.

## Adım 2 — Gürültü ve Unseen Analizi (Tablo 2)

Bu adımda `run_noise_experiment.py` ve `run_unseen_experiment.py` çıktıları kullanılarak modellerin gürültü altındaki davranışı ve unseen pattern yönetimi değerlendirilmiştir. Tablo 2, mevcut deney artefact’larından derlenmiştir.

### Tablo 2 — Gürültü ve Unseen Analizi

| Model | SKAB Orig. F1 | SKAB Gürültülü F1 | BATADAL Orig. F1 | BATADAL Gürültülü F1 | Det. Rate | Map. Acc. |
|-------|---------------|-------------------|------------------|----------------------|-----------|-----------|
| LSTM | 0.2632 | 0.2642 | 0.0000 | 0.0000 | N/A | N/A |
| GRU | 0.2717 | 0.2709 | 0.0334 | 0.0329 | N/A | N/A |
| CNN | 0.2789 | 0.2790 | 0.0000 | 0.0000 | N/A | N/A |
| Automata | 0.0816 | 0.0852 | 0.2821 | 0.2677 | SKAB: 100.0% / BATADAL: 0.0% | SKAB: 0.3352 / BATADAL: 0.5556 |

### Açıklama Notu

- `Det. Rate` ve `Map. Acc.` yalnızca `Automata` modeli için anlamlıdır; çünkü unseen pattern eşleme mekanizması bu modelde açık biçimde tanımlıdır.
- Gürültülü F1 sütunu Gaussian gürültü `std=0.10` senaryosunu göstermektedir.
- Tablo 1 ile Tablo 2 değerlerinin farklı görünmesi beklenen bir durumdur.
- Tablo 1, temel model karşılaştırmasında kullanılan ana değerlendirme protokolünün ortalama sonuçlarını verir.
- Tablo 2 ise ayrı senaryo deneylerinden (`original` ve `noise`, ayrıca unseen örnek altkümeleri) türetilmiştir.
- Özellikle `SKAB` tarafında fold bazlı, `BATADAL` tarafında seed ve senaryo bazlı özetleme farkı bulunduğu için F1 değerleri birebir aynı çıkmaz.

## Adım 3 — Cross-Dataset (Tablo 3)

Bu adımda `run_cross_dataset_experiment.py` çıktıları kullanılarak veri setleri arası aktarım başarımı değerlendirilmiştir. Amaç, bir veri setinde eğitilen modelin diğer veri setine ne ölçüde genellenebildiğini göstermektir.

### Tablo 3 — Cross-Dataset Matrisi

| Train / Test | SKAB | BATADAL |
|-------------|------|---------|
| Train: SKAB | — | 0.3125 ± 0.0299 (1D-CNN) |
| Train: BATADAL | 0.5372 ± 0.0127 (Automata) | — |

### Değerlendirme Notu

- `SKAB -> BATADAL` yönünde en iyi sonuç `1D-CNN` modeli ile elde edilmiştir.
- `BATADAL -> SKAB` yönünde en iyi sonuç `Automata` modeli ile elde edilmiştir.
- Bu deneylerde tüm özellikler boyut uyumu sağlamak için `PCA` ile `PC1` bileşenine indirgenmiştir.

## Adım 4 — Parametre Analizi (Tablo 4)

Bu adımda `window_size ∈ {3,4,5,6}` ve `alphabet_size ∈ {3,4,5,6}` kombinasyonları denenerek `Automata` modelinin parametre duyarlılığı incelenmiştir. Tablo 4 değerleri mevcut `parameter_analysis` artefact’larından alınmıştır.

### Tablo 4a — Window Size Etkisi (`alphabet_size=3` sabit)

| Window Size | SKAB F1 | BATADAL F1 | SKAB State # | BATADAL State # | SKAB Transition Density | BATADAL Transition Density |
|-------------|---------|------------|--------------|-----------------|-------------------------|----------------------------|
| 3 | 0.0859 | 0.2500 | 20.68 | 26.00 | 0.0970 | 0.1050 |
| 4 | 0.0816 | 0.2821 | 37.92 | 76.00 | 0.0494 | 0.0261 |
| 5 | 0.0844 | 0.0548 | 67.92 | 166.00 | 0.0260 | 0.0092 |
| 6 | 0.0909 | 0.3889 | 116.12 | 206.00 | 0.0142 | 0.0063 |

### Tablo 4b — Alphabet Size Etkisi (`window_size=4` sabit)

| Alphabet Size | SKAB F1 | BATADAL F1 | SKAB State # | BATADAL State # | SKAB Transition Density | BATADAL Transition Density |
|---------------|---------|------------|--------------|-----------------|-------------------------|----------------------------|
| 3 | 0.0816 | 0.2821 | 37.92 | 76.00 | 0.0494 | 0.0261 |
| 4 | 0.0969 | 0.3065 | 62.96 | 175.00 | 0.0305 | 0.0095 |
| 5 | 0.1144 | 0.2353 | 97.40 | 258.00 | 0.0196 | 0.0059 |
| 6 | 0.1124 | 0.2584 | 137.40 | 340.00 | 0.0142 | 0.0040 |

### Geçiş Yoğunluğu Notu

- `state_transition_analysis.csv` dosyası mevcut ve geçiş yoğunluğu bilgisini zaten içermektedir.
- İlgili kolon adı `transition_density` olarak kaydedilmiştir.
- Bu nedenle bu adım için script’e ek bir kayıt satırı eklemek gerekmedi.

## Adım 5 — Runtime (Tablo 5)

Bu adımda `run_runtime_analysis.py` çıktıları kullanılarak eğitim ve çıkarım süreleri karşılaştırılmıştır. Tablo 5 değerleri mevcut runtime artefact’larından alınmıştır.

### Tablo 5 — Eğitim ve Çıkarım Süreleri

| Model | SKAB Eğitim (sn) | SKAB Çıkarım (sn) | BATADAL Eğitim (sn) | BATADAL Çıkarım (sn) |
|-------|------------------|-------------------|---------------------|----------------------|
| LSTM | 9.68 | 0.1026 | 2.79 | 0.0196 |
| GRU | 8.69 | 0.0984 | 2.97 | 0.0193 |
| CNN | 11.49 | 0.0807 | 6.22 | 0.0136 |
| Automata | 26.06 | 0.0085 | 0.53 | 0.0046 |

### Değerlendirme Notu

- Bu adım için tablo mevcut `results/runtime/runtime_comparison.csv` çıktısından derlenmiştir.
- `BATADAL` üzerinde `Automata`, eğitim süresinde derin öğrenme modellerinden belirgin biçimde daha hızlıdır.
- `SKAB` üzerinde ise `Automata` eğitimi daha uzun sürse de çıkarım süresi en düşük seviyededir.

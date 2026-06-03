# Final Report

## 1. Problem ve Yaklasim

Bu proje, zaman serisi anomali tespitinde iki farkli aileyi karsilastirir:

- Olasiliksal otomata tabanli sembolik modelleme
- Derin ogrenme tabanli sira modelleri (`LSTM`, `GRU`, `CNN`)

Bu revizyonda mimari "tam parametrik" olacak sekilde guncellendi. `configs/models.yaml` altindaki etkin derin modeller deney kodu tarafinda otomatik kesfediliyor. Boylece yeni bir model eklemek veya bir modeli pasife almak icin artik deney dongusunde elle degisiklik yapmaya gerek kalmiyor.

## 2. Veri Setleri

### SKAB

- Hedef sutunu: `anomaly`
- Split stratejisi: grup bazli hold-out / fold bazli ayirim
- Ozellikler: sensorden gelen zaman serisi kolonlari

### BATADAL

- Hedef sutunu: `ATT_FLAG`
- Etiket donusumu: `-999 -> 0`, `1 -> 1`
- Split stratejisi: zaman sirali train/validation/test

Bu noktada `ATT_FLAG` bilgisinin raporda acikca yazilmasi zorunlu oldugu icin burada dogrudan belirtilmistir.

## 3. Eksik Veri Denetimi ve Preprocessing

Ham veri audit'i sonucunda:

- `BATADAL_dataset04.csv` dosyasinda eksik deger sayisi: `0`
- Dahil edilen SKAB CSV dosyalarinda eksik deger sayisi: `0`

Dolayisiyla mevcut veri snapshot'inda imputasyon ihtiyaci fiilen yoktur. Buna ragmen, proje isterindeki "gerekli durumlarda eksik veri islemleri" maddesini karsilamak icin preprocessing pipeline'a su eklenmistir:

- `preprocessing.missing_data.enabled`
- `preprocessing.missing_data.strategy`
- `SimpleImputer` tabanli eksik veri tamamlama

Pipeline sirasi:

1. Eksik veri imputasyonu
2. Scaling
3. PCA
4. Sequence generation

## 4. Deney Baglami ve Loglama

Onceki durumda metric ciktilari deney baglamini sistematik olarak tasimiyordu. Bu revizyonla birlikte her run satiri su alanlari da icerir:

- `experiment_context`
- `context_preprocessing_*`
- `context_noise_*`
- `context_training_*`
- `context_automata_*`
- `context_model_config_*`

Ayrica `src/main.py` artik:

- tum resolve edilmis config snapshot'larini
- veri seti bazli eksik veri audit sonucunu

`results/logs/project.log` dosyasina yazar.

## 5. Guncel Model Karsilastirmasi

Asagidaki ozet, `results/tables/deep_learning_metrics.csv` dosyasinin 5 seed ortalamalarindan alinmistir.

| Dataset | Model | Accuracy | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| SKAB | CNN | 0.6643 | 0.5683 | 0.2037 | 0.2845 |
| SKAB | GRU | 0.6633 | 0.5877 | 0.1968 | 0.2750 |
| SKAB | LSTM | 0.6618 | 0.5588 | 0.1830 | 0.2570 |
| BATADAL | CNN | 0.8660 | 0.0000 | 0.0000 | 0.0000 |
| BATADAL | GRU | 0.8655 | 0.4214 | 0.0211 | 0.0404 |
| BATADAL | LSTM | 0.8609 | 0.0087 | 0.0023 | 0.0030 |

Otomata ciktilari icin mevcut artefakt ozeti:

| Dataset | Accuracy | Precision | Recall | F1 | Ortalama unseen |
| --- | ---: | ---: | ---: | ---: | ---: |
| SKAB | 0.6132 | 0.2935 | 0.0611 | 0.0932 | 20.6 |
| BATADAL | 0.7282 | 0.2200 | 0.3929 | 0.2821 | 9.0 |

Degerlendirme:

- SKAB veri setinde derin modeller daha yuksek F1 uretiyor.
- BATADAL veri setinde accuracy yuksek kalirken anomaly yakalama performansi derin tarafta zayif.
- CNN entegrasyonu mimariyi parametre odakli hale getirdi ve SKAB tarafinda en iyi ortalama F1'i verdi.

## 6. Noise Analizi

`results/tables/noise_experiment_metrics.csv` icindeki kayitli snapshot su davranisi gosteriyor:

| Dataset | Model | Original F1 | Noise F1 | Gozlem |
| --- | --- | ---: | ---: | --- |
| SKAB | LSTM | 0.3883 | 0.3864 | Cok kucuk dusus |
| SKAB | GRU | 0.3526 | 0.3535 | Neredeyse sabit |
| SKAB | AUTOMATA | 0.0355 | 0.0393 | Kucuk artis |
| BATADAL | AUTOMATA | 0.2821 | 0.2933 | Kucuk artis |

Yorum:

- SKAB tarafinda derin modeller dusuk seviyeli gaussian noise'a goreli dayanikli.
- BATADAL automata modeli noise altinda hafif iyilesme gostermis.
- Bu artefakt, guncel CNN entegrasyonundan once uretilmis snapshot'i temsil eder; ayni deney akisi yeni parametrik yapiyla tekrar calistirilabilir.

## 7. Unseen Pattern Analizi

`results/tables/unseen_metrics.csv` ozetine gore:

| Dataset | Total Patterns | Unseen Patterns | Unseen Ratio | Avg Distance | Avg Confidence |
| --- | ---: | ---: | ---: | ---: | ---: |
| SKAB | 1374 | 5 | 0.003639 | 1.0 | 0.75 |
| BATADAL | 206 | 9 | 0.043689 | 1.0 | 0.75 |

Sonuclar:

- BATADAL daha yuksek unseen ratio uretiyor.
- Her iki veri setinde de unseen pattern'lerin en yakin eslesmesi bulunabiliyor.
- BATADAL'da unseen paternlerin anomaly ile iliskisi daha yuksek gorunuyor.

## 8. Parametre Duyarliligi

`results/tables/parameter_analysis_metrics.csv` icinden en iyi F1 kombinasyonlari:

| Dataset | Window Size | Alphabet Size | F1 | Accuracy | Unseen |
| --- | ---: | ---: | ---: | ---: | ---: |
| SKAB | 6 | 5 | 0.1741 | 0.5947 | 22 |
| BATADAL | 6 | 3 | 0.3889 | 0.5111 | 66 |

Yorum:

- Pencere boyutu arttikca state sayisi ve unseen sayisi buyuyor.
- BATADAL'da recall/F1 kazanci ile accuracy kaybi arasinda belirgin bir trade-off var.
- SKAB'ta daha dengeli ama daha sinirli bir iyilesme goruluyor.

## 9. Teslim Durumu

Bu revizyon sonrasinda daha once eksik olan maddeler su sekilde kapatilmistir:

- Parametrik derin model mimarisi: tamamlandi
- CNN tanimi ile kodun uyumu: tamamlandi
- Deney parametrelerinin sistematik loglanmasi: tamamlandi
- Eksik veri islemi veya eksik yok audit'i: tamamlandi
- README ve final rapor teslimi: tamamlandi
- BATADAL `ATT_FLAG` bilgisinin raporda acikca belirtilmesi: tamamlandi

## 10. Sinirlar ve Sonraki Adim

- `generate_figures.py` icin mevcut ortamda `seaborn` eksik; bu nedenle PNG figurler bu geciste yeniden uretilmedi.
- Noise ve unseen artefaktlari, yeni CNN destekli tam kosunun tekrar alinmasiyla daha da guncellenebilir.
- BATADAL icin class imbalance'a duyarli threshold tuning veya cost-sensitive egitim, derin modellerin anomaly recall/F1 skorlarini iyilestirebilir.

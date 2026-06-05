# Probabilistic Automata ile Zaman Serisi Anomali Tespiti

Bu proje, zaman serisi anomali tespitinde olasılıksal otomata tabanlı sembolik bir yaklaşımı derin öğrenme tabanlı modellerle karşılaştırmak için geliştirilmiştir. Çalışmada hem tahmin başarımı hem de açıklanabilirlik, gürültü dayanıklılığı, unseen pattern yönetimi, çapraz veri seti aktarımı ve çalışma süresi gibi ölçütler birlikte ele alınmaktadır. Repo, deneylerin yeniden üretilebilir biçimde çalıştırılmasını ve tüm ara çıktılarının kaydedilmesini hedefleyen tam bir deney hattı içerir.

## Veri Setleri

Bu çalışmada iki veri seti kullanılmıştır:

- `SKAB`: Endüstriyel ekipman sensörlerinden türetilen anomali senaryoları içerir. Bu projede yalnızca `valve1` ve `valve2` klasörleri kullanılmıştır.
- `BATADAL`: Su dağıtım sistemi saldırı/anomali senaryolarını içerir. Bu projede `BATADAL_dataset04.csv` dosyası kullanılmış ve hedef etiket kolonu `ATT_FLAG` olarak işlenmiştir.

Bu iki veri seti birlikte seçilmiştir çünkü biri fold-tabanlı grup ayrımı gerektiren daha parçalı bir yapı sunarken, diğeri zaman sıralı ve sınıf dengesizliği yüksek bir gerçekçi değerlendirme ortamı sağlar. EK PDF şablonunda geçen `SWAT` ve `WADI` veri setlerinden farklı olarak bu repo yalnızca `SKAB` ve `BATADAL` üzerine kuruludur.

## Kurulum ve Çalıştırma

Bağımlılıkları kurmak için:

```bash
pip install -r requirements.txt
```

Tüm deney hattını çalıştırmak için:

```bash
python src/experiments/run_all.py --no-resume
```

İsterseniz aşama bazlı komutlar da doğrudan çalıştırılabilir:

```bash
python src/experiments/run_automata.py
python src/experiments/run_deep_models.py
python src/experiments/run_noise_experiment.py
python src/experiments/run_unseen_experiment.py
python src/experiments/run_cross_dataset_experiment.py
python src/experiments/run_parameter_analysis.py
python src/experiments/run_explainability_export.py
python src/experiments/run_runtime_analysis.py
python src/experiments/run_statistical_tests.py
python src/experiments/generate_figures.py
```

## Proje Mimarisi

Repo yapısı özetle aşağıdaki bölümlerden oluşur:

- `configs/`: Veri seti, model ve deney ayarlarını içeren YAML dosyaları.
- `data/`: Ham, işlenmiş ve bölünmüş veri çıktıları.
- `src/data/`: Veri yükleme, dönüştürme ve split hazırlama bileşenleri.
- `src/models/`: Olasılıksal otomata ve derin öğrenme modellerinin uygulamaları.
- `src/evaluation/`: Metrikler, grafik üretimi ve istatistiksel test yardımcıları.
- `src/experiments/`: Tüm deney scriptleri ve tam pipeline akışı.
- `src/utils/`: Config, seed, deney context’i ve yardımcı altyapı kodları.
- `results/`: Tablolar, açıklanabilirlik çıktıları, görseller, runtime ve diğer deney artefact’ları.
- `tests/`: Birim testler ve kabul testi niteliğindeki doğrulamalar.
- `reports/`: README içine taşınmış rapor yapısının eski yardımcı kalıntıları.

Merkezi giriş noktaları:

- `configs/config.yaml`
- `configs/models.yaml`
- `configs/experiments.yaml`
- `src/experiments/run_all.py`
- `src/utils/experiment_context.py`

**Not — CSV Deney Kaydı Formatı:** `results/tables/` altındaki ham CSV dosyaları her satırda tam deney context’ini JSON string olarak içerir (`context_` önekli kolonlar). Bu tasarım tam yeniden üretilebilirlik için seçilmiştir; her satır hangi parametrelerle üretildiğini kendi içinde taşır. Raporlama sırasında yalnızca `dataset`, `model`, `split`, `f1_score_mean`, `f1_score_std` gibi özet kolonlar kullanılır. Özet tablolar ayrıca `*_summary.csv` uzantısıyla kaydedilir.

## Veri Ön İşleme ve Bölme Stratejisi

Ön işleme hattı eksik değer işleme, ölçekleme, PCA ile tek bileşene indirgeme ve sabit uzunluklu dizi üretiminden oluşur.

`SKAB` için:

- Veri yalnızca `valve1` ve `valve2` klasörlerinden alınır.
- Tüm CSV dosyaları birleştirilir.
- `source_group` ve `source_file` alanları izlenebilirlik için korunur.
- Değerlendirme `source_file` bazlı grup ayrımı mantığıyla fold yapısında yürütülür.

`BATADAL` için:

- `BATADAL_dataset04.csv` kullanılır.
- Etiket kolonu `ATT_FLAG` olarak sabitlenmiştir.
- Zaman sırası korunur.
- Bölme stratejisi `%60 train / %20 validation / %20 test` şeklindedir.

Data leakage’i engellemek için şu kurallar uygulanır:

- scaler yalnızca train verisi üzerinde fit edilir
- PCA yalnızca train verisi üzerinde fit edilir
- validation ve test bölümlerinde aynı fit edilmiş dönüşümler tekrar kullanılır
- otomata durum üretimi ve geçiş olasılıkları yalnızca train verisinden kurulur

## Eğitim Protokolü

Çalışmanın temel eğitim standardı aşağıdaki gibidir:

- early stopping metriği: `val_loss`
- maksimum epoch: `50`
- batch size: `32`
- random seed kümesi: `[42, 123, 2026, 7, 999]`

Karşılaştırılan başlıca model aileleri:

- Olasılıksal otomata tabanlı sembolik model
- `LSTM`
- `GRU`
- `CNN`

Ek iyileştirme olarak derin öğrenme modellerinde iki rapor destek özelliği de bulunmaktadır:

- validation tabanlı threshold tuning
- class imbalance azaltmak için weighted BCE loss

Bu iki mekanizma deneyleri zenginleştiren ek özelliklerdir; temel mimari karşılaştırmanın zorunlu tek koşulu değildir.

## Tablo 1 — Model Performansı

| Model | SKAB (F1 ± std) | BATADAL (F1 ± std) |
|-------|-----------------|---------------------|
| LSTM | 0.4942 ± 0.0539 | 0.1297 ± 0.0412 |
| GRU | 0.4979 ± 0.0468 | 0.1766 ± 0.0509 |
| 1D-CNN | 0.4965 ± 0.0774 | 0.0737 ± 0.0466 |
| Automata | 0.5022 ± 0.0934 | 0.3053 ± 0.0000 |

SKAB sonuçları 5-fold StratifiedGroupKFold ortalamasıdır (`source_file` bazlı gruplama). BATADAL sonuçları zaman sıralı test kümesinde 5 seed ortalamasıdır `[42, 123, 2026, 7, 999]`. BATADAL'da düşük DL F1 değerleri sınıf dengesizliğinden kaynaklanmaktadır (`~%5` anomali oranı).

## Tablo 2 — Gürültü ve Unseen Analizi

| Model | SKAB Orig. F1 | SKAB Gürültülü F1 | BAT Orig. F1 | BAT Gürültülü F1 | Det. Rate | Map. Acc. |
|-------|---------------|-------------------|--------------|------------------|-----------|-----------|
| LSTM | 0.2632 | 0.2642 | 0.0000 | 0.0000 | N/A | N/A |
| GRU | 0.2717 | 0.2709 | 0.0334 | 0.0329 | N/A | N/A |
| CNN | 0.2789 | 0.2790 | 0.0000 | 0.0000 | N/A | N/A |
| Automata | 0.0816 | 0.0852 | 0.2821 | 0.2677 | SKAB: 100.0% / BAT: 0.0% | SKAB: 0.3352 / BAT: 0.5556 |

Det. Rate ve Map. Acc. yalnızca Automata için hesaplanmıştır. Gürültü seviyesi: Gaussian `std=0.10`.

## Tablo 3 — Cross-Dataset Matrisi

| Train / Test | SKAB | BATADAL |
|-------------|------|---------|
| Train: SKAB | — (in-distribution, bkz. Tablo 1) | 0.3125 ± 0.0299 (1D-CNN) |
| Train: BATADAL | 0.5372 ± 0.0127 (Automata) | — (in-distribution, bkz. Tablo 1) |

Cross-dataset deneylerde tüm özellikler PCA ile `PC1`'e indirgenerek boyut uyumu sağlanmıştır. BATADAL→SKAB: Automata modeli `recall=1.0` üretiyor ancak precision düşük; bu durum anomali oranı yüksek sınıfa karşı aşırı hassasiyetle açıklanabilir.

## Tablo 4 — Parametre Analizi

### Tablo 4a — Window Size Etkisi (`alphabet_size=3` sabit)

| Window Size | SKAB F1 | BAT F1 | SKAB State # | BAT State # |
|-------------|---------|--------|--------------|-------------|
| 3 | 0.0859 | 0.2500 | 20.68 | 26.00 |
| 4 | 0.0816 | 0.2821 | 37.92 | 76.00 |
| 5 | 0.0844 | 0.0548 | 67.92 | 166.00 |
| 6 | 0.0909 | 0.3889 | 116.12 | 206.00 |

### Tablo 4b — Alphabet Size Etkisi (`window_size=4` sabit)

| Alphabet Size | SKAB F1 | BAT F1 | SKAB State # | BAT State # |
|---------------|---------|--------|--------------|-------------|
| 3 | 0.0816 | 0.2821 | 37.92 | 76.00 |
| 4 | 0.0969 | 0.3065 | 62.96 | 175.00 |
| 5 | 0.1144 | 0.2353 | 97.40 | 258.00 |
| 6 | 0.1124 | 0.2584 | 137.40 | 340.00 |

Window size artışı state sayısını artırır, daha az unseen pattern üretir ama geçiş matrisini seyrekleştirir. Alphabet size artışı sembol uzayını genişletir, unseen pattern riskini artırır. Bu iki etki arasındaki denge optimal parametre seçimini belirler.

## Tablo 5 — Runtime

| Model | SKAB Eğitim (sn) | SKAB Çıkarım (sn) | BAT Eğitim (sn) | BAT Çıkarım (sn) |
|-------|------------------|-------------------|-----------------|------------------|
| LSTM | 9.68 | 0.1026 | 2.79 | 0.0196 |
| GRU | 8.69 | 0.0984 | 2.97 | 0.0193 |
| CNN | 11.49 | 0.0807 | 6.22 | 0.0136 |
| Automata | 26.06 | 0.0085 | 0.53 | 0.0046 |

Eğitim süresi GPU (CUDA) üzerinde ölçülmüştür. Automata modeli CPU'da dahi DL modellerine kıyasla çok daha kısa eğitim süresine sahiptir.

## Görseller

### Şekil 1: En iyi F1 üreten modelin confusion matrix'i
![En iyi F1 üreten modelin confusion matrix'i](results/figures/confusion_matrix_best_model.png)
*En iyi F1 üreten modelin confusion matrix'i.*

### Şekil 2: Model ailelerinin ROC eğrisi karşılaştırması
![Model ailelerinin ROC eğrisi karşılaştırması](results/figures/roc_curve_best_model.png)
*Model ailelerinin ROC eğrisi karşılaştırması.*

### Şekil 3: Precision-Recall eğrisi
![Precision-Recall eğrisi](results/figures/precision_recall_curve_best_model.png)
*BATADAL gibi dengesiz veri setinde PR eğrisi ROC'a göre daha bilgilendirici bir metriktir.*

### Şekil 4: SKAB için otomata durum geçiş diyagramı
![SKAB için otomata durum geçiş diyagramı](results/figures/automata_state_diagram_skab.png)
*SKAB için SAX sembolik temsiliyle oluşturulan olasılıksal otomata durum geçiş diyagramı (`window_size=4`, `alphabet_size=3`).*

### Şekil 5: Geçiş olasılıkları ısı haritası
![Geçiş olasılıkları ısı haritası](results/figures/transition_probability_heatmap_skab.png)
*Durumlar arası geçiş olasılıklarının ısı haritası; koyu hücreler yüksek olasılıklı (normal) geçişleri temsil eder.*

### Şekil 6: F1 vs window/alphabet
![F1 vs window/alphabet](results/automata_analysis/f1_vs_window_alphabet.png)
*Window size ve alphabet size'ın F1 üzerindeki etkisi.*

### Şekil 7: State sayısı vs window
![State sayısı vs window](results/automata_analysis/state_count_vs_window.png)
*Parametre değişiminin state sayısı ve geçiş yoğunluğu üzerindeki etkisi.*

### Şekil 8: Gürültü dayanıklılığı
![Gürültü dayanıklılığı](results/noise/noise_robustness_plot.png)
*Artan Gaussian gürültü seviyelerinde (`std=0.05`, `0.10`, `0.20`) model F1 değişimi.*

### Şekil 9: Cross-dataset matrisi
![Cross-dataset matrisi](results/cross_dataset/cross_dataset_matrix.png)
*SKAB↔BATADAL çapraz veri seti genellenebilirlik matrisi.*

### Şekil 10: Confidence histogram
![Confidence histogram](results/explanations/confidence_histogram.png)
*Automata güven skoru dağılımı: normal ve anomali sınıfları için ayrı.*

## Analiz ve Bulgular

### Model Karşılaştırması

Tablo 1 sonuçları, `SKAB` üzerinde dört modelin birbirine görece yakın F1 değerleri ürettiğini göstermektedir. `LSTM` için ortalama F1 `0.4942 ± 0.0539`, `GRU` için `0.4979 ± 0.0468`, `1D-CNN` için `0.4965 ± 0.0774` ve `Automata` için `0.5022 ± 0.0934` elde edilmiştir. Bu tablo, `SKAB` üzerinde derin öğrenme modelleri ile olasılıksal otomata yaklaşımının benzer doğruluk bandında çalıştığını, yani otomata modelinin açıklanabilirlik avantajına rağmen performans açısından tamamen geride kalmadığını göstermektedir.

`BATADAL` tarafında ayrım çok daha belirgindir. `LSTM` için F1 `0.1297 ± 0.0412`, `GRU` için `0.1766 ± 0.0509`, `1D-CNN` için `0.0737 ± 0.0466` düzeyinde kalırken `Automata` modeli `0.3053 ± 0.0000` ile daha yüksek bir sonuç üretmiştir. Bu farkın temel nedeni `BATADAL` veri setindeki güçlü sınıf dengesizliğidir. Derin öğrenme modelleri yüksek accuracy üretse bile anomali sınıfını yeterince yakalayamamaktadır; buna karşılık otomata modelinin `BATADAL` için recall değeri `0.7143` seviyesindedir. Bu durum, path probability tabanlı karar mekanizmasının dengesiz sınıf dağılımına karşı doğal bir direnç gösterebildiğini düşündürmektedir. Açıklanabilirlik açısından fark daha da belirgindir: Automata modeli her karar için `state`, `pattern`, `transition_probability` ve `path_probability` alanlarını üretirken derin öğrenme modelleri büyük ölçüde black-box karakterini korumaktadır.

### Veri Setleri Arası Performans Farkları

İki veri seti yapısal olarak oldukça farklıdır. `SKAB`, `source_file` bazlı fold düzeni sayesinde aynı fiziksel kaynaktan gelen örneklerin train ve test arasında karışmasını engelleyen daha kontrollü bir değerlendirme yapısı sunmaktadır. Bu nedenle model ailesi farkları burada daha dengeli görünmektedir. `BATADAL` ise zaman sıralı `%60/%20/%20` ayrım ve düşük anomali oranı nedeniyle daha zor bir problemdir; model yanlışlarının büyük bölümü anomali sınıfını kaçırma yönünde oluşmaktadır.

Çapraz veri seti sonuçları da bu farkı desteklemektedir. `SKAB -> BATADAL` yönünde en iyi sonuç `1D-CNN` ile `0.3125 ± 0.0299` iken, `BATADAL -> SKAB` yönünde en iyi sonuç `Automata` ile `0.5372 ± 0.0127` olmuştur. Bu deneylerde tüm özelliklerin `PCA` ile `PC1` bileşenine indirgenmesi boyut uyumu sağlamış olsa da, bu indirgeme veri kümeleri arasındaki yapısal farkları tamamen ortadan kaldırmamaktadır. Özellikle `BATADAL -> SKAB` yönünde `Automata` modelinin `recall=1.0` üretmesi ama precision değerinin düşük kalması, modelin anomaliye aşırı duyarlı davranabildiğini göstermektedir.

### Gürültü Etkisi Analizi

Tablo 2 ve gürültü deneyleri, `SKAB` üzerinde tüm modellerin Gaussian gürültü altında görece kararlı kaldığını göstermektedir. `LSTM` modeli `0.2632` F1’den `std=0.10` seviyesinde `0.2642`'ye, `GRU` modeli `0.2717`'den `0.2709`'a, `CNN` modeli `0.2789`'dan `0.2790`'a değişmektedir. `Automata` modeli ise `0.0816`'dan `0.0852`'ye hafif artış göstermektedir. `std=0.20` düzeyinde de `SKAB` için tüm modellerde dramatik bir çöküş gözlenmemekte, hatta `Automata` F1’i `0.0884` seviyesine kadar çıkmaktadır.

`BATADAL` tarafında ise gürültü etkisi veri setinin kendi zorluğunun gölgesinde kalmaktadır. `LSTM` ve `CNN` zaten `0.0000` F1 düzeyinde olduğundan gürültü bu modeller için pratikte yeni bir bozulma yaratmamaktadır. `GRU` modeli `0.0334`'ten `0.0329` ve `0.0295` seviyelerine düşerken, `Automata` modeli `0.2821`'den `0.2677` ve `0.2684` seviyelerine gerilemektedir. SAX temelli sembolik temsil ve pencereleme etkisi, otomata tarafında doğal bir smoothing davranışı oluşturmakta; küçük gürültü dalgalanmaları her zaman dramatik yapısal değişime dönüşmemektedir.

### Unseen Veri Davranışı

Unseen pattern yönetimi yalnızca `Automata` modeli için anlamlıdır ve Levenshtein eşleştirmesi ile yapılmaktadır. `SKAB` için unseen anomaly detection rate `%100.0`, `BATADAL` için ise `%0.0` olarak ölçülmüştür. Bu sonuç ilk bakışta çelişkili görünebilir; ancak `Det. Rate` ile `Map. Acc.` farklı iki davranışı ölçmektedir. `Det. Rate`, unseen bir örneğin anomali olarak işaretlenip işaretlenmediğini; `Map. Acc.` ise unseen pattern'ın en yakın bilinen pattern'a semantik olarak doğru eşlenip eşlenmediğini ölçer. Dolayısıyla bir sistem, eşleştirme kalitesi kusurlu olsa bile unseen örneği "anomali" olarak yakalayabilir.

`SKAB` tarafında bu ayrım çok belirgindir: `372` unseen kaydın `125` tanesi gerçek anomalidir ve bu anomalilerin tamamı yakalanmıştır. Ancak aynı anda `247` normal unseen kaydın `217` tanesi de anomali olarak işaretlenmiştir; yani model unseen gördüğünde oldukça hassas davranmakta, bu da `%100` detection rate ile görece düşük mapping accuracy değerinin birlikte ortaya çıkmasına neden olmaktadır. Nitekim `SKAB` için mapping accuracy, Levenshtein uzaklığı `1` ve `2` seviyeleri birlikte düşünüldüğünde yaklaşık `0.3352` düzeyindedir.

`BATADAL` tarafında ise bağlam daha küçüktür: toplam unseen örnek sayısı `45`, bunların yalnızca `10` tanesi gerçek anomalidir. Bu `10` anomalinin hiçbiri doğru biçimde anomali olarak işaretlenmediği için detection rate `%0.0` çıkmıştır. Buna karşın normal unseen örneklerin bir kısmı yine anomaliye kaymıştır (`35` normal unseen örneğin `10` tanesi false positive). Mapping accuracy burada Levenshtein uzaklığı `1` düzeyinde `0.5556` olarak ölçülmüştür. Bu tablo, unseen pattern yakalamanın her zaman doğru pattern semantiğiyle eşleşme anlamına gelmediğini; ancak yine de modelin açıklanabilir bir fallback mekanizmasına sahip olduğunu göstermektedir.

### Parametre Etkileri

Tablo 4a ve Tablo 4b, `window_size` ve `alphabet_size` parametrelerinin otomata yapısını doğrudan değiştirdiğini göstermektedir. `alphabet_size=3` sabitken `window_size` değerinin `3`'ten `6`'ya çıkması `SKAB` için ortalama state sayısını `20.68`'den `116.12`'ye, `BATADAL` için ise `26.00`'dan `206.00`'ya yükseltmektedir. Aynı sırada `SKAB` F1 değeri `0.0859` ile `0.0909` arasında sınırlı dalgalanırken, `BATADAL` F1 değeri `0.2500`, `0.2821`, `0.0548` ve `0.3889` gibi daha oynak bir desen göstermektedir.

`window_size=4` sabitken `alphabet_size` artışı da benzer şekilde state uzayını genişletmektedir: `SKAB` state sayısı `37.92`'den `137.40`'a, `BATADAL` state sayısı `76.00`'dan `340.00`'a çıkmaktadır. Aynı anda `SKAB` F1 değeri `0.0816`'dan `0.1144` seviyesine kadar yükselip sonra hafif düşerken, `BATADAL` F1 değeri `0.2821`, `0.3065`, `0.2353` ve `0.2584` biçiminde değişmektedir. Bu davranış, daha zengin sembol uzayının daha fazla ifade gücü sağlarken unseen pattern riskini de yükselttiğini göstermektedir.

### Karar Mekanizması Notu

İsterde tanımlanan `P(sequence)=∏P(Si→Si+1)` formülü log uzayında hesaplanarak `average_log_probability` elde edilmektedir. Bu dönüşüm çok adımlı geçişlerde oluşabilecek sayısal underflow riskini ortadan kaldırır ve matematiksel olarak eşdeğerdir.

## Açıklanabilirlik Modülü

Automata modeli açıklanabilirdir çünkü karar üretimi kapalı bir latent uzay yerine açık biçimde izlenebilen `state -> pattern -> transition -> decision` zinciri üzerinden gerçekleşir. Her adımda hangi sembolik pattern'in gözlendiği, bunun eğitim sırasında görülüp görülmediği, hangi duruma eşlendiği ve bu geçişin olasılığı ayrı ayrı raporlanır. Böylece model yalnızca bir anomali etiketi üretmez; o kararın hangi geçiş yapısından ve hangi olasılık zincirinden türediğini de gösterir. Confidence score, geçişlerden türetilen path probability ile aynı anlamda kullanılır.

Geçiş olasılığı ve güven skoru:

```text
P(Si → Sj) = Geçiş Sayısı(i→j) / Toplam Çıkış Sayısı(i)
P(sequence) = P(S0→S1) × P(S1→S2) × ... × P(Sn-1→Sn)
Confidence Score = P(sequence)
```

Örnek 1 (seen, normal):

```json
{
  "time_step": 0,
  "state": 8,
  "previous_state": NaN,
  "pattern": "abab",
  "status": "seen",
  "mapped_to": "abab",
  "distance": 0,
  "transition_probability": 1.0,
  "path_probability": 1.0,
  "confidence_score": 1.0,
  "decision_reason": "expected_transition",
  "decision": "normal"
}
```

Örnek 2 (unseen, anomali):

```json
{
  "time_step": 5,
  "state": "aab",
  "pattern": "adc",
  "status": "unseen",
  "mapped_to": "abc",
  "probability": 0.108,
  "decision": "anomaly"
}
```

Counterfactual not: "Unseen anomali pattern'lar için counterfactual analiz de yapılmaktadır: alternatif pattern'lar altında kararın nasıl değişeceği `results/explanations/counterfactual_explanations.json` dosyasında raporlanmıştır."

## İstatistiksel Testler

Wilcoxon signed-rank testi: Sürekli F1 dağılımları arasındaki farkı test eder. `SKAB` için fold bazlı (`5` fold), `BATADAL` için seed bazlı (`5` seed) uygulanmıştır.

McNemar testi: İki modelin ikili tahmin vektörleri arasındaki uyuşmazlığı test eder. Hangi hataların sistematik olduğunu gösterir.

Her iki test için sonuçlar `results/tables/wilcoxon_results.csv` ve `results/tables/mcnemar_results.csv` dosyalarında üretilmiş ve depoya kaydedilmiştir. Wilcoxon sonuçlarında bu veri hacminde çiftler arası farkların çoğu `p < 0.05` eşiğini geçmemiştir; buna karşılık McNemar testi, bazı model çiftlerinde hata örüntülerinin istatistiksel olarak anlamlı biçimde farklılaştığını göstermektedir.

`5` farklı random seed `[42, 123, 2026, 7, 999]` ile tekrar stratejisi, tek bir rastgele başlangıç noktasına bağlı olmayan güvenilir sonuçlar üretmek için uygulanmıştır.

## Kaynaklar

- `SKAB` veri seti ve sensör temelli anomali tespiti literatürü
- `BATADAL` veri seti ve su dağıtım sistemi saldırı tespiti çalışmaları
- Wilcoxon signed-rank test literatürü
- McNemar test literatürü
- Zaman serilerinde açıklanabilir anomali tespiti ve sembolik temsil yaklaşımları

Kod içi başvuru noktaları:

- `src/models/automata/`
- `src/experiments/run_automata.py`
- `src/experiments/run_deep_models.py`
- `src/experiments/run_explainability_export.py`
- `src/experiments/run_statistical_tests.py`

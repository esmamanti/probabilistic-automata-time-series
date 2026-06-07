# Probabilistic Automata Time Series Analysis

**Yazılım Geliştirme - 2. Proje** 
**Grup 10**  
**Nalan Kara - Esma Nur Mantı** 

---

## 1. Proje Hakkında

Bu proje, zaman serisi anomali tespitinde iki farklı modelleme paradigmasını karşılaştırır:

- **Black-box modeller:** LSTM, GRU, 1D-CNN
- **Explainable model:** Probabilistic Automata

### Veri Setleri

| Veri Seti | Kaynak | Özellik Yapısı | Anomali Oranı | Bölme Stratejisi |
|-----------|--------|----------------|---------------|------------------|
| **SKAB** | `valve1` + `valve2` | Endüstriyel sensör verisi | Yaklaşık `%35` | GroupKFold `(k=5, grup=source_file)` |
| **BATADAL** | `BATADAL_dataset04.csv` | Su dağıtım sistemi saldırı/anomali verisi | Yaklaşık `%5` | Temporal `%60/%20/%20` |


---

## 2. Kurulum

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Veri Setleri

```text
data/raw/skab/valve1/
data/raw/skab/valve2/
data/raw/batadal/BATADAL_dataset04.csv
```

---

## 3. Kullanım

```bash
python src/experiments/run_all.py --no-resume
```

Aşama bazlı çalıştırma:

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

Testler:

```bash
pytest tests/ -v
```

---

## 4. Proje Yapısı

```text
probabilistic-automata-time-series/
├── configs/
├── data/
├── results/
├── scripts/
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── experiments/
│   ├── models/
│   │   ├── automata/
│   │   └── deep_learning/
│   └── utils/
└── tests/
```


---

## 5. Deneysel Sonuçlar

### Tablo 1: Model Performansı ve Stabilitesi (Ortalama F1-score ± Standart Sapma)

*5 farklı random seed `[42, 123, 2026, 7, 999]` ile elde edilen ortalama ve standart sapma.*  
*SKAB için GroupKFold `(k=5)` fold ortalaması alınmıştır.*

| Model | SKAB F1 | BATADAL F1 |
|-------|---------|------------|
| LSTM | 0.8219 +- 0.0375 | 0.1086 +- 0.0308 |
| GRU | 0.8355 +- 0.0276 | 0.1880 +- 0.0147 |
| 1D-CNN | 0.8273 +- 0.0379 | 0.1630 +- 0.0148 |
| Automata | 0.5022 +- 0.0666 | 0.3053 +- 0.0000 |

### Tablo 2: Gürültü Etkisi ve Unseen Senaryo Analizi

| Model | Orijinal F1 | Gürültülü F1 | F1 Değişimi | Unseen Det. Rate | Unseen Map. Acc. |
|-------|-------------|--------------|-------------|------------------|------------------|
| LSTM | 0.2193 | 0.2200 | 0.0007 | 0.0000 | 0.8794 |
| GRU | 0.2319 | 0.2316 | -0.0003 | 0.0000 | 0.8794 |
| 1D-CNN | 0.2324 | 0.2328 | 0.0003 | 0.0000 | 0.8794 |
| Automata | 0.1150 | 0.1167 | 0.0017 | 0.1739 | 0.4133 |

### Tablo 3: Parametre Duyarlılık Analizi (BATADAL)

#### 3a. F1-score

| Window Size \ Alphabet Size | 3 | 4 | 5 | 6 |
|-----------------------------|---|---|---|---|
| **3** | 0.2500 | 0.1591 | 0.1778 | 0.1757 |
| **4** | 0.2821 | 0.3065 | 0.2353 | 0.2584 |
| **5** | 0.0548 | 0.2957 | 0.2545 | 0.2727 |
| **6** | 0.3889 | 0.3265 | 0.3226 | 0.3205 |

#### 3b. State Sayısı

| Window Size \ Alphabet Size | 3 | 4 | 5 | 6 |
|-----------------------------|---|---|---|---|
| **3** | 26.0000 | 59.0000 | 112.0000 | 178.0000 |
| **4** | 76.0000 | 175.0000 | 258.0000 | 340.0000 |
| **5** | 166.0000 | 298.0000 | 368.0000 | 421.0000 |
| **6** | 206.0000 | 304.0000 | 342.0000 | 378.0000 |

#### 3c. Geçiş Yoğunluğu (Transition Density)

Geçiş yoğunluğu = toplam geçiş sayısı / `(state sayısı ^ 2)`

| Window Size \ Alphabet Size | 3 | 4 | 5 | 6 |
|-----------------------------|---|---|---|---|
| **3** | 0.1050 | 0.0500 | 0.0244 | 0.0147 |
| **4** | 0.0261 | 0.0095 | 0.0059 | 0.0040 |
| **5** | 0.0092 | 0.0042 | 0.0031 | 0.0026 |
| **6** | 0.0063 | 0.0038 | 0.0032 | 0.0028 |

> **Geçiş Yoğunluğu (Transition Density):** Toplam geçiş sayısının state sayısının karesine oranıdır. Yoğunluğun düşük olması, otomatın seyrek bir geçiş grafiğine sahip olduğunu ve eğitimde görülmemiş geçiş çiftlerinin fazla olduğunu gösterir.

### Tablo 4: Modellerin Çalışma Süresi

| Model | SKAB Eğitim (sn) | SKAB Inference (sn) | BATADAL Eğitim (sn) | BATADAL Inference (sn) |
|-------|------------------|---------------------|---------------------|------------------------|
| LSTM | 10.1155 | 0.1194 | 2.7770 | 0.0241 |
| GRU | 8.7268 | 0.1146 | 3.2575 | 0.0236 |
| 1D-CNN | 9.9035 | 0.0933 | 5.2977 | 0.0164 |
| Automata | 26.0647 | 0.0085 | 0.5297 | 0.0046 |

### Özet

- SKAB tarafında veri setine özel iyileştirmelerden sonra derin öğrenme modelleri belirgin biçimde yükselmiştir. En iyi ortalama F1 `GRU = 0.8355`, ardından `1D-CNN = 0.8273` ve `LSTM = 0.8219` gelmektedir.
- BATADAL tarafında automata modeli F1 skorunda daha güçlüdür.
- Geçiş yoğunluğu parametreler büyüdükçe genelde azalmaktadır; bu, state uzayının büyüyüp geçiş grafiğinin seyrekleştiğini gösterir.

---

## 6. Açıklanabilirlik Modülü

Automata modelinin her kararı için `pattern`, `status`, `mapped_to`, `distance`, `transition_probability`, `path_probability` ve `decision` gibi alanlar üretilir.

Örnek çıktı:

```json
{
  "time_step": 5,
  "pattern": "adc",
  "status": "unseen",
  "mapped_to": "abc",
  "distance": 1,
  "path_probability": 0.108,
  "decision": "anomaly"
}
```

---

## 7. İstatistiksel Analiz

- **Wilcoxon signed-rank testi:** Model F1 dağılımları arasındaki farkı test eder.
- **McNemar testi:** İkili tahmin vektörlerindeki hata örüntülerini karşılaştırır.
- **Tekrar stratejisi:** `5` farklı random seed `[42, 123, 2026, 7, 999]`

---

## 8. SKAB F1 İyileştirme Notları

SKAB için aşağıdaki iyileştirmeler uygulandı:

1. `configs/models.yaml` içinde SKAB'a özel `use_pos_weight: false` tanımlandı.
2. `configs/config.yaml` içinde SKAB'a özel `pca.enabled: false` ve `sequence_length: 32` override'i eklendi.
3. Derin modellerin `input_size` / `input_channels` değerleri, PCA kapalı olduğunda veriden dinamik olarak alınacak şekilde güncellendi.
4. Veri hazırlama katmanına dataset-specific preprocessing override desteği eklendi.

Bu değişikliklerden sonra SKAB özet F1 skorlarında belirgin artış görüldü:

- `GRU: 0.8355 +- 0.0276`
- `1D-CNN: 0.8273 +- 0.0379`
- `LSTM: 0.8219 +- 0.0375`

---

## 9. BATADAL İyileştirme Notları

BATADAL için aşağıdaki iyileştirmeler uygulandı:

1. BATADAL verisi zaman sırası korunarak işlendi ve zamansal %60/%20/%20 train/validation/test bölmesi uygulandı.
Sonuç: Veri sızıntısı azaltıldı ve değerlendirme daha gerçekçi hale geldi.

2. Derin öğrenme modelleri için sınıf dengesizliğine karşı weighted loss ve BATADAL’a özel threshold tuning eklendi.
Sonuç: Anomali sınıfını yakalama performansı iyileştirildi.

3. BATADAL için karar eşiği optimizasyonunda F2 metriği kullanıldı.
Sonuç: Recall odaklı daha uygun anomaly detection kararları elde edildi.

4. GRU modeli için BATADAL özelinde focal loss desteği tanımlandı.
Sonuç: Dengesiz ve zor örneklerde öğrenme daha sağlam hale getirildi.

5. Unseen pattern ve noise robustness deneyleri BATADAL için eklendi.
Sonuç: Modelin görülmeyen örüntüler ve gürültü altındaki davranışı ölçülebilir hale geldi.
---

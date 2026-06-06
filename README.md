# From Black-Box to Explainability: Probabilistic Automata for Time Series Analysis

**Grup 15**  
**Ders:** Yazilim Gelistirme - 2. Proje

---

## Icindekiler
1. [Proje Hakkinda](#1-proje-hakkinda)
2. [Kurulum](#2-kurulum)
3. [Kullanim](#3-kullanim)
4. [Proje Yapisi](#4-proje-yapisi)
5. [Deneysel Sonuclar](#5-deneysel-sonuclar)
6. [Aciklanabilirlik Modulu](#6-aciklanabilirlik-modulu)
7. [Istatistiksel Analiz](#7-istatistiksel-analiz)
8. [SKAB F1 Iyilestirme Notlari](#8-skab-f1-iyilestirme-notlari)

---

## 1. Proje Hakkinda

Bu proje, zaman serisi anomali tespitinde iki farkli modelleme paradigmasini karsilastirir:

- **Black-box modeller:** LSTM, GRU, 1D-CNN
- **Explainable model:** Probabilistic Automata

### Arastirma Sorusu
Farkli modelleme yaklasimlari, farkli veri kosullari altinda nasil davranmaktadir ve bu davranislar istatistiksel olarak anlamli midir?

### Veri Setleri

| Veri Seti | Kaynak | Ozellik Yapisi | Anomali Orani | Bolme Stratejisi |
|-----------|--------|----------------|---------------|------------------|
| **SKAB** | `valve1` + `valve2` | Endustriyel sensor verisi | Yaklasik `%35` | GroupKFold `(k=5, grup=source_file)` |
| **BATADAL** | `BATADAL_dataset04.csv` | Su dagitim sistemi saldiri/anomali verisi | Yaklasik `%5` | Temporal `%60/%20/%20` |

Bu repo yalnizca `SKAB` ve `BATADAL` veri setleri uzerine kuruludur.

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

## 3. Kullanim

```bash
python src/experiments/run_all.py --no-resume
```

Asama bazli calistirma:

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

## 4. Proje Yapisi

```text
probabilistic-automata-time-series/
├── configs/
├── data/
├── reports/
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

Kritik dosyalar:

- `configs/config.yaml`
- `configs/models.yaml`
- `src/experiments/run_all.py`
- `src/experiments/run_deep_models.py`
- `src/experiments/run_parameter_analysis.py`
- `src/models/automata/`

---

## 5. Deneysel Sonuclar

### Tablo 1: Model Performansi ve Stabilitesi (Ortalama F1-score +- Standart Sapma)

*5 farkli random seed `[42, 123, 2026, 7, 999]` ile elde edilen ortalama ve standart sapma.*  
*SKAB icin GroupKFold `(k=5)` fold ortalamasi alinmistir.*

| Model | SKAB F1 | BATADAL F1 |
|-------|---------|------------|
| LSTM | 0.8219 +- 0.0375 | 0.1086 +- 0.0308 |
| GRU | 0.8355 +- 0.0276 | 0.1880 +- 0.0147 |
| 1D-CNN | 0.8273 +- 0.0379 | 0.1630 +- 0.0148 |
| Automata | 0.5022 +- 0.0666 | 0.3053 +- 0.0000 |

### Tablo 2: Gurultu Etkisi ve Unseen Senaryo Analizi

| Model | Orijinal F1 | Gurultulu F1 | F1 Degisimi | Unseen Det. Rate | Unseen Map. Acc. |
|-------|-------------|--------------|-------------|------------------|------------------|
| LSTM | 0.2193 | 0.2200 | 0.0007 | 0.0000 | 0.8794 |
| GRU | 0.2319 | 0.2316 | -0.0003 | 0.0000 | 0.8794 |
| 1D-CNN | 0.2324 | 0.2328 | 0.0003 | 0.0000 | 0.8794 |
| Automata | 0.1150 | 0.1167 | 0.0017 | 0.1739 | 0.4133 |

### Tablo 3: Parametre Duyarlilik Analizi (BATADAL)

#### 3a. F1-score

| Window Size \ Alphabet Size | 3 | 4 | 5 | 6 |
|-----------------------------|---|---|---|---|
| **3** | 0.2500 | 0.1591 | 0.1778 | 0.1757 |
| **4** | 0.2821 | 0.3065 | 0.2353 | 0.2584 |
| **5** | 0.0548 | 0.2957 | 0.2545 | 0.2727 |
| **6** | 0.3889 | 0.3265 | 0.3226 | 0.3205 |

#### 3b. State Sayisi

| Window Size \ Alphabet Size | 3 | 4 | 5 | 6 |
|-----------------------------|---|---|---|---|
| **3** | 26.0000 | 59.0000 | 112.0000 | 178.0000 |
| **4** | 76.0000 | 175.0000 | 258.0000 | 340.0000 |
| **5** | 166.0000 | 298.0000 | 368.0000 | 421.0000 |
| **6** | 206.0000 | 304.0000 | 342.0000 | 378.0000 |

#### 3c. Gecis Yogunlugu (Transition Density)

Gecis yogunlugu = toplam gecis sayisi / `(state sayisi ^ 2)`

| Window Size \ Alphabet Size | 3 | 4 | 5 | 6 |
|-----------------------------|---|---|---|---|
| **3** | 0.1050 | 0.0500 | 0.0244 | 0.0147 |
| **4** | 0.0261 | 0.0095 | 0.0059 | 0.0040 |
| **5** | 0.0092 | 0.0042 | 0.0031 | 0.0026 |
| **6** | 0.0063 | 0.0038 | 0.0032 | 0.0028 |

> **Gecis Yogunlugu (Transition Density):** Toplam gecis sayisinin state sayisinin karesine oranidir. Yogunlugun dusuk olmasi, otomatanin seyrek bir gecis grafigine sahip oldugunu ve egitimde gorulmemis gecis ciftlerinin fazla oldugunu gosterir.

### Tablo 4: Modellerin Calisma Suresi

| Model | SKAB Egitim (sn) | SKAB Inference (sn) | BATADAL Egitim (sn) | BATADAL Inference (sn) |
|-------|------------------|---------------------|---------------------|------------------------|
| LSTM | 10.1155 | 0.1194 | 2.7770 | 0.0241 |
| GRU | 8.7268 | 0.1146 | 3.2575 | 0.0236 |
| 1D-CNN | 9.9035 | 0.0933 | 5.2977 | 0.0164 |
| Automata | 26.0647 | 0.0085 | 0.5297 | 0.0046 |

### Ozet Bulgular

- SKAB tarafinda veri setine ozel iyilestirmelerden sonra derin ogrenme modelleri belirgin bicimde yukselmistir. En iyi ortalama F1 `GRU = 0.8355`, ardindan `1D-CNN = 0.8273` ve `LSTM = 0.8219` gelmektedir.
- BATADAL tarafinda automata modeli F1 skorunda daha gucludur.
- Gecis yogunlugu parametreler buyudukce genelde azalmaktadir; bu, state uzayinin buyuyup gecis grafinin seyreklestigini gosterir.

---

## 6. Aciklanabilirlik Modulu

Automata modelinin her karari icin `pattern`, `status`, `mapped_to`, `distance`, `transition_probability`, `path_probability` ve `decision` gibi alanlar uretilir.

Ornek cikti:

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

## 7. Istatistiksel Analiz

- **Wilcoxon signed-rank testi:** Model F1 dagilimlari arasindaki farki test eder.
- **McNemar testi:** Ikili tahmin vektorlerindeki hata oruntulerini karsilastirir.
- **Tekrar stratejisi:** `5` farkli random seed `[42, 123, 2026, 7, 999]`

Uretilen dosyalar:

- `results/tables/deep_learning_wilcoxon.csv`
- `results/tables/deep_learning_mcnemar.csv`
- `results/tables/model_comparison_wilcoxon.csv`
- `results/tables/model_comparison_mcnemar.csv`

---

## 8. SKAB F1 Iyilestirme Notlari

SKAB icin asagidaki iyilestirmeler uygulandi:

1. `configs/models.yaml` icinde SKAB'a ozel `use_pos_weight: false` tanimlandi.
2. `configs/config.yaml` icinde SKAB'a ozel `pca.enabled: false` ve `sequence_length: 32` override'i eklendi.
3. Derin modellerin `input_size` / `input_channels` degerleri, PCA kapali oldugunda veriden dinamik olarak alinacak sekilde guncellendi.
4. Veri hazirlama katmanina dataset-specific preprocessing override destegi eklendi.

Bu degisikliklerden sonra SKAB ozet F1 skorlarinda belirgin artis goruldu:

- `GRU: 0.8355 +- 0.0276`
- `1D-CNN: 0.8273 +- 0.0379`
- `LSTM: 0.8219 +- 0.0375`

Bir sonraki mantikli adimlar:

1. SKAB icin `sequence_length` uzerinde `32 / 48 / 64` kucuk bir grid taramasi yapmak.
2. Threshold tuning'de SKAB icin daha yuksek alt esik (`min_threshold`) denemek.
3. SKAB icin model bazli kayip fonksiyonu varyantlari (`BCE` vs `focal`) karsilastirmak.

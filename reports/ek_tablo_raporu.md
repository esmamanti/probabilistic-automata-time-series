---
## Proje Notları

Bu rapor SKAB ve BATADAL veri setleri uzerinde calismaktadir.
SWAT ve WADI veri setleri bu projede kullanilmamistir.
EK sablonundaki SWAT ve WADI sutunlari bu proje icin gecerli degildir;
ilgili hucreler "N/A" olarak birakilmistir.
---

## Tablo 1: Model Performansi ve Stabilitesi (Ortalama F1-score ± Standart Sapma)

| Model    | SKAB            | BATADAL         | SWAT | WADI |
|----------|-----------------|-----------------|------|------|
| LSTM | 0.5021 ± 0.0381 | 0.1086 ± 0.0308 | N/A  | N/A  |
| GRU | 0.4978 ± 0.0334 | 0.1880 ± 0.0147 | N/A  | N/A  |
| 1D-CNN | 0.4912 ± 0.0573 | 0.1630 ± 0.0148 | N/A  | N/A  |
| Automata | 0.5022 ± 0.0666 | 0.3053 ± 0.0000 | N/A  | N/A  |

*5 farkli random seed [42, 123, 2026, 7, 999] ile elde edilen ortalama ve standart sapma.
SKAB icin GroupKFold (k=5) fold ortalamasi alinmistir.*

## Tablo 2: Gurultu Etkisi ve Unseen Senaryo Analizi

| Model    | Orijinal F1 | Gurultulu F1 | F1 Degisimi | Unseen Det. Rate | Unseen Map. Acc. |
|----------|-------------|--------------|-------------|------------------|------------------|
| LSTM | 0.2193 | 0.2200 | 0.0007 | 0.0000 | 0.8794 |
| GRU | 0.2319 | 0.2316 | -0.0003 | 0.0000 | 0.8794 |
| 1D-CNN | 0.2324 | 0.2328 | 0.0003 | 0.0000 | 0.8794 |
| Automata | 0.1150 | 0.1167 | 0.0017 | 0.1739 | 0.4133 |

## Tablo 4a: Automata Parametre Duyarlilik Analizi — F1-score (BATADAL)

| Window Size \ Alphabet Size | 3 | 4 | 5 | 6 |
|-----------------------------|---|---|---|---|
| 3 | 0.2500 | 0.1591 | 0.1778 | 0.1757 |
| 4 | 0.2821 | 0.3065 | 0.2353 | 0.2584 |
| 5 | 0.0548 | 0.2957 | 0.2545 | 0.2727 |
| 6 | 0.3889 | 0.3265 | 0.3226 | 0.3205 |

## Tablo 4b: Automata Parametre Duyarlilik Analizi — State Sayisi

| Window Size \ Alphabet Size | 3 | 4 | 5 | 6 |
|-----------------------------|---|---|---|---|
| 3 | 26.0000 | 59.0000 | 112.0000 | 178.0000 |
| 4 | 76.0000 | 175.0000 | 258.0000 | 340.0000 |
| 5 | 166.0000 | 298.0000 | 368.0000 | 421.0000 |
| 6 | 206.0000 | 304.0000 | 342.0000 | 378.0000 |

## Tablo 4c: Gecis Yogunlugu (Transition Density)

Gecis yogunlugu = toplam gecis sayisi / (state sayisi ^ 2)
(Her parametre kombinasyonu icin hesaplandi)

| Window Size \ Alphabet Size | 3 | 4 | 5 | 6 |
|-----------------------------|---|---|---|---|
| 3 | 0.1050 | 0.0500 | 0.0244 | 0.0147 |
| 4 | 0.0261 | 0.0095 | 0.0059 | 0.0040 |
| 5 | 0.0092 | 0.0042 | 0.0031 | 0.0026 |
| 6 | 0.0063 | 0.0038 | 0.0032 | 0.0028 |

## Tablo 5: Modellerin Calisma Suresi

| Model    | SKAB Egitim (sn) | SKAB Inference (sn) | BATADAL Egitim (sn) | BATADAL Inference (sn) |
|----------|------------------|---------------------|---------------------|------------------------|
| LSTM | 10.1155 | 0.1194 | 2.7770 | 0.0241 |
| GRU | 8.7268 | 0.1146 | 3.2575 | 0.0236 |
| 1D-CNN | 9.9035 | 0.0933 | 5.2977 | 0.0164 |
| Automata | 26.0647 | 0.0085 | 0.5297 | 0.0046 |

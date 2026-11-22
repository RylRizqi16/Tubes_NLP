# T2 — NER (`LOCATION`, `ORGANIZATION`, `PERSON`) & LDA Topic Modeling

T2 membaca `data/processed/master_dataset.jsonl` tanpa mengubah teks, label, maupun split master:
- **NER (`entities != null`)**: 9.257 dokumen (`6.479` train / `1.388` val / `1.390` test — `976` real + `414` synthetic) dengan 21.002 entitas (`LOCATION`: 7.993, `ORGANIZATION`: 6.946, `PERSON`: 6.063) dan `0%` kehilangan/pemotongan span pada `max_length=384`.
- **LDA (`split == "train"` untuk fitting)**: 6.468 dokumen train (`campus_complaints`: 3.500, `synthetic`: 1.932, `sp4n_lapor`: 1.036); `val` (1.386) dan `test` (1.387) hanya untuk evaluasi *held-out perplexity*.

## Menjalankan Lokal (CPU)

Dari root `Tubes_NLP`:

```bash
# 1. Unit & contract tests
python -m unittest src.task2.test_task2 -v

# 2. E1 NER CPU Baseline (Feature BIO2 + Cased Civic Gazetteer & 3-Gate Comma Merger)
python -m src.task2.train_ner_baseline

# 3. LDA Topic Modeling (otomatis memakai gensim bila terpasang, atau fallback sklearn + PMI bigrams)
python -m src.task2.train_lda --k-values 5 8 10 12 15

# 4. Evaluasi 2x2 NER (E1 + seluruh checkpoint IndoBERT yang tersedia)
python -m src.task2.evaluate_ner --split test

# 5. Inferensi CLI (NER + opsional LDA topic)
python -m src.task2.predictor "Lampu jalan di Jl. Setiabudi, Srondol mati 5 hari, mohon Dishub Kota Semarang segera perbaiki." --with-topic
```

## Menjalankan Transformer NER (E2–E4, GPU / Colab)

Gunakan `src/task2/T2_Run.ipynb` di Google Colab GPU (L4/T4), atau jalankan langsung:

```bash
# ner_full (Real + Synthetic, 6.479 train) -> E2 (Raw) & E3 (Main: + Civic Post-Processing)
python -m src.task2.train_ner_transformer \
    --model-name indobenchmark/indobert-base-p1 \
    --seeds 42 123 2025 \
    --epochs 5 --batch-size 16 --lr 3e-5 --max-length 384 --patience 2 --fp16

# ner_real_only (Real-Only Ablation, 4.547 train) -> E4a (Raw) & E4b (+ Civic Post-Processing)
python -m src.task2.train_ner_transformer \
    --model-name indobenchmark/indobert-base-p1 \
    --seeds 42 123 2025 \
    --epochs 5 --batch-size 16 --lr 3e-5 --max-length 384 --patience 2 --fp16 --real-only

# Opsional: tambahkan --use-crf untuk Linear-Chain CRF head (pure PyTorch, tanpa dependency tambahan)
python -m src.task2.evaluate_ner --split test
```

## Ringkasan Eksperimen

| Eksperimen | Konfigurasi | Artefak / Laporan |
| :--- | :--- | :--- |
| **E1 · Baseline (CPU)** | Feature BIO2 (`LogisticRegression`) + Cased Civic Gazetteer & 3-Gate `, ` Post-Processing | `models/task2/ner_baseline/baseline_model.joblib`, `outputs/task2_ner_e1.json` |
| **E2 · Transformer** | IndoBERT (`ner_full`, 3 seed), Raw BIO2 Decoding | `models/task2/ner_full/seed_*/` |
| **E3 · Model Utama** | IndoBERT (`ner_full`, 3 seed) + WordPiece `##` Repair + Cased Civic Gazetteer & 3-Gate `, ` Post-Processing | `outputs/task2_ner_eval_report.json` |
| **E4a / E4b · Ablation** | IndoBERT (`ner_real_only`, 3 seed), Raw vs. + Civic Post-Processing | `models/task2/ner_real_only/seed_*/`, `outputs/task2_ner_eval_report.json` |
| **LDA · Topic Modeling** | 4-Layer Stopwords + Bigrams + Sweep $K \in \{5,8,10,12,15\}$ | `models/task2/lda/`, `outputs/task2_lda_report.json` |

### Hasil Evaluasi NER 2×2 (`Test` Split, 3 Seed: `42, 123, 2025`)

Dari `outputs/task2_ner_eval_report.json` & `src/task2/T2_Run.ipynb`:
- **Span alignment (`max_length=384`)**: `21.002 / 21.002` (`0` terpotong, loss `0,000000`).
- **Oracle upper bound (`gold -> postprocessing`)**: `val (all/real/synthetic)` = **`1,0000`** (`0 / 1.388` dokumen rusak).
- **Val safeguard**: `val (real)` `0,7134` $\rightarrow$ `0,7152` (`+0,0018`, `no_real_regression=True`); `val (synthetic)` `0,6778` $\rightarrow$ **`0,9948`** (`+0,3170`).

| Konfigurasi | Test All (Strict Micro / Macro F1) | Test Real (Strict Micro / Macro F1) | Test Synthetic (Strict Micro / Macro F1) | Partial-Boundary F1 (All / Syn) |
| :--- | :---: | :---: | :---: | :---: |
| **E1 · Baseline (Civic BIO, CPU)** | `0.7900 / 0.7903` | `0.7102 / 0.7070` | `1.0000 / 1.0000` | `0.8554 / 1.0000` |
| **E2 · IndoBERT (`ner_full`, Raw)** | `0.8619 ± 0.0029 / 0.8643 ± 0.0034` | `0.8299 ± 0.0055 / 0.8262 ± 0.0066` | `0.9505 ± 0.0045 / 0.9546 ± 0.0032` | `0.9195 ± 0.0019 / 0.9914 ± 0.0033` |
| **E3 · Main (`ner_full` + Civic Hybrid)** | **`0.8794 ± 0.0032 / 0.8814 ± 0.0038`** | **`0.8361 ± 0.0043 / 0.8327 ± 0.0052`** | **`0.9996 ± 0.0007 / 0.9997 ± 0.0005`** | **`0.9220 ± 0.0019 / 0.9996 ± 0.0007`** |
| **E4a · Ablation (`ner_real_only`, Raw)** | `0.6122 ± 0.0135 / 0.6376 ± 0.0154` | `0.8191 ± 0.0212 / 0.8158 ± 0.0216` | `0.1361 ± 0.0105 / 0.1283 ± 0.0188` | `0.8431 ± 0.0117 / 0.7479 ± 0.0209` |
| **E4b · Ablation (`ner_real_only` + Hybrid)** | `0.8599 ± 0.0157 / 0.8632 ± 0.0157` | `0.8265 ± 0.0191 / 0.8232 ± 0.0195` | `0.9519 ± 0.0197 / 0.9616 ± 0.0151` | `0.9027 ± 0.0098 / 0.9562 ± 0.0161` |

| Konfigurasi | LOCATION F1 | ORGANIZATION F1 | PERSON F1 | `indonlu_nergrit` | `indolem_nerui` | `idner_news_2k` | `synthetic` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1 · Baseline (Civic BIO, CPU)** | `0.7840` | `0.7940` | `0.7930` | `0.5855` | `0.8200` | `0.7728` | `1.0000` |
| **E2 · IndoBERT (`ner_full`, Raw)** | `0.8440 ± 0.0055` | `0.8423 ± 0.0103` | `0.9066 ± 0.0058` | `0.7549 ± 0.0079` | `0.8975 ± 0.0075` | `0.8708 ± 0.0146` | `0.9505 ± 0.0045` |
| **E3 · Main (`ner_full` + Civic Hybrid)** | **`0.8641 ± 0.0036`** | **`0.8657 ± 0.0086`** | **`0.9142 ± 0.0079`** | **`0.7619 ± 0.0049`** | **`0.9047 ± 0.0091`** | **`0.8744 ± 0.0164`** | **`0.9996 ± 0.0007`** |
| **E4a · Ablation (`ner_real_only`, Raw)** | `0.4820 ± 0.0076` | `0.6175 ± 0.0264` | `0.8133 ± 0.0138` | `0.7483 ± 0.0170` | `0.8780 ± 0.0307` | `0.8652 ± 0.0239` | `0.1361 ± 0.0105` |
| **E4b · Ablation (`ner_real_only` + Hybrid)** | `0.8405 ± 0.0100` | `0.8363 ± 0.0253` | `0.9126 ± 0.0122` | `0.7571 ± 0.0175` | `0.8852 ± 0.0250` | `0.8703 ± 0.0257` | `0.9519 ± 0.0197` |

### Hasil Evaluasi LDA Topic Modeling (`outputs/task2_lda_report.json`)
- **Backend Gensim (`T2_Run.ipynb`, `train=6.468`, `vocab=1.342`, `alpha='auto'`, `eta='auto'`)**:
  - **Terpilih**: `K = 15` (`C_v = 0.4391`, `U_mass = -12.9338`, `Topic Diversity = 0.9267`, `Val Perplexity = 1619.67`, `Test Perplexity = 1630.14`; `K = 12` mencapai `Domain Purity = 0.7292`, `NMI = 0.3124`).
  - **Domain Alignment (`K = 15`)**: All (`purity = 0.6674`, `NMI = 0.2412`), Synthetic (`purity = 0.4834`, `NMI = 0.2934`), Campus (`purity = 0.9142`, `NMI = 0.1018`).
- **Backend Sklearn Fallback Lokal (`train=6.468`, `vocab=1.169`, PMI Bigrams)**:
  - **Terpilih**: `K = 15` (`C_v = 0.4841`, `U_mass = -11.4470`, `Topic Diversity = 0.8867`, `Val Perplexity = 803.24`, `Test Perplexity = 784.69`).
  - **Domain Alignment (`K = 15`)**: All (`purity = 0.7359`, `NMI = 0.3032`), Synthetic (`purity = 0.6863`, `NMI = 0.4357`), Campus (`purity = 0.9398`, `NMI = 0.1653`).

## Kontrak Integrasi (`src/schema.py`)

```python
from src.task2 import T2Predictor, predict, predict_topic

t2 = T2Predictor()  # otomatis memuat best seed di models/task2/ner_full/ atau fallback ke E1 CPU
result = t2.predict("Lampu jalan di Jl. Setiabudi, Srondol mati, mohon Dishub Kota Semarang cek.")
# T2Result(entities=[EntitySpan(text='Jl. Setiabudi, Srondol', label='LOCATION', start=15, end=37), ...])

topic = t2.predict_topic("Aplikasi SP4N LAPOR sering error saat login.")
# {'topic_id': ..., 'topic_probability': ..., 'top_keywords': [...], 'topic_distribution': {...}}
```

**Jaminan keamanan offset:**
- `T2Predictor.predict(text)` **tidak** melakukan pembersihan yang mengubah panjang string sebelum ekstraksi offset, sehingga `text[span.start:span.end] == span.text` selalu terpenuhi terhadap string persis yang dikirim pemanggil.
- Setiap keluaran `T2Result.entities` terurut menurut `(start, end)`, bebas tumpang-tindih (*non-overlapping*), dan berada dalam batas `0 <= start < end <= len(text)`.

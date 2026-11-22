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

Hasil terverifikasi saat ini pada E1 CPU Baseline (`outputs/task2_ner_e1.json`):
- **Span alignment (`max_length=384`)**: `21.002 / 21.002` (`0` terpotong, loss `0,000000`).
- **Oracle upper bound (`gold -> postprocessing`)**: `val (all/real/synthetic)` = **`1,0000`** (`0 / 1.388` dokumen rusak).
- **Val safeguard**: `val (real)` `0,7169` $\rightarrow$ `0,7187` (`+0,0018`, `no_real_regression=True`); `val (synthetic)` `0,7029` $\rightarrow$ **`0,9948`** (`+0,2919`).
- **Test E1**: All Strict Micro-F1 **`0,7922`** / Macro-F1 **`0,7929`** (Real: `0,7132` / `0,7101`; Synthetic: **`1,0000`** / **`1,0000`**; Partial-Boundary All: `0,8567`).

Hasil terverifikasi saat ini pada LDA Lokal (`outputs/task2_lda_report.json`, backend `sklearn`, `train=6.468`, `vocab=1.169` setelah 4-layer stopwords + PMI bigrams + `filter_extremes`):
- **Terpilih**: `K = 15` (`C_v = 0,4841`, `U_mass = -11,4470`, `Topic Diversity = 0,8867`, `Val Perplexity = 803,24`, `Test Perplexity = 784,69`), mencakup seluruh 5 domain pengaduan (`Infrastruktur & PU`, `Kesehatan & Sanitasi`, `Layanan & Administrasi`, `Lingkungan`, `Pungli & Tata Kelola`).
- **Domain Alignment**: All (`purity = 0,7359`, `NMI = 0,3032`), Synthetic (`purity = 0,6863`, `NMI = 0,4357`), Campus (`purity = 0,9398`, `NMI = 0,1653`).
- Catatan: Jalankan `python -m src.task2.train_lda --backend gensim` di Colab (`T2_Run.ipynb`) untuk versi `gensim` penuh dengan `alpha='auto'`, `eta='auto'`.

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

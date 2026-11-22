# Panduan Singkat Dataset per Task

Semua task membaca file yang sama: `data/processed/master_dataset.jsonl` (atau `.parquet`).  
Gunakan `split == "train"` untuk melatih, `"val"` untuk tuning, dan `"test"` untuk evaluasi.

| Task (PIC) | Sumber (`source`) yang Dipakai | Kolom yang Dibaca | Catatan Evaluasi (`test`) |
| :--- | :--- | :--- | :--- |
| **T1 · Domain & Intensi**<br>*(Farhan)* | • `synthetic` (domain + intent lengkap)<br>• `campus_complaints` (domain tambahan) | `text`, `domain`, `intent`, `split`, `source` | Filter `domain != null` / `intent != null`. |
| **T2 · NER + LDA**<br>*(Raka)* | • **NER:** `indonlu_nergrit`, `indolem_nerui`, `idner_news_2k`, `synthetic`<br>• **LDA:** `sp4n_lapor`, `campus_complaints`, `synthetic` *(train saja)* | `text`, `entities`, `split`, `source` | Filter `entities != null`. Pisahkan skor `test` nyata vs `synthetic`. |
| **T3 · Sentimen & Sarkasme**<br>*(Darryl)* | • **Sentimen:** `sp4n_lapor`, `ecommerce_sentiment`, `rizalespe_*`, `synthetic`<br>• **Sarkasme:** `twitter_indonesia_sarcastic`, `campus_complaints`, `ecommerce_sentiment`, `synthetic` | `text`, `sentiment`, `is_sarcasm`, `split`, `source` | Filter `sentiment != null` / `is_sarcasm != null`. Pisahkan skor `test` nyata vs `synthetic`. |
| **Priority & End-to-End**<br>*(Integrasi Tim)* | • `synthetic` *(punya semua label T1+T2+T3+urgensi)* | `text`, `domain`, `intent`, `entities`, `sentiment`, `is_sarcasm`, `urgency_level`, `split` | Uji pada `source == "synthetic"` & `split == "test"`. |

## Contoh Filter Cepat (Python)

```python
import pandas as pd, json

df = pd.read_json("data/processed/master_dataset.jsonl", lines=True)

t1_df = df[df["domain"].notna()][["id", "source", "split", "text", "domain", "intent"]]
t2_df = df[df["entities"].notna()][["id", "source", "split", "text", "entities"]]
t3_df = df[df["sentiment"].notna() | df["is_sarcasm"].notna()][["id", "source", "split", "text", "sentiment", "is_sarcasm"]]
e2e_test = df[(df["source"] == "synthetic") & (df["split"] == "test")]
```

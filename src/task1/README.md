# T1 — Domain dan intensi pengaduan

T1 membaca `data/processed/master_dataset.jsonl` dan mengikuti label/split yang sudah ada. Lima domain dan lima intensi sesuai `data/TASK_GUIDE.md` dan presentasi. Label kosong diabaikan untuk target yang bersangkutan: domain memiliki 5.940 contoh, sedangkan intensi hanya 2.760 contoh sintetis. Teks, label, dan split tidak dihitung ulang pada tahap T1.

## Jalankan

Dari root `Tubes_NLP`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.task1.train_baseline
python -m src.task1.predictor "Jalan rusak di depan sekolah, mohon segera diperbaiki."
```

E1 menyetel `C` di validation (0.1, 1, 10), lalu mengevaluasi test sekali. Artefak ada di `models/task1/baseline.joblib`, laporan di `outputs/task1_e1.json`. TF-IDF di-fit hanya pada train. Model domain memakai train nyata dan sintetis; model intensi hanya train sintetis.

Hasil E1 pada dataset repo saat ini: domain test 891 contoh, accuracy dan macro-F1 **1,0000**; intensi test 414 contoh sintetis, accuracy **0,6208** dan macro-F1 **0,4252**. Domain test non-sintetis hanya mencakup dua kelas (`Layanan & Administrasi` dan `Infrastruktur & PU`) dari korpus kampus. Skor sempurna ini tidak membuktikan generalisasi ke keluhan publik nyata di lima domain. Kelas `Saran` dan `Apresiasi` pada intensi mendapat recall 0 pada test; perlu perbaikan data/model sebelum dipakai untuk routing operasional.

Untuk E2–E4, diperlukan unduhan pertama bobot XLM-R dan komputasi GPU. Perintah berikut menjalankan tiga seed untuk masing-masing konfigurasi:

```bash
python -m src.task1.train_transformer --experiments E2 E3 E4_domain E4_intent --seeds 42 43 44
python -m src.task1.compare_experiments
python -m src.task1.predictor "Lampu jalan mati, tolong perbaiki" --checkpoint models/task1/E3/seed_42
```

- E2: XLM-R dua head, cross-entropy.
- E3: XLM-R dua head, Focal Loss γ=2 dan bobot kelas dari train.
- E4: dua model single-task dengan backbone dan cross-entropy setara E2. Laporan E4 hanya menghitung skor head yang dilatih.

Epoch terbaik dipilih pada validation; early stopping memiliki patience 2. Checkpoint dan laporan disimpan per konfigurasi/seed. `compare_experiments` menghitung mean dan simpangan baku dari laporan yang tersedia. Laporan memuat macro-F1, akurasi, recall per kelas, support, dan confusion matrix. Macro-F1 dihitung hanya pada kelas yang hadir pada gold subset; `scored_labels` merekam kelas tersebut. Hasil test dipisah untuk sumber non-sintetis dan sintetis. **Akurasi intensi pada data non-sintetis belum dapat diukur** karena sumber itu tidak memiliki label intensi. Confidence adalah probabilitas model yang belum dikalibrasi.

## Kontrak untuk integrasi

```python
from src.task1.predictor import T1Predictor

t1 = T1Predictor("models/task1/E3/seed_42")  # atau tanpa argumen untuk E1
result = t1.predict("Jalan depan rumah rusak parah")
# T1Result(domain=..., intent=..., domain_confidence=..., intent_confidence=...)
```

Instansiasi satu kali untuk banyak laporan. Saat mengintegrasikan T2 dan T3, tetapkan satu versi teks untuk seluruh modul; offset entitas T2 harus merujuk ke versi teks yang disimpan pada event. Hasil ketiga modul dapat digabung menurut `src/schema.py`. Pilih checkpoint berdasar validation, simpan pilihan itu sebelum melihat metrik test untuk pembandingan final. Jangan menggunakan output single-task E4 untuk kedua label.

Prediktor membersihkan teks baru dengan aturan yang sama seperti dataset (HTML, URL, karakter kontrol, spasi berlebih), sambil mempertahankan negasi dan huruf besar/kecil. Dataset training sudah dibersihkan oleh `scripts/build_master_dataset.py`; tidak perlu menjalankan preprocessing ulang untuk menjalankan T1. Jika data mentah hendak dibangun ulang, unduh dulu seluruh sumber dengan `scripts/download_datasets.py`.

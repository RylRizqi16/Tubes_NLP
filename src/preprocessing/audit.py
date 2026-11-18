"""
Modul Audit dan Verifikasi Dataset (Tahap 1)
Digunakan untuk mengecek konsistensi data, menghitung missing values/duplikasi,
serta memverifikasi total baris riil antar sumber.
"""

import os
import pandas as pd
from typing import Dict, Any


def audit_dataframe(df: pd.DataFrame, source_name: str = "Dataset") -> Dict[str, Any]:
    """Menghasilkan ringkasan kualitas data untuk DataFrame."""
    print(f"\n{'='*20} AUDIT: {source_name} {'='*20}")
    print(f"Total Baris      : {len(df):,}")
    print(f"Total Kolom      : {len(df.columns)}")
    print(f"Daftar Kolom     : {df.columns.tolist()}")
    
    print("\n--- Missing Values ---")
    null_counts = df.isnull().sum()
    print(null_counts[null_counts > 0] if null_counts.sum() > 0 else "Tidak ada missing value.")
    
    dup_rows = df.duplicated().sum()
    print(f"\n--- Duplikasi Baris Lengkap: {dup_rows:,} baris ---")
    
    summary = {
        "source": source_name,
        "rows": len(df),
        "columns": df.columns.tolist(),
        "missing": null_counts.to_dict(),
        "duplicates": int(dup_rows)
    }
    return summary


def inspect_file(filepath: str, text_col: str = None) -> None:
    """Membaca dan mengaudit file CSV/Excel langsung dari path."""
    if not os.path.exists(filepath):
        print(f"File belum ditemukan: {filepath}")
        return

    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(filepath)
    else:
        print(f"Format tidak didukung: {filepath}")
        return

    name = os.path.basename(filepath)
    audit_dataframe(df, source_name=name)

    # Cek sampel teks jika kolom teks diketahui
    if text_col and text_col in df.columns:
        print(f"\n--- 3 Sampel Teks dari Kolom '{text_col}' ---")
        for i, sample in enumerate(df[text_col].dropna().head(3), 1):
            print(f"{i}. {str(sample)[:120]}...")


if __name__ == "__main__":
    # Test audit pada file yang sudah ada
    sp4n_path = os.path.join("data", "raw", "sp4n", "sp4n_lapor.csv")
    if os.path.exists(sp4n_path):
        inspect_file(sp4n_path, text_col="content")

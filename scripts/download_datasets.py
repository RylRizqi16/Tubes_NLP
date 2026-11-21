"""
Download and inspect all public and local raw datasets for the Indonesian Public Complaint NLP project.

Sources handled:
1. Local (restored from git history commit f4c01c94):
   - SP4N LAPOR (`data/raw/sp4n/sp4n_lapor.csv`)
2. Hugging Face (`datasets.load_dataset`):
   - `YogaSWD/indonesian-campus-complaints`
   - `w11wo/twitter_indonesia_sarcastic`
   - `AIbnuHibban/e-commerce-sentiment-bahasa-indonesia`
3. SEACrowd NER datasets (canonical upstream CoNLL/TSV files declared in SEACrowd loading scripts):
   - `SEACrowd/indonlu_nergrit`
   - `SEACrowd/indolem_nerui` (Fold 1 only: train.01.tsv, dev.01.tsv, test.01.tsv)
   - `SEACrowd/idner_news_2k`
4. GitHub (`rizalespe/Dataset-Sentimen-Analisis-Bahasa-Indonesia`):
   - `dataset_tweet_sentiment_cellular_service_provider.csv`
   - `dataset_tweet_sentiment_pilkada_DKI_2017.csv`

Run:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --force
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests
from datasets import Dataset, concatenate_datasets, load_dataset


ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
MANIFEST_PATH = RAW_DIR / "download_manifest.json"
SP4N_GIT_REF = "f4c01c94:Dataset/SP4N LAPOR.csv"
REQUEST_TIMEOUT = 60

NER_SOURCES: Dict[str, Dict[str, Any]] = {
    "indonlu_nergrit": {
        "hf_repo": "SEACrowd/indonlu_nergrit",
        "note": "SEACrowd HF repo uses a legacy loading script requiring seacrowd/torchaudio; downloaded directly from its canonical IndoNLU upstream URLs.",
        "files": {
            "train_preprocess.txt": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/nergrit_ner-grit/train_preprocess.txt",
            "valid_preprocess.txt": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/nergrit_ner-grit/valid_preprocess.txt",
            "test_preprocess.txt": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/nergrit_ner-grit/test_preprocess.txt",
        },
    },
    "indolem_nerui": {
        "hf_repo": "SEACrowd/indolem_nerui",
        "note": "SEACrowd HF repo uses a legacy loading script; downloaded Fold 1 only from canonical IndoLEM upstream URLs to avoid 5-fold duplication.",
        "files": {
            "train.01.tsv": "https://raw.githubusercontent.com/indolem/indolem/main/ner/data/nerui/train.01.tsv",
            "dev.01.tsv": "https://raw.githubusercontent.com/indolem/indolem/main/ner/data/nerui/dev.01.tsv",
            "test.01.tsv": "https://raw.githubusercontent.com/indolem/indolem/main/ner/data/nerui/test.01.tsv",
        },
    },
    "idner_news_2k": {
        "hf_repo": "SEACrowd/idner_news_2k",
        "note": "SEACrowd HF repo uses a legacy loading script; downloaded directly from canonical idner-news-2k upstream URLs.",
        "files": {
            "train.txt": "https://raw.githubusercontent.com/khairunnisaor/idner-news-2k/main/train.txt",
            "dev.txt": "https://raw.githubusercontent.com/khairunnisaor/idner-news-2k/main/dev.txt",
            "test.txt": "https://raw.githubusercontent.com/khairunnisaor/idner-news-2k/main/test.txt",
        },
    },
}

RIZALESPE_BASE_URL = (
    "https://raw.githubusercontent.com/rizalespe/"
    "Dataset-Sentimen-Analisis-Bahasa-Indonesia/master/"
)
RIZALESPE_FILES = [
    "dataset_tweet_sentiment_cellular_service_provider.csv",
    "dataset_tweet_sentiment_pilkada_DKI_2017.csv",
]


def configure_utf8_io() -> None:
    """Ensure UTF-8 console output on Windows."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def download_file(url: str, dest: Path, force: bool = False) -> bool:
    """Download a URL to dest if missing or force=True. Returns True if downloaded."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return False

    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return True


def restore_sp4n_lapor(force: bool = False) -> Dict[str, Any]:
    """Restore SP4N LAPOR.csv from git history commit f4c01c94 into data/raw/sp4n/."""
    dest = RAW_DIR / "sp4n" / "sp4n_lapor.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)

    if not dest.exists() or dest.stat().st_size == 0 or force:
        proc = subprocess.run(
            ["git", "show", SP4N_GIT_REF],
            cwd=ROOT_DIR,
            capture_output=True,
            check=True,
        )
        if not proc.stdout:
            raise RuntimeError(f"Empty output from git show {SP4N_GIT_REF}")
        dest.write_bytes(proc.stdout)
        action = "restored_from_git"
    else:
        action = "cached"

    df = pd.read_csv(dest)
    score_dist = {str(k): int(v) for k, v in df["score"].value_counts(dropna=False).sort_index().items()}
    print(f"[sp4n_lapor] ({action}) path={dest.relative_to(ROOT_DIR)} rows={len(df):,} cols={list(df.columns)}")
    print(f"  score distribution: {score_dist}")

    return {
        "source": "sp4n_lapor",
        "origin": f"git:{SP4N_GIT_REF}",
        "path": str(dest.relative_to(ROOT_DIR)).replace("\\", "/"),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "score_distribution": score_dist,
    }


def inspect_hf_dataframe(df: pd.DataFrame, max_unique: int = 15) -> Dict[str, Dict[str, int]]:
    """Summarize low-cardinality columns in a DataFrame."""
    summary: Dict[str, Dict[str, int]] = {}
    for col in df.columns:
        nunique = df[col].nunique(dropna=False)
        if nunique <= max_unique:
            vc = df[col].value_counts(dropna=False)
            summary[col] = {("null" if pd.isna(k) else str(k)): int(v) for k, v in vc.items()}
    return summary


def download_hf_standard(repo_id: str, slug: str, force: bool = False) -> Dict[str, Any]:
    """Download a Hugging Face dataset via datasets.load_dataset and save splits as Parquet."""
    out_dir = RAW_DIR / "hf" / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_parquets = sorted(out_dir.glob("*.parquet"))
    split_frames: Dict[str, pd.DataFrame] = {}

    if existing_parquets and not force:
        for pq in existing_parquets:
            split_frames[pq.stem] = pd.read_parquet(pq)
        action = "cached"
    else:
        ds_dict = load_dataset(repo_id)
        for split_name, ds_split in ds_dict.items():
            out_path = out_dir / f"{split_name}.parquet"
            ds_split.to_parquet(str(out_path))
            split_frames[split_name] = pd.read_parquet(out_path)
        action = "downloaded"

    total_rows = sum(len(df) for df in split_frames.values())
    first_df = next(iter(split_frames.values()))
    col_summary = inspect_hf_dataframe(pd.concat(split_frames.values(), ignore_index=True))

    print(
        f"[{slug}] ({action}) hf={repo_id} splits={ {k: len(v) for k, v in split_frames.items()} } "
        f"total={total_rows:,} cols={list(first_df.columns)}"
    )
    if col_summary:
        print(f"  label summary: {col_summary}")

    return {
        "source": slug,
        "origin": f"hf:{repo_id}",
        "dir": str(out_dir.relative_to(ROOT_DIR)).replace("\\", "/"),
        "splits": {k: int(len(v)) for k, v in split_frames.items()},
        "total_rows": int(total_rows),
        "columns": list(first_df.columns),
        "label_summary": col_summary,
    }


def download_ecommerce_sentiment(force: bool = False) -> Dict[str, Any]:
    """
    Download AIbnuHibban/e-commerce-sentiment-bahasa-indonesia via datasets,
    ensuring the `category` column from challange.json (including sarcasm) is preserved.
    """
    repo_id = "AIbnuHibban/e-commerce-sentiment-bahasa-indonesia"
    slug = "ecommerce_sentiment"
    out_dir = RAW_DIR / "hf" / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train.parquet"

    if out_path.exists() and not force:
        df = pd.read_parquet(out_path)
        action = "cached"
    else:
        ds_dict = load_dataset(repo_id)
        ds_train: Dataset = ds_dict["train"]
        has_category = "category" in ds_train.column_names and any(
            x is not None for x in ds_train["category"]
        )
        if not has_category:
            base = f"https://huggingface.co/datasets/{repo_id}/resolve/main"
            ds_simple = load_dataset("json", data_files=f"{base}/simple.json", split="train")
            ds_challange = load_dataset("json", data_files=f"{base}/challange.json", split="train")
            ds_simple = ds_simple.add_column("category", [None] * len(ds_simple))
            ds_simple = ds_simple.add_column("subset", ["simple"] * len(ds_simple))
            ds_challange = ds_challange.add_column("subset", ["challange"] * len(ds_challange))
            ds_merged = concatenate_datasets([ds_simple, ds_challange])
        else:
            subsets = ["challange" if c is not None else "simple" for c in ds_train["category"]]
            ds_merged = ds_train.add_column("subset", subsets)

        ds_merged.to_parquet(str(out_path))
        df = pd.read_parquet(out_path)
        action = "downloaded"

    col_summary = inspect_hf_dataframe(df)
    print(
        f"[{slug}] ({action}) hf={repo_id} rows={len(df):,} cols={list(df.columns)}"
    )
    print(f"  label summary: {col_summary}")

    return {
        "source": slug,
        "origin": f"hf:{repo_id}",
        "dir": str(out_dir.relative_to(ROOT_DIR)).replace("\\", "/"),
        "splits": {"train": int(len(df))},
        "total_rows": int(len(df)),
        "columns": list(df.columns),
        "label_summary": col_summary,
    }


def inspect_conll_file(path: Path) -> Dict[str, Any]:
    """Count sentences and BIO tags in a CoNLL/TSV NER file."""
    tags: Counter[str] = Counter()
    sentences = 0
    tokens = 0
    in_sentence = False

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                if in_sentence:
                    sentences += 1
                    in_sentence = False
                continue
            parts = stripped.split("\t") if "\t" in stripped else stripped.split()
            if len(parts) < 2:
                continue
            in_sentence = True
            tokens += 1
            tags[parts[-1]] += 1

    if in_sentence:
        sentences += 1

    return {"sentences": sentences, "tokens": tokens, "tags": dict(tags)}


def download_ner_sources(force: bool = False) -> List[Dict[str, Any]]:
    """Download the 3 SEACrowd NER datasets from their canonical upstream CoNLL/TSV URLs."""
    manifests: List[Dict[str, Any]] = []
    for slug, cfg in NER_SOURCES.items():
        out_dir = RAW_DIR / "ner" / slug
        downloaded_any = False
        file_stats: Dict[str, Any] = {}
        combined_tags: Counter[str] = Counter()
        total_sentences = 0

        for fname, url in cfg["files"].items():
            dest = out_dir / fname
            did_dl = download_file(url, dest, force=force)
            downloaded_any = downloaded_any or did_dl
            stats = inspect_conll_file(dest)
            file_stats[fname] = stats["sentences"]
            total_sentences += stats["sentences"]
            combined_tags.update(stats["tags"])

        action = "downloaded" if downloaded_any else "cached"
        print(
            f"[{slug}] ({action}) hf={cfg['hf_repo']} files={file_stats} "
            f"total_sentences={total_sentences:,}"
        )
        print(f"  note: {cfg['note']}")
        print(f"  BIO tags: {dict(combined_tags)}")

        manifests.append(
            {
                "source": slug,
                "origin": f"hf:{cfg['hf_repo']} (upstream CoNLL/TSV)",
                "note": cfg["note"],
                "dir": str(out_dir.relative_to(ROOT_DIR)).replace("\\", "/"),
                "files": file_stats,
                "total_sentences": int(total_sentences),
                "bio_tags": dict(combined_tags),
            }
        )
    return manifests


def download_rizalespe_sources(force: bool = False) -> List[Dict[str, Any]]:
    """Download selected sentiment CSVs from rizalespe GitHub repository."""
    out_dir = RAW_DIR / "github_rizalespe"
    manifests: List[Dict[str, Any]] = []

    for fname in RIZALESPE_FILES:
        url = RIZALESPE_BASE_URL + fname
        dest = out_dir / fname
        did_dl = download_file(url, dest, force=force)
        action = "downloaded" if did_dl else "cached"
        df = pd.read_csv(dest)
        sent_col = "Sentiment"
        sent_dist = {str(k): int(v) for k, v in df[sent_col].value_counts(dropna=False).items()}

        print(
            f"[rizalespe:{fname}] ({action}) path={dest.relative_to(ROOT_DIR)} "
            f"rows={len(df):,} cols={list(df.columns)} sentiment={sent_dist}"
        )
        manifests.append(
            {
                "source": fname,
                "origin": url,
                "path": str(dest.relative_to(ROOT_DIR)).replace("\\", "/"),
                "rows": int(len(df)),
                "columns": list(df.columns),
                "sentiment_distribution": sent_dist,
            }
        )
    return manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and inspect all raw datasets for the Unified Master Table."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite cached raw files.",
    )
    return parser.parse_args()


def main() -> None:
    configure_utf8_io()
    args = parse_args()
    print("=== Downloading & Inspecting Raw Datasets ===")

    entries: List[Dict[str, Any]] = []
    entries.append(restore_sp4n_lapor(force=args.force))
    entries.append(
        download_hf_standard(
            "YogaSWD/indonesian-campus-complaints",
            "indonesian_campus_complaints",
            force=args.force,
        )
    )
    entries.append(
        download_hf_standard(
            "w11wo/twitter_indonesia_sarcastic",
            "twitter_indonesia_sarcastic",
            force=args.force,
        )
    )
    entries.append(download_ecommerce_sentiment(force=args.force))
    entries.extend(download_ner_sources(force=args.force))
    entries.extend(download_rizalespe_sources(force=args.force))

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": entries,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nManifest saved to: {MANIFEST_PATH.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()

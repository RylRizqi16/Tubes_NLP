"""Task 1 data contract. The master split and text are never modified here."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data/processed/master_dataset.jsonl"
DOMAINS = (
    "Infrastruktur & PU", "Kesehatan & Sanitasi", "Layanan & Administrasi",
    "Lingkungan", "Pungli & Tata Kelola",
)
INTENTS = ("Keluhan", "Permintaan", "Darurat", "Saran", "Apresiasi")
SPLITS = {"train", "val", "test"}


def load_records(path: str | Path = DEFAULT_DATA, split: str | None = None) -> list[dict]:
    if split is not None and split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    records = []
    seen = set()
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("domain") and not row.get("intent"):
                continue
            if row["split"] not in SPLITS or row.get("domain") not in (*DOMAINS, None) or row.get("intent") not in (*INTENTS, None):
                raise ValueError(f"Invalid T1 label or split on line {number}")
            if not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError(f"Empty T1 text on line {number}")
            if row["id"] in seen:
                raise ValueError(f"Duplicate T1 id: {row['id']}")
            seen.add(row["id"])
            if split is None or row["split"] == split:
                records.append(row)
    return records


def labeled(records: list[dict], target: str) -> list[dict]:
    if target not in {"domain", "intent"}:
        raise ValueError(target)
    return [row for row in records if row[target] is not None]

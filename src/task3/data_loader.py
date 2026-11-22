from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_MASTER_PATH = ROOT_DIR / "data" / "processed" / "master_dataset.jsonl"

SENTIMENT_TO_ID: Dict[str, int] = {"Negatif": 0, "Netral": 1, "Positif": 2}
ID_TO_SENTIMENT: Dict[int, str] = {0: "Negatif", 1: "Netral", 2: "Positif"}
SENTIMENT_TO_SCHEMA: Dict[str, str] = {
    "Negatif": "negative",
    "Netral": "neutral",
    "Positif": "positive",
}

NEGATION_FLAG_RE = re.compile(
    r"\b(?:tidak|bukan|belum|gak|ga|gk|nggak|ngak|ora|boro-boro|mana\s+ada|tak|jangan)\b",
    re.IGNORECASE,
)
CODE_MIXING_FLAG_RE = re.compile(
    r"\b(?:wis|suwe|ora|piye|iki|neng|tenan|meneh|wingi|kangelan|tulung|pirang-pirang|"
    r"dina|ono|bgt|blm|yg|gk|jan|dalan)\b",
    re.IGNORECASE,
)


def annotate_linguistic_flags(record: Dict[str, Any]) -> Dict[str, Any]:
    text = str(record.get("text") or "")
    enriched = dict(record)
    enriched["has_negation"] = bool(NEGATION_FLAG_RE.search(text))
    enriched["has_code_mixing"] = bool(CODE_MIXING_FLAG_RE.search(text))
    enriched["is_irony"] = record.get("is_sarcasm") == 1
    return enriched


def load_t3_records(
    data_path: Path | str = DEFAULT_MASTER_PATH,
    split: Optional[str] = None,
    subset: str = "all",
) -> List[Dict[str, Any]]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Master dataset not found at {path}")
    if subset not in {"all", "real", "synthetic"}:
        raise ValueError(f"Invalid subset: {subset!r}. Expected 'all', 'real', or 'synthetic'.")

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("sentiment") is None and row.get("is_sarcasm") is None:
                continue
            if split is not None and row.get("split") != split:
                continue
            is_syn = row.get("source") == "synthetic"
            if subset == "real" and is_syn:
                continue
            if subset == "synthetic" and not is_syn:
                continue
            records.append(annotate_linguistic_flags(row))
    return records


def filter_subset(records: List[Dict[str, Any]], subset: str) -> List[Dict[str, Any]]:
    if subset == "all":
        return records
    if subset == "real":
        return [r for r in records if r.get("source") != "synthetic"]
    if subset == "synthetic":
        return [r for r in records if r.get("source") == "synthetic"]
    if subset == "negation":
        return [r for r in records if r.get("has_negation")]
    if subset == "code_mixing":
        return [r for r in records if r.get("has_code_mixing")]
    if subset == "sarcasm_irony":
        return [r for r in records if r.get("is_irony")]
    raise ValueError(f"Unknown subset filter: {subset!r}")


def summarize_t3_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    sent_count = sum(1 for r in records if r.get("sentiment") is not None)
    sarc_count = sum(1 for r in records if r.get("is_sarcasm") is not None)
    neg_count = sum(1 for r in records if r.get("has_negation"))
    cm_count = sum(1 for r in records if r.get("has_code_mixing"))
    irony_count = sum(1 for r in records if r.get("is_irony"))
    return {
        "total_rows": len(records),
        "sentiment_non_null": sent_count,
        "sarcasm_non_null": sarc_count,
        "negation_rows": neg_count,
        "code_mixing_rows": cm_count,
        "irony_rows": irony_count,
    }

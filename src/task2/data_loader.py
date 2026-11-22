"""
Task 2 Data Loader & Subword BIO2 Offset Alignment Module.

Loads NER records (`entities != null`, 9,257 rows) and LDA records
(`source in {'sp4n_lapor', 'campus_complaints', 'synthetic'}`, 6,468 train rows)
from `data/processed/master_dataset.jsonl` without modifying master splits or texts.

Provides:
- Subword BIO2 7-tag encoding & decoding via `offset_mapping`
  (preserving adjacent same-label `B-X` entities separated by a single space).
- Zero-truncation verification at `max_length=384`.
- Dynamic per-batch padding collator for PyTorch training/evaluation.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.schema import EntitySpan


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_MASTER_PATH = ROOT_DIR / "data" / "processed" / "master_dataset.jsonl"

VALID_SPLITS = {"train", "val", "test"}
ENTITY_LABELS: Tuple[str, ...] = ("LOCATION", "ORGANIZATION", "PERSON")
NER_SOURCES: Tuple[str, ...] = (
    "indonlu_nergrit",
    "indolem_nerui",
    "idner_news_2k",
    "synthetic",
)
LDA_SOURCES: Tuple[str, ...] = (
    "sp4n_lapor",
    "campus_complaints",
    "synthetic",
)

BIO2_TAGS: Tuple[str, ...] = (
    "O",
    "B-LOCATION",
    "I-LOCATION",
    "B-ORGANIZATION",
    "I-ORGANIZATION",
    "B-PERSON",
    "I-PERSON",
)
TAG_TO_ID: Dict[str, int] = {tag: idx for idx, tag in enumerate(BIO2_TAGS)}
ID_TO_TAG: Dict[int, str] = {idx: tag for idx, tag in enumerate(BIO2_TAGS)}
IGNORE_INDEX: int = -100
DEFAULT_MAX_LENGTH: int = 384

# Unicode-safe word + punctuation tokenizer reproducing WordPiece-style token boundaries
# when running CPU unit tests or E1 baseline without downloading HF tokenizer weights.
WORD_PUNCT_TOKEN_RE = re.compile(r"[^\W_]+|[^\w\s]|_")


def load_ner_records(
    data_path: Path | str = DEFAULT_MASTER_PATH,
    split: Optional[str] = None,
    subset: str = "all",
    source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load Task 2 NER records (`entities is not None`) from `master_dataset.jsonl`.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Master dataset not found at {path}")
    if split is not None and split not in VALID_SPLITS:
        raise ValueError(f"Invalid split: {split!r}. Expected one of {VALID_SPLITS}.")
    if subset not in {"all", "real", "synthetic"}:
        raise ValueError(f"Invalid subset: {subset!r}. Expected 'all', 'real', or 'synthetic'.")

    records: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("entities") is None:
                continue

            row_split = row.get("split")
            if row_split not in VALID_SPLITS:
                raise ValueError(f"Invalid split {row_split!r} on line {line_no}")
            if split is not None and row_split != split:
                continue

            row_source = row.get("source")
            is_syn = row_source == "synthetic"
            if subset == "real" and is_syn:
                continue
            if subset == "synthetic" and not is_syn:
                continue
            if source is not None and row_source != source:
                continue

            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Empty NER text on line {line_no}")

            rid = row.get("id")
            if not isinstance(rid, str) or not rid or rid in seen_ids:
                raise ValueError(f"Duplicate or invalid record id {rid!r} on line {line_no}")
            seen_ids.add(rid)

            raw_ents = row.get("entities") or []
            validated_ents: List[Dict[str, Any]] = []
            for ent in sorted(raw_ents, key=lambda e: (int(e["start"]), int(e["end"]))):
                s = int(ent["start"])
                e = int(ent["end"])
                lbl = str(ent["label"])
                if lbl not in ENTITY_LABELS:
                    raise ValueError(f"Invalid entity label {lbl!r} in {rid}")
                if not (0 <= s < e <= len(text)):
                    raise ValueError(f"Out-of-bounds entity span {(s, e)} in {rid} (len={len(text)})")
                validated_ents.append({"start": s, "end": e, "label": lbl})

            for prev_ent, curr_ent in zip(validated_ents, validated_ents[1:]):
                if curr_ent["start"] < prev_ent["end"]:
                    raise ValueError(f"Overlapping gold spans in {rid}: {prev_ent} vs {curr_ent}")

            rec = dict(row)
            rec["entities"] = validated_ents
            records.append(rec)

    return records


def filter_ner_subset(records: Sequence[Dict[str, Any]], subset: str) -> List[Dict[str, Any]]:
    """Filter NER records by `'all'`, `'real'`, `'synthetic'`, or individual source name."""
    if subset == "all":
        return list(records)
    if subset == "real":
        return [r for r in records if r.get("source") != "synthetic"]
    if subset == "synthetic":
        return [r for r in records if r.get("source") == "synthetic"]
    if subset in NER_SOURCES:
        return [r for r in records if r.get("source") == subset]
    raise ValueError(f"Unknown NER subset filter: {subset!r}")


def load_lda_records(
    data_path: Path | str = DEFAULT_MASTER_PATH,
    split: Optional[str] = "train",
    sources: Sequence[str] = LDA_SOURCES,
) -> List[Dict[str, Any]]:
    """
    Load Task 2 LDA topic modeling records (`sp4n_lapor`, `campus_complaints`, `synthetic`).
    By default loads `split == 'train'` strictly to prevent topic leakage.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Master dataset not found at {path}")
    if split is not None and split not in VALID_SPLITS:
        raise ValueError(f"Invalid split: {split!r}. Expected one of {VALID_SPLITS}.")

    allowed_sources = set(sources)
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("source") not in allowed_sources:
                continue
            if split is not None and row.get("split") != split:
                continue
            records.append(row)
    return records


def summarize_ner_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute dataset profile summary for a sequence of NER records."""
    split_counts = Counter(r["split"] for r in records)
    source_counts = Counter(r["source"] for r in records)
    label_counts: Counter[str] = Counter()
    empty_rows = 0
    max_words = 0
    max_chars = 0

    for r in records:
        ents = r.get("entities") or []
        if not ents:
            empty_rows += 1
        for e in ents:
            label_counts[e["label"]] += 1
        text = r["text"]
        max_words = max(max_words, len(text.split()))
        max_chars = max(max_chars, len(text))

    return {
        "total_rows": len(records),
        "non_empty_rows": len(records) - empty_rows,
        "zero_entity_rows": empty_rows,
        "total_spans": sum(label_counts.values()),
        "label_counts": {lbl: label_counts.get(lbl, 0) for lbl in ENTITY_LABELS},
        "split_counts": dict(split_counts),
        "source_counts": dict(source_counts),
        "max_words": max_words,
        "max_chars": max_chars,
    }


def regex_tokenize_with_offsets(
    text: str,
    max_length: int = DEFAULT_MAX_LENGTH,
    add_special_tokens: bool = True,
) -> Dict[str, Any]:
    """
    Deterministic Unicode word+punctuation tokenizer with character `offset_mapping`.
    Used for offline CPU verification and E1 baseline tokenization.
    """
    tokens: List[str] = []
    offsets: List[Tuple[int, int]] = []

    if add_special_tokens:
        tokens.append("[CLS]")
        offsets.append((0, 0))

    limit = max_length - (2 if add_special_tokens else 0)
    for m in WORD_PUNCT_TOKEN_RE.finditer(text):
        if len(tokens) - (1 if add_special_tokens else 0) >= limit:
            break
        tokens.append(m.group(0))
        offsets.append((m.start(), m.end()))

    if add_special_tokens:
        tokens.append("[SEP]")
        offsets.append((0, 0))

    return {
        "tokens": tokens,
        "offset_mapping": offsets,
        "attention_mask": [1] * len(offsets),
    }


def encode_spans_to_bio2(
    offset_mapping: Sequence[Tuple[int, int]],
    entities: Sequence[Dict[str, Any]],
) -> List[int]:
    """
    Align character-level entity spans `[{"start", "end", "label"}]` to subword
    `BIO2` tag IDs using `offset_mapping`.

    Rules:
    - Special/padding tokens with `(0, 0)` (or `tok_start >= tok_end`) receive `IGNORE_INDEX (-100)`.
    - For each entity `(start, end, label)`:
      - First non-special token overlapping `[start, end]` receives `B-<LABEL>`.
      - Subsequent non-special tokens overlapping `[start, end]` receive `I-<LABEL>`.
    - All other valid tokens receive `TAG_TO_ID["O"] (0)`.
    """
    labels: List[int] = [
        IGNORE_INDEX if (s == 0 and e == 0) or s >= e else TAG_TO_ID["O"]
        for s, e in offset_mapping
    ]

    sorted_ents = sorted(entities, key=lambda x: (int(x["start"]), int(x["end"])))
    for ent in sorted_ents:
        ent_start = int(ent["start"])
        ent_end = int(ent["end"])
        ent_label = str(ent["label"])
        b_id = TAG_TO_ID[f"B-{ent_label}"]
        i_id = TAG_TO_ID[f"I-{ent_label}"]

        matched_indices: List[int] = []
        for idx, (tok_start, tok_end) in enumerate(offset_mapping):
            if labels[idx] == IGNORE_INDEX:
                continue
            if max(tok_start, ent_start) < min(tok_end, ent_end):
                matched_indices.append(idx)

        for pos, idx in enumerate(matched_indices):
            labels[idx] = b_id if pos == 0 else i_id

    return labels


def decode_bio2_to_spans(
    text: str,
    offset_mapping: Sequence[Tuple[int, int]],
    tag_ids: Sequence[int],
    tokens: Optional[Sequence[str]] = None,
    repair_subwords: bool = False,
    promote_orphan_i: bool = False,
    repair_orphans: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """
    Decode a sequence of BIO2 tag IDs and `offset_mapping` back into character spans
    `[{"start": int, "end": int, "label": str, "text": str}]`.

    Crucial guarantee:
    - Only merges `B-X` followed by `I-X*` into `(first_tok_start, last_tok_end)`.
    - NEVER merges two consecutive `B-X` tokens across a space, preserving the
      59 adjacent same-label entity pairs in `indolem_nerui` and `idner_news_2k`.

    Parameters
    ----------
    text : str
        Original input string.
    offset_mapping : Sequence[Tuple[int, int]]
        Token `(start, end)` character offsets.
    tag_ids : Sequence[int]
        Predicted BIO2 tag IDs.
    tokens : Optional[Sequence[str]]
        Optional subword strings (e.g. from `tokenizer.convert_ids_to_tokens`) to
        detect WordPiece continuation prefixes (`##...`).
    repair_subwords : bool
        When `True` (enabled for E3 / E4b post-processing), repairs mid-word
        WordPiece continuation splits (`##...` or contiguous alphanumeric subwords
        `tok_start == active_end`) by extending the active entity across the rest
        of the word. Always safe on `val (real)`.
    promote_orphan_i : bool
        When `True` (opt-in only, default `False`), promotes standalone orphan
        `O -> I-X` or `B-Y -> I-X` ($Y \neq X$) tokens to `B-X`. Disabled by
        default because non-CRF classifiers emit noisy standalone `I-X` tags on
        non-entity capitalized words.
    repair_orphans : Optional[bool]
        Backward-compatible alias for `repair_subwords` if passed by callers.
    """
    if repair_orphans is not None:
        repair_subwords = bool(repair_orphans)

    spans: List[Dict[str, Any]] = []
    active_label: Optional[str] = None
    active_start: int = -1
    active_end: int = -1

    def flush_active() -> None:
        nonlocal active_label, active_start, active_end
        if active_label is None or active_start < 0 or active_end <= active_start:
            active_label = None
            active_start = -1
            active_end = -1
            return

        s, e = active_start, active_end
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1

        if 0 <= s < e <= len(text):
            spans.append(
                {
                    "start": s,
                    "end": e,
                    "label": active_label,
                    "text": text[s:e],
                }
            )
        active_label = None
        active_start = -1
        active_end = -1

    n = min(len(offset_mapping), len(tag_ids))
    for idx in range(n):
        tok_start, tok_end = offset_mapping[idx]
        tag_id = int(tag_ids[idx])

        # Skip special tokens ([CLS], [SEP], [PAD])
        if tag_id == IGNORE_INDEX or (tok_start == 0 and tok_end == 0) or tok_start >= tok_end:
            continue

        tag_str = ID_TO_TAG.get(tag_id, "O")
        tok_str = tokens[idx] if tokens is not None and idx < len(tokens) else ""
        is_wordpiece_cont = tok_str.startswith("##") or (
            active_label is not None
            and tok_start == active_end
            and tok_start > 0
            and text[tok_start - 1].isalnum()
            and text[tok_start].isalnum()
        )

        if repair_subwords and is_wordpiece_cont and active_label is not None:
            active_end = max(active_end, tok_end)
            continue

        if tag_str == "O":
            flush_active()
            continue

        prefix, ent_type = tag_str.split("-", 1)
        if prefix == "B":
            flush_active()
            active_label = ent_type
            active_start = tok_start
            active_end = tok_end
        elif prefix == "I":
            if active_label == ent_type:
                active_end = max(active_end, tok_end)
            else:
                flush_active()
                if promote_orphan_i:
                    active_label = ent_type
                    active_start = tok_start
                    active_end = tok_end

    flush_active()
    return spans


def verify_dataset_span_alignment(
    records: Sequence[Dict[str, Any]],
    tokenize_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> Dict[str, Any]:
    """
    Verify round-trip `char spans -> BIO2 subword tags -> char spans` across all
    NER records at `max_length=384`.
    """
    if tokenize_fn is None:
        def _default_tok(txt: str) -> Dict[str, Any]:
            return regex_tokenize_with_offsets(txt, max_length=max_length, add_special_tokens=True)
        tokenize_fn = _default_tok

    total_gold_spans = 0
    exact_recovered_spans = 0
    truncated_spans = 0
    mismatches: List[Dict[str, Any]] = []

    for rec in records:
        text = rec["text"]
        gold_ents = [
            {"start": int(e["start"]), "end": int(e["end"]), "label": str(e["label"])}
            for e in (rec.get("entities") or [])
        ]
        total_gold_spans += len(gold_ents)

        enc = tokenize_fn(text)
        offsets = [tuple(pair) for pair in enc["offset_mapping"]]
        valid_ends = [e for s, e in offsets if s < e]
        max_covered_char = max(valid_ends) if valid_ends else 0

        for ge in gold_ents:
            if ge["end"] > max_covered_char:
                truncated_spans += 1

        bio_ids = encode_spans_to_bio2(offsets, gold_ents)
        decoded = decode_bio2_to_spans(
            text,
            offsets,
            bio_ids,
            repair_subwords=False,
            promote_orphan_i=False,
        )
        decoded_triples = [
            {"start": d["start"], "end": d["end"], "label": d["label"]}
            for d in decoded
        ]

        if decoded_triples == gold_ents:
            exact_recovered_spans += len(gold_ents)
        else:
            gold_set = {(e["start"], e["end"], e["label"]) for e in gold_ents}
            dec_set = {(e["start"], e["end"], e["label"]) for e in decoded_triples}
            exact_recovered_spans += len(gold_set & dec_set)
            if len(mismatches) < 20:
                mismatches.append(
                    {
                        "id": rec["id"],
                        "source": rec["source"],
                        "split": rec["split"],
                        "gold": gold_ents,
                        "decoded": decoded_triples,
                    }
                )

    loss_rate = (
        (total_gold_spans - exact_recovered_spans) / total_gold_spans
        if total_gold_spans > 0
        else 0.0
    )
    return {
        "total_records": len(records),
        "total_gold_spans": total_gold_spans,
        "exact_recovered_spans": exact_recovered_spans,
        "truncated_spans": truncated_spans,
        "alignment_loss_rate": round(loss_rate, 6),
        "mismatches": mismatches,
    }


def spans_to_entity_objects(text: str, spans: Sequence[Dict[str, Any]]) -> List[EntitySpan]:
    """
    Convert internal span dicts `[{"start", "end", "label"}]` into validated
    `EntitySpan(text=text[start:end], label=..., start=..., end=...)` objects
    satisfying `src/schema.py` and non-overlapping invariants.
    """
    ordered = sorted(spans, key=lambda s: (int(s["start"]), int(s["end"])))
    result: List[EntitySpan] = []
    last_end = -1

    for sp in ordered:
        start = int(sp["start"])
        end = int(sp["end"])
        label = str(sp["label"])
        if label not in ENTITY_LABELS:
            raise ValueError(f"Invalid EntitySpan label: {label!r}")
        if not (0 <= start < end <= len(text)):
            raise ValueError(f"Out-of-bounds EntitySpan {(start, end)} for text length {len(text)}")
        if start < last_end:
            continue
        slice_text = text[start:end]
        if not slice_text:
            continue
        result.append(EntitySpan(text=slice_text, label=label, start=start, end=end))
        last_end = end

    return result


class NERTorchDataset:
    """
    Tokenized NER dataset for PyTorch / HuggingFace `AutoTokenizer` (fast tokenizer).
    Stores variable-length sequences up to `max_length=384` for dynamic per-batch padding.
    """

    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        tokenizer: Any,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.items: List[Dict[str, Any]] = []

        for rec in self.records:
            text = rec["text"]
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_offsets_mapping=True,
            )
            offsets = [tuple(x) for x in enc["offset_mapping"]]
            labels = encode_spans_to_bio2(offsets, rec.get("entities") or [])
            tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])
            self.items.append(
                {
                    "id": rec["id"],
                    "source": rec["source"],
                    "split": rec["split"],
                    "text": text,
                    "gold_entities": rec.get("entities") or [],
                    "input_ids": list(enc["input_ids"]),
                    "attention_mask": list(enc["attention_mask"]),
                    "labels": labels,
                    "offset_mapping": offsets,
                    "tokens": tokens,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


def dynamic_pad_collate(batch: Sequence[Dict[str, Any]], pad_token_id: int = 0) -> Dict[str, Any]:
    """
    Dynamic per-batch padding collator for `NERTorchDataset`.
    Pads `input_ids`, `attention_mask`, and `labels` only to the longest sequence
    in the current batch (up to `max_length=384`).
    """
    import torch

    max_len = max(len(item["input_ids"]) for item in batch)
    batch_input_ids: List[List[int]] = []
    batch_attention_mask: List[List[int]] = []
    batch_labels: List[List[int]] = []

    for item in batch:
        cur_len = len(item["input_ids"])
        pad_len = max_len - cur_len
        batch_input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
        batch_attention_mask.append(item["attention_mask"] + [0] * pad_len)
        batch_labels.append(item["labels"] + [IGNORE_INDEX] * pad_len)

    return {
        "ids": [item["id"] for item in batch],
        "sources": [item["source"] for item in batch],
        "texts": [item["text"] for item in batch],
        "gold_entities": [item["gold_entities"] for item in batch],
        "offset_mappings": [item["offset_mapping"] for item in batch],
        "tokens": [item["tokens"] for item in batch],
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
    }

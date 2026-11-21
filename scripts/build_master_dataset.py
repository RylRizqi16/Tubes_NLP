"""
Build the Unified Master Table dataset for Indonesian Public Complaint NLP.

Reads raw datasets prepared by `scripts/download_datasets.py`, cleans text while
preserving negations and intensifiers, converts NER BIO/IOB2 annotations into
sentence texts with verified character offsets, applies source-level stratified
sampling and anti-leakage deduplication, performs per-source 70/15/15 stratified
splits (seed=42), validates all invariants, and writes processed outputs to
`data/processed/`.

Outputs:
- `data/processed/master_dataset.jsonl`
- `data/processed/master_dataset.parquet`
- `data/processed/master_dataset_preview.csv`
- `data/processed/dataset_stats.json`

Run:
    python scripts/build_master_dataset.py
"""

from __future__ import annotations

import html
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 42
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

SCHEMA_KEYS = [
    "id",
    "source",
    "split",
    "text",
    "domain",
    "intent",
    "entities",
    "sentiment",
    "is_sarcasm",
    "urgency_level",
]

VALID_SPLITS = {"train", "val", "test"}
VALID_DOMAINS = {
    "Infrastruktur & PU",
    "Kesehatan & Sanitasi",
    "Layanan & Administrasi",
    "Lingkungan",
    "Pungli & Tata Kelola",
}
VALID_INTENTS = {"Keluhan", "Permintaan", "Darurat", "Saran", "Apresiasi"}
VALID_ENTITY_LABELS = {"LOCATION", "ORGANIZATION", "PERSON"}
VALID_SENTIMENTS = {"Positif", "Netral", "Negatif"}
VALID_URGENCY_LEVELS = {"Low", "Med", "High", "Emergency"}

CAMPUS_MAX_ROWS = 5000
ECOM_MAX_ROWS = 4000
ECOM_PER_CLASS_TARGET = ECOM_MAX_ROWS // 3

HTML_TAG_RE = re.compile(
    r"</?(?:p|br|div|span|a|b|i|u|strong|em|img|ul|ol|li|h[1-6]|table|tr|td|th|"
    r"thead|tbody|script|style|iframe|blockquote|pre|code|hr|font|center|small|"
    r"sub|sup|mark|section|article|header|footer|nav)\b[^>]*>",
    re.IGNORECASE,
)
URL_RE = re.compile(r"(?:https?://\S+|\bwww\.\S+)", re.IGNORECASE)
URL_PLACEHOLDER_RE = re.compile(r"<(?:link|url)>", re.IGNORECASE)
ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff\xad]")
CTRL_RE = re.compile(r"[\x00-\x1f\x7f]+")
WS_RE = re.compile(r"\s+")

ENTITY_LABEL_MAP = {
    "PLACE": "LOCATION",
    "LOC": "LOCATION",
    "LOCATION": "LOCATION",
    "ORGANISATION": "ORGANIZATION",
    "ORGANIZATION": "ORGANIZATION",
    "ORG": "ORGANIZATION",
    "PERSON": "PERSON",
    "PER": "PERSON",
}

CLOSING_PUNCT = {".", ",", ":", ";", "!", "?", ")", "]", "}", "%", "..."}
OPENING_PUNCT = {"(", "[", "{", "$"}
SIMPLE_HTML_TAG_NAMES = {
    "strong",
    "/strong",
    "b",
    "/b",
    "i",
    "/i",
    "em",
    "/em",
    "p",
    "/p",
    "br",
    "br/",
    "div",
    "/div",
    "span",
    "/span",
    "a",
    "/a",
}
ENAMEX_ATTR_RE = re.compile(r'^TYPE="[^"]+">(.+)$', re.IGNORECASE)


def configure_utf8_io() -> None:
    """Ensure UTF-8 console output on Windows."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def clean_text(text: Any) -> str:
    """
    Clean raw text while preserving casing, negation words, intensifiers,
    and semantic dataset placeholders (<PROVIDER_NAME>, <username>, etc.).
    """
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    s = str(text)
    s = html.unescape(s)
    s = ZERO_WIDTH_RE.sub("", s)
    s = HTML_TAG_RE.sub(" ", s)
    s = URL_PLACEHOLDER_RE.sub(" ", s)
    s = URL_RE.sub(" ", s)
    s = CTRL_RE.sub(" ", s)
    s = WS_RE.sub(" ", s).strip()
    return s


def normalize_for_dedup(text: str) -> str:
    """Return canonical lowercase whitespace-collapsed key for deduplication."""
    return WS_RE.sub(" ", text.strip().lower())


def make_record(
    *,
    source: str,
    text: str,
    domain: Optional[str] = None,
    intent: Optional[str] = None,
    entities: Optional[List[Dict[str, Any]]] = None,
    sentiment: Optional[str] = None,
    is_sarcasm: Optional[int] = None,
    urgency_level: Optional[str] = None,
    split: str = "train",
    record_id: str = "",
) -> Dict[str, Any]:
    """Construct a schema-compliant record dictionary."""
    return {
        "id": record_id,
        "source": source,
        "split": split,
        "text": text,
        "domain": domain,
        "intent": intent,
        "entities": entities,
        "sentiment": sentiment,
        "is_sarcasm": int(is_sarcasm) if is_sarcasm is not None else None,
        "urgency_level": urgency_level,
    }


def resolve_within_source_duplicates(
    records: List[Dict[str, Any]],
    label_key_fn: Optional[Callable[[Dict[str, Any]], Any]] = None,
    drop_all_collisions: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Deduplicate normalized text within a source.
    - If `label_key_fn` is None: keep first occurrence.
    - If `drop_all_collisions` is True: drop any text with >1 distinct label.
    - Else: resolve collisions by strict majority vote on label; drop ties.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[normalize_for_dedup(rec["text"])].append(rec)

    kept: List[Dict[str, Any]] = []
    exact_dups_removed = 0
    collisions_majority_resolved = 0
    collisions_dropped = 0

    for group in grouped.values():
        if len(group) == 1:
            kept.append(group[0])
            continue

        if label_key_fn is None:
            kept.append(group[0])
            exact_dups_removed += len(group) - 1
            continue

        label_counts = Counter(label_key_fn(r) for r in group)
        if len(label_counts) == 1:
            kept.append(group[0])
            exact_dups_removed += len(group) - 1
            continue

        if drop_all_collisions:
            collisions_dropped += len(group)
            continue

        ranked = label_counts.most_common()
        top_label, top_count = ranked[0]
        second_count = ranked[1][1]
        if top_count > second_count:
            winner = next(r for r in group if label_key_fn(r) == top_label)
            kept.append(winner)
            collisions_majority_resolved += 1
            exact_dups_removed += len(group) - 1
        else:
            collisions_dropped += len(group)

    stats = {
        "input_rows": len(records),
        "kept_rows": len(kept),
        "exact_dups_removed": exact_dups_removed,
        "collisions_majority_resolved": collisions_majority_resolved,
        "collisions_dropped": collisions_dropped,
    }
    return kept, stats


def filter_seen_cross_source(
    records: List[Dict[str, Any]],
    seen_norms: set[str],
) -> Tuple[List[Dict[str, Any]], int]:
    """Exclude any record whose normalized text already exists in an earlier source."""
    filtered: List[Dict[str, Any]] = []
    dropped = 0
    for rec in records:
        norm = normalize_for_dedup(rec["text"])
        if norm in seen_norms:
            dropped += 1
            continue
        filtered.append(rec)
    return filtered, dropped


def register_seen_norms(records: Sequence[Dict[str, Any]], seen_norms: set[str]) -> None:
    """Register normalized texts of selected records into the global seen set."""
    for rec in records:
        seen_norms.add(normalize_for_dedup(rec["text"]))


def normalize_bio_tag(raw_tag: str) -> str:
    """Normalize BIO/IOB2 tag label to LOCATION, ORGANIZATION, PERSON, or O."""
    tag = raw_tag.strip()
    if tag == "O":
        return "O"
    if "-" not in tag:
        raise ValueError(f"Unexpected NER BIO tag format: {raw_tag!r}")
    prefix, raw_type = tag.split("-", 1)
    prefix = prefix.upper()
    if prefix not in {"B", "I"}:
        raise ValueError(f"Unexpected NER BIO prefix: {raw_tag!r}")
    mapped_type = ENTITY_LABEL_MAP.get(raw_type.upper())
    if mapped_type is None:
        raise ValueError(f"Unexpected NER entity type: {raw_tag!r}")
    return f"{prefix}-{mapped_type}"


def strip_tokenized_html_fragments(
    token_tags: List[Tuple[str, str]],
) -> Tuple[List[Tuple[str, str]], int]:
    """
    Remove tokenized HTML/XML wrapper fragments (such as `< iframe ... >`,
    `< strong > ... < /strong >`, and `< ENAMEX TYPE="LOCATION">Indonesia`)
    while preserving inner content tokens and their BIO tags.
    """
    cleaned: List[Tuple[str, str]] = []
    n = len(token_tags)
    i = 0
    fragments_removed = 0

    while i < n:
        tok, tag = token_tags[i]

        if (
            tok == "<"
            and tag == "O"
            and i + 1 < n
            and token_tags[i + 1][0].lower() == "iframe"
            and token_tags[i + 1][1] == "O"
        ):
            j = i + 2
            consumed = False
            while j < n and token_tags[j][1] == "O":
                if "</iframe" in token_tags[j][0].lower() or token_tags[j][0].endswith("/>"):
                    j += 1
                    if j < n and token_tags[j][0] == ">" and token_tags[j][1] == "O":
                        j += 1
                    consumed = True
                    break
                j += 1
            if consumed:
                fragments_removed += 1
                i = j
                continue

        if (
            tok == "<"
            and tag == "O"
            and i + 2 < n
            and token_tags[i + 1] == ("ENAMEX", "O")
            and token_tags[i + 2][1] == "O"
        ):
            m = ENAMEX_ATTR_RE.match(token_tags[i + 2][0])
            if m:
                cleaned.append((m.group(1), "O"))
                fragments_removed += 1
                i += 3
                continue

        if (
            tok == "<"
            and tag == "O"
            and i + 2 < n
            and token_tags[i + 1][1] == "O"
            and token_tags[i + 1][0].lower() in SIMPLE_HTML_TAG_NAMES
            and token_tags[i + 2] == (">", "O")
        ):
            fragments_removed += 1
            i += 3
            continue

        cleaned.append((tok, tag))
        i += 1

    return cleaned, fragments_removed


def detokenize_and_extract_entities(
    raw_token_tags: List[Tuple[str, str]],
) -> Tuple[str, List[Dict[str, Any]], int]:
    """
    Convert a CoNLL/TSV token+BIO sequence into a cleaned sentence string and
    verified character-offset entities `[{"start", "end", "label"}]`.
    """
    sanitized_pairs, fragments_removed = strip_tokenized_html_fragments(raw_token_tags)

    surviving: List[Tuple[str, str]] = []
    promoted_entity: Optional[str] = None
    prev_surviving_tag = "O"

    for raw_tok, raw_tag in sanitized_pairs:
        norm_tag = normalize_bio_tag(raw_tag)
        tok = clean_text(raw_tok)
        if not tok:
            if norm_tag.startswith("B-"):
                promoted_entity = norm_tag[2:]
            elif norm_tag == "O":
                promoted_entity = None
            continue

        if promoted_entity is not None:
            if norm_tag == f"I-{promoted_entity}":
                norm_tag = f"B-{promoted_entity}"
            promoted_entity = None

        if norm_tag.startswith("I-"):
            ent_type = norm_tag[2:]
            if prev_surviving_tag not in {f"B-{ent_type}", f"I-{ent_type}"}:
                norm_tag = f"B-{ent_type}"

        surviving.append((tok, norm_tag))
        prev_surviving_tag = norm_tag

    if not surviving:
        return "", [], fragments_removed

    tokens_only = [t for t, _ in surviving]
    standalone_sq_indices = [
        idx
        for idx, tok in enumerate(tokens_only)
        if tok == "'" and not (idx + 1 < len(tokens_only) and tokens_only[idx + 1].lower() == "s")
    ]
    pair_single_quotes = len(standalone_sq_indices) > 0 and len(standalone_sq_indices) % 2 == 0
    standalone_sq_set = set(standalone_sq_indices)

    text_parts: List[str] = []
    cur_len = 0
    prev_no_space_after = True
    double_quote_open = False
    single_quote_open = False

    entities: List[Dict[str, Any]] = []
    active_entity: Optional[Dict[str, Any]] = None

    def close_active_entity(end_pos: int) -> None:
        nonlocal active_entity
        if active_entity is None:
            return
        expected_surface = "".join(active_entity["surface_parts"])
        entities.append(
            {
                "start": active_entity["start"],
                "end": end_pos,
                "label": active_entity["label"],
                "_expected": expected_surface,
            }
        )
        active_entity = None

    for idx, (tok, tag) in enumerate(surviving):
        cur_no_space_before = False
        cur_no_space_after = False

        if tok == '"':
            if not double_quote_open:
                double_quote_open = True
                cur_no_space_after = True
            else:
                double_quote_open = False
                cur_no_space_before = True
        elif tok == "'" and idx + 1 < len(tokens_only) and tokens_only[idx + 1].lower() == "s":
            cur_no_space_before = True
            cur_no_space_after = True
        elif idx in standalone_sq_set and pair_single_quotes:
            if not single_quote_open:
                single_quote_open = True
                cur_no_space_after = True
            else:
                single_quote_open = False
                cur_no_space_before = True
        elif tok in CLOSING_PUNCT:
            cur_no_space_before = True
        elif tok in OPENING_PUNCT:
            cur_no_space_after = True

        need_space = (idx > 0) and (not prev_no_space_after) and (not cur_no_space_before)

        if tag == "O" or tag.startswith("B-"):
            close_active_entity(cur_len)

        if need_space:
            text_parts.append(" ")
            cur_len += 1
            if active_entity is not None and tag.startswith("I-"):
                active_entity["surface_parts"].append(" ")

        if tag.startswith("B-"):
            active_entity = {
                "start": cur_len,
                "label": tag[2:],
                "surface_parts": [tok],
            }
        elif tag.startswith("I-"):
            assert active_entity is not None and active_entity["label"] == tag[2:]
            active_entity["surface_parts"].append(tok)

        text_parts.append(tok)
        cur_len += len(tok)
        prev_no_space_after = cur_no_space_after

    close_active_entity(cur_len)
    sentence = "".join(text_parts)

    verified_entities: List[Dict[str, Any]] = []
    for ent in entities:
        start = ent["start"]
        end = ent["end"]
        expected = ent["_expected"]
        actual = sentence[start:end]
        if actual != expected or actual != actual.strip() or not actual:
            raise AssertionError(
                f"NER offset mismatch: actual={actual!r}, expected={expected!r}, sentence={sentence!r}"
            )
        verified_entities.append({"start": start, "end": end, "label": ent["label"]})

    return sentence, verified_entities, fragments_removed


def parse_conll_file(path: Path) -> Tuple[List[Tuple[str, List[Dict[str, Any]]]], int]:
    """Parse one CoNLL/TSV NER file into (sentence, entities) tuples."""
    sentences: List[Tuple[str, List[Dict[str, Any]]]] = []
    current_tokens: List[Tuple[str, str]] = []
    total_fragments_removed = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                if current_tokens:
                    text, ents, frags = detokenize_and_extract_entities(current_tokens)
                    total_fragments_removed += frags
                    if text:
                        sentences.append((text, ents))
                    current_tokens = []
                continue
            parts = stripped.split("\t") if "\t" in stripped else stripped.split()
            if len(parts) < 2:
                continue
            current_tokens.append((parts[0], parts[-1]))

    if current_tokens:
        text, ents, frags = detokenize_and_extract_entities(current_tokens)
        total_fragments_removed += frags
        if text:
            sentences.append((text, ents))

    return sentences, total_fragments_removed


def safe_stratified_sample(
    df: pd.DataFrame,
    n: int,
    strat_cols: Sequence[str],
    seed: int = SEED,
) -> pd.DataFrame:
    """Draw a deterministic stratified sample of size n from df."""
    if len(df) <= n:
        return df.copy()

    keys = df[list(strat_cols)].fillna("NULL").astype(str).agg("|".join, axis=1)
    counts = keys.value_counts()
    rare_keys = set(counts[counts < 2].index)
    if rare_keys:
        keys = keys.where(~keys.isin(rare_keys), "__RARE__")
        if (keys == "__RARE__").sum() < 2:
            top_key = keys.value_counts().index[0]
            keys = keys.replace("__RARE__", top_key)

    use_stratify = keys if keys.nunique() >= 2 and keys.nunique() <= n else None
    sampled, _ = train_test_split(
        df,
        train_size=n,
        random_state=seed,
        stratify=use_stratify,
    )
    return sampled


def _prepare_strat_keys(raw_keys: Sequence[str], min_count: int) -> Optional[List[str]]:
    """Merge rare strata (< min_count) so sklearn train_test_split succeeds."""
    counts = Counter(raw_keys)
    keys = [k if counts[k] >= min_count else "__RARE__" for k in raw_keys]
    counts2 = Counter(keys)
    if "__RARE__" in counts2 and counts2["__RARE__"] < min_count:
        non_rare = [(k, c) for k, c in counts2.items() if k != "__RARE__"]
        if not non_rare:
            return None
        most_common_key = max(non_rare, key=lambda x: x[1])[0]
        keys = [most_common_key if k == "__RARE__" else k for k in keys]
    if len(set(keys)) < 2:
        return None
    return keys


def stratified_split_70_15_15(
    records: List[Dict[str, Any]],
    strat_key_fn: Callable[[Dict[str, Any]], str],
    seed: int = SEED,
) -> List[Dict[str, Any]]:
    """Assign `split` ('train' 70%, 'val' 15%, 'test' 15%) per source."""
    n = len(records)
    if n < 3:
        raise ValueError(f"Source too small to split 70/15/15: {n}")

    indices = list(range(n))
    raw_keys = [strat_key_fn(r) for r in records]
    keys_stage1 = _prepare_strat_keys(raw_keys, min_count=3)

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=seed,
        stratify=keys_stage1,
    )

    temp_raw_keys = [raw_keys[i] for i in temp_idx]
    keys_stage2 = _prepare_strat_keys(temp_raw_keys, min_count=2)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=seed,
        stratify=keys_stage2,
    )

    split_map: Dict[int, str] = {}
    for i in train_idx:
        split_map[i] = "train"
    for i in val_idx:
        split_map[i] = "val"
    for i in test_idx:
        split_map[i] = "test"

    out: List[Dict[str, Any]] = []
    for i, rec in enumerate(records):
        rec_copy = dict(rec)
        rec_copy["split"] = split_map[i]
        out.append(rec_copy)
    return out


def load_sp4n_lapor(seen_norms: set[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load and process SP4N LAPOR local CSV."""
    source = "sp4n_lapor"
    df = pd.read_csv(RAW_DIR / "sp4n" / "sp4n_lapor.csv")
    raw_rows = len(df)

    def map_score(score: Any) -> str:
        s = int(score)
        if s in (1, 2):
            return "Negatif"
        if s == 3:
            return "Netral"
        if s in (4, 5):
            return "Positif"
        raise ValueError(f"Unexpected SP4N score: {score!r}")

    candidates: List[Dict[str, Any]] = []
    empty_dropped = 0
    for _, row in df.iterrows():
        text = clean_text(row["content"])
        if not text:
            empty_dropped += 1
            continue
        candidates.append(
            make_record(
                source=source,
                text=text,
                sentiment=map_score(row["score"]),
            )
        )

    deduped, dedup_stats = resolve_within_source_duplicates(
        candidates, label_key_fn=lambda r: r["sentiment"], drop_all_collisions=False
    )
    filtered, cross_dropped = filter_seen_cross_source(deduped, seen_norms)
    split_records = stratified_split_70_15_15(filtered, lambda r: str(r["sentiment"]))
    register_seen_norms(split_records, seen_norms)

    log = {
        "raw_rows": raw_rows,
        "empty_dropped": empty_dropped,
        **dedup_stats,
        "cross_source_dropped": cross_dropped,
        "final_rows": len(split_records),
    }
    return split_records, log


def load_campus_complaints(seen_norms: set[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load, map, deduplicate, and stratified-sample max 5,000 campus complaint rows."""
    source = "campus_complaints"
    parquets = sorted((RAW_DIR / "hf" / "indonesian_campus_complaints").glob("*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    raw_rows = len(df)

    urgency_map = {
        "Rendah": "Low",
        "Sedang": "Med",
        "Tinggi": "High",
        "Darurat": "Emergency",
        "Kritis": "Emergency",
    }
    layanan_biros = {"ULT", "BAP", "BAK", "BTI", "HUMAS", "PERPUS"}
    unmapped_biros = {"PPKS", "BIMA", "BPM", "BKP"}

    def map_domain(biro: str, topic: str) -> Optional[str]:
        if biro == "BSP" or topic == "FAS":
            return "Infrastruktur & PU"
        if biro in layanan_biros:
            return "Layanan & Administrasi"
        if biro in unmapped_biros:
            return None
        raise ValueError(f"Unexpected target_biro={biro!r}, topic_category={topic!r}")

    df = df.copy()
    df["clean_text"] = df["text_masked"].map(clean_text)
    empty_dropped = int((df["clean_text"] == "").sum())
    df = df[df["clean_text"] != ""].copy()

    df["norm_text"] = df["clean_text"].map(normalize_for_dedup)
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["norm_text"], keep="first").copy()
    within_dups_removed = before_dedup - len(df)

    before_cross = len(df)
    df = df[~df["norm_text"].isin(seen_norms)].copy()
    cross_dropped = before_cross - len(df)

    sampled_df = safe_stratified_sample(
        df,
        n=CAMPUS_MAX_ROWS,
        strat_cols=["target_biro", "priority", "linguistic_style"],
        seed=SEED,
    )

    records: List[Dict[str, Any]] = []
    for _, row in sampled_df.iterrows():
        biro = str(row["target_biro"])
        topic = str(row["topic_category"])
        prio = str(row["priority"])
        style = str(row["linguistic_style"])
        if prio not in urgency_map:
            raise ValueError(f"Unexpected campus priority: {prio!r}")
        records.append(
            make_record(
                source=source,
                text=row["clean_text"],
                domain=map_domain(biro, topic),
                urgency_level=urgency_map[prio],
                is_sarcasm=1 if style == "sarkasme" else 0,
            )
        )

    split_records = stratified_split_70_15_15(
        records,
        lambda r: f"{r['domain'] or 'NONE'}|{r['urgency_level']}|{r['is_sarcasm']}",
    )
    register_seen_norms(split_records, seen_norms)

    log = {
        "raw_rows": raw_rows,
        "empty_dropped": empty_dropped,
        "exact_dups_removed": within_dups_removed,
        "cross_source_dropped": cross_dropped,
        "pool_before_sampling": len(df),
        "final_rows": len(split_records),
    }
    return split_records, log


def load_ner_source(
    source: str,
    rel_dir: str,
    filenames: Sequence[str],
    seen_norms: set[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load, detokenize, deduplicate, and split one SEACrowd CoNLL/TSV NER source."""
    raw_sentences = 0
    total_frags_removed = 0
    candidates: List[Dict[str, Any]] = []

    for fname in filenames:
        sents, frags = parse_conll_file(RAW_DIR / rel_dir / fname)
        raw_sentences += len(sents)
        total_frags_removed += frags
        for text, ents in sents:
            candidates.append(
                make_record(
                    source=source,
                    text=text,
                    entities=ents,
                )
            )

    deduped, dedup_stats = resolve_within_source_duplicates(candidates, label_key_fn=None)
    filtered, cross_dropped = filter_seen_cross_source(deduped, seen_norms)

    def ner_strat_key(rec: Dict[str, Any]) -> str:
        ents = rec["entities"] or []
        if not ents:
            return "NONE"
        return "+".join(sorted({e["label"] for e in ents}))

    split_records = stratified_split_70_15_15(filtered, ner_strat_key)
    register_seen_norms(split_records, seen_norms)

    log = {
        "raw_rows": raw_sentences,
        "tokenized_html_fragments_removed": total_frags_removed,
        **dedup_stats,
        "cross_source_dropped": cross_dropped,
        "final_rows": len(split_records),
    }
    return split_records, log


def load_twitter_sarcastic(seen_norms: set[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load and process w11wo/twitter_indonesia_sarcastic."""
    source = "twitter_indonesia_sarcastic"
    parquets = sorted((RAW_DIR / "hf" / "twitter_indonesia_sarcastic").glob("*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    raw_rows = len(df)

    candidates: List[Dict[str, Any]] = []
    empty_dropped = 0
    for _, row in df.iterrows():
        text = clean_text(row["tweet"])
        if not text:
            empty_dropped += 1
            continue
        lbl = int(row["label"])
        if lbl not in (0, 1):
            raise ValueError(f"Unexpected sarcasm label: {lbl!r}")
        candidates.append(
            make_record(
                source=source,
                text=text,
                is_sarcasm=lbl,
            )
        )

    deduped, dedup_stats = resolve_within_source_duplicates(
        candidates, label_key_fn=lambda r: r["is_sarcasm"], drop_all_collisions=False
    )
    filtered, cross_dropped = filter_seen_cross_source(deduped, seen_norms)
    split_records = stratified_split_70_15_15(filtered, lambda r: str(r["is_sarcasm"]))
    register_seen_norms(split_records, seen_norms)

    log = {
        "raw_rows": raw_rows,
        "empty_dropped": empty_dropped,
        **dedup_stats,
        "cross_source_dropped": cross_dropped,
        "final_rows": len(split_records),
    }
    return split_records, log


def load_ecommerce_sentiment(seen_norms: set[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load AIbnuHibban/e-commerce-sentiment-bahasa-indonesia, deduplicate while
    preserving challange.json sarcasm labels, drop cross-sentiment collisions,
    and draw a strictly balanced 1:1:1 sample (<= 4,000 rows).
    """
    source = "ecommerce_sentiment"
    df = pd.read_parquet(RAW_DIR / "hf" / "ecommerce_sentiment" / "train.parquet")
    raw_rows = len(df)

    sent_map = {"positive": "Positif", "neutral": "Netral", "negative": "Negatif"}
    df = df.copy()
    df["clean_text"] = df["comment"].map(clean_text)
    empty_dropped = int((df["clean_text"] == "").sum())
    df = df[df["clean_text"] != ""].copy()

    df["mapped_sentiment"] = df["sentiment"].map(sent_map)
    if df["mapped_sentiment"].isna().any():
        raise ValueError("Unexpected sentiment value in ecommerce_sentiment.")

    df["norm_text"] = df["clean_text"].map(normalize_for_dedup)
    sent_per_norm = df.groupby("norm_text")["mapped_sentiment"].nunique()
    collision_norms = set(sent_per_norm[sent_per_norm > 1].index)
    collisions_dropped = int(df["norm_text"].isin(collision_norms).sum())
    df = df[~df["norm_text"].isin(collision_norms)].copy()

    df["subset_priority"] = (df["subset"] != "challange").astype(int)
    df["sarc_priority"] = (df["category"].fillna("") != "sarcasm").astype(int)
    df = df.sort_values(["subset_priority", "sarc_priority"], kind="mergesort")
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["norm_text"], keep="first").copy()
    exact_dups_removed = before_dedup - len(df)

    before_cross = len(df)
    df = df[~df["norm_text"].isin(seen_norms)].copy()
    cross_dropped = before_cross - len(df)

    pools = {
        s: df[df["mapped_sentiment"] == s].copy()
        for s in ("Positif", "Netral", "Negatif")
    }
    per_class = min(ECOM_PER_CLASS_TARGET, *(len(p) for p in pools.values()))

    sampled_parts: List[pd.DataFrame] = []
    for s in ("Positif", "Netral", "Negatif"):
        pool = pools[s]
        if len(pool) <= per_class:
            sampled_parts.append(pool)
            continue

        if s == "Negatif":
            sarc_df = pool[pool["category"].fillna("") == "sarcasm"]
            non_sarc_df = pool[pool["category"].fillna("") != "sarcasm"]
            remaining_needed = per_class - len(sarc_df)
            if remaining_needed < 0:
                raise ValueError("More sarcasm rows than per_class quota.")
            non_sarc_sampled = safe_stratified_sample(
                non_sarc_df,
                n=remaining_needed,
                strat_cols=["subset"],
                seed=SEED,
            )
            sampled_parts.append(pd.concat([sarc_df, non_sarc_sampled], ignore_index=True))
        else:
            sampled_parts.append(
                safe_stratified_sample(pool, n=per_class, strat_cols=["subset"], seed=SEED)
            )

    sampled_df = pd.concat(sampled_parts, ignore_index=True)

    records: List[Dict[str, Any]] = []
    for _, row in sampled_df.iterrows():
        is_challange = row["subset"] == "challange" and pd.notna(row["category"])
        is_sarcasm = (1 if str(row["category"]) == "sarcasm" else 0) if is_challange else None
        records.append(
            make_record(
                source=source,
                text=row["clean_text"],
                sentiment=str(row["mapped_sentiment"]),
                is_sarcasm=is_sarcasm,
            )
        )

    split_records = stratified_split_70_15_15(
        records,
        lambda r: f"{r['sentiment']}|{r['is_sarcasm'] if r['is_sarcasm'] is not None else 'NULL'}",
    )
    register_seen_norms(split_records, seen_norms)

    log = {
        "raw_rows": raw_rows,
        "empty_dropped": empty_dropped,
        "collisions_dropped": collisions_dropped,
        "exact_dups_removed": exact_dups_removed,
        "cross_source_dropped": cross_dropped,
        "pool_counts_before_sampling": {k: int(len(v)) for k, v in pools.items()},
        "per_class_quota": int(per_class),
        "final_rows": len(split_records),
    }
    return split_records, log


def load_rizalespe_csv(
    source: str,
    filename: str,
    seen_norms: set[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load and process one GitHub rizalespe tweet sentiment CSV."""
    df = pd.read_csv(RAW_DIR / "github_rizalespe" / filename)
    raw_rows = len(df)
    sent_map = {"positive": "Positif", "negative": "Negatif", "neutral": "Netral"}

    candidates: List[Dict[str, Any]] = []
    empty_dropped = 0
    for _, row in df.iterrows():
        text = clean_text(row["Text Tweet"])
        if not text:
            empty_dropped += 1
            continue
        raw_sent = str(row["Sentiment"]).strip().lower()
        if raw_sent not in sent_map:
            raise ValueError(f"Unexpected sentiment in {filename}: {row['Sentiment']!r}")
        candidates.append(
            make_record(
                source=source,
                text=text,
                sentiment=sent_map[raw_sent],
            )
        )

    deduped, dedup_stats = resolve_within_source_duplicates(
        candidates, label_key_fn=lambda r: r["sentiment"], drop_all_collisions=False
    )
    filtered, cross_dropped = filter_seen_cross_source(deduped, seen_norms)
    split_records = stratified_split_70_15_15(filtered, lambda r: str(r["sentiment"]))
    register_seen_norms(split_records, seen_norms)

    log = {
        "raw_rows": raw_rows,
        "empty_dropped": empty_dropped,
        **dedup_stats,
        "cross_source_dropped": cross_dropped,
        "final_rows": len(split_records),
    }
    return split_records, log


def load_synthetic_edge_cases(
    seen_norms: set[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load pre-split 2,760 synthetic edge-case complaints from data/raw/synthetic/."""
    path = RAW_DIR / "synthetic" / "synthetic_edge_cases.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing synthetic dataset at {path}. Run scripts/generate_synthetic_edge_cases.py first."
        )

    raw_rows = 0
    empty_dropped = 0
    candidates: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_rows += 1
            obj = json.loads(line)
            text = str(obj.get("text") or "").strip()
            if not text:
                empty_dropped += 1
                continue
            candidates.append(
                make_record(
                    source="synthetic",
                    text=text,
                    domain=obj.get("domain"),
                    intent=obj.get("intent"),
                    entities=obj.get("entities") if obj.get("entities") is not None else [],
                    sentiment=obj.get("sentiment"),
                    is_sarcasm=obj.get("is_sarcasm"),
                    urgency_level=obj.get("urgency_level"),
                    split=obj.get("split", "train"),
                )
            )

    filtered, cross_dropped = filter_seen_cross_source(candidates, seen_norms)
    register_seen_norms(filtered, seen_norms)

    log = {
        "raw_rows": raw_rows,
        "empty_dropped": empty_dropped,
        "cross_source_dropped": cross_dropped,
        "final_rows": len(filtered),
    }
    return filtered, log


def assign_sequential_ids(records: List[Dict[str, Any]], prefix: str) -> List[Dict[str, Any]]:
    """Assign deterministic 4-digit zero-padded IDs ordered by split (train, val, test)."""
    split_order = {"train": 0, "val": 1, "test": 2}
    ordered = sorted(records, key=lambda r: split_order[r["split"]])
    for idx, rec in enumerate(ordered, start=1):
        rec["id"] = f"{prefix}_{idx:04d}"
    return ordered


def validate_master_dataset(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run strict schema, anti-leakage, NER span, and cleaning assertions."""
    seen_ids: set[str] = set()
    split_norms: Dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}

    ner_sources = {"indonlu_nergrit", "indolem_nerui", "idner_news_2k", "synthetic"}
    bad_ner_substrings = ("<iframe", "allowfullscreen", "< strong", "enamex", "http://", "https://")

    total_entity_spans = 0
    entity_label_counts: Counter[str] = Counter()
    negation_count = 0
    intensifier_count = 0
    neg_re = re.compile(r"\b(?:tidak|bukan|belum)\b", re.IGNORECASE)
    int_re = re.compile(r"\b(?:parah|sangat)\b", re.IGNORECASE)

    for rec in records:
        if list(rec.keys()) != SCHEMA_KEYS:
            raise AssertionError(f"Invalid keys for {rec.get('id')}: {list(rec.keys())}")

        rid = rec["id"]
        if not isinstance(rid, str) or not rid or rid in seen_ids:
            raise AssertionError(f"Duplicate or empty id: {rid!r}")
        seen_ids.add(rid)

        split = rec["split"]
        if split not in VALID_SPLITS:
            raise AssertionError(f"Invalid split {split!r} in {rid}")

        text = rec["text"]
        if not isinstance(text, str) or not text or text != text.strip():
            raise AssertionError(f"Invalid text in {rid}: {text!r}")

        if rec["domain"] is not None and rec["domain"] not in VALID_DOMAINS:
            raise AssertionError(f"Invalid domain {rec['domain']!r} in {rid}")
        if rec["intent"] is not None and rec["intent"] not in VALID_INTENTS:
            raise AssertionError(f"Invalid intent {rec['intent']!r} in {rid}")
        if rec["sentiment"] is not None and rec["sentiment"] not in VALID_SENTIMENTS:
            raise AssertionError(f"Invalid sentiment {rec['sentiment']!r} in {rid}")
        if rec["is_sarcasm"] is not None:
            if type(rec["is_sarcasm"]) is not int or rec["is_sarcasm"] not in (0, 1):
                raise AssertionError(f"Invalid is_sarcasm {rec['is_sarcasm']!r} in {rid}")
        if rec["urgency_level"] is not None and rec["urgency_level"] not in VALID_URGENCY_LEVELS:
            raise AssertionError(f"Invalid urgency_level {rec['urgency_level']!r} in {rid}")

        if rec["source"] in ner_sources:
            lower_text = text.lower()
            for bad in bad_ner_substrings:
                if bad in lower_text:
                    raise AssertionError(f"Forbidden NER artifact {bad!r} in {rid}: {text!r}")
            if not isinstance(rec["entities"], list):
                raise AssertionError(f"NER record {rid} must have list entities, got {type(rec['entities'])}")
        else:
            if rec["entities"] is not None:
                raise AssertionError(f"Non-NER record {rid} must have null entities.")

        if rec["entities"] is not None:
            for ent in rec["entities"]:
                start, end, label = ent["start"], ent["end"], ent["label"]
                if label not in VALID_ENTITY_LABELS:
                    raise AssertionError(f"Invalid entity label {label!r} in {rid}")
                if not (0 <= start < end <= len(text)):
                    raise AssertionError(f"Out-of-bounds span {(start, end)} in {rid}")
                substr = text[start:end]
                if not substr or substr != substr.strip():
                    raise AssertionError(f"Whitespace-padded entity span {substr!r} in {rid}")
                total_entity_spans += 1
                entity_label_counts[label] += 1

        split_norms[split].add(normalize_for_dedup(text))
        if neg_re.search(text):
            negation_count += 1
        if int_re.search(text):
            intensifier_count += 1

        json.dumps(rec, allow_nan=False, ensure_ascii=False)

    tv_overlap = split_norms["train"] & split_norms["val"]
    tt_overlap = split_norms["train"] & split_norms["test"]
    vt_overlap = split_norms["val"] & split_norms["test"]
    if tv_overlap or tt_overlap or vt_overlap:
        raise AssertionError(
            f"Data leakage across splits! train&val={len(tv_overlap)}, "
            f"train&test={len(tt_overlap)}, val&test={len(vt_overlap)}"
        )

    if negation_count == 0 or intensifier_count == 0:
        raise AssertionError("Negation or intensifier words missing after cleaning!")

    return {
        "total_entity_spans": total_entity_spans,
        "entity_label_counts": dict(entity_label_counts),
        "rows_with_negation_words": negation_count,
        "rows_with_intensifier_words": intensifier_count,
        "cross_split_text_overlap": 0,
    }


def write_outputs(
    records: List[Dict[str, Any]],
    source_logs: Dict[str, Dict[str, Any]],
    validation_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Write JSONL, Parquet, preview CSV, and stats JSON to data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_path = PROCESSED_DIR / "master_dataset.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, allow_nan=False, ensure_ascii=False) + "\n")

    parquet_rows = []
    for rec in records:
        row = dict(rec)
        row["entities"] = (
            json.dumps(row["entities"], ensure_ascii=False)
            if row["entities"] is not None
            else None
        )
        parquet_rows.append(row)
    df_out = pd.DataFrame(parquet_rows, columns=SCHEMA_KEYS)
    df_out["is_sarcasm"] = df_out["is_sarcasm"].astype("Int64")
    parquet_path = PROCESSED_DIR / "master_dataset.parquet"
    df_out.to_parquet(parquet_path, index=False)

    preview_rows: List[Dict[str, Any]] = []
    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_source[rec["source"]].append(rec)
    for src_records in by_source.values():
        subset = (
            [r for r in src_records if r["split"] == "train"][:14]
            + [r for r in src_records if r["split"] == "val"][:3]
            + [r for r in src_records if r["split"] == "test"][:3]
        )
        for rec in subset:
            row = dict(rec)
            row["entities"] = (
                json.dumps(row["entities"], ensure_ascii=False)
                if row["entities"] is not None
                else None
            )
            preview_rows.append(row)
    preview_path = PROCESSED_DIR / "master_dataset_preview.csv"
    pd.DataFrame(preview_rows, columns=SCHEMA_KEYS).to_csv(
        preview_path, index=False, encoding="utf-8-sig"
    )

    split_counts = Counter(r["split"] for r in records)
    source_counts: Dict[str, Dict[str, int]] = {}
    for src, src_records in by_source.items():
        sc = Counter(r["split"] for r in src_records)
        source_counts[src] = {
            "total": len(src_records),
            "train": sc.get("train", 0),
            "val": sc.get("val", 0),
            "test": sc.get("test", 0),
        }

    target_non_null = {
        col: sum(1 for r in records if r[col] is not None)
        for col in ("domain", "intent", "entities", "sentiment", "is_sarcasm", "urgency_level")
    }
    target_distributions = {
        "domain": dict(Counter(r["domain"] for r in records if r["domain"] is not None)),
        "intent": dict(Counter(r["intent"] for r in records if r["intent"] is not None)),
        "sentiment": dict(Counter(r["sentiment"] for r in records if r["sentiment"] is not None)),
        "is_sarcasm": {
            str(k): v
            for k, v in Counter(r["is_sarcasm"] for r in records if r["is_sarcasm"] is not None).items()
        },
        "urgency_level": dict(
            Counter(r["urgency_level"] for r in records if r["urgency_level"] is not None)
        ),
        "ner_non_empty_sentences": sum(
            1 for r in records if isinstance(r["entities"], list) and len(r["entities"]) > 0
        ),
        "ner_empty_sentences": sum(
            1 for r in records if isinstance(r["entities"], list) and len(r["entities"]) == 0
        ),
    }

    samples_per_source: Dict[str, Dict[str, Any]] = {}
    for src, src_records in by_source.items():
        if src in {"indonlu_nergrit", "indolem_nerui", "idner_news_2k"}:
            sample = next((r for r in src_records if r["entities"]), src_records[0])
        elif src == "ecommerce_sentiment":
            sample = next((r for r in src_records if r["is_sarcasm"] == 1), src_records[0])
        elif src == "twitter_indonesia_sarcastic":
            sample = next((r for r in src_records if r["is_sarcasm"] == 1), src_records[0])
        else:
            sample = src_records[0]
        samples_per_source[src] = sample

    stats = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "total_rows": len(records),
        "split_counts": {k: split_counts.get(k, 0) for k in ("train", "val", "test")},
        "source_counts": source_counts,
        "target_non_null": target_non_null,
        "target_distributions": target_distributions,
        "validation": validation_summary,
        "processing_logs": source_logs,
        "samples_per_source": samples_per_source,
    }
    stats_path = PROCESSED_DIR / "dataset_stats.json"
    stats_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return stats


def print_summary_report(stats: Dict[str, Any]) -> None:
    """Print formatted summary of the Unified Master Table."""
    print("\n" + "=" * 72)
    print("UNIFIED MASTER TABLE SUMMARY")
    print("=" * 72)
    print(f"Total rows : {stats['total_rows']:,}")
    print(f"Splits     : {stats['split_counts']}")

    print("\n--- Rows per Source (train / val / test) ---")
    for src, c in stats["source_counts"].items():
        print(
            f"  {src:<30} total={c['total']:>5,}  "
            f"(train={c['train']:>4,}, val={c['val']:>4,}, test={c['test']:>4,})"
        )

    print("\n--- Non-Null Target Counts ---")
    for col, cnt in stats["target_non_null"].items():
        print(f"  {col:<16}: {cnt:>6,}")

    print("\n--- Target Label Distributions ---")
    for col, dist in stats["target_distributions"].items():
        print(f"  {col}: {dist}")

    print("\n--- Validation Checks ---")
    for k, v in stats["validation"].items():
        print(f"  {k}: {v}")

    print("\n--- Sample Output Row per Source ---")
    for src, sample in stats["samples_per_source"].items():
        print(f"\n[{src}]")
        print(json.dumps(sample, ensure_ascii=False))
        if sample["entities"]:
            extracted = [
                f"{sample['text'][e['start']:e['end']]} ({e['label']})"
                for e in sample["entities"]
            ]
            print(f"  Verified spans: {extracted}")


def main() -> None:
    configure_utf8_io()
    seen_norms: set[str] = set()
    all_records: List[Dict[str, Any]] = []
    source_logs: Dict[str, Dict[str, Any]] = {}

    loaders: List[Tuple[str, str, Callable[[], Tuple[List[Dict[str, Any]], Dict[str, Any]]]]] = [
        ("sp4n_lapor", "sp4n", lambda: load_sp4n_lapor(seen_norms)),
        ("campus_complaints", "campus", lambda: load_campus_complaints(seen_norms)),
        (
            "indonlu_nergrit",
            "nergrit",
            lambda: load_ner_source(
                "indonlu_nergrit",
                "ner/indonlu_nergrit",
                ["train_preprocess.txt", "valid_preprocess.txt", "test_preprocess.txt"],
                seen_norms,
            ),
        ),
        (
            "indolem_nerui",
            "nerui",
            lambda: load_ner_source(
                "indolem_nerui",
                "ner/indolem_nerui",
                ["train.01.tsv", "dev.01.tsv", "test.01.tsv"],
                seen_norms,
            ),
        ),
        (
            "idner_news_2k",
            "idner",
            lambda: load_ner_source(
                "idner_news_2k",
                "ner/idner_news_2k",
                ["train.txt", "dev.txt", "test.txt"],
                seen_norms,
            ),
        ),
        (
            "twitter_indonesia_sarcastic",
            "sarc",
            lambda: load_twitter_sarcastic(seen_norms),
        ),
        (
            "ecommerce_sentiment",
            "ecom",
            lambda: load_ecommerce_sentiment(seen_norms),
        ),
        (
            "rizalespe_cellular_provider",
            "cell",
            lambda: load_rizalespe_csv(
                "rizalespe_cellular_provider",
                "dataset_tweet_sentiment_cellular_service_provider.csv",
                seen_norms,
            ),
        ),
        (
            "rizalespe_pilkada_dki",
            "pilkada",
            lambda: load_rizalespe_csv(
                "rizalespe_pilkada_dki",
                "dataset_tweet_sentiment_pilkada_DKI_2017.csv",
                seen_norms,
            ),
        ),
        ("synthetic", "syn", lambda: load_synthetic_edge_cases(seen_norms)),
    ]

    for source_name, prefix, fn in loaders:
        records, log = fn()
        records = assign_sequential_ids(records, prefix)
        all_records.extend(records)
        source_logs[source_name] = log
        print(f"[{source_name}] processed: {log}")

    validation_summary = validate_master_dataset(all_records)
    stats = write_outputs(all_records, source_logs, validation_summary)
    print_summary_report(stats)
    print(f"\nSaved outputs in: {PROCESSED_DIR.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()

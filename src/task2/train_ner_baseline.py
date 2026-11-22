"""
Task 2 E1 CPU Baseline Trainer & Evaluator (`src/task2/train_ner_baseline.py`).

Combines:
1. **Feature-Based Contextual BIO2 Tagger** (`DictVectorizer` + `LogisticRegression`
   with orthography, word-shape, affix, ±2 token context window, and civic prefix features)
   fitted strictly on `split == "train"`.
2. **Cased-Text Civic Gazetteer & 3-Gate `, ` Boundary Post-Processing** (`src/task2/gazetteer.py`).

Saves:
- Model artifact: `models/task2/ner_baseline/baseline_model.joblib`
- Evaluation report: `outputs/task2_ner_e1.json` (including the oracle upper-bound check
  and `val (real)` vs `val (synthetic)` post-processing safeguard check).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from src.task2.data_loader import (
    DEFAULT_MASTER_PATH,
    DEFAULT_MAX_LENGTH,
    IGNORE_INDEX,
    ROOT_DIR,
    decode_bio2_to_spans,
    encode_spans_to_bio2,
    load_ner_records,
    regex_tokenize_with_offsets,
    summarize_ner_records,
    verify_dataset_span_alignment,
)
from src.task2.evaluate_ner import evaluate_predictions_across_subsets
from src.task2.gazetteer import (
    CIVIC_COMPLAINT_SENTENCE_RE,
    apply_civic_postprocessing,
)


DEFAULT_BASELINE_MODEL_DIR = ROOT_DIR / "models" / "task2" / "ner_baseline"
DEFAULT_BASELINE_REPORT_PATH = ROOT_DIR / "outputs" / "task2_ner_e1.json"

PERSON_TITLES = {
    "pak", "bu", "bapak", "ibu", "mas", "mbak", "dr", "prof", "ir", "kh", "sdr", "sdri", "bpk",
}
LOC_CUE_WORDS = {
    "jl", "jalan", "kecamatan", "kec", "kelurahan", "kel", "desa", "kabupaten", "kab",
    "kota", "provinsi", "pasar", "terminal", "stasiun", "pelabuhan", "bandara", "simpang",
    "gunung", "pulau", "sungai", "danau", "pantai", "gua", "dusun",
}
ORG_CUE_WORDS = {
    "dinas", "dishub", "disdukcapil", "dlh", "bpbd", "rsud", "puskesmas", "polsek",
    "polres", "polda", "samsat", "pdam", "pln", "satpol", "inspektorat", "kantor",
    "universitas", "institut", "fakultas", "sekolah", "sma", "smk", "smp", "pt", "cv",
    "bank", "partai", "kementerian", "komisi", "yayasan",
}
LOC_PREPOSITIONS = {"di", "ke", "dari", "menuju", "wilayah", "daerah", "area", "kondisi", "warga"}


def _word_shape(token: str) -> str:
    shape: List[str] = []
    for ch in token[:20]:
        if ch.isupper():
            shape.append("X")
        elif ch.islower():
            shape.append("x")
        elif ch.isdigit():
            shape.append("d")
        else:
            shape.append(ch)
    collapsed = re.sub(
        r"(X+|x+|d+)",
        lambda m: m.group(0)[0] + ("+" if len(m.group(0)) > 1 else ""),
        "".join(shape),
    )
    return collapsed


def extract_token_features(
    tokens: Sequence[str],
    offsets: Sequence[Tuple[int, int]],
    has_civic_cue: bool,
) -> List[Dict[str, Any]]:
    """
    Extract rich orthographic, affix, and ±2 context window features for each
    non-special token in `tokens`.
    """
    valid_indices = [
        i for i, (s, e) in enumerate(offsets) if not (s == 0 and e == 0) and s < e
    ]
    valid_tokens = [tokens[i] for i in valid_indices]
    n = len(valid_tokens)
    feats_list: List[Dict[str, Any]] = []

    for pos, tok in enumerate(valid_tokens):
        low = tok.lower()
        prev1 = valid_tokens[pos - 1] if pos >= 1 else "<BOS>"
        prev2 = valid_tokens[pos - 2] if pos >= 2 else "<BOS2>"
        next1 = valid_tokens[pos + 1] if pos + 1 < n else "<EOS>"
        next2 = valid_tokens[pos + 2] if pos + 2 < n else "<EOS2>"

        prev1_low = prev1.lower()
        prev2_low = prev2.lower()
        next1_low = next1.lower()

        f: Dict[str, Any] = {
            "bias": 1.0,
            "w": low,
            "shape": _word_shape(tok),
            "is_title": tok.istitle(),
            "is_upper": tok.isupper() and any(c.isalpha() for c in tok),
            "is_digit": tok.isdigit(),
            "is_punct": not any(c.isalnum() for c in tok),
            "has_capital": any(c.isupper() for c in tok),
            "pref2": low[:2],
            "pref3": low[:3],
            "suf2": low[-2:],
            "suf3": low[-3:],
            "pos_zero": pos == 0,
            "prev1_w": prev1_low,
            "prev1_shape": _word_shape(prev1) if prev1 != "<BOS>" else "<BOS>",
            "prev1_title": prev1.istitle(),
            "prev2_w": prev2_low,
            "next1_w": next1_low,
            "next1_shape": _word_shape(next1) if next1 != "<EOS>" else "<EOS>",
            "next1_title": next1.istitle(),
            "next2_w": next2.lower(),
            "bigram_prev": f"{prev1_low}|{low}",
            "bigram_next": f"{low}|{next1_low}",
            "is_person_title": low in PERSON_TITLES,
            "prev1_person_title": prev1_low in PERSON_TITLES,
            "prev2_person_title": prev2_low in PERSON_TITLES,
            "is_loc_cue": low in LOC_CUE_WORDS,
            "prev1_loc_cue": prev1_low in LOC_CUE_WORDS,
            "prev2_loc_cue": prev2_low in LOC_CUE_WORDS,
            "prev1_loc_prep": prev1_low in LOC_PREPOSITIONS,
            "is_org_cue": low in ORG_CUE_WORDS,
            "prev1_org_cue": prev1_low in ORG_CUE_WORDS,
            "prev2_org_cue": prev2_low in ORG_CUE_WORDS,
            "civic_sent": has_civic_cue,
        }
        feats_list.append(f)

    return feats_list


def fit_baseline_model(
    train_records: Sequence[Dict[str, Any]],
    c_param: float = 1.5,
) -> Dict[str, Any]:
    """
    Fit the E1 Feature-Based BIO2 Tagger strictly on `train_records`.
    """
    all_features: List[Dict[str, Any]] = []
    all_labels: List[int] = []

    for rec in train_records:
        text = rec["text"]
        has_civic = bool(CIVIC_COMPLAINT_SENTENCE_RE.search(text))
        enc = regex_tokenize_with_offsets(text, max_length=DEFAULT_MAX_LENGTH, add_special_tokens=True)
        tokens = enc["tokens"]
        offsets = enc["offset_mapping"]
        bio_ids = encode_spans_to_bio2(offsets, rec.get("entities") or [])

        valid_bio = [
            bio_ids[i]
            for i, (s, e) in enumerate(offsets)
            if not (s == 0 and e == 0) and s < e and bio_ids[i] != IGNORE_INDEX
        ]
        token_feats = extract_token_features(tokens, offsets, has_civic_cue=has_civic)
        if len(token_feats) != len(valid_bio):
            continue
        all_features.extend(token_feats)
        all_labels.extend(valid_bio)

    vectorizer = DictVectorizer(sparse=True)
    X_train = vectorizer.fit_transform(all_features)
    clf = LogisticRegression(
        C=c_param,
        max_iter=300,
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(X_train, all_labels)

    return {
        "model_type": "E1_Feature_BIO2_Civic_Gazetteer",
        "c_param": c_param,
        "vectorizer": vectorizer,
        "classifier": clf,
        "train_rows": len(train_records),
        "train_tokens": len(all_labels),
    }


def predict_baseline_spans(
    text: str,
    model_bundle: Dict[str, Any],
    use_postprocessing: bool = True,
) -> List[Dict[str, Any]]:
    """
    Predict NER spans for `text` using the E1 baseline model bundle.
    - When `use_postprocessing=False`: returns raw feature-based BIO2 decoded spans.
    - When `use_postprocessing=True`: applies cased civic gazetteer, boundary cleanup,
      Kelurahan/Kecamatan disambiguation, and the 3-gate `, ` address merger on top
      of the exact same decoded spans.
    """
    has_civic = bool(CIVIC_COMPLAINT_SENTENCE_RE.search(text))
    enc = regex_tokenize_with_offsets(text, max_length=DEFAULT_MAX_LENGTH, add_special_tokens=True)
    tokens = enc["tokens"]
    offsets = enc["offset_mapping"]

    valid_indices = [
        i for i, (s, e) in enumerate(offsets) if not (s == 0 and e == 0) and s < e
    ]
    if not valid_indices:
        return []

    feats = extract_token_features(tokens, offsets, has_civic_cue=has_civic)
    X = model_bundle["vectorizer"].transform(feats)
    pred_valid_ids = model_bundle["classifier"].predict(X).tolist()

    full_tag_ids = [IGNORE_INDEX] * len(offsets)
    for idx, tag_id in zip(valid_indices, pred_valid_ids):
        full_tag_ids[idx] = int(tag_id)

    raw_spans = decode_bio2_to_spans(
        text,
        offsets,
        full_tag_ids,
        tokens=tokens,
        repair_subwords=False,
        promote_orphan_i=False,
    )

    if not use_postprocessing:
        return raw_spans

    return apply_civic_postprocessing(text, raw_spans, inject_gazetteer=True)


def load_or_fit_baseline_model(
    data_path: Path = DEFAULT_MASTER_PATH,
    model_dir: Path = DEFAULT_BASELINE_MODEL_DIR,
) -> Dict[str, Any]:
    """Load saved E1 baseline artifact if present, else fit on train and save."""
    artifact_path = model_dir / "baseline_model.joblib"
    if artifact_path.exists():
        return joblib.load(artifact_path)

    train_records = load_ner_records(data_path=data_path, split="train", subset="all")
    bundle = fit_baseline_model(train_records)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_path)
    return bundle


def run_baseline_experiment(
    data_path: Path = DEFAULT_MASTER_PATH,
    model_dir: Path = DEFAULT_BASELINE_MODEL_DIR,
    report_path: Path = DEFAULT_BASELINE_REPORT_PATH,
) -> Dict[str, Any]:
    """
    Execute Phase 1 & Phase 2 E1 baseline training, dataset alignment check,
    oracle post-processing check, 3-gate `, ` merger safeguard verification on `val (real)`,
    and `test` evaluation.
    """
    all_records = load_ner_records(data_path=data_path, split=None, subset="all")
    train_records = [r for r in all_records if r["split"] == "train"]
    val_records = [r for r in all_records if r["split"] == "val"]
    test_records = [r for r in all_records if r["split"] == "test"]

    # 1. Verify Phase 1 zero-truncation and round-trip alignment at max_length=384
    alignment_check = verify_dataset_span_alignment(all_records, max_length=DEFAULT_MAX_LENGTH)
    if alignment_check["truncated_spans"] != 0 or alignment_check["alignment_loss_rate"] != 0.0:
        raise AssertionError(f"Span alignment check failed: {alignment_check}")

    # 2. Verify Oracle Post-Processing Upper Bound on val (feeding gold spans into apply_civic_postprocessing)
    oracle_val = evaluate_predictions_across_subsets(
        val_records,
        lambda r: apply_civic_postprocessing(r["text"], r["entities"], inject_gazetteer=True),
        subsets=("all", "real", "synthetic"),
        collect_errors=0,
    )

    # 3. Tune C on validation set (strict Micro-F1)
    best_c = 1.5
    best_val_f1 = -1.0
    best_bundle: Optional[Dict[str, Any]] = None
    c_candidates: Dict[str, float] = {}

    for c_val in (0.5, 1.5, 3.0):
        bundle = fit_baseline_model(train_records, c_param=c_val)
        val_rep = evaluate_predictions_across_subsets(
            val_records,
            lambda r, b=bundle: predict_baseline_spans(r["text"], b, use_postprocessing=True),
            subsets=("all", "real", "synthetic"),
            collect_errors=0,
        )
        f1 = val_rep["all"]["micro_f1"]
        c_candidates[str(c_val)] = f1
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_c = c_val
            best_bundle = bundle

    assert best_bundle is not None
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = model_dir / "baseline_model.joblib"
    joblib.dump(best_bundle, artifact_path)

    # 4. Isolated Safeguard Check on val: Raw BIO vs Civic Post-Processing on same decoded spans
    val_raw_metrics = evaluate_predictions_across_subsets(
        val_records,
        lambda r: predict_baseline_spans(r["text"], best_bundle, use_postprocessing=False),
        collect_errors=5,
    )
    val_post_metrics = evaluate_predictions_across_subsets(
        val_records,
        lambda r: predict_baseline_spans(r["text"], best_bundle, use_postprocessing=True),
        collect_errors=5,
    )

    val_real_delta = round(
        val_post_metrics["real"]["micro_f1"] - val_raw_metrics["real"]["micro_f1"], 4
    )
    val_syn_delta = round(
        val_post_metrics["synthetic"]["micro_f1"] - val_raw_metrics["synthetic"]["micro_f1"], 4
    )

    # 5. Final test evaluation
    test_metrics = evaluate_predictions_across_subsets(
        test_records,
        lambda r: predict_baseline_spans(r["text"], best_bundle, use_postprocessing=True),
        collect_errors=10,
    )

    report = {
        "experiment": "E1_Baseline_Civic_BIO",
        "dataset_profile": {
            "all": summarize_ner_records(all_records),
            "train": summarize_ner_records(train_records),
            "val": summarize_ner_records(val_records),
            "test": summarize_ner_records(test_records),
        },
        "span_alignment_check_max_len_384": {
            k: v for k, v in alignment_check.items() if k != "mismatches"
        },
        "tuning": {
            "selected_c": best_c,
            "val_micro_f1_by_c": c_candidates,
        },
        "val_postprocessing_safeguard": {
            "oracle_val_all_micro_f1": oracle_val["all"]["micro_f1"],
            "oracle_val_real_micro_f1": oracle_val["real"]["micro_f1"],
            "oracle_val_synthetic_micro_f1": oracle_val["synthetic"]["micro_f1"],
            "raw_val_real_micro_f1": val_raw_metrics["real"]["micro_f1"],
            "post_val_real_micro_f1": val_post_metrics["real"]["micro_f1"],
            "val_real_delta": val_real_delta,
            "raw_val_synthetic_micro_f1": val_raw_metrics["synthetic"]["micro_f1"],
            "post_val_synthetic_micro_f1": val_post_metrics["synthetic"]["micro_f1"],
            "val_synthetic_delta": val_syn_delta,
            "no_real_regression": val_real_delta >= 0.0,
        },
        "val_metrics": val_post_metrics,
        "test_metrics": test_metrics,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Task 2 E1 NER CPU Baseline.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_MASTER_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_BASELINE_MODEL_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_BASELINE_REPORT_PATH)
    args = parser.parse_args()

    report = run_baseline_experiment(
        data_path=args.data_path,
        model_dir=args.model_dir,
        report_path=args.report_path,
    )

    align = report["span_alignment_check_max_len_384"]
    print("=== PHASE 1 · SPAN ALIGNMENT CHECK (max_length=384) ===")
    print(
        f"  Records={align['total_records']:,} | Gold Spans={align['total_gold_spans']:,} | "
        f"Recovered={align['exact_recovered_spans']:,} | Truncated={align['truncated_spans']} | "
        f"Loss={align['alignment_loss_rate']:.6f}"
    )

    sg = report["val_postprocessing_safeguard"]
    print("\n=== PHASE 2 · 3-GATE COMMA MERGER & POST-PROCESSING SAFEGUARD (VAL) ===")
    print(
        f"  Oracle Upper Bound (gold -> post): all={sg['oracle_val_all_micro_f1']:.4f} | "
        f"real={sg['oracle_val_real_micro_f1']:.4f} | synthetic={sg['oracle_val_synthetic_micro_f1']:.4f}"
    )
    print(
        f"  val (real)      Micro-F1: raw={sg['raw_val_real_micro_f1']:.4f} -> "
        f"post={sg['post_val_real_micro_f1']:.4f} (delta={sg['val_real_delta']:+.4f}, "
        f"no_regression={sg['no_real_regression']})"
    )
    print(
        f"  val (synthetic) Micro-F1: raw={sg['raw_val_synthetic_micro_f1']:.4f} -> "
        f"post={sg['post_val_synthetic_micro_f1']:.4f} (delta={sg['val_synthetic_delta']:+.4f})"
    )

    print("\n=== E1 BASELINE (FEATURE BIO2 + CIVIC GAZETTEER) ===")
    for split_key in ("val_metrics", "test_metrics"):
        m_split = report[split_key]
        print(f"\n[{split_key}] latency={m_split['latency_ms_per_sample']} ms/sample")
        for sub in ("all", "real", "synthetic", "indonlu_nergrit", "indolem_nerui", "idner_news_2k"):
            m = m_split[sub]
            pe = m["per_entity"]
            print(
                f"  {sub:<16} docs={m['n_docs']:>4} spans={m['n_gold_spans']:>4} | "
                f"P={m['precision']:.4f} R={m['recall']:.4f} Micro-F1={m['micro_f1']:.4f} "
                f"Macro-F1={m['macro_f1']:.4f} | "
                f"LOC={pe['LOCATION']['f1']:.4f} ORG={pe['ORGANIZATION']['f1']:.4f} PER={pe['PERSON']['f1']:.4f}"
            )

    print(f"\nSaved E1 baseline model  -> {args.model_dir / 'baseline_model.joblib'}")
    print(f"Saved E1 baseline report -> {args.report_path}")


if __name__ == "__main__":
    main()

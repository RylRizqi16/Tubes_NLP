"""
Pure-Python Span-Level & Partial/Token-Level NER Evaluator (`src/task2/evaluate_ner.py`).

Computes:
1. **Primary Metrics**: Strict exact character-span `(start, end, label)` Precision,
   Recall, Micro-F1, Macro-F1, and per-entity F1 (`LOCATION`, `ORGANIZATION`, `PERSON`).
2. **Secondary Diagnostic**: Type-matched Partial-Boundary Overlap F1 and Word-Token F1
   (surfaces boundary-convention differences such as comma-separated civic addresses in E4).
3. **Breakdown Slices**: `all`, `real`, `synthetic`, and each individual source
   (`indonlu_nergrit`, `indolem_nerui`, `idner_news_2k`, `synthetic`), plus latency (`ms/sample`).
4. **2x2 Multi-Seed Ablation Summary**: Evaluates E1, E2, E3, E4a, and E4b (mean ± std across seeds)
   and writes `outputs/task2_ner_eval_report.json`.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from src.task2.data_loader import (
    DEFAULT_MASTER_PATH,
    ENTITY_LABELS,
    NER_SOURCES,
    ROOT_DIR,
    decode_bio2_to_spans,
    filter_ner_subset,
    load_ner_records,
)
from src.task2.gazetteer import apply_civic_postprocessing


DEFAULT_MODELS_DIR = ROOT_DIR / "models" / "task2"
DEFAULT_EVAL_OUTPUT = ROOT_DIR / "outputs" / "task2_ner_eval_report.json"

EVAL_SUBSETS: Tuple[str, ...] = (
    "all",
    "real",
    "synthetic",
    *NER_SOURCES,
)

WORD_RE = re.compile(r"\w+")


def _raw_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def _safe_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec, rec, f1 = _raw_prf(tp, fp, fn)
    return round(prec, 4), round(rec, 4), round(f1, 4)


def _span_word_tokens(text: str, spans: Sequence[Dict[str, Any]]) -> Set[Tuple[int, int, str]]:
    """Extract `(word_start, word_end, label)` word tokens covered by entity spans."""
    word_set: Set[Tuple[int, int, str]] = set()
    for sp in spans:
        s, e, lbl = int(sp["start"]), int(sp["end"]), str(sp["label"])
        if 0 <= s < e <= len(text):
            for m in WORD_RE.finditer(text[s:e]):
                word_set.add((s + m.start(), s + m.end(), lbl))
    return word_set


def compute_ner_metrics(
    records: Sequence[Dict[str, Any]],
    predictions: Sequence[Sequence[Dict[str, Any]]],
    collect_errors: int = 0,
) -> Dict[str, Any]:
    """
    Compute strict exact-span `(start, end, label)` metrics and partial/token-level
    diagnostic metrics for aligned `(records, predictions)`.
    """
    if len(records) != len(predictions):
        raise ValueError(
            f"Record count ({len(records)}) != prediction count ({len(predictions)})"
        )

    tp_by_label = {lbl: 0 for lbl in ENTITY_LABELS}
    fp_by_label = {lbl: 0 for lbl in ENTITY_LABELS}
    fn_by_label = {lbl: 0 for lbl in ENTITY_LABELS}
    support_by_label = {lbl: 0 for lbl in ENTITY_LABELS}
    pred_count_by_label = {lbl: 0 for lbl in ENTITY_LABELS}

    # Secondary diagnostics: partial-boundary overlap & word-token level
    overlap_gold_matched = 0
    overlap_pred_matched = 0
    total_gold_spans = 0
    total_pred_spans = 0

    token_tp = 0
    token_fp = 0
    token_fn = 0

    error_samples: List[Dict[str, Any]] = []

    for rec, pred_spans in zip(records, predictions):
        text = rec["text"]
        gold_spans = rec.get("entities") or []

        gold_set: Set[Tuple[int, int, str]] = {
            (int(e["start"]), int(e["end"]), str(e["label"])) for e in gold_spans
        }
        pred_set: Set[Tuple[int, int, str]] = {
            (int(e["start"]), int(e["end"]), str(e["label"])) for e in pred_spans
        }

        total_gold_spans += len(gold_set)
        total_pred_spans += len(pred_set)

        for s, e, lbl in gold_set:
            if lbl in support_by_label:
                support_by_label[lbl] += 1
        for s, e, lbl in pred_set:
            if lbl in pred_count_by_label:
                pred_count_by_label[lbl] += 1

        for lbl in ENTITY_LABELS:
            g_lbl = {(s, e) for s, e, l in gold_set if l == lbl}
            p_lbl = {(s, e) for s, e, l in pred_set if l == lbl}
            tp = len(g_lbl & p_lbl)
            fp = len(p_lbl - g_lbl)
            fn = len(g_lbl - p_lbl)
            tp_by_label[lbl] += tp
            fp_by_label[lbl] += fp
            fn_by_label[lbl] += fn

        # Partial boundary overlap (same label, non-empty character overlap)
        for gs, ge, gl in gold_set:
            if any(pl == gl and max(gs, ps) < min(ge, pe) for ps, pe, pl in pred_set):
                overlap_gold_matched += 1
        for ps, pe, pl in pred_set:
            if any(gl == pl and max(gs, ps) < min(ge, pe) for gs, ge, gl in gold_set):
                overlap_pred_matched += 1

        # Word-token level overlap
        gold_words = _span_word_tokens(text, gold_spans)
        pred_words = _span_word_tokens(text, pred_spans)
        token_tp += len(gold_words & pred_words)
        token_fp += len(pred_words - gold_words)
        token_fn += len(gold_words - pred_words)

        if collect_errors > 0 and len(error_samples) < collect_errors and gold_set != pred_set:
            error_samples.append(
                {
                    "id": rec["id"],
                    "source": rec["source"],
                    "text": text,
                    "gold": [
                        {"start": s, "end": e, "label": l, "text": text[s:e]}
                        for s, e, l in sorted(gold_set)
                    ],
                    "predicted": [
                        {"start": s, "end": e, "label": l, "text": text[s:e]}
                        for s, e, l in sorted(pred_set)
                    ],
                }
            )

    total_tp = sum(tp_by_label.values())
    total_fp = sum(fp_by_label.values())
    total_fn = sum(fn_by_label.values())
    micro_p, micro_r, micro_f1 = _safe_prf(total_tp, total_fp, total_fn)

    per_entity: Dict[str, Dict[str, Any]] = {}
    unrounded_active_f1s: List[float] = []
    for lbl in ENTITY_LABELS:
        raw_p, raw_r, raw_f1 = _raw_prf(tp_by_label[lbl], fp_by_label[lbl], fn_by_label[lbl])
        per_entity[lbl] = {
            "precision": round(raw_p, 4),
            "recall": round(raw_r, 4),
            "f1": round(raw_f1, 4),
            "support": support_by_label[lbl],
            "predicted": pred_count_by_label[lbl],
        }
        if support_by_label[lbl] > 0:
            unrounded_active_f1s.append(raw_f1)

    macro_f1 = (
        round(sum(unrounded_active_f1s) / len(unrounded_active_f1s), 4)
        if unrounded_active_f1s
        else 0.0
    )

    # Compute partial boundary F1 from unrounded precision and recall
    raw_partial_p = overlap_pred_matched / total_pred_spans if total_pred_spans > 0 else 0.0
    raw_partial_r = overlap_gold_matched / total_gold_spans if total_gold_spans > 0 else 0.0
    partial_f1 = (
        round(2.0 * raw_partial_p * raw_partial_r / (raw_partial_p + raw_partial_r), 4)
        if (raw_partial_p + raw_partial_r) > 0
        else 0.0
    )

    _, _, word_token_f1 = _safe_prf(token_tp, token_fp, token_fn)

    out: Dict[str, Any] = {
        "n_docs": len(records),
        "n_gold_spans": total_gold_spans,
        "n_pred_spans": total_pred_spans,
        "precision": micro_p,
        "recall": micro_r,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "per_entity": per_entity,
        "partial_boundary_f1": partial_f1,
        "token_level_f1": word_token_f1,
    }
    if collect_errors > 0:
        out["error_samples"] = error_samples
    return out


def evaluate_predictions_across_subsets(
    records: Sequence[Dict[str, Any]],
    predict_fn: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
    subsets: Sequence[str] = EVAL_SUBSETS,
    collect_errors: int = 10,
    forward_latency_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run `predict_fn` once over `records`, measure latency (`ms/sample`), and
    compute metrics across `all`, `real`, `synthetic`, and each individual source.
    """
    t0 = time.perf_counter()
    preds_by_id: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        preds_by_id[rec["id"]] = predict_fn(rec)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    decode_latency_ms = round(elapsed_ms / max(1, len(records)), 4)

    report: Dict[str, Any] = {}
    for sub in subsets:
        sub_recs = filter_ner_subset(records, sub)
        sub_preds = [preds_by_id[r["id"]] for r in sub_recs]
        report[sub] = compute_ner_metrics(
            sub_recs,
            sub_preds,
            collect_errors=(collect_errors if sub == "all" else 0),
        )

    if forward_latency_ms is not None:
        report["forward_latency_ms_per_sample"] = round(float(forward_latency_ms), 4)
        report["decode_latency_ms_per_sample"] = decode_latency_ms
        report["latency_ms_per_sample"] = round(float(forward_latency_ms) + decode_latency_ms, 4)
    else:
        report["latency_ms_per_sample"] = decode_latency_ms
    return report


def load_cached_ner_seed_predictions(
    seed_dir: Path,
    split: str,
) -> Optional[Tuple[Dict[str, Dict[str, Any]], Optional[float]]]:
    """
    Load cached subword predictions (`{split}_predictions.jsonl`) and optional
    neural forward-pass latency from `train_meta.json` in a transformer seed directory.
    """
    pred_path = seed_dir / f"{split}_predictions.jsonl"
    if not pred_path.exists():
        return None
    mapping: Dict[str, Dict[str, Any]] = {}
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            mapping[row["id"]] = row

    forward_latency: Optional[float] = None
    meta_path = seed_dir / "train_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            split_meta = meta.get(f"{split}_metrics") or meta.get("val_metrics") or {}
            if "latency_ms_per_sample" in split_meta:
                forward_latency = float(split_meta["latency_ms_per_sample"])
        except Exception:
            forward_latency = None

    return mapping, forward_latency


def decode_cached_prediction_row(
    rec: Dict[str, Any],
    cached_row: Dict[str, Any],
    use_postprocessing: bool,
) -> List[Dict[str, Any]]:
    """
    Decode a cached transformer seed prediction row under either:
    - `use_postprocessing=False` (Raw Decoding: E2 / E4a)
    - `use_postprocessing=True` (Mid-word WordPiece `##` continuation repair + Cased Civic
      Gazetteer & 3-Gate Merger: E3 / E4b, without noisy standalone `O -> I-X` promotion).
    """
    text = rec["text"]
    if "offset_mapping" in cached_row and "tag_ids" in cached_row:
        offsets = [tuple(x) for x in cached_row["offset_mapping"]]
        tag_ids = [int(x) for x in cached_row["tag_ids"]]
        tokens = cached_row.get("tokens")
        spans = decode_bio2_to_spans(
            text,
            offsets,
            tag_ids,
            tokens=tokens,
            repair_subwords=use_postprocessing,
            promote_orphan_i=False,
        )
    else:
        spans = list(cached_row.get("raw_spans") or [])

    if use_postprocessing:
        return apply_civic_postprocessing(text, spans, inject_gazetteer=True)
    return spans


def aggregate_seed_reports(seed_reports: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate `mean ± std` across multiple seed evaluation reports."""
    if not seed_reports:
        return {}

    scalar_keys = (
        "precision",
        "recall",
        "micro_f1",
        "macro_f1",
        "partial_boundary_f1",
        "token_level_f1",
    )
    aggregated: Dict[str, Any] = {}

    for sub in EVAL_SUBSETS:
        if sub not in seed_reports[0]:
            continue
        sub_agg: Dict[str, Any] = {
            "n_docs": seed_reports[0][sub]["n_docs"],
            "n_gold_spans": seed_reports[0][sub]["n_gold_spans"],
        }
        for mk in scalar_keys:
            vals = [float(rep[sub][mk]) for rep in seed_reports if rep[sub].get(mk) is not None]
            sub_agg[mk] = {
                "mean": round(statistics.mean(vals), 4),
                "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
            }
        per_ent_agg: Dict[str, Any] = {}
        for lbl in ENTITY_LABELS:
            f1_vals = [float(rep[sub]["per_entity"][lbl]["f1"]) for rep in seed_reports]
            per_ent_agg[lbl] = {
                "f1": {
                    "mean": round(statistics.mean(f1_vals), 4),
                    "std": round(statistics.stdev(f1_vals), 4) if len(f1_vals) > 1 else 0.0,
                },
                "support": seed_reports[0][sub]["per_entity"][lbl]["support"],
            }
        sub_agg["per_entity"] = per_ent_agg
        aggregated[sub] = sub_agg

    for lat_key in (
        "latency_ms_per_sample",
        "forward_latency_ms_per_sample",
        "decode_latency_ms_per_sample",
    ):
        lat_vals = [float(rep[lat_key]) for rep in seed_reports if lat_key in rep]
        if lat_vals:
            aggregated[lat_key] = {
                "mean": round(statistics.mean(lat_vals), 4),
                "std": round(statistics.stdev(lat_vals), 4) if len(lat_vals) > 1 else 0.0,
            }
    return aggregated


def format_metric(val: Any) -> str:
    if val is None:
        return "-"
    if isinstance(val, dict):
        return f"{val['mean']:.4f} ± {val['std']:.4f}"
    return f"{float(val):.4f}"


def print_ner_eval_table(report: Dict[str, Any], split: str) -> None:
    """Print formatted Markdown tables for Task 2 NER configurations."""
    configs = report["configurations"]
    print(f"\n=== TASK 2 NER EVALUATION SUMMARY ({split.upper()} · STRICT SPAN MICRO-F1 / MACRO-F1) ===")
    print("| Config | All (Micro / Macro F1) | Real (Micro / Macro F1) | Synthetic (Micro / Macro F1) | Partial-Boundary F1 (All / Syn) |")
    print("| :--- | :---: | :---: | :---: | :---: |")

    for name, cfg in configs.items():
        summary = cfg.get("aggregated", cfg.get("single_run", {}))
        all_s = summary.get("all", {})
        real_s = summary.get("real", {})
        syn_s = summary.get("synthetic", {})
        print(
            f"| {name} | "
            f"{format_metric(all_s.get('micro_f1'))} / {format_metric(all_s.get('macro_f1'))} | "
            f"{format_metric(real_s.get('micro_f1'))} / {format_metric(real_s.get('macro_f1'))} | "
            f"{format_metric(syn_s.get('micro_f1'))} / {format_metric(syn_s.get('macro_f1'))} | "
            f"{format_metric(all_s.get('partial_boundary_f1'))} / {format_metric(syn_s.get('partial_boundary_f1'))} |"
        )

    print(f"\n=== PER-ENTITY F1 ({split.upper()} · ALL) & PER-SOURCE MICRO-F1 ===")
    print("| Config | LOCATION F1 | ORGANIZATION F1 | PERSON F1 | nergrit | nerui | idner_2k | synthetic |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for name, cfg in configs.items():
        summary = cfg.get("aggregated", cfg.get("single_run", {}))
        pe = summary.get("all", {}).get("per_entity", {})
        print(
            f"| {name} | "
            f"{format_metric(pe.get('LOCATION', {}).get('f1'))} | "
            f"{format_metric(pe.get('ORGANIZATION', {}).get('f1'))} | "
            f"{format_metric(pe.get('PERSON', {}).get('f1'))} | "
            f"{format_metric(summary.get('indonlu_nergrit', {}).get('micro_f1'))} | "
            f"{format_metric(summary.get('indolem_nerui', {}).get('micro_f1'))} | "
            f"{format_metric(summary.get('idner_news_2k', {}).get('micro_f1'))} | "
            f"{format_metric(summary.get('synthetic', {}).get('micro_f1'))} |"
        )


def run_evaluation(
    data_path: Path = DEFAULT_MASTER_PATH,
    models_dir: Path = DEFAULT_MODELS_DIR,
    output_path: Path = DEFAULT_EVAL_OUTPUT,
    split: str = "test",
) -> Dict[str, Any]:
    """
    Evaluate E1 (CPU Baseline) and all available cached transformer seed groups
    (`ner_full` -> E2/E3; `ner_real_only` -> E4a/E4b) on `split`.
    """
    from src.task2.train_ner_baseline import load_or_fit_baseline_model, predict_baseline_spans

    records = load_ner_records(data_path=data_path, split=split, subset="all")
    baseline_model = load_or_fit_baseline_model(
        data_path=data_path,
        model_dir=models_dir / "ner_baseline",
    )

    e1_metrics = evaluate_predictions_across_subsets(
        records,
        lambda r: predict_baseline_spans(r["text"], baseline_model, use_postprocessing=True),
    )
    configurations: Dict[str, Any] = {
        "E1_Baseline_Civic_BIO": {"single_run": e1_metrics}
    }

    group_specs = [
        (
            models_dir / "ner_full",
            [
                ("E2_IndoBERT_Full_Raw", False),
                ("E3_IndoBERT_Full_Civic_Hybrid", True),
            ],
        ),
        (
            models_dir / "ner_real_only",
            [
                ("E4a_Ablation_RealOnly_Raw", False),
                ("E4b_Ablation_RealOnly_Civic_Hybrid", True),
            ],
        ),
    ]

    found_checkpoints: Dict[str, List[str]] = {}
    for group_dir, variants in group_specs:
        if not group_dir.exists():
            continue
        seed_dirs = sorted([d for d in group_dir.glob("seed_*") if d.is_dir()], key=lambda p: p.name)
        loaded_seeds: List[Tuple[str, Dict[str, Dict[str, Any]], Optional[float]]] = []
        for sd in seed_dirs:
            cached_res = load_cached_ner_seed_predictions(sd, split)
            if cached_res is not None:
                pred_map, fwd_lat = cached_res
                loaded_seeds.append((sd.name, pred_map, fwd_lat))
        if not loaded_seeds:
            continue

        found_checkpoints[group_dir.name] = [s for s, _, _ in loaded_seeds]
        for cfg_name, use_post in variants:
            per_seed: Dict[str, Any] = {}
            seed_list: List[Dict[str, Any]] = []
            for seed_name, pred_map, fwd_lat in loaded_seeds:
                def _pred_fn(
                    rec: Dict[str, Any],
                    pm: Dict[str, Dict[str, Any]] = pred_map,
                    up: bool = use_post,
                    sn: str = seed_name,
                ) -> List[Dict[str, Any]]:
                    rid = rec["id"]
                    if rid not in pm:
                        raise KeyError(
                            f"Missing cached prediction for id={rid!r} in seed={sn!r} (split={split!r})"
                        )
                    return decode_cached_prediction_row(rec, pm[rid], use_postprocessing=up)

                rep = evaluate_predictions_across_subsets(
                    records,
                    _pred_fn,
                    forward_latency_ms=fwd_lat,
                )
                per_seed[seed_name] = rep
                seed_list.append(rep)

            configurations[cfg_name] = {
                "seeds": per_seed,
                "aggregated": aggregate_seed_reports(seed_list),
            }

    report: Dict[str, Any] = {
        "split": split,
        "total_eval_rows": len(records),
        "found_seed_checkpoints": found_checkpoints,
        "configurations": configurations,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Task 2 NER configurations E1-E4.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_MASTER_PATH)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_EVAL_OUTPUT)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    args = parser.parse_args()

    report = run_evaluation(
        data_path=args.data_path,
        models_dir=args.models_dir,
        output_path=args.output_path,
        split=args.split,
    )
    print_ner_eval_table(report, args.split)
    if not report["found_seed_checkpoints"]:
        print(
            "\nNote: No cached IndoBERT seed checkpoints found in "
            f"{args.models_dir}. Run `python -m src.task2.train_ner_transformer` "
            "to unlock E2, E3, E4a, and E4b evaluations."
        )
    print(f"\nSaved full evaluation report -> {args.output_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.priority.engine import (
    DEFAULT_WEIGHTS_PATH,
    PriorityConfig,
    ROOT_DIR,
    compute_urgency,
    normalize_urgency_label,
)
from src.schema import T1Result, T3Result


DEFAULT_DATA_PATH = ROOT_DIR / "data" / "processed" / "master_dataset.jsonl"

ORACLE_SENTIMENT_MAP: Dict[str, Tuple[str, float]] = {
    "Negatif": ("negative", 0.90),
    "Netral": ("neutral", 0.40),
    "Positif": ("positive", 0.05),
}


def load_val_priority_records(data_path: Path = DEFAULT_DATA_PATH) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("split") != "val":
                continue
            if (
                row.get("domain") is None
                or row.get("intent") is None
                or row.get("sentiment") is None
                or row.get("is_sarcasm") is None
                or row.get("urgency_level") is None
            ):
                continue
            records.append(row)
    return records


def score_config(records: List[Dict[str, Any]], cfg: PriorityConfig) -> Dict[str, Any]:
    tp = fp = fn = tn = 0
    em_tp = em_fn = 0
    for rec in records:
        t1 = T1Result(domain=rec["domain"], intent=rec["intent"])
        sent_lbl, sent_score = ORACLE_SENTIMENT_MAP[rec["sentiment"]]
        t3 = T3Result(
            sentiment=sent_lbl,
            sarcasm=bool(rec["is_sarcasm"]),
            sentiment_score=sent_score,
        )
        out = compute_urgency(t1, t3, config=cfg, ablation="full")
        gold_level = normalize_urgency_label(rec["urgency_level"])
        gold_crit = gold_level in ("High", "Emergency")
        pred_crit = out.escalated

        if pred_crit and gold_crit:
            tp += 1
        elif pred_crit and not gold_crit:
            fp += 1
        elif not pred_crit and gold_crit:
            fn += 1
        else:
            tn += 1

        if gold_level == "Emergency":
            if out.priority_level == "Emergency":
                em_tp += 1
            else:
                em_fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    em_recall = em_tp / (em_tp + em_fn) if (em_tp + em_fn) > 0 else 0.0
    return {
        "escalation_precision": round(precision, 4),
        "critical_recall": round(recall, 4),
        "critical_f1": round(f1, 4),
        "emergency_recall": round(em_recall, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def calibrate_weights(
    data_path: Path = DEFAULT_DATA_PATH,
    output_path: Path = DEFAULT_WEIGHTS_PATH,
) -> Dict[str, Any]:
    val_records = load_val_priority_records(data_path)
    if not val_records:
        raise ValueError(f"No validation records with complete priority labels found in {data_path}")

    weight_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    alpha_grid = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3]
    threshold_grid = [6.5, 7.0, 7.5]

    best_overall: Tuple[Tuple[float, float, float], PriorityConfig, Dict[str, Any]] | None = None
    best_fixed_75: Tuple[Tuple[float, float], PriorityConfig, Dict[str, Any]] | None = None

    for w1 in weight_grid:
        for w2 in weight_grid:
            w3 = round(1.0 - w1 - w2, 4)
            if w3 < 0.1 or w3 > 0.6:
                continue
            for alpha in alpha_grid:
                for thr in threshold_grid:
                    cfg = PriorityConfig(
                        w1=w1,
                        w2=w2,
                        w3=w3,
                        alpha=alpha,
                        escalation_threshold=thr,
                        emergency_threshold=max(thr + 0.8, 8.50),
                    )
                    metrics = score_config(val_records, cfg)
                    rank_key = (
                        metrics["critical_f1"],
                        metrics["emergency_recall"],
                        metrics["critical_recall"],
                        -abs(thr - 7.50),
                        -abs(alpha - 0.20),
                    )
                    if best_overall is None or rank_key > best_overall[0]:
                        best_overall = (rank_key, cfg, metrics)
                    if thr == 7.5:
                        fixed_key = (
                            metrics["critical_f1"],
                            metrics["emergency_recall"],
                            metrics["critical_recall"],
                            -abs(alpha - 0.20),
                        )
                        if best_fixed_75 is None or fixed_key > best_fixed_75[0]:
                            best_fixed_75 = (fixed_key, cfg, metrics)

    assert best_overall is not None and best_fixed_75 is not None
    report = {
        "val_samples": len(val_records),
        "best_config": best_overall[1].to_dict(),
        "best_val_metrics": best_overall[2],
        "fixed_750_config": best_fixed_75[1].to_dict(),
        "fixed_750_val_metrics": best_fixed_75[2],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate Priority Engine weights on validation split.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_WEIGHTS_PATH)
    args = parser.parse_args()

    report = calibrate_weights(args.data_path, args.output_path)
    print(f"Calibrated on {report['val_samples']} validation rows.")
    print("Best config:", {k: report["best_config"][k] for k in ("w1", "w2", "w3", "alpha", "escalation_threshold")})
    print("Best val metrics:", report["best_val_metrics"])
    print("Fixed U>7.50 config:", {k: report["fixed_750_config"][k] for k in ("w1", "w2", "w3", "alpha", "escalation_threshold")})
    print("Fixed U>7.50 metrics:", report["fixed_750_val_metrics"])
    print(f"Saved calibrated priority weights -> {args.output_path}")


if __name__ == "__main__":
    main()

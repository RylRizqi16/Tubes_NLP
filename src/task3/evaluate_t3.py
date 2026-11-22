from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.task3.data_loader import (
    DEFAULT_MASTER_PATH,
    ID_TO_SENTIMENT,
    ROOT_DIR,
    filter_subset,
    load_t3_records,
)
from src.task3.rules import apply_rule_engine, predict_e1


DEFAULT_MODELS_DIR = ROOT_DIR / "models" / "task3"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "outputs" / "task3_eval_report.json"

SENTIMENT_LABELS = ["Negatif", "Netral", "Positif"]
EVAL_SUBSETS = ["all", "real", "synthetic", "negation", "code_mixing", "sarcasm_irony"]


def compute_t3_metrics(
    records: Sequence[Dict[str, Any]],
    preds: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    sent_true: List[str] = []
    sent_pred: List[str] = []
    sarc_true: List[int] = []
    sarc_pred: List[int] = []

    for rec, pred in zip(records, preds):
        if rec.get("sentiment") is not None and pred.get("sentiment") is not None:
            sent_true.append(rec["sentiment"])
            sent_pred.append(pred["sentiment"])
        if rec.get("is_sarcasm") is not None and pred.get("is_sarcasm") is not None:
            sarc_true.append(int(rec["is_sarcasm"]))
            sarc_pred.append(int(pred["is_sarcasm"]))

    result: Dict[str, Any] = {
        "n_total": len(records),
        "n_sentiment": len(sent_true),
        "n_sarcasm": len(sarc_true),
        "sentiment_accuracy": None,
        "sentiment_macro_f1": None,
        "sentiment_per_class_f1": None,
        "sentiment_confusion_matrix": None,
        "sarcasm_accuracy": None,
        "sarcasm_f1": None,
        "sarcasm_macro_f1": None,
    }

    if sent_true:
        per_class = f1_score(
            sent_true,
            sent_pred,
            labels=SENTIMENT_LABELS,
            average=None,
            zero_division=0,
        )
        cm = confusion_matrix(sent_true, sent_pred, labels=SENTIMENT_LABELS)
        result["sentiment_accuracy"] = round(float(accuracy_score(sent_true, sent_pred)), 4)
        result["sentiment_macro_f1"] = round(
            float(f1_score(sent_true, sent_pred, labels=SENTIMENT_LABELS, average="macro", zero_division=0)),
            4,
        )
        result["sentiment_per_class_f1"] = {
            lbl: round(float(val), 4) for lbl, val in zip(SENTIMENT_LABELS, per_class)
        }
        result["sentiment_confusion_matrix"] = cm.tolist()

    if sarc_true:
        result["sarcasm_accuracy"] = round(float(accuracy_score(sarc_true, sarc_pred)), 4)
        result["sarcasm_f1"] = round(
            float(f1_score(sarc_true, sarc_pred, pos_label=1, average="binary", zero_division=0)),
            4,
        )
        result["sarcasm_macro_f1"] = round(
            float(f1_score(sarc_true, sarc_pred, labels=[0, 1], average="macro", zero_division=0)),
            4,
        )

    return result


def evaluate_predictor_across_subsets(
    records: List[Dict[str, Any]],
    predict_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    subsets: Sequence[str] = EVAL_SUBSETS,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    preds_by_id: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        preds_by_id[rec["id"]] = predict_fn(rec)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    latency_ms = round(elapsed_ms / max(1, len(records)), 4)

    subset_metrics: Dict[str, Any] = {}
    for sub in subsets:
        sub_recs = filter_subset(records, sub)
        sub_preds = [preds_by_id[r["id"]] for r in sub_recs]
        subset_metrics[sub] = compute_t3_metrics(sub_recs, sub_preds)
    subset_metrics["latency_ms_per_sample"] = latency_ms
    return subset_metrics


def load_cached_seed_predictions(seed_dir: Path, split: str) -> Optional[Dict[str, Dict[str, Any]]]:
    pred_file = seed_dir / f"{split}_predictions.jsonl"
    if not pred_file.exists():
        return None
    mapping: Dict[str, Dict[str, Any]] = {}
    with pred_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            mapping[row["id"]] = row
    return mapping


def aggregate_seed_reports(seed_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not seed_reports:
        return {}
    aggregated: Dict[str, Any] = {}
    metric_keys = ["sentiment_accuracy", "sentiment_macro_f1", "sarcasm_f1", "sarcasm_accuracy"]
    for sub in EVAL_SUBSETS:
        sub_agg: Dict[str, Any] = {
            "n_total": seed_reports[0][sub]["n_total"],
            "n_sentiment": seed_reports[0][sub]["n_sentiment"],
            "n_sarcasm": seed_reports[0][sub]["n_sarcasm"],
        }
        for mk in metric_keys:
            vals = [rep[sub][mk] for rep in seed_reports if rep[sub].get(mk) is not None]
            if not vals:
                sub_agg[mk] = None
                continue
            mean_val = round(statistics.mean(vals), 4)
            std_val = round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0
            sub_agg[mk] = {"mean": mean_val, "std": std_val}
        aggregated[sub] = sub_agg
    lat_vals = [rep.get("latency_ms_per_sample", 0.0) for rep in seed_reports]
    aggregated["latency_ms_per_sample"] = {
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


def print_eval_table(report: Dict[str, Any], split: str) -> None:
    configs = report["configurations"]
    print(f"\n=== TASK 3 EVALUATION SUMMARY ({split.upper()}) ===")
    header = (
        "| Config | Test All (Sent Acc / Macro-F1 / Sarc F1) | "
        "Test Real (Sent Acc / Macro-F1 / Sarc F1) | "
        "Test Synthetic (Sent Acc / Macro-F1 / Sarc F1) |"
    )
    print(header)
    print("| :--- | :---: | :---: | :---: |")

    for name, cfg in configs.items():
        summary = cfg.get("aggregated", cfg.get("single_run", {}))
        cols: List[str] = []
        for sub in ("all", "real", "synthetic"):
            s = summary.get(sub, {})
            cols.append(
                f"{format_metric(s.get('sentiment_accuracy'))} / "
                f"{format_metric(s.get('sentiment_macro_f1'))} / "
                f"{format_metric(s.get('sarcasm_f1'))}"
            )
        print(f"| {name} | {cols[0]} | {cols[1]} | {cols[2]} |")

    print("\n=== HARD LINGUISTIC SUBSETS (Sent Macro-F1 / Sarc F1) ===")
    print("| Config | Negation | Code-Mixing | Sarcasm / Irony |")
    print("| :--- | :---: | :---: | :---: |")
    for name, cfg in configs.items():
        summary = cfg.get("aggregated", cfg.get("single_run", {}))
        cols = []
        for sub in ("negation", "code_mixing", "sarcasm_irony"):
            s = summary.get(sub, {})
            cols.append(
                f"{format_metric(s.get('sentiment_macro_f1'))} / "
                f"{format_metric(s.get('sarcasm_f1'))}"
            )
        print(f"| {name} | {cols[0]} | {cols[1]} | {cols[2]} |")


def run_evaluation(
    data_path: Path = DEFAULT_MASTER_PATH,
    models_dir: Path = DEFAULT_MODELS_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    split: str = "test",
) -> Dict[str, Any]:
    records = load_t3_records(data_path=data_path, split=split, subset="all")

    e1_metrics = evaluate_predictor_across_subsets(records, lambda r: predict_e1(r["text"]))
    configurations: Dict[str, Any] = {
        "E1_Baseline_Lexicon_Rules": {"single_run": e1_metrics}
    }

    seed_dirs = sorted(
        [d for d in models_dir.glob("seed_*") if d.is_dir()],
        key=lambda p: p.name,
    )
    loaded_seeds: List[tuple[str, Dict[str, Dict[str, Any]]]] = []
    for sd in seed_dirs:
        preds = load_cached_seed_predictions(sd, split)
        if preds is not None:
            loaded_seeds.append((sd.name, preds))

    if loaded_seeds:
        ablation_configs = {
            "E2_IndoBERTweet_Only": {"use_negation": False, "use_sarcasm": False},
            "E3_IndoBERTweet_Full_Rules": {"use_negation": True, "use_sarcasm": True},
            "E4a_Ablation_No_Negation": {"use_negation": False, "use_sarcasm": True},
            "E4b_Ablation_No_Sarcasm": {"use_negation": True, "use_sarcasm": False},
        }
        for cfg_name, flags in ablation_configs.items():
            per_seed: Dict[str, Any] = {}
            seed_list: List[Dict[str, Any]] = []
            for seed_name, pred_map in loaded_seeds:
                def _predict_row(rec: Dict[str, Any], pm: Dict[str, Dict[str, Any]] = pred_map, fl: Dict[str, bool] = flags) -> Dict[str, Any]:
                    raw = pm[rec["id"]]
                    if not fl["use_negation"] and not fl["use_sarcasm"]:
                        probs = raw["sent_probs"]
                        sarc_prob = float(raw["sarc_prob"])
                        sent_id = max(range(3), key=lambda i: probs[i])
                        return {
                            "sentiment": ID_TO_SENTIMENT[sent_id],
                            "is_sarcasm": 1 if sarc_prob >= 0.5 else 0,
                        }
                    return apply_rule_engine(
                        rec["text"],
                        sent_probs=raw["sent_probs"],
                        sarc_prob=raw["sarc_prob"],
                        use_negation=fl["use_negation"],
                        use_sarcasm=fl["use_sarcasm"],
                    )

                rep = evaluate_predictor_across_subsets(records, _predict_row)
                per_seed[seed_name] = rep
                seed_list.append(rep)

            configurations[cfg_name] = {
                "seeds": per_seed,
                "aggregated": aggregate_seed_reports(seed_list),
            }

    report: Dict[str, Any] = {
        "split": split,
        "total_eval_rows": len(records),
        "found_seed_checkpoints": [name for name, _ in loaded_seeds],
        "configurations": configurations,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Task 3 configurations E1-E4.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_MASTER_PATH)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    args = parser.parse_args()

    report = run_evaluation(
        data_path=args.data_path,
        models_dir=args.models_dir,
        output_path=args.output_path,
        split=args.split,
    )
    print_eval_table(report, args.split)
    if not report["found_seed_checkpoints"]:
        print(
            "\nNote: No trained IndoBERTweet checkpoints found in "
            f"{args.models_dir}. Run `python -m src.task3.train_indobertweet` "
            "to unlock E2, E3, and E4 evaluations."
        )
    print(f"\nSaved full evaluation report -> {args.output_path}")


if __name__ == "__main__":
    main()

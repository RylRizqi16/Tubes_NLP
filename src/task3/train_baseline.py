from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from src.task3.data_loader import DEFAULT_MASTER_PATH, ROOT_DIR, load_t3_records, summarize_t3_records
from src.task3.evaluate_t3 import evaluate_predictor_across_subsets
from src.task3.rules import predict_e1


DEFAULT_BASELINE_OUTPUT = ROOT_DIR / "outputs" / "task3_e1_baseline.json"


def run_baseline(data_path: Path, output_path: Path) -> Dict[str, Any]:
    val_records = load_t3_records(data_path=data_path, split="val", subset="all")
    test_records = load_t3_records(data_path=data_path, split="test", subset="all")

    val_metrics = evaluate_predictor_across_subsets(val_records, lambda r: predict_e1(r["text"]))
    test_metrics = evaluate_predictor_across_subsets(test_records, lambda r: predict_e1(r["text"]))

    report = {
        "model": "E1_Baseline_Lexicon_Rules",
        "val_summary": summarize_t3_records(val_records),
        "test_summary": summarize_t3_records(test_records),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 3 E1 Baseline evaluation.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_MASTER_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_BASELINE_OUTPUT)
    args = parser.parse_args()

    report = run_baseline(args.data_path, args.output_path)
    print("=== E1 BASELINE (LEXICON + RULES) ===")
    for split_name in ("val_metrics", "test_metrics"):
        print(f"\n[{split_name}] latency={report[split_name]['latency_ms_per_sample']} ms/sample")
        for sub in ("all", "real", "synthetic", "negation", "code_mixing", "sarcasm_irony"):
            m = report[split_name][sub]
            print(
                f"  {sub:<14} n={m['n_total']:>4} | "
                f"sent_acc={m['sentiment_accuracy']} | "
                f"sent_f1={m['sentiment_macro_f1']} | "
                f"sarc_f1={m['sarcasm_f1']}"
            )
    print(f"\nSaved E1 baseline report -> {args.output_path}")


if __name__ == "__main__":
    main()

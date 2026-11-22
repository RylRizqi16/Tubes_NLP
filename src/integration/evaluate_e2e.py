from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from src.integration.pipeline import CivicComplaintPipeline
from src.priority.calibrate import ORACLE_SENTIMENT_MAP
from src.priority.engine import (
    ABLATION_MODES,
    PriorityConfig,
    ROOT_DIR,
    compute_urgency,
    normalize_urgency_label,
)
from src.schema import T1Result, T3Result
from src.task1.evaluate import evaluate as evaluate_t1
from src.task2.evaluate_ner import compute_ner_metrics
from src.task3.evaluate_t3 import compute_t3_metrics


DEFAULT_DATA_PATH = ROOT_DIR / "data" / "processed" / "master_dataset.jsonl"
DEFAULT_REPORT_PATH = ROOT_DIR / "outputs" / "e2e_eval_report.json"
DEFAULT_BEAM_SAMPLES_PATH = ROOT_DIR / "outputs" / "sample_beam_events.jsonl"

SCHEMA_TO_SENTIMENT: Dict[str, str] = {
    "negative": "Negatif",
    "neutral": "Netral",
    "positive": "Positif",
}


def load_test_records(data_path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("split") == "test":
                records.append(row)
    return records


def compute_priority_metrics(
    records: Sequence[Dict[str, Any]],
    escalated_flags: Sequence[bool],
    pred_levels: Sequence[str],
    collect_missed: int = 10,
) -> Dict[str, Any]:
    tp = fp = fn = tn = 0
    em_tp = em_fn = 0
    missed_samples: List[Dict[str, Any]] = []

    for rec, pred_esc, pred_lvl in zip(records, escalated_flags, pred_levels):
        raw_urg = rec.get("urgency_level")
        if raw_urg is None:
            continue
        gold_lvl = normalize_urgency_label(raw_urg)
        gold_crit = gold_lvl in ("High", "Emergency")

        if pred_esc and gold_crit:
            tp += 1
        elif pred_esc and not gold_crit:
            fp += 1
        elif not pred_esc and gold_crit:
            fn += 1
            if len(missed_samples) < collect_missed:
                missed_samples.append(
                    {
                        "id": rec["id"],
                        "text": rec["text"],
                        "gold_urgency": gold_lvl,
                        "pred_urgency": pred_lvl,
                        "gold_domain": rec.get("domain"),
                        "gold_intent": rec.get("intent"),
                    }
                )
        else:
            tn += 1

        if gold_lvl == "Emergency":
            if pred_lvl == "Emergency":
                em_tp += 1
            else:
                em_fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    em_recall = em_tp / (em_tp + em_fn) if (em_tp + em_fn) > 0 else 0.0

    return {
        "n_urgency_labeled": tp + fp + fn + tn,
        "escalation_precision": round(precision, 4),
        "critical_recall": round(recall, 4),
        "critical_f1": round(f1, 4),
        "emergency_recall": round(em_recall, 4),
        "missed_critical_count": fn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "missed_critical_samples": missed_samples,
    }


def run_priority_ablation(
    records: Sequence[Dict[str, Any]],
    t1_inputs: Sequence[T1Result],
    t3_inputs: Sequence[T3Result],
    config: PriorityConfig,
) -> Dict[str, Any]:
    ablation_report: Dict[str, Any] = {}
    for mode in ABLATION_MODES:
        esc_flags: List[bool] = []
        pred_lvls: List[str] = []
        urg_scores: List[float] = []
        for t1_r, t3_r in zip(t1_inputs, t3_inputs):
            po = compute_urgency(t1_r, t3_r, config=config, ablation=mode)
            esc_flags.append(po.escalated)
            pred_lvls.append(po.priority_level)
            urg_scores.append(po.urgency_index)
        metrics = compute_priority_metrics(records, esc_flags, pred_lvls, collect_missed=0)
        metrics.pop("missed_critical_samples", None)
        metrics["mean_urgency_index"] = round(statistics.mean(urg_scores), 4) if urg_scores else 0.0
        ablation_report[mode] = metrics
    return ablation_report


def summarize_latencies(timing_rows: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if not timing_rows:
        return {}
    keys = list(timing_rows[0].keys())
    summary: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = [row[k] for row in timing_rows]
        summary[k] = {
            "mean_ms": round(float(np.mean(vals)), 4),
            "p95_ms": round(float(np.percentile(vals, 95)), 4),
        }
    return summary


def select_beam_samples(beam_events: Sequence[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
    emergency = [e for e in beam_events if e["why"]["priority_level"] == "Emergency"]
    sarcastic = [e for e in beam_events if e["why"]["sarcasm_triggered"] and e not in emergency]
    others = [e for e in beam_events if e not in emergency and e not in sarcastic]
    selected = emergency[:4] + sarcastic[:3] + others[:limit]
    return selected[:limit]


def print_markdown_summary(report: Dict[str, Any]) -> None:
    syn = report["synthetic_e2e"]
    real = report["real_test"]
    print("\n=== END-TO-END PIPELINE EVALUATION SUMMARY ===")
    print("Backends:", report["backends"])
    print("Priority Config:", {k: report["priority_config"][k] for k in ("w1", "w2", "w3", "alpha", "escalation_threshold")})

    print("\n--- Synthetic Test (n={}) Component & Routing Metrics ---".format(syn["n_records"]))
    print(
        f"  T1 Domain Macro-F1 : {syn['components']['t1']['domain']['macro_f1']} | "
        f"Intent Accuracy : {syn['components']['t1']['intent']['accuracy']}"
    )
    print(
        f"  T2 Strict NER F1   : {syn['components']['t2_ner']['micro_f1']} "
        f"(P={syn['components']['t2_ner']['precision']}, R={syn['components']['t2_ner']['recall']}, "
        f"Macro-F1={syn['components']['t2_ner']['macro_f1']})"
    )
    print(
        f"  T3 Sent Acc        : {syn['components']['t3']['sentiment_accuracy']} | "
        f"Sent Macro-F1 : {syn['components']['t3']['sentiment_macro_f1']} | "
        f"Sarcasm F1 : {syn['components']['t3']['sarcasm_f1']}"
    )
    print(
        f"  Routing Coverage   : {syn['routing']['routing_coverage']} | "
        f"NER Org Routing : {syn['routing']['ner_org_routing_rate']}"
    )
    print(f"  Total Latency Mean : {syn['latency_ms']['total']['mean_ms']} ms (p95={syn['latency_ms']['total']['p95_ms']} ms)")

    print("\n--- Priority Ablation (Synthetic Test - Live Pipeline vs Oracle) ---")
    print("| Mode | Live Esc Prec | Live Crit Rec | Live Crit F1 | Live Emer Rec | Oracle Crit F1 | Oracle Emer Rec |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for mode in ABLATION_MODES:
        lv = syn["priority_ablation_live"][mode]
        orc = syn["priority_ablation_oracle"][mode]
        print(
            f"| {mode} | {lv['escalation_precision']:.4f} | {lv['critical_recall']:.4f} | "
            f"{lv['critical_f1']:.4f} | {lv['emergency_recall']:.4f} | "
            f"{orc['critical_f1']:.4f} | {orc['emergency_recall']:.4f} |"
        )

    print("\n--- Real Civic Test (n={}, sources={}) ---".format(real["n_records"], real["sources"]))
    print(
        f"  Escalation Rate    : {real['escalation_rate']} | "
        f"NER Org Routing : {real['ner_org_routing_rate']} | "
        f"Latency Mean : {real['latency_ms']['total']['mean_ms']} ms"
    )
    if real.get("priority_vs_campus_urgency"):
        pm = real["priority_vs_campus_urgency"]
        print(
            f"  Campus Urgency     : Esc Prec={pm['escalation_precision']} | "
            f"Crit Rec={pm['critical_recall']} | Crit F1={pm['critical_f1']} (n={pm['n_urgency_labeled']})"
        )
    if real.get("t3_metrics"):
        tm = real["t3_metrics"]
        print(
            f"  Real T3 Metrics    : Sent Acc={tm['sentiment_accuracy']} | "
            f"Sent Macro-F1={tm['sentiment_macro_f1']} | Sarcasm F1={tm['sarcasm_f1']}"
        )


def evaluate_e2e(
    data_path: Path = DEFAULT_DATA_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    beam_samples_path: Path = DEFAULT_BEAM_SAMPLES_PATH,
    t1_checkpoint: Path | None = None,
    t2_checkpoint: Path | None = None,
    t2_lda_dir: Path | None = None,
    t3_checkpoint: Path | None = None,
    priority_config_path: Path | None = None,
    real_max_samples: int = 300,
) -> Dict[str, Any]:
    pipeline = CivicComplaintPipeline(
        t1_checkpoint=t1_checkpoint,
        t2_checkpoint=t2_checkpoint,
        t2_lda_dir=t2_lda_dir,
        t3_checkpoint=t3_checkpoint,
        priority_config_path=priority_config_path,
    )

    test_records = load_test_records(data_path)
    syn_records = [r for r in test_records if r.get("source") == "synthetic"]
    real_records = [
        r for r in test_records
        if r.get("source") in ("sp4n_lapor", "campus_complaints")
    ][:real_max_samples]

    syn_t1_preds: List[Dict[str, str]] = []
    syn_t1_objs: List[T1Result] = []
    syn_t2_spans: List[List[Dict[str, Any]]] = []
    syn_t3_preds: List[Dict[str, Any]] = []
    syn_t3_objs: List[T3Result] = []
    syn_beams: List[Dict[str, Any]] = []
    syn_timings: List[Dict[str, float]] = []
    syn_esc_flags: List[bool] = []
    syn_pred_lvls: List[str] = []

    oracle_t1_objs: List[T1Result] = []
    oracle_t3_objs: List[T3Result] = []

    for rec in syn_records:
        channel = "laporgub" if "semarang" in rec["text"].lower() else "sp4n"
        out = pipeline.process(rec["text"], channel=channel, reporter_id=rec["id"])
        ce = out["complaint_event"]
        be = out["beam_event"]

        syn_beams.append(be)
        syn_timings.append(out["timings_ms"])
        syn_t1_preds.append({"domain": ce["domain"], "intent": ce["intent"]})
        syn_t1_objs.append(T1Result(domain=ce["domain"], intent=ce["intent"]))
        syn_t2_spans.append(ce["entities"])
        syn_t3_preds.append(
            {
                "sentiment": SCHEMA_TO_SENTIMENT[ce["sentiment"]],
                "is_sarcasm": 1 if ce["sarcasm"] else 0,
            }
        )
        syn_t3_objs.append(
            T3Result(
                sentiment=ce["sentiment"],
                sarcasm=ce["sarcasm"],
                sentiment_score=be["how_many"]["sentiment_score"],
            )
        )
        syn_esc_flags.append(bool(be["why"]["escalated"]))
        syn_pred_lvls.append(str(be["why"]["priority_level"]))

        oracle_t1_objs.append(T1Result(domain=rec["domain"], intent=rec["intent"]))
        s_lbl, s_val = ORACLE_SENTIMENT_MAP[rec["sentiment"]]
        oracle_t3_objs.append(
            T3Result(
                sentiment=s_lbl,
                sarcasm=bool(rec["is_sarcasm"]),
                sentiment_score=s_val,
            )
        )

    t1_syn_eval = evaluate_t1(syn_records, syn_t1_preds)["synthetic"]
    t2_syn_eval = compute_ner_metrics(syn_records, syn_t2_spans)
    t3_syn_eval = compute_t3_metrics(syn_records, syn_t3_preds)
    prio_live_full = compute_priority_metrics(syn_records, syn_esc_flags, syn_pred_lvls, collect_missed=10)

    routed_count = sum(1 for b in syn_beams if bool(b["who"]["assigned_agency"]))
    org_routed = sum(1 for b in syn_beams if b["who"]["routing_source"] == "ner_organization")

    real_t3_preds: List[Dict[str, Any]] = []
    real_beams: List[Dict[str, Any]] = []
    real_timings: List[Dict[str, float]] = []
    real_esc_flags: List[bool] = []
    real_pred_lvls: List[str] = []

    for rec in real_records:
        channel = "sp4n" if rec["source"] == "sp4n_lapor" else "portal_web"
        out = pipeline.process(rec["text"], channel=channel, reporter_id=rec["id"])
        ce = out["complaint_event"]
        be = out["beam_event"]
        real_beams.append(be)
        real_timings.append(out["timings_ms"])
        real_t3_preds.append(
            {
                "sentiment": SCHEMA_TO_SENTIMENT[ce["sentiment"]],
                "is_sarcasm": 1 if ce["sarcasm"] else 0,
            }
        )
        real_esc_flags.append(bool(be["why"]["escalated"]))
        real_pred_lvls.append(str(be["why"]["priority_level"]))

    real_urg_pairs = [
        (rec, esc, lvl)
        for rec, esc, lvl in zip(real_records, real_esc_flags, real_pred_lvls)
        if rec.get("urgency_level") is not None
    ]
    real_prio_eval = (
        compute_priority_metrics(
            [p[0] for p in real_urg_pairs],
            [p[1] for p in real_urg_pairs],
            [p[2] for p in real_urg_pairs],
            collect_missed=5,
        )
        if real_urg_pairs
        else None
    )

    real_org_routed = sum(1 for b in real_beams if b["who"]["routing_source"] == "ner_organization")
    real_esc_count = sum(1 for f in real_esc_flags if f)

    report = {
        "backends": {
            "t1": pipeline.t1_backend,
            "t2_ner": pipeline.t2_predictor.ner_backend,
            "t2_lda": "lda" if pipeline.has_lda else "none",
            "t3": pipeline.t3_backend,
        },
        "priority_config": pipeline.priority_config.to_dict(),
        "synthetic_e2e": {
            "n_records": len(syn_records),
            "components": {
                "t1": t1_syn_eval,
                "t2_ner": t2_syn_eval,
                "t3": t3_syn_eval,
            },
            "priority_live_full": prio_live_full,
            "priority_ablation_live": run_priority_ablation(
                syn_records, syn_t1_objs, syn_t3_objs, pipeline.priority_config
            ),
            "priority_ablation_oracle": run_priority_ablation(
                syn_records, oracle_t1_objs, oracle_t3_objs, pipeline.priority_config
            ),
            "routing": {
                "routing_coverage": round(routed_count / max(1, len(syn_beams)), 4),
                "ner_org_routing_rate": round(org_routed / max(1, len(syn_beams)), 4),
            },
            "latency_ms": summarize_latencies(syn_timings),
        },
        "real_test": {
            "n_records": len(real_records),
            "sources": sorted(list({r["source"] for r in real_records})),
            "escalation_rate": round(real_esc_count / max(1, len(real_beams)), 4),
            "ner_org_routing_rate": round(real_org_routed / max(1, len(real_beams)), 4),
            "t3_metrics": compute_t3_metrics(real_records, real_t3_preds) if real_records else None,
            "priority_vs_campus_urgency": real_prio_eval,
            "latency_ms": summarize_latencies(real_timings),
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    beam_samples = select_beam_samples(syn_beams, limit=10)
    beam_samples_path.parent.mkdir(parents=True, exist_ok=True)
    with beam_samples_path.open("w", encoding="utf-8") as f:
        for ev in beam_samples:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate integrated End-to-End Civic Complaint pipeline.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--beam-samples-path", type=Path, default=DEFAULT_BEAM_SAMPLES_PATH)
    parser.add_argument("--t1-checkpoint", type=Path, default=None)
    parser.add_argument("--t2-checkpoint", type=Path, default=None)
    parser.add_argument("--t2-lda-dir", type=Path, default=None)
    parser.add_argument("--t3-checkpoint", type=Path, default=None)
    parser.add_argument("--priority-config", type=Path, default=None)
    parser.add_argument("--real-max-samples", type=int, default=300)
    args = parser.parse_args()

    report = evaluate_e2e(
        data_path=args.data_path,
        report_path=args.report_path,
        beam_samples_path=args.beam_samples_path,
        t1_checkpoint=args.t1_checkpoint,
        t2_checkpoint=args.t2_checkpoint,
        t2_lda_dir=args.t2_lda_dir,
        t3_checkpoint=args.t3_checkpoint,
        priority_config_path=args.priority_config,
        real_max_samples=args.real_max_samples,
    )
    print_markdown_summary(report)
    print(f"\nSaved E2E evaluation report -> {args.report_path}")
    print(f"Saved sample BEAM events    -> {args.beam_samples_path}")


if __name__ == "__main__":
    main()

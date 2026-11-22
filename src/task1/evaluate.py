"""Comparable T1 reports, with real and synthetic results kept separate."""
from __future__ import annotations

from collections import Counter

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from src.task1.data_loader import DOMAINS, INTENTS


def evaluate(records: list[dict], predictions: list[dict]) -> dict:
    if len(records) != len(predictions):
        raise ValueError("Prediction count does not match record count")
    report = {}
    for subset, rows in (("all", range(len(records))),
                         ("real", (i for i, r in enumerate(records) if r["source"] != "synthetic")),
                         ("synthetic", (i for i, r in enumerate(records) if r["source"] == "synthetic"))):
        indices = list(rows)
        report[subset] = {}
        for target, labels in (("domain", DOMAINS), ("intent", INTENTS)):
            pairs = [(records[i][target], predictions[i][target]) for i in indices
                     if records[i][target] is not None and predictions[i].get(target) is not None]
            counts = Counter(gold for gold, _ in pairs)
            if not pairs:
                report[subset][target] = {"n": 0, "accuracy": None, "macro_f1": None, "recall_by_class": None, "support": {}, "confusion_matrix": None, "labels": list(labels), "scored_labels": []}
                continue
            truth, pred = zip(*pairs)
            scored_labels = [label for label in labels if counts[label] > 0]
            recalls = recall_score(truth, pred, labels=labels, average=None, zero_division=0)
            report[subset][target] = {
                "n": len(pairs),
                "accuracy": round(float(accuracy_score(truth, pred)), 4),
                "macro_f1": round(float(f1_score(truth, pred, labels=scored_labels, average="macro", zero_division=0)), 4),
                "recall_by_class": {label: round(float(value), 4) for label, value in zip(labels, recalls)},
                "support": {label: counts[label] for label in labels},
                "confusion_matrix": confusion_matrix(truth, pred, labels=labels).tolist(),
                "labels": list(labels),
                "scored_labels": scored_labels,
            }
    return report

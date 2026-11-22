"""Small contract checks for masking, splits, and saved-model inference."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.task1.data_loader import load_records
from src.task1.evaluate import evaluate


class Task1ContractTest(unittest.TestCase):
    def test_loader_keeps_original_splits_and_missing_labels(self):
        rows = [
            {"id": "a", "split": "train", "source": "synthetic", "text": "Tolong jalan diperbaiki", "domain": "Infrastruktur & PU", "intent": "Permintaan"},
            {"id": "b", "split": "test", "source": "campus_complaints", "text": "Administrasi lambat", "domain": "Layanan & Administrasi", "intent": None},
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            self.assertEqual([r["id"] for r in load_records(path, "test")], ["b"])
            self.assertIsNone(load_records(path, "test")[0]["intent"])

    def test_metrics_skip_unlabeled_targets_and_absent_classes(self):
        rows = [{"source": "campus_complaints", "domain": "Layanan & Administrasi", "intent": None}]
        report = evaluate(rows, [{"domain": "Layanan & Administrasi", "intent": "Keluhan"}])
        self.assertEqual(report["real"]["domain"]["macro_f1"], 1.0)
        self.assertEqual(report["real"]["domain"]["scored_labels"], ["Layanan & Administrasi"])
        self.assertIsNone(report["real"]["intent"]["accuracy"])


if __name__ == "__main__":
    unittest.main()

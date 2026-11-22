"""
Unit & Contract Tests for Priority Engine & End-to-End Pipeline Integration.

Run from repository root:
    python -m unittest src.integration.test_integration -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.integration.beam import infer_admin_region
from src.integration.evaluate_e2e import _select_balanced_real_records
from src.integration.pipeline import CivicComplaintPipeline, _find_best_t1_seed_dir
from src.priority.engine import PriorityConfig, compute_urgency
from src.schema import T1Result, T3Result


class IntegrationAndPriorityTest(unittest.TestCase):
    def test_find_best_t1_seed_dir_resolves_best_seed_subdirectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            e3_dir = Path(tmp) / "E3"
            s42 = e3_dir / "seed_42"
            s43 = e3_dir / "seed_43"
            s42.mkdir(parents=True)
            s43.mkdir(parents=True)

            (s42 / "model_weights.pt").write_bytes(b"dummy")
            (s42 / "train_meta.json").write_text(
                json.dumps({"history": [{"val_domain_macro_f1": 0.80, "val_intent_macro_f1": 0.70}]}),
                encoding="utf-8",
            )
            (s43 / "model_weights.pt").write_bytes(b"dummy")
            (s43 / "train_meta.json").write_text(
                json.dumps({"history": [{"val_domain_macro_f1": 0.90, "val_intent_macro_f1": 0.85}]}),
                encoding="utf-8",
            )

            resolved = _find_best_t1_seed_dir(e3_dir)
            self.assertEqual(resolved, s43)

    def test_infer_admin_region_word_boundary_prevents_gunungpati_to_pati(self) -> None:
        self.assertEqual(
            infer_admin_region(["Kelurahan Sekaran, Gunungpati"]),
            "Gunungpati",
        )
        self.assertEqual(
            infer_admin_region(["Jl. Kartini, Pati"]),
            "Pati",
        )
        self.assertEqual(
            infer_admin_region(["Jl. Setiabudi, Srondol"]),
            "Srondol",
        )

    def test_no_intent_ablation_does_not_leak_darurat_intent(self) -> None:
        cfg = PriorityConfig(
            w1=0.3,
            w2=0.4,
            w3=0.3,
            alpha=0.2,
            escalation_threshold=7.50,
            emergency_threshold=8.50,
        )
        # Choose domain & sentiment scores where renormalized (w1, w3) score is 8.60 (>= 8.50, < 8.80)
        t1 = T1Result(domain="Pungli & Tata Kelola", intent="Darurat")  # C = 8.5
        t3 = T3Result(sentiment="negative", sarcasm=False, sentiment_score=0.87)  # S = 8.7

        full_out = compute_urgency(t1, t3, config=cfg, ablation="full")
        self.assertEqual(full_out.priority_level, "Emergency")
        self.assertIn("Intensi darurat terdeteksi", full_out.escalation_reasons)

        no_intent_out = compute_urgency(t1, t3, config=cfg, ablation="no_intent")
        self.assertEqual(no_intent_out.priority_level, "High")
        self.assertNotIn("Intensi darurat terdeteksi", no_intent_out.escalation_reasons)

    def test_balanced_real_records_allocation(self) -> None:
        fake_test = (
            [{"id": f"sp_{i}", "source": "sp4n_lapor"} for i in range(223)]
            + [{"id": f"cp_{i}", "source": "campus_complaints"} for i in range(750)]
        )
        sampled = _select_balanced_real_records(fake_test, 300)
        self.assertEqual(len(sampled), 300)
        self.assertEqual(sum(1 for r in sampled if r["source"] == "sp4n_lapor"), 150)
        self.assertEqual(sum(1 for r in sampled if r["source"] == "campus_complaints"), 150)

    def test_end_to_end_pipeline_process(self) -> None:
        pipeline = CivicComplaintPipeline()
        text = "Lampu jalan di Jl. Setiabudi, Srondol mati 5 hari, mohon Dishub Kota Semarang segera perbaiki."
        out = pipeline.process(text, channel="laporgub", reporter_id="citizen_01")
        self.assertIn("complaint_event", out)
        self.assertIn("beam_event", out)
        ce = out["complaint_event"]
        be = out["beam_event"]
        self.assertEqual(ce["text"], text)
        self.assertTrue(len(ce["entities"]) >= 2)
        for ent in ce["entities"]:
            self.assertEqual(ent["text"], text[ent["start"] : ent["end"]])
        self.assertEqual(be["who"]["assigned_agency"], "Dishub Kota Semarang")
        self.assertEqual(be["where"]["admin_region"], "Srondol")


if __name__ == "__main__":
    unittest.main()

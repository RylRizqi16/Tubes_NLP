"""
Task 2 Phase 5 — Unit & Contract Test Suite (`src/task2/test_task2.py`).

Run from repository root:
    python -m unittest src.task2.test_task2 -v
"""

from __future__ import annotations

import unittest
from pathlib import Path

from src.schema import EntitySpan, T2Result
from src.task2.data_loader import (
    DEFAULT_MASTER_PATH,
    DEFAULT_MAX_LENGTH,
    TAG_TO_ID,
    decode_bio2_to_spans,
    encode_spans_to_bio2,
    filter_ner_subset,
    load_lda_records,
    load_ner_records,
    regex_tokenize_with_offsets,
    verify_dataset_span_alignment,
)
from src.task2.evaluate_ner import compute_ner_metrics
from src.task2.gazetteer import (
    apply_civic_postprocessing,
    disambiguate_kelurahan_kecamatan,
    should_merge_comma_locations,
)
from src.task2.predictor import DEFAULT_LDA_MODEL_DIR, T2Predictor, predict, predict_topic
from src.task2.train_lda import build_4layer_stopwords, preprocess_lda_text


class Task2AlignmentAndDecodingTest(unittest.TestCase):
    def test_master_dataset_zero_truncation_and_roundtrip_alignment(self) -> None:
        all_records = load_ner_records(data_path=DEFAULT_MASTER_PATH, split=None, subset="all")
        self.assertEqual(len(all_records), 9257)
        result = verify_dataset_span_alignment(all_records, max_length=DEFAULT_MAX_LENGTH)
        self.assertEqual(result["total_gold_spans"], 21002)
        self.assertEqual(result["exact_recovered_spans"], 21002)
        self.assertEqual(result["truncated_spans"], 0)
        self.assertEqual(result["alignment_loss_rate"], 0.0)

    def test_adjacent_same_label_b_spans_preserved_across_single_space(self) -> None:
        cases = [
            (
                "Mahasiswa Fakultas Psikologi Universitas Indonesia mengadakan seminar.",
                [
                    {"start": 10, "end": 28, "label": "ORGANIZATION"},
                    {"start": 29, "end": 50, "label": "ORGANIZATION"},
                ],
                ["Fakultas Psikologi", "Universitas Indonesia"],
            ),
            (
                "Bantuan dikirim ke Desa Wonolelo Kecamatan Sawangan pagi ini.",
                [
                    {"start": 19, "end": 32, "label": "LOCATION"},
                    {"start": 33, "end": 51, "label": "LOCATION"},
                ],
                ["Desa Wonolelo", "Kecamatan Sawangan"],
            ),
        ]
        for text, gold, expected_texts in cases:
            enc = regex_tokenize_with_offsets(text, max_length=DEFAULT_MAX_LENGTH)
            offsets = enc["offset_mapping"]
            bio_ids = encode_spans_to_bio2(offsets, gold)
            decoded = decode_bio2_to_spans(
                text, offsets, bio_ids, repair_subwords=True, promote_orphan_i=False
            )
            self.assertEqual([d["text"] for d in decoded], expected_texts)
            self.assertEqual(
                [(d["start"], d["end"], d["label"]) for d in decoded],
                [(g["start"], g["end"], g["label"]) for g in gold],
            )

    def test_subword_continuation_repair_vs_orphan_i_opt_in(self) -> None:
        # WordPiece-style split of "Semarang" into "Semar" + "##ang" where "##ang" is tagged O,
        # plus a standalone noisy "O -> I-PERSON" on "Hari".
        text = "Hari di Semarang hujan."
        offsets = [(0, 0), (0, 4), (5, 7), (8, 13), (13, 16), (17, 22), (22, 23), (0, 0)]
        tokens = ["[CLS]", "hari", "di", "semar", "##ang", "hujan", ".", "[SEP]"]
        tag_ids = [
            -100,
            TAG_TO_ID["I-PERSON"],  # noisy standalone O -> I-PERSON
            TAG_TO_ID["O"],
            TAG_TO_ID["B-LOCATION"],
            TAG_TO_ID["O"],  # dropped WordPiece continuation
            TAG_TO_ID["O"],
            TAG_TO_ID["O"],
            -100,
        ]

        # Raw decoding: drops noisy I-PERSON, leaves truncated "Semar"
        raw = decode_bio2_to_spans(text, offsets, tag_ids, tokens=tokens, repair_subwords=False, promote_orphan_i=False)
        self.assertEqual([(s["text"], s["label"]) for s in raw], [("Semar", "LOCATION")])

        # Safe subword repair only (E3/E4b default): fixes "Semarang", still drops noisy I-PERSON
        repaired = decode_bio2_to_spans(text, offsets, tag_ids, tokens=tokens, repair_subwords=True, promote_orphan_i=False)
        self.assertEqual([(s["text"], s["label"]) for s in repaired], [("Semarang", "LOCATION")])

        # Opt-in orphan promotion: also promotes "Hari"
        promoted = decode_bio2_to_spans(text, offsets, tag_ids, tokens=tokens, repair_subwords=True, promote_orphan_i=True)
        self.assertEqual(
            [(s["text"], s["label"]) for s in promoted],
            [("Hari", "PERSON"), ("Semarang", "LOCATION")],
        )


class Task2PostProcessingAndOracleTest(unittest.TestCase):
    def test_oracle_postprocessing_upper_bound_on_val(self) -> None:
        val_records = load_ner_records(data_path=DEFAULT_MASTER_PATH, split="val", subset="all")
        for sub in ("all", "real", "synthetic"):
            sub_recs = filter_ner_subset(val_records, sub)
            post_preds = [
                apply_civic_postprocessing(r["text"], r["entities"], inject_gazetteer=True)
                for r in sub_recs
            ]
            metrics = compute_ner_metrics(sub_recs, post_preds)
            self.assertEqual(
                metrics["micro_f1"],
                1.0,
                f"Oracle upper bound regressed on val({sub}): {metrics}",
            )

    def test_three_gate_comma_merger_and_guards(self) -> None:
        # Gate 3 fast-path: abbreviated Jl. X, Y merges
        t1 = "Lampu mati di Jl. Setiabudi, Srondol sejak kemarin."
        self.assertTrue(should_merge_comma_locations(t1, 14, 27, 29, 36, has_civic_org_in_sentence=False))

        # Gate 2: hierarchical second span must NOT merge
        t2 = "Banjir melanda Desa Sanggeng, Kecamatan Manokwari Barat."
        self.assertFalse(should_merge_comma_locations(t2, 15, 28, 30, 55, has_civic_org_in_sentence=True))

        # Gate 2 with dotted abbreviation (Kec.) must NOT merge
        t2b = "Warga melapor di Desa Sanggeng, Kec. Manokwari Barat."
        self.assertFalse(should_merge_comma_locations(t2b, 17, 30, 32, 52, has_civic_org_in_sentence=True))

        # Gate 1: spelled-out "Jalan" in real news must NOT merge
        t3 = "Konferensi pers di Kemenhub, Jalan Medan Merdeka Barat, Jakarta Pusat."
        self.assertFalse(should_merge_comma_locations(t3, 29, 54, 56, 69, has_civic_org_in_sentence=False))

        # Multi-comma chaining guard: already contains comma -> must NOT chain a third span
        t4 = "Mohon cek Pasar Wage, Purwokerto, Jawa Tengah segera."
        self.assertFalse(should_merge_comma_locations(t4, 10, 32, 34, 45, has_civic_org_in_sentence=True))

    def test_kantor_gating_and_strict_expansion_only(self) -> None:
        # Non-civic encyclopedia text: Kantor must NOT expand onto Kecamatan Talang
        wiki = "Desa ini terletak di sebelah timur Kantor Kecamatan Talang."
        s, e, lbl = disambiguate_kelurahan_kecamatan(
            wiki, 42, 58, "LOCATION", is_civic_sentence=False
        )
        self.assertEqual((wiki[s:e], lbl), ("Kecamatan Talang", "LOCATION"))

        # Civic complaint text: Kantor DOES expand and set ORGANIZATION
        civic = "Mohon perbaikan layanan di Kantor Kecamatan Talang segera."
        s2, e2, lbl2 = disambiguate_kelurahan_kecamatan(
            civic, 34, 50, "LOCATION", is_civic_sentence=True
        )
        self.assertEqual((civic[s2:e2], lbl2), ("Kantor Kecamatan Talang", "ORGANIZATION"))

        # Equal-boundary neural span must NOT have its label overwritten by gazetteer
        news_like = "Warga melapor antrean di RSUD Budhi Asih sangat panjang."
        idx = news_like.index("RSUD Budhi Asih")
        raw = [{"start": idx, "end": idx + len("RSUD Budhi Asih"), "label": "LOCATION"}]
        out = apply_civic_postprocessing(news_like, raw, inject_gazetteer=True)
        self.assertEqual([(x["text"], x["label"]) for x in out], [("RSUD Budhi Asih", "LOCATION")])


class Task2PredictorContractAndLDATest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predictor = T2Predictor()

    def test_predictor_schema_contract_and_invariants(self) -> None:
        samples = [
            "  Lampu jalan di Jl. Setiabudi, Srondol mati 3 hari, mohon Dishub Kota Semarang segera perbaiki.  ",
            "Terima kasih kepada Pak Budi dan Puskesmas Tembalang atas pelayanan yang cepat!",
            "Keluhan warga: sampah menumpuk di Pasar Klewer, Solo dan belum diangkut DLH Jawa Tengah.",
            "Cek http://example.com/aduan   dengan  spasi   ganda di Kantor Kecamatan Genuk.",
        ]
        for text in samples:
            res = self.predictor.predict(text)
            self.assertIsInstance(res, T2Result)
            prev_end = -1
            for sp in res.entities:
                self.assertIsInstance(sp, EntitySpan)
                self.assertIn(sp.label, ("LOCATION", "ORGANIZATION", "PERSON"))
                self.assertTrue(0 <= sp.start < sp.end <= len(text))
                # Exact character alignment to caller's unmodified string
                self.assertEqual(sp.text, text[sp.start : sp.end])
                self.assertGreaterEqual(sp.start, prev_end)
                prev_end = sp.end
            self.assertGreater(len(res.entities), 0, f"Expected entities in {text!r}")

        # Module-level singleton helper matches
        res_fn = predict(samples[0])
        self.assertIsInstance(res_fn, T2Result)
        self.assertEqual(res_fn.entities, self.predictor.predict(samples[0]).entities)

    def test_zero_entity_and_empty_text_handling(self) -> None:
        self.assertEqual(self.predictor.predict("").entities, [])
        self.assertEqual(self.predictor.predict("   \n\t  ").entities, [])
        neutral = self.predictor.predict("terima kasih banyak atas bantuannya hari ini.")
        self.assertIsInstance(neutral, T2Result)
        self.assertEqual(neutral.entities, [])

    def test_lda_preprocessing_and_predict_topic(self) -> None:
        train_lda = load_lda_records(data_path=DEFAULT_MASTER_PATH, split="train")
        self.assertEqual(len(train_lda), 6468)
        self.assertTrue(all(r["split"] == "train" for r in train_lda))

        sw = build_4layer_stopwords()
        tokens = preprocess_lda_text(
            "Dear team, mohon min [NAMA] [NIM] segera cek server aplikasi error di Semarang bareng Pak Budi!",
            stopwords=sw,
        )
        # Bilingual English, slang, masks, boilerplate, and synthetic city/person slots removed
        for blocked in ("dear", "team", "mohon", "min", "nama", "nim", "segera", "semarang", "pak", "budi"):
            self.assertNotIn(blocked, tokens)
        for kept in ("server", "aplikasi", "error"):
            self.assertIn(kept, tokens)

        if (DEFAULT_LDA_MODEL_DIR / "lda_meta.json").exists():
            topic_res = self.predictor.predict_topic(
                "Aplikasi SP4N LAPOR sering error dan server lambat saat login."
            )
            self.assertIn("topic_id", topic_res)
            self.assertIn("topic_probability", topic_res)
            self.assertIn("top_keywords", topic_res)
            self.assertIn("topic_distribution", topic_res)
            self.assertGreater(len(topic_res["top_keywords"]), 0)
            self.assertAlmostEqual(sum(topic_res["topic_distribution"].values()), 1.0, places=2)

            singleton_res = predict_topic(
                "Aplikasi SP4N LAPOR sering error dan server lambat saat login."
            )
            self.assertEqual(singleton_res["topic_id"], topic_res["topic_id"])


if __name__ == "__main__":
    unittest.main()

"""
Stable Task 2 Inference Interface (`src/task2/predictor.py`).

Exposes:
- `T2Predictor.predict(text) -> T2Result` (strictly compliant with `src/schema.py`)
- `T2Predictor.predict_topic(text) -> Dict[str, Any]` (LDA topic inference)
- Module-level singleton helpers `predict(text, checkpoint_dir=None) -> T2Result`
  and `predict_topic(text, lda_dir=None) -> Dict[str, Any]`.

Safety & Integration Guarantees:
- Never applies length-changing text cleaning before computing NER offsets, so
  every `EntitySpan(start, end)` indexes the exact string passed by the caller.
- Enforces hard output invariants on every span:
  `0 <= span.start < span.end <= len(text)`, `span.text == text[span.start:span.end]`,
  sorted by `(start, end)` with zero overlapping spans.
- Auto-discovers the best validation seed in `models/task2/ner_full/` with graceful
  CPU fallback to `models/task2/ner_baseline/baseline_model.joblib` (E1) or civic
  gazetteer rules if no neural checkpoint is present locally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from src.schema import EntitySpan, T2Result
from src.task2.data_loader import (
    DEFAULT_MAX_LENGTH,
    ROOT_DIR,
    decode_bio2_to_spans,
    spans_to_entity_objects,
)
from src.task2.gazetteer import apply_civic_postprocessing, extract_civic_gazetteer_spans
from src.task2.train_lda import (
    DEFAULT_LDA_MODEL_DIR,
    PMIBigramPhraser,
    build_4layer_stopwords,
    preprocess_lda_text,
)
from src.task2.train_ner_baseline import DEFAULT_BASELINE_MODEL_DIR, predict_baseline_spans


DEFAULT_MODELS_DIR = ROOT_DIR / "models" / "task2"
DEFAULT_NER_FULL_DIR = DEFAULT_MODELS_DIR / "ner_full"


def _assert_span_invariants(text: str, entities: List[EntitySpan]) -> List[EntitySpan]:
    """Verify hard integration invariants on every emitted `EntitySpan`."""
    prev_end = -1
    for sp in entities:
        if not (0 <= sp.start < sp.end <= len(text)):
            raise AssertionError(f"Out-of-bounds EntitySpan: {sp} for len(text)={len(text)}")
        if sp.text != text[sp.start : sp.end]:
            raise AssertionError(
                f"Offset mismatch: span.text={sp.text!r} != text[{sp.start}:{sp.end}]={text[sp.start:sp.end]!r}"
            )
        if sp.start < prev_end:
            raise AssertionError(f"Overlapping or unsorted EntitySpan: {sp} after end={prev_end}")
        prev_end = sp.end
    return entities


class T2Predictor:
    """Unified Task 2 predictor for NER (`T2Result`) and LDA topic inference."""

    def __init__(
        self,
        checkpoint_dir: Optional[Path | str] = None,
        lda_dir: Optional[Path | str] = None,
        use_postprocessing: bool = True,
    ) -> None:
        self.use_postprocessing = use_postprocessing
        self.checkpoint_dir = self._resolve_transformer_checkpoint(checkpoint_dir)
        self.lda_dir = Path(lda_dir) if lda_dir is not None else DEFAULT_LDA_MODEL_DIR

        # NER backend state
        self.ner_backend: str = "gazetteer"
        self.model = None
        self.tokenizer = None
        self.device = None
        self.max_length: int = DEFAULT_MAX_LENGTH
        self.baseline_bundle: Optional[Dict[str, Any]] = None

        # Lazy-loaded LDA backend state
        self._lda_loaded: bool = False
        self._lda_backend: Optional[str] = None
        self._lda_meta: Optional[Dict[str, Any]] = None
        self._lda_stopwords = None
        self._lda_model = None
        self._lda_dictionary = None
        self._lda_phraser = None
        self._lda_vectorizer = None

        self._init_ner_backend(checkpoint_dir)

    @staticmethod
    def _resolve_transformer_checkpoint(checkpoint_dir: Optional[Path | str]) -> Optional[Path]:
        if checkpoint_dir is not None:
            p = Path(checkpoint_dir)
            if p.is_dir() and (p / "model_weights.pt").exists():
                return p
            # If user pointed to a group directory (e.g. models/task2/ner_full)
            summary_path = p / "training_summary.json"
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    best_dir = p / str(summary.get("best_seed_dir", ""))
                    if (best_dir / "model_weights.pt").exists():
                        return best_dir
                except Exception:
                    pass
            if p.is_dir():
                for cand in sorted(p.glob("seed_*")):
                    if (cand / "model_weights.pt").exists():
                        return cand
            return None

        # Auto-discover best validation seed in models/task2/ner_full/
        if DEFAULT_NER_FULL_DIR.exists():
            summary_path = DEFAULT_NER_FULL_DIR / "training_summary.json"
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    best_dir = DEFAULT_NER_FULL_DIR / str(summary.get("best_seed_dir", ""))
                    if (best_dir / "model_weights.pt").exists():
                        return best_dir
                except Exception:
                    pass

            # Otherwise choose seed with highest val micro_f1 from train_meta.json
            best_ckpt: Optional[Path] = None
            best_score = -1.0
            for cand in sorted(DEFAULT_NER_FULL_DIR.glob("seed_*")):
                if not (cand / "model_weights.pt").exists():
                    continue
                score = 0.0
                meta_path = cand / "train_meta.json"
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        score = float(meta.get("best_val_micro_f1", 0.0))
                    except Exception:
                        score = 0.0
                if best_ckpt is None or score > best_score:
                    best_ckpt = cand
                    best_score = score
            if best_ckpt is not None:
                return best_ckpt

        return None

    def _init_ner_backend(self, explicit_checkpoint: Optional[Path | str]) -> None:
        # 1. Try loading IndoBERT transformer checkpoint if available
        if self.checkpoint_dir is not None:
            try:
                self._load_transformer(self.checkpoint_dir)
                self.ner_backend = "indobert"
                return
            except ImportError:
                # Fall through to CPU E1 baseline if torch/transformers are not installed
                pass

        # 2. Try loading explicit joblib path or default E1 CPU baseline artifact
        baseline_path = DEFAULT_BASELINE_MODEL_DIR / "baseline_model.joblib"
        if explicit_checkpoint is not None:
            p = Path(explicit_checkpoint)
            if p.is_file() and p.suffix == ".joblib":
                baseline_path = p

        if baseline_path.exists():
            self.baseline_bundle = joblib.load(baseline_path)
            self.ner_backend = "e1_baseline"
            return

        # 3. Last-resort zero-artifact fallback: cased civic gazetteer rules
        self.ner_backend = "gazetteer"

    def _load_transformer(self, ckpt_dir: Path) -> None:
        import torch
        from transformers import AutoTokenizer
        from src.task2.train_ner_transformer import DEFAULT_MODEL_NAME, IndoBERTNER

        meta_path = ckpt_dir / "train_meta.json"
        model_name = str(ckpt_dir) if (ckpt_dir / "config.json").exists() else DEFAULT_MODEL_NAME
        dropout = 0.1
        use_crf = False
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            dropout = float(meta.get("dropout", 0.1))
            use_crf = bool(meta.get("use_crf", False))
            self.max_length = int(meta.get("max_length", DEFAULT_MAX_LENGTH))

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
        self.model = IndoBERTNER(
            model_name=model_name,
            dropout=dropout,
            use_crf=use_crf,
            from_pretrained=False,
        )
        self.model.load_state_dict(
            torch.load(ckpt_dir / "model_weights.pt", map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> T2Result:
        """
        Predict NER entities for `text` without altering `text` before offset extraction.
        Returns `T2Result(entities=[EntitySpan(...)])` strictly matching `src/schema.py`.
        """
        if not isinstance(text, str):
            raise TypeError(f"T2Predictor.predict expects str, got {type(text).__name__}")
        if not text.strip():
            return T2Result(entities=[])

        if self.ner_backend == "indobert" and self.model is not None and self.tokenizer is not None:
            import torch

            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = [(int(s), int(e)) for s, e in enc["offset_mapping"][0].tolist()]
            tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())

            with torch.no_grad():
                tag_ids = self.model.predict_tag_ids(
                    input_ids=enc["input_ids"].to(self.device),
                    attention_mask=enc["attention_mask"].to(self.device),
                    offset_mappings=[offsets],
                )[0]

            raw_spans = decode_bio2_to_spans(
                text,
                offsets,
                tag_ids,
                tokens=tokens,
                repair_subwords=self.use_postprocessing,
                promote_orphan_i=False,
            )
            final_spans = (
                apply_civic_postprocessing(text, raw_spans, inject_gazetteer=True)
                if self.use_postprocessing
                else raw_spans
            )
        elif self.ner_backend == "e1_baseline" and self.baseline_bundle is not None:
            final_spans = predict_baseline_spans(
                text,
                self.baseline_bundle,
                use_postprocessing=self.use_postprocessing,
            )
        else:
            final_spans = extract_civic_gazetteer_spans(text)

        entities = spans_to_entity_objects(text, final_spans)
        _assert_span_invariants(text, entities)
        return T2Result(entities=entities)

    def _load_lda_backend(self) -> None:
        if self._lda_loaded:
            return
        meta_path = self.lda_dir / "lda_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"LDA model metadata not found at {meta_path}. "
                "Run `python -m src.task2.train_lda` first."
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        backend = str(meta.get("backend", "sklearn"))
        self._lda_meta = meta
        self._lda_backend = backend
        self._lda_stopwords = build_4layer_stopwords()

        if backend == "gensim":
            try:
                from gensim.corpora import Dictionary
                from gensim.models import LdaModel
                from gensim.models.phrases import Phraser

                self._lda_model = LdaModel.load(str(self.lda_dir / "lda_gensim.model"))
                self._lda_dictionary = Dictionary.load(str(self.lda_dir / "dictionary.gensim"))
                self._lda_phraser = Phraser.load(str(self.lda_dir / "bigram_phraser.gensim"))
            except ImportError:
                sklearn_path = self.lda_dir / "lda_sklearn.joblib"
                if not sklearn_path.exists():
                    raise
                bundle = joblib.load(sklearn_path)
                self._lda_backend = "sklearn"
                self._lda_model = bundle["lda_model"]
                self._lda_vectorizer = bundle["vectorizer"]
                phraser = PMIBigramPhraser()
                phraser.bigrams = {tuple(pair) for pair in bundle.get("bigrams", [])}
                self._lda_phraser = phraser
                self._lda_meta = {
                    **meta,
                    "backend": "sklearn",
                    "selected_k": int(bundle.get("selected_k", meta.get("selected_k", 15))),
                    "topics_keywords": bundle.get("topics_keywords", meta.get("topics_keywords", {})),
                }
        else:
            bundle = joblib.load(self.lda_dir / "lda_sklearn.joblib")
            self._lda_model = bundle["lda_model"]
            self._lda_vectorizer = bundle["vectorizer"]
            phraser = PMIBigramPhraser()
            phraser.bigrams = {tuple(pair) for pair in bundle.get("bigrams", [])}
            self._lda_phraser = phraser

        self._lda_loaded = True

    def predict_topic(self, text: str) -> Dict[str, Any]:
        """Infer LDA topic distribution and top keywords for `text`."""
        if not isinstance(text, str):
            raise TypeError(f"T2Predictor.predict_topic expects str, got {type(text).__name__}")
        self._load_lda_backend()
        assert self._lda_meta is not None

        k = int(self._lda_meta["selected_k"])
        keywords_map: Dict[str, List[str]] = self._lda_meta["topics_keywords"]
        raw_tokens = preprocess_lda_text(text, stopwords=self._lda_stopwords)

        if self._lda_backend == "gensim":
            bigram_tokens = list(self._lda_phraser[raw_tokens])
            bow = self._lda_dictionary.doc2bow(bigram_tokens)
            dist = self._lda_model.get_document_topics(bow, minimum_probability=0.0)
            probs = [0.0] * k
            for tid, prob in dist:
                probs[int(tid)] = float(prob)
        else:
            bigram_tokens = self._lda_phraser.transform_doc(raw_tokens)
            X = self._lda_vectorizer.transform([" ".join(bigram_tokens)])
            if X.nnz == 0:
                probs = [1.0 / k] * k
            else:
                probs = self._lda_model.transform(X)[0].tolist()

        best_tid = int(np.argmax(probs))
        return {
            "topic_id": best_tid,
            "topic_probability": round(float(probs[best_tid]), 4),
            "top_keywords": list(keywords_map.get(str(best_tid), [])),
            "topic_distribution": {t: round(float(p), 4) for t, p in enumerate(probs)},
            "tokens": bigram_tokens,
            "backend": self._lda_backend,
        }


_DEFAULT_PREDICTOR: Optional[T2Predictor] = None
_DEFAULT_PREDICTOR_KEY: Optional[Tuple[Optional[str], Optional[str]]] = None


def _get_singleton(
    checkpoint_dir: Optional[Path | str] = None,
    lda_dir: Optional[Path | str] = None,
) -> T2Predictor:
    global _DEFAULT_PREDICTOR, _DEFAULT_PREDICTOR_KEY
    key = (
        str(Path(checkpoint_dir).resolve()) if checkpoint_dir is not None else None,
        str(Path(lda_dir).resolve()) if lda_dir is not None else None,
    )
    if _DEFAULT_PREDICTOR is None or _DEFAULT_PREDICTOR_KEY != key:
        _DEFAULT_PREDICTOR = T2Predictor(checkpoint_dir=checkpoint_dir, lda_dir=lda_dir)
        _DEFAULT_PREDICTOR_KEY = key
    return _DEFAULT_PREDICTOR


def predict(text: str, checkpoint_dir: Optional[Path | str] = None) -> T2Result:
    """Module-level singleton NER helper returning `T2Result`."""
    return _get_singleton(checkpoint_dir=checkpoint_dir).predict(text)


def predict_topic(text: str, lda_dir: Optional[Path | str] = None) -> Dict[str, Any]:
    """Module-level singleton LDA topic helper."""
    return _get_singleton(lda_dir=lda_dir).predict_topic(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 2 NER (& optional LDA topic) inference.")
    parser.add_argument("text", type=str, help="Input Indonesian text.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional NER checkpoint path.")
    parser.add_argument("--lda-dir", type=Path, default=DEFAULT_LDA_MODEL_DIR)
    parser.add_argument("--with-topic", action="store_true", help="Also infer LDA topic distribution.")
    args = parser.parse_args()

    predictor = T2Predictor(checkpoint_dir=args.checkpoint, lda_dir=args.lda_dir)
    t2_res = predictor.predict(args.text)
    payload: Dict[str, Any] = {
        "ner_backend": predictor.ner_backend,
        "entities": [e.to_dict() for e in t2_res.entities],
    }
    if args.with_topic:
        payload["topic"] = predictor.predict_topic(args.text)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.schema import T3Result
from src.task3.data_loader import ROOT_DIR, SENTIMENT_TO_SCHEMA
from src.task3.rules import apply_rule_engine, predict_e1


DEFAULT_MODELS_DIR = ROOT_DIR / "models" / "task3"


class T3Predictor:

    def __init__(self, checkpoint_dir: Optional[Path | str] = None) -> None:
        self.checkpoint_dir = self._resolve_checkpoint_dir(checkpoint_dir)
        self.model = None
        self.tokenizer = None
        self.device = None
        self.max_length = 128

        if self.checkpoint_dir is not None:
            self._load_transformer(self.checkpoint_dir)

    @staticmethod
    def _resolve_checkpoint_dir(checkpoint_dir: Optional[Path | str]) -> Optional[Path]:
        if checkpoint_dir is not None:
            p = Path(checkpoint_dir)
            return p if (p / "model_weights.pt").exists() else None
        if not DEFAULT_MODELS_DIR.exists():
            return None
        for candidate in sorted(DEFAULT_MODELS_DIR.glob("seed_*")):
            if (candidate / "model_weights.pt").exists():
                return candidate
        return None

    def _load_transformer(self, ckpt_dir: Path) -> None:
        import torch
        from transformers import AutoTokenizer
        from src.task3.train_indobertweet import DEFAULT_MODEL_NAME, IndoBERTweetMultiTask

        meta_path = ckpt_dir / "train_meta.json"
        model_name = DEFAULT_MODEL_NAME
        dropout = 0.2
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            model_name = meta.get("model_name", DEFAULT_MODEL_NAME)
            dropout = float(meta.get("dropout", 0.2))
            self.max_length = int(meta.get("max_length", 128))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        self.model = IndoBERTweetMultiTask(model_name=model_name, dropout=dropout, from_pretrained=False)
        self.model.load_state_dict(
            torch.load(ckpt_dir / "model_weights.pt", map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        text: str,
        use_negation: bool = True,
        use_sarcasm: bool = True,
    ) -> T3Result:
        if self.model is not None and self.tokenizer is not None and self.device is not None:
            import torch

            enc = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                sent_logits, sarc_logits = self.model(
                    input_ids=enc["input_ids"].to(self.device),
                    attention_mask=enc["attention_mask"].to(self.device),
                )
                sent_probs = torch.softmax(sent_logits, dim=-1).squeeze(0).cpu().tolist()
                sarc_prob = float(torch.softmax(sarc_logits, dim=-1)[0, 1].cpu().item())

            out = apply_rule_engine(
                text,
                sent_probs=sent_probs,
                sarc_prob=sarc_prob,
                use_negation=use_negation,
                use_sarcasm=use_sarcasm,
            )
            return T3Result(
                sentiment=SENTIMENT_TO_SCHEMA[out["sentiment"]],
                sarcasm=bool(out["is_sarcasm"]),
                sentiment_score=float(out["sentiment_score"]),
            )

        out = predict_e1(text)
        return T3Result(
            sentiment=SENTIMENT_TO_SCHEMA[out["sentiment"]],
            sarcasm=bool(out["is_sarcasm"]),
            sentiment_score=float(out["sentiment_score"]),
        )


_DEFAULT_PREDICTOR: Optional[T3Predictor] = None


def predict(text: str, checkpoint_dir: Optional[Path | str] = None) -> T3Result:
    global _DEFAULT_PREDICTOR
    if checkpoint_dir is not None:
        return T3Predictor(checkpoint_dir=checkpoint_dir).predict(text)
    if _DEFAULT_PREDICTOR is None:
        _DEFAULT_PREDICTOR = T3Predictor()
    return _DEFAULT_PREDICTOR.predict(text)

"""Stable T1 inference interface for the future T2/T3 integration pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.schema import T1Result
from src.task1.data_loader import DOMAINS, INTENTS, ROOT
from src.task1.preprocess import clean_text


class T1Predictor:
    def __init__(self, checkpoint: str | Path = ROOT / "models/task1/baseline.joblib"):
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"T1 checkpoint missing: {self.checkpoint}. Train E1 or supply an E2/E3 checkpoint directory.")
        self.transformer = self.checkpoint.is_dir()
        if self.transformer:
            import torch
            from transformers import AutoTokenizer
            from src.task1.train_transformer import T1Model

            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
            metadata = json.loads((self.checkpoint / "train_meta.json").read_text(encoding="utf-8"))
            if metadata["experiment"] not in ("E2", "E3"):
                raise ValueError("T1Predictor requires both trained heads; use E2 or E3")
            self.max_length = metadata["max_length"]
            self.model = T1Model(str(self.checkpoint), pretrained=False)
            self.model.load_state_dict(torch.load(self.checkpoint / "model_weights.pt", map_location=self.device, weights_only=True))
            self.model.to(self.device).eval()
        else:
            import joblib
            self.models = joblib.load(self.checkpoint)
            if set(self.models) != {"domain", "intent"}:
                raise ValueError("Invalid T1 baseline artifact")

    def predict(self, text: str) -> T1Result:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("T1 input must be nonempty text")
        text = clean_text(text)
        if not text:
            raise ValueError("T1 input is empty after cleaning")
        if self.transformer:
            import torch
            encoded = self.tokenizer(text, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt")
            with torch.no_grad():
                d, i = self.model(encoded["input_ids"].to(self.device), encoded["attention_mask"].to(self.device))
                dp = torch.softmax(d, dim=-1)[0]
                ip = torch.softmax(i, dim=-1)[0]
                di, ii = dp.argmax().item(), ip.argmax().item()
                return T1Result(DOMAINS[di], INTENTS[ii], float(dp[di]), float(ip[ii]))
        outputs = {}
        for target in ("domain", "intent"):
            model = self.models[target]
            probs = model.predict_proba([text])[0]
            index = probs.argmax()
            outputs[target] = (str(model.classes_[index]), float(probs[index]))
        return T1Result(outputs["domain"][0], outputs["intent"][0], outputs["domain"][1], outputs["intent"][1])


def predict(text: str, checkpoint: str | Path = ROOT / "models/task1/baseline.joblib") -> T1Result:
    return T1Predictor(checkpoint).predict(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/task1/baseline.joblib")
    args = parser.parse_args()
    print(json.dumps(vars(predict(args.text, args.checkpoint)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

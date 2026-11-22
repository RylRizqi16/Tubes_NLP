"""E1: word + character TF-IDF and independent logistic heads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from src.task1.data_loader import DEFAULT_DATA, ROOT, labeled, load_records
from src.task1.evaluate import evaluate


def make_pipeline(c: float) -> Pipeline:
    return Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
        ])),
        ("classifier", LogisticRegression(C=c, max_iter=1000, class_weight="balanced")),
    ])


def run(data: Path, model_dir: Path, report_path: Path) -> dict:
    train, val, test = (load_records(data, split) for split in ("train", "val", "test"))
    models, tuning = {}, {}
    for target in ("domain", "intent"):
        tr, va = labeled(train, target), labeled(val, target)
        if not tr or not va:
            raise ValueError(f"Missing train or validation data for {target}")
        candidates = []
        for c in (0.1, 1.0, 10.0):
            model = make_pipeline(c)
            model.fit([r["text"] for r in tr], [r[target] for r in tr])
            guesses = [str(p) for p in model.predict([r["text"] for r in va])]
            score = round(float(f1_score([r[target] for r in va], guesses, average="macro", zero_division=0)), 4)
            candidates.append((score, c, model))
        score, best_c, model = max(candidates, key=lambda item: (item[0], -item[1]))
        models[target] = model
        tuning[target] = {"selected_c": best_c, "validation_macro_f1": score,
                          "candidates": {str(c): s for s, c, _ in candidates}}
    def predict(rows):
        if not rows:
            return []
        texts = [r["text"] for r in rows]
        domains = models["domain"].predict(texts)
        intents = models["intent"].predict(texts)
        return [{"domain": str(d), "intent": str(i)} for d, i in zip(domains, intents)]
    report = {"experiment": "E1", "selection": tuning,
              "validation": evaluate(val, predict(val)), "test": evaluate(test, predict(test)),
              "data": str(data), "train_count": len(train)}
    model_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, model_dir / "baseline.joblib")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/task1")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/task1_e1.json")
    args = parser.parse_args()
    result = run(args.data, args.model_dir, args.report)
    print(json.dumps({k: result[k] for k in ("selection", "validation", "test")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

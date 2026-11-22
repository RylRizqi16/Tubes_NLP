"""Summarize E1-E4 reports using the same metric and source subsets."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from src.task1.data_loader import ROOT


def summarize(output_dir: Path) -> dict:
    paths = [output_dir / "task1_e1.json", *sorted(output_dir.glob("task1_E*_seed_*.json"))]
    result = {}
    for path in paths:
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        experiment = report["experiment"]
        result.setdefault(experiment, {"files": [], "metrics": {}})["files"].append(str(path))
        for split in ("validation", "test"):
            for subset in ("all", "real", "synthetic"):
                for target in ("domain", "intent"):
                    for metric in ("accuracy", "macro_f1"):
                        value = report[split][subset][target][metric]
                        if value is not None:
                            key = f"{split}.{subset}.{target}.{metric}"
                            result[experiment]["metrics"].setdefault(key, []).append(value)
    for entry in result.values():
        entry["metrics"] = {key: {"mean": round(statistics.mean(values), 4),
                                    "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
                                    "runs": len(values)}
                            for key, values in entry["metrics"].items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    result = summarize(args.output_dir)
    path = args.output_dir / "task1_comparison.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, value in result.items():
        print(f"{name}: {len(value['files'])} run(s)")
        for key in ("test.all.domain.macro_f1", "test.synthetic.intent.accuracy", "test.synthetic.intent.macro_f1"):
            if key in value["metrics"]:
                print(f"  {key}: {value['metrics'][key]}")
    print(path)


if __name__ == "__main__":
    main()

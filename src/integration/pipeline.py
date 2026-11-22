from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.integration.beam import build_beam_event, build_complaint_event
from src.priority.engine import (
    DEFAULT_WEIGHTS_PATH,
    PriorityConfig,
    ROOT_DIR,
    compute_urgency,
    load_priority_config,
)
from src.task1.predictor import T1Predictor
from src.task2.predictor import T2Predictor
from src.task3.predictor import T3Predictor


def resolve_t1_checkpoint(explicit: Optional[Path | str] = None) -> Path:
    if explicit is not None:
        return Path(explicit)
    candidates = [
        ROOT_DIR / "models" / "task1" / "E3",
        ROOT_DIR / "models" / "task1" / "E2",
        ROOT_DIR / "models" / "task1" / "baseline.joblib",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[-1]


class CivicComplaintPipeline:
    def __init__(
        self,
        t1_checkpoint: Optional[Path | str] = None,
        t2_checkpoint: Optional[Path | str] = None,
        t2_lda_dir: Optional[Path | str] = None,
        t3_checkpoint: Optional[Path | str] = None,
        priority_config_path: Optional[Path | str] = None,
    ) -> None:
        t1_path = resolve_t1_checkpoint(t1_checkpoint)
        self.t1_predictor = T1Predictor(checkpoint=t1_path)
        self.t1_backend = "transformer" if t1_path.is_dir() else "e1_baseline"

        self.t2_predictor = T2Predictor(checkpoint_dir=t2_checkpoint, lda_dir=t2_lda_dir)
        self.has_lda = (self.t2_predictor.lda_dir / "lda_meta.json").exists()

        self.t3_predictor = T3Predictor(checkpoint_dir=t3_checkpoint)
        self.t3_backend = "indobertweet" if self.t3_predictor.model is not None else "e1_baseline"

        cfg_path = Path(priority_config_path) if priority_config_path is not None else DEFAULT_WEIGHTS_PATH
        self.priority_config: PriorityConfig = load_priority_config(cfg_path)

    def process(
        self,
        text: str,
        channel: str = "portal_web",
        reporter_id: Optional[str] = None,
        reported_at: Optional[str] = None,
        ablation: str = "full",
        t3_use_negation: bool = True,
        t3_use_sarcasm: bool = True,
    ) -> Dict[str, Any]:
        total_t0 = time.perf_counter()

        t0 = time.perf_counter()
        t2_res = self.t2_predictor.predict(text)
        t2_ner_ms = (time.perf_counter() - t0) * 1000.0

        topic_info: Optional[Dict[str, Any]] = None
        t2_lda_ms = 0.0
        if self.has_lda:
            t0 = time.perf_counter()
            raw_topic = self.t2_predictor.predict_topic(text)
            t2_lda_ms = (time.perf_counter() - t0) * 1000.0
            topic_info = {
                "topic_id": raw_topic["topic_id"],
                "topic_probability": raw_topic["topic_probability"],
                "top_keywords": raw_topic["top_keywords"][:5],
            }

        t0 = time.perf_counter()
        t1_res = self.t1_predictor.predict(text)
        t1_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        t3_res = self.t3_predictor.predict(
            text,
            use_negation=t3_use_negation,
            use_sarcasm=t3_use_sarcasm,
        )
        t3_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        prio = compute_urgency(t1_res, t3_res, config=self.priority_config, ablation=ablation)
        prio_ms = (time.perf_counter() - t0) * 1000.0

        total_ms = (time.perf_counter() - total_t0) * 1000.0

        complaint_event = build_complaint_event(text, t1_res, t2_res, t3_res, prio)
        beam_event = build_beam_event(
            text=text,
            t1=t1_res,
            t2=t2_res,
            t3=t3_res,
            prio=prio,
            channel=channel,
            reporter_id=reporter_id,
            reported_at=reported_at,
            topic=topic_info,
            latency_ms=total_ms,
        )

        return {
            "complaint_event": complaint_event.to_dict(),
            "beam_event": beam_event,
            "timings_ms": {
                "t2_ner": round(t2_ner_ms, 4),
                "t2_lda": round(t2_lda_ms, 4),
                "t1": round(t1_ms, 4),
                "t3": round(t3_ms, 4),
                "priority": round(prio_ms, 4),
                "total": round(total_ms, 4),
            },
            "backends": {
                "t1": self.t1_backend,
                "t2_ner": self.t2_predictor.ner_backend,
                "t2_lda": "lda" if self.has_lda else "none",
                "t3": self.t3_backend,
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end Civic Complaint NLP pipeline.")
    parser.add_argument("text", type=str)
    parser.add_argument("--channel", type=str, default="portal_web")
    parser.add_argument("--t1-checkpoint", type=Path, default=None)
    parser.add_argument("--t2-checkpoint", type=Path, default=None)
    parser.add_argument("--t2-lda-dir", type=Path, default=None)
    parser.add_argument("--t3-checkpoint", type=Path, default=None)
    parser.add_argument("--priority-config", type=Path, default=None)
    args = parser.parse_args()

    pipeline = CivicComplaintPipeline(
        t1_checkpoint=args.t1_checkpoint,
        t2_checkpoint=args.t2_checkpoint,
        t2_lda_dir=args.t2_lda_dir,
        t3_checkpoint=args.t3_checkpoint,
        priority_config_path=args.priority_config,
    )
    out = pipeline.process(args.text, channel=args.channel)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

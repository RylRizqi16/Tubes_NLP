from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.schema import T1Result, T3Result


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS_PATH = ROOT_DIR / "outputs" / "priority_weights.json"

DEFAULT_DOMAIN_RISK_SCORES: Dict[str, float] = {
    "Pungli & Tata Kelola": 8.5,
    "Kesehatan & Sanitasi": 8.0,
    "Infrastruktur & PU": 7.0,
    "Lingkungan": 7.0,
    "Layanan & Administrasi": 5.5,
}

DEFAULT_INTENT_SEVERITY_SCORES: Dict[str, float] = {
    "Darurat": 10.0,
    "Keluhan": 7.5,
    "Permintaan": 5.5,
    "Saran": 2.5,
    "Apresiasi": 0.5,
}

SLA_HOURS_BY_PRIORITY: Dict[str, int] = {
    "Emergency": 2,
    "High": 24,
    "Medium": 72,
    "Low": 168,
}

ABLATION_MODES = ("full", "no_sarcasm", "no_sentiment", "no_domain", "no_intent")


def normalize_urgency_label(label: str) -> str:
    cleaned = str(label).strip()
    if cleaned == "Med":
        return "Medium"
    return cleaned


@dataclass
class PriorityConfig:
    w1: float = 0.30
    w2: float = 0.45
    w3: float = 0.25
    alpha: float = 0.20
    escalation_threshold: float = 7.50
    emergency_threshold: float = 8.50
    medium_threshold: float = 4.50
    domain_risk_scores: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_DOMAIN_RISK_SCORES)
    )
    intent_severity_scores: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_INTENT_SEVERITY_SCORES)
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PriorityOutput:
    urgency_index: float
    priority_level: str
    escalated: bool
    sla_hours: int
    escalation_reasons: List[str]
    components: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_priority_config(path: Path | str = DEFAULT_WEIGHTS_PATH) -> PriorityConfig:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return PriorityConfig()
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    best = raw.get("best_config", raw)
    return PriorityConfig(
        w1=float(best.get("w1", 0.30)),
        w2=float(best.get("w2", 0.45)),
        w3=float(best.get("w3", 0.25)),
        alpha=float(best.get("alpha", 0.20)),
        escalation_threshold=float(best.get("escalation_threshold", 7.50)),
        emergency_threshold=float(best.get("emergency_threshold", 8.50)),
        medium_threshold=float(best.get("medium_threshold", 4.50)),
        domain_risk_scores=dict(best.get("domain_risk_scores", DEFAULT_DOMAIN_RISK_SCORES)),
        intent_severity_scores=dict(best.get("intent_severity_scores", DEFAULT_INTENT_SEVERITY_SCORES)),
    )


def _sentiment_intensity_score(t3: T3Result) -> float:
    if t3.sentiment_score is not None:
        return min(10.0, max(0.0, float(t3.sentiment_score) * 10.0))
    fallback = {"negative": 8.5, "neutral": 3.5, "positive": 0.5}
    return fallback.get(str(t3.sentiment).lower(), 3.5)


def _ablation_weights(cfg: PriorityConfig, ablation: str) -> tuple[float, float, float, float]:
    if ablation not in ABLATION_MODES:
        raise ValueError(f"Unknown ablation mode {ablation!r}. Expected one of {ABLATION_MODES}.")
    w1, w2, w3, alpha = cfg.w1, cfg.w2, cfg.w3, cfg.alpha
    if ablation == "no_sarcasm":
        alpha = 0.0
    elif ablation == "no_sentiment":
        w3 = 0.0
    elif ablation == "no_domain":
        w1 = 0.0
    elif ablation == "no_intent":
        w2 = 0.0

    if ablation in ("no_sentiment", "no_domain", "no_intent"):
        total = w1 + w2 + w3
        if total > 0:
            w1, w2, w3 = w1 / total, w2 / total, w3 / total
    return w1, w2, w3, alpha


def compute_urgency(
    t1: T1Result,
    t3: T3Result,
    config: Optional[PriorityConfig] = None,
    ablation: str = "full",
) -> PriorityOutput:
    cfg = config or PriorityConfig()
    w1, w2, w3, alpha = _ablation_weights(cfg, ablation)

    c_score = float(cfg.domain_risk_scores.get(t1.domain, 5.5))
    i_score = float(cfg.intent_severity_scores.get(t1.intent, 5.5))
    s_score = _sentiment_intensity_score(t3)
    f_flag = 1 if t3.sarcasm else 0

    raw_u = (w1 * c_score + w2 * i_score + w3 * s_score) * (1.0 + alpha * f_flag)
    urgency_index = round(min(10.0, max(0.0, raw_u)), 4)

    if (t1.intent == "Darurat" and urgency_index >= cfg.emergency_threshold) or urgency_index >= 8.8:
        priority_level = "Emergency"
    elif urgency_index > cfg.escalation_threshold:
        priority_level = "High"
    elif urgency_index >= cfg.medium_threshold:
        priority_level = "Medium"
    else:
        priority_level = "Low"

    escalated = urgency_index > cfg.escalation_threshold or priority_level in ("High", "Emergency")
    sla_hours = SLA_HOURS_BY_PRIORITY[priority_level]

    reasons: List[str] = []
    if t1.intent == "Darurat":
        reasons.append("Intensi darurat terdeteksi")
    if f_flag == 1 and alpha > 0.0:
        reasons.append("Terindikasi sarkasme (eskalasi faktor α)")
    if s_score >= 7.0 and w3 > 0.0:
        reasons.append("Intensitas sentimen negatif tinggi")
    if c_score >= 8.0 and w1 > 0.0:
        reasons.append(f"Domain berisiko tinggi: {t1.domain}")
    if urgency_index > cfg.escalation_threshold:
        reasons.append(f"Melampaui ambang eskalasi U > {cfg.escalation_threshold:.2f}")
    if not reasons:
        reasons.append("Laporan rutin dalam batas SLA standar")

    return PriorityOutput(
        urgency_index=urgency_index,
        priority_level=priority_level,
        escalated=escalated,
        sla_hours=sla_hours,
        escalation_reasons=reasons,
        components={
            "C": round(c_score, 4),
            "I": round(i_score, 4),
            "S": round(s_score, 4),
            "F": float(f_flag),
        },
    )

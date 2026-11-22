from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from src.priority.engine import PriorityOutput
from src.schema import ComplaintEvent, EntitySpan, T1Result, T2Result, T3Result
from src.task3.data_loader import CODE_MIXING_FLAG_RE, NEGATION_FLAG_RE


DEFAULT_AGENCY_BY_DOMAIN: Dict[str, str] = {
    "Infrastruktur & PU": "Dinas PUPR / Dishub",
    "Kesehatan & Sanitasi": "Dinas Kesehatan",
    "Layanan & Administrasi": "Disdukcapil / Biro Pelayanan Publik",
    "Lingkungan": "Dinas Lingkungan Hidup (DLH)",
    "Pungli & Tata Kelola": "Inspektorat Daerah / Satpol PP",
}

REGION_HINTS = (
    "Semarang", "Surakarta", "Solo", "Kendal", "Cilacap", "Magelang", "Banyumas",
    "Purwokerto", "Demak", "Kudus", "Pati", "Brebes", "Kebumen", "Jepara",
    "Tegal", "Salatiga", "Pekalongan", "Klaten", "Boyolali", "Wonogiri",
    "Ungaran", "Ambarawa", "Tembalang", "Banyumanik", "Pedurungan", "Genuk",
    "Gunungpati", "Srondol", "Ngaliyan", "Meteseh", "Tugu", "Bumiayu", "Kroya",
    "Mranggen", "Kaliwungu", "Sukorejo", "Karanganyar", "Mangkang",
)


def route_agency(domain: str, entities: List[EntitySpan]) -> Dict[str, Any]:
    orgs = [e.text for e in entities if e.label == "ORGANIZATION"]
    if orgs:
        return {
            "assigned_agency": orgs[0],
            "routing_source": "ner_organization",
            "organizations_detected": orgs,
        }
    return {
        "assigned_agency": DEFAULT_AGENCY_BY_DOMAIN.get(domain, "Sekretariat Daerah"),
        "routing_source": "domain_default",
        "organizations_detected": [],
    }


def infer_admin_region(locations: List[str]) -> str:
    joined = " ".join(locations)
    for region in REGION_HINTS:
        if re.search(rf"\b{re.escape(region)}\b", joined, flags=re.IGNORECASE):
            return region
    if locations:
        return locations[0]
    return "Jawa Tengah / Umum"


def anonymize_reporter(reporter_id: Optional[str], text: str) -> str:
    seed_val = reporter_id if reporter_id else text
    digest = hashlib.sha256(seed_val.encode("utf-8")).hexdigest()[:12]
    return f"anon_{digest}"


def build_complaint_event(
    text: str,
    t1: T1Result,
    t2: T2Result,
    t3: T3Result,
    prio: PriorityOutput,
) -> ComplaintEvent:
    return ComplaintEvent(
        text=text,
        domain=t1.domain,
        intent=t1.intent,
        entities=[e.to_dict() for e in t2.entities],
        sentiment=t3.sentiment,
        sarcasm=bool(t3.sarcasm),
        urgency=float(prio.urgency_index),
        priority_level=prio.priority_level,
    )


def build_beam_event(
    text: str,
    t1: T1Result,
    t2: T2Result,
    t3: T3Result,
    prio: PriorityOutput,
    channel: str = "portal_web",
    reporter_id: Optional[str] = None,
    reported_at: Optional[str] = None,
    topic: Optional[Dict[str, Any]] = None,
    latency_ms: float = 0.0,
) -> Dict[str, Any]:
    now_dt = datetime.now(timezone.utc)
    if reported_at:
        try:
            rep_dt = datetime.fromisoformat(reported_at.replace("Z", "+00:00"))
        except ValueError:
            rep_dt = now_dt
    else:
        rep_dt = now_dt
    sla_deadline = rep_dt + timedelta(hours=prio.sla_hours)

    locations = [e.text for e in t2.entities if e.label == "LOCATION"]
    persons = [e.text for e in t2.entities if e.label == "PERSON"]
    routing = route_agency(t1.domain, t2.entities)
    anon_id = anonymize_reporter(reporter_id, text)
    event_hash = hashlib.sha1(f"{anon_id}:{text}".encode("utf-8")).hexdigest()[:16]

    return {
        "event_type": "Citizen Submits Civic Complaint Report",
        "event_id": f"beam_{event_hash}",
        "who": {
            "reporter_id": anon_id,
            "assigned_agency": routing["assigned_agency"],
            "routing_source": routing["routing_source"],
            "organizations_detected": routing["organizations_detected"],
            "persons_mentioned": persons,
        },
        "what": {
            "text": text,
            "domain": t1.domain,
            "intent": t1.intent,
            "sentiment": t3.sentiment,
            "entities": [e.to_dict() for e in t2.entities],
            "language_flags": {
                "negation": bool(NEGATION_FLAG_RE.search(text)),
                "code_mixing": bool(CODE_MIXING_FLAG_RE.search(text)),
                "sarcasm": bool(t3.sarcasm),
            },
            "topic": topic,
        },
        "when": {
            "reported_at": rep_dt.isoformat(),
            "inferred_at": now_dt.isoformat(),
            "sla_hours": prio.sla_hours,
            "sla_deadline": sla_deadline.isoformat(),
        },
        "where": {
            "locations": locations,
            "admin_region": infer_admin_region(locations),
        },
        "how_many": {
            "word_count": len(text.split()),
            "domain_confidence": t1.domain_confidence,
            "intent_confidence": t1.intent_confidence,
            "sentiment_score": t3.sentiment_score,
            "urgency_index": prio.urgency_index,
            "latency_ms": round(float(latency_ms), 4),
        },
        "why": {
            "escalated": prio.escalated,
            "priority_level": prio.priority_level,
            "escalation_reasons": prio.escalation_reasons,
            "sarcasm_triggered": bool(t3.sarcasm),
            "urgency_components": prio.components,
        },
        "how": {
            "channel": channel,
        },
    }

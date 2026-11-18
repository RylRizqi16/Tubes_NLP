"""
Kontrak Schema JSON Antar Modul - NLP Klasifikasi Pengaduan Publik Indonesia.
Digunakan oleh:
- T1 (Farhan): Domain & Intent
- T2 (Raka): NER Entities
- T3 (Darryl): Sentiment & Sarcasm
- Priority Engine & Pipeline Integration
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class EntitySpan:
    text: str
    label: str  # 'LOCATION' | 'ORGANIZATION' | 'PERSON'
    start: int
    end: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class T1Result:
    domain: str
    intent: str
    domain_confidence: Optional[float] = None
    intent_confidence: Optional[float] = None


@dataclass
class T2Result:
    entities: List[EntitySpan]


@dataclass
class T3Result:
    sentiment: str  # 'positive' | 'neutral' | 'negative'
    sarcasm: bool
    sentiment_score: Optional[float] = None


@dataclass
class ComplaintEvent:
    text: str
    domain: str
    intent: str
    entities: List[Dict[str, Any]]
    sentiment: str
    sarcasm: bool
    urgency: float
    priority_level: Optional[str] = None  # 'Low' | 'Medium' | 'High' | 'Emergency'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

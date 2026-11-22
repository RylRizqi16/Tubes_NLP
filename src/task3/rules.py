from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

from src.task3.data_loader import ID_TO_SENTIMENT, SENTIMENT_TO_ID


POSITIVE_WORDS = {
    "bagus", "baik", "mantap", "mantab", "hebat", "keren", "luar", "biasa",
    "terbaik", "top", "markotop", "salut", "juara", "cepat", "sigap", "gercep",
    "ramah", "bersih", "rapi", "nyaman", "lancar", "berhasil", "sukses",
    "membantu", "terbantu", "teratasi", "beres", "selesai", "puas", "memuaskan",
    "alhamdulillah", "terima", "kasih", "makasih", "apresiasi", "aman", "sehat",
    "terima_kasih", "luar_biasa", "sangat_bagus", "sangat_puas", "sukses_selalu",
}

NEGATIVE_WORDS = {
    "rusak", "parah", "berlubang", "lubang", "mati", "gelap", "macet", "error",
    "eror", "lemot", "lambat", "lelet", "lama", "molor", "antre", "antrean",
    "mengular", "bau", "busuk", "kotor", "sampah", "meluber", "banjir", "mampet",
    "tergenang", "menggenang", "bocor", "jebol", "putus", "roboh", "tumbang",
    "miring", "retak", "amblas", "licin", "mengelupas", "menjuntai", "bahaya",
    "membahayakan", "celaka", "tertabrak", "kritis", "darurat", "urgent",
    "kebakaran", "longsor", "keracunan", "wabah", "tercemar", "limbah",
    "kecewa", "mengecewakan", "kesal", "marah", "resah", "susah", "sulit",
    "kangelan", "gagal", "kosong", "kurang", "buruk", "jelek", "busuk",
    "percuma", "sia", "maksa", "pungli", "pelicin", "memeras", "premanisme",
    "oknum", "korupsi", "nilep", "ditipu", "penipuan", "cacat", "penyok",
    "hilang", "telat", "terlambat", "dibiarkan", "diabaikan", "terlewat",
    "terblokir", "tertahan", "mengganggu", "muntah", "diarsipkan", "nol",
    "hancur", "ancur", "bobrok", "brengsek", "payah", "kacau", "ruwet",
}

PRAISE_HYPERBOLE_CUES = {
    "mantap", "mantab", "hebat", "keren", "luar biasa", "terbaik", "top",
    "markotop", "salut", "juara", "juara satu", "selamat", "terima kasih",
    "makasih", "bravo", "prestasi", "prestasinya", "bagus banget", "keren banget",
    "hebat banget", "mantap sekali", "luar biasa memang", "sukses", "mengagumkan",
}

COMPLAINT_ISSUE_CUES = {
    "mati", "rusak", "berlubang", "lubang", "lubangnya", "macet", "error", "eror",
    "lemot", "lambat", "molor", "antre", "antrean", "nunggu", "menunggu", "jam",
    "berbulan-bulan", "berbulan", "bertahun", "minggu", "sampah", "bau", "banjir",
    "mampet", "jebol", "putus", "roboh", "tumbang", "retak", "amblas", "celaka",
    "hampir", "tertabrak", "pungli", "oknum", "pelicin", "ditipu", "diarsipkan",
    "dibiarkan", "diabaikan", "nol", "gagal", "kosong", "kacau", "mengelupas",
    "kangelan", "ora beres", "ora ditangani", "belum beres", "belum selesai",
    "belum ditangani", "tanpa tindakan", "tanpa kejelasan", "wisata baru", "kolam lele",
    "awet", "malah", "statusnya ditutup", "sabar ekstra", "pirang-pirang dina",
}

EXPLICIT_NEGATIONS = {
    "tidak", "tak", "bukan", "bukannya", "belum", "blm", "gak", "ga", "gk",
    "nggak", "ngak", "enggak", "ndak", "ora", "jangan", "tanpa", "anti",
}

IMPLICIT_NEGATION_PATTERNS = [
    re.compile(r"\bboro-boro\b", re.IGNORECASE),
    re.compile(r"\bmana\s+ada\b", re.IGNORECASE),
    re.compile(r"\bbelum\s+(?:pernah|juga|ada|beres|selesai|ditangani|direspon)\b", re.IGNORECASE),
    re.compile(r"\btidak\s+(?:pernah|kunjung|ada\s+(?:solusi|perkembangan|tindakan|balasan|kejelasan))\b", re.IGNORECASE),
    re.compile(r"\bora\s+(?:ono|iso|beres|ditangani)\b", re.IGNORECASE),
    re.compile(r"\b(?:bukan|bukannya)\s+(?:tidak|kami\s+tidak|bermaksud)\b", re.IGNORECASE),
    re.compile(r"\bjangan\s+sampai\b", re.IGNORECASE),
    re.compile(r"\btanpa\s+(?:kejelasan|tindakan|solusi|kepastian|karcis|kwitansi|dasar)\b", re.IGNORECASE),
]

SARCASM_PATTERNS = [
    re.compile(r"\b(?:mantap|mantab|hebat|keren|salut|juara|top|luar\s+biasa|selamat|terima\s+kasih)\b.{0,120}\b(?:mati|rusak|lubang|berlubang|nunggu|antre|jam|bulan|tahun|diarsipkan|ditipu|pungli|celaka|awet|kolam|wisata|malah|dibiarkan|tanpa\s+tindakan|parah)\b", re.IGNORECASE),
    re.compile(r"\b(?:itulah\s+hebatnya|prestasi(?:nya)?|lanjutkan\s+prestasinya|jangan\s+terlalu\s+cepat|kapan-kapan\s+.*\s+mampir|pengalaman\s+luar\s+biasa)\b", re.IGNORECASE),
    re.compile(r"\b(?:kasih\s+bintang|bintang\s+1|auto\s+bintang)\b", re.IGNORECASE),
    re.compile(r"(?::D|:v|😂|🤣|🙃).{0,40}\b(?:parah|rusak|bohong|antre|lama|nol|gagal)\b", re.IGNORECASE),
]

INTENSIFIERS = {
    "sangat", "amat", "banget", "bgt", "bener", "parah", "sekali", "tenan",
    "ekstra", "total", "terlalu", "paling", "sungguh", "betul",
}

TOKEN_RE = re.compile(r"[a-zA-Z0-9_-]+")


def tokenize_for_rules(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def analyze_rules(text: str) -> Dict[str, Any]:
    lowered = text.lower()
    tokens = tokenize_for_rules(text)

    explicit_neg_hits = [t for t in tokens if t in EXPLICIT_NEGATIONS]
    implicit_neg_hits = [p.pattern for p in IMPLICIT_NEGATION_PATTERNS if p.search(lowered)]

    score = 0.0
    negation_window = 0
    intensifier_boost = 1.0
    pos_hits = 0
    neg_hits = 0

    for tok in tokens:
        if tok in EXPLICIT_NEGATIONS:
            negation_window = 3
            continue
        if tok in INTENSIFIERS:
            intensifier_boost = 1.5
            continue

        polarity = 0.0
        if tok in POSITIVE_WORDS:
            polarity = 1.0
            pos_hits += 1
        elif tok in NEGATIVE_WORDS:
            polarity = -1.0
            neg_hits += 1

        if polarity != 0.0:
            if negation_window > 0:
                polarity = -0.9 * polarity
            score += polarity * intensifier_boost
            intensifier_boost = 1.0

        if negation_window > 0:
            negation_window -= 1

    if implicit_neg_hits:
        score -= 1.5 * len(implicit_neg_hits)

    praise_hits = [c for c in PRAISE_HYPERBOLE_CUES if re.search(rf"\b{re.escape(c)}\b", lowered)]
    issue_hits = [c for c in COMPLAINT_ISSUE_CUES if re.search(rf"\b{re.escape(c)}\b", lowered)]
    sarcasm_pattern_hits = [p.pattern for p in SARCASM_PATTERNS if p.search(lowered)]

    sarcasm_triggered = bool(
        sarcasm_pattern_hits or (praise_hits and (issue_hits or neg_hits > 0 or implicit_neg_hits))
    )

    if sarcasm_triggered:
        score = min(score, -1.8)

    if score <= -0.5:
        rule_sentiment = "Negatif"
    elif score >= 0.6 and not implicit_neg_hits:
        rule_sentiment = "Positif"
    else:
        rule_sentiment = "Netral"

    raw_intensity = min(1.0, max(0.0, (-score) / 4.0))
    if rule_sentiment == "Positif":
        raw_intensity = min(0.15, raw_intensity)
    elif rule_sentiment == "Netral":
        raw_intensity = max(0.30, min(0.55, 0.40 + (-score) * 0.08))

    return {
        "lexicon_score": round(score, 4),
        "pos_hits": pos_hits,
        "neg_hits": neg_hits,
        "explicit_negation_triggered": bool(explicit_neg_hits),
        "implicit_negation_triggered": bool(implicit_neg_hits),
        "negation_triggered": bool(explicit_neg_hits or implicit_neg_hits),
        "sarcasm_rule_triggered": sarcasm_triggered,
        "praise_cues": praise_hits,
        "issue_cues": issue_hits,
        "rule_sentiment": rule_sentiment,
        "rule_neg_intensity": round(raw_intensity, 4),
    }


def predict_e1(text: str) -> Dict[str, Any]:
    info = analyze_rules(text)
    return {
        "sentiment": info["rule_sentiment"],
        "is_sarcasm": 1 if info["sarcasm_rule_triggered"] else 0,
        "sentiment_score": info["rule_neg_intensity"],
        "rule_details": info,
    }


def apply_rule_engine(
    text: str,
    sent_probs: Sequence[float],
    sarc_prob: float,
    use_negation: bool = True,
    use_sarcasm: bool = True,
    sarcasm_threshold: float = 0.50,
) -> Dict[str, Any]:
    probs: List[float] = [float(p) for p in sent_probs]
    total = sum(probs)
    if total > 0:
        probs = [p / total for p in probs]
    else:
        probs = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

    info = analyze_rules(text)
    model_pred_id = max(range(3), key=lambda i: probs[i])
    model_sentiment = ID_TO_SENTIMENT[model_pred_id]
    model_sarcasm = 1 if sarc_prob >= sarcasm_threshold else 0

    final_probs = list(probs)
    final_sarcasm = model_sarcasm
    negation_override = False
    sarcasm_override = False

    if use_negation:
        if info["implicit_negation_triggered"] and model_sentiment in {"Positif", "Netral"}:
            if probs[SENTIMENT_TO_ID["Negatif"]] >= 0.15 or info["neg_hits"] > 0:
                shift = 0.35 if model_sentiment == "Positif" else 0.28
                final_probs[0] = min(0.98, final_probs[0] + shift)
                final_probs[1] = max(0.01, final_probs[1] - shift * 0.5)
                final_probs[2] = max(0.01, final_probs[2] - shift * 0.5)
                negation_override = True
        elif info["explicit_negation_triggered"] and info["rule_sentiment"] == "Negatif":
            if model_sentiment == "Positif" and probs[2] < 0.78:
                final_probs[0] = min(0.95, final_probs[0] + 0.30)
                final_probs[2] = max(0.02, final_probs[2] - 0.30)
                negation_override = True

    norm = sum(final_probs)
    final_probs = [p / norm for p in final_probs]

    if use_sarcasm:
        if info["sarcasm_rule_triggered"]:
            final_sarcasm = 1
            sarcasm_override = True
        elif sarc_prob >= 0.35 and info["praise_cues"] and info["issue_cues"]:
            final_sarcasm = 1
            sarcasm_override = True

        if final_sarcasm == 1 and ID_TO_SENTIMENT[max(range(3), key=lambda i: final_probs[i])] == "Positif":
            final_probs = [
                max(final_probs[0], 0.72),
                min(final_probs[1], 0.18),
                min(final_probs[2], 0.10),
            ]
            s_norm = sum(final_probs)
            final_probs = [p / s_norm for p in final_probs]
            sarcasm_override = True

    final_id = max(range(3), key=lambda i: final_probs[i])
    final_sentiment = ID_TO_SENTIMENT[final_id]

    neg_intensity = final_probs[0] + 0.4 * final_probs[1]
    if final_sarcasm == 1:
        neg_intensity = max(neg_intensity, 0.75)
    neg_intensity = round(min(1.0, max(0.0, neg_intensity)), 4)

    return {
        "sentiment": final_sentiment,
        "is_sarcasm": final_sarcasm,
        "sentiment_score": neg_intensity,
        "sent_probs": [round(p, 6) for p in final_probs],
        "sarc_prob": round(float(sarc_prob), 6),
        "negation_override": negation_override,
        "sarcasm_override": sarcasm_override,
        "rule_details": info,
    }

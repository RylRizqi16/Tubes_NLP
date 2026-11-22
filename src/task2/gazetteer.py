"""
Indonesian Civic Regex, Gazetteer & Boundary Post-Processing Module (`src/task2/gazetteer.py`).

Used by:
- **E1 (CPU Baseline)**: High-precision civic regex + gazetteer extractor combined with
  a trained token/phrase BIO model.
- **E3 & E4b (Hybrid Post-Processing)**: Cased-text boundary cleanup, WordPiece continuation
  repair, 3-gate civic address `, ` merger, and `Kelurahan`/`Kecamatan` Org-vs-Loc
  disambiguation on top of IndoBERT predictions.
- **LDA Preprocessing (`train_lda.py`)**: Generic synthetic city/person/slot stopword
  set so topic models cluster around complaint semantics rather than random geographic slots.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


# ---------------------------------------------------------------------------
# 1. Civic Prefixes, Cues & Regex Patterns (Original Cased Text)
# ---------------------------------------------------------------------------

# Note: `Jl.` ends with `.`, so it must be matched before `\s+` (not `\b`, which fails between `.` and `' '`).
# Spelled-out `Jalan` is intentionally excluded from comma-merging because real news NER
# (`indolem_nerui`, `idner_news_2k`) always splits `"Jalan Sudirman, Jakarta"` into two spans,
# whereas civic complaint addresses use `"Jl. <Street>, <Area>"` or
# `"Kecamatan/Kelurahan/Desa/Pasar/Terminal/Stasiun/Simpang <X>, <Y>"`.
CIVIC_LOC_PREFIX_RE = re.compile(
    r"^(?:(?:Jl\.|Kec\.|Kel\.)(?=\s|$)|(?:Kecamatan|Kelurahan|Desa|Pasar|Terminal|Stasiun|Simpang|Alun-Alun|Alun\s*-\s*Alun|Pelabuhan)\b)",
    re.IGNORECASE,
)

HIERARCHICAL_SECOND_SPAN_RE = re.compile(
    r"^(?:(?:Kel\.|Kec\.|Kab\.|Prov\.|Jl\.)(?=\s|$)|(?:Desa|Kelurahan|Kecamatan|Kabupaten|Kota|Provinsi|Distrik|Jalan)\b)",
    re.IGNORECASE,
)

ABBREV_JL_PREFIX_RE = re.compile(r"^Jl\.\s+\S", re.IGNORECASE)

# Explicit civic agency pattern for `has_civic_org` (excludes political parties or news orgs like PDIP/Kemenhub)
CIVIC_ORG_SPAN_RE = re.compile(
    r"\b(?:Dishub|Disdukcapil|DLH|PDAM|BPBD|RSUD|Puskesmas|Polsek|Samsat|Dinas\s+(?:PUPR|Kesehatan|Sosial|Pendidikan)|"
    r"Kantor\s+(?:Kecamatan|Kelurahan)|Satpol\s+PP|Inspektorat\s+Daerah|PLN\s+UP3)\b",
    re.IGNORECASE,
)

# Sentence-level civic complaint cues for Gate 3 of the 3-gate `, ` address merger
CIVIC_COMPLAINT_SENTENCE_RE = re.compile(
    r"(?:\b(?:mohon|tolong|tolonglah|tulung|warga|lapor|laporan|keluhan|aduan|aduannya|perbaiki|diperbaiki|"
    r"ditangani|menindaklanjuti|ditindaklanjuti|oknum|darurat|urgent|bahaya|"
    r"berlubang|sampah|mampet|antrean|pungli|pungutan\s+liar|dishub|disdukcapil|"
    r"puskesmas|rsud|pdam|dlh|bpbd|satpol|polsek|samsat|inspektorat|"
    r"dinas\s+(?:pupr|kesehatan|sosial|pendidikan)|masukan\s+untuk|usul\s+kepada|"
    r"saran\s+(?:buat|kami)|terima\s+kasih|apresiasi|permisi\s+admin|alhamdulillah|"
    r"hebat\s+(?:banget|sekali)|keren\s+banget|top\s+markotop|juara\s+satu|"
    r"luar\s+biasa|bukan\s+bermaksud|waduh|aduh|piye\s+iki)\b|Jl\.\s+[A-Z])",
    re.IGNORECASE,
)

# Discourse / complaint words that must never be absorbed as the tail of a proper name
NON_ENTITY_TAIL_WORDS: Set[str] = {
    "Mohon", "Tolong", "Tulung", "Terima", "Masalah", "Kinerja", "Warga", "Sekarang",
    "Sudah", "Sampai", "Kata", "Jangan", "Segera", "Kondisi", "Laporan", "Saran",
    "Usul", "Masukan", "Apresiasi", "Salut", "Keren", "Hebat", "Mantap", "Waduh",
    "Aduh", "Halo", "Piye", "Min", "Permisi", "Info", "Hingga", "Bukan", "Bukannya",
    "Kami", "Saya", "Selamat", "Luar", "Wah", "Juara", "Alhamdulillah", "Bahaya",
    "Darurat", "Urgent",
}

# Preceding context cues (last ~45 chars before span) for Kelurahan/Kecamatan disambiguation
LOCATIVE_PRECEDING_RE = re.compile(
    r"(?:\b(?:di|warga|kondisi|wilayah|area|daerah|meninjau|lokasi|pelayanan|keren\s+ya|"
    r"turun\s+ke|bergerak\s+ke|sop\s+di|cek)\s+)$",
    re.IGNORECASE,
)

INSTITUTIONAL_PRECEDING_RE = re.compile(
    r"(?:\b(?:tim|petugas|kinerja|kerja|bantuan|pertimbangan|kepada|untuk|oleh|bersama|"
    r"terima\s+kasih|hebat\s+banget|luar\s+biasa(?:\s+memang)?|tolong(?:\s+segera)?|"
    r"tulung|mohon|minta|meminta|solusi\s+dari|sekali\s+dari|cepat\s+dari|saran\s+buat|"
    r"saran\s+kami|usul\s+kepada|masukan\s+untuk|lapor\s+berkali-kali\s+ke|resmi\s+ke|"
    r"info\s+ke|terusin\s+ke|saat\s+ini|kapan-kapan|diselesaikan|baru|halo|waduh|aduh|"
    r"piye\s+iki)\s+)$",
    re.IGNORECASE,
)

KANTOR_PREFIX_LOOKBEHIND_RE = re.compile(r"(?:^|\s)(Kantor\s+)$", re.IGNORECASE)

# High-precision cased regex extractors for civic complaints (E1 & E3/E4b recall recovery).
# Note: Token characters exclude `.` so patterns never bridge across sentence-ending periods.
CASED_CIVIC_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    # PERSON: Indonesian honorifics + Capitalized Name(s) (1 to 2 tokens after honorific)
    (
        "PERSON",
        re.compile(
            r"\b(?:Pak|Bu|Bapak|Ibu|Mas|Mbak)\s+[A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'-]+)?\b"
        ),
    ),
    # ORGANIZATION: Kantor Kecamatan / Kelurahan + Name
    (
        "ORGANIZATION",
        re.compile(
            r"\bKantor\s+(?:Kecamatan|Kelurahan)\s+[A-Z][a-zA-Z0-9-]+(?:\s+[A-Z][a-zA-Z0-9-]+)?\b"
        ),
    ),
    # ORGANIZATION: Explicit Indonesian civic agencies
    (
        "ORGANIZATION",
        re.compile(
            r"\b(?:Dishub|Disdukcapil|DLH|BPBD|RSUD|Puskesmas|Polsek|Samsat)\s+"
            r"(?:Kota\s+|Kabupaten\s+|Provinsi\s+)?[A-Z][a-zA-Z0-9-]+(?:\s+[A-Z][a-zA-Z0-9-]+)?\b"
        ),
    ),
    (
        "ORGANIZATION",
        re.compile(
            r"\bDinas\s+(?:PUPR|Kesehatan|Sosial|Pendidikan|Perhubungan|Lingkungan\s+Hidup)"
            r"(?:\s+(?:Kota|Kabupaten|Provinsi))?(?:\s+[A-Z][a-zA-Z0-9-]+){0,2}\b"
        ),
    ),
    (
        "ORGANIZATION",
        re.compile(
            r"\b(?:PDAM\s+Tirta\s+Moedal|PLN\s+UP3\s+[A-Z][a-zA-Z]+|Satpol\s+PP|Inspektorat\s+Daerah)\b"
        ),
    ),
    # LOCATION: Abbreviated Jl. Street + optional ", Area"
    (
        "LOCATION",
        re.compile(
            r"\bJl\.\s+[A-Z0-9][a-zA-Z0-9-]*(?:\s+[A-Z0-9][a-zA-Z0-9-]*){0,2}"
            r"(?:,\s*[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})?\b"
        ),
    ),
    # LOCATION: Civic landmarks / administrative areas (with optional ", Area" gated on context)
    (
        "LOCATION",
        re.compile(
            r"\b(?:Kecamatan|Kelurahan|Desa|Kabupaten|Pasar|Terminal|Stasiun|Pelabuhan|Simpang\s+Lima|Alun-Alun)\s+"
            r"[A-Z][a-zA-Z0-9-]+(?:\s+[A-Z][a-zA-Z0-9-]+){0,2}"
            r"(?:,\s*[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})?\b"
        ),
    ),
    (
        "LOCATION",
        re.compile(
            r"\bSimpang\s+Lima(?:,\s*[A-Z][a-zA-Z]+)?\b"
        ),
    ),
]


# ---------------------------------------------------------------------------
# 2. Synthetic Slot Stopwords for LDA (prevents proper-noun slot clustering)
# ---------------------------------------------------------------------------

SYNTHETIC_ENTITY_SLOT_STOPWORDS: Set[str] = {
    # Honorifics & titles
    "pak", "bu", "bapak", "ibu", "mas", "mbak",
    # Synthetic person names
    "budi", "siti", "hendra", "agung", "joko", "rina", "slamet", "dewi",
    "agus", "lestari", "wahyu", "rizki", "arif", "endang", "eko", "fajar",
    "wati", "darmawan", "nurul", "teguh", "putri", "cahyo",
    # Synthetic Central Java / city / street proper nouns randomly sampled across all 5 domains
    "semarang", "surakarta", "solo", "tembalang", "banyumanik", "gunungpati",
    "sekaran", "kendal", "ungaran", "purwokerto", "demak", "karanganyar",
    "salatiga", "pekalongan", "cilacap", "magelang", "mangkang", "tugu",
    "tirtonadi", "pati", "bumiayu", "brebes", "tawang", "kudus", "klaten",
    "sukorejo", "boyolali", "pandanaran", "pedurungan", "kebumen", "jepara",
    "ngaliyan", "tegal", "klewer", "ambarawa", "kaliwungu", "siliwangi",
    "setiabudi", "srondol", "genuk", "gajahmada", "wonogiri", "meteseh",
    "sumoharjo", "kroya", "mranggen", "banyumas", "tugurejo", "moewardi",
    "moedal", "tirta", "melati", "pemuda", "diponegoro", "kartini", "katamso",
    "sudirman", "riyadi", "veteran", "pahlawan", "jawa", "tengah", "barat",
    "timur", "utara", "selatan",
}


# ---------------------------------------------------------------------------
# 3. Span Cleanup, Disambiguation & 3-Gate Comma Address Merger
# ---------------------------------------------------------------------------

def _trim_non_entity_tail_words(text: str, start: int, end: int) -> int:
    """Trim trailing discourse/complaint starter words if greedily matched at span end."""
    cur_end = end
    while cur_end > start:
        substr = text[start:cur_end]
        parts = substr.split()
        if len(parts) <= 1:
            break
        last_word = parts[-1].strip(".,:;!?)\"'")
        if last_word in NON_ENTITY_TAIL_WORDS:
            idx = substr.rfind(parts[-1])
            if idx > 0:
                cur_end = start + idx
                while cur_end > start and text[cur_end - 1].isspace():
                    cur_end -= 1
                continue
        break
    return cur_end


def snap_and_clean_span_bounds(
    text: str,
    start: int,
    end: int,
    strip_trailing_period: bool = False,
) -> Optional[Tuple[int, int]]:
    """
    Clean whitespace and stray boundary punctuation on a candidate span `(start, end)`.
    Also snaps partial mid-word subword cuts to complete alphanumeric boundaries,
    and preserves matched parenthesized nicknames such as `'(Jokowi)'` or `'(SBY)'`.
    """
    n = len(text)
    s = max(0, min(n, int(start)))
    e = max(0, min(n, int(end)))
    if s >= e:
        return None

    # Snap outward if a subword split cut through the middle of an alphanumeric token
    while s > 0 and text[s - 1].isalnum() and text[s].isalnum():
        s -= 1
    while e < n and text[e - 1].isalnum() and text[e].isalnum():
        e += 1

    # Strip leading/trailing whitespace
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1

    # Strip stray leading/trailing commas, colons, semicolons, exclamation/question marks
    while s < e and text[s] in ",:;!?":
        s += 1
    while e > s and text[e - 1] in ",:;!?":
        e -= 1

    # Handle parentheses: if span starts with '(' (e.g. '(Jokowi' in idner_news_2k)
    # and the immediately following char in text is ')', snap e += 1 to complete '(Jokowi)'
    substr = text[s:e]
    if substr.startswith("(") and ")" not in substr:
        if e < n and text[e] == ")":
            e += 1
        else:
            s += 1
    elif substr.endswith(")") and "(" not in substr:
        if s > 0 and text[s - 1] == "(":
            s -= 1
        else:
            e -= 1

    if strip_trailing_period:
        substr = text[s:e]
        if substr.endswith(".") and not re.search(
            r"(?:Tbk|Ltd|Inc|Corp|Co|Jr|Sr|Dr|Prof|Ir|St|Jl|[A-Z])\.$", substr
        ):
            e -= 1

    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1

    if s >= e:
        return None
    return s, e


def disambiguate_kelurahan_kecamatan(
    text: str,
    start: int,
    end: int,
    label: str,
    is_civic_sentence: bool = True,
) -> Tuple[int, int, str]:
    """
    Apply `Kelurahan / Kecamatan` Org-vs-Loc disambiguation rules:
    1. When `is_civic_sentence=True`:
       - If preceded by `'Kantor '` (e.g., `'Kantor Kecamatan Genuk'`, `'Kantor Kelurahan Ngaliyan'`),
         expand left to include `'Kantor '` and force label = `'ORGANIZATION'`.
       - If span starts with `'Kantor Kecamatan'` or `'Kantor Kelurahan'`, force `'ORGANIZATION'`.
       - If span starts with bare `'Kelurahan '` or `'Kecamatan '`:
         - If it contains `', '` (e.g. `'Kelurahan Sekaran, Gunungpati'`), force `'LOCATION'`.
         - Else if preceded by locative cues (`di`, `warga`, `kondisi`, `wilayah`, `area`, `daerah`,
           `meninjau`, `turun ke`, `bergerak ke`, `pelayanan`, `cek`), force `'LOCATION'`.
         - Else if preceded by institutional cues (`tim`, `petugas`, `kinerja`, `kerja`, `bantuan`,
           `kepada`, `terima kasih`, `oleh`, `mohon`, `tolong`, `saran`, `usul`, `masukan`, `dari`),
           set `'ORGANIZATION'`.
         - Default for bare `'Kelurahan/Kecamatan'`: `'LOCATION'`.
    2. When `is_civic_sentence=False` (general news/encyclopedia text like `nergrit_1723`):
       - Preserves original span boundaries and label without expanding `'Kantor '`.
    """
    if not is_civic_sentence:
        return start, end, label

    span_str = text[start:end]

    # Check if "Kantor " immediately precedes a bare Kecamatan/Kelurahan span
    if re.match(r"^(?:Kecamatan|Kelurahan)\b", span_str, re.IGNORECASE):
        prefix_window = text[max(0, start - 10) : start]
        m_kantor = KANTOR_PREFIX_LOOKBEHIND_RE.search(prefix_window)
        if m_kantor:
            new_start = start - len(m_kantor.group(1))
            return new_start, end, "ORGANIZATION"

    if re.match(r"^Kantor\s+(?:Kecamatan|Kelurahan)\b", span_str, re.IGNORECASE):
        return start, end, "ORGANIZATION"

    if re.match(r"^(?:Kecamatan|Kelurahan)\b", span_str, re.IGNORECASE):
        if "," in span_str:
            return start, end, "LOCATION"
        before_ctx = text[max(0, start - 45) : start]
        if LOCATIVE_PRECEDING_RE.search(before_ctx):
            return start, end, "LOCATION"
        if INSTITUTIONAL_PRECEDING_RE.search(before_ctx):
            return start, end, "ORGANIZATION"
        return start, end, "LOCATION"

    return start, end, label


def should_merge_comma_locations(
    text: str,
    loc1_start: int,
    loc1_end: int,
    loc2_start: int,
    loc2_end: int,
    has_civic_org_in_sentence: bool = False,
) -> bool:
    """
    Evaluate the 3-Gate Civic Address `, ` Merger condition for two consecutive
    `LOCATION` spans `Loc1` and `Loc2`:

    - Must be separated strictly by `", "` (`text[loc1_end:loc2_start] == ", "`).
    - Multi-comma guard: Neither `Loc1` nor `Loc2` may already contain a comma `","`
      (prevents triple-chaining across `A, B, C`).
    - **Gate 1 (Civic prefix on Loc1)**: `Loc1` starts with `Jl.`, `Kecamatan`,
      `Kelurahan`, `Desa`, `Pasar`, `Terminal`, `Stasiun`, `Simpang`, `Alun-Alun`, or `Pelabuhan`.
    - **Gate 2 (Second-span hierarchical exclusion)**: `Loc2` must NOT start with
      `Desa`, `Kelurahan`, `Kecamatan`, `Kabupaten`, `Kota`, `Provinsi`, `Distrik`, `Jalan`, or `Jl.`
      (protects 89 / 102 real NER pairs such as `'Desa Sanggeng, Kecamatan Manokwari Barat'`).
    - **Gate 3 (Fast-path vs. Civic-context gate)**:
      - Fast-path: Always merge if `Loc1` starts with abbreviated `"Jl."` (904 in synthetic, 0 in real).
      - Otherwise: Merge only when sentence-level civic complaint cues (`CIVIC_COMPLAINT_SENTENCE_RE`)
        or an explicit civic agency (`has_civic_org_in_sentence`) are present in `text`.
    """
    if text[loc1_end:loc2_start] != ", ":
        return False

    loc1_text = text[loc1_start:loc1_end]
    loc2_text = text[loc2_start:loc2_end]

    # Multi-comma chaining guard (never chain A, B, C)
    if "," in loc1_text or "," in loc2_text:
        return False

    # Gate 1: Civic prefix on Loc1
    if not CIVIC_LOC_PREFIX_RE.search(loc1_text):
        return False

    # Gate 2: Second-span hierarchical exclusion
    if HIERARCHICAL_SECOND_SPAN_RE.search(loc2_text):
        return False
    if len(loc2_text.split()) > 3:
        return False

    # Gate 3: Fast-path for abbreviated "Jl." or sentence-level civic complaint cues
    if ABBREV_JL_PREFIX_RE.search(loc1_text):
        return True

    if has_civic_org_in_sentence or CIVIC_COMPLAINT_SENTENCE_RE.search(text):
        return True

    return False


def extract_civic_gazetteer_spans(text: str) -> List[Dict[str, Any]]:
    """
    Extract high-precision civic entity spans (`LOCATION`, `ORGANIZATION`, `PERSON`)
    directly from cased text using `CASED_CIVIC_PATTERNS` and disambiguation rules.
    Resolves overlaps by keeping longer matches first.
    """
    has_civic_ctx = bool(CIVIC_COMPLAINT_SENTENCE_RE.search(text))
    candidates: List[Dict[str, Any]] = []

    for raw_label, pattern in CASED_CIVIC_PATTERNS:
        for m in pattern.finditer(text):
            s, e = m.start(), m.end()
            e = _trim_non_entity_tail_words(text, s, e)
            if s >= e:
                continue
            matched_str = text[s:e]

            # If a LOCATION match includes ", <Area>", verify Gate 1–3
            if raw_label == "LOCATION" and "," in matched_str:
                comma_idx = matched_str.index(",")
                loc1_end = s + comma_idx
                loc2_start = loc1_end + 1
                while loc2_start < e and text[loc2_start].isspace():
                    loc2_start += 1
                if not should_merge_comma_locations(
                    text,
                    s,
                    loc1_end,
                    loc2_start,
                    e,
                    has_civic_org_in_sentence=has_civic_ctx,
                ):
                    e = loc1_end

            bounds = snap_and_clean_span_bounds(text, s, e, strip_trailing_period=True)
            if bounds is None:
                continue
            s, e = bounds
            s, e, final_label = disambiguate_kelurahan_kecamatan(
                text, s, e, raw_label, is_civic_sentence=True
            )
            candidates.append(
                {
                    "start": s,
                    "end": e,
                    "label": final_label,
                    "text": text[s:e],
                }
            )

    candidates.sort(key=lambda sp: (-(sp["end"] - sp["start"]), sp["start"]))
    accepted: List[Dict[str, Any]] = []

    for cand in candidates:
        c_start, c_end = cand["start"], cand["end"]
        overlaps = any(
            max(c_start, acc["start"]) < min(c_end, acc["end"])
            for acc in accepted
        )
        if not overlaps:
            accepted.append(cand)

    accepted.sort(key=lambda sp: (sp["start"], sp["end"]))
    return accepted


def apply_civic_postprocessing(
    text: str,
    raw_spans: Sequence[Dict[str, Any]],
    inject_gazetteer: bool = True,
) -> List[Dict[str, Any]]:
    """
    Apply full E3 / E4b Cased-Text Civic Gazetteer & Boundary Post-Processing to decoded spans:
    1. Snap & clean character boundaries (fix partial-word subword cuts and stray punctuation).
    2. Disambiguate `Kantor Kecamatan/Kelurahan` vs bare `Kelurahan/Kecamatan` (`ORG` vs `LOC`)
       when in civic complaint context.
    3. Optionally inject high-precision cased civic gazetteer matches (`Pak/Bu`, `Jl. X, Y`,
       `Dishub/Disdukcapil/DLH/Puskesmas/RSUD/BPBD/Kantor Kecamatan`) when the sentence has
       civic complaint cues. Only replaces existing overlapping model spans when the gazetteer
       span strictly expands/merges fragmented boundaries (never overwriting an identical-boundary
       model prediction's label).
    4. Apply the 3-Gate Civic Address `, ` Merger to adjacent `LOCATION` spans separated by `", "`.
    5. Deduplicate and enforce strict non-overlapping sorted order `(start, end)`.
    """
    is_civic_sentence = bool(CIVIC_COMPLAINT_SENTENCE_RE.search(text))
    cleaned: List[Dict[str, Any]] = []

    for sp in raw_spans:
        bounds = snap_and_clean_span_bounds(
            text,
            int(sp["start"]),
            int(sp["end"]),
            strip_trailing_period=is_civic_sentence,
        )
        if bounds is None:
            continue
        s, e = bounds
        if is_civic_sentence:
            e = _trim_non_entity_tail_words(text, s, e)
            if s >= e:
                continue
        s, e, lbl = disambiguate_kelurahan_kecamatan(
            text, s, e, str(sp["label"]), is_civic_sentence=is_civic_sentence
        )
        cleaned.append({"start": s, "end": e, "label": lbl, "text": text[s:e]})

    # Resolve any overlaps introduced by Kantor prefix expansion (keep longer span)
    cleaned.sort(key=lambda sp: (-(sp["end"] - sp["start"]), sp["start"]))
    deduped: List[Dict[str, Any]] = []
    for sp in cleaned:
        if not any(max(sp["start"], ex["start"]) < min(sp["end"], ex["end"]) for ex in deduped):
            deduped.append(sp)

    # Step 3: Inject high-precision cased civic gazetteer spans when sentence is a civic complaint
    if inject_gazetteer and is_civic_sentence:
        gaz_spans = extract_civic_gazetteer_spans(text)
        for gsp in gaz_spans:
            gs, ge = gsp["start"], gsp["end"]
            overlapping = [
                ex for ex in deduped if max(gs, ex["start"]) < min(ge, ex["end"])
            ]
            if not overlapping:
                deduped.append(gsp)
            else:
                # Only replace overlapping model spans when the gazetteer match strictly expands
                # a truncated span or merges >= 2 fragmented spans (never override an equal-boundary span's label)
                all_enclosed = all(gs <= ex["start"] and ex["end"] <= ge for ex in overlapping)
                strictly_wider = len(overlapping) >= 2 or (
                    (ge - gs) > (overlapping[0]["end"] - overlapping[0]["start"])
                )
                if all_enclosed and strictly_wider:
                    deduped = [
                        ex for ex in deduped if not (max(gs, ex["start"]) < min(ge, ex["end"]))
                    ]
                    deduped.append(gsp)

    deduped.sort(key=lambda sp: (sp["start"], sp["end"]))

    # Step 4: Apply 3-Gate Civic Address `, ` Merger across consecutive LOCATION spans
    has_civic_org = any(
        sp["label"] == "ORGANIZATION" and bool(CIVIC_ORG_SPAN_RE.search(sp["text"]))
        for sp in deduped
    )
    merged: List[Dict[str, Any]] = []
    for sp in deduped:
        if (
            merged
            and merged[-1]["label"] == "LOCATION"
            and sp["label"] == "LOCATION"
            and should_merge_comma_locations(
                text,
                merged[-1]["start"],
                merged[-1]["end"],
                sp["start"],
                sp["end"],
                has_civic_org_in_sentence=has_civic_org,
            )
        ):
            new_start = merged[-1]["start"]
            new_end = sp["end"]
            merged[-1] = {
                "start": new_start,
                "end": new_end,
                "label": "LOCATION",
                "text": text[new_start:new_end],
            }
        else:
            merged.append(sp)

    return merged

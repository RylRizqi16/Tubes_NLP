"""
Task 2 Phase 3 — LDA Topic Modeling Pipeline (`src/task2/train_lda.py`).

Fits strictly on `split == "train"` (6,468 docs across `campus_complaints`,
`synthetic`, and `sp4n_lapor`) with zero leakage from `val`/`test`.

Features:
1. 4-Layer Stopword & Normalization Pipeline (independent of NER text):
   - Layer 1: Indonesian stopwords
   - Layer 2: English stopwords (handles 17.3% bilingual `campus_complaints` rows)
   - Layer 3: Indonesian colloquial slang & abbreviations
   - Layer 4: Complaint boilerplate, anonymization masks (`[NAMA]`, `[NIM]`), and
     generic synthetic proper-noun slots (`SYNTHETIC_ENTITY_SLOT_STOPWORDS`)
2. Bigram construction (`gensim.models.Phrases` or pure-Python PMI bigram fallback)
   + dictionary filtering (`no_below=10`, `no_above=0.35`, `keep_n=10000`).
3. Topic count sweep over `K in {5, 8, 10, 12, 15}`, selecting best `K` by
   `C_v` coherence (with topic diversity tie-breaker).
4. Diagnostics: `C_v`, `U_mass`, topic diversity (unique top-10 words ratio),
   held-out `val`/`test` perplexity, `Topic x 5 Domains` cross-tabulation
   (Topic Purity + NMI, overall + split by `synthetic` vs. `campus_complaints`),
   and `Topic x Source` cross-tabulation.
5. Saves winning model artifacts to `models/task2/lda/` and report to
   `outputs/task2_lda_report.json`.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import joblib
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import normalized_mutual_info_score

from src.task2.data_loader import DEFAULT_MASTER_PATH, LDA_SOURCES, ROOT_DIR, load_lda_records
from src.task2.gazetteer import SYNTHETIC_ENTITY_SLOT_STOPWORDS


DEFAULT_LDA_MODEL_DIR = ROOT_DIR / "models" / "task2" / "lda"
DEFAULT_LDA_REPORT_PATH = ROOT_DIR / "outputs" / "task2_lda_report.json"
DEFAULT_K_VALUES: Tuple[int, ...] = (5, 8, 10, 12, 15)

DOMAINS: Tuple[str, ...] = (
    "Infrastruktur & PU",
    "Kesehatan & Sanitasi",
    "Layanan & Administrasi",
    "Lingkungan",
    "Pungli & Tata Kelola",
)

# ---------------------------------------------------------------------------
# 1. 4-Layer Stopword Sets & Text Normalization
# ---------------------------------------------------------------------------

INDONESIAN_STOPWORDS: Set[str] = {
    "ada", "adalah", "adanya", "adapun", "agak", "agaknya", "agar", "akan", "akankah",
    "akhir", "akhiri", "akhirnya", "aku", "akulah", "amat", "amatlah", "anda", "andalah",
    "antar", "antara", "antaranya", "apa", "apaan", "apabila", "apakah", "apalagi", "apatah",
    "artinya", "asal", "asalkan", "atas", "atau", "ataukah", "ataupun", "awal", "awalnya",
    "bagai", "bagaikan", "bagaimana", "bagaimanakah", "bagaimanapun", "bagi", "bagian",
    "bahkan", "bahwa", "bahwasanya", "baik", "bakal", "bakalan", "balik", "banyak", "baru",
    "bawah", "beberapa", "begini", "beginian", "beginikah", "beginilah", "begitu", "begitukah",
    "begitulah", "begitupun", "bekerja", "belakang", "belakangan", "belum", "belumlah", "benar",
    "benarkah", "benarlah", "berada", "berakhir", "berakhirlah", "berakhirnya", "berapa",
    "berapakah", "berapalah", "berapapun", "berarti", "berawal", "berbagai", "berdatangan",
    "beri", "berikan", "berikut", "berikutnya", "berjumlah", "berkali", "berkata", "berkehendak",
    "berkeinginan", "berkenaan", "berlainan", "berlalu", "berlangsung", "berlebihan", "bermacam",
    "bermaksud", "bermula", "bersama", "bersiap", "bertanya", "berturut", "bertutur", "berujar",
    "berupa", "besar", "betul", "betulkah", "biasa", "biasanya", "bila", "bilakah", "bisa",
    "bisakah", "boleh", "bolehkah", "bolehlah", "buat", "bukan", "bukankah", "bukanlah",
    "bukannya", "bulan", "bung", "cara", "caranya", "cukup", "cukupkah", "cukuplah", "cuma",
    "dahulu", "dalam", "dan", "dapat", "dari", "daripada", "datang", "dekat", "demi", "demikian",
    "demikianlah", "dengan", "depan", "di", "dia", "diakhiri", "diakhirinya", "dialah", "diantara",
    "diantaranya", "diberi", "diberikan", "diberikannya", "dibuat", "dibuatnya", "didapat",
    "didatangkan", "digunakan", "diibaratkan", "diibaratkannya", "diingat", "diingatkan",
    "diinginkan", "dijawab", "dijelaskan", "dijelaskannya", "dikarenakan", "dikatakan",
    "dikatakannya", "dikerjakan", "diketahui", "diketahuinya", "dikira", "dilakukan", "dilalui",
    "dilihat", "dimaksud", "dimaksudkan", "dimaksudkannya", "dimaksudnya", "diminta", "dimintai",
    "dimisalkan", "dimulai", "dimulailah", "dimulainya", "dimungkinkan", "dini", "dipastikan",
    "diperbuat", "diperbuatnya", "dipergunakan", "diperkirakan", "diperlihatkan", "diperlukan",
    "diperlukannya", "dipersoalkan", "dipertanyakan", "dipunyai", "diri", "dirinya", "disampaikan",
    "disebut", "disebutkan", "disebutkannya", "disini", "disinilah", "ditambahkan", "ditandaskan",
    "ditanya", "ditanyai", "ditanyakan", "ditegaskan", "ditujukan", "ditunjuk", "ditunjuki",
    "ditunjukkan", "ditunjukkannya", "ditunjuknya", "dituturkan", "dituturkannya", "diucapkan",
    "diucapkannya", "diungkapkan", "dong", "dua", "dulu", "empat", "enggak", "enggaknya", "entah",
    "entahlah", "guna", "gunakan", "hal", "hampir", "hanya", "hanyalah", "hari", "harus",
    "haruslah", "harusnya", "hendak", "hendaklah", "hendaknya", "hingga", "ia", "ialah", "ibarat",
    "ibaratkan", "ibaratnya", "ibu", "ikut", "ingat", "ingin", "inginkah", "inginkan", "ini",
    "inikah", "inilah", "itu", "itukah", "itulah", "jadi", "jadilah", "jadinya", "jangan",
    "jangankan", "janganlah", "jauh", "jawab", "jawaban", "jawabnya", "jelas", "jelaskan",
    "jelaslah", "jelasnya", "jika", "jikalau", "juga", "jumlah", "jumlahnya", "justru", "kala",
    "kalau", "kalaulah", "kalaupun", "kalian", "kami", "kamilah", "kamu", "kamulah", "kan",
    "kapan", "kapankah", "kapanpun", "karena", "karenanya", "kasus", "kata", "katakan",
    "katakanlah", "katanya", "ke", "keadaan", "kebetulan", "kecil", "kedua", "keduanya",
    "keinginan", "kelamaan", "kelihatan", "kelihatannya", "kelima", "keluar", "kembali",
    "kemudian", "kemungkinan", "kemungkinannya", "kenapa", "kepada", "kepadanya", "kesampaian",
    "keseluruhan", "keseluruhannya", "keterlaluan", "ketika", "khususnya", "kini", "kinilah",
    "kira", "kiranya", "kita", "kitalah", "kok", "kurang", "lagi", "lagian", "lah", "lain",
    "lainnya", "lalu", "lama", "lamanya", "lanjut", "lanjutnya", "lebih", "lewat", "lima",
    "luar", "macam", "maka", "makanya", "makin", "malah", "malahan", "mampu", "mampukah",
    "mana", "manakala", "manalagi", "masa", "masalah", "masalahnya", "masih", "masihkah",
    "masing", "mau", "maupun", "melainkan", "melakukan", "melalui", "melihat", "melihatnya",
    "memang", "memastikan", "memberi", "memberikan", "membuat", "memerlukan", "memihak",
    "meminta", "memintakan", "memisalkan", "memperbuat", "mempergunakan", "memperkirakan",
    "memperlihatkan", "mempersiapkan", "mempersoalkan", "mempertanyakan", "mempunyai", "memulai",
    "memungkinkan", "menaiki", "menambahkan", "menandaskan", "menanti", "menanyakan", "menegaskan",
    "mengakhiri", "mengapa", "mengatakan", "mengatakannya", "mengenai", "mengerjakan", "mengetahui",
    "menggunakan", "menghendaki", "mengibaratkan", "mengibaratkannya", "mengingat", "mengingatkan",
    "menginginkan", "mengira", "mengucapkan", "mengucapkannya", "mengungkapkan", "menjadi",
    "menjawab", "menjelaskan", "menuju", "menunjuk", "menunjuki", "menunjukkan", "menunjuknya",
    "menurut", "menuturkan", "menyampaikan", "menyangkut", "menyatakan", "menyebutkan",
    "menyeluruh", "menyiapkan", "merasa", "mereka", "merekalah", "merupakan", "meski", "meskipun",
    "meyakini", "meyakinkan", "minta", "mirip", "misal", "misalkan", "misalnya", "mula", "mulai",
    "mulailah", "mulanya", "mungkin", "mungkinkah", "nah", "naik", "namun", "nanti", "nantinya",
    "nyaris", "nyatanya", "oleh", "olehnya", "orang", "pada", "padahal", "padanya", "pak",
    "paling", "panjang", "pantas", "para", "pasti", "pastilah", "penting", "pentingnya", "per",
    "percuma", "perlu", "perlukah", "perlunya", "pernah", "persoalan", "pertama", "pertanyaan",
    "pertanyakan", "pihak", "pihaknya", "pukul", "pula", "pun", "punya", "rasa", "rasanya",
    "rata", "rupanya", "saat", "saatnya", "saja", "sajalah", "saling", "sama", "sambil", "sampai",
    "sampaikan", "sana", "sangat", "sangatlah", "satu", "saya", "sayalah", "se", "sebab",
    "sebabnya", "sebagai", "sebagaimana", "sebagainya", "sebagian", "sebaik", "sebaiknya",
    "sebaliknya", "sebanyak", "sebegini", "sebegitu", "sebelum", "sebelumnya", "sebenarnya",
    "seberapa", "sebesar", "sebetulnya", "sebisanya", "sebuah", "sebut", "sebutlah", "sebutnya",
    "secara", "secukupnya", "sedang", "sedangkan", "sedemikian", "sedikit", "sedikitnya", "seenaknya",
    "segala", "segalanya", "segera", "seharusnya", "sehingga", "seingat", "sejak", "sejauh",
    "sejenak", "sejumlah", "sekadar", "sekadarnya", "sekali", "sekalian", "sekaligus", "sekalipun",
    "sekarang", "sekecil", "seketika", "sekiranya", "sekitar", "sekitarnya", "sekurang", "sekurangnya",
    "sela", "selain", "selaku", "selalu", "selama", "selamanya", "selanjutnya", "seluruh",
    "seluruhnya", "semacam", "semakin", "semampu", "semampunya", "semasa", "semasih", "semata",
    "semaunya", "sementara", "semisal", "semisalnya", "sempat", "semua", "semuanya", "semula",
    "sendiri", "sendirian", "sendirinya", "seolah", "seorang", "sepanjang", "sepantasnya",
    "sepantasnya", "seperlunya", "seperti", "sepertinya", "sepihak", "sering", "seringnya", "serta",
    "serupa", "sesaat", "sesama", "sesampai", "sesegera", "sesekali", "seseorang", "sesuatu",
    "sesuatunya", "sesudah", "sesudahnya", "setelah", "setempat", "setengah", "seterusnya",
    "setiap", "setiba", "setibanya", "setidaknya", "setinggi", "seusai", "sewaktu", "siap",
    "siapa", "siapakah", "siapapun", "sini", "sinilah", "soal", "soalnya", "suatu", "sudah",
    "sudahkah", "sudahlah", "supaya", "tadi", "tadinya", "tahu", "tahun", "tak", "tambah",
    "tambahnya", "tampak", "tampaknya", "tandas", "tandasnya", "tanpa", "tanya", "tanyakan",
    "tanyanya", "tapi", "tegas", "tegasnya", "telah", "tempat", "tengah", "tentang", "tentu",
    "tentulah", "tentunya", "tepat", "terakhir", "terasa", "terbanyak", "terdahulu", "terdapat",
    "terdiri", "terhadap", "terhadapnya", "teringat", "terjadi", "terjadilah", "terjadinya",
    "terkira", "terlalu", "terlebih", "terlihat", "termasuk", "ternyata", "tersampaikan", "tersebut",
    "tersebutlah", "tertentu", "tertuju", "terus", "terutama", "tetap", "tetapi", "tiap", "tiba",
    "tidak", "tidakkah", "tidaklah", "tiga", "tinggi", "toh", "tunjuk", "turut", "tutur", "tuturnya",
    "ucap", "ucapnya", "ujar", "ujarnya", "umum", "umumnya", "ungkap", "ungkapnya", "untuk", "usah",
    "usai", "waduh", "wah", "wahai", "waktu", "waktunya", "walau", "walaupun", "wong", "yaitu",
    "yakin", "yakni", "yang",
}

ENGLISH_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "cannot", "could", "dear", "did", "didn", "do", "does", "doesn", "doing",
    "don", "down", "during", "each", "ensure", "even", "few", "for", "from", "further", "get",
    "got", "had", "has", "have", "having", "he", "hello", "her", "here", "hers", "herself", "hi",
    "him", "himself", "his", "hope", "how", "i", "if", "in", "into", "is", "isn", "it", "its",
    "itself", "just", "keep", "kindly", "know", "let", "like", "make", "may", "me", "might",
    "more", "most", "much", "must", "my", "myself", "need", "no", "nor", "not", "now", "of",
    "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own",
    "please", "really", "regards", "same", "see", "she", "should", "since", "so", "some", "still",
    "such", "team", "than", "thank", "thanks", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "us", "use", "very", "want", "was", "we", "well", "were", "what",
    "when", "where", "which", "while", "who", "whom", "why", "will", "with", "without", "would",
    "you", "your", "yours", "yourself", "yourselves",
}

INDONESIAN_SLANG_STOPWORDS: Set[str] = {
    "min", "admin", "mimin", "bgt", "banget", "bngt", "tlg", "tulung", "yg", "yng", "gak", "ga",
    "gk", "nggak", "ngak", "nga", "blm", "belom", "dgn", "dg", "utk", "tuk", "sdh", "udh", "udah",
    "dah", "sy", "sya", "aku", "gue", "gw", "lu", "lo", "mhs", "mw", "mau", "sblm", "tgl",
    "makasi", "makasih", "makasii", "trimakasih", "trims", "thx", "tq", "nya", "aja", "aj",
    "dong", "dongs", "nih", "ni", "sih", "deh", "lho", "loh", "tenan", "ora", "wis", "piye",
    "neng", "iki", "meneh", "wingi", "ono", "jan", "dina", "pirang", "kangelan", "bkn", "krn",
    "karna", "jd", "jdi", "tp", "tpi", "klo", "kl", "kalo", "kalok", "dlm", "trs", "trus", "sm",
    "sma", "bs", "bsa", "lg", "lgi", "skrg", "skrang", "jg", "jga", "org", "orng", "smg", "semoga",
    "kak", "kk", "gan", "bro", "sis", "plis", "pls", "please", "btw", "fyi", "asap", "wkwk",
    "wkwkwk", "hehe", "haha", "huhu", "yaa", "yaaa", "yah", "iya", "iyaa", "gitu", "gini", "gtu",
    "kayak", "kyk", "kek", "biar", "buat", "pake", "pakai", "sampe", "ampe", "bener", "beneran",
    "emang", "emg", "cuman", "cuma", "doang", "bangett", "parah", "bikin", "kasih", "dikasih",
    "ngasih", "tau", "tahu", "liat", "lihat", "bilang", "malah", "padahal", "soalnya", "soal",
    "terus", "jadi", "kalo", "gimana", "gmn", "knp", "kenapa", "apaan", "sbg", "dll", "dsb",
    "dst", "dkk", "tsb", "yg", "dr", "ke", "di", "pd", "tdk", "tidak", "bisa", "ada",
}

COMPLAINT_BOILERPLATE_STOPWORDS: Set[str] = {
    "mohon", "tolong", "yth", "selamat", "pagi", "siang", "sore", "malam", "terima", "kasih",
    "permisi", "perhatian", "perhatiannya", "lapor", "laporan", "melaporkan", "info", "informasi",
    "warga", "segera", "ditindaklanjuti", "tindaklanjuti", "tindak", "lanjut", "ditangani",
    "tangani", "nama", "nim", "pak", "bu", "bapak", "ibu", "mas", "mbak", "halo", "hai",
    "waduh", "aduh", "mantap", "hebat", "keren", "salut", "apresiasi", "saran", "usul",
    "masukan", "keluhan", "aduan", "pengaduan", "masalah", "kondisi", "wilayah", "daerah",
    "area", "lokasi", "jalan", "kecamatan", "kelurahan", "desa", "kabupaten", "kota", "provinsi",
    "kantor", "dinas", "petugas", "tim", "kinerja", "kerja", "minggu", "jam", "kemarin",
    "sekarang", "semalam", "terkait", "bantuan", "bantuannya", "solusi", "solusinya",
    "terimakasih", "assalamualaikum", "wassalamualaikum", "salam", "hormat", "kepada",
    "berkali", "resmi", "terusin", "perbaiki", "diperbaiki", "perbaikan", "tolonglah",
    "mohonlah", "sebaiknya", "bagusnya", "harap", "harapan", "berharap", "semoga",
    "banget", "sekali", "sangat", "kurang", "lebih", "sudah", "belum", "masih", "lagi",
    "saja", "hanya", "terus", "juga", "atau", "dan", "yang", "untuk", "dengan", "dari",
    "pada", "dalam", "bisa", "dapat", "akan", "harus", "perlu", "ingin", "mau", "buat",
    "baik", "bagus", "buruk", "jelek", "lama", "cepat", "lambat", "ramah", "mudah", "sulit",
    "susah", "jelas", "biasa", "luar", "juara", "alhamdulillah", "diabaikan", "menindaklanjuti",
    "terbantu", "membantu", "beres", "muluk", "nyaman", "sabar", "ekstra", "awet", "prestasi",
    "prestasinya", "diarsipkan", "tindakan", "cek", "dicek", "gercep", "kirim", "genting",
    "nyawa", "membahayakan", "pelayanan", "kampus", "mahasiswa", "gara", "butuh", "memohon",
    "kendala", "pas", "giliran", "jadwal", "masyarakat", "mengambil", "turun", "bergerak",
    "menuntaskan", "teratasi", "direspons", "direspon", "tanggapan", "tanggapannya",
    "evaluasi", "berkala", "terulang", "keulang", "rutin", "dicegah", "pengecekan",
    "meninjau", "mengganggu", "aktivitas", "perkembangan", "sepele", "kejadian", "berulang",
    "celaka", "mampir", "menuntut", "diselesaikan", "menunggu", "korban", "barusan",
    "evakuasi", "menindak", "pertimbangan", "markotop", "wisata", "pokoknya", "depan",
    "ditengok", "boro", "muluk", "balasan", "aduannya", "statusnya", "ditutup", "responsnya",
    "berlarut", "kejelasan", "menyalahkan", "sigap",
}


def build_4layer_stopwords() -> Set[str]:
    """Assemble the complete 4-layer LDA stopword set (including synthetic slot proper nouns)."""
    sw: Set[str] = set()
    sw.update(INDONESIAN_STOPWORDS)
    sw.update(ENGLISH_STOPWORDS)
    sw.update(INDONESIAN_SLANG_STOPWORDS)
    sw.update(COMPLAINT_BOILERPLATE_STOPWORDS)
    sw.update(SYNTHETIC_ENTITY_SLOT_STOPWORDS)

    # Optionally enrich with NLTK stopwords if already downloaded locally (never trigger network)
    try:
        from nltk.corpus import stopwords as nltk_sw

        for lang in ("indonesian", "english"):
            try:
                sw.update(w.lower() for w in nltk_sw.words(lang))
            except LookupError:
                pass
    except ImportError:
        pass

    return sw


URL_RE = re.compile(r"(?:https?://\S+|www\.\S+)", re.IGNORECASE)
BRACKET_MASK_RE = re.compile(r"\[[^\]]{1,30}\]")
NON_ALPHA_RE = re.compile(r"[^a-z\s]+")


def preprocess_lda_text(
    text: str,
    stopwords: Optional[Set[str]] = None,
    min_token_len: int = 3,
) -> List[str]:
    """
    Clean and tokenize one document for LDA (completely separate from NER text):
    - Lowercase
    - Strip URLs, anonymization brackets (`[NAMA]`, `[NIM]`), digits, and punctuation
    - Filter 4-layer stopwords and short tokens (`len < min_token_len`)
    """
    sw = stopwords if stopwords is not None else build_4layer_stopwords()
    s = text.lower()
    s = URL_RE.sub(" ", s)
    s = BRACKET_MASK_RE.sub(" ", s)
    s = NON_ALPHA_RE.sub(" ", s)
    tokens: List[str] = []
    for tok in s.split():
        if len(tok) < min_token_len:
            continue
        if tok in sw:
            continue
        tokens.append(tok)
    return tokens


# ---------------------------------------------------------------------------
# 2. Pure-Python PMI Bigram Builder & Coherence Metrics (C_v, U_mass, Diversity)
# ---------------------------------------------------------------------------

class PMIBigramPhraser:
    """
    Lightweight Gensim-compatible bigram phraser fitted strictly on `train` tokens.
    Joins consecutive tokens `(w1, w2)` into `w1_w2` when:
      count(w1, w2) >= min_count and (count(w1, w2) - min_count) * N / (count(w1) * count(w2)) >= threshold.
    """

    def __init__(self, min_count: int = 10, threshold: float = 10.0) -> None:
        self.min_count = min_count
        self.threshold = threshold
        self.bigrams: Set[Tuple[str, str]] = set()

    def fit(self, tokenized_docs: Sequence[Sequence[str]]) -> "PMIBigramPhraser":
        uni: Counter[str] = Counter()
        bi: Counter[Tuple[str, str]] = Counter()
        total_tokens = 0

        for doc in tokenized_docs:
            total_tokens += len(doc)
            uni.update(doc)
            for a, b in zip(doc, doc[1:]):
                bi[(a, b)] += 1

        vocab_size = max(1, len(uni))
        accepted: Set[Tuple[str, str]] = set()
        for (a, b), cnt in bi.items():
            if cnt < self.min_count:
                continue
            denom = uni[a] * uni[b]
            if denom <= 0:
                continue
            score = ((cnt - self.min_count) * vocab_size) / denom
            if score >= self.threshold:
                accepted.add((a, b))

        self.bigrams = accepted
        return self

    def transform_doc(self, tokens: Sequence[str]) -> List[str]:
        out: List[str] = []
        i = 0
        n = len(tokens)
        while i < n:
            if i + 1 < n and (tokens[i], tokens[i + 1]) in self.bigrams:
                out.append(f"{tokens[i]}_{tokens[i + 1]}")
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        return out

    def transform(self, tokenized_docs: Sequence[Sequence[str]]) -> List[List[str]]:
        return [self.transform_doc(doc) for doc in tokenized_docs]


def compute_topic_diversity(topics_top_words: Sequence[Sequence[str]], top_n: int = 10) -> float:
    """Fraction of unique words among the top-`top_n` words across all K topics."""
    if not topics_top_words:
        return 0.0
    all_words: List[str] = []
    for words in topics_top_words:
        all_words.extend(words[:top_n])
    if not all_words:
        return 0.0
    return round(len(set(all_words)) / len(all_words), 4)


def compute_umass_coherence(
    topics_top_words: Sequence[Sequence[str]],
    tokenized_docs: Sequence[Sequence[str]],
    top_n: int = 10,
    eps: float = 1e-12,
) -> float:
    """Compute U_mass topic coherence over document co-occurrence counts."""
    word_doc_sets: Dict[str, Set[int]] = defaultdict(set)
    target_words = {w for topic in topics_top_words for w in topic[:top_n]}
    for doc_idx, doc in enumerate(tokenized_docs):
        for w in set(doc):
            if w in target_words:
                word_doc_sets[w].add(doc_idx)

    topic_scores: List[float] = []
    for topic in topics_top_words:
        words = list(topic[:top_n])
        pair_scores: List[float] = []
        for i in range(1, len(words)):
            wi = words[i]
            docs_i = word_doc_sets.get(wi, set())
            for j in range(0, i):
                wj = words[j]
                docs_j = word_doc_sets.get(wj, set())
                co_occ = len(docs_i & docs_j)
                denom = len(docs_j)
                if denom > 0:
                    pair_scores.append(math.log((co_occ + eps) / denom))
        if pair_scores:
            topic_scores.append(sum(pair_scores) / len(pair_scores))

    return round(sum(topic_scores) / len(topic_scores), 4) if topic_scores else 0.0


def compute_cv_coherence_pure(
    topics_top_words: Sequence[Sequence[str]],
    tokenized_docs: Sequence[Sequence[str]],
    window_size: int = 110,
    top_n: int = 10,
    eps: float = 1e-12,
) -> float:
    """
    Pure-Python implementation of Röder et al. (2015) C_v topic coherence
    (sliding window size 110, NPMI confirmation measure, cosine similarity to topic context vector).
    Used when `gensim` is not installed locally.
    """
    target_words = {w for topic in topics_top_words for w in topic[:top_n]}
    if not target_words:
        return 0.0

    word_win_count: Counter[str] = Counter()
    pair_win_count: Counter[Tuple[str, str]] = Counter()
    total_windows = 0

    for doc in tokenized_docs:
        n = len(doc)
        if n == 0:
            continue
        if n <= window_size:
            windows = [doc]
        else:
            windows = [doc[i : i + window_size] for i in range(0, n - window_size + 1)]
        for win in windows:
            total_windows += 1
            present = sorted(set(win) & target_words)
            for idx_a, wa in enumerate(present):
                word_win_count[wa] += 1
                for wb in present[idx_a + 1 :]:
                    pair_win_count[(wa, wb)] += 1

    if total_windows == 0:
        return 0.0

    def _npmi(wa: str, wb: str) -> float:
        if wa == wb:
            return 1.0
        p_a = word_win_count[wa] / total_windows
        p_b = word_win_count[wb] / total_windows
        key = (wa, wb) if wa < wb else (wb, wa)
        p_ab = pair_win_count[key] / total_windows
        if p_a <= 0 or p_b <= 0:
            return 0.0
        num = math.log((p_ab + eps) / (p_a * p_b))
        den = -math.log(p_ab + eps)
        return num / den if den > 0 else 0.0

    topic_scores: List[float] = []
    for topic in topics_top_words:
        words = list(topic[:top_n])
        m = len(words)
        if m < 2:
            continue
        # Context vector v(w_i) = [NPMI(w_i, w_j) for j in 0..m-1]
        mat = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            for j in range(m):
                mat[i, j] = _npmi(words[i], words[j])
        # Topic sum vector v_all = sum_i v(w_i)
        v_all = mat.sum(axis=0)
        norm_all = float(np.linalg.norm(v_all))
        if norm_all <= 0:
            topic_scores.append(0.0)
            continue
        cos_vals: List[float] = []
        for i in range(m):
            vi = mat[i]
            ni = float(np.linalg.norm(vi))
            if ni > 0:
                cos_vals.append(float(np.dot(vi, v_all) / (ni * norm_all)))
            else:
                cos_vals.append(0.0)
        topic_scores.append(sum(cos_vals) / len(cos_vals))

    return round(sum(topic_scores) / len(topic_scores), 4) if topic_scores else 0.0


# ---------------------------------------------------------------------------
# 3. Interpretability Cross-Tabulations (Topic x Domain & Topic x Source)
# ---------------------------------------------------------------------------

def compute_topic_domain_alignment(
    records: Sequence[Dict[str, Any]],
    doc_topics: Sequence[int],
    k_topics: int,
    source_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute Topic x 5 Domains cross-tabulation, Topic Purity, and NMI on
    domain-labeled records (optionally filtered to `synthetic` or `campus_complaints`).
    """
    y_domain: List[str] = []
    y_topic: List[int] = []
    crosstab: Dict[str, Dict[str, int]] = {
        f"topic_{t}": {d: 0 for d in DOMAINS} for t in range(k_topics)
    }

    for rec, tid in zip(records, doc_topics):
        if source_filter is not None and rec.get("source") != source_filter:
            continue
        dom = rec.get("domain")
        if dom not in DOMAINS:
            continue
        y_domain.append(dom)
        y_topic.append(int(tid))
        crosstab[f"topic_{int(tid)}"][dom] += 1

    n_labeled = len(y_domain)
    if n_labeled == 0:
        return {
            "n_labeled_docs": 0,
            "topic_purity": 0.0,
            "nmi": 0.0,
            "crosstab": crosstab,
        }

    majority_sum = sum(max(dom_counts.values()) for dom_counts in crosstab.values())
    purity = round(majority_sum / n_labeled, 4)
    nmi = round(float(normalized_mutual_info_score(y_domain, y_topic)), 4)

    dominant_domain_per_topic: Dict[str, Dict[str, Any]] = {}
    for t_key, dom_counts in crosstab.items():
        tot = sum(dom_counts.values())
        if tot == 0:
            dominant_domain_per_topic[t_key] = {"dominant_domain": None, "share": 0.0, "docs": 0}
        else:
            best_dom, best_cnt = max(dom_counts.items(), key=lambda x: x[1])
            dominant_domain_per_topic[t_key] = {
                "dominant_domain": best_dom,
                "share": round(best_cnt / tot, 4),
                "docs": tot,
            }

    return {
        "n_labeled_docs": n_labeled,
        "topic_purity": purity,
        "nmi": nmi,
        "dominant_domain_per_topic": dominant_domain_per_topic,
        "crosstab": crosstab,
    }


def compute_topic_source_crosstab(
    records: Sequence[Dict[str, Any]],
    doc_topics: Sequence[int],
    k_topics: int,
) -> Dict[str, Dict[str, int]]:
    """Compute Topic x Source (`sp4n_lapor`, `campus_complaints`, `synthetic`) counts."""
    crosstab: Dict[str, Dict[str, int]] = {
        f"topic_{t}": {src: 0 for src in LDA_SOURCES} for t in range(k_topics)
    }
    for rec, tid in zip(records, doc_topics):
        src = rec.get("source")
        if src in LDA_SOURCES:
            crosstab[f"topic_{int(tid)}"][src] += 1
    return crosstab


def collect_representative_docs(
    records: Sequence[Dict[str, Any]],
    topic_probs: Sequence[Sequence[float]],
    k_topics: int,
    top_docs: int = 3,
) -> Dict[str, List[Dict[str, Any]]]:
    """Collect the top-`top_docs` highest-probability complaint snippets per topic."""
    per_topic: Dict[int, List[Tuple[float, Dict[str, Any]]]] = defaultdict(list)
    for rec, probs in zip(records, topic_probs):
        for t in range(k_topics):
            per_topic[t].append((float(probs[t]), rec))

    result: Dict[str, List[Dict[str, Any]]] = {}
    for t in range(k_topics):
        ranked = sorted(per_topic[t], key=lambda x: -x[0])[:top_docs]
        result[f"topic_{t}"] = [
            {
                "id": r["id"],
                "source": r["source"],
                "domain": r.get("domain"),
                "prob": round(p, 4),
                "text_snippet": r["text"][:180],
            }
            for p, r in ranked
        ]
    return result


# ---------------------------------------------------------------------------
# 4. Gensim & Sklearn Backends
# ---------------------------------------------------------------------------

def _gensim_available() -> bool:
    try:
        import gensim  # noqa: F401
        return True
    except ImportError:
        return False


def _run_gensim_sweep(
    train_records: Sequence[Dict[str, Any]],
    val_records: Sequence[Dict[str, Any]],
    test_records: Sequence[Dict[str, Any]],
    train_tokens_raw: List[List[str]],
    val_tokens_raw: List[List[str]],
    test_tokens_raw: List[List[str]],
    k_values: Sequence[int],
    no_below: int,
    no_above: float,
    keep_n: int,
    seed: int,
    model_dir: Path,
) -> Dict[str, Any]:
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel, LdaModel, Phrases
    from gensim.models.phrases import Phraser

    phrases = Phrases(train_tokens_raw, min_count=10, threshold=10.0)
    bigram_phraser = Phraser(phrases)

    train_docs = [bigram_phraser[doc] for doc in train_tokens_raw]
    val_docs = [bigram_phraser[doc] for doc in val_tokens_raw]
    test_docs = [bigram_phraser[doc] for doc in test_tokens_raw]

    dictionary = Dictionary(train_docs)
    raw_vocab_size = len(dictionary)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    dictionary.compactify()

    train_corpus = [dictionary.doc2bow(doc) for doc in train_docs]
    val_corpus = [dictionary.doc2bow(doc) for doc in val_docs]
    test_corpus = [dictionary.doc2bow(doc) for doc in test_docs]

    sweep_results: List[Dict[str, Any]] = []
    best_model = None
    best_entry: Optional[Dict[str, Any]] = None
    best_key: Tuple[float, float] = (-1.0, -1.0)

    for k in k_values:
        lda = LdaModel(
            corpus=train_corpus,
            id2word=dictionary,
            num_topics=k,
            random_state=seed,
            passes=15,
            iterations=200,
            alpha="auto",
            eta="auto",
            per_word_topics=False,
        )
        topics_top_words = [[w for w, _ in lda.show_topic(t, topn=10)] for t in range(k)]

        cm_cv = CoherenceModel(
            model=lda, texts=train_docs, dictionary=dictionary, coherence="c_v", processes=1
        )
        cv = round(float(cm_cv.get_coherence()), 4)

        cm_umass = CoherenceModel(
            model=lda, corpus=train_corpus, dictionary=dictionary, coherence="u_mass", processes=1
        )
        umass = round(float(cm_umass.get_coherence()), 4)

        diversity = compute_topic_diversity(topics_top_words, top_n=10)
        val_perp = round(float(math.exp(-lda.log_perplexity(val_corpus))), 4) if val_corpus else 0.0
        test_perp = round(float(math.exp(-lda.log_perplexity(test_corpus))), 4) if test_corpus else 0.0

        # Train document topic assignments for interpretability cross-tabulations
        train_probs: List[List[float]] = []
        train_top_topics: List[int] = []
        for bow in train_corpus:
            dist = lda.get_document_topics(bow, minimum_probability=0.0)
            row_p = [0.0] * k
            for tid, prob in dist:
                row_p[tid] = float(prob)
            train_probs.append(row_p)
            train_top_topics.append(int(np.argmax(row_p)))

        dom_all = compute_topic_domain_alignment(train_records, train_top_topics, k)
        entry = {
            "k": k,
            "coherence_c_v": cv,
            "coherence_u_mass": umass,
            "topic_diversity": diversity,
            "val_perplexity": val_perp,
            "test_perplexity": test_perp,
            "domain_purity_all": dom_all["topic_purity"],
            "domain_nmi_all": dom_all["nmi"],
            "topics": {
                f"topic_{t}": [
                    {"word": w, "weight": round(float(wt), 5)}
                    for w, wt in lda.show_topic(t, topn=10)
                ]
                for t in range(k)
            },
        }
        sweep_results.append(entry)

        # Select best K by C_v coherence, with topic diversity tie-breaker
        cand_key = (cv, diversity)
        if cand_key > best_key:
            best_key = cand_key
            best_model = lda
            best_entry = {
                "entry": entry,
                "train_probs": train_probs,
                "train_top_topics": train_top_topics,
                "topics_top_words": topics_top_words,
            }

    assert best_model is not None and best_entry is not None
    best_k = int(best_entry["entry"]["k"])

    model_dir.mkdir(parents=True, exist_ok=True)
    best_model.save(str(model_dir / "lda_gensim.model"))
    dictionary.save(str(model_dir / "dictionary.gensim"))
    bigram_phraser.save(str(model_dir / "bigram_phraser.gensim"))

    topics_keywords = {
        str(t): best_entry["topics_top_words"][t] for t in range(best_k)
    }
    meta = {
        "backend": "gensim",
        "selected_k": best_k,
        "vocab_size_raw": raw_vocab_size,
        "vocab_size_filtered": len(dictionary),
        "no_below": no_below,
        "no_above": no_above,
        "keep_n": keep_n,
        "seed": seed,
        "topics_keywords": topics_keywords,
    }
    (model_dir / "lda_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {
        "backend": "gensim",
        "vocab_stats": {
            "raw_vocab_size": raw_vocab_size,
            "filtered_vocab_size": len(dictionary),
            "no_below": no_below,
            "no_above": no_above,
            "keep_n": keep_n,
        },
        "sweep_results": sweep_results,
        "selected_k": best_k,
        "best_metrics": best_entry["entry"],
        "topic_x_domain_all": compute_topic_domain_alignment(
            train_records, best_entry["train_top_topics"], best_k
        ),
        "topic_x_domain_synthetic": compute_topic_domain_alignment(
            train_records, best_entry["train_top_topics"], best_k, source_filter="synthetic"
        ),
        "topic_x_domain_campus": compute_topic_domain_alignment(
            train_records, best_entry["train_top_topics"], best_k, source_filter="campus_complaints"
        ),
        "topic_x_source": compute_topic_source_crosstab(
            train_records, best_entry["train_top_topics"], best_k
        ),
        "representative_docs": collect_representative_docs(
            train_records, best_entry["train_probs"], best_k, top_docs=3
        ),
    }


def _run_sklearn_sweep(
    train_records: Sequence[Dict[str, Any]],
    val_records: Sequence[Dict[str, Any]],
    test_records: Sequence[Dict[str, Any]],
    train_tokens_raw: List[List[str]],
    val_tokens_raw: List[List[str]],
    test_tokens_raw: List[List[str]],
    k_values: Sequence[int],
    no_below: int,
    no_above: float,
    keep_n: int,
    seed: int,
    model_dir: Path,
) -> Dict[str, Any]:
    from sklearn.feature_extraction.text import CountVectorizer

    phraser = PMIBigramPhraser(min_count=10, threshold=10.0).fit(train_tokens_raw)
    train_docs = phraser.transform(train_tokens_raw)
    val_docs = phraser.transform(val_tokens_raw)
    test_docs = phraser.transform(test_tokens_raw)

    raw_vocab = {w for doc in train_docs for w in doc}

    train_joined = [" ".join(doc) for doc in train_docs]
    val_joined = [" ".join(doc) for doc in val_docs]
    test_joined = [" ".join(doc) for doc in test_docs]

    vectorizer = CountVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        lowercase=False,
        token_pattern=None,
        min_df=no_below,
        max_df=no_above,
        max_features=keep_n,
    )
    X_train = vectorizer.fit_transform(train_joined)
    X_val = vectorizer.transform(val_joined)
    X_test = vectorizer.transform(test_joined)
    vocab_names = vectorizer.get_feature_names_out().tolist()

    sweep_results: List[Dict[str, Any]] = []
    best_model = None
    best_entry: Optional[Dict[str, Any]] = None
    best_key: Tuple[float, float] = (-1.0, -1.0)

    for k in k_values:
        lda = LatentDirichletAllocation(
            n_components=k,
            learning_method="batch",
            max_iter=25,
            random_state=seed,
            n_jobs=1,
        )
        train_probs_arr = lda.fit_transform(X_train)
        train_probs = train_probs_arr.tolist()
        train_top_topics = np.argmax(train_probs_arr, axis=1).tolist()

        # Extract top-10 normalized topic words
        topics_top_words: List[List[str]] = []
        topics_dict: Dict[str, List[Dict[str, Any]]] = {}
        for t in range(k):
            comp = lda.components_[t]
            comp_norm = comp / max(1e-12, float(comp.sum()))
            top_idx = np.argsort(comp_norm)[::-1][:10]
            words_t = [vocab_names[idx] for idx in top_idx]
            topics_top_words.append(words_t)
            topics_dict[f"topic_{t}"] = [
                {"word": vocab_names[idx], "weight": round(float(comp_norm[idx]), 5)}
                for idx in top_idx
            ]

        cv = compute_cv_coherence_pure(topics_top_words, train_docs, window_size=110, top_n=10)
        umass = compute_umass_coherence(topics_top_words, train_docs, top_n=10)
        diversity = compute_topic_diversity(topics_top_words, top_n=10)
        val_perp = round(float(lda.perplexity(X_val)), 4) if X_val.shape[0] > 0 else 0.0
        test_perp = round(float(lda.perplexity(X_test)), 4) if X_test.shape[0] > 0 else 0.0

        dom_all = compute_topic_domain_alignment(train_records, train_top_topics, k)
        entry = {
            "k": k,
            "coherence_c_v": cv,
            "coherence_u_mass": umass,
            "topic_diversity": diversity,
            "val_perplexity": val_perp,
            "test_perplexity": test_perp,
            "domain_purity_all": dom_all["topic_purity"],
            "domain_nmi_all": dom_all["nmi"],
            "topics": topics_dict,
        }
        sweep_results.append(entry)

        cand_key = (cv, diversity)
        if cand_key > best_key:
            best_key = cand_key
            best_model = lda
            best_entry = {
                "entry": entry,
                "train_probs": train_probs,
                "train_top_topics": train_top_topics,
                "topics_top_words": topics_top_words,
            }

    assert best_model is not None and best_entry is not None
    best_k = int(best_entry["entry"]["k"])

    model_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "backend": "sklearn",
        "lda_model": best_model,
        "vectorizer": vectorizer,
        "bigrams": sorted(phraser.bigrams),
        "selected_k": best_k,
        "topics_keywords": {str(t): best_entry["topics_top_words"][t] for t in range(best_k)},
    }
    joblib.dump(bundle, model_dir / "lda_sklearn.joblib")

    meta = {
        "backend": "sklearn",
        "selected_k": best_k,
        "vocab_size_raw": len(raw_vocab),
        "vocab_size_filtered": len(vocab_names),
        "bigrams_learned": len(phraser.bigrams),
        "no_below": no_below,
        "no_above": no_above,
        "keep_n": keep_n,
        "seed": seed,
        "topics_keywords": bundle["topics_keywords"],
    }
    (model_dir / "lda_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {
        "backend": "sklearn",
        "vocab_stats": {
            "raw_vocab_size": len(raw_vocab),
            "filtered_vocab_size": len(vocab_names),
            "bigrams_learned": len(phraser.bigrams),
            "no_below": no_below,
            "no_above": no_above,
            "keep_n": keep_n,
        },
        "sweep_results": sweep_results,
        "selected_k": best_k,
        "best_metrics": best_entry["entry"],
        "topic_x_domain_all": compute_topic_domain_alignment(
            train_records, best_entry["train_top_topics"], best_k
        ),
        "topic_x_domain_synthetic": compute_topic_domain_alignment(
            train_records, best_entry["train_top_topics"], best_k, source_filter="synthetic"
        ),
        "topic_x_domain_campus": compute_topic_domain_alignment(
            train_records, best_entry["train_top_topics"], best_k, source_filter="campus_complaints"
        ),
        "topic_x_source": compute_topic_source_crosstab(
            train_records, best_entry["train_top_topics"], best_k
        ),
        "representative_docs": collect_representative_docs(
            train_records, best_entry["train_probs"], best_k, top_docs=3
        ),
    }


def run_lda_experiment(
    data_path: Path = DEFAULT_MASTER_PATH,
    model_dir: Path = DEFAULT_LDA_MODEL_DIR,
    report_path: Path = DEFAULT_LDA_REPORT_PATH,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    backend: str = "auto",
    no_below: int = 10,
    no_above: float = 0.35,
    keep_n: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run the full Phase 3 LDA topic modeling sweep and save model + report."""
    if backend not in {"auto", "gensim", "sklearn"}:
        raise ValueError(f"Unsupported LDA backend: {backend!r}")
    resolved_backend = (
        ("gensim" if _gensim_available() else "sklearn") if backend == "auto" else backend
    )
    if resolved_backend == "gensim" and not _gensim_available():
        raise ImportError("Gensim backend requested, but `gensim` is not installed.")

    # Load splits strictly respecting split boundaries (fit ONLY on train)
    train_records = load_lda_records(data_path=data_path, split="train", sources=LDA_SOURCES)
    val_records = load_lda_records(data_path=data_path, split="val", sources=LDA_SOURCES)
    test_records = load_lda_records(data_path=data_path, split="test", sources=LDA_SOURCES)

    stopwords = build_4layer_stopwords()
    train_tokens = [preprocess_lda_text(r["text"], stopwords=stopwords) for r in train_records]
    val_tokens = [preprocess_lda_text(r["text"], stopwords=stopwords) for r in val_records]
    test_tokens = [preprocess_lda_text(r["text"], stopwords=stopwords) for r in test_records]

    if resolved_backend == "gensim":
        result = _run_gensim_sweep(
            train_records=train_records,
            val_records=val_records,
            test_records=test_records,
            train_tokens_raw=train_tokens,
            val_tokens_raw=val_tokens,
            test_tokens_raw=test_tokens,
            k_values=k_values,
            no_below=no_below,
            no_above=no_above,
            keep_n=keep_n,
            seed=seed,
            model_dir=model_dir,
        )
    else:
        result = _run_sklearn_sweep(
            train_records=train_records,
            val_records=val_records,
            test_records=test_records,
            train_tokens_raw=train_tokens,
            val_tokens_raw=val_tokens,
            test_tokens_raw=test_tokens,
            k_values=k_values,
            no_below=no_below,
            no_above=no_above,
            keep_n=keep_n,
            seed=seed,
            model_dir=model_dir,
        )

    report = {
        "task": "Task2_LDA_Topic_Modeling",
        "sources": list(LDA_SOURCES),
        "split_counts": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
        },
        "train_source_counts": dict(Counter(r["source"] for r in train_records)),
        "stopword_count": len(stopwords),
        **result,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Task 2 LDA Topic Modeling.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_MASTER_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_LDA_MODEL_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_LDA_REPORT_PATH)
    parser.add_argument("--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "gensim", "sklearn"])
    parser.add_argument("--no-below", type=int, default=10)
    parser.add_argument("--no-above", type=float, default=0.35)
    parser.add_argument("--keep-n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = run_lda_experiment(
        data_path=args.data_path,
        model_dir=args.model_dir,
        report_path=args.report_path,
        k_values=args.k_values,
        backend=args.backend,
        no_below=args.no_below,
        no_above=args.no_above,
        keep_n=args.keep_n,
        seed=args.seed,
    )

    print(
        f"=== TASK 2 PHASE 3 · LDA TOPIC MODELING (backend={report['backend']}, "
        f"train={report['split_counts']['train']:,}, vocab={report['vocab_stats']['filtered_vocab_size']:,}) ==="
    )
    print("| K | C_v Coherence | U_mass | Topic Diversity | Val Perplexity | Test Perplexity | Domain Purity | NMI |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for row in report["sweep_results"]:
        mark = " *" if row["k"] == report["selected_k"] else ""
        print(
            f"| {row['k']}{mark} | {row['coherence_c_v']:.4f} | {row['coherence_u_mass']:.4f} | "
            f"{row['topic_diversity']:.4f} | {row['val_perplexity']:.2f} | "
            f"{row['test_perplexity']:.2f} | {row['domain_purity_all']:.4f} | "
            f"{row['domain_nmi_all']:.4f} |"
        )

    best_k = report["selected_k"]
    print(f"\nSelected K={best_k} Top Keywords:")
    for t_key, words in report["best_metrics"]["topics"].items():
        kw = ", ".join(w["word"] for w in words[:8])
        dom_info = report["topic_x_domain_all"]["dominant_domain_per_topic"][t_key]
        src_counts = report["topic_x_source"][t_key]
        print(
            f"  {t_key:<8} | dom={str(dom_info['dominant_domain']):<22} ({dom_info['share']:.2f}) | "
            f"src={src_counts} | {kw}"
        )

    print(
        f"\nDomain Alignment -> All: purity={report['topic_x_domain_all']['topic_purity']:.4f}, "
        f"NMI={report['topic_x_domain_all']['nmi']:.4f} | "
        f"Synthetic: purity={report['topic_x_domain_synthetic']['topic_purity']:.4f}, "
        f"NMI={report['topic_x_domain_synthetic']['nmi']:.4f} | "
        f"Campus: purity={report['topic_x_domain_campus']['topic_purity']:.4f}, "
        f"NMI={report['topic_x_domain_campus']['nmi']:.4f}"
    )
    print(f"\nSaved LDA model  -> {args.model_dir}")
    print(f"Saved LDA report -> {args.report_path}")


if __name__ == "__main__":
    main()

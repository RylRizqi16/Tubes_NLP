"""
Generate 2,760 synthetic Indonesian public complaint edge cases for the Unified Master Table.

Covers all 5 domains, 5 intents, 3 NER entity types (LOCATION, ORGANIZATION, PERSON)
with verified character offsets, 3 sentiments, sarcasm flag, and 4 urgency levels,
including rare classes (Pungli & Tata Kelola, Darurat/Emergency) and linguistic
edge cases (sarcasm/irony, implicit/explicit negation, code-mixing/slang).

Templates are partitioned into disjoint train/val/test template families (70/15/15)
before slot-filling so test/val sentences never leak exact training phrasing.

Output:
    data/raw/synthetic/synthetic_edge_cases.jsonl

Run:
    python scripts/generate_synthetic_edge_cases.py
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SEED = 42
TARGET_TOTAL = 2760
SPLIT_QUOTAS = {"train": 1932, "val": 414, "test": 414}

ROOT_DIR = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT_DIR / "data" / "raw" / "synthetic" / "synthetic_edge_cases.jsonl"

VALID_DOMAINS = {
    "Infrastruktur & PU",
    "Kesehatan & Sanitasi",
    "Layanan & Administrasi",
    "Lingkungan",
    "Pungli & Tata Kelola",
}
VALID_INTENTS = {"Keluhan", "Permintaan", "Darurat", "Saran", "Apresiasi"}
VALID_SENTIMENTS = {"Positif", "Netral", "Negatif"}
VALID_URGENCY_LEVELS = {"Low", "Med", "High", "Emergency"}

ENTITY_POOLS: Dict[str, Dict[str, List[str]]] = {
    "train": {
        "LOCATION": [
            "Jl. Melati, Tembalang",
            "Jl. Pemuda, Semarang Tengah",
            "Kecamatan Banyumanik",
            "Kelurahan Sekaran, Gunungpati",
            "Jl. Slamet Riyadi, Surakarta",
            "Kabupaten Kendal",
            "Kecamatan Ungaran Barat",
            "Jl. Jenderal Sudirman, Purwokerto",
            "Simpang Lima, Semarang",
            "Pasar Johar, Semarang",
            "Desa Karanganyar, Demak",
            "Jl. Diponegoro, Salatiga",
            "Kecamatan Pekalongan Timur",
            "Kabupaten Cilacap",
            "Jl. Ahmad Yani, Magelang",
            "Kelurahan Mangkang, Tugu",
            "Terminal Tirtonadi, Solo",
            "Jl. Kartini, Pati",
            "Kecamatan Bumiayu, Brebes",
            "Pelabuhan Tanjung Emas",
            "Stasiun Tawang, Semarang",
            "Jl. Brigjen Katamso, Kudus",
            "Alun-Alun Klaten",
            "Desa Sukorejo, Boyolali",
        ],
        "ORGANIZATION": [
            "Dishub Kota Semarang",
            "Dinas PUPR Jawa Tengah",
            "Dinas Kesehatan",
            "Puskesmas Tembalang",
            "RSUD Tugurejo",
            "Disdukcapil Kota Semarang",
            "DLH Kota Surakarta",
            "PDAM Tirta Moedal",
            "Satpol PP",
            "BPBD Jawa Tengah",
            "Polsek Banyumanik",
            "Kantor Kecamatan Ungaran",
            "Kelurahan Sekaran",
            "Dinas Sosial",
            "PLN UP3 Semarang",
            "Dinas Pendidikan",
            "Samsat Semarang",
            "Inspektorat Daerah",
        ],
        "PERSON": [
            "Pak Budi",
            "Bu Siti",
            "Pak Hendra",
            "Mas Agung",
            "Pak Joko",
            "Bu Rina",
            "Pak Slamet",
            "Mbak Dewi",
            "Pak Agus",
            "Bu Lestari",
            "Pak Wahyu",
            "Mas Rizki",
        ],
    },
    "val": {
        "LOCATION": [
            "Jl. Pandanaran, Semarang",
            "Kecamatan Pedurungan",
            "Jl. Pahlawan, Kebumen",
            "Kabupaten Jepara",
            "Kelurahan Ngaliyan",
            "Jl. Veteran, Tegal",
            "Pasar Klewer, Solo",
            "Kecamatan Ambarawa",
            "Desa Kaliwungu, Kendal",
            "Jl. Siliwangi, Semarang Barat",
        ],
        "ORGANIZATION": [
            "Dinas PUPR Kendal",
            "Puskesmas Pedurungan",
            "Disdukcapil Solo",
            "DLH Jawa Tengah",
            "RSUD Moewardi",
            "Dishub Surakarta",
            "BPBD Cilacap",
            "Kantor Kelurahan Ngaliyan",
        ],
        "PERSON": [
            "Pak Arif",
            "Bu Endang",
            "Pak Eko",
            "Mas Fajar",
            "Bu Wati",
        ],
    },
    "test": {
        "LOCATION": [
            "Jl. Setiabudi, Srondol",
            "Kecamatan Genuk, Semarang",
            "Jl. Gajahmada, Semarang",
            "Kabupaten Wonogiri",
            "Kelurahan Meteseh",
            "Jl. Urip Sumoharjo, Solo",
            "Kecamatan Kroya, Cilacap",
            "Pasar Wage, Purwokerto",
            "Desa Mranggen, Demak",
            "Jl. Tentara Pelajar, Magelang",
        ],
        "ORGANIZATION": [
            "Dinas PUPR Semarang",
            "Puskesmas Genuk",
            "Disdukcapil Magelang",
            "DLH Banyumas",
            "RSUD Banyumas",
            "Dishub Jawa Tengah",
            "BPBD Demak",
            "Kantor Kecamatan Genuk",
        ],
        "PERSON": [
            "Pak Darmawan",
            "Bu Nurul",
            "Pak Teguh",
            "Mbak Putri",
            "Pak Cahyo",
        ],
    },
}

DOMAIN_TOPICS: Dict[str, Dict[str, List[str]]] = {
    "Infrastruktur & PU": {
        "issue": [
            "lampu jalan depan sekolah mati total",
            "jalan berlubang sedalam 20 cm",
            "aspal jalan utama mengelupas dan licin",
            "trotoar rusak parah sampai membahayakan pejalan kaki",
            "drainase mampet bikin air menggenang tiap hujan",
            "rambu lalu lintas di persimpangan miring dan hampir roboh",
            "jembatan penghubung antar desa retak parah",
            "kabel listrik dan tiang PJU menjuntai rendah ke jalan",
        ],
        "emergency_issue": [
            "jembatan gantung putus total akibat banjir bandang",
            "tiang listrik roboh menimpa badan jalan dan kabel masih bertegangan",
            "jalan raya amblas sedalam 2 meter dan memutus akses warga",
            "tanggul sungai jebol mengancam puluhan rumah warga",
        ],
        "orgs": ["Dishub Kota Semarang", "Dinas PUPR Jawa Tengah", "PLN UP3 Semarang", "Dinas PUPR Kendal", "Dinas PUPR Semarang", "Dishub Jawa Tengah", "Dishub Surakarta"],
    },
    "Kesehatan & Sanitasi": {
        "issue": [
            "antrean loket pendaftaran puskesmas mengular sejak subuh",
            "stok obat generik dan vitamin di poli umum sering kosong",
            "toilet umum dan ruang tunggu klinik bau menyengat tidak terurus",
            "pelayanan IGD lambat padahal pasien sudah menunggu lama",
            "jadwal dokter poli anak tidak sesuai pengumuman",
            "air bersih di ruang rawat inap mati sejak kemarin malam",
            "fogging nyamuk demam berdarah belum juga dilakukan",
        ],
        "emergency_issue": [
            "belasan warga keracunan makanan massal dan butuh ambulans segera",
            "pasien kritis di IGD belum tertangani karena ambulans rujukan tidak tersedia",
            "wabah demam berdarah melonjak tajam di RT kami dan dua anak dirawat kritis",
            "pipa air bersih tercemar limbah medis dan menyebabkan warga muntah-muntah",
        ],
        "orgs": ["Dinas Kesehatan", "Puskesmas Tembalang", "RSUD Tugurejo", "Puskesmas Pedurungan", "RSUD Moewardi", "Puskesmas Genuk", "RSUD Banyumas"],
    },
    "Layanan & Administrasi": {
        "issue": [
            "pengurusan KTP elektronik molor berminggu-minggu tanpa kepastian",
            "sistem antrean online dukcapil sering error saat jam kerja",
            "berkas pengajuan KK dan akta kelahiran tidak ada kejelasan status",
            "loket pelayanan tutup lebih awal sebelum jam operasional selesai",
            "verifikasi bansos tidak tepat sasaran dan warga rentan malah terlewat",
            "portal pengaduan dan aplikasi pajak daerah gagal login terus",
            "surat domisili usaha belum ditandatangani sejak dua minggu lalu",
        ],
        "emergency_issue": [
            "data kependudukan warga lansia terblokir mendadak saat butuh rujukan operasi darurat BPJS",
            "sistem layanan darurat 112 tidak bisa dihubungi saat terjadi kebakaran rumah warga",
            "pencairan bantuan darurat korban bencana tertahan karena kesalahan input NIK massal",
        ],
        "orgs": ["Disdukcapil Kota Semarang", "Kantor Kecamatan Ungaran", "Kelurahan Sekaran", "Dinas Sosial", "Samsat Semarang", "Disdukcapil Solo", "Kantor Kelurahan Ngaliyan", "Disdukcapil Magelang", "Kantor Kecamatan Genuk"],
    },
    "Lingkungan": {
        "issue": [
            "tumpukan sampah di TPS liar meluber ke badan jalan dan bau busuk",
            "selokan penuh sampah plastik sehingga banjir saat hujan deras",
            "asap pembakaran sampah pabrik mengganggu pernapasan warga",
            "saluran air berwarna hitam pekat diduga tercemar limbah rumah tangga",
            "pohon besar di pinggir jalan sudah miring dan akarnya terangkat",
            "truk pengangkut sampah sudah lima hari tidak mengambil sampah warga",
        ],
        "emergency_issue": [
            "pohon beringin besar tumbang menimpa dua motor dan menutup jalan utama",
            "limbah B3 beracun bocor ke sungai dan menyebabkan ratusan ikan mati mendadak",
            "kebakaran lahan kosong mendekati permukiman padat penduduk",
            "longsor tebing menutup akses jalan dan menimbun satu warung warga",
        ],
        "orgs": ["DLH Kota Surakarta", "BPBD Jawa Tengah", "DLH Jawa Tengah", "BPBD Cilacap", "DLH Banyumas", "BPBD Demak"],
    },
    "Pungli & Tata Kelola": {
        "issue": [
            "oknum petugas meminta uang pelicin Rp100 ribu untuk mempercepat surat",
            "parkir liar mematok tarif Rp10 ribu tanpa karcis resmi",
            "pengambilan berkas gratis di kelurahan masih dimintai biaya administrasi",
            "pendaftaran bantuan UMKM dipotong oknum sebesar Rp50 ribu per warga",
            "ada pungutan liar saat mengurus surat pindah domisili tanpa kwitansi",
            "petugas lapangan menarik iuran keamanan tanpa dasar aturan resmi",
        ],
        "emergency_issue": [
            "oknum memeras warga Rp500 ribu sambil mengancam mencabut bantuan darurat bencana",
            "aksi premanisme dan pungli paksa menghentikan ambulans dan bantuan logistik warga",
        ],
        "orgs": ["Satpol PP", "Polsek Banyumanik", "Inspektorat Daerah", "Kantor Kecamatan Ungaran", "Kelurahan Sekaran", "Kantor Kelurahan Ngaliyan", "Kantor Kecamatan Genuk"],
    },
}

TEMPLATES: List[Tuple[str, str, str, str, int, str, str]] = [
    ("train", "sarkasme", "Keluhan", "Negatif", 1, "High",
     "Mantap sekali pelayanan di {LOC}, {ISSUE}. Terima kasih {ORG} atas pengalaman luar biasa ini."),
    ("train", "sarkasme", "Keluhan", "Negatif", 1, "Med",
     "Hebat banget {ORG}, di {LOC} {ISSUE} dibiarkan berbulan-bulan. Kinerja terbaik pokoknya!"),
    ("train", "sarkasme", "Permintaan", "Negatif", 1, "High",
     "Keren ya {LOC} sekarang punya wisata baru karena {ISSUE}. Tolong {ORG} jangan terlalu cepat perbaiki."),
    ("train", "sarkasme", "Keluhan", "Negatif", 1, "Med",
     "Luar biasa {PER} dan tim {ORG}, laporan soal {ISSUE} di {LOC} sukses diarsipkan tanpa tindakan."),
    ("train", "sarkasme", "Keluhan", "Negatif", 1, "Low",
     "Selamat atas prestasinya, di {LOC} {ISSUE} masih awet sampai tahun depan."),

    ("train", "negasi_implisit", "Keluhan", "Negatif", 0, "High",
     "Bukan tidak menghargai kerja {ORG}, tapi di {LOC} {ISSUE} belum pernah ditangani sampai sekarang."),
    ("train", "negasi_implisit", "Permintaan", "Negatif", 0, "Med",
     "Saya tidak bilang {PER} tidak bekerja, namun masalah {ISSUE} di {LOC} tidak kunjung selesai. Mohon {ORG} cek."),
    ("train", "negasi_implisit", "Keluhan", "Negatif", 0, "Med",
     "Sudah lapor berkali-kali ke {ORG} soal {ISSUE} di {LOC}, tapi boro-boro diperbaiki, ditengok saja belum."),
    ("train", "negasi_implisit", "Permintaan", "Netral", 0, "Low",
     "Tidak perlu muluk-muluk, warga {LOC} hanya minta {ORG} menindaklanjuti {ISSUE} yang belum beres."),
    ("train", "negasi_implisit", "Keluhan", "Negatif", 0, "High",
     "Mana ada warga yang nyaman kalau di {LOC} {ISSUE} dibiarkan tanpa solusi dari {ORG}."),

    ("train", "campur_kode", "Permintaan", "Negatif", 0, "High",
     "Tulung {ORG} cek {LOC}, {ISSUE} wis suwe banget ora ditangani, warga kangelan lewat."),
    ("train", "campur_kode", "Keluhan", "Negatif", 0, "Med",
     "Piye iki {ORG}, neng {LOC} {ISSUE} parah bgt. Kata {PER} blm ada anggaran terus."),
    ("train", "campur_kode", "Permintaan", "Netral", 0, "Med",
     "Min tolong info ke {ORG}, area {LOC} {ISSUE}. Wis lapor minggu wingi tapi blm direspon."),
    ("train", "campur_kode", "Keluhan", "Negatif", 0, "High",
     "Jan parah tenan kondisi {LOC}, {ISSUE} bikin macet tiap pagi. Mohon {ORG} gercep."),
    ("train", "campur_kode", "Saran", "Netral", 0, "Low",
     "Saran buat {ORG}, di {LOC} sebaiknya pasang jadwal rutin biar {ISSUE} ora keulang meneh."),

    ("train", "darurat_pungli", "Darurat", "Negatif", 0, "Emergency",
     "DARURAT! Di {LOC} baru saja {EMERGENCY_ISSUE}. Mohon {ORG} dan {PER} segera kirim tim ke lokasi sekarang juga!"),
    ("train", "darurat_pungli", "Darurat", "Negatif", 0, "Emergency",
     "Tolong segera {ORG}! Kondisi genting di {LOC}, {EMERGENCY_ISSUE} dan membahayakan nyawa warga."),
    ("train", "darurat_pungli", "Keluhan", "Negatif", 0, "High",
     "Mohon {ORG} usut dugaan pelanggaran di {LOC}: {ISSUE}. Warga resah dan butuh tindakan tegas."),

    ("train", "normal", "Saran", "Netral", 0, "Low",
     "Usul kepada {ORG} agar di {LOC} dibuatkan kanal monitoring terbuka terkait {ISSUE}."),
    ("train", "normal", "Saran", "Positif", 0, "Low",
     "Terima kasih tanggapannya kemarin, saran kami {ORG} perlu evaluasi berkala di {LOC} supaya {ISSUE} tidak terulang."),
    ("train", "normal", "Apresiasi", "Positif", 0, "Low",
     "Terima kasih banyak untuk {ORG} dan {PER}, masalah {ISSUE} di {LOC} sudah beres dan warga sangat terbantu."),
    ("train", "normal", "Apresiasi", "Positif", 0, "Low",
     "Apresiasi setinggi-tingginya kepada tim {ORG} yang sigap menangani {ISSUE} di {LOC} hari ini."),
    ("train", "normal", "Permintaan", "Netral", 0, "Med",
     "Mohon bantuan {ORG} untuk meninjau {LOC} karena {ISSUE} mulai mengganggu aktivitas warga."),

    ("val", "sarkasme", "Keluhan", "Negatif", 1, "High",
     "Wah top markotop kinerja {ORG}, di {LOC} {ISSUE} makin parah tiap minggu. Keren banget!"),
    ("val", "sarkasme", "Permintaan", "Negatif", 1, "Med",
     "Terima kasih sudah bikin warga {LOC} sabar ekstra karena {ISSUE}, tolong {ORG} lanjutkan prestasinya."),

    ("val", "negasi_implisit", "Keluhan", "Negatif", 0, "High",
     "Bukannya kami tidak sabar, tapi di {LOC} {ISSUE} tidak ada perkembangan sama sekali dari {ORG}."),
    ("val", "negasi_implisit", "Permintaan", "Negatif", 0, "Med",
     "Hingga saat ini {ORG} belum juga turun ke {LOC}, padahal {ISSUE} bukan masalah sepele."),

    ("val", "campur_kode", "Keluhan", "Negatif", 0, "High",
     "Waduh {ORG}, neng {LOC} {ISSUE} wis suwe ora beres-beres. Tolonglah {PER} dicek langsung."),
    ("val", "campur_kode", "Permintaan", "Netral", 0, "Med",
     "Halo {ORG}, daerah {LOC} {ISSUE} bgt nih, blm ada petugas yg dateng dari kemarin."),

    ("val", "darurat_pungli", "Darurat", "Negatif", 0, "Emergency",
     "BAHAYA! Kejadian darurat di {LOC}: {EMERGENCY_ISSUE}. Segera kerahkan petugas {ORG} sekarang!"),
    ("val", "darurat_pungli", "Keluhan", "Negatif", 0, "High",
     "Laporan resmi ke {ORG} mengenai {ISSUE} di {LOC}, kata {PER} kejadian ini sering berulang."),

    ("val", "normal", "Saran", "Netral", 0, "Low",
     "Masukan untuk {ORG}, sebaiknya di {LOC} dilakukan pengecekan rutin agar {ISSUE} bisa dicegah."),
    ("val", "normal", "Apresiasi", "Positif", 0, "Low",
     "Mantap, laporan di {LOC} langsung direspons cepat oleh {ORG} dan {PER}. Masalah {ISSUE} kini teratasi."),

    ("test", "sarkasme", "Keluhan", "Negatif", 1, "High",
     "Luar biasa memang {ORG}, {ISSUE} di {LOC} dibiarkan sampai warga hampir celaka. Salut!"),
    ("test", "sarkasme", "Permintaan", "Negatif", 1, "Med",
     "Juara satu deh pelayanan {LOC}, {ISSUE} awet banget. Kapan-kapan {ORG} mampir lihat ya."),

    ("test", "negasi_implisit", "Keluhan", "Negatif", 0, "High",
     "Kami tidak menuntut hal berlebihan, namun di {LOC} {ISSUE} tidak pernah diselesaikan {ORG}."),
    ("test", "negasi_implisit", "Permintaan", "Negatif", 0, "Med",
     "Jangan sampai menunggu korban dulu baru {ORG} bergerak ke {LOC}, soalnya {ISSUE} belum ditangani."),

    ("test", "campur_kode", "Keluhan", "Negatif", 0, "High",
     "Aduh {ORG}, kondisi {LOC} parah tenan, {ISSUE} wis pirang-pirang dina ora ono tindak lanjut."),
    ("test", "campur_kode", "Permintaan", "Netral", 0, "Med",
     "Min tolong terusin ke {ORG}, wilayah {LOC} {ISSUE} nih, dari kemarin blm ada yg cek."),

    ("test", "darurat_pungli", "Darurat", "Negatif", 0, "Emergency",
     "URGENT! Di {LOC} barusan {EMERGENCY_ISSUE}. Warga butuh evakuasi cepat dari {ORG} dan {PER}!"),
    ("test", "darurat_pungli", "Keluhan", "Negatif", 0, "High",
     "Warga {LOC} melaporkan {ISSUE} dan meminta {ORG} segera menindak oknum terkait."),

    ("test", "normal", "Saran", "Netral", 0, "Low",
     "Mohon pertimbangan {ORG} untuk memperbaiki SOP di {LOC} supaya {ISSUE} tidak terjadi lagi."),
    ("test", "normal", "Apresiasi", "Positif", 0, "Low",
     "Alhamdulillah, {ORG} bersama {PER} bergerak cepat menuntaskan {ISSUE} di {LOC}. Terima kasih!"),
]

NO_ENTITY_TEMPLATES: Dict[str, List[Tuple[str, str, str, str, int, str, str]]] = {
    "train": [
        ("train", "negasi_implisit", "Keluhan", "Negatif", 0, "Med",
         "Laporan soal {ISSUE} sudah masuk seminggu lalu tapi belum ada balasan sama sekali."),
        ("train", "sarkasme", "Keluhan", "Negatif", 1, "Med",
         "Keren banget sistem aduannya, {ISSUE} dilaporkan malah statusnya langsung ditutup."),
        ("train", "campur_kode", "Permintaan", "Netral", 0, "Low",
         "Min tulung dicek ya, {ISSUE} masih kejadian terus dari kemarin pagi."),
    ],
    "val": [
        ("val", "negasi_implisit", "Keluhan", "Negatif", 0, "Med",
         "Bukan bermaksud menyalahkan, tapi {ISSUE} tidak kunjung beres sejak awal bulan."),
    ],
    "test": [
        ("test", "sarkasme", "Keluhan", "Negatif", 1, "High",
         "Hebat sekali responsnya, {ISSUE} dibiarkan berlarut tanpa kejelasan."),
    ],
}

DOMAIN_WEIGHTS = {
    "Infrastruktur & PU": 0.25,
    "Layanan & Administrasi": 0.25,
    "Kesehatan & Sanitasi": 0.20,
    "Lingkungan": 0.20,
    "Pungli & Tata Kelola": 0.10,
}


def pick_weighted_domain(rng: random.Random, force_rare: bool = False) -> str:
    if force_rare:
        return rng.choices(
            ["Pungli & Tata Kelola", "Infrastruktur & PU", "Kesehatan & Sanitasi", "Lingkungan", "Layanan & Administrasi"],
            weights=[0.45, 0.15, 0.15, 0.15, 0.10],
            k=1,
        )[0]
    domains = list(DOMAIN_WEIGHTS.keys())
    weights = list(DOMAIN_WEIGHTS.values())
    return rng.choices(domains, weights=weights, k=1)[0]


def render_with_entities(
    template: str,
    slots: Dict[str, Tuple[str, Optional[str]]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Render template string and track exact character spans for entity slots.
    slots maps placeholder name (e.g. 'LOC') -> (value_str, entity_label_or_None).
    """
    pattern = re.compile(r"\{([A-Z_]+)\}")
    parts: List[str] = []
    entities: List[Dict[str, Any]] = []
    cursor = 0
    out_len = 0

    for match in pattern.finditer(template):
        lit = template[cursor : match.start()]
        parts.append(lit)
        out_len += len(lit)

        key = match.group(1)
        val, label = slots[key]
        start = out_len
        parts.append(val)
        out_len += len(val)
        end = out_len

        if label is not None:
            entities.append({"start": start, "end": end, "label": label})

        cursor = match.end()

    tail = template[cursor:]
    parts.append(tail)
    text = "".join(parts)

    for ent in entities:
        s, e = ent["start"], ent["end"]
        assert 0 <= s < e <= len(text), f"Invalid span {(s, e)} for text len {len(text)}"
        assert text[s:e].strip() == text[s:e] and text[s:e], f"Bad span slice: '{text[s:e]}'"

    return text, entities


def pick_org_for_split(rng: random.Random, domain: str, split: str) -> str:
    pool = ENTITY_POOLS[split]["ORGANIZATION"]
    domain_orgs = [o for o in DOMAIN_TOPICS[domain]["orgs"] if o in pool]
    if domain_orgs:
        return rng.choice(domain_orgs)
    return rng.choice(pool)


def generate_for_split(
    split: str,
    quota: int,
    rng: random.Random,
    seen_texts: set[str],
    counter_start: int,
) -> List[Dict[str, Any]]:
    split_templates = [t for t in TEMPLATES if t[0] == split]
    no_ent_templates = NO_ENTITY_TEMPLATES[split]
    records: List[Dict[str, Any]] = []
    idx = counter_start

    prefixes = ["", "", "", "[Laporan Warga] ", "Mohon perhatian, ", "Info warga: ", "Permisi admin, "]
    suffixes = ["", "", "", " Mohon segera ditindaklanjuti.", " Tolong jangan diabaikan.", " Terima kasih."]

    attempts = 0
    max_attempts = quota * 30
    while len(records) < quota and attempts < max_attempts:
        attempts += 1
        use_no_entity = rng.random() < 0.08
        if use_no_entity:
            _, phenomenon, intent, sentiment, is_sarc, urgency, tpl = rng.choice(no_ent_templates)
        else:
            _, phenomenon, intent, sentiment, is_sarc, urgency, tpl = rng.choice(split_templates)

        if phenomenon == "darurat_pungli" and intent != "Darurat":
            domain = "Pungli & Tata Kelola"
        elif intent == "Darurat":
            domain = rng.choices(
                ["Infrastruktur & PU", "Kesehatan & Sanitasi", "Lingkungan", "Layanan & Administrasi", "Pungli & Tata Kelola"],
                weights=[0.30, 0.25, 0.25, 0.12, 0.08],
                k=1,
            )[0]
        else:
            domain = pick_weighted_domain(rng, force_rare=(phenomenon == "darurat_pungli"))

        loc = rng.choice(ENTITY_POOLS[split]["LOCATION"])
        org = pick_org_for_split(rng, domain, split)
        per = rng.choice(ENTITY_POOLS[split]["PERSON"])
        issue = rng.choice(DOMAIN_TOPICS[domain]["issue"])
        em_issue = rng.choice(DOMAIN_TOPICS[domain]["emergency_issue"])

        pref = rng.choice(prefixes)
        suf = rng.choice(suffixes) if not tpl.endswith("!") else ""
        full_tpl = f"{pref}{tpl}{suf}".strip()

        slots: Dict[str, Tuple[str, Optional[str]]] = {
            "LOC": (loc, "LOCATION"),
            "ORG": (org, "ORGANIZATION"),
            "PER": (per, "PERSON"),
            "ISSUE": (issue, None),
            "EMERGENCY_ISSUE": (em_issue, None),
        }

        text, entities = render_with_entities(full_tpl, slots)
        norm = re.sub(r"\s+", " ", text.lower()).strip()
        if norm in seen_texts:
            continue
        seen_texts.add(norm)

        assert domain in VALID_DOMAINS
        assert intent in VALID_INTENTS
        assert sentiment in VALID_SENTIMENTS
        assert urgency in VALID_URGENCY_LEVELS

        idx += 1
        records.append(
            {
                "id": f"syn_{idx:04d}",
                "source": "synthetic",
                "split": split,
                "text": text,
                "domain": domain,
                "intent": intent,
                "entities": entities,
                "sentiment": sentiment,
                "is_sarcasm": is_sarc,
                "urgency_level": urgency,
                "phenomenon": phenomenon,
            }
        )

    if len(records) != quota:
        raise RuntimeError(f"Could only generate {len(records)}/{quota} unique rows for split={split}")
    return records


def main() -> None:
    rng = random.Random(SEED)
    seen_texts: set[str] = set()
    all_records: List[Dict[str, Any]] = []

    offset = 0
    for split in ("train", "val", "test"):
        quota = SPLIT_QUOTAS[split]
        recs = generate_for_split(split, quota, rng, seen_texts, offset)
        all_records.extend(recs)
        offset += len(recs)

    assert len(all_records) == TARGET_TOTAL, f"Expected {TARGET_TOTAL}, got {len(all_records)}"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Generated {len(all_records):,} synthetic edge cases -> {OUT_PATH.relative_to(ROOT_DIR)}")
    print("Splits      :", dict(Counter(r["split"] for r in all_records)))
    print("Domains     :", dict(Counter(r["domain"] for r in all_records)))
    print("Intents     :", dict(Counter(r["intent"] for r in all_records)))
    print("Sentiments  :", dict(Counter(r["sentiment"] for r in all_records)))
    print("Is Sarcasm  :", dict(Counter(r["is_sarcasm"] for r in all_records)))
    print("Urgency     :", dict(Counter(r["urgency_level"] for r in all_records)))
    print("Phenomena   :", dict(Counter(r["phenomenon"] for r in all_records)))


if __name__ == "__main__":
    main()

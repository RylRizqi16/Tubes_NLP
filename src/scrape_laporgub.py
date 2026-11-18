"""
Scraper untuk LaporGub Jawa Tengah (laporgub.jatengprov.go.id)
================================================================
Dibuat untuk keperluan riset akademik (Tugas Besar NLP - Klasifikasi
Pengaduan Publik Indonesia).

BACA DULU SEBELUM MENJALANKAN
------------------------------
1. Cek robots.txt situs ini SENDIRI di browser sebelum memakai script ini:
       https://laporgub.jatengprov.go.id/robots.txt
   Script ini TIDAK memeriksa atau mematuhi robots.txt secara otomatis.
   Kamu yang bertanggung jawab memastikan penggunaannya sesuai aturan situs.

2. Script ini TIDAK berjalan otomatis / terjadwal / di-trigger pihak lain.
   Kamu yang menjalankan setiap perintah secara manual dari terminal, dan
   kamu yang memutuskan kapan, berapa halaman, dan seberapa cepat proses
   scraping berjalan (lewat argumen --start-page/--end-page/--delay).

3. Ini server milik Pemprov Jateng untuk layanan publik sungguhan. Pakai
   delay yang wajar (default 1.5 detik/request), jangan paralel besar-besaran,
   dan hentikan kalau diminta admin situs.

4. HASIL SCRAPING BELUM AMAN DIPAKAI LANGSUNG. Banyak teks aduan memuat nama
   lengkap pelapor, nomor HP, dsb. Jalankan redact_pii.py (file terpisah)
   pada hasilnya sebelum dipakai untuk training atau dibagikan ke siapa pun.

5. Parsing di script ini bersifat BEST-EFFORT berdasarkan struktur halaman
   yang teramati saat script ini ditulis. Markup situs bisa berubah kapan
   saja. SELALU uji dulu dengan rentang kecil (mis. --start-page 1 --end-page 2)
   dan periksa manual hasil CSV-nya sebelum menjalankan untuk ribuan halaman.
   Kalau hasil parsing berantakan, buka "View Page Source" di browser pada
   salah satu halaman dan sesuaikan fungsi parse_listing_page / parse_detail_page
   di bawah.

Cara pakai
----------
--contact-email WAJIB diisi di semua perintah (email asli kamu, masuk ke
User-Agent supaya admin situs bisa menghubungi kalau ada masalah traffic).

Tahap 1 - kumpulkan daftar kode aduan dari halaman listing:
    python scrape_laporgub.py --contact-email nama@kampus.ac.id listing \\
        --start-page 1 --end-page 50 --output listing.csv

Tahap 2 - ambil teks lengkap tiap aduan dari halaman detail:
    python scrape_laporgub.py --contact-email nama@kampus.ac.id detail \\
        --input listing.csv --output detail.csv --delay 1.5

Kalau file CSV akan dibuka di Excel dengan regional setting Indonesia/Eropa
(yang pakai ';' sebagai delimiter default), tambahkan --csv-delimiter ";"
di kedua tahap di atas supaya kolom tidak menumpuk jadi satu di Excel.

Script ini resumable: kalau terhenti (Ctrl+C, error jaringan, dsb), jalankan
ulang perintah yang sama dan kode yang sudah pernah tersimpan di --output
akan dilewati.
"""

import argparse
import csv
import os
import re
import sys
import time
import random

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Perlu install dulu: pip install requests beautifulsoup4 --break-system-packages")
    sys.exit(1)

BASE_URL = "https://laporgub.jatengprov.go.id"
LISTING_URL = BASE_URL + "/jelajah"
DETAIL_URL_TMPL = BASE_URL + "/detail/{code}.html"

# Email kontak sekarang WAJIB diisi lewat --contact-email (lihat main()), bukan
# ditulis langsung di source. Ini supaya script tidak bisa "lupa" ter-commit
# atau dijalankan dengan placeholder yang tidak pernah diganti.
USER_AGENT_TMPL = (
    "Mozilla/5.0 (compatible; AcademicResearchBot/1.0; "
    "Tugas Besar NLP Klasifikasi Pengaduan Publik; "
    "contact: {email})"
)

CODE_PATTERN = re.compile(r"/detail/([A-Z0-9]{8,})\.html")

# Dipakai bersama oleh parse_listing_page (lewat CARD_PATTERN) dan
# parse_detail_page, supaya keduanya konsisten mengenali nama kabupaten/kota
# yang mengandung huruf beraksen (contoh: "KOTA BANJARNEGARA" vs varian lain).
# Sebelumnya CARD_PATTERN mendukung [A-ZÉÈ...] tapi parse_detail_page hanya
# [A-Z...] saja, sehingga lokasi beraksen gagal terdeteksi di halaman detail.
LOCATION_ALTERNATION = r"KABUPATEN [A-ZÉÈ\s./]+|KOTA [A-ZÉÈ\s./]+|LAIN-LAIN"

# Pola heuristik untuk satu kartu aduan di halaman listing, berdasar urutan
# teks yang teramati: KODE  KAB/KOTA  TANGGAL  jumlah_ditandai  "ditandai"  cuplikan  STATUS
CARD_PATTERN = re.compile(
    r"(?P<code>[A-Z]{2,4}\d{8,})\s+"
    r"(?P<location>" + LOCATION_ALTERNATION + r")\s+"
    r"(?P<date>\d{1,2}\s\w{3}\s\d{4})\s+"
    r"\d+\s+ditandai\s+"
    r"(?P<snippet>.*?)\s+"
    r"(?P<status>Disposisi|Verifikasi|Diproses|Terkirim|Ditutup|Selesai)",
    re.DOTALL,
)

# CSS selector yang dicoba LEBIH DULU, sebelum jatuh ke regex-on-text.
# Ini TEBAKAN GENERIK (nama kelas umum dipakai banyak template Laravel/Bootstrap
# untuk daftar kartu) - BUKAN hasil inspeksi markup situs ini secara langsung.
# WAJIB divalidasi: buka salah satu halaman listing di browser, klik kanan
# salah satu kartu aduan -> "Inspect", lalu cocokkan/ganti selector di bawah
# dengan class/tag yang benar-benar dipakai situs. Kalau tidak ada yang cocok,
# kode otomatis jatuh ke CARD_PATTERN (regex-on-text), jadi tidak merusak apa pun.
LISTING_CARD_SELECTORS = [
    ".card", "[class*='card-item']", "[class*='list-item']",
    "article", "li[class*='item']",
]
DETAIL_META_SELECTORS = [
    "[class*='sidebar']", "[class*='meta']", "[class*='detail-header']",
    "[class*='info']", "header",
]


def fetch(url, headers, max_retries=3, timeout=20):
    """Ambil halaman dengan retry sederhana. Return None kalau gagal total."""
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                return resp.text
            print(f"  [!] Status {resp.status_code} untuk {url} (percobaan {attempt})")
        except requests.RequestException as e:
            print(f"  [!] Error fetch {url}: {e} (percobaan {attempt})")
        time.sleep(2 * attempt)
    return None


def _parse_card_element(card):
    """
    Coba ekstrak satu kartu aduan dari elemen HTML-nya langsung (bukan dari
    teks yang sudah diratakan). Lebih tahan terhadap whitespace/kartu yang
    sedikit beda bentuk dibanding regex-on-text, tapi masih bergantung pada
    CARD_PATTERN untuk membaca field-field di dalam teks kartu tsb, karena
    kita tidak tahu nama class spesifik per field (lihat komentar di
    LISTING_CARD_SELECTORS). Return dict atau None kalau tidak ada kode di
    dalam kartu ini sama sekali.
    """
    link = card.find("a", href=CODE_PATTERN)
    if link is None:
        return None
    code_match = CODE_PATTERN.search(link.get("href", ""))
    if not code_match:
        return None
    code = code_match.group(1)

    card_text = card.get_text(separator=" ", strip=True)
    m = CARD_PATTERN.search(card_text)
    if m:
        return {
            "code": code,
            "location": m.group("location").strip(),
            "date": m.group("date").strip(),
            "snippet": m.group("snippet").strip(),
            "status": m.group("status").strip(),
        }
    # Kode ketemu lewat href, tapi field lain gagal di-parse dari teks kartu.
    return {"code": code, "location": "", "date": "", "snippet": "", "status": ""}


def parse_listing_page(html):
    """
    Ekstrak daftar aduan dari satu halaman listing.
    Return list of dict: {code, location, date, snippet, status}

    Tiga tingkat fallback, dari yang paling akurat ke paling kasar:
      1. CSS selector (LISTING_CARD_SELECTORS) per kartu, field-field dibaca
         dari teks kartu itu sendiri (lebih tahan terhadap elemen lain di
         halaman yang ikut ketangkap kalau kita regex seluruh halaman).
      2. Kalau tidak ada selector yang cocok sama sekali: CARD_PATTERN pada
         teks seluruh halaman (perilaku lama).
      3. Kalau itu juga gagal total: ambil semua kode dari href saja, field
         lain dikosongkan.
    Semua tingkat ini best-effort. Kalau banyak field kosong, cek manual
    struktur HTML asli (lihat komentar LISTING_CARD_SELECTORS) dan sesuaikan.
    """
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find(id="main-content") or soup

    # Tingkat 1: CSS selector per kartu.
    for selector in LISTING_CARD_SELECTORS:
        cards = main.select(selector)
        if not cards:
            continue
        results = []
        for card in cards:
            parsed = _parse_card_element(card)
            if parsed:
                results.append(parsed)
        if results:
            return results

    # Tingkat 2: regex pada teks seluruh halaman (perilaku lama, sebelum ada
    # CSS selector di atas).
    print("  [i] Tidak ada CSS selector kartu yang cocok, jatuh ke regex-on-text.")
    text = main.get_text(separator=" ", strip=True)
    results = []
    for m in CARD_PATTERN.finditer(text):
        results.append({
            "code": m.group("code"),
            "location": m.group("location").strip(),
            "date": m.group("date").strip(),
            "snippet": m.group("snippet").strip(),
            "status": m.group("status").strip(),
        })
    if results:
        return results

    # Tingkat 3: minimal ambil semua kode dari href.
    print("  [i] Regex-on-text juga gagal, jatuh ke kode dari href saja (field lain kosong).")
    codes = set(CODE_PATTERN.findall(html))
    return [{"code": c, "location": "", "date": "", "snippet": "", "status": ""} for c in codes]


def parse_detail_page(html, code):
    """
    Ekstrak teks lengkap + metadata dari satu halaman detail aduan.
    Return dict. Field yang gagal dideteksi akan tetap ada tapi kosong,
    supaya tidak menghentikan proses scraping keseluruhan.

    Metadata (lokasi/tanggal/status) dicari DUA TAHAP:
      1. Di dalam elemen "meta/sidebar/header" (DETAIL_META_SELECTORS) kalau
         ada yang cocok - supaya tidak salah tangkap nama kota atau tanggal
         yang disebut di DALAM isi aduan warga, bukan di metadata aduan.
      2. Kalau tidak ada elemen semacam itu yang cocok, baru cari di
         full_text seluruh halaman (perilaku lama, lebih berisiko salah
         tangkap tapi lebih baik daripada kosong).
    """
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find(id="main-content") or soup
    full_text = main.get_text(separator="\n", strip=True)

    location_re = re.compile(r"\b(" + LOCATION_ALTERNATION + r")\b")
    date_re = re.compile(r"\b\d{1,2}\s\w{3}\s\d{4}\b")
    status_re = re.compile(r"\b(Disposisi|Verifikasi|Diproses|Terkirim|Ditutup|Selesai)\b")

    meta_text = ""
    for selector in DETAIL_META_SELECTORS:
        found = main.select(selector)
        if found:
            meta_text = " ".join(el.get_text(separator=" ", strip=True) for el in found)
            break

    def search_meta_then_full(pattern):
        m = pattern.search(meta_text) if meta_text else None
        if m:
            return m
        return pattern.search(full_text)

    location_match = search_meta_then_full(location_re)
    date_match = search_meta_then_full(date_re)
    status_match = search_meta_then_full(status_re)

    return {
        "code": code,
        "location": location_match.group(0).strip() if location_match else "",
        "date": date_match.group(0).strip() if date_match else "",
        "status": status_match.group(0).strip() if status_match else "",
        # raw_text disimpan utuh supaya tidak ada informasi yang hilang kalau
        # heuristik di atas kurang akurat; bisa diparsing ulang belakangan.
        "raw_text": full_text,
    }


def load_existing_codes(path, delimiter=",", key="code"):
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        return {row[key] for row in reader if key in row}


def append_rows(path, rows, fieldnames, delimiter=","):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_listing(args, headers):
    fieldnames = ["code", "location", "date", "snippet", "status", "page"]
    seen_codes = load_existing_codes(args.output, delimiter=args.csv_delimiter)
    print(f"Sudah ada {len(seen_codes)} kode tersimpan sebelumnya di {args.output}")

    for page in range(args.start_page, args.end_page + 1):
        url = f"{LISTING_URL}?page={page}"
        print(f"[listing] halaman {page} -> {url}")
        html = fetch(url, headers)
        if html is None:
            print(f"  [!] Lewati halaman {page}, gagal diambil setelah retry.")
            continue

        entries = parse_listing_page(html)
        new_entries = [e for e in entries if e["code"] not in seen_codes]
        for e in new_entries:
            e["page"] = page
            seen_codes.add(e["code"])

        if new_entries:
            append_rows(args.output, new_entries, fieldnames, delimiter=args.csv_delimiter)
        print(f"  -> {len(entries)} ditemukan, {len(new_entries)} baru disimpan")

        time.sleep(args.delay + random.uniform(0, 0.5))

    print(f"Selesai. Total kode unik tersimpan: {len(seen_codes)}")


def run_detail(args, headers):
    if not os.path.exists(args.input):
        print(f"File input tidak ditemukan: {args.input}")
        sys.exit(1)

    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=args.csv_delimiter)
        all_codes = [row["code"] for row in reader if row.get("code")]

    fieldnames = ["code", "location", "date", "status", "raw_text"]
    done_codes = load_existing_codes(args.output, delimiter=args.csv_delimiter)
    todo_codes = [c for c in all_codes if c not in done_codes]
    print(f"Total {len(all_codes)} kode, {len(done_codes)} sudah selesai, {len(todo_codes)} sisa.")

    for i, code in enumerate(todo_codes, 1):
        url = DETAIL_URL_TMPL.format(code=code)
        print(f"[detail] {i}/{len(todo_codes)} -> {code}")
        html = fetch(url, headers)
        if html is None:
            print(f"  [!] Lewati {code}, gagal diambil setelah retry.")
            continue

        row = parse_detail_page(html, code)
        append_rows(args.output, [row], fieldnames, delimiter=args.csv_delimiter)

        time.sleep(args.delay + random.uniform(0, 0.5))

    print("Selesai.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--contact-email", required=True,
        help="Email asli kamu, dicantumkan di User-Agent supaya admin situs bisa "
             "menghubungi kalau ada masalah dengan traffic scraping ini. WAJIB diisi.",
    )
    parser.add_argument(
        "--csv-delimiter", default=",",
        help="Delimiter CSV (default ','). Pakai ';' kalau file akan dibuka di Excel "
             "dengan regional setting Indonesia/Eropa, supaya Excel tidak menumpuk "
             "semua kolom jadi satu kolom A.",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_list = sub.add_parser("listing", help="Scrape halaman listing /jelajah untuk kumpulkan kode aduan")
    p_list.add_argument("--start-page", type=int, default=1)
    p_list.add_argument("--end-page", type=int, required=True)
    p_list.add_argument("--output", default="listing.csv")
    p_list.add_argument("--delay", type=float, default=1.5, help="Jeda detik antar request (default 1.5)")

    p_detail = sub.add_parser("detail", help="Scrape halaman detail tiap kode aduan dari hasil listing")
    p_detail.add_argument("--input", default="listing.csv", help="CSV hasil tahap listing")
    p_detail.add_argument("--output", default="detail.csv")
    p_detail.add_argument("--delay", type=float, default=1.5, help="Jeda detik antar request (default 1.5)")

    args = parser.parse_args()

    if "@" not in args.contact_email or args.contact_email.strip("<>").upper() == "ISI_EMAIL_KAMU":
        print("--contact-email harus email asli yang valid (ada tanda '@'), bukan placeholder.")
        sys.exit(1)

    headers = {"User-Agent": USER_AGENT_TMPL.format(email=args.contact_email)}

    if args.mode == "listing":
        run_listing(args, headers)
    elif args.mode == "detail":
        run_detail(args, headers)


if __name__ == "__main__":
    main()

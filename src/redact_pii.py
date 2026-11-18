"""
Redaksi identitas pribadi (PII) untuk hasil scraping LaporGub
===============================================================
Jalankan ini pada hasil scrape_laporgub.py SEBELUM dataset dipakai untuk
training, dianalisis, atau dibagikan ke siapa pun (termasuk dosen/asisten
kalau nanti dilampirkan contoh mentah).

Ini WAJIB karena teks aduan warga sering memuat nama lengkap, nomor HP,
dan kadang NIK secara eksplisit dalam kalimat (mis. "Nama: Budi Santoso,
HP 0812xxxxxxx").

Dua mode:
1. Mode regex saja (default, tidak perlu install apa-apa tambahan selain
   pandas) - menangkap pola eksplisit: NIK 16 digit, nomor HP, dan baris
   berformat "Nama: ...". Ini TIDAK menangkap nama yang disebut tanpa
   label eksplisit di tengah kalimat.
2. Mode spaCy (opsional, lebih baik) - kalau sudah ada model spaCy Bahasa
   Indonesia (sesuai rencana T2 kalian), gunakan NER untuk mask entitas
   PERSON di mana pun munculnya, tidak hanya setelah "Nama:".

PENTING: regex saja TIDAK CUKUP untuk menjamin semua nama ter-mask. Untuk
dataset yang akan disimpan/dibagikan dalam jangka panjang, sangat disarankan
pakai mode spaCy atau review manual pada sampel acak setelah proses ini.

Cara pakai
----------
    python redact_pii.py --input detail.csv --output detail_redacted.csv --text-col raw_text
    python redact_pii.py --input detail.csv --output detail_redacted.csv --text-col raw_text --use-spacy
"""

import argparse
import csv
import re
import sys

NIK_PATTERN = re.compile(r"\b\d{16}\b")
PHONE_PATTERN = re.compile(r"\b(?:\+?62|0)8\d{8,11}\b")
NAME_LABEL_PATTERN = re.compile(
    r"(Nama\s*[:\-]\s*)([A-Z][A-Za-z.'\u2019\s]{2,60}?)(?=[,\n]|$|\s{2,}|Alamat|HP|No\.?\s*HP|Telp)",
    re.IGNORECASE,
)


def redact_regex(text):
    if not text:
        return text
    text = NIK_PATTERN.sub("[NIK_DIREDAKSI]", text)
    text = PHONE_PATTERN.sub("[NOMOR_HP_DIREDAKSI]", text)
    text = NAME_LABEL_PATTERN.sub(lambda m: m.group(1) + "[NAMA_DIREDAKSI]", text)
    return text


def load_spacy_model():
    try:
        import spacy
    except ImportError:
        print("spaCy belum terinstall. Jalankan: pip install spacy --break-system-packages")
        print("lalu download model Bahasa Indonesia yang kalian pakai untuk T2.")
        sys.exit(1)
    try:
        # Sesuaikan nama model dengan yang dipakai di notebook T2_ner_lda.ipynb
        return spacy.load("xx_ent_wiki_sm")
    except OSError:
        print("Model spaCy tidak ditemukan. Sesuaikan nama model di fungsi load_spacy_model().")
        print("Contoh: python -m spacy download xx_ent_wiki_sm")
        sys.exit(1)


def redact_with_spacy(text, nlp):
    if not text:
        return text
    text = redact_regex(text)  # tetap jalankan regex dulu untuk NIK/HP
    doc = nlp(text)
    redacted = text
    # Ganti dari belakang supaya index tidak bergeser
    for ent in sorted(doc.ents, key=lambda e: e.start_char, reverse=True):
        if ent.label_ in ("PERSON", "PER"):
            redacted = redacted[: ent.start_char] + "[NAMA_DIREDAKSI]" + redacted[ent.end_char :]
    return redacted


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-col", default="raw_text", help="Nama kolom yang berisi teks aduan")
    parser.add_argument("--use-spacy", action="store_true", help="Pakai NER spaCy, bukan regex saja")
    args = parser.parse_args()

    nlp = load_spacy_model() if args.use_spacy else None

    with open(args.input, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if args.text_col not in fieldnames:
        print(f"Kolom '{args.text_col}' tidak ada di {args.input}. Kolom tersedia: {fieldnames}")
        sys.exit(1)

    for row in rows:
        original = row[args.text_col]
        row[args.text_col] = redact_with_spacy(original, nlp) if args.use_spacy else redact_regex(original)

    with open(args.output, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Selesai. {len(rows)} baris diproses -> {args.output}")
    print("Tetap lakukan spot-check manual pada sampel acak sebelum dataset dianggap aman dipakai.")


if __name__ == "__main__":
    main()

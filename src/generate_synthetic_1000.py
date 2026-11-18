"""
Generator Skala Besar Dataset Sintetis Pengaduan Publik Indonesia (>= 1000 Sampel)
Membangun korpus sintetis pengaduan publik dengan anotasi ground truth presisi untuk:
- Task 1 (Farhan): Domain & Intent
- Task 2 (Raka): Named Entity Recognition (NER)
- Task 3 (Darryl): Sentiment & Sarcasm Detection
- Priority Engine: Urgency & Priority Level
"""

import os
import csv
import json
import random
from typing import List, Dict, Any, Tuple
import pandas as pd

from src.schema import ComplaintEvent
from src.preprocessing.audit import audit_dataframe


# =========================================================================
# KATALOG ENTITAS LENGKAP JAWA TENGAH & INDONESIA
# =========================================================================

CITIES = [
    "Semarang", "Surakarta", "Solo", "Magelang", "Pekalongan", "Salatiga", "Tegal",
    "Banyumas", "Purwokerto", "Kudus", "Pati", "Jepara", "Brebes", "Cilacap",
    "Klaten", "Boyolali", "Sukoharjo", "Karanganyar", "Sragen", "Wonogiri",
    "Kendal", "Batang", "Pemalang", "Purbalingga", "Banjarnegara", "Kebumen",
    "Wonosobo", "Temanggung", "Grobogan", "Purwodadi", "Blora", "Rembang", "Demak"
]

STREETS = [
    "Jalan Pemuda", "Jalan Pandanaran", "Jalan Slamet Riyadi", "Jalan Brigjen Sudiarto",
    "Jalan Pahlawan", "Jalan Veteran", "Jalan Ahmad Yani", "Jalan Jenderal Sudirman",
    "Jalan Raya Pantura", "Jalan Diponegoro", "Jalan Gajah Mada", "Jalan Majapahit",
    "Jalan Dr. Radjiman", "Jalan Piere Tendean", "Jalan Kolonel Sugiono", "Jalan Kaligawe",
    "Jalan MT Haryono", "Jalan Sultan Agung", "Jalan Dr. Wahidin", "Jalan Imam Bonjol"
]

VILLAGES = [
    "Desa Ketitangwetan", "Kelurahan Jebres", "Desa Tuntang", "Kelurahan Mintaragen",
    "Desa Karangpucung", "Desa Jatisrono", "Kelurahan Sendangguwo", "Desa Sokaraja Kulon",
    "Desa Tambakmulyo", "Kelurahan Banyumanik", "Kelurahan Mangkubumen", "Desa Candirejo",
    "Kelurahan Kaligawe", "Desa Sambirejo", "Desa Banyudono", "Desa Kedungwuni",
    "Desa Dukun", "Kelurahan Krapyak", "Kelurahan Pedurungan", "Kelurahan Gemolong",
    "Desa Karanganyar", "Kelurahan Nusukan", "Desa Kradenan", "Desa Karangmalang"
]

PERSONS = [
    "Pak Ganjar Pranowo", "Mas Gibran Rakabuming", "Mas Nana Sudjana",
    "Pak Hendrar Prihadi", "Bu Risma", "dr. Budi Gunadi Sadikin",
    "Lurah Joko", "Lurah Bambang", "Kepala Desa Supardi", "Pak Camat Wahyu"
]

DURATIONS = [
    "sudah 2 bulan lebih", "sejak seminggu lalu", "sudah hampir 6 bulan",
    "sejak musim hujan kemarin", "hampir setahun terakhir", "sudah 3 minggu",
    "dari awal bulan lalu", "sejak kemarin sore", "sudah berbulan-bulan"
]

IMPACTS = [
    "sangat membahayakan keselamatan pengendara motor",
    "mengakibatkan kemacetan panjang setiap jam berangkat kerja",
    "mengganggu aktivitas warga dan anak sekolah",
    "sudah memakan korban luka-luka jatuh terpeleset",
    "menimbulkan bau tidak sedap dan sarang penyakit",
    "membuat warga sangat resah dan terganggu",
    "mengancam keselamatan anak-anak kecil yang melintas"
]

CLOSINGS = [
    "Mohon dinas terkait segera melakukan perbaikan.",
    "Tolong petugas segera turun langsung ke lapangan.",
    "Kami memohon tindak lanjut secepatnya dari pihak berwenang.",
    "Harap segera diselesaikan sebelum timbul korban jiwa.",
    "Mohon dievaluasi kinerjanya dan segera diperbaiki.",
    "Tolong jangan dibiarkan berlarut-larut tanpa penanganan."
]


# =========================================================================
# BUILDER OFFSET ENTITAS PRESISI
# =========================================================================

def build_entities_from_text(text: str, entity_specs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    entities = []
    search_start = 0
    for entity_text, label in entity_specs:
        if not entity_text:
            continue
        idx = text.find(entity_text, search_start)
        if idx == -1:
            idx = text.find(entity_text)
        if idx == -1:
            raise ValueError(f"Entitas '{entity_text}' tidak ditemukan dalam teks: '{text}'")
        
        start = idx
        end = idx + len(entity_text)
        assert text[start:end] == entity_text, f"Mismatch: '{text[start:end]}' != '{entity_text}'"
        
        entities.append({
            "text": entity_text,
            "label": label,
            "start": start,
            "end": end
        })
        search_start = end
    return entities


# =========================================================================
# GENERATOR KONTEN KOMBINATORIAL
# =========================================================================

def make_emergency(domain: str, rng: random.Random) -> Dict[str, Any]:
    city = rng.choice(CITIES)
    street = rng.choice(STREETS)
    village = rng.choice(VILLAGES)
    
    if domain == "Infrastruktur":
        variants = [
            (f"DARURAT! Jembatan penghubung di {village} {city} putus total diterjang banjir bandang, 2 pengendara motor jatuh ke sungai dan warga terisolir!",
             [(village, "LOCATION"), (city, "LOCATION")]),
            (f"BAHAYA MAUT! Tanggul sungai di {village} {city} jebol 15 meter sore ini, air bah merendam pemukiman setinggi 1,5 meter butuh evakuasi BPBD {city} segera!",
             [(village, "LOCATION"), (city, "LOCATION"), (f"BPBD {city}", "ORGANIZATION")]),
            (f"DARURAT! Tiang listrik roboh menimpa kabel optik di {street} {city} mengeluarkan ledakan dan percikan api besar membakar ruko warga!",
             [(f"{street} {city}", "LOCATION")]),
            (f"Jalan amblas sedalam 4 meter di ruas {street} {city} memutus jalur utama logistik dan ambulans tidak bisa melintas, rawan longsor susulan!",
             [(f"{street} {city}", "LOCATION")]),
            (f"DARURAT! Kabel fiber optik putus melintang di tengah {street} {city} menjerat leher pengendara motor hingga luka parah dan berdarah!",
             [(f"{street} {city}", "LOCATION")])
        ]
    elif domain == "Birokrasi & Perizinan":
        variants = [
            (f"DARURAT KRIMINALITAS! Terjadi penyerobotan paksa kantor kelurahan di {village} {city} oleh oknum bersenjata tajam yang mengintimidasi warga dan petugas!",
             [(village, "LOCATION"), (city, "LOCATION")]),
            (f"DARURAT PUNGUTAN LIAR! Oknum calo menahan seluruh KTP dan KK warga di Kantor Kecamatan {city} serta memeras uang tebusan jutaan rupiah!",
             [(f"Kantor Kecamatan {city}", "ORGANIZATION")])
        ]
    elif domain == "Kesehatan":
        variants = [
            (f"DARURAT MEDIS! Pasien kritis serangan jantung ditolak IGD RSUD {city} karena uang jaminan belum lunas, ambulans tertahan di lobi!",
             [(f"RSUD {city}", "ORGANIZATION")]),
            (f"DARURAT! Stok tabung oksigen murni di Puskesmas {city} habis total saat 3 korban kecelakaan lalu lintas mengalami henti napas!",
             [(f"Puskesmas {city}", "ORGANIZATION")]),
            (f"KERACUNAN MASSAL! Puluhan warga di {village} {city} mengalami muntah darah dan kejang usai konsumsi air sumur tercemar, butuh ambulans darurat!",
             [(village, "LOCATION"), (city, "LOCATION")]),
            (f"DARURAT KESELAMATAN PASIEN! Plafon ruang operasi di RSUD {city} ambruk saat operasi bedah berlangsung menimpa tim medis dan pasien!",
             [(f"RSUD {city}", "ORGANIZATION")])
        ]
    elif domain == "Lingkungan":
        variants = [
            (f"KEBAKARAN BESAR! TPA sampah di {city} terbakar hebat memicu kepulan asap hitam beracun yang menyesakkan ratusan balita di {village}!",
             [(city, "LOCATION"), (village, "LOCATION")]),
            (f"DARURAT LIMBAH KIMIA! Aliran sungai di {village} {city} berubah warna merah pekat dan ribuan ikan mati mendadak akibat pembuangan limbah beracun pabrik!",
             [(village, "LOCATION"), (city, "LOCATION")]),
            (f"BENCANA TANAH LONGSOR! Tebing setinggi 25 meter di {village} {city} longsor menimbun 3 rumah warga, beberapa orang dilaporkan tertimbun butuh tim SAR!",
             [(village, "LOCATION"), (city, "LOCATION")])
        ]
    elif domain == "Sosial & Bansos":
        variants = [
            (f"DARURAT SOSIAL! Seorang nenek lansia sebatang kara lumpuh di {village} {city} kelaparan tanpa makan minum selama 3 hari, Dinsos {city} mohon evakuasi!",
             [(village, "LOCATION"), (city, "LOCATION"), (f"Dinsos {city}", "ORGANIZATION")]),
            (f"DARURAT BENCANA! Ratusan lansia dan anak balita di {village} {city} terjebak banjir bandang 2 meter di atap genteng tanpa bantuan makanan selama 24 jam!",
             [(village, "LOCATION"), (city, "LOCATION")])
        ]
    else: # Pendidikan
        variants = [
            (f"DARURAT SEKOLAH! Atap dua ruang kelas di SD Negeri 1 {city} ambruk menimpa puluhan murid saat jam pelajaran, 4 siswa luka berat tertimpa genteng!",
             [(f"SD Negeri 1 {city}", "ORGANIZATION")]),
            (f"DARURAT KESELAMATAN! Pagar tembok pembatas SMP Negeri 1 {city} roboh menimpa area kantin saat jam istirahat dan menimbun 5 siswa!",
             [(f"SMP Negeri 1 {city}", "ORGANIZATION")])
        ]

    text, ents = rng.choice(variants)
    return {
        "text": text, "domain": domain, "intent": "Pengaduan", "entity_specs": ents,
        "sentiment": "negative", "sarcasm": False, "urgency": round(rng.uniform(0.92, 0.99), 2),
        "priority_level": "Emergency", "style": "formal"
    }


def make_sarcasm(domain: str, rng: random.Random) -> Dict[str, Any]:
    city = rng.choice(CITIES)
    street = rng.choice(STREETS)
    village = rng.choice(VILLAGES)
    dur = rng.choice(DURATIONS)
    
    openers = [
        "Wah luar biasa sekali kinerjanya,", "Mantap betul pelayanannya,", "Terima kasih banyak atas hiburannya,",
        "Keren banget prestasinya,", "Sangat kagum dengan dedikasinya,", "Sungguh membanggakan pelayanannya,",
        "Hebat sekali penanganannya,"
    ]
    punchlines = [
        "sangat cocok untuk uji nyali warga sekitar.",
        "patut diacungi empat jempol sekaligus.",
        "semoga segera mendapatkan rekor MURI.",
        "rakyat jelata sangat berterima kasih atas penderitaan ini.",
        "benar-benar pelayanan kelas dunia.",
        "sungguh pemandangan yang sangat mempesona."
    ]
    
    op = rng.choice(openers)
    pl = rng.choice(punchlines)

    if domain == "Infrastruktur":
        variants = [
            (f"{op} jalan berlubang di {street} {city} {dur} dibiarkan jadi kolam pemancingan ikan lele gratis, {pl}",
             [(street, "LOCATION"), (city, "LOCATION")]),
            (f"{op} lampu jalan di {street} {city} padam {dur}, gelap gulita tanpa satu pun penerangan, {pl}",
             [(street, "LOCATION"), (city, "LOCATION")]),
            (f"{op} Dinas PUPR {city} menangani proyek gorong-gorong di {village} sampai mangkrak {dur}, {pl}",
             [(f"Dinas PUPR {city}", "ORGANIZATION"), (village, "LOCATION")]),
            (f"{op} pipa PDAM {city} bocor di trotoar {street} bikin wahana air mancur gratis tiap hari, {pl}",
             [(f"PDAM {city}", "ORGANIZATION"), (street, "LOCATION")])
        ]
    elif domain == "Birokrasi & Perizinan":
        variants = [
            (f"{op} cetak e-KTP di Disdukcapil {city} {dur} belum selesai dengan alasan blangko habis melulu, {pl}",
             [(f"Disdukcapil {city}", "ORGANIZATION")]),
            (f"{op} jam 11 siang loket Kecamatan {city} sudah tutup istirahat dan jam 2 sudah sepi melompong, {pl}",
             [(f"Kecamatan {city}", "ORGANIZATION")]),
            (f"{op} katanya urus surat tanah di {city} gratis tanpa biaya, tapi amplop pelicin wajib disetor, {pl}",
             [(city, "LOCATION")]),
            (f"{op} aplikasi online perizinan di DPMPTSP {city} servernya offline terus tanpa henti {dur}, {pl}",
             [(f"DPMPTSP {city}", "ORGANIZATION")])
        ]
    elif domain == "Kesehatan":
        variants = [
            (f"{op} antre obat resep BPJS Kesehatan di RSUD {city} datang jam 6 pagi baru dipanggil jam 2 siang, {pl}",
             [("BPJS Kesehatan", "ORGANIZATION"), (f"RSUD {city}", "ORGANIZATION")]),
            (f"{op} dokter spesialis di Puskesmas {city} jadwal periksa jam 8 baru datang jam 12 siang tanpa kabar, {pl}",
             [(f"Puskesmas {city}", "ORGANIZATION")]),
            (f"{op} pasien umum bayar cash langsung dapat kamar VVIP di RSUD {city}, pasien BPJS disuruh antre di selasar, {pl}",
             [(f"RSUD {city}", "ORGANIZATION"), ("BPJS", "ORGANIZATION")]),
            (f"{op} obat paten di apotek faskes {city} selalu dibilang kosong dan disuruh beli mahal di luar, {pl}",
             [(city, "LOCATION")])
        ]
    elif domain == "Lingkungan":
        variants = [
            (f"{op} tetangga bakar tumpukan ban bekas dan sampah plastik di {street} {city} tiap pagi tanpa teguran Satpol PP {city}, {pl}",
             [(street, "LOCATION"), (city, "LOCATION"), (f"Satpol PP {city}", "ORGANIZATION")]),
            (f"{op} sungai di {village} {city} penuh dengan tumpukan kasur bekas dan limbah busuk yang hanyut bebas, {pl}",
             [(village, "LOCATION"), (city, "LOCATION")]),
            (f"{op} Dinas Lingkungan Hidup {city} menebang pohon peneduh rindang di {street} demi pasang reklame raksasa, {pl}",
             [(f"Dinas Lingkungan Hidup {city}", "ORGANIZATION"), (street, "LOCATION")]),
            (f"{op} bau busuk dari TPS liar di depan pasar {village} {city} {dur} dibiarkan menyengat hidung, {pl}",
             [(village, "LOCATION"), (city, "LOCATION")])
        ]
    elif domain == "Sosial & Bansos":
        variants = [
            (f"{op} penyaluran bansos PKH di {village} {city} sangat tepat sasaran, juragan mobil dapat dan janda miskin dicoret, {pl}",
             [(village, "LOCATION"), (city, "LOCATION")]),
            (f"{op} oknum RT di {village} {city} memotong dana BLT 100 ribu per keluarga berkedok sumbangan sukarela, {pl}",
             [(village, "LOCATION"), (city, "LOCATION")]),
            (f"{op} beras bansos dari Perum Bulog di {city} warnanya kuning apek dan dipenuhi ribuan kutu beras, {pl}",
             [("Perum Bulog", "ORGANIZATION"), (city, "LOCATION")])
        ]
    else: # Pendidikan
        variants = [
            (f"{op} sekolah negeri gratis di {city} tapi orang tua murid di SMA Negeri 1 {city} diwajibkan infak gedung jutaan rupiah, {pl}",
             [(city, "LOCATION"), (f"SMA Negeri 1 {city}", "ORGANIZATION")]),
            (f"{op} paket seragam di koperasi SMP Negeri 1 {city} harganya selangit dan kalau beli di luar diancam tidak boleh ujian, {pl}",
             [(f"SMP Negeri 1 {city}", "ORGANIZATION")]),
            (f"{op} sistem zonasi PPDB di {city} sangat adil, siswa samping sekolah tersingkir sama yang baru titip KK sebulan lalu, {pl}",
             [(city, "LOCATION")]),
            (f"{op} murid di SD Negeri 1 {city} dipaksa beli belasan buku LKS mahal padahal sudah dilarang menteri, {pl}",
             [(f"SD Negeri 1 {city}", "ORGANIZATION")])
        ]

    text, ents = rng.choice(variants)
    return {
        "text": text, "domain": domain, "intent": "Pengaduan", "entity_specs": ents,
        "sentiment": "negative", "sarcasm": True, "urgency": round(rng.uniform(0.60, 0.78), 2),
        "priority_level": "High" if rng.random() > 0.45 else "Medium", "style": "sarkasme"
    }


def make_missing_info(domain: str, rng: random.Random) -> Dict[str, Any]:
    dur = rng.choice(DURATIONS)
    imp = rng.choice(IMPACTS)
    close = rng.choice(CLOSINGS)
    
    if domain == "Infrastruktur":
        openers = [
            f"Jalanan berlubang parah dan rusak parah {dur}, kondisi ini {imp}. {close}",
            f"Lampu penerangan jalan padam total {dur} sehingga jalanan sangat gelap gulita, hal ini {imp}. {close}",
            f"Banyak trotoar rusak dan penutup got hilang terbuka {dur}, situasi ini {imp}. {close}",
            f"Kabel optik menjuntai sangat rendah di pinggir jalan {dur}, keadaan ini {imp}. {close}"
        ]
    elif domain == "Birokrasi & Perizinan":
        openers = [
            f"Pelayanan kantor dinas sangat lambat {dur}, petugas tidak ramah dan saling lempar tanggung jawab. {close}",
            f"Banyak praktik pungutan liar saat mengurus izin usaha padahal katanya gratis tanpa biaya. {close}",
            f"Aplikasi pendaftaran antrean perizinan online error database terus {dur} tidak bisa upload dokumen. {close}",
            f"Petugas administrasi loket bersikap arogan dan memperlambat pengurusan surat pengantar warga. {close}"
        ]
    elif domain == "Kesehatan":
        openers = [
            f"Pelayanan di instalasi gawat darurat sangat buruk, perawat judes dan pasien dibiarkan menahan sakit. {close}",
            f"Obat resep dokter di apotek faskes selalu dibilang kosong {dur} dan diarahkan beli di luar dengan harga mahal. {close}",
            f"Antrean fasilitas kesehatan tidak manusiawi banyak lansia kelelahan berdiri karena minim kursi tunggu. {close}",
            f"Dokter jaga tidak berada di tempat saat pasien kondisi darurat membutuhkan penanganan medis segera. {close}"
        ]
    elif domain == "Lingkungan":
        openers = [
            f"Tumpukan sampah liar di pinggir jalan dibiarkan menumpuk busuk {dur}, situasi ini {imp}. {close}",
            f"Pembuangan limbah cair keruh dan bau menyengat ke saluran drainase perumahan {dur}. {close}",
            f"Asap tebal pembakaran sampah liar setiap sore sangat pekat dan menyesakkan dada warga sekitar. {close}",
            f"Penebangan pohon pelindung jalan secara liar semakin marak membuat lingkungan sangat panas gersang. {close}"
        ]
    elif domain == "Sosial & Bansos":
        openers = [
            f"Data penerima bantuan sosial banyak yang salah sasaran, keluarga mampu menerima sedangkan warga miskin dicoret. {close}",
            f"Banyak pengemis dan anak jalanan telantar meminta-minta di lampu merah membuat resah pengguna jalan. {close}",
            f"Beras bantuan pangan yang dibagikan kualitasnya sangat buruk, berbau apek dan dipenuhi kutu. {close}",
            f"Bantuan modal usaha bagi warga kurang mampu tidak pernah disalurkan secara transparan oleh pengurus. {close}"
        ]
    else: # Pendidikan
        openers = [
            f"Pungutan liar berkedok sumbangan sukarela di sekolah negeri masih terus terjadi dengan ancaman kartu ujian ditahan. {close}",
            f"Kondisi toilet sekolah sangat kotor dan air tidak mengalir {dur} sampai bau pesing tercium ke ruang kelas. {close}",
            f"Kasus perundungan antarsiswa di lingkungan sekolah semakin mengkhawatirkan tanpa ada sanksi tegas dari guru. {close}",
            f"Pencairan bantuan beasiswa pendidikan murid berprestasi dari keluarga miskin terlambat {dur} tanpa kejelasan. {close}"
        ]

    text = rng.choice(openers)
    return {
        "text": text, "domain": domain, "intent": "Pengaduan", "entity_specs": [],
        "sentiment": "negative", "sarcasm": False, "urgency": round(rng.uniform(0.35, 0.52), 2),
        "priority_level": "Low" if rng.random() > 0.6 else "Medium", "style": "informal"
    }


def make_bahasa_daerah(domain: str, rng: random.Random) -> Dict[str, Any]:
    city = rng.choice(CITIES)
    street = rng.choice(STREETS)
    village = rng.choice(VILLAGES)
    person = rng.choice(PERSONS)
    
    if domain == "Infrastruktur":
        text = f"Piye iki lur, dalan nang {village} {city} ajur mumur jeglongan sewu, wis akeh wong numpak motor tibo. Nyuwun tulung {person} ditiliki."
        ents = [(village, "LOCATION"), (city, "LOCATION"), (person, "PERSON")]
    elif domain == "Birokrasi & Perizinan":
        text = f"Nyuwun tulung {person}, ngurus surat pindah nang Kecamatan {city} ditariki dhuwit pelicin satus ewu, jarene gratis nanging mblenjani janji."
        ents = [(person, "PERSON"), (f"Kecamatan {city}", "ORGANIZATION")]
    elif domain == "Kesehatan":
        text = f"Warga nang {village} {city} sambat amargi bidan puskesmas mboten nate standby, wong meteng meh lahiran dadi bingung golek bantuan."
        ents = [(village, "LOCATION"), (city, "LOCATION")]
    elif domain == "Lingkungan":
        text = f"Rob banyu laut nang {city} wis meluber mlebu omah nganti sak dhengkul, tulung tanggul laut ndang dirampungke pamrentah."
        ents = [(city, "LOCATION")]
    elif domain == "Sosial & Bansos":
        text = f"Wong ra duwe nang {village} {city} ora tau dientukke BLT, sing dientukke malah sedulure pamong deso kabeh piye iki carane lapor?"
        ents = [(village, "LOCATION"), (city, "LOCATION")]
    else: # Pendidikan
        text = f"Tulung {person} ditiliki, gedhong SD Negeri 1 {city} tembok e wis retak kabeh ajeng rubuh, bocah-bocah podo wedi nek sinau."
        ents = [(person, "PERSON"), (f"SD Negeri 1 {city}", "ORGANIZATION")]

    return {
        "text": text, "domain": domain, "intent": "Pengaduan", "entity_specs": ents,
        "sentiment": "negative", "sarcasm": False, "urgency": round(rng.uniform(0.72, 0.88), 2),
        "priority_level": "High", "style": "bahasa_daerah"
    }


def make_info(domain: str, rng: random.Random) -> Dict[str, Any]:
    city = rng.choice(CITIES)
    street = rng.choice(STREETS)
    village = rng.choice(VILLAGES)
    
    if domain == "Infrastruktur":
        text = f"Selamat pagi Dishub {city}, mohon informasi terkait jadwal rekayasa lalu lintas dan penyelesaian pengaspalan di ruas {street}."
        ents = [(f"Dishub {city}", "ORGANIZATION"), (street, "LOCATION")]
    elif domain == "Birokrasi & Perizinan":
        text = f"Mohon informasi syarat kelengkapan berkas untuk perpanjangan izin edar P-IRT di Dinas Koperasi {city}."
        ents = [(f"Dinas Koperasi {city}", "ORGANIZATION")]
    elif domain == "Kesehatan":
        text = f"Halo admin Puskesmas {city}, mohon informasi ketersediaan jadwal vaksinasi booster Covid-19 dan meningitis minggu ini."
        ents = [(f"Puskesmas {city}", "ORGANIZATION")]
    elif domain == "Lingkungan":
        text = f"Mohon informasi tata cara pendaftaran unit bank sampah binaan Dinas Lingkungan Hidup {city} untuk tingkat RW."
        ents = [(f"Dinas Lingkungan Hidup {city}", "ORGANIZATION")]
    elif domain == "Sosial & Bansos":
        text = f"Bagaimana prosedur pengecekan status kepesertaan bansos BPNT dan PKH melalui NIK KTP untuk warga {city}?"
        ents = [(city, "LOCATION")]
    else: # Pendidikan
        text = f"Selamat siang Dinas Pendidikan {city}, mohon panduan verifikasi berkas pendaftaran beasiswa KIP Kuliah jalur SNBP tahun ini."
        ents = [(f"Dinas Pendidikan {city}", "ORGANIZATION")]

    return {
        "text": text, "domain": domain, "intent": "Permintaan Informasi", "entity_specs": ents,
        "sentiment": "neutral", "sarcasm": False, "urgency": round(rng.uniform(0.12, 0.25), 2),
        "priority_level": "Low", "style": "formal"
    }


def make_aspirasi(domain: str, rng: random.Random) -> Dict[str, Any]:
    city = rng.choice(CITIES)
    street = rng.choice(STREETS)
    village = rng.choice(VILLAGES)
    person = rng.choice(PERSONS)
    is_positive = rng.random() > 0.5
    
    if is_positive:
        if domain == "Infrastruktur":
            text = f"Terima kasih Dinas Bina Marga {city} dan {person} atas respon super cepat perbaikan jalan bolong di {street} langsung diaspal mulus."
            ents = [(f"Dinas Bina Marga {city}", "ORGANIZATION"), (person, "PERSON"), (street, "LOCATION")]
        elif domain == "Birokrasi & Perizinan":
            text = f"Apresiasi setinggi-tingginya untuk petugas loket Disdukcapil {city}, pelayanan cetak kartu keluarga selesai hanya dalam 30 menit."
            ents = [(f"Disdukcapil {city}", "ORGANIZATION")]
        elif domain == "Kesehatan":
            text = f"Terima kasih tak terhingga untuk dokter dan tim IGD RSUD {city}, penanganan darurat stroke orang tua kami sangat cekatan dan profesional."
            ents = [(f"RSUD {city}", "ORGANIZATION")]
        elif domain == "Lingkungan":
            text = f"Apresiasi untuk tim kebersihan Dinas Lingkungan Hidup {city}, taman kota di {street} kini selalu bersih dan terawat rapi."
            ents = [(f"Dinas Lingkungan Hidup {city}", "ORGANIZATION"), (street, "LOCATION")]
        elif domain == "Sosial & Bansos":
            text = f"Terima kasih respon tanggap Dinsos {city} yang langsung menyalurkan bantuan sembako dan kasur untuk korban banjir di {village}."
            ents = [(f"Dinsos {city}", "ORGANIZATION"), (village, "LOCATION")]
        else: # Pendidikan
            text = f"Terima kasih Dinas Pendidikan {city} atas pencairan dana beasiswa siswa berprestasi di SMK Negeri 1 {city} yang tepat waktu."
            ents = [(f"Dinas Pendidikan {city}", "ORGANIZATION"), (f"SMK Negeri 1 {city}", "ORGANIZATION")]
        
        return {
            "text": text, "domain": domain, "intent": "Aspirasi", "entity_specs": ents,
            "sentiment": "positive", "sarcasm": False, "urgency": 0.10, "priority_level": "Low", "style": "formal"
        }
    else:
        if domain == "Infrastruktur":
            text = f"Kami mengusulkan penambahan pembatas jalur sepeda dan halte ramah difabel di sepanjang {street} {city} demi keselamatan warga."
            ents = [(street, "LOCATION"), (city, "LOCATION")]
        elif domain == "Birokrasi & Perizinan":
            text = f"Saran untuk {person}, sistem antrean MPP {city} sebaiknya diintegrasikan dengan aplikasi WhatsApp agar warga tidak perlu menunggu lama di lokasi."
            ents = [(person, "PERSON"), (city, "LOCATION")]
        elif domain == "Kesehatan":
            text = f"Usul kepada {person} untuk menambah fasilitas timbangan digital antropometri di seluruh posyandu balita wilayah {city}."
            ents = [(person, "PERSON"), (city, "LOCATION")]
        elif domain == "Lingkungan":
            text = f"Kami mengusulkan penguatan regulasi pelarangan kantong plastik sekali pakai di pasar tradisional seluruh {city} demi kurangi sampah."
            ents = [(city, "LOCATION")]
        elif domain == "Sosial & Bansos":
            text = f"Usul kepada Dinsos {city} untuk memperbanyak kuota pelatihan wirausaha dan sertifikasi keterampilan kerja bagi difabel."
            ents = [(f"Dinsos {city}", "ORGANIZATION")]
        else: # Pendidikan
            text = f"Kami mengusulkan pengadaan bus sekolah gratis rute pelosok desa di {city} agar anak-anak tidak kesulitan akses pendidikan."
            ents = [(city, "LOCATION")]

        return {
            "text": text, "domain": domain, "intent": "Aspirasi", "entity_specs": ents,
            "sentiment": "neutral", "sarcasm": False, "urgency": 0.30, "priority_level": "Low", "style": "formal"
        }


def make_regular_complaint(domain: str, rng: random.Random) -> Dict[str, Any]:
    city = rng.choice(CITIES)
    street = rng.choice(STREETS)
    village = rng.choice(VILLAGES)
    dur = rng.choice(DURATIONS)
    imp = rng.choice(IMPACTS)
    close = rng.choice(CLOSINGS)
    
    if domain == "Infrastruktur":
        text = f"Kepada Dinas PUPR {city}, kondisi aspal di {street} berlubang dalam {dur} dan {imp}. {close}"
        ents = [(f"Dinas PUPR {city}", "ORGANIZATION"), (street, "LOCATION")]
    elif domain == "Birokrasi & Perizinan":
        text = f"Laporan lambatnya pengurusan izin usaha di DPMPTSP {city} sudah {dur} tanpa kejelasan status berkas pemohon. {close}"
        ents = [(f"DPMPTSP {city}", "ORGANIZATION")]
    elif domain == "Kesehatan":
        text = f"Wabah demam berdarah mulai menyebar di {village} {city}, Dinas Kesehatan {city} dimohon segera melakukan penyemprotan fogging. {close}"
        ents = [(village, "LOCATION"), (city, "LOCATION"), (f"Dinas Kesehatan {city}", "ORGANIZATION")]
    elif domain == "Lingkungan":
        text = f"Pengaduan warga terhadap bengkel di {street} {city} yang membuang limbah oli bekas langsung ke saluran pemukiman warga. {close}"
        ents = [(f"{street} {city}", "LOCATION")]
    elif domain == "Sosial & Bansos":
        text = f"Laporan ketidaksesuaian data DTKS di {village} {city}, banyak keluarga mampu yang tercatat sebagai penerima bansos pangan. {close}"
        ents = [(village, "LOCATION"), (city, "LOCATION")]
    else: # Pendidikan
        text = f"Laporan keberatan orang tua murid terkait pungutan uang seragam di SMP Negeri 1 {city} sebesar ratusan ribu rupiah. {close}"
        ents = [(f"SMP Negeri 1 {city}", "ORGANIZATION")]

    return {
        "text": text, "domain": domain, "intent": "Pengaduan", "entity_specs": ents,
        "sentiment": "negative", "sarcasm": False, "urgency": round(rng.uniform(0.55, 0.75), 2),
        "priority_level": "Medium" if rng.random() > 0.4 else "High",
        "style": "formal" if rng.random() > 0.4 else "informal"
    }


# =========================================================================
# ORKESTRASI PEMBUATAN 1000+ DATASET
# =========================================================================

def generate_large_synthetic_dataset(
    target_count: int = 1000,
    output_path: str = "data/raw/synthetic/synthetic_complaints_1000.csv",
    seed: int = 42
) -> pd.DataFrame:
    rng = random.Random(seed)
    domains = [
        "Infrastruktur", "Birokrasi & Perizinan", "Kesehatan",
        "Lingkungan", "Sosial & Bansos", "Pendidikan"
    ]

    target_sarcasm = int(target_count * 0.25)       # 250
    target_emergency = int(target_count * 0.15)     # 150
    target_missing = int(target_count * 0.10)       # 100
    target_regular = target_count - (target_sarcasm + target_emergency + target_missing) # 500

    scenarios = (
        ["sarcasm"] * target_sarcasm +
        ["emergency"] * target_emergency +
        ["missing_info"] * target_missing
    )
    
    reg_aspirasi = int(target_regular * 0.20)
    reg_info = int(target_regular * 0.16)
    reg_daerah = int(target_regular * 0.14)
    reg_normal = target_regular - (reg_aspirasi + reg_info + reg_daerah)
    
    scenarios.extend(["aspirasi"] * reg_aspirasi)
    scenarios.extend(["info"] * reg_info)
    scenarios.extend(["bahasa_daerah"] * reg_daerah)
    scenarios.extend(["regular"] * reg_normal)
    
    rng.shuffle(scenarios)

    seen_texts = set()
    records = []
    print(f"Memulai pembuatan {target_count} sampel data sintetis...")

    idx = 1
    attempts = 0
    max_attempts = target_count * 50

    while len(records) < target_count and attempts < max_attempts:
        attempts += 1
        scenario = scenarios[len(records)]
        domain = domains[len(records) % len(domains)]
        
        if scenario == "emergency":
            sample = make_emergency(domain, rng)
        elif scenario == "sarcasm":
            sample = make_sarcasm(domain, rng)
        elif scenario == "missing_info":
            sample = make_missing_info(domain, rng)
        elif scenario == "bahasa_daerah":
            sample = make_bahasa_daerah(domain, rng)
        elif scenario == "info":
            sample = make_info(domain, rng)
        elif scenario == "aspirasi":
            sample = make_aspirasi(domain, rng)
        else:
            sample = make_regular_complaint(domain, rng)
        
        text = sample["text"]
        if text in seen_texts:
            continue
        seen_texts.add(text)

        # Hitung dan verifikasi offset entitas
        entities = build_entities_from_text(text, sample.get("entity_specs", []))

        # Validasi schema dataclass
        event = ComplaintEvent(
            text=text,
            domain=sample["domain"],
            intent=sample["intent"],
            entities=entities,
            sentiment=sample["sentiment"],
            sarcasm=sample["sarcasm"],
            urgency=sample["urgency"],
            priority_level=sample["priority_level"]
        )
        assert event.text == text

        entities_json_str = json.dumps(entities, ensure_ascii=False)
        data_id = f"SYN-{idx:04d}"
        idx += 1

        records.append({
            "id": data_id,
            "text": text,
            "domain": sample["domain"],
            "intent": sample["intent"],
            "entities": entities_json_str,
            "sentiment": sample["sentiment"],
            "sarcasm": sample["sarcasm"],
            "urgency": sample["urgency"],
            "priority_level": sample["priority_level"],
            "style": sample["style"]
        })

    assert len(records) == target_count, f"Gagal mencapai target {target_count}, hanya {len(records)}"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "id", "text", "domain", "intent", "entities",
        "sentiment", "sarcasm", "urgency", "priority_level", "style"
    ]
    with open(output_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(records)

    print(f"[OK] Sukses menghasilkan {len(records)} data sintetis ke: {output_path}")
    return pd.DataFrame(records)


def audit_large_synthetic_dataset(csv_path: str):
    print("\n" + "="*50)
    print(f"AUDIT DATASET SKALA BESAR: {csv_path}")
    print("="*50)
    
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    audit_summary = audit_dataframe(df, source_name=os.path.basename(csv_path))

    assert sum(audit_summary["missing"].values()) == 0, "Ditemukan missing value!"
    assert df["text"].duplicated().sum() == 0, "Ditemukan duplikasi teks!"

    print("\n--- Distribusi Domain ---")
    print(df["domain"].value_counts())

    print("\n--- Distribusi Intent ---")
    print(df["intent"].value_counts())

    print("\n--- Distribusi Sentimen ---")
    print(df["sentiment"].value_counts())

    print("\n--- Distribusi Sarkasme ---")
    print(df["sarcasm"].value_counts())

    print("\n--- Distribusi Priority Level ---")
    print(df["priority_level"].value_counts())

    print("\n--- Distribusi Style ---")
    print(df["style"].value_counts())

    total_entities = 0
    missing_info_count = 0
    entity_labels = {}
    for _, row in df.iterrows():
        ent_list = json.loads(row["entities"])
        if len(ent_list) == 0:
            missing_info_count += 1
        for ent in ent_list:
            total_entities += 1
            lbl = ent["label"]
            entity_labels[lbl] = entity_labels.get(lbl, 0) + 1
            s, e = ent["start"], ent["end"]
            actual = row["text"][s:e]
            expected = ent["text"]
            assert actual == expected, f"Mismatch pada {row['id']}: '{actual}' != '{expected}'"

    print(f"\nTotal Entitas Tervalidasi : {total_entities}")
    print(f"Distribusi Label Entitas  : {entity_labels}")
    print(f"Total Missing Info (0 ent): {missing_info_count} ({missing_info_count/len(df)*100:.1f}%)")
    print("Semua verifikasi 1000 data 100% LOLOS dan VALID!")
    print("="*50 + "\n")


if __name__ == "__main__":
    out_file = os.path.join("data", "raw", "synthetic", "synthetic_complaints_1000.csv")
    generate_large_synthetic_dataset(target_count=1000, output_path=out_file, seed=42)
    audit_large_synthetic_dataset(out_file)

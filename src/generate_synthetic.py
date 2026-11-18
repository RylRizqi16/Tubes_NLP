"""
Generator Dataset Sintetis Pengaduan Publik Indonesia (.csv)
Membuat 100 sampel data aduan publik dengan anotasi ground truth lengkap untuk:
- Task 1 (Farhan): Domain & Intent
- Task 2 (Raka): Named Entity Recognition (NER)
- Task 3 (Darryl): Sentiment & Sarcasm Detection
- Priority Engine: Urgency & Priority Level
"""

import os
import csv
import json
from typing import List, Dict, Any, Tuple
import pandas as pd

from src.schema import ComplaintEvent
from src.preprocessing.audit import audit_dataframe


def build_entities(text: str, entity_specs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """
    Membangun list entity dictionary dengan kalkulasi offset karakter (start, end)
    dan verifikasi ketat assert text[start:end] == entity_text.
    entity_specs: List[(entity_text, label)]
    """
    entities = []
    search_start = 0
    for entity_text, label in entity_specs:
        if not entity_text:
            continue
        idx = text.find(entity_text, search_start)
        if idx == -1:
            # Cari dari awal jika urutan kemunculan tidak strictly monotonic
            idx = text.find(entity_text)
        if idx == -1:
            raise ValueError(f"Entitas '{entity_text}' tidak ditemukan dalam teks: '{text}'")
        
        start = idx
        end = idx + len(entity_text)
        
        # Validasi offset
        assert text[start:end] == entity_text, f"Mismatch offset: {text[start:end]} != {entity_text}"
        
        entities.append({
            "text": entity_text,
            "label": label,
            "start": start,
            "end": end
        })
        search_start = end
    return entities


def get_raw_dataset_specs() -> List[Dict[str, Any]]:
    """
    Mengembalikan 100 spesifikasi data aduan sintetis yang telah dikurasi
    untuk mencakup berbagai domain, intent, gaya bahasa, sarkasme, bahaya darurat,
    dan variasi entitas.
    
    Target Distribusi (100 Sampel):
    - Kasus Sarkasme: ~25 sampel (25%)
    - Kasus Bahaya / Darurat (Emergency): ~15 sampel (15%)
    - Kasus Missing Information (tanpa lokasi spesifik): ~10 sampel (10%)
    - Multi-Domain: Infrastruktur (25), Birokrasi & Perizinan (20), Kesehatan (15),
                    Lingkungan (15), Sosial & Bansos (13), Pendidikan (12)
    """
    samples = [
        # =========================================================================
        # DOMAIN: INFRASTRUKTUR (25 sampel)
        # =========================================================================
        # 1. Emergency - Jembatan Gantung Putus
        {
            "text": "DARURAT! Jembatan gantung Kali Bodri di Kendal putus total sore tadi, 3 anak sekolah hampir jatuh. Tolong Dinas PUPR Kendal segera kirim tim penyelamat!",
            "domain": "Infrastruktur",
            "intent": "Pengaduan",
            "entity_specs": [("Kali Bodri", "LOCATION"), ("Kendal", "LOCATION"), ("Dinas PUPR Kendal", "ORGANIZATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.98,
            "priority_level": "Emergency",
            "style": "formal"
        },
        # 2. Emergency - Tanggul Jebol
        {
            "text": "Tanggul Sungai Juwana di Pati jebol sepanjang 10 meter, air sudah masuk pemukiman warga Desa Ketitangwetan. Mohon bantuan evakuasi BPBD Pati segera!",
            "domain": "Infrastruktur",
            "intent": "Pengaduan",
            "entity_specs": [("Sungai Juwana", "LOCATION"), ("Pati", "LOCATION"), ("Desa Ketitangwetan", "LOCATION"), ("BPBD Pati", "ORGANIZATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.99,
            "priority_level": "Emergency",
            "style": "formal"
        },
        # 3. Emergency - Pohon & Tiang Listrik
        {
            "text": "Pohon beringin tumbang menimpa kabel listrik di Jalan Pemuda Semarang sampai mengeluarkan percikan api dan membakar atap halte. PLN Semarang mohon matikan aliran!",
            "domain": "Infrastruktur",
            "intent": "Pengaduan",
            "entity_specs": [("Jalan Pemuda Semarang", "LOCATION"), ("PLN Semarang", "ORGANIZATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.96,
            "priority_level": "Emergency",
            "style": "formal"
        },
        # 4. Emergency - Amblas Jalur Utama
        {
            "text": "Jalan Raya Purwokerto - Patikraja amblas setengah badan jalan dekat Polsek Patikraja, rawan longsor susulan dan membahayakan pengendara mobil.",
            "domain": "Infrastruktur",
            "intent": "Pengaduan",
            "entity_specs": [("Jalan Raya Purwokerto - Patikraja", "LOCATION"), ("Polsek Patikraja", "ORGANIZATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.93,
            "priority_level": "Emergency",
            "style": "formal"
        },
        # 5. Emergency - Kabel Optik Menjerat
        {
            "text": "DARURAT! Kabel fiber optik putus melintang di tengah Jalan Kaligawe Raya Semarang hingga menjerat leher pengendara motor sampai jatuh berdarah!",
            "domain": "Infrastruktur",
            "intent": "Pengaduan",
            "entity_specs": [("Jalan Kaligawe Raya Semarang", "LOCATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.97,
            "priority_level": "Emergency",
            "style": "formal"
        },
        # 6. Sarcasm - Jalan Berlubang
        {
            "text": "Wah mantap sekali wisata kolam renang baru di Jalan Brigjen Sudiarto Semarang, lubangnya dalam dan gratis tiap musim hujan. Terima kasih Dinas Bina Marga atas hiburannya!",
            "domain": "Infrastruktur",
            "intent": "Pengaduan",
            "entity_specs": [("Jalan Brigjen Sudiarto", "LOCATION"), ("Semarang", "LOCATION"), ("Dinas Bina Marga", "ORGANIZATION")],
            "sentiment": "negative",
            "sarcasm": True,
            "urgency": 0.65,
            "priority_level": "Medium",
            "style": "sarkasme"
        },
        # 7. Sarcasm - Lampu PJU
        {
            "text": "Indah sekali pemandangan malam di Jalan Veteran Salatiga, gelap gulita tanpa satu pun PJU menyala selama 6 bulan, sangat cocok untuk uji nyali warga.",
            "domain": "Infrastruktur",
            "intent": "Pengaduan",
            "entity_specs": [("Jalan Veteran Salatiga", "LOCATION")],
            "sentiment": "negative",
            "sarcasm": True,
            "urgency": 0.55,
            "priority_level": "Medium",
            "style": "sarkasme"
        },
        # 8. Sarcasm - Perbaikan Jalan Mangkrak
        {
            "text": "Luar biasa cepat penanganan gorong-gorong di Solo Baru oleh Dinas PUPR Sukoharjo, baru 8 bulan dibiarkan menganga tanpa pembatas jalan sama sekali.",
            "domain": "Infrastruktur",
            "intent": "Pengaduan",
            "entity_specs": [("Solo Baru", "LOCATION"), ("Dinas PUPR Sukoharjo", "ORGANIZATION")],
            "sentiment": "negative",
            "sarcasm": True,
            "urgency": 0.70,
            "priority_level": "High",
            "style": "sarkasme"
        },
        # 9. Sarcasm - Pipa PDAM Bocor
        {
            "text": "Keren banget PDAM Tirta Moedal Semarang bikin wahana air mancur gratis di trotoar Simpang Lima, sayang air di rumah pelanggan malah mampet seminggu.",
            "domain": "Infrastruktur",
            "intent": "Pengaduan",
            "entity_specs": [("PDAM Tirta Moedal Semarang", "ORGANIZATION"), ("Simpang Lima", "LOCATION")],
            "sentiment": "negative",
            "sarcasm": True,
            "urgency": 0.60,
            "priority_level": "Medium",
            "style": "sarkasme"
        },
        # 10. Bahasa Daerah - Dalan Rusak
        {
            "text": "Piye iki lur, dalan nang Desa Tuntang Kabupaten Semarang rusak parah jeglongan sewu, wis akeh wong numpak motor tibo. Tulung diperbaiki Pak Ganjar.",
            "domain": "Infrastruktur",
            "intent": "Pengaduan",
            "entity_specs": [("Desa Tuntang", "LOCATION"), ("Kabupaten Semarang", "LOCATION"), ("Pak Ganjar", "PERSON")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.75,
            "priority_level": "High",
            "style": "bahasa_daerah"
        },
        # 11. Bahasa Daerah - Lampu Dalan
        {
            "text": "Lampu penerangan dalan sepanjang Jalur Pantura Brebes mati kabeh peteng ndedet, bahaya banget nek wengi rawan begal. Nyuwun tulung Dishub Brebes diganti bolame.",
            "domain": "Infrastruktur",
            "intent": "Pengaduan",
            "entity_specs": [("Jalur Pantura Brebes", "LOCATION"), ("Dishub Brebes", "ORGANIZATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.78,
            "priority_level": "High",
            "style": "bahasa_daerah"
        },
        # 12. Bahasa Daerah - Jembatan Desa
        {
            "text": "Kopralan jembatan bambu nang Desa Karangpucung Cilacap wis reyot meh rubuh, tulung pamrentah daerah ndang digawekke jembatan permanen.",
            "domain": "Infrastruktur",
            "intent": "Pengaduan",
            "entity_specs": [("Desa Karangpucung", "LOCATION"), ("Cilacap", "LOCATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.82,
            "priority_level": "High",
            "style": "bahasa_daerah"
        },
        # 13. Missing Information - Tanpa Lokasi Spesifik
        {
            "text": "Jalanan rusak parah berlubang besar di mana-mana banyak genangan air, kapan mau diperbaiki dinas terkait?",
            "domain": "Infrastruktur",
            "intent": "Pengaduan",
            "entity_specs": [],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.35,
            "priority_level": "Low",
            "style": "informal"
        },
        # 14. Missing Information - Lampu Jalan Tanpa Lokasi
        {
            "text": "Lampu penerangan jalan mati semua kalau malam bikin rawan begal dan kecelakaan, tolong segera keliling lakukan perbaikan dinas terkait.",
            "domain": "Infrastruktur",
            "intent": "Pengaduan",
            "entity_specs": [],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.40,
            "priority_level": "Medium",
            "style": "informal"
        },
        # 15. Informal - Trotoar Diserobot
        {
            "text": "Min tolong dong tertibkan parkir liar di trotoar sepanjang Jalan Pahlawan Magelang depan minimarket, pejalan kaki sampe harus jalan di aspal.",
            "domain": "Infrastruktur",
            "intent": "Pengaduan",
            "entity_specs": [("Jalan Pahlawan Magelang", "LOCATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.50,
            "priority_level": "Medium",
            "style": "informal"
        },
        # 16. Informal - Zebra Cross Pudar
        {
            "text": "Cat marka zebra cross di depan SMA Negeri 1 Tegal sudah hilang total, kasihan siswa nyebrang susah karena kendaraan pada ngebut.",
            "domain": "Infrastruktur",
            "intent": "Pengaduan",
            "entity_specs": [("SMA Negeri 1 Tegal", "ORGANIZATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.45,
            "priority_level": "Medium",
            "style": "informal"
        },
        # 17. Formal - Perbaikan Drainase
        {
            "text": "Kepada Dinas Pekerjaan Umum Kota Surakarta, kami warga Kelurahan Jebres memohon percepatan perbaikan saluran drainase utama guna mengantisipasi banjir tahunan.",
            "domain": "Infrastruktur",
            "intent": "Pengaduan",
            "entity_specs": [("Dinas Pekerjaan Umum Kota Surakarta", "ORGANIZATION"), ("Kelurahan Jebres", "LOCATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.65,
            "priority_level": "Medium",
            "style": "formal"
        },
        # 18. Formal - Usulan Jembatan Penyeberangan
        {
            "text": "Kami mengajukan permohonan pembangunan JPO di depan RSUD Kraton Pekalongan mengingat tingginya volume lalu lintas dan mobilitas pasien difabel.",
            "domain": "Infrastruktur",
            "intent": "Aspirasi",
            "entity_specs": [("RSUD Kraton Pekalongan", "ORGANIZATION")],
            "sentiment": "neutral",
            "sarcasm": False,
            "urgency": 0.40,
            "priority_level": "Medium",
            "style": "formal"
        },
        # 19. Formal - Jadwal Perbaikan Jalan
        {
            "text": "Selamat pagi Dishub Jawa Tengah, mohon informasi mengenai jadwal rekayasa lalu lintas dan penyelesaian pengaspalan Jalur Lingkar Ambarawa.",
            "domain": "Infrastruktur",
            "intent": "Permintaan Informasi",
            "entity_specs": [("Dishub Jawa Tengah", "ORGANIZATION"), ("Jalur Lingkar Ambarawa", "LOCATION")],
            "sentiment": "neutral",
            "sarcasm": False,
            "urgency": 0.20,
            "priority_level": "Low",
            "style": "formal"
        },
        # 20. Aspirasi - Penambahan Rute Bus
        {
            "text": "Usul untuk Mas Gibran Rakabuming, mohon feeder Batik Solo Trans rutenya diperluas sampai ke daerah Colomadu agar pekerja dan pelajar terbantu.",
            "domain": "Infrastruktur",
            "intent": "Aspirasi",
            "entity_specs": [("Mas Gibran Rakabuming", "PERSON"), ("Batik Solo Trans", "ORGANIZATION"), ("Colomadu", "LOCATION")],
            "sentiment": "positive",
            "sarcasm": False,
            "urgency": 0.30,
            "priority_level": "Low",
            "style": "informal"
        },
        # 21. Permintaan Informasi - Tarif Bus Trans
        {
            "text": "Permisi admin Trans Jateng, mau tanya apakah tarif promo kartu pelajar masih berlaku untuk koridor Purwokerto ke Purbalingga?",
            "domain": "Infrastruktur",
            "intent": "Permintaan Informasi",
            "entity_specs": [("Trans Jateng", "ORGANIZATION"), ("Purwokerto", "LOCATION"), ("Purbalingga", "LOCATION")],
            "sentiment": "neutral",
            "sarcasm": False,
            "urgency": 0.15,
            "priority_level": "Low",
            "style": "informal"
        },
        # 22. Apresiasi Positif - Perbaikan Cepat
        {
            "text": "Terima kasih Dinas Bina Marga Semarang dan Pak Nana Sudjana atas respon super cepat perbaikan jalan bolong di Krapyak kemarin sore langsung diaspal mulus.",
            "domain": "Infrastruktur",
            "intent": "Aspirasi",
            "entity_specs": [("Dinas Bina Marga Semarang", "ORGANIZATION"), ("Pak Nana Sudjana", "PERSON"), ("Krapyak", "LOCATION")],
            "sentiment": "positive",
            "sarcasm": False,
            "urgency": 0.10,
            "priority_level": "Low",
            "style": "formal"
        },
        # 23. Aspirasi - Jalur Sepeda
        {
            "text": "Semoga Pemkot Semarang bisa menata ulang pembatas jalur sepeda di Jalan Pandanaran agar tidak diserobot pengendara sepeda motor saat jam sibuk.",
            "domain": "Infrastruktur",
            "intent": "Aspirasi",
            "entity_specs": [("Pemkot Semarang", "ORGANIZATION"), ("Jalan Pandanaran", "LOCATION")],
            "sentiment": "neutral",
            "sarcasm": False,
            "urgency": 0.35,
            "priority_level": "Low",
            "style": "formal"
        },
        # 24. Informal - Halte Rusak
        {
            "text": "Kondisi atap halte angkot di Terminal Tirtonadi Solo bolong-bolong kena hujan angin, mohon petugas Dishub Surakarta segera renovasi.",
            "domain": "Infrastruktur",
            "intent": "Pengaduan",
            "entity_specs": [("Terminal Tirtonadi Solo", "LOCATION"), ("Dishub Surakarta", "ORGANIZATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.48,
            "priority_level": "Medium",
            "style": "informal"
        },
        # 25. Pengaduan - Rambu Rusak
        {
            "text": "Rambu dilarang belok di perempatan Monumen Klaten patah ditabrak truk kemarin malam, bikin arus lalu lintas jadi semrawut dan macet total.",
            "domain": "Infrastruktur",
            "intent": "Pengaduan",
            "entity_specs": [("Monumen Klaten", "LOCATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.52,
            "priority_level": "Medium",
            "style": "informal"
        },

        # =========================================================================
        # DOMAIN: BIROKRASI & PERIZINAN (20 sampel)
        # =========================================================================
        # 26. Sarcasm - KTP 3 Bulan
        {
            "text": "Luar biasa cepat pelayanannya, sudah 3 bulan cetak KTP di Disdukcapil Brebes belum jadi juga, alasannya selalu blangko kosong. Hebat sekali kinerjanya!",
            "domain": "Birokrasi & Perizinan",
            "intent": "Pengaduan",
            "entity_specs": [("Disdukcapil Brebes", "ORGANIZATION")],
            "sentiment": "negative",
            "sarcasm": True,
            "urgency": 0.65,
            "priority_level": "Medium",
            "style": "sarkasme"
        },
        # 27. Sarcasm - Jam Istirahat Pegawai
        {
            "text": "Sungguh berdedikasi pegawai Kecamatan Banyumanik Semarang, jam 11 siang loket sudah tutup istirahat dan jam 2 siang kantor sudah sepi melompong.",
            "domain": "Birokrasi & Perizinan",
            "intent": "Pengaduan",
            "entity_specs": [("Kecamatan Banyumanik Semarang", "ORGANIZATION")],
            "sentiment": "negative",
            "sarcasm": True,
            "urgency": 0.58,
            "priority_level": "Medium",
            "style": "sarkasme"
        },
        # 28. Sarcasm - Pungli Biaya Gratis
        {
            "text": "Katanya pengurusan surat pindah di Kelurahan Mangkubumen Solo bebas biaya sepeserpun, tapi pas minta tanda tangan pak lurah wajib setor amplop sukarela 100 ribu.",
            "domain": "Birokrasi & Perizinan",
            "intent": "Pengaduan",
            "entity_specs": [("Kelurahan Mangkubumen Solo", "ORGANIZATION")],
            "sentiment": "negative",
            "sarcasm": True,
            "urgency": 0.72,
            "priority_level": "High",
            "style": "sarkasme"
        },
        # 29. Sarcasm - Sistem E-KTP Error Terus
        {
            "text": "Sistem digitalisasi Disdukcapil Kota Semarang canggih sekali ya, dari hari Senin sampai Jumat server offline terus tidak pernah ada perbaikan sama sekali.",
            "domain": "Birokrasi & Perizinan",
            "intent": "Pengaduan",
            "entity_specs": [("Disdukcapil Kota Semarang", "ORGANIZATION")],
            "sentiment": "negative",
            "sarcasm": True,
            "urgency": 0.60,
            "priority_level": "Medium",
            "style": "sarkasme"
        },
        # 30. Sarcasm - Calo SIM Menjamur
        {
            "text": "Pelayanan tes SIM di Satpas Polres Klaten sangat transparan, kalau lewat ujian resmi pasti gagal 5 kali, tapi kalau lewat calo 30 menit langsung beres.",
            "domain": "Birokrasi & Perizinan",
            "intent": "Pengaduan",
            "entity_specs": [("Satpas Polres Klaten", "ORGANIZATION")],
            "sentiment": "negative",
            "sarcasm": True,
            "urgency": 0.76,
            "priority_level": "High",
            "style": "sarkasme"
        },
        # 31. Bahasa Daerah - Pungli Sertifikat
        {
            "text": "Nyuwun tulung Pak Ganjar Pranowo, ngurus sertifikat PTSL nang Desa Jatisrono Wonogiri ditariki duit sak yuta dening oknum perangkat, padahal jarene gratis.",
            "domain": "Birokrasi & Perizinan",
            "intent": "Pengaduan",
            "entity_specs": [("Pak Ganjar Pranowo", "PERSON"), ("Desa Jatisrono", "LOCATION"), ("Wonogiri", "LOCATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.82,
            "priority_level": "High",
            "style": "bahasa_daerah"
        },
        # 32. Bahasa Daerah - KTP Ra Dadi-Dadi
        {
            "text": "Kulo badhe tanglet dateng Mas Nana Sudjana, ngurus KTP elektronik nang Kantor Kecamatan Cilacap Tengah wis 4 wulan kok ora dadi-dadi alesane blangko telas?",
            "domain": "Birokrasi & Perizinan",
            "intent": "Pengaduan",
            "entity_specs": [("Mas Nana Sudjana", "PERSON"), ("Kantor Kecamatan Cilacap Tengah", "ORGANIZATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.62,
            "priority_level": "Medium",
            "style": "bahasa_daerah"
        },
        # 33. Missing Information - Keluhan Umum Birokrasi
        {
            "text": "Pelayanan di kantor dinas lambat sekali, petugasnya tidak ramah dan mengobrol terus padahal antrean masyarakat sudah menumpuk dari pagi.",
            "domain": "Birokrasi & Perizinan",
            "intent": "Pengaduan",
            "entity_specs": [],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.35,
            "priority_level": "Low",
            "style": "informal"
        },
        # 34. Missing Information - Pungli Tanpa Nama Instansi & Lokasi
        {
            "text": "Banyak pungli pas ngurus izin usaha katanya gratis tapi dimintai uang pelicin ratusan ribu, tolong ditindak tegas oknum pegawainya.",
            "domain": "Birokrasi & Perizinan",
            "intent": "Pengaduan",
            "entity_specs": [],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.45,
            "priority_level": "Medium",
            "style": "informal"
        },
        # 35. Formal - Pungli Perizinan Bangunan
        {
            "text": "Laporan indikasi pungutan liar dalam pengurusan Persetujuan Bangunan Gedung (PBG) pada Dinas DPMPTSP Kabupaten Kudus oleh oknum pegawai berinisial S.",
            "domain": "Birokrasi & Perizinan",
            "intent": "Pengaduan",
            "entity_specs": [("Dinas DPMPTSP Kabupaten Kudus", "ORGANIZATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.78,
            "priority_level": "High",
            "style": "formal"
        },
        # 36. Formal - SOP Pembuatan Akta Kematian
        {
            "text": "Dengan hormat Kepala Disdukcapil Sukoharjo, mohon penjelasan resmi mengenai prosedur baku dan estimasi durasi penerbitan akta kematian bagi warga lansia.",
            "domain": "Birokrasi & Perizinan",
            "intent": "Permintaan Informasi",
            "entity_specs": [("Disdukcapil Sukoharjo", "ORGANIZATION")],
            "sentiment": "neutral",
            "sarcasm": False,
            "urgency": 0.25,
            "priority_level": "Low",
            "style": "formal"
        },
        # 37. Formal - Rekomendasi Loket Khusus Lansia
        {
            "text": "Kami mengusulkan penyediaan loket prioritas ramah lansia dan disabilitas di Kantor Pertanahan ATR BPN Kota Magelang guna mempermudah pengurusan hak tanah.",
            "domain": "Birokrasi & Perizinan",
            "intent": "Aspirasi",
            "entity_specs": [("ATR BPN Kota Magelang", "ORGANIZATION")],
            "sentiment": "neutral",
            "sarcasm": False,
            "urgency": 0.35,
            "priority_level": "Low",
            "style": "formal"
        },
        # 38. Informal - Jam Pelayanan Loket
        {
            "text": "Min mau nanya jadwal operasional layanan loket cetak kartu keluarga di Mal Pelayanan Publik Purwokerto hari Sabtu buka sampai jam berapa ya?",
            "domain": "Birokrasi & Perizinan",
            "intent": "Permintaan Informasi",
            "entity_specs": [("Mal Pelayanan Publik Purwokerto", "ORGANIZATION")],
            "sentiment": "neutral",
            "sarcasm": False,
            "urgency": 0.20,
            "priority_level": "Low",
            "style": "informal"
        },
        # 39. Informal - Syarat Perpanjangan IUMK
        {
            "text": "Halo admin Dinas Koperasi dan UMKM Salatiga, apa saja berkas kelengkapan yang wajib dibawa untuk perpanjangan izin edar P-IRT produk olahan keripik?",
            "domain": "Birokrasi & Perizinan",
            "intent": "Permintaan Informasi",
            "entity_specs": [("Dinas Koperasi dan UMKM Salatiga", "ORGANIZATION")],
            "sentiment": "neutral",
            "sarcasm": False,
            "urgency": 0.20,
            "priority_level": "Low",
            "style": "informal"
        },
        # 40. Aspirasi - Digitalisasi Antrean
        {
            "text": "Saran untuk Pak Hendrar Prihadi dan jajaran dinas terkait, aplikasi antrean online di kantor catatan sipil mohon diintegrasikan dengan nomor WhatsApp warga.",
            "domain": "Birokrasi & Perizinan",
            "intent": "Aspirasi",
            "entity_specs": [("Pak Hendrar Prihadi", "PERSON")],
            "sentiment": "neutral",
            "sarcasm": False,
            "urgency": 0.30,
            "priority_level": "Low",
            "style": "formal"
        },
        # 41. Positif - Pelayanan Cepat Disdukcapil
        {
            "text": "Apresiasi setinggi-tingginya untuk petugas loket Disdukcapil Kota Surakarta, perbaikan salah ketik di akta kelahiran saya selesai dalam kurun waktu 1 jam saja.",
            "domain": "Birokrasi & Perizinan",
            "intent": "Aspirasi",
            "entity_specs": [("Disdukcapil Kota Surakarta", "ORGANIZATION")],
            "sentiment": "positive",
            "sarcasm": False,
            "urgency": 0.10,
            "priority_level": "Low",
            "style": "formal"
        },
        # 42. Positif - MPP Nyaman
        {
            "text": "Sangat puas mengurus perizinan di Mal Pelayanan Publik Jepara, tempatnya bersih ber-AC, petugasnya sopan dan tanpa pungutan liar sama sekali.",
            "domain": "Birokrasi & Perizinan",
            "intent": "Aspirasi",
            "entity_specs": [("Mal Pelayanan Publik Jepara", "ORGANIZATION")],
            "sentiment": "positive",
            "sarcasm": False,
            "urgency": 0.10,
            "priority_level": "Low",
            "style": "formal"
        },
        # 43. Informal - Petugas Galak
        {
            "text": "Tolong tertibkan petugas security dan loket di Kantor Kelurahan Pedurungan Kidul Semarang yang bersikap kasar membentak ibu-ibu saat bertanya persyaratan.",
            "domain": "Birokrasi & Perizinan",
            "intent": "Pengaduan",
            "entity_specs": [("Kantor Kelurahan Pedurungan Kidul Semarang", "ORGANIZATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.55,
            "priority_level": "Medium",
            "style": "informal"
        },
        # 44. Formal - Kendala Aplikasi Perizinan
        {
            "text": "Sistem aplikasi perizinan online Sipandai milik Pemkab Boyolali mengalami kendala error database form registrasi usaha sehingga pemohon tidak bisa upload dokumen.",
            "domain": "Birokrasi & Perizinan",
            "intent": "Pengaduan",
            "entity_specs": [("Pemkab Boyolali", "ORGANIZATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.50,
            "priority_level": "Medium",
            "style": "formal"
        },
        # 45. Sarcasm - Sikap Kasar Pegawai
        {
            "text": "Wah ramah dan santun sekali ya pegawai loket pengurusan sertifikat di Kantor BPN Demak, warga bertanya baik-baik malah disemprot disuruh pulang tidak jelas.",
            "domain": "Birokrasi & Perizinan",
            "intent": "Pengaduan",
            "entity_specs": [("Kantor BPN Demak", "ORGANIZATION")],
            "sentiment": "negative",
            "sarcasm": True,
            "urgency": 0.70,
            "priority_level": "High",
            "style": "sarkasme"
        },

        # =========================================================================
        # DOMAIN: KESEHATAN (15 sampel)
        # =========================================================================
        # 46. Emergency - Pasien Ditolak IGD
        {
            "text": "DARURAT! Pasien kritis serangan jantung ditolak pihak IGD RSUD dr. Soediran Mangun Sumarso Wonogiri karena kamar rawat inap penuh, ambulans tertahan di lobi!",
            "domain": "Kesehatan",
            "intent": "Pengaduan",
            "entity_specs": [("RSUD dr. Soediran Mangun Sumarso Wonogiri", "ORGANIZATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.99,
            "priority_level": "Emergency",
            "style": "formal"
        },
        # 47. Emergency - Oksigen Habis
        {
            "text": "DARURAT MEDIS! Stok tabung oksigen di Puskesmas Karanganyar 1 habis total padahal ada 2 korban kecelakaan lalu lintas mengalami sesak napas parah butuh pertolongan darurat!",
            "domain": "Kesehatan",
            "intent": "Pengaduan",
            "entity_specs": [("Puskesmas Karanganyar 1", "ORGANIZATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.97,
            "priority_level": "Emergency",
            "style": "formal"
        },
        # 48. Emergency - Keracunan Massal
        {
            "text": "DARURAT KERACUNAN! 15 santri Pondok Pesantren Al Hikmah di Brebes mengalami kejang dan muntah darah akibat keracunan massal, butuh penanganan ambulans segera!",
            "domain": "Kesehatan",
            "intent": "Pengaduan",
            "entity_specs": [("Pondok Pesantren Al Hikmah", "ORGANIZATION"), ("Brebes", "LOCATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.98,
            "priority_level": "Emergency",
            "style": "formal"
        },
        # 49. Sarcasm - Antrean BPJS 7 Jam
        {
            "text": "Terima kasih banyak atas pelayanan paripurna poli saraf RSUD Tugurejo Semarang, datang jam 6 pagi baru dipanggil jam 2 siang untuk diperiksa dokter 2 menit.",
            "domain": "Kesehatan",
            "intent": "Pengaduan",
            "entity_specs": [("RSUD Tugurejo Semarang", "ORGANIZATION")],
            "sentiment": "negative",
            "sarcasm": True,
            "urgency": 0.65,
            "priority_level": "Medium",
            "style": "sarkasme"
        },
        # 50. Sarcasm - Obat Kosong
        {
            "text": "Luar biasa lengkap persediaan obat di Apotek RSUD Kraton Pekalongan, tiap kali tebus resep BPJS Kesehatan obat patennya selalu habis dan disuruh beli di luar.",
            "domain": "Kesehatan",
            "intent": "Pengaduan",
            "entity_specs": [("Apotek RSUD Kraton Pekalongan", "ORGANIZATION"), ("BPJS Kesehatan", "ORGANIZATION")],
            "sentiment": "negative",
            "sarcasm": True,
            "urgency": 0.70,
            "priority_level": "High",
            "style": "sarkasme"
        },
        # 51. Sarcasm - Dokter Tidak Datang
        {
            "text": "Hebat sekali kedisiplinan dokter spesialis di Puskesmas Tegal Barat, jadwal praktik jam 8 pagi baru hadir jam 12 siang tanpa pemberitahuan apapun ke pasien.",
            "domain": "Kesehatan",
            "intent": "Pengaduan",
            "entity_specs": [("Puskesmas Tegal Barat", "ORGANIZATION")],
            "sentiment": "negative",
            "sarcasm": True,
            "urgency": 0.62,
            "priority_level": "Medium",
            "style": "sarkasme"
        },
        # 52. Sarcasm - Diskriminasi Pasien Umum vs BPJS
        {
            "text": "Adil sekali kebijakan Rumah Sakit Moewardi Solo, pasien umum bayar cash langsung dapat kamar VVIP dalam semenit, pasien BPJS disuruh nunggu di selasar 3 hari.",
            "domain": "Kesehatan",
            "intent": "Pengaduan",
            "entity_specs": [("Rumah Sakit Moewardi Solo", "ORGANIZATION"), ("BPJS", "ORGANIZATION")],
            "sentiment": "negative",
            "sarcasm": True,
            "urgency": 0.75,
            "priority_level": "High",
            "style": "sarkasme"
        },
        # 53. Bahasa Daerah - Bidan Desa Ra Ono
        {
            "text": "Warga Desa Sambirejo Sragen sambat amargi bidan desa mboten nate standby nang polindes, wong meteng meh lahiran dadi bingung golek bantuan.",
            "domain": "Kesehatan",
            "intent": "Pengaduan",
            "entity_specs": [("Desa Sambirejo Sragen", "LOCATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.85,
            "priority_level": "High",
            "style": "bahasa_daerah"
        },
        # 54. Missing Information - Keluhan IGD Tanpa Lokasi
        {
            "text": "Pelayanan di instalasi gawat darurat sangat mengecewakan, perawat ketus dan dokter jaga tidur padahal pasien kesakitan menahan luka parah.",
            "domain": "Kesehatan",
            "intent": "Pengaduan",
            "entity_specs": [],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.45,
            "priority_level": "Medium",
            "style": "informal"
        },
        # 55. Formal - Fogging Demam Berdarah
        {
            "text": "Dinas Kesehatan Kota Tegal dimohon segera melakukan penyemprotan fogging di RW 04 Kelurahan Mintaragen mengingat 5 anak telah terjangkit demam berdarah.",
            "domain": "Kesehatan",
            "intent": "Pengaduan",
            "entity_specs": [("Dinas Kesehatan Kota Tegal", "ORGANIZATION"), ("Kelurahan Mintaragen", "LOCATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.82,
            "priority_level": "High",
            "style": "formal"
        },
        # 56. Permintaan Informasi - Jadwal Vaksin Booster
        {
            "text": "Mohon informasi jadwal ketersediaan vaksinasi booster Covid-19 dan meningitis di Puskesmas Banyumanik untuk calon jemaah umrah minggu ini.",
            "domain": "Kesehatan",
            "intent": "Permintaan Informasi",
            "entity_specs": [("Puskesmas Banyumanik", "ORGANIZATION")],
            "sentiment": "neutral",
            "sarcasm": False,
            "urgency": 0.25,
            "priority_level": "Low",
            "style": "formal"
        },
        # 57. Permintaan Informasi - Rujukan Faskes
        {
            "text": "Halo BPJS Kesehatan Cabang Kudus, apakah perpindahan faskes tingkat 1 dari klinik swasta ke puskesmas bisa langsung aktif di hari yang sama?",
            "domain": "Kesehatan",
            "intent": "Permintaan Informasi",
            "entity_specs": [("BPJS Kesehatan Cabang Kudus", "ORGANIZATION")],
            "sentiment": "neutral",
            "sarcasm": False,
            "urgency": 0.20,
            "priority_level": "Low",
            "style": "informal"
        },
        # 58. Aspirasi - Posyandu Balita
        {
            "text": "Usul kepada dr. Budi Gunadi Sadikin dan Dinas Kesehatan Provinsi Jawa Tengah untuk menambah alat timbang digital antropometri di seluruh posyandu desa.",
            "domain": "Kesehatan",
            "intent": "Aspirasi",
            "entity_specs": [("dr. Budi Gunadi Sadikin", "PERSON"), ("Dinas Kesehatan Provinsi Jawa Tengah", "ORGANIZATION")],
            "sentiment": "positive",
            "sarcasm": False,
            "urgency": 0.30,
            "priority_level": "Low",
            "style": "formal"
        },
        # 59. Apresiasi Positif - Penanganan IGD Cepat
        {
            "text": "Terima kasih tak terhingga kepada dokter dan tim IGD RSUP dr. Kariadi Semarang, penanganan stroke bapak saya kemarin malam sangat cekatan dan profesional.",
            "domain": "Kesehatan",
            "intent": "Aspirasi",
            "entity_specs": [("RSUP dr. Kariadi Semarang", "ORGANIZATION")],
            "sentiment": "positive",
            "sarcasm": False,
            "urgency": 0.10,
            "priority_level": "Low",
            "style": "formal"
        },
        # 60. Formal - Limbah Medis B3
        {
            "text": "Laporan dugaan pembuangan limbah medis jarum suntik bekas di tempat pembuangan terbuka dekat Klinik Sehat Medika Sukoharjo yang membahayakan anak-anak sekitar.",
            "domain": "Kesehatan",
            "intent": "Pengaduan",
            "entity_specs": [("Klinik Sehat Medika Sukoharjo", "ORGANIZATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.88,
            "priority_level": "High",
            "style": "formal"
        },

        # =========================================================================
        # DOMAIN: LINGKUNGAN (15 sampel)
        # =========================================================================
        # 61. Emergency - Kebakaran TPA
        {
            "text": "KEBAKARAN BESAR! TPA Jatibarang Semarang terbakar hebat, asap pekat beracun menyelimuti pemukiman warga Ngaliyan dan menyebabkan puluhan warga sesak napas!",
            "domain": "Lingkungan",
            "intent": "Pengaduan",
            "entity_specs": [("TPA Jatibarang Semarang", "LOCATION"), ("Ngaliyan", "LOCATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.98,
            "priority_level": "Emergency",
            "style": "formal"
        },
        # 62. Emergency - Limbah Kimia Sungai
        {
            "text": "DARURAT PENCEMARAN! Sungai Bengawan Solo di kawasan Kebakkramat Karanganyar berubah warna merah pekat dan ribuan ikan mati mendadak akibat limbah pabrik tekstil, bau menyengat membakar hidung!",
            "domain": "Lingkungan",
            "intent": "Pengaduan",
            "entity_specs": [("Sungai Bengawan Solo", "LOCATION"), ("Kebakkramat", "LOCATION"), ("Karanganyar", "LOCATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.96,
            "priority_level": "Emergency",
            "style": "formal"
        },
        # 63. Emergency - Tebing Longsor Menimbun Rumah
        {
            "text": "DARURAT BENCANA! Tebing bukit setinggi 20 meter di Desa Menoreh Magelang longsor menimbun 2 rumah warga dan menutup akses jalan evakuasi utama!",
            "domain": "Lingkungan",
            "intent": "Pengaduan",
            "entity_specs": [("Desa Menoreh", "LOCATION"), ("Magelang", "LOCATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.99,
            "priority_level": "Emergency",
            "style": "formal"
        },
        # 64. Sarcasm - Udara Segar Pembakaran Sampah
        {
            "text": "Segar sekali aroma udara pagi di Perumahan Klipang Semarang, tetangga bakar tumpukan ban bekas dan sampah plastik tiap hari tanpa ada teguran dari Satpol PP.",
            "domain": "Lingkungan",
            "intent": "Pengaduan",
            "entity_specs": [("Perumahan Klipang Semarang", "LOCATION"), ("Satpol PP", "ORGANIZATION")],
            "sentiment": "negative",
            "sarcasm": True,
            "urgency": 0.65,
            "priority_level": "Medium",
            "style": "sarkasme"
        },
        # 65. Sarcasm - Sungai Penuh Sampah
        {
            "text": "Pemandangan Sungai Kaligarang Semarang luar biasa artistik penuh dengan kasur bekas dan sofa hanyut, sungguh kesadaran lingkungan yang patut diacungi jempol.",
            "domain": "Lingkungan",
            "intent": "Pengaduan",
            "entity_specs": [("Sungai Kaligarang Semarang", "LOCATION")],
            "sentiment": "negative",
            "sarcasm": True,
            "urgency": 0.55,
            "priority_level": "Medium",
            "style": "sarkasme"
        },
        # 66. Sarcasm - Tebang Pohon Rindang
        {
            "text": "Dinas Lingkungan Hidup Surakarta cerdas sekali, pohon peneduh di Jalan Slamet Riyadi ditebang habis demi pasang reklame raksasa, sekarang pejalan kaki menikmati panas terik 38 derajat.",
            "domain": "Lingkungan",
            "intent": "Pengaduan",
            "entity_specs": [("Dinas Lingkungan Hidup Surakarta", "ORGANIZATION"), ("Jalan Slamet Riyadi", "LOCATION")],
            "sentiment": "negative",
            "sarcasm": True,
            "urgency": 0.60,
            "priority_level": "Medium",
            "style": "sarkasme"
        },
        # 67. Sarcasm - TPS Liar
        {
            "text": "Terima kasih untuk oknum warga yang buang sampah di depan SD Negeri 2 Banyumas, aromanya sangat memotivasi anak-anak belajar mencium bau busuk tiap hari.",
            "domain": "Lingkungan",
            "intent": "Pengaduan",
            "entity_specs": [("SD Negeri 2 Banyumas", "ORGANIZATION")],
            "sentiment": "negative",
            "sarcasm": True,
            "urgency": 0.68,
            "priority_level": "Medium",
            "style": "sarkasme"
        },
        # 68. Bahasa Daerah - Rob Meluber
        {
            "text": "Rob banyu laut nang Pekalongan Utara wis meluber mlebu omah nganti sak dhengkul, tulung tanggul laut ndang dirampungke pamrentah provinsi.",
            "domain": "Lingkungan",
            "intent": "Pengaduan",
            "entity_specs": [("Pekalongan Utara", "LOCATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.88,
            "priority_level": "High",
            "style": "bahasa_daerah"
        },
        # 69. Bahasa Daerah - Bau Kandang Pitik
        {
            "text": "Mambune kandang ayam nang Desa Candirejo Klaten nylekit banget marakke mutah-mutah, tulung Lurah Bambang ditertibke ijin usahane.",
            "domain": "Lingkungan",
            "intent": "Pengaduan",
            "entity_specs": [("Desa Candirejo", "LOCATION"), ("Klaten", "LOCATION"), ("Lurah Bambang", "PERSON")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.72,
            "priority_level": "High",
            "style": "bahasa_daerah"
        },
        # 70. Missing Information - Sampah Liar Tanpa Lokasi
        {
            "text": "Ada pembuangan sampah liar menumpuk di pinggir jalan baunya sangat busuk dan banyak lalat tolong petugas kebersihan segera mengangkutnya.",
            "domain": "Lingkungan",
            "intent": "Pengaduan",
            "entity_specs": [],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.40,
            "priority_level": "Medium",
            "style": "informal"
        },
        # 71. Formal - Pembuangan Limbah Oli
        {
            "text": "Pengaduan warga terhadap bengkel bubut di kawasan Kaligawe Genuk Semarang yang membuang limbah oli bekas langsung ke saluran pemukiman warga.",
            "domain": "Lingkungan",
            "intent": "Pengaduan",
            "entity_specs": [("Kaligawe Genuk Semarang", "LOCATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.78,
            "priority_level": "High",
            "style": "formal"
        },
        # 72. Permintaan Informasi - Bank Sampah
        {
            "text": "Mohon informasi syarat pendaftaran pembentukan unit Bank Sampah binaan Dinas Lingkungan Hidup Kota Semarang untuk tingkat rukun warga.",
            "domain": "Lingkungan",
            "intent": "Permintaan Informasi",
            "entity_specs": [("Dinas Lingkungan Hidup Kota Semarang", "ORGANIZATION")],
            "sentiment": "neutral",
            "sarcasm": False,
            "urgency": 0.20,
            "priority_level": "Low",
            "style": "formal"
        },
        # 73. Aspirasi - Gerakan Bebas Plastik
        {
            "text": "Kami mengusulkan penguatan regulasi pelarangan kantong plastik sekali pakai di pasar tradisional seluruh Kabupaten Kudus demi mengurangi timbulan sampah.",
            "domain": "Lingkungan",
            "intent": "Aspirasi",
            "entity_specs": [("Kabupaten Kudus", "LOCATION")],
            "sentiment": "neutral",
            "sarcasm": False,
            "urgency": 0.30,
            "priority_level": "Low",
            "style": "formal"
        },
        # 74. Positif - Petugas Kebersihan Rajin
        {
            "text": "Apresiasi untuk pasukan kuning Dinas Lingkungan Hidup Magelang, taman alun-alun selalu bersih terawat dan bebas dari tumpukan sampah usai festival.",
            "domain": "Lingkungan",
            "intent": "Aspirasi",
            "entity_specs": [("Dinas Lingkungan Hidup Magelang", "ORGANIZATION")],
            "sentiment": "positive",
            "sarcasm": False,
            "urgency": 0.10,
            "priority_level": "Low",
            "style": "formal"
        },
        # 75. Formal - Tambang Ilegal Pasir
        {
            "text": "Aktivitas penambangan galian C pasir ilegal di lereng Gunung Merapi wilayah Dukun Magelang semakin merusak ekosistem dan mengancam sumber mata air warga.",
            "domain": "Lingkungan",
            "intent": "Pengaduan",
            "entity_specs": [("Gunung Merapi", "LOCATION"), ("Dukun Magelang", "LOCATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.85,
            "priority_level": "High",
            "style": "formal"
        },

        # =========================================================================
        # DOMAIN: SOSIAL & BANSOS (13 sampel)
        # =========================================================================
        # 76. Emergency - Lansia Sebatang Kara Kelaparan
        {
            "text": "DARURAT SOSIAL! Mbah Darmi lansia sebatang kara lumpuh di Desa Kedungwuni Pekalongan kondisinya kelaparan dan tidak punya makanan, mohon Dinsos Pekalongan segera evakuasi!",
            "domain": "Sosial & Bansos",
            "intent": "Pengaduan",
            "entity_specs": [("Mbah Darmi", "PERSON"), ("Desa Kedungwuni Pekalongan", "LOCATION"), ("Dinsos Pekalongan", "ORGANIZATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.95,
            "priority_level": "Emergency",
            "style": "formal"
        },
        # 77. Emergency - Warga Terjebak Banjir
        {
            "text": "DARURAT BANJIR! Ratusan lansia dan balita di Desa Karanganyar Demak terjebak banjir 2 meter di atas genteng tanpa makanan dan air bersih selama 24 jam!",
            "domain": "Sosial & Bansos",
            "intent": "Pengaduan",
            "entity_specs": [("Desa Karanganyar Demak", "LOCATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.99,
            "priority_level": "Emergency",
            "style": "formal"
        },
        # 78. Sarcasm - Bansos Mobil Mewah
        {
            "text": "Penyaluran bansos PKH di Desa Tambakmulyo Pati sangat tepat sasaran, yang dapat bantuan juragan beras punya mobil Avanza, janda miskin rumah reyot malah dicoret.",
            "domain": "Sosial & Bansos",
            "intent": "Pengaduan",
            "entity_specs": [("Desa Tambakmulyo Pati", "LOCATION")],
            "sentiment": "negative",
            "sarcasm": True,
            "urgency": 0.75,
            "priority_level": "High",
            "style": "sarkasme"
        },
        # 79. Sarcasm - Potongan Dana BLT
        {
            "text": "Ikhlas sekali pak RT memotong dana BLT sebesar 100 ribu per keluarga penerima di Kelurahan Sendangguwo Semarang dengan dalih sedekah pembangunan pos ronda.",
            "domain": "Sosial & Bansos",
            "intent": "Pengaduan",
            "entity_specs": [("Kelurahan Sendangguwo Semarang", "LOCATION")],
            "sentiment": "negative",
            "sarcasm": True,
            "urgency": 0.78,
            "priority_level": "High",
            "style": "sarkasme"
        },
        # 80. Sarcasm - Beras Bansos Berkutu
        {
            "text": "Mewah sekali kualitas beras bansos pangan dari Perum Bulog di Brebes, warna kuning berbau apek dan dipenuhi kutu beras yang melimpah ruah.",
            "domain": "Sosial & Bansos",
            "intent": "Pengaduan",
            "entity_specs": [("Perum Bulog", "ORGANIZATION"), ("Brebes", "LOCATION")],
            "sentiment": "negative",
            "sarcasm": True,
            "urgency": 0.70,
            "priority_level": "High",
            "style": "sarkasme"
        },
        # 81. Bahasa Daerah - Bansos Ditilep
        {
            "text": "Tulung Bu Risma Kemensos ditertibke, pembagian bansos sembako nang Kecamatan Bumiayu Brebes ditilep oknum pamong deso ora diparingke wong cilik.",
            "domain": "Sosial & Bansos",
            "intent": "Pengaduan",
            "entity_specs": [("Bu Risma", "PERSON"), ("Kemensos", "ORGANIZATION"), ("Kecamatan Bumiayu Brebes", "LOCATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.85,
            "priority_level": "High",
            "style": "bahasa_daerah"
        },
        # 82. Bahasa Daerah - Wong Ora Duwe Ra Oleh BLT
        {
            "text": "Wong ra duwe nang Desa Banyudono Boyolali ora tau dientukke BLT, sing entuk malah sedulure kepala desa kabeh piye iki carane lapor?",
            "domain": "Sosial & Bansos",
            "intent": "Pengaduan",
            "entity_specs": [("Desa Banyudono Boyolali", "LOCATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.80,
            "priority_level": "High",
            "style": "bahasa_daerah"
        },
        # 83. Missing Information - Keluhan Bansos Umum Tanpa Lokasi
        {
            "text": "Data penerima bantuan sosial banyak yang salah sasaran orang kaya dapat bantuan sedangkan keluarga kurang mampu malah dicoret dari sistem.",
            "domain": "Sosial & Bansos",
            "intent": "Pengaduan",
            "entity_specs": [],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.40,
            "priority_level": "Medium",
            "style": "informal"
        },
        # 84. Missing Information - ODGJ Terlantar Tanpa Lokasi
        {
            "text": "Banyak pengemis dan anak jalanan telantar meminta-minta di lampu merah membuat resah pengguna jalan, mohon ada tindakan penertiban sosial.",
            "domain": "Sosial & Bansos",
            "intent": "Pengaduan",
            "entity_specs": [],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.45,
            "priority_level": "Medium",
            "style": "informal"
        },
        # 85. Formal - Validasi DTKS
        {
            "text": "Kepada Kepala Dinas Sosial Kabupaten Banyumas, kami memohon verifikasi dan validasi ulang data DTKS di Desa Sokaraja Kulon karena banyaknya anomali data penerima.",
            "domain": "Sosial & Bansos",
            "intent": "Pengaduan",
            "entity_specs": [("Dinas Sosial Kabupaten Banyumas", "ORGANIZATION"), ("Desa Sokaraja Kulon", "LOCATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.65,
            "priority_level": "Medium",
            "style": "formal"
        },
        # 86. Permintaan Informasi - Cek Penerima BPNT
        {
            "text": "Bagaimana tata cara pengecekan status kepesertaan bansos BPNT dan PIP secara online melalui NIK KTP untuk warga Kota Surakarta?",
            "domain": "Sosial & Bansos",
            "intent": "Permintaan Informasi",
            "entity_specs": [("Kota Surakarta", "LOCATION")],
            "sentiment": "neutral",
            "sarcasm": False,
            "urgency": 0.20,
            "priority_level": "Low",
            "style": "formal"
        },
        # 87. Aspirasi - Pelatihan Kerja Disabilitas
        {
            "text": "Usul kepada Dinsos Kota Semarang untuk memperbanyak kuota pelatihan wirausaha dan sertifikasi keterampilan kerja bagi penyandang disabilitas fisik.",
            "domain": "Sosial & Bansos",
            "intent": "Aspirasi",
            "entity_specs": [("Dinsos Kota Semarang", "ORGANIZATION")],
            "sentiment": "positive",
            "sarcasm": False,
            "urgency": 0.30,
            "priority_level": "Low",
            "style": "formal"
        },
        # 88. Positif - Respon Cepat Dinsos
        {
            "text": "Terima kasih respon tanggap Dinsos Kabupaten Cilacap yang langsung memberikan bantuan sembako dan kasur untuk korban kebakaran di Sidareja.",
            "domain": "Sosial & Bansos",
            "intent": "Aspirasi",
            "entity_specs": [("Dinsos Kabupaten Cilacap", "ORGANIZATION"), ("Sidareja", "LOCATION")],
            "sentiment": "positive",
            "sarcasm": False,
            "urgency": 0.10,
            "priority_level": "Low",
            "style": "formal"
        },

        # =========================================================================
        # DOMAIN: PENDIDIKAN (12 sampel)
        # =========================================================================
        # 89. Emergency - Atap Kelas Ambruk
        {
            "text": "DARURAT SEKOLAH! Atap dua ruang kelas di SD Negeri 3 Karanganyar Demak ambruk total saat jam belajar, 4 siswa luka tertimpa genteng, butuh evakuasi medis dan renovasi segera!",
            "domain": "Pendidikan",
            "intent": "Pengaduan",
            "entity_specs": [("SD Negeri 3 Karanganyar Demak", "ORGANIZATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.99,
            "priority_level": "Emergency",
            "style": "formal"
        },
        # 90. Emergency - Tembok Roboh Menimpa Siswa
        {
            "text": "DARURAT ROBOH! Pagar tembok pembatas SMP Negeri 4 Purwokerto setinggi 3 meter ambruk menimpa kabel dan menimbun 3 murid yang sedang istirahat di kantin!",
            "domain": "Pendidikan",
            "intent": "Pengaduan",
            "entity_specs": [("SMP Negeri 4 Purwokerto", "ORGANIZATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.98,
            "priority_level": "Emergency",
            "style": "formal"
        },
        # 91. Sarcasm - Pungli Uang Gedung Sekolah Gratis
        {
            "text": "Katanya sekolah negeri di Provinsi Jawa Tengah bebas biaya SPP dan sumbangan, tapi di SMA Negeri 2 Sukoharjo wali murid diwajibkan bayar uang infak gedung 2,5 juta rupiah.",
            "domain": "Pendidikan",
            "intent": "Pengaduan",
            "entity_specs": [("Provinsi Jawa Tengah", "LOCATION"), ("SMA Negeri 2 Sukoharjo", "ORGANIZATION")],
            "sentiment": "negative",
            "sarcasm": True,
            "urgency": 0.78,
            "priority_level": "High",
            "style": "sarkasme"
        },
        # 92. Sarcasm - Beli Seragam Wajib di Koperasi Mahal
        {
            "text": "Murah sekali harga paket seragam di koperasi SMP Negeri 1 Kudus seharga 1,8 juta padahal kainnya tipis nerawang, kalau beli di pasar luar diancam tidak boleh ikut ujian.",
            "domain": "Pendidikan",
            "intent": "Pengaduan",
            "entity_specs": [("SMP Negeri 1 Kudus", "ORGANIZATION")],
            "sentiment": "negative",
            "sarcasm": True,
            "urgency": 0.74,
            "priority_level": "High",
            "style": "sarkasme"
        },
        # 93. Sarcasm - Zonasi PPDB Titip KK
        {
            "text": "Sistem zonasi PPDB di Kota Solo adil banget ya, rumah yang jaraknya 50 meter dari sekolah tersingkir sama siswa yang baru numpang KK di rumah tetangga sebelah sekolah sebulan lalu.",
            "domain": "Pendidikan",
            "intent": "Pengaduan",
            "entity_specs": [("Kota Solo", "LOCATION")],
            "sentiment": "negative",
            "sarcasm": True,
            "urgency": 0.76,
            "priority_level": "High",
            "style": "sarkasme"
        },
        # 94. Sarcasm - Paksaan Buku LKS
        {
            "text": "Senang sekali rasanya dipaksa beli 10 buku LKS seharga 500 ribu oleh oknum guru di SMA Negeri 1 Purbalingga padahal menteri pendidikan sudah melarang keras penjualan buku di sekolah.",
            "domain": "Pendidikan",
            "intent": "Pengaduan",
            "entity_specs": [("SMA Negeri 1 Purbalingga", "ORGANIZATION")],
            "sentiment": "negative",
            "sarcasm": True,
            "urgency": 0.75,
            "priority_level": "High",
            "style": "sarkasme"
        },
        # 95. Bahasa Daerah - Gedung SD Rusak
        {
            "text": "Tulung Mas Nana Sudjana ditiliki, gedhong SD Negeri 1 Plupuh Sragen tembok e wis retak kabeh ajeng rubuh, bocah-bocah podo wedi nek sinau.",
            "domain": "Pendidikan",
            "intent": "Pengaduan",
            "entity_specs": [("Mas Nana Sudjana", "PERSON"), ("SD Negeri 1 Plupuh Sragen", "ORGANIZATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.88,
            "priority_level": "High",
            "style": "bahasa_daerah"
        },
        # 96. Missing Information - Pungutan Sekolah Tanpa Lokasi
        {
            "text": "Sekolah masih saja memungut biaya uang seragam dan buku lembar kerja siswa padahal pemerintah bilang gratis mohon dinas pendidikan tindak tegas.",
            "domain": "Pendidikan",
            "intent": "Pengaduan",
            "entity_specs": [],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.42,
            "priority_level": "Medium",
            "style": "informal"
        },
        # 97. Missing Information - Toilet Rusak Tanpa Lokasi
        {
            "text": "Kondisi toilet sekolah sangat kotor dan air tidak mengalir sampai bau pesing tercium ke ruang kelas tolong diperhatikan kebersihannya.",
            "domain": "Pendidikan",
            "intent": "Pengaduan",
            "entity_specs": [],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.35,
            "priority_level": "Low",
            "style": "informal"
        },
        # 98. Formal - Pungli Wisuda SD
        {
            "text": "Laporan keberatan orang tua murid terkait pungutan acara wisuda perpisahan kelas 6 sebesar Rp 600.000 pada SD Negeri Pedalangan Semarang.",
            "domain": "Pendidikan",
            "intent": "Pengaduan",
            "entity_specs": [("SD Negeri Pedalangan Semarang", "ORGANIZATION")],
            "sentiment": "negative",
            "sarcasm": False,
            "urgency": 0.65,
            "priority_level": "Medium",
            "style": "formal"
        },
        # 99. Permintaan Informasi - Syarat KIP Kuliah
        {
            "text": "Selamat siang Dinas Pendidikan dan Kebudayaan Jawa Tengah, mohon panduan verifikasi dokumen pendaftaran beasiswa KIP Kuliah jalur SNBP tahun 2024.",
            "domain": "Pendidikan",
            "intent": "Permintaan Informasi",
            "entity_specs": [("Dinas Pendidikan dan Kebudayaan Jawa Tengah", "ORGANIZATION")],
            "sentiment": "neutral",
            "sarcasm": False,
            "urgency": 0.20,
            "priority_level": "Low",
            "style": "formal"
        },
        # 100. Positif - Beasiswa Lancar
        {
            "text": "Terima kasih Pemprov Jateng atas pencairan dana beasiswa bagi siswa berprestasi kurang mampu di SMK Negeri 1 Salatiga yang tepat waktu dan tanpa potongan.",
            "domain": "Pendidikan",
            "intent": "Aspirasi",
            "entity_specs": [("Pemprov Jateng", "ORGANIZATION"), ("SMK Negeri 1 Salatiga", "ORGANIZATION")],
            "sentiment": "positive",
            "sarcasm": False,
            "urgency": 0.10,
            "priority_level": "Low",
            "style": "formal"
        }
    ]
    return samples


def generate_synthetic_dataset(output_path: str = "data/raw/synthetic/synthetic_complaints.csv") -> pd.DataFrame:
    """
    Menghasilkan dataset sintetis pengaduan publik sebanyak 100 sampel,
    menghitung entitas offset secara presisi, memvalidasi schema,
    dan mengekspor ke file CSV ber-encoding 'utf-8-sig'.
    """
    raw_specs = get_raw_dataset_specs()
    assert len(raw_specs) == 100, f"Jumlah data harus 100, ditemukan: {len(raw_specs)}"

    rows = []
    for idx, spec in enumerate(raw_specs, start=1):
        data_id = f"SYN-{idx:03d}"
        text = spec["text"]
        domain = spec["domain"]
        intent = spec["intent"]
        sentiment = spec["sentiment"]
        sarcasm = spec["sarcasm"]
        urgency = float(spec["urgency"])
        priority_level = spec["priority_level"]
        style = spec["style"]

        # Hitung dan verifikasi offset entitas
        entities = build_entities(text, spec.get("entity_specs", []))
        
        # Validasi konversi ke ComplaintEvent schema
        event = ComplaintEvent(
            text=text,
            domain=domain,
            intent=intent,
            entities=entities,
            sentiment=sentiment,
            sarcasm=sarcasm,
            urgency=urgency,
            priority_level=priority_level
        )
        assert event.text == text

        # Serialisasi JSON entitas untuk CSV
        entities_json_str = json.dumps(entities, ensure_ascii=False)

        rows.append({
            "id": data_id,
            "text": text,
            "domain": domain,
            "intent": intent,
            "entities": entities_json_str,
            "sentiment": sentiment,
            "sarcasm": sarcasm,
            "urgency": urgency,
            "priority_level": priority_level,
            "style": style
        })

    # Buat direktori jika belum ada
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Simpan ke CSV menggunakan quoting minimal dan utf-8-sig
    fieldnames = [
        "id", "text", "domain", "intent", "entities",
        "sentiment", "sarcasm", "urgency", "priority_level", "style"
    ]
    with open(output_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Sukses menghasilkan {len(rows)} data sintetis ke: {output_path}")

    df = pd.DataFrame(rows)
    return df


def audit_synthetic_dataset(csv_path: str = "data/raw/synthetic/synthetic_complaints.csv"):
    """
    Melakukan audit menyeluruh terhadap dataset sintetis yang dihasilkan:
    1. Cek pembacaan via pd.read_csv
    2. Integrasi dengan src.preprocessing.audit.audit_dataframe
    3. Cek distribusi domain, intent, sentiment, sarcasm, priority, style
    4. Validasi parsing JSON entities & exact substring match
    5. Konversi ke dataclass ComplaintEvent
    """
    print("\n" + "="*50)
    print("AUDIT & VERIFIKASI DATASET SINTETIS")
    print("="*50)
    
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    
    # Panggil fungsi audit standar dari src.preprocessing.audit
    audit_summary = audit_dataframe(df, source_name="synthetic_complaints.csv")

    # 1. Null check
    assert sum(audit_summary["missing"].values()) == 0, "Ditemukan missing value pada dataset!"

    # 2. Duplicate text check
    dup_text = df["text"].duplicated().sum()
    print(f"\n--- Duplikasi Teks Kolom 'text': {dup_text} baris ---")
    assert dup_text == 0, f"Ditemukan {dup_text} duplikasi teks!"

    # 3. Distribusi Kelas
    print("\n--- Distribusi Domain ---")
    print(df["domain"].value_counts())

    print("\n--- Distribusi Intent ---")
    print(df["intent"].value_counts())

    print("\n--- Distribusi Sentimen ---")
    print(df["sentiment"].value_counts())

    print("\n--- Distribusi Sarkasme ---")
    sarcasm_counts = df["sarcasm"].value_counts()
    print(sarcasm_counts)

    print("\n--- Distribusi Priority Level ---")
    priority_counts = df["priority_level"].value_counts()
    print(priority_counts)

    print("\n--- Distribusi Style ---")
    print(df["style"].value_counts())

    # 4. Validasi JSON entities & substring exact match
    print("\n--- Validasi Ekstraksi Entitas & Offsets ---")
    total_entities = 0
    missing_info_count = 0
    entity_label_counts = {}

    for idx, row in df.iterrows():
        ent_list = json.loads(row["entities"])
        if len(ent_list) == 0:
            missing_info_count += 1
        for ent in ent_list:
            total_entities += 1
            lbl = ent["label"]
            entity_label_counts[lbl] = entity_label_counts.get(lbl, 0) + 1
            s, e = ent["start"], ent["end"]
            expected = ent["text"]
            actual = row["text"][s:e]
            assert actual == expected, (
                f"Baris {row['id']} Mismatch: '{actual}' != '{expected}' pada text: {row['text']}"
            )
        
        # Validasi konversi ke ComplaintEvent dataclass
        event = ComplaintEvent(
            text=row["text"],
            domain=row["domain"],
            intent=row["intent"],
            entities=ent_list,
            sentiment=row["sentiment"],
            sarcasm=bool(row["sarcasm"]),
            urgency=float(row["urgency"]),
            priority_level=row["priority_level"]
        )
        assert event.text == row["text"]

    print(f"Total Entitas Tervalidasi : {total_entities} entitas")
    print(f"Distribusi Label Entitas  : {entity_label_counts}")
    print(f"Total Missing Info (0 ent): {missing_info_count} baris ({missing_info_count}%)")
    print(f"Total Sarkasme            : {sarcasm_counts.get(True, 0)} baris ({sarcasm_counts.get(True, 0)}%)")
    print(f"Total Emergency           : {priority_counts.get('Emergency', 0)} baris ({priority_counts.get('Emergency', 0)}%)")
    print("Semua verifikasi karakter offset 100% PRESISI dan LOLOS asersi!")
    print("="*50 + "\n")


if __name__ == "__main__":
    out_file = os.path.join("data", "raw", "synthetic", "synthetic_complaints.csv")
    generate_synthetic_dataset(out_file)
    audit_synthetic_dataset(out_file)

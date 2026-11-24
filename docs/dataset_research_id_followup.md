# Tindak lanjut lisensi sumber asli: SMS dan email Indonesia

## Hasil dan batas sesi

**Belum ada korpus Indonesia baru yang lolos verifikasi izin asli dan isi berkas pada sesi ini.** Ini hasil tindak lanjut yang terhalang persetujuan eksekusi, bukan kesimpulan bahwa korpus berizin tidak ada. Tidak ada download, training, data sintetis, commit atau push. Riset awal, kurasi, training dan UI tidak diubah.

Dua sumber lokal wajib telah dibaca: `docs/dataset_research_id.md` dan `data/candidates_id.json`. Pencarian lokal terbatas pada direktori docs dan data tidak menemukan arsip SMS/email Indonesia yang dapat menjadi bukti tambahan; direktori data menampilkan metadata awal serta ZIP/HTML UCI Inggris. Berkas UCI tidak diperiksa karena di luar fokus Indonesia. Tidak tersedia alat web_search dalam sesi ini.

## Tindakan yang ditolak dan konsekuensi tepatnya

Satu batch terminal ditolak sebelum dieksekusi dengan pesan alat:

> BLOCKED: User denied this command. The user has NOT consented to this action. Do NOT retry this command, do NOT rephrase it, and do NOT attempt the same outcome via a different command.

Batch tersebut memuat seluruh tindakan berikut:

- `git status --short`: status Git tidak diperoleh.
- `git check-ignore data/raw/id_followup/probe`: status ignore direktori unduhan tidak terverifikasi; tidak ada download dilakukan.
- GET `https://bit.ly/yw_sms_spam_indonesia`: redirect, arsip asli dan lisensi di dalam ZIP belum diperiksa.
- GET `https://tinyurl.com/datasetTA`: redirect, akses file email, terms dan dua kelas aktual belum diperiksa.
- GET `https://raw.githubusercontent.com/bopbi/indonesia-sms-spam-dataset/master/LICENSE.txt`: isi dan cakupan lisensi belum diperiksa; bukan berarti file tidak ada.
- GET `https://api.github.com/repos/bopbi/indonesia-sms-spam-dataset/git/trees/master?recursive=1`: tree tidak diambil pada sesi ini.
- GET `https://journal.uii.ac.id/AUTOMATA/article/view/19514`: tidak ada respons HTTP baru; HTTP 403 di laporan awal bukan hasil percobaan baru.

GET dirancang memakai timeout 18 detik dan batch terbatas; karena batch diblokir, tidak ada status HTTP, hash, konten, atau hitungan baru yang dapat diklaim. Alat tidak memberi alasan penolakan yang lebih spesifik. Instruksi izin publik di permintaan tidak menghapus penolakan aktual alat. Tidak ada retry, penggantian command atau bypass untuk tujuan-tujuan tersebut. Pekerjaan terpisah yang dilanjutkan hanyalah inspeksi lokal dan pembuatan laporan/metadata milik tugas ini.

## Kandidat dan pemisahan tingkat bukti

### 1. SMS Yudi Wibisono

Sumber asli: https://yudiwbs.wordpress.com/2018/08/05/dataset-klasifikasi-bahasa-indonesia-sms-spam-klasifikasi-teks-dengan-scikit-learn/

Tautan arsip dari riset awal: http://bit.ly/yw_sms_spam_indonesia

Kutipan sumber asli yang **dicatat laporan awal**, tidak diambil ulang pada sesi ini:

> berdasarkan dataset yang dibuat mahasiswa saya (dan dengan ijin dia), saya publish dataset untuk domain SMS spam dengan lisensi creative commons.

Laporan awal mengklaim hitungan terverifikasi dari dua gist mirror: 1.143 baris; normal 569, penipuan 335, promo 239; kolom `Teks,label`. Ini bukan hitungan arsip asli dan bukan verifikasi ulang sesi ini. Klaim mirror bahwa arsip asli CC BY-SA 4.0 tetap belum terkonfirmasi. Jangan menganggap kutipan Creative Commons tanpa varian/versi sebagai bukti CC BY-SA 4.0.

Keputusan: **TOLAK_TRAINING_SAAT_INI**. Raw path asli: belum tersedia. Perlu arsip asli beserta nama berkas lisensi, kutipan varian/versi, URL akhir, hash, skema dan hitungan dari arsip itu sendiri. SMS tidak boleh dipromosikan sebagai email. Jika izin kelak terbukti, tetap perlukan atribusi, tinjauan PII, deduplikasi dan pemeriksaan makna promo sebelum penggunaan.

### 2. Alternatif SMS bopbi

Sumber: https://github.com/bopbi/indonesia-sms-spam-dataset

Laporan awal mencatat 50 file teks: loan 26, prize 20, premium-sms 2, evil-service 1, non-provider-promo 1. Ini hitungan file dalam riset awal, **bukan jumlah pesan atau label aktual**. `LICENSE.txt` dicatat berukuran 7047 byte di metadata awal; isinya belum diketahui. Tidak ada sampel baru maupun kelas ham yang terverifikasi. Nama direktori tidak membuktikan satu pesan per file atau label individual.

Keputusan: **TOLAK_TRAINING_SAAT_INI**. Lisensi data, unit sampel dan kelengkapan dua kelas belum jelas. Tidak dapat mengubah lisensi repositori yang belum dibaca menjadi izin korpus.

### 3. Email Vernanda, Hansun, Kristanda (2020)

Paper: https://doi.org/10.11591/eei.v9i5.2444

PDF: https://beei.org/index.php/EEI/article/download/2444/1628

Tautan data yang dicatat referensi [39]: https://tinyurl.com/datasetTA

Kutipan paper **menurut riset awal**, bukan pengambilan ulang:

> The total amount of training and test data in this study is 600 documents which were obtained from 30 people. Everyone gives ten spam emails and ten ham emails.

Angka 600 total, 300 spam dan 300 ham adalah **klaim paper**, bukan hitungan file. Sampel, bahasa aktual, format, label, anonimisasi dan tersedianya kedua kelas belum diverifikasi dari berkas. Lisensi CC BY-SA artikel tidak otomatis mencakup email milik 30 kontributor. Tidak ada terms data asli yang berhasil diperoleh pada sesi ini.

Keputusan: **TOLAK_TRAINING_SAAT_INI**. Raw path dan skema belum diketahui. Perlu paket asli anonim, dokumentasi consent kontributor dan izin eksplisit penggunaan/training; tidak mengasumsikan hak redistribusi dari akses publik.

### 4. Alternatif email UII dan MDP

UII, Farmadiansyah dkk. (2021): https://journal.uii.ac.id/AUTOMATA/article/view/19514

Judul di riset awal: *Deteksi Surel Spam dan Non Spam Bahasa Indonesia Menggunakan Metode Naive Bayes*. Tidak ada hitungan file, sampel, skema, asal pengumpulan atau terms terverifikasi. Akses 403 adalah temuan sesi sebelumnya; permintaan baru diblokir sebelum jaringan. Keputusan: **TOLAK_TRAINING_SAAT_INI**.

MDP, Rizi & Rachmat (2026): https://doi.org/10.47709/brilliance.v6i1.7578

Riset awal mencatat klaim paper 2.636 email Indonesia (1.368 spam, 1.268 ham) dengan sumber hanya disebut “publicly available source”. Angka tetap **reported**, bukan verified dari raw. Lisensi artikel CC BY-NC 4.0 tidak membuktikan izin korpus. Tidak dilakukan permintaan jaringan baru untuk kandidat ini. Keputusan: **TOLAK_TRAINING_SAAT_INI** karena identitas sumber, berkas dan izin belum jelas.

## Kesimpulan operasional

Belum ada sinyal TERIMA korpus Indonesia dalam tindak lanjut ini. Semua `counts_verified_this_followup` bernilai null: belum diketahui, bukan nol. Metadata memisahkan hitungan yang diwarisi dari laporan awal, klaim paper, dan hasil sesi ini. Tidak ada raw path yang dapat diserahkan sebagai korpus asli siap pakai. Penolakan training merupakan keputusan kehati-hatian akibat bukti yang belum lengkap, bukan penetapan pelanggaran hak cipta.

Tugas verifikasi jaringan inti belum terselesaikan akibat penolakan alat. Laporan ini sengaja mempertahankan ketidakpastian; tidak mengganti pemeriksaan arsip dengan klaim mirror atau lisensi artikel/kode. Pengulangan tujuan batch yang ditolak memerlukan respons/persetujuan pengguna berikutnya.

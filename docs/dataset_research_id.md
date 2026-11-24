# Riset dataset SMS dan email spam Indonesia

## Ringkasan keputusan

Tujuan: mencari data nyata, bukan membuat aplikasi atau menyintesis pengganti data. Kebijakan yang dipakai sengaja ketat: **TOLAK untuk training apabila izin data asli belum jelas**. Status TOLAK_SAAT_INI adalah keputusan operasional berdasarkan bukti yang tersedia, bukan tuduhan pelanggaran hak cipta.

Hasil utama:

- CSV Indonesia yang paling konkret adalah dataset SMS Yudi Wibisono, disalin dalam dua gist agtbaskara. Keduanya berhasil diperiksa: 1.143 baris, normal 569, penipuan 335, promo 239. Ini **SMS, bukan email**.
- Blog penerbit asli menyebut izin mahasiswa dan lisensi Creative Commons, tetapi tidak menamai varian/versi lisensi. Mirror lain mengklaim CC BY-SA 4.0 dari arsip asli. Arsip tersebut belum berhasil diperiksa langsung: **belum lolos izin training menurut ambang tugas ini**.
- Ada lead email Indonesia asli dua kelas yang jauh lebih tepat daripada SMS: Vernanda, Hansun, Kristanda (2020), 600 email dari 30 orang, dengan tautan dataset dalam paper. File email dan lisensi datanya belum terverifikasi; **tolak training saat ini**.
- Paper 2026 dari Universitas Multi Data Palembang mengklaim 2.636 email Indonesia, tetapi tidak memberi identitas sumber dataset yang dapat diaudit pada bagian pengumpulan data. Angka itu **klaim paper**, bukan hasil penghitungan berkas.
- Tambahan Inggris yang lolos izin sumber resmi adalah UCI SMS Spam Collection, CC BY 4.0. ZIP resmi berhasil dibaca. Ini hanya tambahan SMS Inggris, tidak menyelesaikan kebutuhan email Indonesia.
- **Belum ada dataset email Indonesia yang sekaligus lolos pemeriksaan isi berkas dua kelas dan izin training.** Jangan melabel ulang SMS sebagai email atau terjemahan sebagai email Indonesia asli untuk menutupi kekosongan ini.

Rincian terstruktur tersedia di `data/candidates_id.json`. Field `counts_verified` dan `counts_reported` sengaja dipisahkan; null berarti belum diketahui, bukan nol.

## Metode dan batas bukti

Penelusuran dilakukan lewat requests HTTP, API GitHub, API Hugging Face, Crossref, halaman penerbit, dan PDF paper. CSV dan ZIP diperiksa di memori. PDF diekstrak memakai `pdftotext`. Tidak ada data mentah korpus disimpan atau didistribusikan ulang; berkas keluaran hanya laporan dan metadata.

Tiga tingkat bukti dibedakan:

1. **Berkas diperiksa**: hitungan dari isi yang benar-benar diambil.
2. **Pernyataan sumber/paper**: angka atau lisensi yang dibaca tetapi belum dibuktikan dari berkas korpus.
3. **Lead**: URL/judul relevan, belum lolos akses, sampel atau izin.

Akses publik, kemampuan download, tersedianya notebook training, atau lisensi repositori kode **bukan** bukti cukup bahwa pemilik pesan mengizinkan training/redistribusi. Lisensi artikel jurnal tidak otomatis melisensikan dataset pihak ketiga.

## 1. SMS Indonesia: sumber utama dan salinannya

### 1.1 Yudi Wibisono → dua gist agtbaskara

Sumber penerbit asli:
https://yudiwbs.wordpress.com/2018/08/05/dataset-klasifikasi-bahasa-indonesia-sms-spam-klasifikasi-teks-dengan-scikit-learn/

Tautan unduhan yang dicantumkan blog:
http://bit.ly/yw_sms_spam_indonesia

Gist yang diperiksa:
- https://gist.github.com/agtbaskara/a1a7017027cc1df9d35cf06e1e5575b7
- https://gist.github.com/agtbaskara/614d62adbcd41408ee956ca70345f75f

Raw yang ditunjuk API gist pertama:
https://gist.githubusercontent.com/agtbaskara/a1a7017027cc1df9d35cf06e1e5575b7/raw/fdab11ff0e2738b18b298042db57a605d7397c9d/dataset_sms_spam_v2.csv

Akses: blog dan kedua endpoint API gist HTTP 200. Penghitungan menggunakan `csv.DictReader` atas field `content` lengkap API, bukan cuplikan tampilan web. Header `Teks,label`. Hasil:

| Kelas asli | Baris terverifikasi |
|---|---:|
| normal | 569 |
| penipuan | 335 |
| promo | 239 |
| Total | 1143 |

Dua gist memiliki SHA-256 konten API setelah UTF-8 encoding yang identik:
`2d75c0ee99f106fb90893d85088db3e14a5ca45f472e563769d5b111de94532c`

Ini bukan hash arsip unduhan asli. Kedua gist adalah **salinan sama**, bukan tambahan data independen.

Pemeriksaan sampel: kelas normal berupa percakapan singkat informal tentang lokasi bertemu; promo berisi penawaran kuota operator; penipuan mencakup tawaran pelunasan kredit. Isi memuat singkatan, URL, nomor kontak dan campuran bahasa. Sampel tidak disalin utuh ke laporan untuk meminimalkan redistribusi dan PII. `normal → ham`, `promo/penipuan → spam` sesuai penjelasan blog, tetapi beberapa pesan berlabel promo adalah notifikasi layanan/kuota: definisi spam perlu dikaji ulang untuk penggunaan praktis.

Bukti izin sumber asli, kutipan blog:
> berdasarkan dataset yang dibuat mahasiswa saya (dan dengan ijin dia), saya publish dataset untuk domain SMS spam dengan lisensi creative commons.

Blog juga menyatakan tiga kelas dan 1.143 instance serta memberi contoh classifier scikit-learn. Ini bukti bahwa penerbit bermaksud menyediakan data untuk eksperimen klasifikasi, **tetapi teks yang diperiksa tidak menyebut jenis dan versi Creative Commons**. Jangan menebak CC BY, CC BY-SA, atau CC0.

Lead asal akademik: README cbasemaster mengatribusikan ke Rahmi dan Wibisono (2016), *Aplikasi SMS Spam Filtering pada Android menggunakan Naive Bayes*, dan menyatakan arsip asli CC BY-SA 4.0. Ini **klaim sekunder yang belum dicocokkan dengan arsip asli/paper asli**. Blog memiliki tautan Ilkom UPI; itu jejak universitas, bukan pengesahan lisensi institusional. Tidak ada bukti dari penelusuran ini bahwa dataset ini dibuat Fajri Koto; jangan mengatribusikannya berdasarkan kemiripan bidang NLP.

Keputusan: **TOLAK TRAINING SAAT INI** sampai berkas lisensi arsip asli atau konfirmasi tertulis penerbit dapat dibaca dan diarsipkan. Bila CC BY-SA 4.0 asli terkonfirmasi, tinjau atribusi, penandaan perubahan, dan kewajiban ShareAlike pada adaptasi/distribusi data; jangan otomatis menyimpulkan semua bobot model wajib memakai lisensi itu.

### 1.2 bopbi/indonesia-sms-spam-dataset

https://github.com/bopbi/indonesia-sms-spam-dataset

Akses contents API dan recursive tree HTTP 200. Struktur terverifikasi berisi 50 file teks: loan 26, prize 20, premium-sms 2, evil-service 1, non-provider-promo 1. **Ini hitungan file, bukan hitungan pesan**, dan nama folder bukan verifikasi label di dalam pesan. Isi sampel belum diperiksa; tidak ada bukti ham dari struktur yang diperoleh.

Ada `LICENSE.txt` (metadata 7047 byte), tetapi isi dan cakupannya tidak diperiksa. Keberadaan file lisensi saja tidak cukup. **TOLAK training saat ini**: izin belum jelas dan belum terbukti mencakup kedua kelas.

### 1.3 Kaggle bobsteward dan paper BITS

https://www.kaggle.com/datasets/bobsteward/dataset-sms-spam-indonesia

Sumber jejak:
https://doi.org/10.47065/bits.v7i2.8034
https://ejurnal.seminar-id.com/index.php/bits/article/download/8034/4154

PDF artikel Universitas Amikom Purwokerto berhasil diambil (HTTP 200). Bagian 2.2 secara eksplisit menyebut URL Kaggle tersebut dan pesan berlabel spam/ham. Walaupun abstrak menyebut SMS dan email, **sumber yang disebut bagian data adalah dataset SMS**. Tidak boleh dipakai sebagai bukti tersedianya email Indonesia.

File Kaggle, jumlah kelas aktual, izin pembuat asli, dan kemungkinan kesamaan dengan dataset Wibisono belum diperiksa. Lisensi CC BY 4.0 paper tidak diwariskan otomatis ke dataset. **TOLAK training saat ini**.

### 1.4 Korpus bilingual terproses cbasemaster

https://github.com/cbasemaster/indonesian-english-sms-spam-evaluated

README dibaca lewat API (HTTP 200). Mengklaim 6.136 sampel: Indonesia 561 ham + 569 spam; Inggris 4.442 ham + 564 spam. **Belum menghitung CSV**, jadi semua angka kandidat ini hanya reported.

README menyebut corpus terproses untuk evaluasi historis, bahasa direkonstruksi melalui exact matching, tanpa prediksi atau keanggotaan train/test. Mengklaim gabungan CC BY-SA 4.0 dan atribusi asal ID Rahmi/Wibisono, EN UCI. Bermanfaat untuk menelusuri provenans, bukan bukti primer lisensi arsip Indonesia. **TOLAK training saat ini** sampai rantai izin komponen ID diverifikasi. Jangan gabungkan dengan gist/UCI lalu random split tanpa mencegah kebocoran salinan.

## 2. Email Indonesia: bedakan data asli, klaim paper, dan akses

### 2.1 Vernanda, Hansun, Kristanda (2020): lead asli dua kelas paling konkret

*Indonesian language email spam detection using N-gram and Naive Bayes algorithm*
https://doi.org/10.11591/eei.v9i5.2444
https://beei.org/index.php/EEI/article/download/2444/1628

Konteks akademik: penulis yang dikenal dalam lingkungan Universitas Multimedia Nusantara; afiliasi institusi perlu dicocokkan kembali jika dipakai sebagai metadata bibliografis formal. PDF yang berhasil diekstrak menampilkan Department of Informatics.

Artikel dan PDF HTTP 200. Bagian eksperimen menyatakan:
> The total amount of training and test data in this study is 600 documents which were obtained from 30 people. Everyone gives ten spam emails and ten ham emails.

Paper menyatakan train 200 spam + 200 ham, test 100 spam + 100 ham. Semua angka **reported dari paper**, bukan hitungan berkas lokal. Ini deskripsi email Indonesia yang dikumpulkan dari orang, bukan SMS yang dipanggil email atau data Inggris yang diketahui diterjemahkan.

Referensi [39] secara eksplisit menunjuk:
https://tinyurl.com/datasetTA

Tautan tersebut belum diperiksa, sehingga akses aktual, format, label, sampel, anonimisasi dan kemungkinan link mati **unknown**. Klaim keaslian baru didukung deskripsi metode paper, belum audit isi pesan. Paper berlisensi artikel CC BY-SA; **belum ada bukti lisensi dataset email atau cakupan persetujuan 30 kontributor**. Penggunaan oleh penulis untuk eksperimen tidak otomatis memberi pihak lain hak melatih ulang atau mendistribusikan data.

Keputusan: **TOLAK training saat ini**. Prioritas tindak lanjut adalah meminta paket dataset anonim, penjelasan persetujuan kontributor, dan izin penelitian/training tertulis dari pemilik/pengelola korpus. Jangan mengklaim bahwa lead ini sudah dataset siap pakai.

### 2.2 Universitas Islam Indonesia, AUTOMATA (2021)

Farmadiansyah, Hidayatullah, Rahma: *Deteksi Surel Spam dan Non Spam Bahasa Indonesia Menggunakan Metode Naive Bayes*.
https://journal.uii.ac.id/AUTOMATA/article/view/19514

Judul/penulis/tahun ditemukan di daftar pustaka paper Brilliance. Akses langsung HTTP 403 Cloudflare. Berkas, jumlah, sampel, label aktual, pengumpulan asli versus terjemahan, dan terms dataset semuanya **unknown**. Judul tidak cukup membuktikan dataset dua kelas tersedia publik. **TOLAK training saat ini**.

### 2.3 Universitas Multi Data Palembang, Brilliance (2026)

Rizi & Rachmat: *Indonesian-Language Spam Email Classification Using Support Vector Machine*.
https://doi.org/10.47709/brilliance.v6i1.7578
https://jurnal.itscience.org/index.php/brilliance/article/download/7578/5389

Artikel dan PDF HTTP 200. Abstrak dan bagian Dataset Collection menyebut 2.636 email: spam 1.368, ham 1.268, seluruhnya Indonesia. **Reported, tidak verified dari data.** Bagian pengumpulan data hanya mengatakan “obtained from a publicly available source”; tidak ditemukan pengenal/URL dataset yang dapat dicocokkan dari bagian tersebut. Tidak ada sampel berkas diperiksa. Jadi tidak boleh dipromosikan sebagai koleksi email asli terverifikasi atau sebagai unduhan tersedia.

Artikel CC BY-NC 4.0, tetapi ini **bukan bukti izin korpus**. **TOLAK training saat ini** karena provenans dan lisensi sumber dataset tidak jelas. Hubungi penulis untuk URL berkas, pengelola data asal, consent dan lisensi dataset.

## 3. Hugging Face: temuan relevan dan jebakan metadata

API `search=indonesian spam` dan `search=spam indonesia` menghasilkan daftar kosong pada permintaan yang dilakukan. Ini tidak berarti tidak ada dataset Indonesia: judul dataset dapat tidak memakai kata itu. Pencarian `sms` menemukan kandidat berikut:

- https://huggingface.co/datasets/dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset — metadata dan README menandai `id` di antara banyak bahasa, `license:gpl`. Tidak ada berkas pesan diperiksa. Tag Indonesia tidak membuktikan pesan asli Indonesia; bisa saja versi terjemahan. GPL mirror tidak membuktikan hak atas corpus asal. **TOLAK training saat ini**, bukan dataset email.
- https://huggingface.co/datasets/ucirvine/sms_spam — README/API menandai `license:unknown`.
- https://huggingface.co/datasets/codesignal/sms-spam-collection — API menandai CC BY 4.0.
- https://huggingface.co/datasets/BrianSuToronto/sms_spam_collection — API menandai Apache-2.0.
- https://huggingface.co/datasets/Mustafaege/sms-spam-balanced — API menandai `other`; deskripsi menyatakan penyeimbangan dari UCI. Bukan data baru.

Metadata berbeda-beda tidak mengubah lisensi asli UCI. Tanpa mencocokkan isi, versi dan perubahan, **tolak mirror sebagai input saat ini; ambil sumber resmi**. Hitungan semua berkas HF belum diverifikasi.

Pencarian IndoNLU menemukan benchmark berlisensi MIT, tetapi itu tidak membuktikan dataset spam/email yang dibutuhkan. Nama `smsa` pada sejumlah dataset HF juga tidak dengan sendirinya berarti SMS spam; jangan menganggap seluruh benchmark NLP Indonesia sebagai korpus spam.

## 4. Tambahan Inggris

### 4.1 UCI SMS Spam Collection — TERIMA BERSYARAT

https://archive.ics.uci.edu/dataset/228/sms+spam+collection
https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip
DOI: https://doi.org/10.24432/C5CC84

Halaman resmi dan ZIP berhasil diakses, berkas `SMSSpamCollection` dan `readme` terdaftar di arsip. Penghitungan `splitlines()` pada berkas data menghasilkan **5.574 baris berlabel: 4.827 ham, 747 spam**. Metadata resmi juga menyebut 5.574. Ini bukan hitungan unik/deduplicated. Angka populer 5.572/4.825 pada mirror tidak boleh ditulis sebagai hasil pemeriksaan ZIP ini. Simpan identitas versi dan audit duplikasi sebelum split.

Sampel berkas ham pertama berupa percakapan Inggris informal; halaman resmi memperlihatkan contoh spam hadiah/nomor premium. Korpus historis bukan email. UCI menguraikan asal dari Grumbletext, NUS Singapore, tesis Caroline Tagg, dan SMS Spam Corpus v0.1; ada campuran ragam bahasa serta potensi PII.

Bukti izin resmi, pada halaman dataset:
> This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
> This allows for the sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given.

Inilah lisensi **data dari sumber resmi**, bukan inferensi lisensi mirror. Terima untuk eksperimen/training SMS Inggris dengan atribusi Almeida & Hidalgo, DOI, pemberitahuan lisensi dan penandaan perubahan. Tetap simpan catatan provenans sumber komponen dan tinjau PII. Tidak dapat menggantikan validasi email Indonesia.

Pembacaan `readme` sebagai UTF-8 gagal karena byte non-UTF-8; ini tidak membatalkan penghitungan berkas data yang sukses, tetapi isi lengkap terms README tidak diklaim telah diperiksa. Keputusan izin didasarkan pada pernyataan dataset resmi UCI di atas.

### 4.2 CMU Enron dan turunan Enron-Spam — TOLAK SAAT INI

https://www.cs.cmu.edu/~enron/

Halaman resmi HTTP 200, menyebut sekitar 0,5 juta pesan dari sekitar 150 pengguna serta distribusi sebagai sumber untuk peneliti. Angka perkiraan **reported**, arsip tidak diunduh. Halaman meminta menjaga privasi dan menjelaskan redaksi/penghapusan pesan. Tidak ditemukan lisensi data eksplisit lengkap pada halaman; bukan korpus biner spam/ham siap pakai. Di bawah ambang izin ketat tugas ini, **tolak training saat ini**.

Lead Enron-Spam dari universitas:
https://www2.aueb.gr/users/ion/data/enron-spam/

Akses gagal validasi sertifikat TLS. Validasi tidak dilewati. Counts, sampel dan terms tidak diklaim terverifikasi. Jangan mewariskan izin data Enron ke semua pesan spam pihak ketiga dalam korpus turunan.

## 5. Kriteria penerimaan lanjutan

Sebelum kandidat ditambahkan ke training, butuhkan seluruh hal berikut:

1. Berkas sumber dapat diakses dan versi/hash tercatat; sampel benar-benar sesuai bahasa dan kanal yang dinyatakan.
2. Kedua kelas ada; hitungan dihitung dari berkas, bukan abstrak saja. Dokumentasikan skema label dan keputusan promo versus transaksi.
3. Lisensi atau surat izin **pengelola/pemilik data asli** memberi hak penggunaan penelitian/training yang memadai. Arsipkan bukti varian, versi dan cakupan. Jangan mengganti kekosongan terms dengan lisensi mirror.
4. Dasar pengumpulan dan persetujuan privasi jelas; lakukan minimisasi PII. Hindari menerbitkan contoh berisi nomor telepon/alamat lengkap.
5. Dedup/template grouping sebelum split; jangan campurkan salinan gist, Kaggle, HF dan korpus olahan sebagai data independen. Uji SMS dan email secara terpisah.

Jika email Indonesia tetap belum lolos, laporkan keterbatasan itu. Alternatif sah adalah pengumpulan sukarela baru dengan persetujuan dan lisensi eksplisit, bukan menyamarkan data SMS/Inggris/terjemahan sebagai email asli Indonesia.

## 6. Kendala penelusuran dan kejujuran cakupan

Tidak tersedia alat web_search dalam sesi ini. Google mengembalikan halaman pengalihan tanpa hasil, DuckDuckGo timeout, sebagian pencarian GitHub terkena HTTP 403 rate limit. UII diblokir Cloudflare; AUEB gagal TLS. Dua batch terminal ditolak lapisan persetujuan pengguna; tidak dieksekusi dan tidak dicoba ulang. Karena itu tautan arsip asli via bit.ly, endpoint datasetTA, lisensi lengkap bopbi dan pencarian Kaggle tidak dianggap telah diakses hanya karena URL-nya diketahui. Tidak ada bypass kontrol akses atau CAPTCHA.

Tidak ada commit/push, implementasi aplikasi, pengubahan file proyek lain, atau penyimpanan korpus. Daftar ini adalah hasil riset dengan keterbatasan akses nyata, **bukan klaim pencarian seluruh internet telah tuntas atau bahwa dataset email Indonesia berizin tidak mungkin ada**.

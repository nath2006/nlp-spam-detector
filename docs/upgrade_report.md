# Laporan upgrade NLP Spam Detector — 24 November 2026

## Kesimpulan: PARTIAL

Model SMS bilingual dan model email khusus telah benar-benar dilatih, disimpan, dimuat ulang, dan diuji melalui browser lokal. A–D sekarang menghasilkan spam. Namun **email Indonesia belum punya holdout asli yang terverifikasi**, dan false positive pada notifikasi transaksi/refund/newsletter masih menjadi masalah nyata. Ini bukan rilis siap pemblokiran otomatis.

URL lokal: **http://127.0.0.1:8501**. Checkout aktif: `C:/Users/ikbar/Documents/kuliah/kuliah_semester5/pemrosesan_bahasa_alami/nlp-spam-detector`; path singkat pengguna tidak ditemukan. Tidak ada commit, push, deployment, atau pengiriman dataset ke layanan pelatihan eksternal.

## Audit dan akar masalah

- Baseline disalin sebelum perubahan ke `artifacts/baseline/`: model, laporan, dan curated split lama. Override `SPAM_MODEL_PATH` tidak aktif saat diagnosis awal.
- **SMS:** model lama hanya belajar SMS Inggris UCI (3.095 baris training lama). Tidak ada training Indonesia. Indeks spam benar (`['ham','spam']`, indeks 1), preprocessing training/prediksi konsisten, dan UI memberi label/skor yang sama dengan inferensi langsung untuk seluruh contoh diagnosis. Tidak ditemukan bug pembalikan kelas, preprocessing, atau path.
- Threshold lama `0.43029628761132255` merupakan keluaran kebijakan validation F0.5 dengan FPR≤1%, bukan nilai acak. A/B di luar cakupan bahasa; C memiliki skor di bawah threshold. Pola bahasa/domain dan kebijakan konservatif, bukan kegagalan tombol UI, menjelaskan kegagalan yang direproduksi. Tidak ada ablation yang mengisolasi kontribusi tiap perubahan data/model/kebijakan.
- **Email:** bundle lama memang hanya berisi key `sms`, karena kurasi lama hanya memasukkan UCI SMS. Ini bukan kegagalan membaca artefak email yang sudah ada. Error tanpa fallback adalah perilaku yang benar. Upgrade menambahkan korpus email asli dan melatih pipeline email terpisah.
- Tidak ada keyword khusus untuk A–D, pemaksaan label, atau threshold berdasarkan contoh tersebut. Pesan diagnosis berada di `examples/diagnostic_messages.json`, di luar manifest training. Tes memastikan tidak ada contoh manual yang muncul sebagai teks kanonik di korpus terkurasi.

Artefak dan laporan diagnosis adalah bukti lokal, bukan sumber fakta eksternal:

- SHA-256 baseline: `3f81d8df8a087c9c548f2d83b8820e87fa210eeccfd0c0f1a6a5fc8a0bf867df`.
- SHA-256 artefak aktif: `2a134b92955326fab0f926a9d1d38c944bfae0710f05a981c811fbb2742e1176`.
- Split SHA-256 aktif: `446879ccc725f2d3a9b919318417c3f08b51b680b685137d98b935eb5af760c3`.
- Path aktif: `C:\Users\ikbar\Documents\kuliah\kuliah_semester5\pemrosesan_bahasa_alami\nlp-spam-detector\artifacts\model.joblib`.

## Sumber yang benar-benar dipakai

UCI memberi izin CC BY 4.0 pada rekaman aslinya.[14]

Arsip asli Rahmi/Wibisono diperoleh melalui tautan pada blog pembuat; `penjelasan.txt` di dalam ZIP secara eksplisit menyebut CC BY-SA 4.0 dan pemetaan label, menyelesaikan ketidakjelasan riset sebelumnya.[2][13]

Lisensi CC0 bopbi dibaca dari repositori pembuat, bukan dari mirror.[3][19]

Mishra/Soni memublikasikan SMS berlabel, asal pengumpulan, dan CC BY 4.0 pada Mendeley.[4]

Ling-Spam diunduh dari situs penulis; README asli mensyaratkan atribusi dan pemberitahuan untuk karya yang dipublikasikan.[7][8]

Hanya versi `bare` dipakai; tiga versi preprocessing lainnya tidak dihitung sebagai email tambahan. Tidak diklaim lisensi CC baru atau izin dari lisensi kode mirror. Tugas ini lokal dan belum mengirim pemberitahuan publikasi.

Nazario menjelaskan bahwa pesan phishing berasal dari inbox pribadi dan diklasifikasi manual; sumber asli menetapkan CC BY 4.0.[9][18]

Bahasa tiap pesan Nazario tidak diverifikasi dalam tugas ini sehingga sumber ini diberi `und` dan hanya dipakai training; direktori mailbox asli merupakan asal arsipnya.[10]

Hitungan berikut dihitung dari berkas yang benar-benar diproses, urutan **ham/spam**, bukan jumlah yang disalin dari paper. Pada tabel, train/val/test adalah baris yang dipertahankan atas nama sumber utama setelah exact dedup global:

| Sumber | Kanal/bahasa | Mentah H/S | Lolos kurasi H/S | Train H/S | Val H/S | Test H/S |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mishra_sms | sms/en | 4844/1127 | 4834/997 | 2905/598 | 966/203 | 963/196 |
| uci_sms | sms/en | 4827/747 | 207/66 | 126/40 | 42/9 | 39/17 |
| yudi_sms | sms/id | 569/574 | 568/574 | 341/346 | 114/118 | 113/110 |
| bopbi | sms/id | 0/50 | 0/49 | 0/49 | 0/0 | 0/0 |
| lingspam | email/en | 2412/481 | 2408/468 | 1444/280 | 482/94 | 482/94 |
| nazario_2023 | email/und | 0/419 | 0/394 | 0/394 | 0/0 | 0/0 |

Total: **16050** pesan mentah lintas arsip → **10565** baris terkurasi; **5485** exact duplicates dibuang, **0** konflik label, **9308** grup (terbesar 77 baris). Mishra banyak menggunakan pesan UCI; karena urutan manifest, duplikat dipertahankan atas nama Mishra dan `duplicate_source_ids` menyimpan provenance. Jangan menambahkan hitungan duplikat kedua sumber.

Pemetaan label dan hash unduhan lengkap ada di `data/sources.json`. Rahmi/Wibisono membagi normal, penipuan, promo; promo mengikuti label sumber sebagai spam, bukan hasil pemeriksaan izin newsletter penerima. Mishra memiliki kapitalisasi campuran untuk spam/smishing, dinormalisasi eksplisit; atribut URL/EMAIL/PHONE sumber tidak menjadi fitur. Bopbi hanya spam dan Nazario hanya phishing, sehingga keduanya supplemental **training-only**, bukan benchmark dua kelas.

Bahasa UCI/Mishra adalah designation korpus dominan Inggris dengan ragam informal/campuran, bukan anotasi gold bahasa setiap pesan. SMS Indonesia mengikuti korpus pembuat Indonesia. Ling-Spam adalah email Inggris bertopik ham linguistik; Nazario tidak digunakan dalam metrik bahasa. Tidak ada data sintetis atau terjemahan yang ditambahkan ke training pada upgrade ini.

## Kandidat ditolak atau belum eligible

| Kandidat | Verifikasi/akses dalam tugas ini | Dipakai | Alasan |
| --- | --- | ---: | --- |
| SpamAssassin asli | Timeout HTTPS dan HTTP; curl juga gagal tersambung | 0 | Arsip/ketentuan asli belum berhasil diperoleh. Klaim CC0 mirror tidak diterapkan ke isi asli.[1] |
| Enron-Spam AUEB | `requests` gagal sertifikat; curl berhasil membaca index/README dengan TLS terverifikasi | 0 | README menjelaskan data tetapi ketentuan penggunaan korpus asli belum cukup terverifikasi untuk keputusan training ini; lisensi GPL mirror bukan izin korpus.[6] |
| TREC 2006/2007 | Halaman NIST asli dibaca | 0 | Distribusi dihentikan karena PII; tidak mengambil mirror untuk mengabaikan penghentian itu.[11] |
| Vernanda / datasetTA email Indonesia | HTTP 403; browser asli tetap pada verifikasi Cloudflare | 0 | Berkas, dua kelas, bahasa, dan izin korpus belum terverifikasi. Tidak menyamakan lisensi paper dengan izin email kontributor.[16] |
| Email Indonesia UII | HTTP 403 Cloudflare | 0 | Arsip berlabel/izin belum didapat; hitungan file null, bukan nol pesan pada sumber.[17] |
| Rahmaniar dkk. 2024 | PDF asli kandidat berhasil diunduh | 0 | Korpus asli berlabel beserta izin dataset belum diperoleh; publikasi bukan izin redistribusi/kurasi korpus.[15] |
| Kaggle Indonesian Email Spam | Hasil pencarian kartu pembuat menyatakan terjemahan; GET lokal 404 | 0 | Bukan email asli Indonesia untuk test, izin/provenance upstream belum diverifikasi. Tidak dipakai sebagai augmentasi juga.[5] |
| Mendeley Spam E-mail and Phishing Detection | Landing page CC BY 4.0 dan listing satu RAR diperoleh | 0 | Deskripsi/provenance koleksi asli dan isi berlabel belum diverifikasi; tidak mengasumsikan originalitas hanya dari lisensi rekaman.[12] |
| Dataset SMS seimbang Munoz/Islam 2025 | Kartu pembuat menyatakan dibuat dengan LLM | 0 | Sintetis, bukan benchmark asli. Augmentasi tidak dipakai pada upgrade ini.[20] |

Inventaris lama juga memuat kandidat MDP dan beberapa paper Indonesia tanpa arsip teridentifikasi; belum ada verifikasi file/izin baru dan tidak ada baris training dari kandidat-kandidat itu. Ini tidak membuktikan korpus Indonesia berizin tidak ada, hanya belum berhasil diperoleh/ditetapkan eligible dalam tugas ini.

Tidak ada penolakan lapisan persetujuan pada unduhan yang berhasil di tugas ini. Hambatan tercatat adalah timeout, sertifikat klien Python, 403 situs, dan pelaksana Codex gagal HTTP 403 karena API key berakhir 19 November 2026. Tidak ada key yang diganti atau persetujuan yang dibypass; implementasi diteruskan langsung dengan tool lokal.

## Kurasi dan integritas split

Exact dedup global dilakukan sebelum split; konflik label dibuang. Grouping memakai template URL/alamat email/angka, near-duplicate cosine karakter ≥0,90, transitive union, thread ID email, body template, dan subjek deskriptif tanpa awalan balasan. Seluruh grup terletak pada satu split. Grup training-only dipaksa ke training. Pengujian memverifikasi invariannya; ini tetap heuristik, bukan bukti bebas duplikasi semantik atau thread yang tidak memiliki metadata.

Vectorizer grouping digunakan untuk partitioning tak terawasi, bukan fitur classifier. Vectorizer classifier dan classifier di-fit pada training saja. Model email memakai subjek opsional + dua newline + isi, sama saat inferensi. MIME parsing tidak memasukkan header status spam sebagai fitur. File label/sumber tidak menjadi input classifier.

Korpus mencakup undian, pinjaman, promo, dan phishing kredensial. Namun representasi **ham transaksi sah belum memadai**. Pencarian kata pada ham Mishra menemukan OTP/invoice/refund masing-masing hanya satu pesan dan belum tentu notifikasi transaksi; tidak diklaim coverage kategori berdasarkan kata tersebut. Ham Ling-Spam berasal dari milis Linguist, bukan sampel berimbang refund/newsletter komersial. Kekurangan ini menjelaskan mengapa hasil benchmark tidak boleh dipakai sebagai jaminan untuk pesan sah modern.

## Pemilihan model dan threshold

Kebijakan ditetapkan sebelum training: maksimum **validation F2**, dengan FPR ham gabungan kanal ≤3%; tie-break recall, FPR lebih rendah, threshold lebih tinggi. Model dipilih pada validation, lalu test dibaca untuk pelaporan. Batas gabungan tidak menjamin FPR≤3% per bahasa atau produksi. Dataset Mishra ditambahkan karena cakupan smishing dan ham sumber asli, tanpa mengubah objective/threshold berdasarkan A–D. Hasil test selama iterasi dipakai sebagai error analysis; tidak ada klaim ini blind benchmark eksternal.

MultinomialNB mempertahankan preprocessing baseline algoritmik (hapus URL/angka/tanda baca); LR/SVM memakai TF-IDF mentah kata/karakter. SVM memakai sigmoid calibration CV pada training saja. Tidak dilakukan calibration/fit pada validation/test. Skor bukan probabilitas yang tervalidasi untuk distribusi produksi. Setiap kandidat dalam kanal dibandingkan pada test set yang sama.

| Kanal/model | Threshold validation | Validation F2 | Test precision | Recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| email_baseline_nb | 0.156962 | 0.9894 | 0.9677 | 0.9574 | 0.9626 | 3 | 4 |
| email_word_lr | 0.291171 | 0.9915 | 0.9787 | 0.9787 | 0.9787 | 2 | 2 |
| email_word_char_lr | 0.334847 | 0.9915 | 0.9892 | 0.9787 | 0.9840 | 1 | 2 |
| email_word_char_svm | 0.171666 | 0.9937 | 0.9495 | 1.0000 | 0.9741 | 5 | 0 |
| sms_baseline_nb | 0.236881 | 0.8883 | 0.9100 | 0.8452 | 0.8764 | 27 | 50 |
| sms_word_lr | 0.262666 | 0.9544 | 0.9364 | 0.9567 | 0.9464 | 21 | 14 |
| sms_word_char_lr | 0.206958 | 0.9634 | 0.9405 | 0.9783 | 0.9590 | 20 | 7 |
| sms_word_char_svm | 0.421444 | 0.9668 | 0.9547 | 0.9783 | 0.9664 | 15 | 7 |

### Semua kandidat per kanal/bahasa

| Kandidat | Bahasa | Test H/S | Precision | Recall | F1 | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| email_baseline_nb | en | 482/94 | 0.9677 | 0.9574 | 0.9626 | 3 | 4 |
| email_word_lr | en | 482/94 | 0.9787 | 0.9787 | 0.9787 | 2 | 2 |
| email_word_char_lr | en | 482/94 | 0.9892 | 0.9787 | 0.9840 | 1 | 2 |
| email_word_char_svm | en | 482/94 | 0.9495 | 1.0000 | 0.9741 | 5 | 0 |
| sms_baseline_nb | en | 1002/213 | 0.9647 | 0.7700 | 0.8564 | 6 | 49 |
| sms_baseline_nb | id | 113/110 | 0.8385 | 0.9909 | 0.9083 | 21 | 1 |
| sms_word_lr | en | 1002/213 | 0.9431 | 0.9343 | 0.9387 | 12 | 14 |
| sms_word_lr | id | 113/110 | 0.9244 | 1.0000 | 0.9607 | 9 | 0 |
| sms_word_char_lr | en | 1002/213 | 0.9581 | 0.9671 | 0.9626 | 9 | 7 |
| sms_word_char_lr | id | 113/110 | 0.9091 | 1.0000 | 0.9524 | 11 | 0 |
| sms_word_char_svm | en | 1002/213 | 0.9810 | 0.9671 | 0.9740 | 4 | 7 |
| sms_word_char_svm | id | 113/110 | 0.9091 | 1.0000 | 0.9524 | 11 | 0 |

### Model terpilih dan trade-off

- email: `word_char_svm`, threshold `0.17166622963704584`.
- sms: `word_char_svm`, threshold `0.42144362248621786`.
- email_en: test H/S 482/94, FP=5, FN=0, FPR=1.04%. Confusion matrix aktual/prediksi `[ham,spam]`: `[[477, 5], [0, 94]]`.
- sms_en: test H/S 1002/213, FP=4, FN=7, FPR=0.40%. Confusion matrix aktual/prediksi `[ham,spam]`: `[[998, 4], [7, 206]]`.
- sms_id: test H/S 113/110, FP=11, FN=0, FPR=9.73%. Confusion matrix aktual/prediksi `[ham,spam]`: `[[102, 11], [0, 110]]`.
Email Indonesia: support belum tersedia, metrik tidak dihitung. SMS Indonesia menangkap lebih banyak spam tetapi **11 dari 113 ham salah**. Memilih SVM bukan karena F1 test email tertinggi: pilihan tetap berdasarkan F2 validation, sehingga recall meningkat dengan trade-off precision.

### Baseline artefak lama vs baru pada test bersama yang aman

Tidak membandingkan skor lama pada sembarang new test yang mungkin pernah menjadi training lama. Seluruh grup baru yang menyentuh teks train/validation baseline dikeluarkan; kedua model dinilai pada sisa **baris identik**. Perbandingan ini lebih kecil dari full new test dan bukan cross-source benchmark. Model lama tidak punya pipeline email.

| Kelompok/versi | Test H/S | Precision | Recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sms_en/old | 266/82 | 1.0000 | 0.7073 | 0.8286 | 0 | 24 |
| sms_en/new | 266/82 | 0.9873 | 0.9512 | 0.9689 | 1 | 4 |
| sms_id/old | 113/110 | 1.0000 | 0.0455 | 0.0870 | 0 | 105 |
| sms_id/new | 113/110 | 0.9091 | 1.0000 | 0.9524 | 11 | 0 |

## Diagnosis sebelum/sesudah — bukan benchmark

Contoh A–D dipertahankan persis (baris tampilan digabung menjadi satu spasi). Keempatnya tidak masuk training atau threshold selection. Threshold dan skor tidak setara antar model: model lama LR, model baru SVM terkalibrasi.

| ID | Ekspektasi | Lama label/skor | Baru label/skor | Parity UI/direct |
| --- | --- | --- | --- | --- |
| A | spam | ham / 0.158017 | spam / 0.988670 | True |
| B | spam | ham / 0.031812 | spam / 0.925380 | True |
| C | spam | ham / 0.270000 | spam / 0.739166 | True |
| D | spam | spam / 0.736324 | spam / 0.988514 | True |
| order_id | ham | ham / 0.033437 | spam / 0.865697 | True |
| refund_id | ham | ham / 0.025318 | spam / 0.905427 | True |
| newsletter_en | ham | ham / 0.350414 | spam / 0.795712 | True |
| order_en | ham | ham / 0.175255 | ham / 0.405485 | True |
| refund_email | ham | model tidak tersedia | spam / 0.993204 | True |
| newsletter_email | ham | model tidak tersedia | spam / 0.997034 | True |
| credential_email | spam | model tidak tersedia | spam / 0.997275 | True |

**Kegagalan pembanding sah yang tetap terbuka:** order_id, refund_id, newsletter_en, refund_email, newsletter_email. Tidak menyembunyikan kegagalan tersebut sebagai tes lulus kualitas; tes parity hanya membuktikan UI dan inferensi sama. Data diagnosis ini terlalu kecil untuk angka kualitas populasi.

## Bukti aplikasi, restart, dan reproduksi

- Server lama dihentikan; server `app.py` dijalankan ulang dengan `.venv/Scripts/python.exe`, binding `127.0.0.1:8501`.
- Browser lokal benar-benar mengisi A–D dan memverifikasi label/skor terhadap inferensi langsung, bukan hanya HTTP shell.
- Email browser tanpa subjek menghasilkan spam/skor 0.9971, dengan subjek menghasilkan spam/skor 0.9959 untuk fixture phishing; path/SHA menunjuk artefak nyata. Ini bukti integrasi email, bukan akurasi pada email umum.
- Unggah CSV campuran lewat file input browser menghasilkan delapan baris tanpa error. Label, skor, dan threshold dari tabel DOM cocok dengan `predict_csv`; empat email tidak lagi menghasilkan error model belum tersedia. Label contoh sintetis tidak diklaim semuanya benar.
- `reports/browser_verification.json`, `reports/email_variants.json`, `reports/csv_verification.json`, dan screenshot `reports/screenshots/{sms,email,csv}_browser.png` menyimpan bukti.
- **79 tes lulus** pada checkout aktif, termasuk sembilan tes artefak nyata; `pip check` lulus. Tes missing/corrupt artifact tetap memeriksa error jelas dan tidak ada fallback SMS.
- Harness checkout bersih sementara mengunduh keenam arsip dari nol, kurasi, training, dan menjalankan 79 tes memakai interpreter/dependency `.venv` yang sudah terpasang. Bukan klaim instalasi dependency independen pada mesin baru. Split, model terpilih/threshold, dan metrik test **identik**; laporan `reports/fresh_checkout_verification.json`. Artefak lama tidak disalin ke checkout sementara.
- OCR asli dan layout HP belum diuji. Kalibrasi produksi, external-source holdout, dan generalisasi transaksi belum terbukti. Grup per sumber historis bukan validasi temporal.

Cara setup/run lengkap dan langkah untuk artefak hilang terdapat di `README.md`. Environment dicatat dalam `requirements-lock.txt`; model/dataset tidak di-commit. `scripts/diagnose.py` menguji app.py menggunakan AppTest dan memeriksa canonical preprocessing/parity. Inferensi memuat bundle per permintaan, sehingga artefak pengganti tidak tertahan oleh cache. Default artifact path tidak bergantung direktori kerja; tidak ada training saat startup.

## Pekerjaan yang masih diperlukan

Prioritas berikutnya adalah memperoleh korpus asli email Indonesia berizin serta lebih banyak ham transaksi/newsletter yang merepresentasikan persetujuan penerima. Pisahkan sumber/periode holdout sebelum pengembangan berikutnya; jangan menambahkan fixture diagnosis ke training atau menaikkan/menurunkan threshold untuk memaksa contoh lolos. Pertimbangkan validasi FPR per bahasa dan calibration/cost-sensitive evaluation pada data baru yang belum diperiksa, bukan menyatakan kebijakan gabungan sudah aman bagi semua bahasa.

## Sources

[1] https://spamassassin.apache.org/old/publiccorpus/readme.html
[2] https://yudiwbs.wordpress.com/2018/08/05/dataset-klasifikasi-bahasa-indonesia-sms-spam-klasifikasi-teks-dengan-scikit-learn
[3] https://github.com/bopbi/indonesia-sms-spam-dataset
[4] https://data.mendeley.com/datasets/f45bkkt8pr/1
[5] https://www.kaggle.com/datasets/gevabriel/indonesian-email-spam
[6] https://www2.aueb.gr/users/ion/data/enron-spam
[7] https://www2.aueb.gr/users/ion/publications.html
[8] https://www2.aueb.gr/users/ion/data/lingspam_public.tar.gz
[9] https://monkey.org/~jose/phishing/README.txt
[10] https://monkey.org/~jose/phishing
[11] https://trec.nist.gov/data/spam.html
[12] https://data.mendeley.com/datasets/shj94nrczy/1
[13] https://drive.google.com/file/d/1-stKadfTgJLtYsHWqXhGO3nTjKVFxm_Q/view?usp=sharing
[14] https://archive.ics.uci.edu/dataset/228/sms+spam+collection
[15] https://accesson.kr/kjai/assets/pdf/46934/journal-12-4-9.pdf
[16] https://tinyurl.com/datasetTA
[17] https://journal.uii.ac.id/AUTOMATA/article/view/19514
[18] https://monkey.org/~jose/phishing/LICENSE.txt
[19] https://github.com/bopbi/indonesia-sms-spam-dataset/blob/main/LICENSE.txt
[20] https://data.mendeley.com/datasets/vmg875v4xs/1

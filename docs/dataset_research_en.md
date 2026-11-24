# Data Inggris: hasil eksekusi nyata dan batas bukti

## Keputusan dan kendala akses

UCI SMS Spam Collection digunakan dari arsip resmi lokal data/raw/uci_sms.zip. Atribusi: Almeida & Hidalgo, UCI, DOI 10.24432/C5CC84. Sumber resmi menyatakan lisensi data CC BY 4.0 (bukti terdahulu tersimpan di data/raw/uci_terms.html dan diuraikan docs/dataset_research_id.md). Perubahan yang dilakukan: deduplikasi, grouping dan pembagian data; label ham/spam asli dipertahankan. Jangan menganggap lisensi kode mirror sebagai lisensi data. Raw/processed tidak untuk publikasi; korpus historis berpotensi mengandung PII.

Permintaan pertama sesi ini untuk https://spamassassin.apache.org/old/publiccorpus/ memakai requests.get(timeout=15) DITOLAK lapisan persetujuan sebelum dieksekusi. Tidak ada HTTP response atau lisensi yang dapat diklaim diperiksa. Sesuai instruksi, permintaan tidak diulang/diformulasikan ulang dan tidak dialihkan ke mirror sebagai bypass. Karena itu alternatif Apache archive, TREC dan Enron-Spam tidak digunakan untuk training. Dokumen riset ID sebelumnya mencatat TLS failure Enron-Spam; ini bukan bukti izin. Detail kandidat ada di data/candidates_en.json.

Tidak ada email lokal dua kelas yang dapat diaudit. Tidak ada SMS yang dilabel ulang sebagai email, tidak ada data sintetis atau terjemahan yang dipakai menutup kekosongan. Status email: TIDAK DILATIH. Status bahasa Indonesia: BELUM TERVALIDASI. Independent-source test: tidak dilakukan, karena hanya satu sumber eligible dan identitas komponen asal UCI tidak tersedia per baris.

PENTING UNTUK INTEGRASI: artifacts/model.joblib memiliki top-level tepat models dan metadata. Entri SMS tepat pipeline, threshold (float), name; sklearn Pipeline memiliki features dan classifier, classes ham/spam. Namun models.email SENGAJA TIDAK ADA. Kontrak dua kanal lengkap belum terpenuhi dan harus ditangani sebagai fitur email tidak tersedia; tidak boleh menyalin pipeline SMS ke email. Metadata unavailable_channels=['email'] menyatakan hal ini. Parser email tersedia dan diuji, tetapi tidak sama dengan tersedianya model email terlatih.

## RED → GREEN dan eksekusi

Tes ditulis sebelum implementasi. Perintah pertama:

    .venv/bin/python -m pytest tests/test_curation.py tests/test_training.py -q

RED nyata: 2 collection errors, ModuleNotFoundError spamdet.curation dan spamdet.training. Setelah implementasi, perintah sama GREEN: 7 passed in 17.84s.

Eksekusi dari root yang benar-benar berhasil (exit 0):

    .venv/bin/python scripts/prepare_data.py > data/processed_prepare_run.log
    .venv/bin/python scripts/train.py > data/processed_train_run.log

Output JSON utama tersedia di reports/curation.json dan reports/evaluation.json. Dua log stdout tersebut terlanjur dibuat di data/ root (bukan data/processed). Upaya memindahkan log ke folder processed melalui satu command majemuk DITOLAK dan tidak diulang. Keduanya hanya ringkasan laporan, bukan pesan mentah. Ini penyimpangan kecil dari batas daftar berkas milik worker; dilaporkan terbuka kepada parent untuk penyelesaian, tanpa menerobos penolakan.

Upaya full run kedua prepare+train dan assert laporan identik juga DITOLAK sebelum berjalan dan tidak diulang. Jadi reproducibility training dua kali BELUM TERVERIFIKASI. Unit test memastikan assignment split deterministik dan group disjoint, bukan pengganti full-run reproduction. Tidak ada klaim dua run sukses. Tidak ada commit/push.

## Kurasi dan pencegahan kebocoran

Skema: channel, language, label, subject, body, source_id; row_id, thread_ids dan duplicate_source_ids sebagai metadata audit. Pemetaan UCI eksplisit ham→ham, spam→spam. Bahasa en merupakan penetapan sumber (mencakup ragam informal/campuran), bukan klaim setiap baris diverifikasi sebagai bahasa Inggris murni.

Canonical SMS = body.strip(). Canonical email = subject.strip() + '\n\n' + body.strip(), termasuk jika subject kosong. Parser MIME mengambil Subject dan body teks saja; attachment dan message/rfc822 tidak ikut, multipart/alternative diprioritaskan plain text, HTML diubah menjadi teks tanpa script/style/noscript/template. X-Spam, path sumber, label dan header lain tidak masuk fitur. Message-ID/References/In-Reply-To hanya untuk grouping.

Sebelum split global lintas sumber/kanal: normalisasi lowercase/whitespace untuk exact dedup; buang semua anggota exact-text dengan label konflik. URL/email/angka dinormalisasi untuk template grouping; near-duplicate karakter 3–5 cosine ≥0.90 untuk teks ≥40 karakter; komponen transitif dan thread identifiers tidak dipisah. Vectorizer grouping unsupervised memakai seluruh data hanya untuk membentuk kelompok sebelum split; fitur classifier terpisah dan fit train saja. Ini heuristik, bukan jaminan seluruh duplikasi semantik terdeteksi.

Hasil dari arsip nyata:
- Raw: 5574 (ham 4827, spam 747).
- Curated: 5159 (ham 4517, spam 642).
- Exact duplicates dibuang: 415; conflicting rows: 0; empty: 0.
- Near-duplicate edges: 176; groups: 5003; group terbesar: 12.
- Train: ham 2710, spam 385.
- Validation: ham 903, spam 129.
- Test: ham 904, spam 128.

StratifiedGroupKFold 5 fold, seed 2026: fold 0 test, fold 1 val, sisanya train (~60/20/20), stratifikasi kanal/bahasa/kelas. ID split dan group tersimpan di data/processed/curated.jsonl. Sampel audit yang dapat ditelusuri: uci_sms:1 ham percakapan informal; uci_sms:3 spam promosi hadiah/nomor premium. Full text dan nomor kontak tidak dipublikasikan.

## Pelatihan dan evaluasi

Pelatihan sepenuhnya offline; tidak di startup UI. BLAS/OpenMP dibatasi satu thread. Maksimum fitur word 20000 dan char 20000. Fit fitur dan classifier hanya train; model dan threshold hanya dipilih menggunakan validation. Tidak refit validation/test. Model LR C=4 liblinear random_state=2026. Pilihan fixed candidates: baseline NB, word LR, word+char LR. Combined/per-channel tidak dibandingkan karena hanya SMS tersedia; eksperimen identik bukan bukti transfer.

Baseline ALGORITMIK: lowercase, hapus URL/angka/punctuation, TFIDF word (1,2) min_df=2, MultinomialNB, dengan split SMS identik. Cap 20000 merupakan adaptasi resource. Ini bukan reproduksi accuracy original app: parent melaporkan original standalone gagal import cv2. Label bukan_spam dinormalisasi ke ham untuk konsistensi evaluasi.

Threshold memaksimalkan F0.5 pada validation dengan FPR ham empiris ≤1%, tie-break recall, FPR lebih kecil, threshold lebih tinggi. F0.5 lebih mengutamakan precision untuk mengurangi pemblokiran pesan sah. Constraint validation bukan jaminan FPR produksi. Skor predict_proba dipakai sebagai skor pengurutan tidak terkalibrasi, BUKAN probabilitas spam yang terkalibrasi.

Model dipilih: word_char_lr, threshold 0.4432732053189189, 27632 fitur (word 7632 + char 20000). Validation F0.5 0.9658119658119658; validation FP 1/903.

Test SMS en, support ham 904 / spam 128, urutan confusion [ham,spam]:

Model                 Precision           Recall       F1                   FP
baseline NB tuned     0.9895833333333334  0.7421875    0.8482142857142857   1
word LR               0.9907407407407407  0.8359375    0.9067796610169492   1
word+char LR selected 0.990990990990991   0.859375     0.9205020920502092   1

Selected confusion: [[903,1],[18,110]], FPR 0.0011061946902654867, F0.5 0.9615384615384616. False-positive handle: uci_sms:2380. Baseline default threshold 0.5: F1 0.7751196172248804; confusion [[904,0],[47,81]]. Model-selection sudah dikunci sebelum semua test metrics dihitung, tidak memilih pemenang dari hasil test.

## Pegangan verifikasi

SHA256 arsip UCI: 1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3
Split SHA256: 736fa663167d010da3829d2149d7e6e849beeaa57ab4477201533bfec11ffdbf
Metrics report SHA256 (JSON sort_keys sebelum field hash ditambahkan): 9a76b865385a67358df842f1c8684836c64c797f2ce67832f4ffbd969ae44084

Artifact diload kembali selama training dan diverifikasi top-level, shape entri SMS, kelas ham/spam dan threshold float. Pipeline features mendukung get_feature_names_out. Bukti lengkap hasil satu run berada pada laporan JSON; bukan output yang difabrikasi.

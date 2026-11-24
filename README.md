# NLP Spam Detector

Aplikasi Python **Streamlit** untuk deteksi SMS, email, dan CSV campuran. Dikembangkan dari [nlp-spam-detector](https://github.com/nath2006/nlp-spam-detector) untuk pembelajaran NLP. Startup **tidak mengunduh dataset atau melatih model**.

## Status: PARTIAL

Upgrade lokal diverifikasi pada **24 November 2026**. Model nyata SMS dan email tersedia, tetapi email Indonesia belum tervalidasi dan cakupan pesan transaksi sah masih kurang.

| Kanal/bahasa | Cakupan |
| --- | --- |
| SMS Inggris | UCI + Mishra/Soni + India SMS, dominan Inggris/Hinglish dengan ragam informal/campuran; bukan anotasi bahasa per baris |
| SMS Indonesia | Rahmi/Wibisono, bopbi dan augmentasi SMS sintetis training-only; belum ada holdout asli baru Indonesia |
| Email Inggris | Ling-Spam, tambahan phishing Nazario training-only |
| Email Indonesia | **Belum tervalidasi**; menerima input bukan bukti dukungan bahasa |

**Status SMS terbaru:** recall dipulihkan dengan `sms_recall_lr_13`, threshold `0.5990758319506397`. B/C/D kembali SPAM dan tiga tes regresi tetap utuh; **90 tes lulus**. SMS Indonesia historis: FP 0→1, FN 10→3, recall 90,91%→97,27%; seluruh cohort SMS memiliki FPR test <1%. Pesanan/refund SMS tetap ham, tetapi newsletter SMS Inggris berlangganan kini salah SPAM. Email tidak diubah. Jangan gunakan sebagai pemblokir otomatis; skor bukan kepastian.

Laporan aktif: [docs/sms_recall_report.md](docs/sms_recall_report.md), bukti `reports/sms_recall/`. SHA-256 artefak `f8545bf00c72d11bbf5aabba358d5697091b6dbd1d8146c753f98e41186c2445`. Tabel/model/test pada bagian **Riwayat precision** di bawah menggambarkan putaran sebelumnya, bukan artefak aktif. Belum ada holdout Indonesia asli baru yang lolos verifikasi; seluruh test putaran recall merupakan reuse, bukan blind baru.

Riwayat precision: [docs/precision_report.md](docs/precision_report.md). Riwayat upgrade awal: [docs/upgrade_report.md](docs/upgrade_report.md). Manifest dasar: [data/sources.json](data/sources.json); tambahan precision: [data/precision_sources.json](data/precision_sources.json), termasuk URL asli/unduhan, SHA-256, izin, dan pemetaan label. Riset `docs/dataset_research_*.md` dan inventaris `data/candidates_*.json` adalah catatan **historis**, bukan status terbaru.

## Instalasi checkout baru — Windows PowerShell

Jalankan dari root proyek yang berisi `app.py`. Environment teruji: **Windows 11, Python 3.14.3**, Streamlit 1.67.0, scikit-learn 1.9.1. Perlu internet untuk instalasi/unduhan, dan `curl.exe` untuk download dengan verifikasi TLS.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -X utf8 scripts/download_data.py
.\.venv\Scripts\python.exe -X utf8 scripts/prepare_data.py
.\.venv\Scripts\python.exe -X utf8 scripts/train.py
.\.venv\Scripts\python.exe -X utf8 scripts/download_data.py --manifest data/precision_sources.json
.\.venv\Scripts\python.exe -X utf8 scripts/refine_model.py snapshot
.\.venv\Scripts\python.exe -X utf8 scripts/refine_model.py freeze
.\.venv\Scripts\python.exe -X utf8 scripts/refine_model.py train
.\.venv\Scripts\python.exe -X utf8 scripts/refine_model.py evaluate
.\.venv\Scripts\python.exe -X utf8 scripts/recover_sms_recall.py train
.\.venv\Scripts\python.exe -X utf8 scripts/recover_sms_recall.py evaluate
.\.venv\Scripts\python.exe -X utf8 scripts/recover_sms_recall.py promote
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
```

Buka **http://127.0.0.1:8501**. Biarkan terminal terbuka; **Ctrl+C** menghentikan server. Ganti `--server.port 8501` menjadi `--server.port 8502` jika port terpakai, lalu buka `http://127.0.0.1:8502`. Server hanya diikat ke localhost.

Tidak perlu mengaktifkan environment atau mengubah execution policy. Environment lama `venv` tidak dihapus; upgrade memakai **`.venv`**, jangan mencampur interpreter. `requirements-lock.txt` merekam versi yang diuji; `requirements.txt` berisi rentang versi. Jangan memakai berkas lama salah eja `requiretments.txt` untuk instalasi baru.

Urutan instalasi di atas untuk checkout baru tanpa snapshot/holdout. Pada checkout yang sudah berjalan, cukup perintah startup berikutnya. Jangan menjalankan ulang `prepare_data.py`/`train.py` dasar karena akan mengganti artefak/laporan aktif dengan baseline. `snapshot` mempertahankan dan memverifikasi snapshot lama, tidak memperbaruinya; `freeze` menolak perubahan dataset/manifest yang sudah dibekukan. Model pembanding tersimpan di `artifacts/history/precision_round/`. Tahap `evaluate` memasang kandidat terpilih dan menulis laporan. Reproduksi seluruh pipeline precision dari checkout kosong belum diuji; verifikasi fresh-checkout terdahulu hanya berlaku untuk baseline.

Putaran recall juga menyimpan snapshot di `artifacts/sms_recall/before/`; `train` menolak jika plan putaran sudah ada. `evaluate` hanya mengevaluasi pilihan validation; `promote` memasang SMS hanya jika gate per-cohort lolos dan email tetap identik. Jangan menghapus lock untuk mengulang tuning pada test yang sama. Checkout ini sudah selesai menjalankannya; cukup startup biasa.

### Startup berikutnya — tidak perlu training ulang

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
```

Artefak default `artifacts/model.joblib` ditemukan relatif terhadap modul proyek, bukan direktori kerja. UI menunjukkan path, SHA-256, kelas, indeks spam, threshold, dan bahasa test. Inferensi memuat artefak dari disk, bukan cache model lama.

Override lokasi model:

```powershell
$env:SPAM_MODEL_PATH = "C:\lokasi\tepercaya\model.joblib"
```

Hapus override dengan `Remove-Item Env:SPAM_MODEL_PATH -ErrorAction SilentlyContinue`. **Muat hanya joblib tepercaya**: deserialisasi dapat mengeksekusi kode.

### Git Bash dan Linux/macOS

Di Windows Git Bash, ganti `.\.venv\Scripts\python.exe` dengan `.venv/Scripts/python.exe` pada seluruh perintah di atas. Perintah awal tetap `python -m venv .venv`.

Linux/macOS (belum diuji dalam upgrade ini):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -X utf8 scripts/download_data.py
python -X utf8 scripts/prepare_data.py
python -X utf8 scripts/train.py
python -X utf8 scripts/download_data.py --manifest data/precision_sources.json
python -X utf8 scripts/refine_model.py snapshot
python -X utf8 scripts/refine_model.py freeze
python -X utf8 scripts/refine_model.py train
python -X utf8 scripts/refine_model.py evaluate
python -X utf8 scripts/recover_sms_recall.py train
python -X utf8 scripts/recover_sms_recall.py evaluate
python -X utf8 scripts/recover_sms_recall.py promote
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
```

Override POSIX: `export SPAM_MODEL_PATH=/lokasi/tepercaya/model.joblib`. Platform/Python lain belum diverifikasi; perubahan versi dependency dapat mengubah metrik dan kompatibilitas artefak. Gunakan lock pada environment Windows yang sesuai untuk reproduksi.

## Dataset dan alur offline

`scripts/download_data.py` membaca enam sumber dalam manifest, mengunduh dengan curl melalui TLS terverifikasi, dan memeriksa SHA-256. Cache valid dilewati; cache rusak ditolak tanpa ditimpa. Jangan menonaktifkan verifikasi TLS/hash. Pada lingkungan ini curl Windows berhasil mengakses AUEB ketika chain sertifikat tidak dikenali `requests` Python.

| Sumber asli | Pesan sebelum kurasi | Izin/catatan |
| --- | ---: | --- |
| UCI SMS | 5.574 | CC BY 4.0.[14] |
| Mishra/Soni SMS phishing | 5.971 | CC BY 4.0; banyak overlap UCI.[4] |
| Rahmi/Wibisono SMS Indonesia | 1.143 | CC BY-SA 4.0 dari `penjelasan.txt` ZIP asli.[2][13] |
| bopbi SMS | 50 | CC0 1.0; satu kelas, training-only.[19] |
| Ling-Spam `bare` | 2.893 | Ketentuan asli: atribusi dan pemberitahuan untuk karya yang dipublikasikan; bukan lisensi CC/Apache baru.[8] |
| Nazario phishing 2023 | 419 | CC BY 4.0; bahasa campuran/tidak diverifikasi (`und`), training-only.[9][18] |

Atribusi lengkap dan ketentuan ada di manifest/laporan. Tugas ini lokal, tidak mempublikasikan atau mengirim pemberitahuan kepada pembuat Ling-Spam. Pertahankan atribusi, tautan lisensi, dan catatan perubahan sebelum membagikan karya. Lisensi artikel/kode/mirror tidak otomatis menjadi izin korpus asli. Raw/processed data dan artefak diabaikan Git; clone baru memerlukan pipeline offline di atas.

`prepare_data.py` hanya membaca arsip lokal:

1. Pemetaan label eksplisit: UCI ham/spam; Mishra ham versus spam/Smishing (case-insensitive); Rahmi/Wibisono 0→ham, 1/2→spam; Ling-Spam `spmsg*.txt`→spam dan file `bare` lain→ham; kategori bopbi dan phishing Nazario→spam.
2. Exact dedup global, membuang konflik label; grouping template URL/email/angka dan near-duplicate cosine karakter ≥0,90.
3. Email juga dikelompokkan menurut body template, subjek deskriptif tanpa awalan balasan, dan RFC thread IDs jika tersedia.
4. Grouped split seed 2026 sekitar 60/20/20; grup training-only tidak masuk holdout. Ini tetap heuristik, bukan jaminan bebas seluruh duplikasi semantik.

Urutan manifest menentukan pemilik baris duplikat; sumber duplikat tetap dicatat. Mishra berada lebih dulu, sehingga banyak pesan UCI dihitung pada Mishra, bukan dijumlahkan dua kali. Tidak ada SMS yang diberi ulang kanal email.

Model SMS/email terpisah. Email memakai `subject.strip() + '\n\n' + body.strip()` yang sama pada training dan prediksi, tanpa header label/alamat pengirim sebagai fitur. Kurasi MIME memilih teks/plain atau mengubah HTML menjadi teks; tidak menjalankan script, attachment, tautan, atau gambar.

## Riwayat precision — model dan evaluasi sebelum pemulihan recall

Empat kandidat per kanal: TF-IDF kata + MultinomialNB (`baseline_nb`, dengan pembersihan URL/angka/tanda baca), kata + Logistic Regression, kata+karakter + Logistic Regression, dan kata+karakter + Linear SVM dengan sigmoid calibration dari CV training saja.

Putaran dasar sebelumnya menggunakan FPR gabungan. Putaran precision kini memilih **hanya dari validation**: maksimum rata-rata F2 per kelompok, dengan FPR ham ≤3% pada **setiap** kelompok yang memiliki ≥100 ham dan ≥30 spam. Satu threshold per kanal; tidak ada detektor bahasa atau routing threshold per bahasa. Kelompok kecil dilaporkan tanpa klaim memenuhi constraint. Classifier/vectorizer di-fit pada training saja; vectorizer grouping tidak dipakai classifier. Contoh diagnosis/manual di `examples/` **tidak masuk training, tuning, atau benchmark**. Augmentasi sintetis dari sumber terpisah ditandai training-only, bukan evaluasi asli.

Holdout baru yang dibekukan sebelum tuning: **506 ham + 290 spam SMS Inggris/Hinglish** dari India SMS. Ini split berkelompok dari sumber tambahan yang juga memasok train/validation, **bukan unseen-source test**. Bahasa bukan anotasi per baris.

| Model SMS pada holdout baru yang sama | Precision | Recall | F1 | FPR ham | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline tetap | 0,8571 | 0,9310 | 0,8926 | 8,89% | 45 | 20 |
| Baseline threshold dituning ulang | 0,9834 | 0,6138 | 0,7558 | 0,59% | 3 | 112 |
| LR kata+karakter + augmentasi (terpilih) | 1,0000 | 0,9379 | 0,9680 | 0% | 0 | 18 |
| SVM kata+karakter + augmentasi | 1,0000 | 0,7897 | 0,8825 | 0% | 0 | 61 |

Artefak historis putaran precision: SMS `word_char_lr_aug`, threshold `0.7172193110759366`; email mempertahankan pipeline SVM lama (`current_retuned`) dan threshold `0.17166622963704584`. SHA-256 `817d81044bce4b5e80d402159324334321b9567296a0a371052e1b873f3cbfe0`. Artefak aktif terbaru tercantum di atas dan laporan recall.

**Trade-off test historis (bukan holdout baru):** SMS Indonesia, 113 ham/110 spam: FP 11→0, FN 0→10, recall 100%→90,91%. SMS Inggris, 1.000 ham/206 spam: FP 4→0, FN 7→45, recall 96,60%→78,16%. Email Inggris, 443 ham/56 spam: tetap FP 4/FN 0. Support berkurang dari laporan lama karena grup gabungan yang melintasi split dikarantina; baseline dan kandidat dibandingkan pada baris identik. Hasil penuh per kelompok/validation ada di laporan precision. Belum ada holdout asli baru SMS Indonesia atau email Inggris/Indonesia. FPR empiris nol bukan jaminan produksi.

Tambahan data: India SMS 2.267 pesan (ham 1.522/spam 745), MIT yang dinyatakan berlaku untuk dataset; setelah kurasi dipakai train 793, validation 437, test baru 796, overlap dikeluarkan 25.[21][22]

OpenTextShield 15.027 pesan **sintetis**, MIT; subset Inggris/Indonesia 1.903 (ham 1.088/spam 815), seluruhnya training-only.[25][26]

Metadata, pembatasan sampel, atribusi, dan URL ada di manifest tambahan serta laporan precision. Label sumber tidak menyediakan consent per penerima; promo berlangganan tidak otomatis dapat dibedakan dari promosi tanpa izin.

Training menulis artefak setelah load-back berhasil, serta `reports/evaluation.json`. Kurasi menulis `data/processed/curated.jsonl` dan `reports/curation.json`. Training ulang memperbarui laporan; startup biasa tidak.

## Menggunakan UI dan CSV

- **SMS:** masukkan isi pesan lalu Deteksi; OCR opsional.
- **Email:** isi wajib, subjek opsional; pipeline email asli, tidak fallback SMS. Input Indonesia tetap belum tervalidasi.
- **Massal:** CSV UTF-8 berheader `channel,body`, `subject` opsional; kanal persis `sms`/`email`. Template: [examples/batch_template.csv](examples/batch_template.csv).
- **Evaluasi:** sumber dataset, support, precision/recall/F1, FP/FN, dan laporan lengkap.

Batas: SMS 10.000 karakter, email 100.000 karakter isi, subjek 1.000 karakter; CSV 2 MiB/1.000 baris. Isi kosong ditolak. CSV rusak ditolak; kesalahan baris dilaporkan per baris. Ekspor `row,label,display,score_model,threshold,error` tidak berisi isi/subjek pesan dan menetralkan formula spreadsheet. Row dimulai dari 2, setelah header.

## Tes dan bukti UI

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -X utf8 scripts/diagnose.py --output reports/sms_recall/diagnosis.json
```

Verifikasi aktif terbaru: **90 tes lulus**, exit code pytest **0**, tiga assertion B/C/D tetap utuh. Smoke browser SMS memakai artefak baru; seluruh 8 label/skor cocok dengan inferensi langsung, termasuk newsletter sah yang salah SPAM. Bukti `reports/sms_recall/browser_sms.json` dan `reports/sms_recall/tests.xml`; pip check bersih.

Riwayat precision sebelumnya: **83 tes lulus, 3 gagal (86 total)**; exit code pytest **1**. Assertion kualitas `test_manual_sms_regression_and_real_ui_parity[1/2/3]` saat itu menandai regresi B/C/D dan tidak dihapus/dilonggarkan. Tes `test_real_artifact.py` memakai model nyata dan dilewati pada clone yang belum dilatih; fixture sintetis lain menguji integrasi, bukan kualitas deteksi.

Riwayat verifikasi precision: setelah restart, browser menguji 8 SMS, 6 variasi email (dengan/tanpa subjek), serta unggahan 8 baris CSV campuran. Label/skor sama dengan inferensi langsung. EasyOCR asli dan unggahan gambar `examples/ocr_smoke.png` menghasilkan teks Indonesia/Inggris, prediksi ham `0.0122`; parity UI/direct lulus. Emulasi viewport **390×844** menguji OCR, navigasi, dan Evaluasi Model tanpa overflow horizontal dokumen; tabel/kode punya scroll internal. Bukan tes HP fisik. Bukti: `reports/precision_verification.json`, `reports/precision_browser_*.json`, `reports/precision_mobile.json`, `reports/ocr_smoke.json`, dan `reports/screenshots/precision_*.png`. Root dan health HTTP 200.

**Riwayat saja:** reproduksi baseline terdahulu di direktori sementara berhasil dengan 79 tes, split/model/metrik identik, memakai `.venv` yang sudah terpasang. `reports/fresh_checkout_verification.json` bukan bukti reproduksi putaran precision terbaru atau instalasi pada mesin baru.

## OCR, troubleshooting, dan privasi

OCR opsional: `.\.venv\Scripts\python.exe -m pip install -r requirements-ocr.txt`. Bobot pertama dapat diunduh; instalasi dan smoke test EasyOCR CPU sudah diverifikasi pada environment ini (ada warning deprecation PyTorch/pin-memory, bukan kegagalan). Batas gambar 10 MiB/20 juta piksel. Deteksi teks tidak memerlukan OCR.

- Model hilang: jalankan download → prepare → train; periksa path/SHA dan `SPAM_MODEL_PATH`.
- Email belum tersedia: artefak lama SMS-only/path override dapat menjadi penyebab. Jalankan pipeline aktif dan restart, bukan fallback SMS.
- Hash berbeda: hentikan dan verifikasi perubahan sumber, jangan mengganti hash sembarangan.
- Unduhan gagal: catat sumber yang gagal; jangan mengklaim dataset berhasil dipakai. Unduhan manual hanya sah bila izin dan hash cocok.
- Artefak tidak kompatibel: gunakan environment yang sesuai dan latih ulang; jangan memuat joblib sembarang.
- Prediksi salah: audit kanal, data, kelas, skor, threshold, FP/FN; jangan menyesuaikan threshold berdasarkan satu contoh.

Pesan pengguna diproses di memori sesi, tidak secara default ditulis ke disk/log. Widget bisa mempertahankan input selama sesi aktif. Bukti diagnosis/screenshot hanya menggunakan contoh manual/sintetis, bukan pesan pribadi pengguna. Dataset publik dapat mengandung informasi pribadi: jangan mempublikasikan raw/processed data atau artefak tanpa peninjauan izin/privasi.

Tidak ada commit, push, deployment, atau penetapan lisensi kode baru dalam upgrade ini.

## Sources

[2] https://yudiwbs.wordpress.com/2018/08/05/dataset-klasifikasi-bahasa-indonesia-sms-spam-klasifikasi-teks-dengan-scikit-learn
[4] https://data.mendeley.com/datasets/f45bkkt8pr/1
[8] https://www2.aueb.gr/users/ion/data/lingspam_public.tar.gz
[9] https://monkey.org/~jose/phishing/README.txt
[13] https://drive.google.com/file/d/1-stKadfTgJLtYsHWqXhGO3nTjKVFxm_Q/view?usp=sharing
[14] https://archive.ics.uci.edu/dataset/228/sms+spam+collection
[18] https://monkey.org/~jose/phishing/LICENSE.txt
[19] https://github.com/bopbi/indonesia-sms-spam-dataset/blob/main/LICENSE.txt
[21] https://github.com/junioralive/india-spam-sms-classification
[22] https://raw.githubusercontent.com/junioralive/india-spam-sms-classification/main/LICENSE
[25] https://github.com/TelecomsXChangeAPi/OpenTextShield
[26] https://raw.githubusercontent.com/TelecomsXChangeAPi/OpenTextShield/main/evals/generate_legit_notices.py

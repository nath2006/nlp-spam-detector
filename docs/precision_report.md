# Laporan precision refinement — 24 November 2026

## Keputusan dan status

**PARTIAL — bukan kandidat siap produksi.** Model SMS yang dipilih pada validation mengurangi false positive, tetapi mengorbankan recall pada test historis dan gagal pada contoh spam manual B/C/D. Pipeline email Inggris dipertahankan; refund/newsletter email sah tetap gagal. Belum ditemukan korpus email Indonesia asli berlabel dengan izin yang dapat diverifikasi untuk putaran ini.

Checkout ditentukan dengan `git rev-parse --show-toplevel`, bukan path hardcoded dari laporan lama. Tidak ada commit, push, publikasi dataset, atau deployment. Delegasi dokumentasi gagal HTTP 403 karena kredensial provider kedaluwarsa; dokumentasi diselesaikan langsung tanpa mengubah kredensial.

Server lokal setelah restart: `http://127.0.0.1:8501`, root dan `/_stcore/health` HTTP 200. Startup memakai `.venv/Scripts/python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false`.

| Identitas | SHA-256 / nilai |
| --- | --- |
| Snapshot model pembanding | `2a134b92955326fab0f926a9d1d38c944bfae0710f05a981c811fbb2742e1176` |
| Model aktif | `817d81044bce4b5e80d402159324334321b9567296a0a371052e1b873f3cbfe0` |
| Frozen membership manifest | `28d9ee62dc8e5a9bd22163d77122bdd587e984810420652e8395c3a82e95d957` |
| Frozen processed rows | `b3e0629b3c7670632936071eff8a353043fc3d23f75f884b58e8d0d5c74bad33` |
| SMS | `word_char_lr_aug`, threshold `0.7172193110759366` |
| Email | `current_retuned` (pipeline SVM lama), threshold `0.17166622963704584` |

Snapshot tidak ditimpa: `artifacts/history/precision_round/`. Laporan lama tetap di `reports/history/precision_round/`; baseline UCI yang lebih tua terpisah di `artifacts/baseline/`.

## Audit 11 false positive SMS Indonesia

Hanya row ID, hash teks, kategori, dan skor dicatat di `reports/precision_error_audit.json`; body pesan, nomor, dan URL pribadi tidak dipublikasikan.

| Pola konteks sah | Jumlah |
| --- | ---: |
| Harga/perumahan | 1 |
| Koordinasi kelompok | 2 |
| Pesan pribadi/pekerjaan pendek | 2 |
| Layanan/transaksi sah | 5 |
| Diskusi teknis | 1 |

Seluruh 11 sekarang diprediksi ham. Ini audit ulang test historis yang sudah diperiksa, **bukan 11 contoh blind baru**. Kelompok ini konsisten dengan masalah cakupan ham dan pemisahan skor pada kosakata layanan/komersial; kategori merupakan diagnosis manusia, bukan bukti kausal bahwa kata tertentu menyebabkan seluruh kesalahan. Tidak ditambahkan aturan keyword atau whitelist untuk meloloskan contoh.

Label asli Rahmi/Wibisono: 0 normal, 1 penipuan, 2 promosi; sumber tidak menyediakan anotasi consent penerima yang membedakan newsletter/promo berlangganan dari promosi tanpa izin. Label promosi tidak diubah otomatis menjadi ham.[2][13] Dataset India juga hanya memberi label ham/spam, bukan consent individual.[21]

## Sumber tambahan dan kelayakan

| Sumber | Kanal/bahasa | Jumlah mentah dan label | Izin/status penggunaan |
| --- | --- | --- | --- |
| JuniorAlive India SMS | SMS Inggris/Hinglish, penetapan bahasa tingkat sumber; bukan gold per baris | 2.267: ham 1.522, spam 745 | MIT secara eksplisit diterapkan pada dataset di README asli; digunakan setelah kurasi.[21][22] |
| OpenTextShield legitimate notices + phishing twins | SMS sintetis; dipilih Inggris dan Indonesia, bukan email | 15.027: ham 12.000, phishing 3.027 | MIT untuk fixture/generator in-house; subset 1.903, training-only, tidak menjadi evaluasi asli.[25][26][27] |
| High-Accuracy Email Classifier Indonesian | Email Indonesia terjemahan/adaptasi, bukan email asli Indonesia | Enam kategori termasuk spam, promotions, verify_code; jumlah baris tidak diverifikasi dalam putaran ini, dipakai 0 | Card menyatakan Apache-2.0 dan terjemahan mesin. Ditolak sebagai korpus uji asli; tidak dipakai sebagai augmentasi kali ini.[23] |
| SpamShield | Kanal/bahasa/label/jumlah belum terverifikasi dari korpus yang dapat diakses | Dipakai 0 | Card mendapat HTTP 401; akses dan izin belum cukup untuk diterima.[28] |
| Kandidat Kaggle email Indonesia terdahulu | Belum dapat diverifikasi | Dipakai 0 | Halaman mendapat 404; artikel atau judul dataset tidak cukup sebagai bukti lisensi dan data.[5] |
| DatasetTA / artikel UII | Kandidat email Indonesia; korpus/izin/jumlah belum terverifikasi | Dipakai 0 | Shortlink/artikel mendapat 403; tidak diklaim sebagai korpus berhasil diunduh.[16][17] |
| TREC05 | Kandidat email historis, bukan bukti korpus asli Indonesia | Dipakai 0 | README ditelusuri, tidak diadopsi sebagai korpus Indonesia atau tambahan training putaran ini.[24] |

Bukti retrieval: `reports/precision_retrieval.json`, `reports/indonesian_email_retrieval.json`, metadata/hash/URL di `data/precision_sources.json`, salinan bukti sumber di direktori raw lokal. Tidak ada klaim semua kandidat yang ditelusuri berhasil diverifikasi.

India: 27 baris kosong/media-placeholder dikeluarkan sebelum grouping; 2.240 eligible. Audit gabungan mencatat 189 duplicate dikeluarkan secara global. Penggunaan India final: train ham 509/spam 284, validation ham 281/spam 156, new test ham 506/spam 290, overlap dikeluarkan ham 19/spam 6. Baris hilang lainnya dari jumlah eligible merupakan dedup global. Jumlah dipakai dihitung dari pemilik baris setelah dedup, bukan menjumlah ulang duplikat sumber.

OpenTextShield: hanya Inggris/Indonesia; maksimum 50 contoh Inggris per pasangan kategori/label dipilih deterministik menurut hash, seluruh contoh Indonesia yang memenuhi seleksi dipertahankan. Subset training 1.903: ham 1.088/spam 815; bahasa en 1.550/id 353. Fixture/generator menyediakan pola notifikasi sah dan pasangan phishing; keberadaan template transaksi tidak membuktikan performa transaksi nyata atau consent penerima. Seluruh subset ini training-only. Sumber SMS tidak diubah kanal menjadi email.

## Freeze, grouping, dan batas evaluasi

`reports/frozen_precision_holdout.json` dibekukan pada `2026-11-24T10:58:14.068779+00:00`; pemilihan validation dicatat pada `2026-11-24T11:02:37.564152+00:00`. Manifest menyimpan hash dataset, source manifest, model baseline, serta row ID/group/hash semua 796 anggota test baru.

Exact dedup/conflict, template URL/email/angka, near-duplicate karakter cosine ≥0,90, email body/subjek/thread diproses global. Keanggotaan grup historis juga dipertahankan; grup gabungan yang menjembatani split historis dikarantina, bukan dipindahkan ke train. Grup tambahan yang bersentuhan dengan baseline tidak boleh masuk test baru. Synthetic hanya train. Pemeriksaan programatik memverifikasi grup aktif tidak melintasi split dan semua contoh manual tidak ada dalam dataset.

Test baru hanya **SMS Inggris/Hinglish dari India, 506 ham/290 spam**. Sumber tambahan ini juga memasok train/validation: test berkelompok yang belum dipakai tuning, **bukan source-disjoint/unseen-source benchmark**. Rasio awal grup eksternal 40/20/40 berbasis hash, bukan pembagian acak ulang setelah melihat metrik.

SMS Indonesia dan email Inggris tetap test historis yang telah diperiksa dalam pengembangan sebelumnya, bukan holdout blind baru. Setelah penggabungan/karantina, support SMS Inggris menjadi 1.000 ham/206 spam (sebelumnya 1.002/213), email Inggris menjadi 443/56 (sebelumnya 482/94). SMS Indonesia tetap 113/110. Semua perbandingan lama/kandidat dalam tabel berikut menggunakan baris identik. Grouping tetap heuristik, bukan jaminan tanpa duplikasi semantik sama sekali.

## Pemilihan hanya pada validation

Tujuan: maksimum rata-rata F2 per kelompok, FPR ham ≤3% pada setiap kelompok dengan ≥100 ham dan ≥30 spam; tie-break recall dan kunci kandidat deterministik. Satu threshold per kanal, **tanpa detektor bahasa atau threshold berdasarkan bahasa**. Bahasa untuk evaluasi berasal dari metadata sumber. Target empiris bukan confidence bound atau jaminan produksi. Kelompok kecil tidak diklaim memenuhi constraint; email Indonesia tidak punya sampel.

| Kandidat | Kelompok | Ham/spam validation | FPR | Recall | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline tetap | sms_en | 1.289/367 | 1,71% | 94,82% | 22 | 19 |
| Baseline tetap | sms_id | 114/118 | 8,77% | 100% | 10 | 0 |
| Baseline retuned | sms_en | 1.289/367 | 0,31% | 80,93% | 4 | 70 |
| Baseline retuned | sms_id | 114/118 | 2,63% | 88,14% | 3 | 14 |
| LR augmented — terpilih | sms_en | 1.289/367 | 0,16% | 88,83% | 2 | 41 |
| LR augmented — terpilih | sms_id | 114/118 | 2,63% | 93,22% | 3 | 8 |
| SVM augmented | sms_en | 1.289/367 | 0,08% | 88,01% | 1 | 44 |
| SVM augmented | sms_id | 114/118 | 2,63% | 91,53% | 3 | 10 |
| Email lama/retuned identik | email_en | 477/90 | 0,63% | 100% | 3 | 0 |

Threshold baseline SMS tetap 0,421444; retuned 0,896368; LR 0,717219; SVM 0,703631. Macro F2 SMS LR 0,923872, SVM 0,913736, baseline retuned 0,868779. Baseline tetap tidak layak constraint SMS-ID. Pemilihan disimpan sebelum evaluasi test; contoh diagnosis tidak digunakan untuk menentukan model/threshold. Semua angka lengkap di `reports/precision_validation.json`.

## Perbandingan pada holdout baru yang sama

Support setiap baris: **506 ham/290 spam**.

| Model SMS | Precision | Recall | F1 | FPR | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline tetap | 0,8571 | 0,9310 | 0,8926 | 8,89% | 45 | 20 |
| Baseline retuned | 0,9834 | 0,6138 | 0,7558 | 0,59% | 3 | 112 |
| LR augmented — terpilih | 1,0000 | 0,9379 | 0,9680 | 0% | 0 | 18 |
| SVM augmented | 1,0000 | 0,7897 | 0,8825 | 0% | 0 | 61 |

Menurunkan FP hanya dengan threshold baseline sangat mengurangi recall. Kandidat LR memberi kompromi validation terbaik, tetapi hasil test eksternal tidak menghapus kegagalan pada domain historis berikut.

## Perbandingan test historis, bukan blind baru

| Kelompok/model | Ham/spam | Precision | Recall | F1 | FPR | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SMS-ID baseline | 113/110 | 0,9091 | 1,0000 | 0,9524 | 9,73% | 11 | 0 |
| SMS-ID LR aktif | 113/110 | 1,0000 | 0,9091 | 0,9524 | 0% | 0 | 10 |
| SMS-EN baseline | 1.000/206 | 0,9803 | 0,9660 | 0,9731 | 0,40% | 4 | 7 |
| SMS-EN LR aktif | 1.000/206 | 1,0000 | 0,7816 | 0,8774 | 0% | 0 | 45 |
| Email-EN baseline/aktif | 443/56 | 0,9333 | 1,0000 | 0,9655 | 0,90% | 4 | 0 |

**Trade-off penting:** target penurunan false positive tercapai secara empiris pada kelompok historis ini, tetapi SMS-ID kehilangan 10 spam dan SMS-EN kehilangan 45 spam. Email Indonesia belum dievaluasi. Email Inggris yang historis dan bertopik ham sempit tidak mewakili seluruh email transaksi/newsletter modern. Semua kandidat, confusion matrix dan support ada di `reports/precision_evaluation.json`.

## Pesan sah pembanding dan regresi manual

Diagnosis-only, bukan benchmark representatif atau data tuning:

| Kontrol | Sebelum | Sesudah |
| --- | --- | --- |
| Pesanan SMS Indonesia | Salah SPAM | HAM, skor 0,1265 |
| Refund SMS Indonesia | Salah SPAM | HAM, skor 0,6904 |
| Newsletter SMS Inggris berlangganan | Salah SPAM | HAM, skor 0,4327 |
| Pesanan SMS Inggris | HAM | HAM, skor 0,1272 |
| Refund email Inggris | Salah SPAM | Tetap salah SPAM, skor 0,9932 dengan subjek |
| Newsletter email Inggris berlangganan | Salah SPAM | Tetap salah SPAM, skor 0,9970 dengan subjek |
| Spam A | SPAM | SPAM, skor 0,7696 |
| Spam B | SPAM | Salah HAM, skor 0,6296 |
| Spam C | SPAM | Salah HAM, skor 0,6656 |
| Spam D | SPAM | Salah HAM, skor 0,4771 |

Refund/newsletter email juga tetap false positive tanpa subjek. Uji email permintaan kredensial terdeteksi spam dengan/tanpa subjek. Bukti `reports/precision_diagnosis_after.json`, snapshot diagnosis lama, dan browser email. Tidak tersedia benchmark transaksi/newsletter asli Indonesia berlabel per kategori, sehingga tidak ada klaim FPR kategori produksi. Contoh OTP/invoice dalam augmentasi bukan holdout independen.

## Verifikasi UI, OCR, mobile, dan tes

- Restart proses Streamlit, kemudian hash artefak nyata terlihat di UI. Root dan health HTTP 200.
- Browser: 8 SMS (A–D dan 4 kontrol sah), 6 variasi email (3 pesan dengan/tanpa subjek); semua label/skor tampilan sama dengan inferensi langsung.
- Unggah CSV campuran 8 baris melalui file uploader; label, skor, threshold setiap baris dibandingkan programatik dengan inferensi langsung, tanpa error baris. Prediksi salah model tetap dicatat sebagai masalah kualitas, bukan kegagalan uploader.
- EasyOCR CPU asli membaca gambar sintetis `examples/ocr_smoke.png`: `Besok rapat pukul 10 pagi. Please bring the meeting agenda.`; ham, skor 0,012240417568353612. Jalur ekstraksi langsung dan upload browser sama. Warning PyTorch deprecation/pin-memory tidak menggagalkan smoke test.
- Viewport emulasi 390×844: OCR, menu navigasi, dan Evaluasi Model diperiksa; lebar dokumen 390, tidak overflow horizontal. Tabel/kode mempunyai scroll internal. Bukan pengujian HP fisik.
- Halaman Evaluasi Model membedakan test historis, holdout baru, validation, target kelompok, sumber tambahan, serta warning regresi B/C/D dan email Indonesia belum tervalidasi.
- **86 tes: 83 lulus, 3 gagal; pytest exit 1.** Kegagalan `test_manual_sms_regression_and_real_ui_parity[1]`, `[2]`, `[3]` adalah regresi kualitas B/C/D. Assertion dipertahankan, tidak dihapus/dilonggarkan untuk menjadikan suite hijau. Parity browser tetap terverifikasi terpisah.
- `pip check`: `No broken requirements found.`; `git diff --check` lulus pada pemeriksaan kode/UI.

Bukti gabungan `reports/precision_verification.json`; output pytest `reports/precision_test_output.txt` dan `reports/precision_test_results.xml`; screenshot `reports/screenshots/precision_*.png`. Hasil 79 tes dan fresh-checkout dalam laporan upgrade terdahulu hanya riwayat baseline. Tidak dilakukan ulang reproduksi precision dari checkout kosong/mesin baru.

## Reproduksi dan langkah berikutnya

README memisahkan instalasi baru dan startup biasa. Setelah baseline dasar tersedia, unduh manifest tambahan, jalankan `scripts/refine_model.py snapshot`, `freeze`, `train`, lalu `evaluate`. Snapshot lama diverifikasi hash, bukan diganti; frozen rows/manifest berubah menyebabkan penolakan. `evaluate` memverifikasi hash kandidat/holdout dan memuat balik artefak sebelum memasang. Jangan rerun training dasar pada checkout aktif kecuali memang ingin mengganti artefak.

Prioritas lanjutan: korpus asli Indonesia ham transaksi dan spam modern yang berizin, anotasi consent/kategori yang dapat diaudit, holdout baru yang tidak pernah dipakai pengembangan, dan peninjauan regresi recall tanpa menyesuaikan threshold ke contoh manual. Email Inggris tetap tersedia dengan warning cakupan; email Indonesia tetap **belum tervalidasi**. Kandidat lokal tidak diterima sebagai pemblokir otomatis.

## Sources

[1] https://spamassassin.apache.org/old/publiccorpus/readme.html
[2] https://yudiwbs.wordpress.com/2018/08/05/dataset-klasifikasi-bahasa-indonesia-sms-spam-klasifikasi-teks-dengan-scikit-learn
[3] https://github.com/bopbi/indonesia-sms-spam-dataset
[5] https://www.kaggle.com/datasets/gevabriel/indonesian-email-spam
[13] https://drive.google.com/file/d/1-stKadfTgJLtYsHWqXhGO3nTjKVFxm_Q/view?usp=sharing
[16] https://tinyurl.com/datasetTA
[17] https://journal.uii.ac.id/AUTOMATA/article/view/19514
[21] https://github.com/junioralive/india-spam-sms-classification
[22] https://raw.githubusercontent.com/junioralive/india-spam-sms-classification/main/LICENSE
[23] https://huggingface.co/datasets/chairulridjal/high-accuracy-email-classifier-indonesian/raw/main/README.md
[24] https://trec.nist.gov/data/spam/trec05p-1.tgz.README.txt
[25] https://github.com/TelecomsXChangeAPi/OpenTextShield
[26] https://raw.githubusercontent.com/TelecomsXChangeAPi/OpenTextShield/main/evals/generate_legit_notices.py
[27] https://raw.githubusercontent.com/TelecomsXChangeAPi/OpenTextShield/main/docs/LABELING_GUIDE.md
[28] https://huggingface.co/datasets/M-Arjun/SpamShield-Datasets

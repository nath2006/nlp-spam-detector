# Pemulihan recall SMS — 24 November 2026

## Keputusan

**Kandidat SMS dipasang setelah lolos evaluasi; tiga tes regresi tetap utuh dan kini lulus.** Email tidak dilatih ulang, tidak diubah fitur/threshold-nya, dan fingerprint komponen email sebelum/sesudah identik. Status kualitas keseluruhan tetap **PARTIAL**: masih ada FN historis dan false positive pada newsletter SMS sah pembanding.

Model aktif: `sms_recall_lr_13`, threshold `0.5990758319506397`, satu threshold SMS tanpa routing bahasa.

SHA-256 artefak aktif: `f8545bf00c72d11bbf5aabba358d5697091b6dbd1d8146c753f98e41186c2445`.

Snapshot model yang aktif sebelum putaran ini dan laporan evaluasinya: `artifacts/sms_recall/before/`. SHA-256 model sebelum: `817d81044bce4b5e80d402159324334321b9567296a0a371052e1b873f3cbfe0`. Baseline SVM pembanding tetap `artifacts/history/precision_round/model.joblib`, SHA-256 `2a134b92955326fab0f926a9d1d38c944bfae0710f05a981c811fbb2742e1176`.

Checkout aktif diresolusikan melalui Git. Tidak ada commit, push, deployment, atau perubahan fitur email.

## Diagnosis dan perubahan

Gejala direproduksi lebih dahulu: tiga assertion B/C/D gagal pada model aktif sebelumnya; FN test historis Indonesia 10, FP 0. Menaikkan threshold pada baseline saja telah terbukti mengorbankan recall. Pada model aktif, satu threshold hasil validation dibatasi kelompok Indonesia sementara training didominasi bahasa Inggris dan augmentasi sintetis; ini menunjukkan masalah pemisahan skor dan bobot cakupan, bukan mapping kelas atau UI yang terbalik.

Pencarian dibatasi dan dicatat sebelum fitting di `reports/sms_recall/plan.json`: 14 varian LR plus dua pipeline lama dengan threshold retuned. Variasi C 1/4/16, bobot sintetis 0,1 atau 1, penyeimbangan bahasa/label aktif atau tidak; dua varian terakhir memperluas fitur. Tidak ada percobaan tambahan setelah melihat test atau diagnosis manual.

Kandidat terpilih memakai:

- TF-IDF kata 1–2 dan karakter 2–5; `min_df=1`, batas fitur karakter 40.000.
- Logistic Regression C=4, liblinear, seed tetap.
- Bobot contoh berbanding terbalik dengan jumlah **pesan asli** tiap pasangan bahasa/label; contoh sintetis dikalikan 0,1, lalu seluruh bobot dinormalisasi. Data/split tidak ditambah atau diubah.
- Threshold memaksimalkan macro F2 validation dengan FPR ≤3% pada setiap kelompok layak (≥100 ham/≥30 spam), memakai selector yang sudah ada.
- Ranking kandidat berdasarkan macro recall, recall bahasa terendah, macro F2, FP lebih rendah, lalu nama deterministik. **B–D tidak dibaca untuk training, pemilihan fitur/model, atau threshold.** Seluruh contoh manual tetap di luar dataset.

Ablation validation mendukung arah perbaikan: pada C=4 dan bobot sintetis 0,1, recall Inggris/Indonesia berubah dari 90,74%/95,76% tanpa balancing menjadi 94,82%/98,31% dengan balancing; perluasan fitur menjadi 95,64%/98,31%. Perubahan dilakukan bersama dalam kandidat akhir; ini bukan bukti bahwa satu faktor saja menjelaskan seluruh peningkatan.

## Validation kandidat terpilih versus model sebelumnya

| Kelompok | Ham/spam | Model | FP | FN | FPR | Recall |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| SMS Inggris | 1.289/367 | Sebelum | 2 | 41 | 0,16% | 88,83% |
| SMS Inggris | 1.289/367 | Kandidat | 6 | 16 | 0,47% | 95,64% |
| SMS Indonesia | 114/118 | Sebelum | 3 | 8 | 2,63% | 93,22% |
| SMS Indonesia | 114/118 | Kandidat | 3 | 2 | 2,63% | 98,31% |

Selection lock disimpan sebelum evaluasi test di `reports/sms_recall/validation.json`. Seluruh varian, spesifikasi, threshold, support dan metrik tersimpan, bukan hanya pemenang.

## Perbandingan pada data uji identik

Baseline = SVM sebelum precision refinement; sebelum = LR model aktif saat permintaan ini dimulai; sesudah = kandidat yang dipilih hanya dari validation.

| Kelompok/cohort | Ham/spam | Model | FP | FN | FPR | Recall |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Indonesia historis | 113/110 | Baseline | 11 | 0 | 9,73% | 100% |
| Indonesia historis | 113/110 | Sebelum | 0 | 10 | 0% | 90,91% |
| Indonesia historis | 113/110 | Sesudah | 1 | 3 | 0,88% | 97,27% |
| Inggris historis | 1.000/206 | Baseline | 4 | 7 | 0,40% | 96,60% |
| Inggris historis | 1.000/206 | Sebelum | 0 | 45 | 0% | 78,16% |
| Inggris historis | 1.000/206 | Sesudah | 1 | 17 | 0,10% | 91,75% |
| Inggris/Hinglish eksternal | 506/290 | Baseline | 45 | 20 | 8,89% | 93,10% |
| Inggris/Hinglish eksternal | 506/290 | Sebelum | 0 | 18 | 0% | 93,79% |
| Inggris/Hinglish eksternal | 506/290 | Sesudah | 1 | 2 | 0,20% | 99,31% |

Gate promosi dideklarasikan sebelum training: cohort/support sama, FPR setiap cohort ≤3%, tidak menambah FN pada cohort mana pun versus model aktif sebelumnya, serta menurunkan FN setidaknya pada satu cohort. **Hanya kandidat validation terpilih** diuji untuk keputusan promosi; tidak diganti dengan runner-up berdasarkan hasil test/manual. Gate diterima. Penambahan FP dari nol menjadi satu per cohort merupakan trade-off eksplisit, bukan klaim FP tetap nol.

Baseline masih memiliki recall historis lebih tinggi, terutama Inggris, tetapi memiliki lebih banyak FP. Pemulihan belum sempurna: 3 spam Indonesia dan 17 spam Inggris historis masih lolos.

### Batas holdout Indonesia

Tidak ada holdout Indonesia asli baru yang berhasil diverifikasi dalam pencarian terbatas ini. Mirror `penjelasan.txt` mengidentifikasi Rahmi/Wibisono 2016, versi Agustus 2018, 1.143 pesan, CC BY-SA 4.0; ini bukan sumber asli baru.[31]

Kartu dataset campuran mengidentifikasi sumber UCI dan `agtbaskara/dataset_sms_spam_v2`, sedangkan kartu Kaggle SMS Indonesia menyebut fork Yudi; keduanya tidak diterima sebagai bukti holdout independen baru.[29][32]

Pencarian juga memunculkan artikel tentang 4.125 SMS; jumlah tersebut berasal dari hasil pencarian/artikel, bukan hitungan korpus yang diunduh. Unduhan korpus asli beserta izin datanya belum diverifikasi, sehingga tidak digunakan.[33]

Bukti dan keputusan di `reports/sms_recall/source_research.json`. Ini bukan klaim bahwa korpus lain tidak ada. Test Indonesia **historis dan sudah pernah diperiksa**. Cohort Inggris/Hinglish eksternal juga sudah diperiksa pada putaran sebelumnya dan berbagi sumber dengan training/validation, bukan unseen-source/blind baru. Group split dan checksum frozen sebelumnya tetap dipertahankan. Tidak dibuat holdout sintetis untuk menggantikan evaluasi asli.

## Diagnosis manual dan smoke test sesudah seleksi

| Contoh | Sebelum | Sesudah | Skor baru |
| --- | --- | --- | ---: |
| Spam B | HAM, salah | SPAM | 0,7996 |
| Spam C | HAM, salah | SPAM | 0,7781 |
| Spam D | HAM, salah | SPAM | 0,7935 |
| Pesanan SMS Indonesia | HAM | HAM | 0,2192 |
| Refund SMS Indonesia | HAM | HAM | 0,5058 |
| Pesanan SMS Inggris | HAM | HAM | 0,4070 |
| Newsletter SMS Inggris berlangganan | HAM | **SPAM, salah** | 0,6924 |

A juga tetap spam. **Regresi newsletter sah tidak disembunyikan.** Hasil diagnosis tidak dipakai memilih ulang model/threshold atau menambahkan whitelist. Tidak ada benchmark consent/newsletter representatif untuk menyimpulkan FPR kategori produksi. Metrik test yang mendukung promosi bukan jaminan aman memblokir otomatis.

Tiga assertion B/C/D tidak diubah; SHA-256 file `tests/test_real_artifact.py` diperiksa sama dengan sebelum eksperimen. Setelah pemasangan, full suite **90 tes lulus**, pytest exit 0; `pip check` bersih. Suite hijau tidak mencakup klaim bahwa semua kontrol ham lolos: newsletter tetap dicatat salah dalam diagnosis terpisah.

Streamlit direstart dan smoke test browser menguji 8 SMS. Label/skor semua cocok dengan inferensi langsung dan hash artefak baru terlihat di UI. Klik awal sempat membaca hasil input sebelumnya karena rerun Streamlit belum selesai; pengujian diulang setelah menunggu commit/rerun dan membandingkan setiap hasil, tanpa perubahan model. Server localhost port **8501**, root dan health diperiksa. Bukti browser `reports/sms_recall/browser_sms.json`, diagnosis `reports/sms_recall/diagnosis.json`, JUnit `reports/sms_recall/tests.xml`, screenshot `reports/screenshots/sms_recall_*.png`.

Email pipeline, threshold, nama, metadata historis email, dan skor diagnosis tidak berubah; ini diverifikasi dengan fingerprint joblib dan perbandingan laporan, bukan hanya asumsi karena tidak dilatih ulang. Perubahan `app.py` hanya mengganti pemberitahuan evaluasi SMS yang sudah kedaluwarsa; perilaku deteksi email tidak diubah.

## Reproduksi dan batas kerja

Dari checkout tanpa hasil putaran recall (dengan artefak precision dan frozen dataset tersedia):

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/recover_sms_recall.py train
.\.venv\Scripts\python.exe -X utf8 scripts/recover_sms_recall.py evaluate
.\.venv\Scripts\python.exe -X utf8 scripts/recover_sms_recall.py promote
.\.venv\Scripts\python.exe -m pytest tests/ -q
```

`train` menolak putaran yang sudah memiliki `reports/sms_recall/plan.json`; `promote` menolak bila hash artefak aktif berbeda dari snapshot atau gate gagal. Jangan menghapus lock/snapshot untuk menjalankan ulang pencarian pada test yang sama. Startup biasa tetap perintah Streamlit di README, tanpa training ulang. Reproduksi seluruh putaran pada mesin/checkout kosong belum dilakukan.

Berhenti setelah kandidat terpilih dievaluasi, dipasang secara bersyarat, dan diverifikasi. Langkah lanjutan membutuhkan holdout Indonesia baru yang asli dan berizin serta data newsletter consent-aware; bukan tuning tambahan ke B–D atau ke satu newsletter manual.

## Sources

[29] https://www.kaggle.com/datasets/crystalbaby/indonesian-english-sms-spamham-dataset
[31] https://raw.githubusercontent.com/Andikazidanef15/Sentiment-Analysis-on-Indonesian-SMS-Dataset/main/penjelasan.txt
[32] https://www.kaggle.com/datasets/gevabriel/indonesian-sms-spam
[33] http://www.icicelb.org/ellb/contents/2022/10/elb-13-10-12.pdf

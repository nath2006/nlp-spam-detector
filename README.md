 NLP Spam Detector

Aplikasi web Python berbasis **Streamlit** untuk mendeteksi spam pada SMS, menyiapkan alur deteksi email, dan memproses pesan melalui CSV. Proyek ini dikembangkan dari [nlp-spam-detector](https://github.com/nath2006/nlp-spam-detector) untuk pembelajaran Natural Language Processing (NLP).

UI, inferensi, kurasi dataset, dan training dipisahkan. Aplikasi menggunakan artefak model lokal; **tidak mengunduh dataset atau melatih model saat startup**.

## Status proyek

**PARTIAL** — model SMS Inggris tersedia pada lingkungan pengembangan, tetapi target dua kanal dan dua bahasa belum sepenuhnya terpenuhi.

| Kanal / bahasa | Status |
| --- | --- |
| SMS Inggris | Model dilatih dan dievaluasi menggunakan data asli UCI |
| SMS Indonesia | Input dapat diproses oleh model SMS yang sama, tetapi kinerjanya belum tervalidasi dan belum dilatih dengan korpus Indonesia |
| Email Inggris | Form dan validasi tersedia; model email belum tersedia |
| Email Indonesia | Form dan validasi tersedia; model email dan evaluasi data asli belum tersedia |

Aplikasi tidak menggunakan model SMS sebagai pengganti model email. Jika model kanal tidak tersedia, UI menampilkan pesan yang jelas, bukan prediksi buatan.

## Fitur

### 1. Deteksi SMS

- Input teks SMS dengan hasil `SPAM` atau `BUKAN SPAM`.
- Skor model dan ambang keputusan.
- Penjelasan kontribusi fitur linear yang berasal dari model.
- Contoh sintetis Bahasa Indonesia dan Inggris untuk mencoba antarmuka.
- Unggah screenshot melalui OCR opsional. Dependensi OCR hanya dimuat ketika pengguna menekan **Ekstrak & deteksi**; kegagalan OCR tidak memblokir deteksi teks.

### 2. Deteksi Email

- Subjek opsional dan isi email wajib.
- Pemilihan model berdasarkan kanal, tanpa fallback ke model SMS.
- Contoh sintetis Indonesia dan Inggris.
- Isi diproses sebagai teks, bukan HTML aktif; aplikasi tidak otomatis membuka tautan atau mengambil gambar dari email.
- Pada versi ini, prediksi email menampilkan status model belum tersedia.

### 3. Deteksi Massal

- Unggah CSV UTF-8 dengan kolom `channel`, `body`, dan `subject` opsional.
- Unduh template sintetis atau gunakan template langsung dari UI.
- Hasil prediksi dan alasan kegagalan per baris.
- Unduh hasil tanpa menyertakan isi dan subjek pesan.

### 4. Evaluasi Model

- Menampilkan sumber dataset, komposisi kelas, precision, recall, dan F1 dari laporan evaluasi offline.
- Menandai kelompok yang belum tervalidasi.
- Menyediakan laporan lengkap, termasuk confusion matrix.

## Stack dan persyaratan

- Python; lingkungan pengembangan telah menjalankan tes menggunakan Python 3.11.
- Streamlit, scikit-learn, pandas, NumPy, dan joblib.
- Beautiful Soup untuk parsing HTML email pada pipeline kurasi.
- EasyOCR dan OpenCV sebagai dependensi opsional.
- pytest dan Streamlit AppTest untuk pengujian.

Perintah berikut dijalankan dari root proyek pada Linux/macOS. Pada sistem yang hanya menyediakan `python3`, gunakan `python3` untuk membuat virtual environment.

## Setup

```bash
cd ~/documents/kuliah/nlp-spam-detector
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Alternatif jika menggunakan `uv`:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Berkas lama `requiretments.txt` dipertahankan untuk kompatibilitas, tetapi instalasi baru sebaiknya memakai `requirements.txt`.

### OCR opsional

Instal hanya jika memerlukan ekstraksi screenshot:

```bash
python -m pip install -r requirements-ocr.txt
```

EasyOCR dapat mengunduh bobot model pada penggunaan pertama dan membutuhkan koneksi internet serta ruang disk tambahan. Tanpa dependensi/bobot OCR, deteksi teks tetap dapat digunakan selama artefak model SMS tersedia.

## Menjalankan aplikasi

Aktifkan `.venv`, lalu jalankan:

```bash
python -m streamlit run app.py --server.address 127.0.0.1
```

Buka **http://127.0.0.1:8501**. Jika port tersebut sudah dipakai:

```bash
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502
```

Pilih halaman melalui sidebar. Artefak default berada di:

```text
artifacts/model.joblib
```

Direktori `artifacts/` diabaikan Git, sehingga hasil clone baru tidak otomatis memiliki model. Ikuti langkah persiapan dataset dan training di bawah jika artefak belum tersedia.

Untuk memakai lokasi artefak lain:

```bash
export SPAM_MODEL_PATH=/lokasi/tepercaya/model.joblib
python -m streamlit run app.py --server.address 127.0.0.1
```

**Keamanan:** muat hanya artefak joblib yang dibuat sendiri atau berasal dari sumber tepercaya. Deserialisasi joblib dapat menjalankan kode; jangan memakai unggahan model dari pihak yang tidak dipercaya.

## Contoh penggunaan

### SMS

1. Buka **Deteksi SMS**.
2. Masukkan isi pesan atau pilih contoh lalu tekan **Isi contoh**.
3. Opsional: aktifkan kontribusi fitur.
4. Tekan **Deteksi**.

Contoh buatan:

```text
Indonesia: Besok kita bertemu di perpustakaan pukul sembilan.
English: Congratulations! Claim your fictional prize at https://example.com/claim.
```

Contoh hanya untuk mencoba alur, bukan data evaluasi atau jaminan label tertentu. Model saat ini dilatih pada SMS Inggris, sehingga keluaran untuk Indonesia tidak boleh dianggap tervalidasi.

### Email

Buka **Deteksi Email**, isi subjek jika diperlukan, lalu isi badan email dan tekan **Deteksi**. Isi tetap wajib meskipun subjek sudah diisi. Jika artefak tidak memiliki model email, aplikasi menampilkan status belum tersedia.

### CSV

Template lengkap: [examples/batch_template.csv](examples/batch_template.csv).

```csv
channel,body,subject
sms,"Besok kita bertemu di kampus.",
sms,"Please call me after class.",
email,"Please review the attached meeting agenda.","Meeting agenda"
email,"Terima kasih, jadwal rapat sudah saya terima.",
```

Ketentuan:

| Input | Batas / aturan |
| --- | --- |
| `channel` | Nilai persis `sms` atau `email` |
| `body` | Teks wajib, tidak boleh kosong atau hanya whitespace |
| SMS | Maksimal 10.000 karakter |
| Email | Maksimal 100.000 karakter isi |
| `subject` | Opsional, maksimal 1.000 karakter |
| CSV | UTF-8, maksimal 2 MiB dan 1.000 baris data |
| Screenshot OCR | Maksimal 10 MiB dan 20 juta piksel |

Gunakan tanda kutip CSV jika teks berisi koma atau baris baru. Baris dengan kanal tidak valid, isi kosong, panjang berlebih, jumlah kolom salah, atau model kanal yang tidak tersedia menghasilkan alasan kegagalan per baris. CSV yang rusak secara keseluruhan ditolak.

Hasil unduhan memuat `row`, `label`, `display`, `score_model`, `threshold`, dan `error`. Nomor `row` dimulai dari 2 karena baris pertama adalah header. Nilai string berisiko formula spreadsheet dinetralkan pada ekspor.

## Dataset dan izin penggunaan

Inventaris dan catatan riset:

- [Riset dataset Indonesia](docs/dataset_research_id.md)
- [Tindak lanjut sumber Indonesia](docs/dataset_research_id_followup.md)
- [Riset dan eksekusi dataset Inggris](docs/dataset_research_en.md)
- Metadata: [candidates_id.json](data/candidates_id.json), [candidates_id_followup.json](data/candidates_id_followup.json), dan [candidates_en.json](data/candidates_en.json)

| Kelompok | Sumber / kandidat | Keputusan |
| --- | --- | --- |
| SMS Inggris | UCI SMS Spam Collection, Almeida & Hidalgo, DOI [10.24432/C5CC84](https://doi.org/10.24432/C5CC84) | Dipakai; sumber resmi menyatakan CC BY 4.0 |
| SMS Indonesia | Dataset Yudi Wibisono / gist agtbaskara dan alternatif terkait | Tidak dipakai; varian/versi izin sumber asli belum terverifikasi lengkap |
| Email Inggris | SpamAssassin dan kandidat alternatif dalam inventaris | Tidak dipakai; akses/verifikasi korpus belum terselesaikan |
| Email Indonesia | Kandidat paper Vernanda dkk., UII, dan MDP | Tidak dipakai; berkas, asal, label, atau izin data belum terverifikasi lengkap |

Lisensi artikel, kode repositori, atau mirror tidak otomatis menjadi lisensi dataset asli. Angka yang dikutip dari paper dibedakan dari jumlah yang dihitung langsung dari berkas. Tidak ada SMS yang diberi ulang kanal email untuk mengisi kekurangan data. Data sintetis pada contoh/fixture tidak dipakai sebagai test set kualitas model.

### Menyiapkan SMS UCI

Sumber resmi: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), lisensi [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

```bash
mkdir -p data/raw
curl --fail --location --max-time 120 \
  'https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip' \
  --output data/raw/uci_sms.zip
python scripts/prepare_data.py
```

Jika `data/raw/uci_sms.zip` sudah tersedia, tidak perlu mengunduh ulang. Skrip kurasi membaca arsip lokal dan tidak mengakses jaringan.

SHA-256 arsip yang dipakai pada laporan saat ini:

```text
1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3
```

Verifikasi lokal bila diperlukan:

```bash
sha256sum data/raw/uci_sms.zip
```

Pertahankan atribusi Almeida & Hidalgo, tautan sumber/DOI, lisensi, dan catatan perubahan berupa deduplikasi, grouping, serta pembagian data. Data mentah dan terkurasi diabaikan Git karena potensi informasi pribadi; jangan mempublikasikannya tanpa peninjauan izin dan privasi.

### Kurasi dan pembagian data

Skema utama: `channel`, `language`, `label`, `subject`, `body`, dan `source_id`. Metadata tambahan mencakup identitas baris, grup, dan split. Label UCI dipetakan eksplisit: `ham → ham`, `spam → spam`.

Pipeline melakukan:

1. Normalisasi teks untuk deduplikasi exact dan penghapusan kelompok label konflik.
2. Grouping template dengan normalisasi URL, alamat email, dan angka.
3. Grouping near-duplicate menggunakan cosine karakter ≥ 0,90 untuk teks minimal 40 karakter, serta pengelompokan thread ketika metadata tersedia.
4. Pembagian grup deterministik sekitar 60% training, 20% validation, dan 20% test dengan seed `2026`.

Vectorizer grouping hanya digunakan untuk membentuk kelompok sebelum split, bukan sebagai fitur classifier. Preprocessing dan classifier untuk prediksi di-fit pada training saja. Grouping tetap heuristik, bukan jaminan seluruh duplikasi semantik telah ditemukan.

Hasil kurasi yang tercatat di [reports/curation.json](reports/curation.json):

| Tahap | Ham | Spam |
| --- | ---: | ---: |
| Mentah | 4.827 | 747 |
| Lolos kurasi | 4.517 | 642 |
| Training | 2.710 | 385 |
| Validation | 903 | 129 |
| Test | 904 | 128 |

Sebanyak 415 duplikasi exact dibuang. Data terkurasi disimpan lokal pada `data/processed/curated.jsonl`.

**Batas implementasi:** `prepare_data.py` saat ini hanya memasukkan UCI SMS. Menaruh arsip email atau Indonesia di `data/raw/` tidak otomatis mengikutsertakannya; perlu adapter sumber, verifikasi izin, audit label/bahasa, dan integrasi kurasi terlebih dahulu.

## Training dan evaluasi

```bash
source .venv/bin/activate
python scripts/prepare_data.py
python scripts/train.py
```

Training sekaligus membandingkan model, memilih threshold dari validation, mengevaluasi test set, dan menyimpan:

- `artifacts/model.joblib` — pipeline dan threshold per kanal yang benar-benar dilatih.
- `reports/evaluation.json` — hasil eksperimen, metrik, confusion matrix, serta keterbatasan.
- `reports/curation.json` — komposisi sumber, audit kurasi, dan identitas split.

Tidak ada perintah evaluasi terpisah yang diperlukan: `scripts/train.py` menjalankan training dan evaluasi offline dalam satu alur. Eksekusi ulang akan memperbarui artefak dan laporan tersebut.

Model pembanding:

- **Baseline algoritmik:** pembersihan URL/angka/tanda baca + TF-IDF kata + Multinomial Naive Bayes.
- TF-IDF kata + Logistic Regression.
- **Model terpilih:** TF-IDF kata dan karakter + Logistic Regression.

Model dan threshold dipilih dari validation dengan memaksimalkan F0.5, dengan batas false-positive rate ham empiris ≤ 1%. Test set tidak dipakai untuk memilih model. Batas validation ini bukan jaminan false-positive rate di produksi.

### Hasil test SMS Inggris

Angka berikut berasal dari `reports/evaluation.json`, dengan spam sebagai kelas positif. Seluruh model memakai test set yang sama: **904 ham dan 128 spam**.

| Model | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Baseline Naive Bayes, threshold validation | 0,9896 | 0,7422 | 0,8482 |
| TF-IDF kata + Logistic Regression | 0,9907 | 0,8359 | 0,9068 |
| TF-IDF kata + karakter + Logistic Regression | **0,9910** | **0,8594** | **0,9205** |

Threshold model terpilih: `0.4432732053189189`.

Confusion matrix model terpilih, urutan aktual/prediksi `[ham, spam]`:

```text
                  Prediksi ham   Prediksi spam
Aktual ham                 903               1
Aktual spam                 18             110
```

Baseline di atas mereproduksi pendekatan algoritmik pada split pembanding baru; **bukan** bukti akurasi aplikasi awal pada dataset Indonesia. Startup baseline sebelumnya gagal karena dependensi `cv2` tidak tersedia.

SMS Indonesia, email Inggris, dan email Indonesia belum memiliki hasil evaluasi yang layak pada versi ini. Tidak ada klaim akurasi 98%, validasi lintas sumber, kalibrasi probabilitas, atau keunggulan model gabungan SMS/email. Laporan mencatat satu run training yang berhasil; pengulangan penuh untuk membuktikan reproduksibilitas hasil dua run belum terverifikasi.

### Makna skor dan penjelasan

Skor model berasal dari keluaran classifier pada rentang 0–1, tetapi **belum dikalibrasi sebagai probabilitas**. Skor bukan persentase kepastian, akurasi, maupun bukti bahwa pesan aman.

Penjelasan linear menghitung kontribusi fitur dari TF-IDF dan koefisien classifier, bukan daftar kata kunci buatan. Kontribusi positif mendukung spam dan negatif mendukung ham. Penjelasan tidak membuktikan niat pengirim.

## Menggunakan inferensi dari Python

```python
from spamdet.inference import ModelError, predict, predict_csv

result = predict("sms", "Please call me after class.", explain=True)
print(result["display"], result["score_model"])

rows = predict_csv("channel,body\nsms,Please call me after class.\n")
print(rows)

try:
    result = predict("email", "Please review the meeting agenda.", subject="Meeting")
except ModelError as exc:
    print(exc)  # Model email belum tersedia pada artefak saat ini.
```

`predict` juga menerima `artifact_path` jika lokasi model perlu ditentukan secara eksplisit. Input tidak valid menghasilkan `ValueError`; model hilang/rusak/tidak tersedia menghasilkan `ModelError`.

## Pengujian

```bash
python -m pytest tests/ -q
```

Tes terarah:

```bash
python -m pytest tests/test_curation.py tests/test_training.py -q
python -m pytest tests/test_inference.py tests/test_ocr.py -q
python -m pytest tests/test_app.py -q
```

Verifikasi saat pembaruan README: **55 tes lulus** melalui `.venv/bin/python -m pytest tests/ -q`.

Tes mencakup kurasi/split, logika training, validasi SMS/email/CSV, penjelasan fitur, penanganan OCR, dan alur UI melalui Streamlit AppTest. Tes yang menggunakan model fixture sintetis memverifikasi integrasi dan perilaku aplikasi, **bukan** kualitas deteksi pada pesan asli. Kualitas model pada data asli dilaporkan terpisah di `reports/evaluation.json`.

AppTest bukan pengganti pengujian layout desktop/HP di browser. README ini tidak mengklaim smoke test browser, keberhasilan OCR dengan bobot asli, atau uptime server hanya berdasarkan kelulusan tes unit/AppTest.

## Struktur proyek

```text
app.py                          Entry point dan navigasi Streamlit
spamdet/
  inference.py                  Validasi, prediksi, penjelasan, dan CSV
  ocr.py                        Ekstraksi screenshot opsional
  curation.py                   Parsing sumber, deduplikasi, grouping, dan split
  training.py                   Perbandingan model, threshold, dan evaluasi
scripts/
  prepare_data.py                Kurasi dari dataset lokal
  train.py                       Training dan evaluasi offline
examples/batch_template.csv      Contoh sintetis aman dibagikan
data/
  candidates_*.json              Inventaris kandidat dataset
  raw/                          Arsip lokal; diabaikan Git
  processed/                    Data terkurasi; diabaikan Git
artifacts/model.joblib          Model lokal; diabaikan Git
reports/                        Statistik kurasi dan hasil evaluasi
```

Dokumen riset berada di `docs/`; seluruh tes berada di `tests/`. Tidak diperlukan frontend JavaScript terpisah atau API eksternal saat inferensi.

## Privasi dan keterbatasan

- Pesan pengguna diproses di memori sesi; aplikasi tidak secara default menulis isi pesan ke disk atau log. Widget dapat mempertahankan nilai selama sesi aktif.
- Hasil ekspor CSV tidak memuat isi/subjek pesan, tetapi berkas unduhan yang disimpan pengguna tetap perlu dijaga.
- EasyOCR dapat menyimpan bobot model pada cache dependensinya; ini berbeda dari penyimpanan pesan/screenshot oleh aplikasi.
- Korpus UCI bersifat historis dan penetapan bahasa mengikuti sumber, termasuk ragam informal/campuran; hasilnya bukan bukti generalisasi ke semua bahasa atau spam modern.
- Riset email dan izin dataset Indonesia belum tuntas, termasuk tindakan jaringan yang ditolak lapisan persetujuan sebagaimana dicatat di dokumen riset.
- Model dapat salah menandai pesan sah atau melewatkan spam. Gunakan sebagai alat bantu, bukan pemblokir otomatis atau pengganti pemeriksaan keamanan.

## Pemecahan masalah

| Masalah | Tindakan |
| --- | --- |
| Model tidak ditemukan | Jalankan persiapan dataset dan training; periksa `artifacts/model.joblib` atau `SPAM_MODEL_PATH` |
| Model email belum tersedia | Ini batas versi saat ini; jangan menggantinya dengan model SMS |
| Artefak tidak kompatibel | Gunakan environment dependency yang sesuai lalu latih ulang; jangan memuat joblib sembarang |
| `data/raw/uci_sms.zip` tidak ditemukan | Unduh ZIP resmi UCI sesuai langkah persiapan dataset |
| OCR tidak tersedia atau unduhan bobot gagal | Instal dependency OCR/periksa koneksi; gunakan input teks sementara |
| CSV gagal divalidasi | Periksa UTF-8, header, quoting, nilai kanal, batas ukuran, dan pesan kesalahan |
| Port Streamlit terpakai | Pilih port lain melalui `--server.port` |

## Catatan lisensi

UCI SMS Spam Collection memiliki ketentuan data CC BY 4.0 sebagaimana dijelaskan di atas. Status lisensi kandidat lain tercantum pada inventaris dan tidak boleh diasumsikan sama. Repositori saat ini tidak memiliki berkas lisensi kode terpisah; README ini tidak menetapkan lisensi baru atas kode atau data pihak lain.

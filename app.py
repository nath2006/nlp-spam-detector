"""Streamlit UI; inference never trains, logs, or persists user content."""
import csv
import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from spamdet.inference import ModelError, export_csv, predict, predict_csv, model_status

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / 'examples' / 'batch_template.csv'
EVALUATION = ROOT / 'reports' / 'evaluation.json'
PAGES = ['Deteksi SMS', 'Deteksi Email', 'Deteksi Massal', 'Evaluasi Model']

st.set_page_config(page_title='Detektor Spam SMS & Email', layout='wide')
st.title('Detektor Spam SMS & Email')
st.warning('Email Indonesia belum tervalidasi. Cakupan SMS dan email mengikuti artefak serta laporan evaluasi lokal; bukan jaminan keamanan. Periksa pesan secara mandiri.')
st.warning('Pesan sah seperti notifikasi pesanan, refund, OTP, invoice, dan newsletter berlangganan dapat salah ditandai spam. Data transaksi sah masih terbatas; jangan gunakan untuk pemblokiran otomatis.')
st.caption('Pemrosesan pesan di memori sesi saja, tanpa log atau penyimpanan isi pesan. Skor model bukan ukuran kepastian atau akurasi.')
page = st.sidebar.radio('Navigasi', PAGES)
st.header(page)


def show_result(result):
    st.subheader(result['display'])
    a, b = st.columns(2)
    a.metric('Skor model', f"{result['score_model']:.4f}")
    b.metric('Ambang keputusan', f"{result['threshold']:.4f}")
    st.caption('Label SPAM bila skor model ≥ ambang keputusan. Skor ini belum tentu terkalibrasi.')
    explanation = result.get('explanation')
    if explanation:
        with st.expander('Penjelasan kontribusi fitur model'):
            if explanation.get('available'):
                st.caption('Kontribusi positif mendukung spam; negatif mendukung ham. Ini penjelasan model, bukan bukti niat pengirim.')
                st.json({key: value for key, value in explanation.items() if key != 'features'})
                st.dataframe(pd.DataFrame(explanation['features'], columns=['feature', 'contribution']), hide_index=True, width='stretch')
            else:
                st.text(explanation['reason'])


def detect(channel, body, subject='', explain=False):
    try:
        show_result(predict(channel, body, subject, explain=explain))
    except (ValueError, ModelError) as exc:
        st.error(str(exc))


def examples(channel):
    with TEMPLATE.open(encoding='utf-8', newline='') as handle:
        return [row for row in csv.DictReader(handle) if row['channel'] == channel]


def fill_example(channel, rows):
    chosen = rows[st.session_state[channel + '_example']]
    st.session_state[channel + '_body'] = chosen['body']
    if channel == 'email':
        st.session_state['email_subject'] = chosen['subject']


if page in PAGES[:2]:
    channel = 'sms' if page == PAGES[0] else 'email'
    status = model_status(channel)
    st.info(status['message'])
    if status['available']:
        st.caption(f"Artefak: {status['artifact_path']} · SHA-256: {status['artifact_sha256']}")
        st.caption(f"Kelas: {status['classes']} · Indeks spam: {status['spam_index']} · Threshold: {status['threshold']:.6f} · Bahasa test: {', '.join(status['evaluated_languages']) or 'tidak tercatat'}")
    sample_rows = examples(channel)
    st.caption('Contoh sintetis untuk mencoba antarmuka, bukan data evaluasi atau jaminan hasil.')
    labels = ['Indonesia · ham', 'Indonesia · spam', 'English · ham', 'English · spam']
    st.selectbox('Pilih contoh sintetis', range(4), format_func=lambda index: labels[index], key=channel+'_example')
    st.button('Isi contoh', key=channel+'_fill', on_click=fill_example, args=(channel, sample_rows))
    subject = st.text_input('Subjek (opsional, maksimal 1000 karakter)', key='email_subject') if channel == 'email' else ''
    body = st.text_area('Isi pesan (wajib)', key=channel+'_body', height=180, help='Maksimal 10.000 karakter SMS / 100.000 karakter email.')
    explain = st.checkbox('Tampilkan kontribusi fitur', key=channel+'_explain')
    if st.button('Deteksi', key=channel+'_detect', type='primary'):
        detect(channel, body, subject, explain)
    if channel == 'sms':
        with st.expander('Screenshot SMS (OCR opsional)'):
            st.caption('Maksimal 10 MiB dan 20 juta piksel. OCR baru dimuat setelah tombol ditekan; pertama kali dapat mengunduh model.')
            image = st.file_uploader('Unggah screenshot', type=['png','jpg','jpeg','webp'], key='ocr_upload', max_upload_size=10)
            if st.button('Ekstrak & deteksi', key='ocr_detect'):
                if image is None:
                    st.error('Pilih gambar terlebih dahulu.')
                else:
                    from spamdet.ocr import extract_text, OCRError
                    try:
                        with st.spinner('Memproses OCR opsional...'):
                            extracted = extract_text(image.getvalue())
                        st.code(extracted, language='text')
                        detect('sms', extracted, explain=explain)
                    except OCRError as exc:
                        st.error(str(exc))
elif page == PAGES[2]:
    st.caption('CSV UTF-8 ≤ 2 MiB, maksimal 1000 baris data. Header channel,body wajib; subject opsional. Hasil tidak menyertakan isi/subjek pesan.')
    template = TEMPLATE.read_bytes()
    st.download_button('Unduh template CSV sintetis', data=template, file_name='batch_template.csv', mime='text/csv')
    uploaded = st.file_uploader('Unggah CSV', type=['csv'], key='batch_upload', max_upload_size=2)
    use_example = st.button('Gunakan template contoh', key='batch_example')
    run = st.button('Proses CSV', key='batch_detect', type='primary')
    if use_example or run:
        if not use_example and uploaded is None:
            st.error('Unggah CSV terlebih dahulu atau gunakan template contoh.')
        else:
            try:
                results = predict_csv(template if use_example else uploaded.getvalue())
                st.dataframe(pd.DataFrame(results), hide_index=True, width='stretch')
                st.download_button('Unduh hasil CSV', export_csv(results), 'hasil_deteksi.csv', 'text/csv')
            except (ValueError, ModelError) as exc:
                st.error(str(exc))
else:
    st.caption('Metrik di bawah dibaca dari laporan evaluasi offline, bukan dihitung dari pesan pengguna. Tidak ada pelatihan saat aplikasi dibuka.')
    try:
        report = json.loads(EVALUATION.read_text(encoding='utf-8'))
    except FileNotFoundError:
        st.info('Laporan evaluasi belum tersedia. Jalankan evaluasi offline untuk membuat reports/evaluation.json.')
    except (OSError, ValueError):
        st.error('Laporan evaluasi tidak dapat dibaca atau format JSON tidak valid.')
    else:
        refined = report.get('schema_version', 0) >= 3
        st.subheader('Test historis — bukan holdout baru' if refined else 'Kinerja pada test set asli')
        if refined:
            sms_notice = ('Recall SMS ditingkatkan; B, C, D diperiksa pada laporan diagnosis terpisah. Contoh newsletter SMS sah masih salah ditandai spam. '
                          if report.get('sms_recall') else 'Trade-off precision: contoh spam manual B, C, D kini lolos sebagai ham. ')
            st.warning(sms_notice + 'Refund dan newsletter email sah masih salah ditandai spam. Model belum layak untuk pemblokiran otomatis.')
            st.caption('Test historis sudah diperiksa dalam pengembangan sebelumnya. Angka ini bukan evaluasi blind baru; subset dapat berkurang karena penggabungan grup dan karantina konflik antar-split.')
        measured = report.get('by_channel_language', {})
        groups = {'sms_en':'SMS Inggris', 'sms_id':'SMS Indonesia',
                  'email_en':'Email Inggris', 'email_id':'Email Indonesia'}
        summary = []
        for key, name in groups.items():
            result = measured.get(key)
            support = result.get('support', {}) if result else {}
            count = sum(support.values())
            summary.append({'Kelompok':name,
                'Status': ('Dievaluasi (sampel kecil)' if min(support.values(), default=0) < 100 else 'Dievaluasi') if result else 'Belum tervalidasi',
                'Jumlah uji':count, 'Ham':support.get('ham',0), 'Spam':support.get('spam',0),
                'Precision spam':result.get('precision') if result else None,
                'Recall spam':result.get('recall') if result else None,
                'F1 spam':result.get('f1') if result else None,
                'FPR ham':result.get('false_positive_rate') if result else None,
                'False positive':result.get('false_positives') if result else None,
                'False negative':result.get('false_negatives') if result else None})
        st.dataframe(pd.DataFrame(summary), hide_index=True, width='stretch')
        st.caption('Metrik di atas memakai spam sebagai kelas positif; nol jumlah uji berarti belum ada evaluasi yang layak, bukan akurasi nol.')
        if refined:
            def comparison_table(comparisons):
                result_rows = []
                for model, values in comparisons.items():
                    for group, measured_group in values.items():
                        support = measured_group.get('support', {})
                        result_rows.append({'Model':model, 'Kelompok':group,
                            'Ham':support.get('ham',0), 'Spam':support.get('spam',0),
                            'Precision':measured_group.get('precision'), 'Recall':measured_group.get('recall'),
                            'F1':measured_group.get('f1'), 'FPR ham':measured_group.get('false_positive_rate'),
                            'FP':measured_group.get('false_positives'), 'FN':measured_group.get('false_negatives'),
                            'Layak constraint validation':measured_group.get('constraint_eligible')})
                return pd.DataFrame(result_rows)
            st.subheader('Holdout eksternal baru — SMS Inggris/Hinglish')
            st.caption('Sumber India asli berlabel; bahasa merupakan penetapan tingkat sumber, bukan anotasi setiap pesan. Tidak memvalidasi SMS Indonesia atau email. Grup dibekukan sebelum tuning; overlap baseline dikeluarkan.')
            st.dataframe(comparison_table(report.get('new_holdout',{})), hide_index=True, width='stretch')
            st.subheader('Pemilihan pada validation per kelompok')
            validation = report.get('validation', {})
            st.text(validation.get('policy',''))
            st.caption('Satu threshold per kanal; tidak ada detektor bahasa atau routing threshold per bahasa. Target empiris FPR ≤3% hanya untuk kelompok dengan ≥100 ham dan ≥30 spam, bukan jaminan produksi.')
            st.dataframe(comparison_table({key:value.get('by_group',{}) for key,value in validation.get('candidates',{}).items()}), hide_index=True, width='stretch')
            st.json({'selected':report.get('selected'), 'frozen_sha256':report.get('frozen_sha256'), 'artifact_sha256':report.get('artifact_sha256')})
            st.subheader('Tambahan sumber precision')
            st.json(report.get('source_audit',{}))
        try:
            curated = json.loads((ROOT / 'reports' / 'curation.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            st.info('Laporan sumber kurasi belum tersedia.')
        else:
            st.subheader('Sumber dataset yang digunakan')
            for source, details in curated.get('sources', {}).items():
                st.text(f"{source}: {details.get('attribution', source)} — {details.get('license', 'izin tidak tercatat')}")
                st.json({'mentah':details.get('raw_counts',{}), 'lolos_kurasi':details.get('curated_counts',{})})
        missing = [name for key, name in groups.items() if not measured.get(key)]
        if missing:
            st.info('Belum tervalidasi: ' + ', '.join(missing) + '. Tidak ada fallback SMS untuk email.')
        st.caption('Evaluasi dalam sumber historis dan grouping heuristik tidak menjamin generalisasi ke email transaksi, pesan modern, atau sumber lain. Batas false-positive validation bukan batas produksi; data kecil tidak diklaim memenuhi target.')
        with st.expander('Laporan evaluasi lengkap dan confusion matrix'):
            st.json(report)

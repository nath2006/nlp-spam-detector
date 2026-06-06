import streamlit as st
import cv2
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import tempfile
import os
import easyocr
import sys

# ============ KONFIGURASI AWAL ============
# Inisialisasi EasyOCR (hanya sekali)
@st.cache_resource
def init_ocr():
    """Inisialisasi EasyOCR dengan support Bahasa Indonesia dan Inggris"""
    return easyocr.Reader(['id', 'en'], gpu=False)  # gpu=False untuk kompatibilitas

# ============ KONFIGURASI STREAMLIT ============
st.set_page_config(
    page_title="SMS Spam Detector - EasyOCR",
    layout="wide"
)

# Title
st.title("SMS Spam Detection dengan EasyOCR")
st.markdown("Aplikasi ini mendeteksi pesan SPAM dari teks atau gambar screenshot (tanpa install Tesseract!)")

# Sidebar
with st.sidebar:
    st.header("Tentang Aplikasi")
    st.info(
        """
        - Menggunakan **Naive Bayes** dengan TF-IDF
        - Dataset SMS Spam Collection
        - **OCR menggunakan EasyOCR** (support Bahasa Indonesia)
        - Akurasi model: ~98%
        """
    )
    
    st.header("⚙️ Pengaturan")
    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    ngram_range = st.selectbox("N-Gram Range", [(1,1), (1,2), (1,3)], index=1)
    
    st.header("📦 Library yang Digunakan")
    st.code("""
    - streamlit
    - easyocr (OCR)
    - scikit-learn
    - pandas
    - opencv-python
    """)

# ============ FUNGSI TRAINING MODEL ============
@st.cache_resource
def train_model(test_size_param, ngram_param):
    """Training model spam detection"""
    # Load dataset
    url = "https://gist.githubusercontent.com/agtbaskara/a1a7017027cc1df9d35cf06e1e5575b7/raw/59870e27ca217d77ac0d8d8dc100551c0dcd14b3/dataset_sms_spam_v2.csv"
    df = pd.read_csv(url)
    
    # Map labels
    def map_label(label):
        label = str(label).strip().lower()
        return "bukan_spam" if label == "normal" else "spam"
    
    df["target"] = df["label"].apply(map_label)
    
    # Clean text
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    df["clean_text"] = df["Teks"].apply(clean_text)
    
    # Split data
    X = df["clean_text"]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size_param, random_state=42, stratify=y
    )
    
    # Train model
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=ngram_param, min_df=2)),
        ("clf", MultinomialNB())
    ])
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy, y_test, y_pred, df

# ============ FUNGSI CLEANING TEXT ============
def clean_text(text):
    """Membersihkan teks dari karakter tidak perlu"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ============ FUNGSI OCR DENGAN EASYOCR ============
def extract_text_from_image(image_file, ocr_reader):
    """Ekstrak teks dari gambar menggunakan EasyOCR"""
    # Simpan file sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
        tmp_file.write(image_file.getvalue())
        tmp_path = tmp_file.name
    
    # Baca gambar dengan OpenCV
    img = cv2.imread(tmp_path)
    
    # Preprocessing untuk OCR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # OCR dengan EasyOCR
    results = ocr_reader.readtext(gray)
    
    # Gabungkan semua teks yang terdeteksi
    extracted_text = ' '.join([result[1] for result in results])
    
    # Hapus file sementara
    os.unlink(tmp_path)
    
    return extracted_text

# ============ FUNGSI DETEKSI KATA KUNCI ============
def detect_keywords(text):
    """Deteksi kata kunci spam dalam teks"""
    keywords = ["hadiah", "promo", "gratis", "uang", "bonus", "penawaran", 
                "menang", "klik", "transfer", "winner", "prize", "free", 
                "discount", "cash", "jackpot", "undian", "lotre"]
    detected = [kw for kw in keywords if kw in text.lower()]
    return detected

# ============ INITIALISASI ============
# Inisialisasi EasyOCR
with st.spinner("Memuat EasyOCR (pertama kali akan download model, butuh ~500MB)..."):
    try:
        ocr_reader = init_ocr()
        st.success("EasyOCR berhasil dimuat!")
    except Exception as e:
        st.error(f"Gagal memuat EasyOCR: {e}")
        st.info("Pastikan koneksi internet stabil untuk download model pertama kali")
        st.stop()

# Training model
with st.spinner("Melatih model spam detection..."):
    try:
        model, accuracy, y_test, y_pred, df = train_model(test_size, ngram_range)
        st.success(f"Model berhasil dilatih dengan akurasi: **{accuracy*100:.2f}%**")
    except Exception as e:
        st.error(f"Gagal training model: {e}")
        st.stop()

# ============ UI MAIN ============
# Tabs
tab1, tab2, tab3 = st.tabs(["Input Teks", "Upload Gambar", "Evaluasi Model"])

# ============ TAB 1: INPUT TEKS ============
with tab1:
    st.header("Deteksi Spam dari Teks")
    
    text_input = st.text_area(
        "Masukkan pesan SMS:",
        placeholder="Contoh: Selamat! Anda memenangkan hadiah Rp 100.000.000. Klik link untuk klaim...",
        height=150
    )
    
    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        detect_btn = st.button("Deteksi Spam", key="text_btn", type="primary")
    
    if detect_btn and text_input:
        with st.spinner("Menganalisis..."):
            clean_text_input = clean_text(text_input)
            
            # Prediksi
            prediction = model.predict([clean_text_input])[0]
            proba = model.predict_proba([clean_text_input])[0]
            
            spam_score = proba[list(model.classes_).index("spam")]
            non_spam_score = proba[list(model.classes_).index("bukan_spam")]
            
            # Hasil
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == "spam":
                    st.error(f"## **SPAM**")
                    st.metric("Confidence", f"{spam_score*100:.2f}%")
                else:
                    st.success(f"##**BUKAN SPAM**")
                    st.metric("Confidence", f"{non_spam_score*100:.2f}%")
            
            with col2:
                st.metric("Probabilitas Spam", f"{spam_score*100:.2f}%", 
                         delta="" if spam_score > 0.5 else "")
                st.metric("Probabilitas Bukan Spam", f"{non_spam_score*100:.2f}%")
            
            # Kata kunci
            detected_keywords = detect_keywords(clean_text_input)
            if detected_keywords:
                st.warning(f"Kata kunci spam terdeteksi: {', '.join(detected_keywords)}")
            
            # Detail
            with st.expander("📄 Lihat detail teks yang diproses"):
                st.code(f"Teks asli: {text_input}\n\nTeks bersih: {clean_text_input}")
    
    elif detect_btn and not text_input:
        st.warning("Silakan masukkan teks terlebih dahulu")

# ============ TAB 2: UPLOAD GAMBAR ============
with tab2:
    st.header("Deteksi Spam dari Gambar Screenshot")
    st.info("""
    **Tips untuk hasil OCR terbaik:**
    - Gambar harus jelas dan tidak buram
    - Kontras teks dengan background baik
    - Hindari gambar dengan banyak noise
    """)
    
    uploaded_file = st.file_uploader(
        "Pilih gambar screenshot SMS",
        type=['png', 'jpg', 'jpeg', 'webp'],
        help="Upload gambar yang berisi teks SMS"
    )
    
    if uploaded_file:
        # Preview gambar
        col_img1, col_img2 = st.columns([1, 1])
        
        with col_img1:
            st.image(uploaded_file, caption="Gambar yang diupload", use_container_width=True)
        
        with col_img2:
            if st.button("Ekstrak & Deteksi", key="image_btn", type="primary"):
                with st.spinner("Memproses OCR dengan EasyOCR..."):
                    try:
                        # Ekstrak teks
                        ocr_text = extract_text_from_image(uploaded_file, ocr_reader)
                        
                        if ocr_text and len(ocr_text.strip()) > 0:
                            st.success(f"Berhasil mengekstrak {len(ocr_text)} karakter")
                            
                            # Tampilkan hasil OCR
                            st.subheader("Hasil OCR:")
                            st.code(ocr_text, language="text")
                            
                            # Clean dan prediksi
                            clean_ocr = clean_text(ocr_text)
                            prediction = model.predict([clean_ocr])[0]
                            proba = model.predict_proba([clean_ocr])[0]
                            
                            spam_score = proba[list(model.classes_).index("spam")]
                            non_spam_score = proba[list(model.classes_).index("bukan_spam")]
                            
                            # Hasil deteksi
                            st.subheader("Hasil Deteksi:")
                            col_res1, col_res2 = st.columns(2)
                            
                            with col_res1:
                                if prediction == "spam":
                                    st.error(f"## **SPAM**")
                                    st.metric("Confidence", f"{spam_score*100:.2f}%")
                                else:
                                    st.success(f"## **BUKAN SPAM**")
                                    st.metric("Confidence", f"{non_spam_score*100:.2f}%")
                            
                            with col_res2:
                                st.metric("Probabilitas Spam", f"{spam_score*100:.2f}%")
                                st.metric("Probabilitas Bukan Spam", f"{non_spam_score*100:.2f}%")
                            
                            # Kata kunci
                            detected_keywords = detect_keywords(clean_ocr)
                            if detected_keywords:
                                st.warning(f"Kata kunci spam terdeteksi: {', '.join(detected_keywords)}")
                            
                            # Teks bersih
                            with st.expander("Lihat teks yang sudah dibersihkan"):
                                st.code(clean_ocr)
                        
                        else:
                            st.error("Tidak dapat mengekstrak teks dari gambar. Pastikan gambar jelas dan mengandung teks.")
                    
                    except Exception as e:
                        st.error(f"Error saat memproses OCR: {str(e)}")
                        st.info("Coba upload gambar dengan kualitas lebih baik")

# ============ TAB 3: EVALUASI MODEL ============
with tab3:
    st.header("Evaluasi Performa Model")
    
    # Metrik
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Total Data", len(df))
    with col_m2:
        st.metric("Jumlah Spam", (df["target"] == "spam").sum())
    with col_m3:
        st.metric("Jumlah Bukan Spam", (df["target"] == "bukan_spam").sum())
    with col_m4:
        st.metric("Akurasi Model", f"{accuracy*100:.2f}%")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred, labels=["bukan_spam", "spam"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Bukan Spam", "Spam"])
    disp.plot(ax=ax, cmap="Blues")
    st.pyplot(fig)
    
    # Interpretasi Confusion Matrix
    tn, fp, fn, tp = cm.ravel()
    col_int1, col_int2, col_int3, col_int4 = st.columns(4)
    with col_int1:
        st.info(f"True Negatives: {tn}")
    with col_int2:
        st.error(f"False Positives: {fp}")
    with col_int3:
        st.warning(f"False Negatives: {fn}")
    with col_int4:
        st.success(f"True Positives: {tp}")
    
    # Distribusi Dataset
    st.subheader("Distribusi Dataset")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    colors = ['green', 'red']
    df["target"].value_counts().plot(kind="bar", ax=ax2, color=colors)
    ax2.set_title("Distribusi Spam vs Bukan Spam")
    ax2.set_xlabel("Label")
    ax2.set_ylabel("Jumlah")
    ax2.set_xticklabels(['Bukan Spam', 'Spam'], rotation=0)
    st.pyplot(fig2)
    
    # Sample Data
    st.subheader("Sample Dataset (10 baris pertama)")
    st.dataframe(
        df[["Teks", "target"]].head(10),
        use_container_width=True,
        hide_index=False
    )
    
    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

# ============ FOOTER ============
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.caption("Built with Streamlit")
with col_f2:
    st.caption("Naive Bayes + TF-IDF")
with col_f3:
    st.caption("EasyOCR for image processing")

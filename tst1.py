import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import os

# =========================
# IMAGE PROCESSING FUNCTIONS
# =========================

def resize_image(img, width):
    h, w = img.shape[:2]
    ratio = width / w
    return cv2.resize(img, (width, int(h * ratio)))

def normalize_image(img):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

def median_denoising(img, k):
    return cv2.medianBlur(img, k)

def nlm_denoising(img, h):
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)

def dark_channel(img, size=15):
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return cv2.erode(min_channel, kernel)

def dehazing(img):
    img = img.astype(np.float64) / 255
    dark = dark_channel(img)
    A = np.max(img)
    t = 1 - 0.95 * dark / A
    t = np.clip(t, 0.1, 1)
    J = np.zeros_like(img)
    for i in range(3):
        J[:, :, i] = (img[:, :, i] - A) / t + A
    return np.clip(J * 255, 0, 255).astype(np.uint8)

def sharpening(img):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(img, -1, kernel)

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(2.0, (8,8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)

def contour_overlay(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = img.copy()
    cv2.drawContours(overlay, contours, -1, (0,255,0), 1)
    return overlay

def load_css(css_file):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(base_dir, css_file)

    if not os.path.exists(css_path):
        st.error(f"❌ File CSS tidak ditemukan di: {css_path}")
        return

    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# =========================
# QUALITY METRICS (SIMPLE)
# =========================

def blur_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def haze_score(img):
    return np.mean(dark_channel(img.astype(float)/255))

# =========================
# STREAMLIT UI
# =========================

st.set_page_config("Surveillance Image Enhancement", layout="wide")
load_css("jamrud.css")
st.title("📷 Optimasi Citra Pengawasan (Surveillance Image Enhancement)")
tab_app, tab_about = st.tabs(["🔧 Aplikasi", "ℹ️ About"])

#======
#tab app
#======
with tab_app:
    uploaded = st.file_uploader(
        "Upload gambar CCTV",
        type=["jpg", "png", "jpeg"]
    )
if uploaded:
    img = np.array(Image.open(uploaded).convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Sidebar Parameters
    st.sidebar.header("⚙️ Parameter")
    resize_w = st.sidebar.slider("Resize Width", 300, 1000, 600)
    denoise_method = st.sidebar.selectbox("Denoising Method", ["Median", "NLM"])
    median_k = st.sidebar.slider("Median Kernel", 3, 9, 3, step=2)
    nlm_h = st.sidebar.slider("NLM Strength", 5, 30, 10)

    use_dehaze = st.sidebar.checkbox("Dehazing", True)
    use_sharp = st.sidebar.checkbox("Deblurring", True)
    use_clahe = st.sidebar.checkbox("CLAHE", True)
    use_contour = st.sidebar.checkbox("Contour Overlay", True)
    


    # =========================
    # PIPELINE
    # =========================
    proc = resize_image(img, resize_w)
    proc = normalize_image(proc)

    if denoise_method == "Median":
        proc = median_denoising(proc, median_k)
    else:
        proc = nlm_denoising(proc, nlm_h)

    if use_dehaze:
        proc = dehazing(proc)
    if use_sharp:
        proc = sharpening(proc)
    if use_clahe:
        proc = apply_clahe(proc)

    overlay = contour_overlay(proc) if use_contour else proc

    # =========================
    # DISPLAY
    # =========================
    col1, col2, col3 = st.columns(3)
    col1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Before", use_container_width=True)
    col2.image(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB), caption="After", use_container_width=True)
    col3.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Contour Overlay", use_container_width=True)

    # =========================
    # METRICS
    # =========================
    st.subheader("📊 Penilaian Otomatis")
    st.write(f"**Blur Score:** {blur_score(proc):.2f}","(blur score adalah hasil dari tingkat ketidak jelasan dalam sebuah gambar.)")
    st.write(f"**Haze Score:** {haze_score(proc):.4f}","(ini merupakan hasil dari haze meter (pengukur kabut) dalam sebuah gambar.)")

    # =========================
    # HISTOGRAM
    # =========================
    st.subheader("📈 Histogram Intensitas")
    fig, ax = plt.subplots()
    ax.hist(cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY).ravel(), bins=256)
    st.pyplot(fig)
    
    # ================================
    # penjelasan fungsi dan juga hasil
    # ================================
    
    st.write()
    
    # =========================
    # DOWNLOAD
    # =========================
    result_pil = Image.fromarray(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    result_pil.save(buf, format="PNG")

    st.download_button(
        "⬇️ Download Hasil",
        data=buf.getvalue(),
        file_name="enhanced_result.png",
        mime="image/png"
    )

#=======
#About
#=======


with tab_about:
    st.markdown("""
    ## ℹ️ Tentang Aplikasi

    Aplikasi **Optimasi Citra Pengawasan** digunakan untuk
    **meningkatkan kualitas gambar CCTV** yang buram, gelap,
    berkabut, atau penuh noise agar objek lebih mudah dikenali.

    ### ⚙️ Fitur Utama
    - Upload gambar CCTV
    - Pengurangan noise (Median & NLM)
    - Pengurangan kabut (Dehazing)
    - Penajaman dan peningkatan kontras
    - Perbandingan gambar sebelum dan sesudah
    - Download hasil gambar

    ### 📊 Analisis
    - Blur Score (ketajaman gambar)
    - Haze Score (tingkat kabut)
    - Histogram intensitas

    ### 👥 Disusun Oleh
    1. Deo Kharisma Pratama (23010074)  
    2. Muhammad Irfan       (23010016)    
    3. Muhammad Isa Dawud   (23010045)
    4. Rahmat Ilahi         (230100)

    **Catatan:**  
    Aplikasi ini menggunakan teknik pengolahan citra klasik
    tanpa machine learning.
    """)

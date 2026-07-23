import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# ====================================
# CONFIG
# ====================================
st.set_page_config(
    page_title="Deteksi Katarak Mata",
    page_icon="👁️",
    layout="centered"
)
st.markdown("""
    <style>
    [data-testid="stFileUploaderDropzoneInput"] + div {display: none;}
    </style>
""", unsafe_allow_html=True)

# ====================================
# MODEL
# ====================================
MODEL_PATH = "./hasil_training/weights/best.pt"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model tidak ditemukan di path: {MODEL_PATH}")
        st.stop()
    return YOLO(MODEL_PATH)

model = load_model()

# ====================================
# SESSION STATE
# ====================================
if "hasil_deteksi" not in st.session_state:
    st.session_state.hasil_deteksi = None

def reset_ke_halaman_utama():
    st.session_state.hasil_deteksi = None

# ====================================
# TITLE
# ====================================
st.title("👁️ Deteksi Katarak Mata")

# ====================================
# HALAMAN UTAMA
# ====================================
if st.session_state.hasil_deteksi is None:

    st.write("Upload gambar mata untuk mendeteksi katarak menggunakan YOLOv11n.")
    st.divider()

    with st.expander("ℹ️ Apa itu Katarak?"):
        st.write("""
        **Katarak** adalah kondisi ketika lensa mata yang normalnya jernih menjadi keruh 
        sehingga cahaya yang masuk ke retina terhambat dan menyebabkan penglihatan menjadi 
        kabur. Jika tidak ditangani, katarak dapat menyebabkan kebutaan permanen. 
        Katarak merupakan salah satu penyebab utama kebutaan di Indonesia, dengan sekitar 
        81% kasus kebutaan disebabkan oleh katarak.
        
        **Gejala Katarak:**
        - Penglihatan kabur, buram, atau berkabut
        - Silau saat melihat cahaya terang atau lampu
        - Warna terlihat pudar atau kekuningan
        - Penglihatan ganda pada satu mata
        - Sulit melihat di malam hari
        - Sering mengganti kacamata karena ukuran terus berubah
        
        **Faktor Penyebab:**
        - Penuaan (usia di atas 60 tahun)
        - Diabetes melitus
        - Cedera atau trauma pada mata
        - Paparan sinar ultraviolet berlebihan
        - Penggunaan obat steroid jangka panjang
        - Riwayat keluarga dengan katarak
        
        **Cara Pencegahan:**
        - Gunakan kacamata hitam saat beraktivitas di luar ruangan
        - Kontrol kadar gula darah secara rutin
        - Hindari merokok
        - Konsumsi makanan kaya antioksidan seperti sayur dan buah
        - Lakukan pemeriksaan mata secara berkala
        
        **Penanganan:**
        Katarak dapat ditangani melalui operasi pengangkatan lensa yang keruh dan 
        digantikan dengan lensa buatan (Intraocular Lens/IOL). Operasi ini merupakan 
        prosedur yang aman dan efektif dengan tingkat keberhasilan yang tinggi.
        
        ---
        ⚠️ **Catatan Penting:** Aplikasi ini hanya sebagai alat bantu deteksi awal 
        berbasis kecerdasan buatan dan tidak menggantikan diagnosis dokter. 
        Segera konsultasikan dengan dokter mata untuk pemeriksaan lebih lanjut.
        """)

    uploaded_file = st.file_uploader(
        "Upload Gambar Mata",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name

        try:
            with st.spinner("Menganalisis gambar..."):
                results = model.predict(
                    source=tmp_path,
                    conf=0.45,
                    iou=0.5,
                    save=True
                )

                result_image_path = os.path.join(
                    results[0].save_dir,
                    os.path.basename(tmp_path)
                )

                boxes = results[0].boxes
                names = results[0].names

                if boxes is not None and len(boxes) > 0:
                    best_box = max(boxes, key=lambda x: float(x.conf[0]))
                    cls_id = int(best_box.cls[0])
                    conf = float(best_box.conf[0])
                    label = names[cls_id]
                else:
                    label = None
                    conf = None

            st.session_state.hasil_deteksi = {
                "gambar_asli": image,
                "gambar_hasil_path": result_image_path if os.path.exists(result_image_path) else None,
                "label": label,
                "conf": conf,
            }
            st.rerun()

        finally:
            os.unlink(tmp_path)

# ====================================
# HALAMAN HASIL
# ====================================
else:
    hasil = st.session_state.hasil_deteksi

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Gambar Asli")
        st.image(hasil["gambar_asli"], use_container_width=True)

    with col2:
        st.subheader("Hasil Deteksi")
        if hasil["gambar_hasil_path"]:
            st.image(hasil["gambar_hasil_path"], use_container_width=True)
        else:
            st.warning("Gambar hasil deteksi tidak ditemukan.")

    st.divider()
    st.subheader("Hasil Prediksi")

    if hasil["label"] is not None:
        if hasil["label"].lower() == "cataract":
            st.error(f"⚠️ KATARAK TERDETEKSI ({hasil['conf']*100:.1f}%)")
            st.write("Segera konsultasikan dengan dokter mata.")
        else:
            st.success(f"✅ MATA NORMAL ({hasil['conf']*100:.1f}%)")
            st.write("Tidak terdeteksi katarak pada gambar ini.")
    else:
        st.warning("⚠️ Tidak ada objek yang terdeteksi. Coba gunakan gambar yang lebih jelas.")

    st.divider()
    st.button("🔄 Upload gambar lain", on_click=reset_ke_halaman_utama, use_container_width=True)
import streamlit as st
import pandas as pd
import joblib
import os

# =========================
# LOAD DATASET & MODEL
# =========================

@st.cache_data
def load_data():
    return pd.read_csv("Water_Quality.csv")

@st.cache_resource
def load_model():
    return joblib.load("model_air.pkl")

try:
    df = load_data()
    model = load_model()
except FileNotFoundError as e:
    st.error(f"File tidak ditemukan: {e}")
    st.stop()

# =========================
# JUDUL
# =========================

st.title("Klasifikasi Kualitas Air")
st.write("Sistem klasifikasi kualitas air menggunakan algoritma Random Forest.")

# =========================
# DATASET
# =========================

st.subheader("Dataset Air")

st.write("Jumlah Dataset :", len(df))

jumlah = st.selectbox(
    "Jumlah Data Ditampilkan",
    [10, 20, 30, 50, 100]
)

st.dataframe(df.sample(n=jumlah, random_state=42))

# =========================
# INPUT DATA
# =========================

st.subheader("Input Data Air")

ph = st.number_input(
    "pH",
    min_value=float(df["pH"].min()),
    max_value=float(df["pH"].max()),
    value=float(df["pH"].mean())
)

turbidity = st.number_input(
    "Turbidity (NTU)",
    min_value=float(df["Turbidity (NTU)"].min()),
    max_value=float(df["Turbidity (NTU)"].max()),
    value=float(df["Turbidity (NTU)"].mean())
)

temperature = st.number_input(
    "Temperature (°C)",
    min_value=float(df["Temperature (°C)"].min()),
    max_value=float(df["Temperature (°C)"].max()),
    value=float(df["Temperature (°C)"].mean())
)

do = st.number_input(
    "DO (mg/L)",
    min_value=float(df["DO (mg/L)"].min()),
    max_value=float(df["DO (mg/L)"].max()),
    value=float(df["DO (mg/L)"].mean())
)

bod = st.number_input(
    "BOD (mg/L)",
    min_value=float(df["BOD (mg/L)"].min()),
    max_value=float(df["BOD (mg/L)"].max()),
    value=float(df["BOD (mg/L)"].mean())
)

st.caption("Masukkan nilai parameter sesuai rentang data yang digunakan dalam pelatihan model.")

# =========================
# PREDIKSI
# =========================

if st.button("Prediksi"):

    data = pd.DataFrame(
        [[ph, turbidity, temperature, do, bod]],
        columns=[
            "pH",
            "Turbidity (NTU)",
            "Temperature (°C)",
            "DO (mg/L)",
            "BOD (mg/L)"
        ]
    )

    hasil = model.predict(data)
    kategori = str(hasil[0]).lower()

    warna = {
    "jernih": "🟢",
    "keruh": "🟡",
    "kotor": "🔴"
    }

    st.subheader("🔍 Hasil Prediksi")

    st.success(
    f"{warna.get(kategori, '')} Hasil Klasifikasi Air: **{kategori.upper()}**"
    )

# Tambahkan kode ini di sini
if kategori == "jernih":
    st.success(
        "Air diprediksi berada pada kategori **Jernih** karena kombinasi nilai parameter masih berada pada rentang kualitas air yang baik."
    )

elif kategori == "keruh":
    st.warning(
        "Air diprediksi berada pada kategori **Keruh** karena terdapat beberapa parameter yang menunjukkan penurunan kualitas air."
    )

elif kategori == "kotor":
    st.error(
        "Air diprediksi berada pada kategori **Kotor** karena kombinasi parameter menunjukkan kualitas air yang buruk."
    )
# =========================
# PARAMETER YANG DIMASUKKAN
# =========================

    st.write("### 📋 Parameter yang Dimasukkan")

    input_df = pd.DataFrame({
    "Parameter": [
        "pH",
        "Turbidity (NTU)",
        "Temperature (°C)",
        "DO (mg/L)",
        "BOD (mg/L)"
    ],
    "Nilai": [
        ph,
        turbidity,
        temperature,
        do,
        bod
    ]
    })

    st.dataframe(input_df, hide_index=True)

# =========================
# TAMPILKAN GAMBAR
# =========================

    gambar = {
        "jernih": "Gambar/Jernih.jpg",
        "keruh": "Gambar/Keruh.jpg",
        "kotor": "Gambar/Kotor.jpg"
    }

    if kategori in gambar and os.path.exists(gambar[kategori]):

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.image(
                gambar[kategori],
                caption=f"Ilustrasi Air {kategori.capitalize()}",
                use_container_width=True
            )

        st.info(
             f"""
                Model Random Forest mengklasifikasikan sampel air sebagai **{kategori.upper()}**
                berdasarkan nilai parameter **pH, Turbidity, Temperature, DO, dan BOD**
                yang dimasukkan oleh pengguna.

                **Catatan:** Gambar di atas merupakan ilustrasi kondisi air sesuai
                hasil klasifikasi, bukan gambar yang diproses oleh model.
                """
            )
    # =========================
    # PROBABILITAS
    # =========================

    proba = model.predict_proba(data)[0]
    confidence = max(proba)

    st.metric(
    label="Tingkat Keyakinan Model",
    value=f"{confidence*100:.2f}%"
    )

    st.progress(float(confidence))
    kelas = model.classes_

    st.write("### Probabilitas Tiap Kategori")

    probabilitas = pd.DataFrame({
        "Kategori": kelas,
        "Probabilitas (%)": [round(p * 100, 2) for p in proba]
    })

    st.dataframe(probabilitas, hide_index=True)

# =========================
# FEATURE IMPORTANCE
# =========================

st.subheader("Tingkat Pengaruh Fitur (Feature Importance)")

importance_df = pd.DataFrame({
    "Fitur": [
        "pH",
        "Turbidity (NTU)",
        "Temperature (°C)",
        "DO (mg/L)",
        "BOD (mg/L)"
    ],
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.dataframe(importance_df, hide_index=True)

st.bar_chart(
    importance_df.set_index("Fitur")
)
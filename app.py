"""Aplikasi skrining awal risiko depresi mahasiswa (LightGBM + LIME).

Model: LightGBM dengan 10 fitur terpilih, dilatih dengan SMOTE pada data latih.
Aplikasi ini adalah alat bantu skrining awal, bukan alat diagnosis.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from lime.lime_tabular import LimeTabularExplainer

BASE = Path(__file__).parent
LIME_SEED = 42
LIME_NUM_SAMPLES = 5000

st.set_page_config(page_title="Skrining Awal Risiko Depresi Mahasiswa", page_icon="🧠", layout="centered")

# Nama fitur model -> label yang mudah dibaca pada hasil LIME
LABELS = {
    "Have you ever had suicidal thoughts ?_Yes": "Pernah berpikir untuk bunuh diri",
    "Academic Pressure": "Tekanan akademik",
    "Dietary Habits_Unhealthy": "Pola makan tidak sehat",
    "Dietary Habits_Moderate": "Pola makan sedang",
    "Financial Stress": "Tekanan finansial",
    "Age": "Usia",
    "Work/Study Hours": "Jam belajar/kerja per hari",
    "Study Satisfaction": "Kepuasan belajar",
    "Sleep Duration_Less than 5 hours": "Tidur kurang dari 5 jam",
    "Sleep Duration_More than 8 hours": "Tidur lebih dari 8 jam",
}


@st.cache_resource(show_spinner="Memuat model...")
def load_artifacts():
    model = joblib.load(BASE / "best_model.pkl")
    x_raw = np.asarray(joblib.load(BASE / "lime_training_data_raw.pkl"), dtype=float)
    meta = json.loads((BASE / "lime_meta.json").read_text(encoding="utf-8"))
    return model, x_raw, meta


def to_model_space(a_raw, sel, mean, scale):
    """Satuan asli -> ruang model (fitur numerik distandardisasi seperti saat pelatihan)."""
    a = np.array(a_raw, dtype=float).copy()
    for j, f in enumerate(sel):
        if f in mean:
            a[:, j] = (a[:, j] - mean[f]) / scale[f]
    return a


def build_row(values, sel):
    """Susun satu baris fitur (satuan asli) mengikuti urutan fitur terpilih."""
    raw = {
        "Have you ever had suicidal thoughts ?_Yes": 1.0 if values["suicidal"] == "Ya" else 0.0,
        "Academic Pressure": float(values["academic"]),
        "Dietary Habits_Unhealthy": 1.0 if values["diet"] == "Tidak sehat" else 0.0,
        "Dietary Habits_Moderate": 1.0 if values["diet"] == "Sedang" else 0.0,
        "Financial Stress": float(values["financial"]),
        "Age": float(values["age"]),
        "Work/Study Hours": float(values["hours"]),
        "Study Satisfaction": float(values["satisfaction"]),
        "Sleep Duration_Less than 5 hours": 1.0 if values["sleep"] == "Kurang dari 5 jam" else 0.0,
        "Sleep Duration_More than 8 hours": 1.0 if values["sleep"] == "Lebih dari 8 jam" else 0.0,
    }
    return np.array([[raw[f] for f in sel]], dtype=float)


def explain(model, x_raw, meta, row_raw):
    sel, mean, scale = meta["selected_features"], meta["numeric_mean"], meta["numeric_scale"]
    cat_idx = meta["categorical_features_idx"]

    def predict_fn(a_raw):
        xm = pd.DataFrame(to_model_space(a_raw, sel, mean, scale), columns=sel)
        return model.predict_proba(xm)

    explainer = LimeTabularExplainer(
        training_data=x_raw,
        feature_names=sel,
        class_names=["Tidak depresi", "Depresi"],
        categorical_features=cat_idx,
        categorical_names={j: ["0", "1"] for j in cat_idx},
        mode="classification",
        discretize_continuous=True,
        random_state=LIME_SEED,
    )
    exp = explainer.explain_instance(
        row_raw[0], predict_fn, num_features=len(sel), num_samples=LIME_NUM_SAMPLES, labels=(1,)
    )
    score = exp.score[1] if isinstance(exp.score, dict) else exp.score
    return exp.as_list(label=1), float(score)


def pretty_rule(rule):
    for k, v in sorted(LABELS.items(), key=lambda kv: -len(kv[0])):
        rule = rule.replace(k, v)
    return rule.replace("=1", " = Ya").replace("=0", " = Tidak")


# ----------------------------------------------------------------------------- UI
st.title("🧠 Skrining Awal Risiko Depresi Mahasiswa")
st.warning(
    "Aplikasi ini hanya alat bantu **skrining awal** dan **bukan diagnosis**. "
    "Hasilnya tidak menggantikan penilaian psikolog atau tenaga kesehatan profesional. "
    "Jika Anda atau seseorang yang Anda kenal sedang berada dalam kondisi darurat atau berpikir untuk menyakiti diri, "
    "segera hubungi tenaga kesehatan atau layanan darurat setempat."
)

try:
    model, x_raw, meta = load_artifacts()
except FileNotFoundError as e:
    st.error(f"Berkas model tidak ditemukan: {e.filename}. Pastikan best_model.pkl, lime_training_data_raw.pkl, dan lime_meta.json berada satu folder dengan app.py.")
    st.stop()

sel = meta["selected_features"]

with st.sidebar:
    st.header("Pengaturan")
    threshold = st.slider("Ambang keputusan", 0.20, 0.80, 0.50, 0.05,
                          help="Probabilitas minimum agar hasil dinyatakan 'terindikasi berisiko'. "
                               "Ambang lebih rendah menangkap lebih banyak kasus tetapi menambah alarm palsu.")
    show_lime = st.checkbox("Tampilkan penjelasan LIME", value=True)
    st.caption("Data yang dimasukkan tidak disimpan oleh aplikasi ini.")

with st.form("form_skrining"):
    st.subheader("Data mahasiswa")
    c1, c2 = st.columns(2)
    with c1:
        suicidal = st.radio("Pernah berpikir untuk bunuh diri?", ["Tidak", "Ya"], horizontal=True)
        academic = st.slider("Tekanan akademik (0 = tidak ada, 5 = sangat tinggi)", 0, 5, 3)
        financial = st.slider("Tekanan finansial (1 = rendah, 5 = sangat tinggi)", 1, 5, 3)
        satisfaction = st.slider("Kepuasan belajar (0 = tidak puas, 5 = sangat puas)", 0, 5, 3)
    with c2:
        age = st.number_input("Usia (tahun)", min_value=17, max_value=60, value=22, step=1)
        hours = st.slider("Jam belajar/kerja per hari", 0, 12, 6)
        diet = st.selectbox("Pola makan", ["Sehat", "Sedang", "Tidak sehat"])
        sleep = st.selectbox("Durasi tidur per hari", ["Kurang dari 5 jam", "5-6 jam", "7-8 jam", "Lebih dari 8 jam"], index=2)
    submitted = st.form_submit_button("Lihat hasil skrining", type="primary")

if submitted:
    values = dict(suicidal=suicidal, academic=academic, financial=financial, satisfaction=satisfaction,
                  age=age, hours=hours, diet=diet, sleep=sleep)
    row_raw = build_row(values, sel)
    xm = pd.DataFrame(to_model_space(row_raw, sel, meta["numeric_mean"], meta["numeric_scale"]), columns=sel)
    prob = float(model.predict_proba(xm)[0, 1])

    st.subheader("Hasil")
    st.metric("Probabilitas depresi menurut model", f"{prob:.1%}")
    st.progress(min(max(prob, 0.0), 1.0))
    if prob >= threshold:
        st.error(f"Terindikasi berisiko pada ambang {threshold:.2f}. Disarankan penilaian lanjutan oleh konselor atau tenaga profesional.")
    else:
        st.success(f"Tidak terindikasi berisiko pada ambang {threshold:.2f}. Hasil ini bukan jaminan; "
                   "jika ada keluhan, tetap konsultasikan dengan konselor atau tenaga profesional.")

    if show_lime:
        with st.spinner("Menghitung penjelasan LIME..."):
            items, r2 = explain(model, x_raw, meta, row_raw)
        st.subheader("Faktor yang memengaruhi prediksi ini (LIME)")
        names = [pretty_rule(r) for r, _ in items][::-1]
        weights = [w for _, w in items][::-1]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.barh(names, weights, color=["#d9534f" if w > 0 else "#4a7fb5" for w in weights])
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Kontribusi terhadap probabilitas depresi (kanan = mendorong ke depresi)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            f"Kecocokan model linear lokal LIME (R²) untuk kasus ini: {r2:.2f}. Nilai rendah berarti penjelasan perlu dibaca sebagai "
            "indikasi, bukan sebab-akibat."
        )

with st.expander("Tentang model"):
    st.markdown(
        "- Model: LightGBM dengan 10 fitur terpilih, dilatih dengan SMOTE hanya pada data latih.\n"
        "- Data latih: Student Depression Dataset (mahasiswa di India); label bukan hasil diagnosis klinis terdokumentasi.\n"
        "- Kinerja pada data uji (threshold 0,5): F1-Score 0,871 dan AUC-ROC 0,921.\n"
        "- Penjelasan LIME bersifat lokal dan acak; gunakan sebagai pendukung, bukan penentu."
    )

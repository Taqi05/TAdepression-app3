"""
Deteksi Risiko Depresi Mahasiswa — Streamlit App
==================================================
Pipeline preprocessing di file ini HARUS sama persis dengan yang dipakai
di notebook (TA_GridSearchCV_7MODEL.ipynb), supaya prediksi & LIME
konsisten dengan yang dilaporkan di Tugas Akhir.

File pendukung yang WAJIB ada di folder yang sama (hasil dari notebook):
- best_model.pkl          -> model terbaik (hasil GridSearchCV)
- scaler.pkl               -> StandardScaler (fit di X_train)
- ohe.pkl                  -> OneHotEncoder (fit di X_train)
- ordinal_encoder.pkl      -> OrdinalEncoder untuk Degree_Level
- selected_features.pkl    -> daftar 10 fitur hasil feature selection
- lime_training_data.pkl   -> X_train_sel_sm (dipakai rekonstruksi LIME explainer)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lime.lime_tabular

st.set_page_config(page_title="Deteksi Risiko Depresi Mahasiswa", page_icon="🧠", layout="centered")

# ── Mapping domain knowledge (HARUS sama persis dengan notebook) ─────────────
TIER_1 = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad']
TIER_2 = ['Pune', 'Ahmedabad', 'Jaipur', 'Lucknow', 'Surat', 'Bhopal', 'Indore', 'Nagpur', 'Patna']
TIER_3 = ['Agra', 'Faridabad', 'Ghaziabad', 'Kalyan', 'Kanpur', 'Ludhiana',
          'Meerut', 'Nashik', 'Rajkot', 'Srinagar', 'Thane', 'Vadodara',
          'Varanasi', 'Vasai-Virar', 'Visakhapatnam']
ALL_CITIES = TIER_1 + TIER_2 + TIER_3

def map_city_tier(city: str) -> str:
    if city in TIER_1:
        return 'Tier 1'
    elif city in TIER_2:
        return 'Tier 2'
    elif city in TIER_3:
        return 'Tier 3'
    return 'Other'

OHE_COLS = [
    'Gender',
    'City_Tier',
    'Dietary Habits',
    'Sleep Duration',
    'Have you ever had suicidal thoughts ?',
    'Family History of Mental Illness',
    'Profession',
]

NUM_COLS_SCALE = [
    'Age', 'CGPA', 'Work Pressure', 'Academic Pressure',
    'Study Satisfaction', 'Job Satisfaction',
    'Work/Study Hours', 'Financial Stress', 'Degree_Level',
]

DEGREE_LEVELS = ['High School', 'Bachelor', 'Master', 'Doctorate', 'Other']


# ── Load model & artefak (di-cache biar cuma load sekali) ────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    ohe = joblib.load('ohe.pkl')
    ordinal_encoder = joblib.load('ordinal_encoder.pkl')
    selected_features = list(joblib.load('selected_features.pkl'))
    return model, scaler, ohe, ordinal_encoder, selected_features


@st.cache_resource
def load_lime_explainer(_selected_features):
    # _selected_features diberi underscore biar Streamlit gak coba hash list-nya
    training_data = joblib.load('lime_training_data.pkl')
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(training_data),
        feature_names=list(_selected_features),
        class_names=['Tidak Depresi', 'Depresi'],
        mode='classification',
        discretize_continuous=True,
        random_state=42,
    )
    return explainer


model, scaler, ohe, ordinal_encoder, selected_features = load_artifacts()
explainer = load_lime_explainer(selected_features)


def predict_fn(x):
    return model.predict_proba(x)


# ── Preprocessing 1 baris input user, urutan HARUS sama kayak notebook ───────
def preprocess_input(raw: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw])

    # 1. Feature engineering City -> City_Tier
    df['City_Tier'] = df['City'].apply(map_city_tier)
    df = df.drop(columns=['City'])

    # 2. Ordinal encode Degree_Level
    df['Degree_Level'] = ordinal_encoder.transform(df[['Degree_Level']])

    # 3. One-hot encode kolom kategorikal
    ohe_out = ohe.transform(df[OHE_COLS])
    ohe_feature_names = ohe.get_feature_names_out(OHE_COLS).tolist()
    ohe_df = pd.DataFrame(ohe_out, columns=ohe_feature_names, index=df.index)

    df_num = df.drop(columns=OHE_COLS)
    df_full = pd.concat([df_num, ohe_df], axis=1)

    # 4. Scaling fitur numerik (termasuk Degree_Level yang udah ordinal)
    cols_to_scale = [c for c in NUM_COLS_SCALE if c in df_full.columns]
    df_full[cols_to_scale] = scaler.transform(df_full[cols_to_scale])

    # 5. Ambil hanya 10 fitur hasil feature selection, urutan HARUS sama
    df_selected = df_full[selected_features]
    return df_selected


# ── UI ─────────────────────────────────────────────────────────────────────
st.title("🧠 Deteksi Risiko Depresi Mahasiswa")
st.caption("Model machine learning hasil Tugas Akhir — GridSearchCV + SMOTE + Feature Selection + LIME")

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=15, max_value=60, value=21)
        city = st.selectbox("City", sorted(ALL_CITIES))
        profession = st.selectbox("Profession", ["Student", "Civil Engineer", "Teacher", "Doctor", "Other"])
        degree_level = st.selectbox("Degree Level", DEGREE_LEVELS)
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.01)
        sleep_duration = st.selectbox(
            "Sleep Duration",
            ["'Less than 5 hours'", "'5-6 hours'", "'7-8 hours'", "'More than 8 hours'"],
        )
        dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])

    with col2:
        academic_pressure = st.slider("Academic Pressure (0-5)", 0, 5, 3)
        work_pressure = st.slider("Work Pressure (0-5)", 0, 5, 0)
        study_satisfaction = st.slider("Study Satisfaction (0-5)", 0, 5, 3)
        job_satisfaction = st.slider("Job Satisfaction (0-5)", 0, 5, 0)
        work_study_hours = st.number_input("Work/Study Hours per hari", min_value=0, max_value=24, value=6)
        financial_stress = st.slider("Financial Stress (1-5)", 1, 5, 3)
        suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts ?", ["No", "Yes"])
        family_history = st.selectbox("Family History of Mental Illness", ["No", "Yes"])

    submitted = st.form_submit_button("Prediksi")

if submitted:
    raw_input = {
        'Gender': gender,
        'Age': age,
        'City': city,
        'Profession': profession,
        'Degree_Level': degree_level,
        'CGPA': cgpa,
        'Sleep Duration': sleep_duration,
        'Dietary Habits': dietary_habits,
        'Academic Pressure': academic_pressure,
        'Work Pressure': work_pressure,
        'Study Satisfaction': study_satisfaction,
        'Job Satisfaction': job_satisfaction,
        'Work/Study Hours': work_study_hours,
        'Financial Stress': financial_stress,
        'Have you ever had suicidal thoughts ?': suicidal_thoughts,
        'Family History of Mental Illness': family_history,
    }

    X_input = preprocess_input(raw_input)
    proba = predict_fn(X_input.values)[0]
    pred = int(proba[1] > 0.5)

    if pred == 1:
        st.error(f"⚠️ Prediksi: **Berisiko Depresi** (probabilitas {proba[1]*100:.1f}%)")
    else:
        st.success(f"✅ Prediksi: **Tidak Berisiko Depresi** (probabilitas {proba[1]*100:.1f}%)")

    st.progress(float(proba[1]))

    st.caption(
        "Catatan: ini adalah output model machine learning untuk keperluan akademik/Tugas Akhir, "
        "bukan diagnosis klinis. Jika mengalami tekanan psikologis, disarankan berkonsultasi "
        "dengan profesional kesehatan mental."
    )

    with st.expander("Lihat data yang diproses ke model"):
        st.table(X_input.T.rename(columns={0: "Nilai"}))

    st.subheader("🔍 Interpretasi Model (LIME)")
    st.caption("Menunjukkan fitur mana yang paling mendorong / menahan prediksi risiko depresi untuk kasus ini.")

    exp = explainer.explain_instance(
        data_row=X_input.values[0],
        predict_fn=predict_fn,
        num_features=len(selected_features),
    )

    for feat, weight in exp.as_list():
        arah = "🔴 Meningkatkan risiko" if weight > 0 else "🟢 Menurunkan risiko"
        st.write(f"**{feat}** &nbsp; `{weight:+.4f}` &nbsp; {arah}")

    fig = exp.as_pyplot_figure()
    st.pyplot(fig)

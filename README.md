# Deteksi Risiko Depresi Mahasiswa

Aplikasi Streamlit untuk memprediksi risiko depresi mahasiswa, dibangun dari
model machine learning hasil Tugas Akhir (GridSearchCV + SMOTE + Feature
Selection + LIME).

## Isi folder

```
app.py                     # aplikasi Streamlit
requirements.txt           # dependency Python
best_model.pkl              # model terbaik hasil GridSearchCV       (dari notebook)
scaler.pkl                  # StandardScaler, fit di X_train         (dari notebook)
ohe.pkl                     # OneHotEncoder, fit di X_train          (dari notebook)
ordinal_encoder.pkl         # OrdinalEncoder untuk Degree_Level      (dari notebook)
selected_features.pkl       # daftar 10 fitur hasil feature selection (dari notebook)
lime_training_data.pkl      # data referensi untuk rekonstruksi LIME  (dari notebook)
```

File-file `.pkl` **tidak dibuat di sini** — semuanya dihasilkan dari notebook
`TA_GridSearchCV_7MODEL.ipynb` (lihat bagian "SIMPAN MODEL" di notebook).
Download semua file itu dari Colab lalu taruh sejajar dengan `app.py` sebelum
menjalankan atau deploy aplikasi ini.

## Menjalankan secara lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud

1. Push folder ini (termasuk semua file `.pkl`) ke repo GitHub.
2. Buka [share.streamlit.io](https://share.streamlit.io), hubungkan ke repo tsb.
3. Pilih `app.py` sebagai entry point, klik Deploy.

## Catatan penting

- Pipeline preprocessing di `app.py` (mapping City → City_Tier, Ordinal
  encoding Degree_Level, One-Hot Encoding, StandardScaler, lalu seleksi 10
  fitur) **harus tetap sinkron** dengan pipeline di notebook. Kalau notebook
  diubah (mis. fitur baru, kategori baru), `app.py` juga harus di-update.
- Disclaimer di aplikasi ("bukan diagnosis klinis, konsultasikan ke
  profesional kesehatan mental") sengaja dipertahankan — jangan dihapus,
  karena output ini murni untuk keperluan akademik/Tugas Akhir.

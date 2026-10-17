# Deteksi Risiko Depresi Mahasiswa

Aplikasi Streamlit untuk skrining awal risiko depresi mahasiswa, dibangun dari model hasil Tugas Akhir:
LightGBM dengan 10 fitur terpilih, dioptimasi dengan GridSearchCV, dilatih dengan SMOTE pada data latih,
dan dilengkapi penjelasan prediksi dengan LIME.

> Aplikasi ini adalah alat bantu skrining awal dan **bukan diagnosis klinis**. Hasilnya tidak menggantikan
> penilaian psikolog atau tenaga kesehatan profesional.

## Isi folder

```
app.py                      # aplikasi Streamlit
requirements.txt            # dependency Python
best_model.pkl              # model LightGBM akhir                     (dari notebook)
lime_training_data_raw.pkl  # data latih asli (satuan asli) untuk LIME (dari notebook)
lime_meta.json              # daftar 10 fitur, rata-rata dan simpangan baku fitur numerik (dari notebook)
```

Berkas `.pkl` dan `.json` tidak dibuat di sini. Semuanya dihasilkan oleh notebook penelitian
(bagian "Simpan Model & Artefak"). Unduh dari notebook, lalu letakkan sejajar dengan `app.py`.

## Menjalankan secara lokal

```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud

1. Unggah folder ini (termasuk semua berkas di atas) ke repo GitHub.
2. Buka [share.streamlit.io](https://share.streamlit.io/) dan hubungkan ke repo ini.
3. Pilih `app.py` sebagai entry point, lalu klik Deploy.

## Catatan penting

- Aplikasi menanyakan 8 hal (pikiran bunuh diri, tekanan akademik, tekanan finansial, kepuasan belajar, usia,
  jam belajar/kerja, pola makan, durasi tidur) yang dipetakan ke 10 fitur model. Fitur numerik distandardisasi
  memakai rata-rata dan simpangan baku dari `lime_meta.json`.
- Jika notebook diubah (misalnya jumlah atau daftar fitur terpilih), fungsi pembentuk fitur di `app.py`
  harus ikut disesuaikan, dan semua berkas model diunggah ulang bersamaan.
- Versi `scikit-learn` dan `lightgbm` pada `requirements.txt` sengaja dikunci sama dengan versi saat model dilatih,
  karena berkas model peka terhadap versi.
- Model dilatih pada Student Depression Dataset (mahasiswa di India), dan labelnya bukan hasil diagnosis klinis
  terdokumentasi. Disclaimer di aplikasi sengaja dipertahankan; aplikasi ini untuk keperluan akademik.
- Aplikasi tidak menyimpan data yang dimasukkan pengguna.

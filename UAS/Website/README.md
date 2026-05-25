# Clustering Experiment Platform

Platform eksperimen clustering berbasis website untuk membandingkan dataset, teknik preprocessing, dan model clustering.

Versi ini sudah berfungsi untuk:

- Upload dataset CSV
- Simpan dataset
- Pilih dataset
- Pilih teknik preprocessing
- Wajib ordinal encoding untuk `GameDifficulty`
  - Easy = 0
  - Medium = 1
  - Hard = 2
- Jalankan Agglomerative Hierarchical Clustering
- Hitung evaluasi:
  - Silhouette Score
  - Davies-Bouldin Index
  - Calinski-Harabasz Score
- Simpan hasil eksperimen
- Tampilkan visualisasi:
  - Dendrogram
  - Distribusi cluster
  - Rata-rata fitur per cluster
  - PCA 2D
- Download hasil CSV

## Struktur Project

```text
clustering_experiment_platform/
├── backend/
└── frontend/
```

## Cara Menjalankan Backend

Masuk ke folder backend:

```bash
cd backend
```

Buat virtual environment:

```bash
python -m venv venv
```

Aktifkan venv Windows:

```bash
venv\Scripts\activate
```

Install dependency:

```bash
pip install -r requirements.txt
```

Jalankan API:

```bash
uvicorn main:app --reload --port 8000
```

Buka API docs:

```text
http://127.0.0.1:8000/docs
```

## Cara Menjalankan Frontend

Buka terminal baru, masuk ke folder frontend:

```bash
cd frontend
```

Install dependency:

```bash
npm install
```

Jalankan frontend:

```bash
npm run dev
```

Buka website:

```text
http://127.0.0.1:5173
```

## Catatan Dataset Gaming Behavior

Untuk dataset `online_gaming_behavior_dataset.csv`, fitur default yang digunakan adalah:

```text
Age
PlayTimeHours
InGamePurchases
GameDifficulty
SessionsPerWeek
AvgSessionDurationMinutes
PlayerLevel
AchievementsUnlocked
```

Kolom yang tidak dipakai sebagai fitur clustering:

```text
PlayerID
Gender
Location
GameGenre
EngagementLevel
```

Alasan:

- `PlayerID` hanya identitas.
- `EngagementLevel` merupakan label/kategori, sedangkan clustering adalah unsupervised learning.
- `GameDifficulty` wajib di-encoding karena berupa teks ordinal.
- `Easy = 0`, `Medium = 1`, `Hard = 2` karena Easy < Medium < Hard.

## Cara Menambah Model Baru

Tambahkan file model baru di:

```text
backend/app/cluster_models/
```

Contoh:

```text
kmeans.py
dbscan.py
```

Lalu daftarkan di:

```text
backend/app/registry.py
```

dengan pola:

```python
CLUSTER_MODELS = {
    AgglomerativeModel.id: AgglomerativeModel(),
    # KMeansModel.id: KMeansModel(),
    # DBSCANModel.id: DBSCANModel(),
}
```

## Cara Menambah Teknik Preprocessing Baru

Tambahkan file preprocessor baru di:

```text
backend/app/preprocessors/
```

Lalu daftarkan di:

```text
backend/app/registry.py
```

dengan pola:

```python
PREPROCESSORS = {
    StandardPreprocessor.id: StandardPreprocessor(),
    MinMaxPreprocessor.id: MinMaxPreprocessor(),
    RobustPreprocessor.id: RobustPreprocessor(),
}
```

## Status

Ini adalah full build MVP modular. Agglomerative Clustering sudah berjalan. K-Means dan DBSCAN bisa ditambahkan tanpa membongkar struktur utama.

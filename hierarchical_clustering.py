# ============================================================
# PROGRAM 2
# IMPLEMENTASI HIERARCHICAL CLUSTERING
# Input  : dataset_sample/online_gaming_behavior_sample_1000.csv
# Output : gambar visualisasi + CSV hasil clustering
# Metode : Agglomerative Hierarchical Clustering
# Linkage: Ward
# Metric : Euclidean
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import dendrogram, linkage


# ============================================================
# 0. KONFIGURASI
# ============================================================

INPUT_SAMPLE_PATH = os.path.join(
    "dataset_sample",
    "online_gaming_behavior_sample_1000.csv"
)

OUTPUT_IMAGE_DIR = "visualisasi_hasil_clustering"
OUTPUT_CSV_DIR = "output_hasil_clustering"

DENDROGRAM_SAMPLE_SIZE = 100
FINAL_CLUSTER = 4
RANDOM_STATE = 42

SHOW_PLOTS = False
# Ubah menjadi True jika ingin gambar ikut tampil saat program berjalan


# ============================================================
# 1. BUAT FOLDER OUTPUT
# ============================================================

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)


def save_figure(filename):
    """
    Menyimpan gambar visualisasi ke folder output.
    """
    filepath = os.path.join(OUTPUT_IMAGE_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Gambar berhasil disimpan: {filepath}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def create_agglomerative_model(n_clusters):
    """
    Membuat model AgglomerativeClustering.
    Kompatibel untuk scikit-learn versi baru dan lama.
    """
    try:
        return AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="euclidean",
            linkage="ward"
        )
    except TypeError:
        return AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity="euclidean",
            linkage="ward"
        )


# ============================================================
# 2. LOAD DATA SAMPLE 1000
# ============================================================

if not os.path.exists(INPUT_SAMPLE_PATH):
    raise FileNotFoundError(
        f"File sample tidak ditemukan: {INPUT_SAMPLE_PATH}\n"
        "Jalankan dulu program 01_buat_sample_1000.py"
    )

df_sample = pd.read_csv(INPUT_SAMPLE_PATH)

print("\n=== DATA SAMPLE ===")
print("Jumlah baris dan kolom:", df_sample.shape)
print("\n5 data pertama:")
print(df_sample.head())


# ============================================================
# 3. VALIDASI KOLOM
# ============================================================

required_columns = [
    "PlayerID",
    "Age",
    "PlayTimeHours",
    "InGamePurchases",
    "GameDifficulty",
    "SessionsPerWeek",
    "AvgSessionDurationMinutes",
    "PlayerLevel",
    "AchievementsUnlocked"
]

missing_columns = [
    col for col in required_columns
    if col not in df_sample.columns
]

if missing_columns:
    raise ValueError(
        f"Kolom berikut tidak ditemukan pada dataset sample: {missing_columns}"
    )


# ============================================================
# 4. CEK MISSING VALUE DAN DUPLIKAT
# ============================================================

print("\n=== CEK MISSING VALUE ===")
print(df_sample.isnull().sum())

print("\n=== CEK DATA DUPLIKAT ===")
print("Jumlah data duplikat:", df_sample.duplicated().sum())

df_sample = df_sample.drop_duplicates().reset_index(drop=True)


# ============================================================
# 5. PILIH FITUR UNTUK CLUSTERING
# ============================================================

features = [
    "Age",
    "PlayTimeHours",
    "InGamePurchases",
    "GameDifficulty",
    "SessionsPerWeek",
    "AvgSessionDurationMinutes",
    "PlayerLevel",
    "AchievementsUnlocked"
]

data = df_sample[features].copy()

print("\n=== DATA FITUR SEBELUM PREPROCESSING ===")
print(data.head())


# ============================================================
# 6. ENCODING GAME DIFFICULTY
# ============================================================
# GameDifficulty bersifat ordinal:
# Easy < Medium < Hard
#
# Easy   = 0
# Medium = 1
# Hard   = 2

difficulty_mapping = {
    "Easy": 0,
    "Medium": 1,
    "Hard": 2
}

data["GameDifficulty"] = (
    data["GameDifficulty"]
    .astype(str)
    .str.strip()
    .map(difficulty_mapping)
)


# ============================================================
# 7. KONVERSI SEMUA FITUR KE NUMERIK
# ============================================================

for col in features:
    data[col] = pd.to_numeric(data[col], errors="coerce")


# ============================================================
# 8. HAPUS DATA KOSONG SETELAH PREPROCESSING
# ============================================================

print("\n=== MISSING VALUE SETELAH PREPROCESSING ===")
print(data.isnull().sum())

valid_index = data.dropna().index

data = data.loc[valid_index].reset_index(drop=True)
df_sample = df_sample.loc[valid_index].reset_index(drop=True)

print("\nJumlah data setelah preprocessing:", data.shape)

if len(data) < 2:
    raise ValueError(
        "Data terlalu sedikit setelah preprocessing. "
        "Periksa isi dataset atau mapping GameDifficulty."
    )


# ============================================================
# 9. SCALING DATA
# ============================================================
# StandardScaler digunakan karena Hierarchical Clustering
# berbasis jarak.

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

scaled_df = pd.DataFrame(scaled_data, columns=features)

print("\n=== DATA SETELAH SCALING ===")
print(scaled_df.head())


# ============================================================
# 10. DENDROGRAM
# ============================================================
# Dendrogram dibuat dari sebagian data agar visualisasi tetap terbaca.

dendrogram_sample_size = min(DENDROGRAM_SAMPLE_SIZE, len(scaled_df))

dendro_sample = scaled_df.sample(
    n=dendrogram_sample_size,
    random_state=RANDOM_STATE
)

linked = linkage(dendro_sample, method="ward")

plt.figure(figsize=(16, 8))
dendrogram(linked)
plt.title("Dendrogram Hierarchical Clustering")
plt.xlabel("Data Player")
plt.ylabel("Jarak")
plt.tight_layout()
save_figure("01_dendrogram_hierarchical_clustering.png")


# ============================================================
# 11. EVALUASI JUMLAH CLUSTER
# ============================================================
# Silhouette Score: semakin tinggi semakin baik.
# Davies-Bouldin Index: semakin rendah semakin baik.

print("\n=== EVALUASI JUMLAH CLUSTER ===")

evaluation_results = []

max_k = min(6, len(scaled_df) - 1)

for k in range(2, max_k + 1):
    model_eval = create_agglomerative_model(k)
    labels_eval = model_eval.fit_predict(scaled_df)

    silhouette = silhouette_score(scaled_df, labels_eval)
    dbi = davies_bouldin_score(scaled_df, labels_eval)

    evaluation_results.append({
        "Jumlah Cluster": k,
        "Silhouette Score": silhouette,
        "Davies-Bouldin Index": dbi
    })

    print(f"Cluster: {k}")
    print(f"Silhouette Score     : {silhouette:.4f}")
    print(f"Davies-Bouldin Index : {dbi:.4f}")
    print("-" * 40)

evaluation_df = pd.DataFrame(evaluation_results)

print("\n=== TABEL EVALUASI CLUSTER ===")
print(evaluation_df)


# ============================================================
# 12. SIMPAN HASIL EVALUASI CLUSTER
# ============================================================

evaluation_csv_path = os.path.join(
    OUTPUT_CSV_DIR,
    "evaluasi_cluster_gaming_behavior.csv"
)

evaluation_df.to_csv(evaluation_csv_path, index=False)
print(f"\nFile evaluasi cluster disimpan: {evaluation_csv_path}")


# ============================================================
# 13. VISUALISASI SILHOUETTE SCORE
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(
    evaluation_df["Jumlah Cluster"],
    evaluation_df["Silhouette Score"],
    marker="o"
)
plt.title("Silhouette Score untuk Beberapa Jumlah Cluster")
plt.xlabel("Jumlah Cluster")
plt.ylabel("Silhouette Score")
plt.xticks(evaluation_df["Jumlah Cluster"])
plt.grid(True)
plt.tight_layout()
save_figure("02_silhouette_score.png")


# ============================================================
# 14. VISUALISASI DAVIES-BOULDIN INDEX
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(
    evaluation_df["Jumlah Cluster"],
    evaluation_df["Davies-Bouldin Index"],
    marker="o"
)
plt.title("Davies-Bouldin Index untuk Beberapa Jumlah Cluster")
plt.xlabel("Jumlah Cluster")
plt.ylabel("Davies-Bouldin Index")
plt.xticks(evaluation_df["Jumlah Cluster"])
plt.grid(True)
plt.tight_layout()
save_figure("03_davies_bouldin_index.png")


# ============================================================
# 15. MODEL FINAL HIERARCHICAL CLUSTERING
# ============================================================

final_model = create_agglomerative_model(FINAL_CLUSTER)

cluster_labels = final_model.fit_predict(scaled_df)


# ============================================================
# 16. SIMPAN HASIL CLUSTER KE DATAFRAME
# ============================================================
# result_df = data numerik hasil preprocessing + Cluster
# df_result = data asli + Cluster

result_df = data.copy()
result_df["Cluster"] = cluster_labels

df_result = df_sample.copy()
df_result["Cluster"] = cluster_labels

print("\n=== CONTOH HASIL CLUSTERING ===")
print(df_result[[
    "PlayerID",
    "Age",
    "GameDifficulty",
    "PlayTimeHours",
    "SessionsPerWeek",
    "PlayerLevel",
    "AchievementsUnlocked",
    "Cluster"
]].head())


# ============================================================
# 17. DISTRIBUSI JUMLAH DATA SETIAP CLUSTER
# ============================================================

cluster_counts = result_df["Cluster"].value_counts().sort_index()

print("\n=== JUMLAH DATA PER CLUSTER ===")
print(cluster_counts)

cluster_counts_csv_path = os.path.join(
    OUTPUT_CSV_DIR,
    "jumlah_data_per_cluster.csv"
)

cluster_counts.to_csv(cluster_counts_csv_path, header=["Jumlah Data"])
print(f"File jumlah data per cluster disimpan: {cluster_counts_csv_path}")

plt.figure(figsize=(8, 5))
cluster_counts.plot(kind="bar")
plt.title("Jumlah Pemain pada Setiap Cluster")
plt.xlabel("Cluster")
plt.ylabel("Jumlah Pemain")
plt.xticks(rotation=0)
plt.tight_layout()
save_figure("04_jumlah_pemain_per_cluster.png")


# ============================================================
# 18. RINGKASAN RATA-RATA FITUR PER CLUSTER
# ============================================================

cluster_summary = result_df.groupby("Cluster")[features].mean()

print("\n=== RATA-RATA FITUR PER CLUSTER ===")
print(cluster_summary)

cluster_summary_csv_path = os.path.join(
    OUTPUT_CSV_DIR,
    "ringkasan_cluster_gaming_behavior.csv"
)

cluster_summary.to_csv(cluster_summary_csv_path)
print(f"File ringkasan cluster disimpan: {cluster_summary_csv_path}")


# ============================================================
# 19. VISUALISASI RATA-RATA FITUR PER CLUSTER
# ============================================================

cluster_summary.plot(kind="bar", figsize=(14, 7))
plt.title("Rata-rata Fitur pada Setiap Cluster")
plt.xlabel("Cluster")
plt.ylabel("Nilai Rata-rata")
plt.xticks(rotation=0)
plt.tight_layout()
save_figure("05_rata_rata_fitur_per_cluster.png")


# ============================================================
# 20. PCA UNTUK VISUALISASI 2D
# ============================================================

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_df)

df_result["PCA1"] = pca_result[:, 0]
df_result["PCA2"] = pca_result[:, 1]

pca_info = pd.DataFrame({
    "Komponen": ["PCA1", "PCA2"],
    "Explained Variance Ratio": pca.explained_variance_ratio_
})

print("\n=== EXPLAINED VARIANCE RATIO PCA ===")
print(pca_info)

pca_info_csv_path = os.path.join(
    OUTPUT_CSV_DIR,
    "pca_explained_variance.csv"
)

pca_info.to_csv(pca_info_csv_path, index=False)
print(f"File PCA explained variance disimpan: {pca_info_csv_path}")

plt.figure(figsize=(9, 6))
plt.scatter(
    df_result["PCA1"],
    df_result["PCA2"],
    c=df_result["Cluster"],
    alpha=0.7
)
plt.title("Visualisasi Cluster Menggunakan PCA 2D")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.tight_layout()
save_figure("06_visualisasi_cluster_pca_2d.png")


# ============================================================
# 21. INTERPRETASI AWAL CLUSTER
# ============================================================

print("\n=== INTERPRETASI AWAL CLUSTER ===")

global_mean = cluster_summary.mean()
interpretation_rows = []

for cluster_id in sorted(result_df["Cluster"].unique()):
    row = cluster_summary.loc[cluster_id]

    if (
        row["PlayTimeHours"] < global_mean["PlayTimeHours"]
        and row["SessionsPerWeek"] < global_mean["SessionsPerWeek"]
        and row["AvgSessionDurationMinutes"] < global_mean["AvgSessionDurationMinutes"]
    ):
        tipe = "Casual Player"
        alasan = "Waktu bermain, frekuensi sesi, dan durasi sesi cenderung rendah."

    elif (
        row["PlayTimeHours"] >= global_mean["PlayTimeHours"]
        and row["SessionsPerWeek"] >= global_mean["SessionsPerWeek"]
        and row["PlayerLevel"] >= global_mean["PlayerLevel"]
    ):
        tipe = "Hardcore Grinder"
        alasan = "Waktu bermain, frekuensi sesi, dan level pemain cenderung tinggi."

    elif row["AchievementsUnlocked"] >= global_mean["AchievementsUnlocked"]:
        tipe = "Achievement Hunter"
        alasan = "Jumlah achievement yang dibuka cenderung tinggi."

    elif row["InGamePurchases"] >= global_mean["InGamePurchases"]:
        tipe = "Active Spender"
        alasan = "Kecenderungan melakukan pembelian dalam game lebih tinggi."

    else:
        tipe = "Balanced Player"
        alasan = "Perilaku bermain berada pada tingkat sedang atau seimbang."

    interpretation_rows.append({
        "Cluster": cluster_id,
        "Tipe Pemain": tipe,
        "Alasan": alasan
    })

    print(f"\nCluster {cluster_id}")
    print("Tipe   :", tipe)
    print("Alasan :", alasan)

interpretation_df = pd.DataFrame(interpretation_rows)

interpretation_csv_path = os.path.join(
    OUTPUT_CSV_DIR,
    "interpretasi_awal_cluster.csv"
)

interpretation_df.to_csv(interpretation_csv_path, index=False)
print(f"\nFile interpretasi awal cluster disimpan: {interpretation_csv_path}")


# ============================================================
# 22. SIMPAN DATA HASIL CLUSTERING
# ============================================================

hasil_cluster_csv_path = os.path.join(
    OUTPUT_CSV_DIR,
    "hasil_clustering_gaming_behavior.csv"
)

df_result.to_csv(hasil_cluster_csv_path, index=False)
print(f"\nFile hasil clustering disimpan: {hasil_cluster_csv_path}")


# ============================================================
# 23. SIMPAN DATA PREPROCESSING DAN SCALING
# ============================================================

data_preprocessed_path = os.path.join(
    OUTPUT_CSV_DIR,
    "data_preprocessing_numerik.csv"
)

scaled_data_path = os.path.join(
    OUTPUT_CSV_DIR,
    "data_setelah_scaling.csv"
)

result_df.to_csv(data_preprocessed_path, index=False)
scaled_df.to_csv(scaled_data_path, index=False)

print(f"File data preprocessing disimpan: {data_preprocessed_path}")
print(f"File data scaling disimpan: {scaled_data_path}")


# ============================================================
# 24. RINGKASAN AKHIR
# ============================================================

print("\n=== RINGKASAN AKHIR IMPLEMENTASI ===")
print(f"Jumlah data sample            : {len(df_sample)}")
print(f"Jumlah data setelah cleaning  : {len(data)}")
print(f"Jumlah fitur clustering       : {len(features)}")
print(f"Jumlah cluster final          : {FINAL_CLUSTER}")
print(f"Folder gambar                 : {OUTPUT_IMAGE_DIR}")
print(f"Folder output CSV             : {OUTPUT_CSV_DIR}")

print("\n=== FILE GAMBAR YANG DISIMPAN ===")
print("01_dendrogram_hierarchical_clustering.png")
print("02_silhouette_score.png")
print("03_davies_bouldin_index.png")
print("04_jumlah_pemain_per_cluster.png")
print("05_rata_rata_fitur_per_cluster.png")
print("06_visualisasi_cluster_pca_2d.png")

print("\n=== FILE CSV YANG DISIMPAN ===")
print("hasil_clustering_gaming_behavior.csv")
print("ringkasan_cluster_gaming_behavior.csv")
print("evaluasi_cluster_gaming_behavior.csv")
print("jumlah_data_per_cluster.csv")
print("pca_explained_variance.csv")
print("interpretasi_awal_cluster.csv")
print("data_preprocessing_numerik.csv")
print("data_setelah_scaling.csv")

print("\nProgram hierarchical clustering selesai dijalankan.")
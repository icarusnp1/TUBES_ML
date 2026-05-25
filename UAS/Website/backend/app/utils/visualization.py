import os
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage


def _save_current_plot(path: str):
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def create_visualizations(
    x_scaled: pd.DataFrame,
    raw_data: pd.DataFrame,
    labels,
    experiment_dir: str,
    feature_columns: list[str],
    dendrogram_sample_size: int = 100
) -> dict:
    image_dir = os.path.join(experiment_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    image_paths = {}

    # 1. Dendrogram
    sample_size = min(dendrogram_sample_size, len(x_scaled))
    dendro_sample = x_scaled.sample(n=sample_size, random_state=42)
    linked = linkage(dendro_sample, method="ward")

    plt.figure(figsize=(16, 8))
    dendrogram(linked)
    plt.title("Dendrogram Hierarchical Clustering")
    plt.xlabel("Data Sample")
    plt.ylabel("Jarak")
    dendro_path = os.path.join(image_dir, "01_dendrogram.png")
    _save_current_plot(dendro_path)
    image_paths["dendrogram"] = dendro_path

    # 2. Cluster counts
    cluster_counts = pd.Series(labels).value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    cluster_counts.plot(kind="bar")
    plt.title("Jumlah Data pada Setiap Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Jumlah Data")
    plt.xticks(rotation=0)
    counts_path = os.path.join(image_dir, "02_cluster_counts.png")
    _save_current_plot(counts_path)
    image_paths["cluster_counts"] = counts_path

    # 3. Feature mean per cluster
    result = raw_data.copy()
    result["Cluster"] = labels
    cluster_summary = result.groupby("Cluster")[feature_columns].mean()

    cluster_summary.plot(kind="bar", figsize=(14, 7))
    plt.title("Rata-rata Fitur pada Setiap Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Nilai Rata-rata")
    plt.xticks(rotation=0)
    summary_path = os.path.join(image_dir, "03_feature_means.png")
    _save_current_plot(summary_path)
    image_paths["feature_means"] = summary_path

    # 4. PCA 2D
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(x_scaled)

    plt.figure(figsize=(9, 6))
    plt.scatter(
        pca_result[:, 0],
        pca_result[:, 1],
        c=labels,
        alpha=0.7
    )
    plt.title("Visualisasi Cluster Menggunakan PCA 2D")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    pca_path = os.path.join(image_dir, "04_pca_2d.png")
    _save_current_plot(pca_path)
    image_paths["pca_2d"] = pca_path

    return image_paths

import pandas as pd
from sklearn.cluster import KMeans

from app.cluster_models.base import BaseClusterModel


class KMeansModel(BaseClusterModel):
    id = "kmeans"
    name = "K-Means Clustering"
    description = (
        "Algoritma clustering berbasis partisi yang memisahkan data "
        "ke dalam K kelompok yang saling eksklusif dengan meminimalkan "
        "jarak antara setiap titik data dan pusat klasternya (centroid)."
    )

    def fit_predict(self, x_scaled: pd.DataFrame, params: dict):
        # Mengambil parameter dari request web (default n_clusters = 5)
        n_clusters = int(params.get("n_clusters", 5))
        init_method = params.get("init", "k-means++")
        random_state = int(params.get("random_state", 42))

        # Inisialisasi dan melatih model K-Means
        model = KMeans(
            n_clusters=n_clusters,
            init=init_method,
            random_state=random_state
        )
        
        # Mengembalikan array label klaster (misal: [0, 1, 4, 2, ...])
        return model.fit_predict(x_scaled)

    def get_config(self, params: dict) -> dict:
        # Menyimpan riwayat konfigurasi yang digunakan pengguna
        return {
            "model_id": self.id,
            "model_name": self.name,
            "n_clusters": int(params.get("n_clusters", 5)),
            "init": params.get("init", "k-means++"),
            "random_state": int(params.get("random_state", 42)),
            "note": "Menggunakan k-means++ untuk inisialisasi centroid yang optimal."
        }

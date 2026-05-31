import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import warnings

from app.cluster_models.base import BaseClusterModel

class DBSCANModel(BaseClusterModel):
    id = "dbscan"
    name = "DBSCAN (Density-Based Spatial Clustering of Applications with Noise)"
    description = (
        "Algoritma clustering berbasis kepadatan. Menemukan cluster dengan bentuk apa pun "
        "dan mendeteksi noise/outlier (ditandai dengan cluster -1)."
    )

    def fit_predict(self, x_scaled: pd.DataFrame, params: dict):
        eps_param = params.get("eps", "auto")
        min_samples_param = params.get("min_samples", "auto")

        X = x_scaled.values

        is_eps_auto = eps_param is None or str(eps_param).strip().lower() == "auto"
        is_min_samples_auto = min_samples_param is None or str(min_samples_param).strip().lower() == "auto"

        # Tentukan min_samples_final
        if is_min_samples_auto:
            # Rule of thumb: minPts = 2 * Dimensi
            dimensi = X.shape[1]
            min_samples_final = max(3, 2 * dimensi)
        else:
            try:
                min_samples_final = int(min_samples_param)
            except ValueError:
                min_samples_final = 5

        # Tentukan eps_final
        if is_eps_auto:
            from sklearn.neighbors import NearestNeighbors
            # Jarak ke tetangga terdekat ke-(min_samples_final)
            n_neigh = min_samples_final
            
            # Pastikan jumlah data cukup
            if len(X) < n_neigh:
                eps_final = 0.3
            else:
                knn = NearestNeighbors(n_neighbors=n_neigh)
                knn.fit(X)
                distances, _ = knn.kneighbors(X)
                
                # Ambil jarak ke tetangga terjauh untuk K-distance
                k_distances = distances[:, -1]
                k_distances_sorted = np.sort(k_distances)
                
                # Algoritma mencari siku (Elbow point) dengan mencari jarak terjauh titik kurva ke garis lurus
                n_points = len(k_distances_sorted)
                p1 = np.array([0, k_distances_sorted[0]])
                p2 = np.array([n_points - 1, k_distances_sorted[-1]])
                
                max_dist = 0
                elbow_idx = 0
                for i in range(n_points):
                    p0 = np.array([i, k_distances_sorted[i]])
                    # Jarak tegak lurus dari titik p0 ke garis p1-p2
                    dist = np.abs(np.cross(p2 - p1, p1 - p0)) / np.linalg.norm(p2 - p1)
                    if dist > max_dist:
                        max_dist = dist
                        elbow_idx = i
                        
                eps_final = round(float(k_distances_sorted[elbow_idx]), 4)
                
                # Fallback safeguard jika jaraknya bernilai 0
                if eps_final <= 0.0:
                    eps_final = 0.3
        else:
            try:
                eps_final = float(eps_param)
            except ValueError:
                eps_final = 0.3
            
        params["resolved_eps"] = eps_final
        params["resolved_min_samples"] = min_samples_final

        model = DBSCAN(eps=eps_final, min_samples=min_samples_final)
        labels = model.fit_predict(X)
        return labels

    def get_config(self, params: dict) -> dict:
        eps_param = params.get("eps", "auto")
        resolved_eps = params.get("resolved_eps", eps_param)
        min_samples_param = params.get("min_samples", "auto")
        resolved_min_samples = params.get("resolved_min_samples", min_samples_param)
        
        return {
            "model_id": self.id,
            "model_name": self.name,
            "eps": eps_param,
            "resolved_eps": resolved_eps,
            "min_samples": min_samples_param,
            "resolved_min_samples": resolved_min_samples,
            "note": f"DBSCAN mendeteksi noise/outlier (label -1). Epsilon final: {resolved_eps}, min_samples final: {resolved_min_samples}."
        }

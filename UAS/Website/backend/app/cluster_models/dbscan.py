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
        min_samples = int(params.get("min_samples", 5))

        X = x_scaled.values

        if eps_param == "auto" or eps_param is None or str(eps_param).strip().lower() == "auto":
            # Grid search eps seperti script user (menggunakan rentang eps_range yang disediakan user)
            eps_range = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
            best_eps = 0.3
            best_sil = -2.0

            # Mengabaikan warning jika suatu eps tidak menghasilkan >1 cluster
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                for eps_val in eps_range:
                    db = DBSCAN(eps=eps_val, min_samples=min_samples)
                    labels = db.fit_predict(X)
                    
                    n_cl = len(set(labels)) - (1 if -1 in labels else 0)
                    if n_cl > 1:
                        # Metrik silhouette dihitung hanya pada titik non-noise
                        mask_no_noise = labels != -1
                        # Pastikan setelah filter noise masih ada >= 2 cluster valid untuk silhouette
                        valid_labels = set(labels[mask_no_noise])
                        if np.sum(mask_no_noise) > 2 and len(valid_labels) > 1:
                            sil = silhouette_score(X[mask_no_noise], labels[mask_no_noise])
                            if sil > best_sil:
                                best_sil = sil
                                best_eps = eps_val
            
            eps_final = best_eps
            # Simpan eps_final di params agar dapat dibaca di get_config
            params["resolved_eps"] = eps_final
        else:
            try:
                eps_final = float(eps_param)
            except ValueError:
                eps_final = 0.3
            params["resolved_eps"] = eps_final

        model = DBSCAN(eps=eps_final, min_samples=min_samples)
        labels = model.fit_predict(X)
        return labels

    def get_config(self, params: dict) -> dict:
        eps_param = params.get("eps", "auto")
        resolved_eps = params.get("resolved_eps", eps_param)
        min_samples = int(params.get("min_samples", 5))
        
        return {
            "model_id": self.id,
            "model_name": self.name,
            "eps": eps_param,
            "resolved_eps": resolved_eps,
            "min_samples": min_samples,
            "note": f"DBSCAN mendeteksi noise/outlier (label -1). Nilai epsilon final yang digunakan: {resolved_eps}."
        }

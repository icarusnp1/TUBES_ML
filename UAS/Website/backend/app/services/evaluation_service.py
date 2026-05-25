import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


def evaluate_clustering(x_scaled: pd.DataFrame, labels) -> dict:
    unique_labels = sorted(set(labels))

    # DBSCAN nantinya punya label -1 sebagai noise.
    non_noise_labels = [label for label in unique_labels if label != -1]

    result = {
        "n_clusters_detected": len(non_noise_labels),
        "noise_count": int(np.sum(np.array(labels) == -1)),
        "silhouette_score": None,
        "davies_bouldin_index": None,
        "calinski_harabasz_score": None,
        "evaluation_note": ""
    }

    if len(set(labels)) < 2:
        result["evaluation_note"] = "Evaluasi tidak valid karena hanya terbentuk satu cluster."
        return result

    try:
        result["silhouette_score"] = float(silhouette_score(x_scaled, labels))
    except Exception as e:
        result["evaluation_note"] += f"Silhouette error: {e}. "

    try:
        result["davies_bouldin_index"] = float(davies_bouldin_score(x_scaled, labels))
    except Exception as e:
        result["evaluation_note"] += f"Davies-Bouldin error: {e}. "

    try:
        result["calinski_harabasz_score"] = float(calinski_harabasz_score(x_scaled, labels))
    except Exception as e:
        result["evaluation_note"] += f"Calinski-Harabasz error: {e}. "

    return result

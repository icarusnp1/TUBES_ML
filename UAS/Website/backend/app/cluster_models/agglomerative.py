import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from app.cluster_models.base import BaseClusterModel


class AgglomerativeModel(BaseClusterModel):
    id = "agglomerative"
    name = "Agglomerative Hierarchical Clustering"
    description = (
        "Metode hierarchical clustering dari bawah ke atas. "
        "Setiap data awalnya dianggap sebagai cluster sendiri, "
        "lalu cluster yang mirip digabung bertahap."
    )

    def _create_model(self, n_clusters: int, metric: str, linkage: str):
        if linkage == "ward":
            metric = "euclidean"

        try:
            return AgglomerativeClustering(
                n_clusters=n_clusters,
                metric=metric,
                linkage=linkage
            )
        except TypeError:
            return AgglomerativeClustering(
                n_clusters=n_clusters,
                affinity=metric,
                linkage=linkage
            )

    def fit_predict(self, x_scaled: pd.DataFrame, params: dict):
        n_clusters = int(params.get("n_clusters", 4))
        linkage = params.get("linkage", "ward")
        metric = params.get("metric", "euclidean")

        if linkage == "ward":
            metric = "euclidean"

        model = self._create_model(n_clusters, metric, linkage)
        return model.fit_predict(x_scaled)

    def get_config(self, params: dict) -> dict:
        linkage = params.get("linkage", "ward")
        metric = params.get("metric", "euclidean")

        if linkage == "ward":
            metric = "euclidean"

        return {
            "model_id": self.id,
            "model_name": self.name,
            "n_clusters": int(params.get("n_clusters", 4)),
            "metric": metric,
            "linkage": linkage,
            "note": "Jika linkage='ward', metric otomatis memakai Euclidean."
        }

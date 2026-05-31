import os
import uuid
from datetime import datetime

import pandas as pd

from app.core.config import EXPERIMENT_DIR, EXPERIMENTS_METADATA_FILE
from app.registry import PREPROCESSORS, CLUSTER_MODELS, DEFAULT_GAMING_FEATURES
from app.services.dataset_service import get_dataset
from app.services.evaluation_service import evaluate_clustering
from app.utils.visualization import create_visualizations
from app.utils.json_store import read_json_list, append_json_item


def _to_file_url(path: str) -> str:
    normalized = path.replace("\\", "/")
    marker = "/storage/"
    if marker in normalized:
        relative = normalized.split(marker, 1)[1]
    else:
        relative = os.path.basename(normalized)
    return f"/files/{relative}"


def run_experiment(payload: dict) -> dict:
    dataset_id = payload["dataset_id"]
    preprocessing_id = payload.get("preprocessing_id", "ordinal_standard")
    model_id = payload.get("model_id", "agglomerative")
    model_params = payload.get("model_params", {})
    feature_columns = payload.get("feature_columns") or DEFAULT_GAMING_FEATURES
    experiment_name = payload.get("experiment_name") or "Eksperimen Clustering"

    dataset_meta = get_dataset(dataset_id)
    df = pd.read_csv(dataset_meta["path"])

    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"Kolom fitur tidak ditemukan: {missing_features}")

    if preprocessing_id not in PREPROCESSORS:
        raise ValueError("Teknik preprocessing tidak ditemukan.")

    if model_id not in CLUSTER_MODELS:
        raise ValueError("Model clustering tidak ditemukan.")

    preprocessor = PREPROCESSORS[preprocessing_id]
    model = CLUSTER_MODELS[model_id]

    # raw_data = data hasil encoding/imputasi namun belum scaling.
    # x_scaled = data final setelah scaling untuk masuk ke model clustering.
    raw_data, x_scaled, preprocessing_info = preprocessor.transform(df, feature_columns)

    processed_feature_columns = list(raw_data.columns)

    labels = model.fit_predict(x_scaled, model_params)
    evaluation = evaluate_clustering(x_scaled, labels)

    experiment_id = str(uuid.uuid4())
    exp_dir = os.path.join(EXPERIMENT_DIR, experiment_id)
    os.makedirs(exp_dir, exist_ok=True)

    # Data asli + label cluster, untuk download hasil akhir
    result_df = df.loc[raw_data.index].reset_index(drop=True).copy()
    result_df["Cluster"] = labels

    # Data preprocessing numerik + cluster, untuk analisis ringkasan
    raw_numeric_df = raw_data.copy()
    raw_numeric_df["Cluster"] = labels

    cluster_summary = raw_numeric_df.groupby("Cluster")[processed_feature_columns].mean()
    cluster_counts = raw_numeric_df["Cluster"].value_counts().sort_index()

    result_path = os.path.join(exp_dir, "hasil_clustering.csv")
    summary_path = os.path.join(exp_dir, "ringkasan_cluster.csv")
    counts_path = os.path.join(exp_dir, "jumlah_cluster.csv")
    preprocessed_path = os.path.join(exp_dir, "data_preprocessing_numerik.csv")
    scaled_path = os.path.join(exp_dir, "data_scaled.csv")

    result_df.to_csv(result_path, index=False)
    cluster_summary.to_csv(summary_path)
    cluster_counts.to_csv(counts_path, header=["Jumlah Data"])
    raw_numeric_df.to_csv(preprocessed_path, index=False)
    x_scaled.to_csv(scaled_path, index=False)

    model_config = model.get_config(model_params)
    title_suffix = ""
    if model_id == "dbscan":
        resolved_eps = model_config.get("resolved_eps")
        min_samples = model_config.get("resolved_min_samples", model_config.get("min_samples"))
        if resolved_eps is not None:
            title_suffix = f"\n(eps={resolved_eps}, minPts={min_samples})"
    elif model_id == "kmeans":
        k = model_config.get("n_clusters")
        if k is not None:
            title_suffix = f"\n(k={k})"

    image_paths = create_visualizations(
        x_scaled=x_scaled,
        raw_data=raw_data,
        labels=labels,
        experiment_dir=exp_dir,
        feature_columns=processed_feature_columns,
        dendrogram_sample_size=int(payload.get("dendrogram_sample_size", 100)),
        title_suffix=title_suffix
    )

    image_urls = {key: _to_file_url(path) for key, path in image_paths.items()}

    metadata = {
        "id": experiment_id,
        "experiment_name": experiment_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "id": dataset_meta["id"],
            "filename": dataset_meta["original_filename"],
            "rows": dataset_meta["rows"],
            "columns": dataset_meta["columns"]
        },
        "preprocessing": {
            "id": preprocessing_id,
            "name": preprocessor.name,
            "description": preprocessor.description,
            "info": preprocessing_info
        },
        "model": model.get_config(model_params),
        "selected_input_features": feature_columns,
        "processed_output_features": processed_feature_columns,
        "evaluation": evaluation,
        "cluster_counts": {str(k): int(v) for k, v in cluster_counts.to_dict().items()},
        "cluster_summary": cluster_summary.reset_index().to_dict(orient="records"),
        "files": {
            "result_csv": _to_file_url(result_path),
            "summary_csv": _to_file_url(summary_path),
            "counts_csv": _to_file_url(counts_path),
            "preprocessed_csv": _to_file_url(preprocessed_path),
            "scaled_csv": _to_file_url(scaled_path),
        },
        "images": image_urls
    }

    append_json_item(EXPERIMENTS_METADATA_FILE, metadata)
    return metadata


def list_experiments() -> list:
    return read_json_list(EXPERIMENTS_METADATA_FILE)


def get_experiment(experiment_id: str) -> dict:
    experiments = list_experiments()
    for exp in experiments:
        if exp["id"] == experiment_id:
            return exp
    raise ValueError("Eksperimen tidak ditemukan.")

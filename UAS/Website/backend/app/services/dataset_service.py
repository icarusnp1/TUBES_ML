import os
import uuid
import shutil
from datetime import datetime

import pandas as pd
from fastapi import UploadFile

from app.core.config import DATASET_DIR, DATASETS_METADATA_FILE
from app.utils.json_store import read_json_list, append_json_item


def save_dataset(file: UploadFile) -> dict:
    if not file.filename.lower().endswith(".csv"):
        raise ValueError("File harus berformat CSV.")

    dataset_id = str(uuid.uuid4())
    safe_filename = file.filename.replace(" ", "_")
    stored_filename = f"{dataset_id}_{safe_filename}"
    stored_path = os.path.join(DATASET_DIR, stored_filename)

    with open(stored_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    df = pd.read_csv(stored_path)

    metadata = {
        "id": dataset_id,
        "original_filename": file.filename,
        "stored_filename": stored_filename,
        "path": stored_path,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    append_json_item(DATASETS_METADATA_FILE, metadata)
    return metadata


def list_datasets() -> list:
    return read_json_list(DATASETS_METADATA_FILE)


def get_dataset(dataset_id: str) -> dict:
    datasets = list_datasets()
    for dataset in datasets:
        if dataset["id"] == dataset_id:
            return dataset
    raise ValueError("Dataset tidak ditemukan.")


def preview_dataset(dataset_id: str, limit: int = 10) -> dict:
    dataset = get_dataset(dataset_id)
    df = pd.read_csv(dataset["path"])

    return {
        "dataset": dataset,
        "preview": df.head(limit).fillna("").to_dict(orient="records"),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
    }

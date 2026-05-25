import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

STORAGE_DIR = os.path.join(BASE_DIR, "storage")
DATASET_DIR = os.path.join(STORAGE_DIR, "datasets")
EXPERIMENT_DIR = os.path.join(STORAGE_DIR, "experiments")
METADATA_DIR = os.path.join(STORAGE_DIR, "metadata")

DATASETS_METADATA_FILE = os.path.join(METADATA_DIR, "datasets.json")
EXPERIMENTS_METADATA_FILE = os.path.join(METADATA_DIR, "experiments.json")

for folder in [STORAGE_DIR, DATASET_DIR, EXPERIMENT_DIR, METADATA_DIR]:
    os.makedirs(folder, exist_ok=True)

for file_path in [DATASETS_METADATA_FILE, EXPERIMENTS_METADATA_FILE]:
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("[]")

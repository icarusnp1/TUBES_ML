# ============================================================
# PROGRAM 1
# MEMBUAT SAMPLE 1000 DATA DARI DATASET ASLI
# Dataset: online_gaming_behavior_dataset.csv
# Output : dataset_sample/online_gaming_behavior_sample_1000.csv
# ============================================================

import os
import pandas as pd


# ============================================================
# KONFIGURASI
# ============================================================

INPUT_CSV_PATH = "online_gaming_behavior_dataset.csv"

OUTPUT_DIR = "dataset_sample"
OUTPUT_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "online_gaming_behavior_sample_1000.csv"
)

SAMPLE_SIZE = 1000
RANDOM_STATE = 42


# ============================================================
# BUAT FOLDER OUTPUT
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATASET ASLI
# ============================================================

df = pd.read_csv(INPUT_CSV_PATH)

print("\n=== DATASET ASLI ===")
print("Jumlah baris dan kolom:", df.shape)
print("\n5 data pertama:")
print(df.head())


# ============================================================
# VALIDASI JUMLAH DATA
# ============================================================

sample_size = min(SAMPLE_SIZE, len(df))

if len(df) < SAMPLE_SIZE:
    print(f"\nDataset kurang dari {SAMPLE_SIZE} baris.")
    print(f"Program akan mengambil seluruh data sebanyak {len(df)} baris.")
else:
    print(f"\nDataset memiliki cukup data. Program mengambil {SAMPLE_SIZE} baris.")


# ============================================================
# SAMPLING DATA
# ============================================================
# random_state membuat hasil sampling tetap sama
# setiap kali program dijalankan.

df_sample = df.sample(
    n=sample_size,
    random_state=RANDOM_STATE
).reset_index(drop=True)


# ============================================================
# SIMPAN HASIL SAMPLE
# ============================================================

df_sample.to_csv(OUTPUT_CSV_PATH, index=False)

print("\n=== SAMPLE BERHASIL DIBUAT ===")
print("Jumlah baris dan kolom sample:", df_sample.shape)
print("File disimpan di:", OUTPUT_CSV_PATH)

print("\n5 data pertama sample:")
print(df_sample.head())
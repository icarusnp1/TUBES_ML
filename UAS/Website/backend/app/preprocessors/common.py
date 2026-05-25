import pandas as pd

DIFFICULTY_MAPPING = {
    "Easy": 0,
    "Medium": 1,
    "Hard": 2
}


def apply_common_cleaning(df: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, dict]:
    """
    Membersihkan dan mengubah kolom pilihan user menjadi data numerik siap scaling.

    Aturan:
    1. Kolom numerik dipakai langsung.
    2. GameDifficulty wajib ordinal encoding:
       Easy = 0, Medium = 1, Hard = 2 karena Easy < Medium < Hard.
    3. Kolom kategori lain otomatis One-Hot Encoding.
    4. Missing value numerik diisi median.
    5. Missing value kategori diisi modus.
    """

    if not feature_columns:
        raise ValueError("Minimal pilih satu kolom fitur untuk clustering.")

    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"Kolom fitur tidak ditemukan: {missing_features}")

    selected = df[feature_columns].copy()

    info = {
        "input_features": feature_columns,
        "numeric_features": [],
        "ordinal_features": [],
        "one_hot_features": [],
        "output_features": [],
        "ordinal_encoding": {},
        "one_hot_encoding": {},
        "missing_value_strategy": {
            "numeric": "median",
            "categorical": "mode"
        },
        "missing_values_before": selected.isnull().sum().to_dict(),
        "missing_values_after": {},
    }

    numeric_parts = []
    categorical_parts = []

    for col in selected.columns:
        series = selected[col]

        # 1. GameDifficulty memakai ordinal encoding wajib
        if col == "GameDifficulty":
            cleaned = series.astype(str).str.strip()

            # Jika ada missing value, astype(str) dapat menghasilkan "nan".
            # Kita kembalikan nilai invalid menjadi NaN setelah mapping.
            encoded = cleaned.map(DIFFICULTY_MAPPING)

            if encoded.isnull().sum() > 0:
                median_value = encoded.median()
                if pd.isna(median_value):
                    median_value = 1  # default Medium jika seluruhnya gagal mapping
                encoded = encoded.fillna(median_value)

            encoded = encoded.astype(float)
            numeric_parts.append(encoded.to_frame(col))

            info["ordinal_features"].append(col)
            info["ordinal_encoding"][col] = {
                "Easy": 0,
                "Medium": 1,
                "Hard": 2,
                "reason": "GameDifficulty memiliki urutan alami: Easy < Medium < Hard."
            }

        # 2. Kolom numerik dipakai langsung
        elif pd.api.types.is_numeric_dtype(series):
            numeric_series = pd.to_numeric(series, errors="coerce")

            if numeric_series.isnull().sum() > 0:
                median_value = numeric_series.median()
                if pd.isna(median_value):
                    median_value = 0
                numeric_series = numeric_series.fillna(median_value)

            numeric_parts.append(numeric_series.to_frame(col))
            info["numeric_features"].append(col)

        # 3. Kolom kategori lain memakai One-Hot Encoding
        else:
            cat_series = series.astype("object")

            if cat_series.isnull().sum() > 0:
                mode_value = cat_series.mode(dropna=True)
                fill_value = mode_value.iloc[0] if len(mode_value) > 0 else "Unknown"
                cat_series = cat_series.fillna(fill_value)

            cat_series = cat_series.astype(str).str.strip()
            cat_series = cat_series.replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})

            categorical_parts.append((col, cat_series))
            info["one_hot_features"].append(col)

    # Gabungkan fitur numerik dan ordinal
    if numeric_parts:
        processed = pd.concat(numeric_parts, axis=1)
    else:
        processed = pd.DataFrame(index=selected.index)

    # One-hot encoding untuk fitur kategorikal non-ordinal
    for col, cat_series in categorical_parts:
        dummies = pd.get_dummies(cat_series, prefix=col, dtype=float)

        # Menyimpan daftar hasil kolom one-hot untuk metadata
        info["one_hot_encoding"][col] = list(dummies.columns)

        processed = pd.concat([processed, dummies], axis=1)

    if processed.shape[1] < 2:
        raise ValueError(
            "Jumlah fitur hasil preprocessing kurang dari 2. "
            "Pilih minimal dua fitur numerik/kategori agar clustering lebih bermakna."
        )

    # Pastikan semuanya numerik float
    for col in processed.columns:
        processed[col] = pd.to_numeric(processed[col], errors="coerce")

    # Jika masih ada NaN setelah proses, isi dengan median kolom
    for col in processed.columns:
        if processed[col].isnull().sum() > 0:
            median_value = processed[col].median()
            if pd.isna(median_value):
                median_value = 0
            processed[col] = processed[col].fillna(median_value)

    info["missing_values_after"] = processed.isnull().sum().to_dict()
    info["output_features"] = list(processed.columns)
    info["output_feature_count"] = len(processed.columns)

    return processed, info

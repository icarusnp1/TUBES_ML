import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from app.preprocessors.base import BasePreprocessor


class MeanDropMinMaxPreprocessor(BasePreprocessor):
    id = "mean_drop_minmax"
    name = "Mean Imputation + Drop Duplicates + MinMax"
    description = (
        "Penanganan Missing Value menggunakan mean, Penanganan data duplikat "
        "dengan cara dihapus, dan normalisasi menggunakan MinMaxScaler. "
        "Cocok untuk Country Dataset."
    )

    def transform(self, df: pd.DataFrame, feature_columns: list[str]):
        if not feature_columns:
            raise ValueError("Minimal pilih satu kolom fitur untuk clustering.")

        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            raise ValueError(f"Kolom fitur tidak ditemukan: {missing_features}")

        selected = df[feature_columns].copy()

        info = {
            "input_features": feature_columns,
            "missing_values_before": selected.isnull().sum().to_dict(),
        }

        # 1. Fill missing value with mean untuk kolom numerik
        for col in selected.columns:
            if pd.api.types.is_numeric_dtype(selected[col]):
                mean_val = selected[col].mean()
                if pd.isna(mean_val):
                    mean_val = 0
                selected[col] = selected[col].fillna(mean_val)
            else:
                mode_val = selected[col].mode(dropna=True)
                fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "Unknown"
                selected[col] = selected[col].fillna(fill_val)

        # 2. Drop duplicates
        before_drop = len(selected)
        selected = selected.drop_duplicates(keep='first')
        after_drop = len(selected)
        
        info["duplicates_dropped"] = before_drop - after_drop

        # 3. Ordinal Encoding untuk kolom bertipe teks/kategori
        from sklearn.preprocessing import LabelEncoder
        for col in selected.columns:
            if selected[col].dtype == 'object' or selected[col].dtype.name == 'category':
                le = LabelEncoder()
                # Ubah semua jadi string untuk menghindari error tipe campuran
                selected[col] = selected[col].astype(str) 
                selected[col] = le.fit_transform(selected[col])

        # 4. Pastikan semuanya terdeteksi sebagai tipe data numerik
        for col in selected.columns:
            selected[col] = pd.to_numeric(selected[col], errors='coerce').fillna(0)

        # 3. MinMaxScaler
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(selected)
        scaled_df = pd.DataFrame(scaled_values, columns=selected.columns, index=selected.index)

        info["missing_values_after"] = selected.isnull().sum().to_dict()
        info["output_features"] = list(selected.columns)
        info["output_feature_count"] = len(selected.columns)
        info["scaler"] = "MinMaxScaler"
        
        return selected, scaled_df, info

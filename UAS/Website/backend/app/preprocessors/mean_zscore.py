import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from app.preprocessors.base import BasePreprocessor


class MeanZScorePreprocessor(BasePreprocessor):
    id = "mean_zscore"
    name = "Mean Imputation + Z-Score"
    description = (
        "Mengisi nilai yang kosong (missing values) menggunakan rata-rata (mean), "
        "kemudian menstandarkan fitur dengan menghilangkan nilai rata-rata (mean = 0) "
        "dan menskalakannya ke varians unit (Z-Score Normalization)."
    )

    def transform(self, df: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        if not feature_columns:
            raise ValueError("Minimal pilih satu kolom fitur untuk clustering.")
            
        selected = df[feature_columns].copy()
        
        info = {
            "input_features": feature_columns,
            "missing_values_before": selected.isnull().sum().to_dict(),
        }
        
        # Pastikan data berupa numerik karena mean imputation butuh angka
        for col in selected.columns:
            selected[col] = pd.to_numeric(selected[col], errors='coerce')

        # 1. Mean Imputation menggunakan SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        np_imputed = imputer.fit_transform(selected)
        df_imputed = pd.DataFrame(np_imputed, columns=selected.columns, index=selected.index)

        # 2. Z-Score Normalization menggunakan StandardScaler
        scaler = StandardScaler()
        np_scaled = scaler.fit_transform(df_imputed)
        df_scaled = pd.DataFrame(np_scaled, columns=selected.columns, index=selected.index)
        
        info["missing_values_after"] = df_scaled.isnull().sum().to_dict()
        info["output_features"] = feature_columns
        info["scaler"] = "StandardScaler"
        
        return df_imputed, df_scaled, info

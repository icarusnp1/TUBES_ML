import pandas as pd
from sklearn.preprocessing import RobustScaler

from app.preprocessors.base import BasePreprocessor
from app.preprocessors.common import apply_common_cleaning


class RobustPreprocessor(BasePreprocessor):
    id = "ordinal_robust"
    name = "Ordinal + One-Hot + RobustScaler"
    description = (
        "GameDifficulty menggunakan ordinal encoding "
        "(Easy=0, Medium=1, Hard=2). Kolom kategori lain otomatis One-Hot Encoding. "
        "Seluruh fitur numerik akhir discales menggunakan RobustScaler."
    )

    def transform(self, df: pd.DataFrame, feature_columns: list[str]):
        raw_data, info = apply_common_cleaning(df, feature_columns)

        scaler = RobustScaler()
        scaled = scaler.fit_transform(raw_data)
        scaled_df = pd.DataFrame(scaled, columns=raw_data.columns)

        info["scaler"] = "RobustScaler"
        info["scaler_reason"] = "RobustScaler lebih tahan terhadap outlier dibanding scaler biasa."

        return raw_data, scaled_df, info

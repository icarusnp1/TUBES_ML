import pandas as pd
from sklearn.preprocessing import StandardScaler

from app.preprocessors.base import BasePreprocessor
from app.preprocessors.common import apply_common_cleaning


class StandardPreprocessor(BasePreprocessor):
    id = "ordinal_standard"
    name = "Ordinal + One-Hot + StandardScaler"
    description = (
        "GameDifficulty menggunakan ordinal encoding "
        "(Easy=0, Medium=1, Hard=2). Kolom kategori lain otomatis One-Hot Encoding. "
        "Seluruh fitur numerik akhir discales menggunakan StandardScaler."
    )

    def transform(self, df: pd.DataFrame, feature_columns: list[str]):
        raw_data, info = apply_common_cleaning(df, feature_columns)

        scaler = StandardScaler()
        scaled = scaler.fit_transform(raw_data)
        scaled_df = pd.DataFrame(scaled, columns=raw_data.columns)

        info["scaler"] = "StandardScaler"
        info["scaler_reason"] = (
            "StandardScaler digunakan karena clustering berbasis jarak "
            "dan fitur memiliki skala nilai yang berbeda."
        )

        return raw_data, scaled_df, info

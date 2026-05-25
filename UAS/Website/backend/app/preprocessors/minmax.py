import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from app.preprocessors.base import BasePreprocessor
from app.preprocessors.common import apply_common_cleaning


class MinMaxPreprocessor(BasePreprocessor):
    id = "ordinal_minmax"
    name = "Ordinal + One-Hot + MinMaxScaler"
    description = (
        "GameDifficulty menggunakan ordinal encoding "
        "(Easy=0, Medium=1, Hard=2). Kolom kategori lain otomatis One-Hot Encoding. "
        "Seluruh fitur numerik akhir diubah ke rentang 0 sampai 1."
    )

    def transform(self, df: pd.DataFrame, feature_columns: list[str]):
        raw_data, info = apply_common_cleaning(df, feature_columns)

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(raw_data)
        scaled_df = pd.DataFrame(scaled, columns=raw_data.columns)

        info["scaler"] = "MinMaxScaler"
        info["scaler_reason"] = "MinMaxScaler membuat seluruh fitur berada pada rentang 0 sampai 1."

        return raw_data, scaled_df, info

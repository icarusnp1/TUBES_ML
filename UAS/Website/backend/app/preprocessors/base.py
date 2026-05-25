from abc import ABC, abstractmethod
import pandas as pd


class BasePreprocessor(ABC):
    id: str
    name: str
    description: str

    @abstractmethod
    def transform(self, df: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        Return:
        - raw_selected_df: dataframe fitur setelah encoding/imputasi namun sebelum scaling
        - scaled_df: dataframe final yang siap dipakai model
        - info: metadata preprocessing
        """
        pass

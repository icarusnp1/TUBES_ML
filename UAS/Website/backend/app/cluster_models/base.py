from abc import ABC, abstractmethod
import pandas as pd


class BaseClusterModel(ABC):
    id: str
    name: str
    description: str

    @abstractmethod
    def fit_predict(self, x_scaled: pd.DataFrame, params: dict):
        pass

    @abstractmethod
    def get_config(self, params: dict) -> dict:
        pass

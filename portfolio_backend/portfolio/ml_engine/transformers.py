from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class RollingStandardScaler(BaseEstimator, TransformerMixin):
    window: int = 60
    min_periods: int = 20

    def fit(self, X: Any, y: Any = None) -> "RollingStandardScaler":
        return self

    def transform(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            values = X.values
            columns = list(X.columns)
        else:
            values = np.asarray(X)
            columns = None
        if values.ndim != 2:
            return values
        df = pd.DataFrame(values, columns=columns)
        mean = df.rolling(self.window, min_periods=self.min_periods).mean()
        std = df.rolling(self.window, min_periods=self.min_periods).std().replace(0, np.nan)
        scaled = (df - mean) / std
        return scaled.fillna(0.0).values

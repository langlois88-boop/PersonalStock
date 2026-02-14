from typing import List

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS: List[str] = [
    "rsi_14",
    "vol_zscore",
    "return_20d",
    "roe",
    "debt_to_equity",
    "news_sentiment",
    "news_count",
    "fred_rate",
]


def build_model(random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=500,
                max_depth=5,
                min_samples_split=8,
                min_samples_leaf=10,
                max_features="sqrt",
                random_state=random_state,
                class_weight="balanced",
            ),
        ),
    ])


def train_model(X: np.ndarray, y: np.ndarray) -> Pipeline:
    model = build_model()
    model.fit(X, y)
    return model


def save_model(model: Pipeline, path: str) -> None:
    joblib.dump(model, path)


def load_model(path: str) -> Pipeline:
    return joblib.load(path)

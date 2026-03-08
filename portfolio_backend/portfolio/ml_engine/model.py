from typing import List

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .feature_registry import RECOMMENDER_FEATURE_NAMES


FEATURE_COLUMNS: List[str] = list(RECOMMENDER_FEATURE_NAMES)


def build_model(random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
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


def load_model_payload(path: str) -> dict:
    payload = joblib.load(path)
    if isinstance(payload, dict):
        model = payload.get("model")
        features = payload.get("features")
        model_type = payload.get("model_type")
        return {
            "model": model,
            "features": features,
            "model_type": model_type,
        }
    return {"model": payload, "features": FEATURE_COLUMNS, "model_type": None}


def load_model(path: str) -> Pipeline:
    payload = load_model_payload(path)
    return payload.get("model")

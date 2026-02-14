from typing import Dict, List

import numpy as np

from .model import FEATURE_COLUMNS, load_model
from .processor import DataMerger


def build_feature_vector(features: Dict[str, float]) -> np.ndarray:
    row = [features.get(col) if features.get(col) is not None else 0.0 for col in FEATURE_COLUMNS]
    return np.array(row, dtype=float)


def generate_recommendations(symbols: List[str], model_path: str) -> List[Dict[str, float]]:
    merger = DataMerger()
    model = load_model(model_path)

    results = []
    for symbol in symbols:
        features = merger.merge(symbol)
        if not features:
            continue
        vector = build_feature_vector(features)
        proba = float(model.predict_proba([vector])[0][1])
        results.append({
            "symbol": symbol,
            "prob_up_15d": round(proba, 4),
            "recommendation": "BUY" if proba >= 0.7 else "HOLD",
            "features": features,
        })

    results.sort(key=lambda x: x["prob_up_15d"], reverse=True)
    return results

import os
from typing import Dict, List

import numpy as np

from .model import FEATURE_COLUMNS, load_model_payload
from .processor import DataMerger


def build_feature_vector(features: Dict[str, float], feature_list: List[str]) -> np.ndarray:
    row = [features.get(col) if features.get(col) is not None else 0.0 for col in feature_list]
    return np.array(row, dtype=float)


def generate_recommendations(symbols: List[str], model_path: str) -> List[Dict[str, float]]:
    merger = DataMerger()
    payload = load_model_payload(model_path)
    model = payload.get("model")
    feature_list = payload.get("features") or FEATURE_COLUMNS
    debug = os.getenv('RECOMMENDER_DEBUG', 'false').lower() in {'1', 'true', 'yes', 'y'}
    buy_threshold = float(os.getenv('RECOMMENDER_BUY_THRESHOLD', '0.7'))

    results = []
    if debug:
        print(f"\n--- SCANNER START ({len(symbols)} tickers) ---")
    for symbol in symbols:
        features = merger.merge(symbol)
        if not features:
            if debug:
                print(f"❌ {symbol}: Aucune donnée (Merger vide)")
            continue
        vector = build_feature_vector(features, feature_list)
        if np.all(vector == 0):
            if debug:
                print(f"⚠️ {symbol}: Vecteur nul (Problème de données techniques)")
            continue
        if hasattr(model, "predict_proba"):
            score = float(model.predict_proba([vector])[0][1])
            recommendation = "BUY" if score >= buy_threshold else "HOLD"
            if debug:
                status_icon = "✅" if recommendation == "BUY" else "⏳"
                print(f"{status_icon} {symbol}: Score={score:.4f} | Recommandation: {recommendation}")
            results.append({
                "symbol": symbol,
                "prob_up_15d": round(score, 4),
                "recommendation": recommendation,
                "features": features,
            })
        else:
            pred = float(model.predict([vector])[0])
            recommendation = "BUY" if pred >= 0.02 else "HOLD"
            if debug:
                print(f"📊 {symbol}: Prédiction retour={pred:.2%}")
            results.append({
                "symbol": symbol,
                "predicted_20d_return": round(pred, 4),
                "recommendation": recommendation,
                "features": features,
            })

    results.sort(key=lambda x: x.get("prob_up_15d", x.get("predicted_20d_return", 0)), reverse=True)
    return results

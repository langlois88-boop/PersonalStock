from __future__ import annotations

import os

from ..ml_engine.engine.data_fusion import DataFusionEngine
from ..ml_engine.backtester import (
    apply_feature_weighting_to_signal,
    get_model_path,
    load_or_train_model,
)

# 2026-08-23 : DanasBroker (chat LLM local via Ollama/deepseek-r1) et
# MarketContext ont été retirés -- testé en direct (ask_danas) : 21s de
# latence, tags <think> non nettoyés dans la réponse brute, et le
# contexte portefeuille/marché censé être injecté ne l'était pas
# vraiment ("Je n'ai accès qu'aux données fournies" en réponse à une
# question sur le portefeuille). Ni AICenterView (frontend) ni
# useAICenter.js ne l'utilisaient de toute façon -- aucune perte
# fonctionnelle réelle. DanasMLRouter ci-dessous reste : c'est un
# routeur de signal ML pur (pas de LLM), toujours utilisé par de vraies
# tâches de scan dans tasks.py.


class DanasMLRouter:
    def __init__(self) -> None:
        self.default_universe = os.getenv("DANAS_DEFAULT_UNIVERSE", "BLUECHIP")

    def infer_universe(self, symbol: str, price: float | None, market_type: str | None) -> str:
        if market_type:
            return market_type
        if price is not None and price < 7:
            return "PENNY"
        return self.default_universe

    def predict(self, symbol: str, market_type: str | None = None) -> dict[str, float | None]:
        try:
            engine = DataFusionEngine(symbol)
            frame = engine.fuse_all()
            if frame is None or frame.empty:
                return {"signal": None, "confidence": None}

            try:
                price = float(frame.tail(1).iloc[0].get("Close") or 0.0)
            except Exception:
                price = None

            universe = self.infer_universe(symbol, price, market_type)
            payload = load_or_train_model(frame, model_path=get_model_path(universe))
            if not payload or not payload.get("model"):
                return {"signal": None, "confidence": None}

            last_row = frame.tail(1).copy()
            feature_list = payload.get("features") or []
            for col in feature_list:
                if col not in last_row.columns:
                    last_row[col] = 0.0
            features = last_row[feature_list].fillna(0).values
            try:
                proba = float(payload["model"].predict_proba(features)[0][1])
                signal = 1.0 if proba >= 0.5 else 0.0
                confidence = max(proba, 1.0 - proba)
            except Exception:
                raw = float(payload["model"].predict(features)[0]) if hasattr(payload["model"], "predict") else 0.0
                signal = 1.0 if raw >= 0.5 else 0.0
                confidence = abs(raw - 0.5) + 0.5

            weighted = apply_feature_weighting_to_signal(float(signal), last_row.iloc[0], symbol)
            return {
                "signal": round(weighted, 4),
                "confidence": round(float(confidence) * 100, 2) if confidence is not None else None,
            }
        except Exception:
            return {"signal": None, "confidence": None}

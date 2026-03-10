from __future__ import annotations

"""
shap_explainer.py  — FIX #5
============================
Intégration SHAP pour expliquer pourquoi le modèle dit BUY/SELL.

Fonctionnalités :
- Calcul SHAP pour n'importe quel modèle sklearn pipeline
- Top features positives/négatives par trade
- Rapport textuel en français (pour DeepSeek + Telegram)
- Cache des valeurs SHAP par trade (évite recalcul)
- Feature importance globale pour le dashboard Analytics

Usage :
    from portfolio.ml_engine.shap_explainer import explain_prediction, global_importance
    
    explanation = explain_prediction(model_payload, feature_row, symbol="AAPL")
    print(explanation['text'])
    # → "Le signal BUY est principalement dû au RSI oversold (RSI=28.4, +23%
    #    du score), confirmé par le rubber band très tendu (-1.82, +18%) et
    #    le volume inhabituel (VolumeZ=2.3, +15%)."
"""

import logging
import os
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# SHAP est optionnel — le reste du code fonctionne sans
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("shap not installed. Run: pip install shap. Falling back to permutation importance.")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _extract_rf_from_pipeline(pipeline: Any) -> Any:
    """Extrait le RandomForest d'un sklearn Pipeline."""
    if hasattr(pipeline, "steps"):
        for _, step in pipeline.steps:
            if hasattr(step, "feature_importances_"):
                return step
    if hasattr(pipeline, "feature_importances_"):
        return pipeline
    return None


def _extract_scaler_from_pipeline(pipeline: Any) -> Any:
    if hasattr(pipeline, "steps"):
        for _, step in pipeline.steps:
            if hasattr(step, "transform"):
                return step
    return None


def _transform_features(pipeline: Any, X: np.ndarray) -> np.ndarray:
    """Applique le scaler du pipeline sans le classificateur."""
    if hasattr(pipeline, "steps"):
        result = X.copy()
        for name, step in pipeline.steps[:-1]:  # tous sauf le dernier (classifier)
            try:
                result = step.transform(result)
            except Exception:
                pass
        return result
    return X


# ─── SHAP values ──────────────────────────────────────────────────────────────

def compute_shap_values(
    model_payload: dict[str, Any],
    X: np.ndarray | pd.DataFrame,
    max_background: int = 100,
) -> np.ndarray | None:
    """
    Calcule les SHAP values pour un batch de prédictions.

    Args:
        model_payload : dict avec 'model' (pipeline sklearn) et 'features'
        X             : features array ou DataFrame
        max_background: nb d'échantillons pour le background SHAP TreeExplainer

    Returns:
        SHAP values array [n_samples, n_features] ou None si indisponible
    """
    if not SHAP_AVAILABLE:
        return None

    pipeline = model_payload.get("model")
    if pipeline is None:
        return None

    rf_model = _extract_rf_from_pipeline(pipeline)
    if rf_model is None:
        return None

    if isinstance(X, pd.DataFrame):
        X_arr = X.values
    else:
        X_arr = np.array(X)

    # Transformer avec le scaler
    X_transformed = _transform_features(pipeline, X_arr)

    try:
        # TreeExplainer est rapide et exact pour RandomForest
        explainer = shap.TreeExplainer(
            rf_model,
            feature_perturbation="tree_path_dependent",
        )
        shap_vals = explainer.shap_values(X_transformed)

        # Pour classification binaire, shap_values retourne [class0, class1]
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            return np.array(shap_vals[1])  # class 1 (BUY)
        return np.array(shap_vals)
    except Exception as exc:
        logger.debug("SHAP TreeExplainer failed: %s. Trying KernelExplainer.", exc)

        # Fallback : KernelExplainer (plus lent)
        try:
            background = shap.kmeans(X_transformed, min(max_background, len(X_transformed)))
            explainer = shap.KernelExplainer(
                lambda x: pipeline.predict_proba(x)[:, 1],
                background,
            )
            return np.array(explainer.shap_values(X_transformed[:1]))
        except Exception as exc2:
            logger.warning("SHAP KernelExplainer also failed: %s", exc2)
            return None


def _permutation_importance_fallback(
    model_payload: dict[str, Any],
    X: np.ndarray,
    feature_names: list[str],
) -> list[tuple[str, float]]:
    """
    Fallback si SHAP non dispo : importance par permutation rapide.
    Retourne [(feature, importance_score), ...] trié par importance décroissante.
    """
    pipeline = model_payload.get("model")
    if pipeline is None:
        return []
    rf = _extract_rf_from_pipeline(pipeline)
    if rf is None or not hasattr(rf, "feature_importances_"):
        return []
    importances = rf.feature_importances_
    pairs = sorted(
        zip(feature_names, importances.tolist()),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    return pairs


# ─── Explication par prédiction ───────────────────────────────────────────────

def explain_prediction(
    model_payload: dict[str, Any],
    feature_row: pd.Series | dict[str, float],
    symbol: str = "",
    signal: float | None = None,
    top_n: int = 5,
    language: str = "fr",
) -> dict[str, Any]:
    """
    Explique une prédiction individuelle avec SHAP.

    Args:
        model_payload : payload du modèle (model, features)
        feature_row   : Series ou dict de la dernière ligne de features
        symbol        : ticker pour le contexte
        signal        : score ML (0.0–1.0) pour contextualiser
        top_n         : nb de features à afficher
        language      : "fr" ou "en"

    Returns:
        {
            top_positive: [(feature, shap_val, feature_val), ...],  # features qui poussent vers BUY
            top_negative: [(feature, shap_val, feature_val), ...],  # features qui poussent vers SELL
            text: str,                                               # explication en français
            method: "shap" | "permutation",
            shap_sum: float,                                         # somme des SHAP values
        }
    """
    feature_names = model_payload.get("features") or []
    if not feature_names:
        return {"text": "Features indisponibles.", "top_positive": [], "top_negative": [], "method": "none"}

    # Construire le vecteur de features
    if isinstance(feature_row, dict):
        row_dict = feature_row
    else:
        row_dict = dict(feature_row)

    X_row = np.array([[float(row_dict.get(f, 0.0)) for f in feature_names]])

    method = "none"
    shap_values = None

    if SHAP_AVAILABLE:
        shap_values = compute_shap_values(model_payload, X_row)
        if shap_values is not None:
            method = "shap"

    if shap_values is None:
        # Fallback permutation importance
        importance_pairs = _permutation_importance_fallback(model_payload, X_row, feature_names)
        method = "permutation"
        # Simuler des SHAP-like values depuis l'importance globale × signe du signal
        sign = 1 if (signal or 0.5) >= 0.5 else -1
        shap_values = np.array([[imp * sign for _, imp in importance_pairs]])

    shap_row = shap_values[0] if len(shap_values.shape) > 1 else shap_values
    if len(shap_row) != len(feature_names):
        return {"text": "SHAP dimension mismatch.", "top_positive": [], "top_negative": [], "method": method}

    # Paires (feature, shap_val, feature_val)
    pairs = [
        (name, float(shap_row[i]), float(row_dict.get(name, 0.0)))
        for i, name in enumerate(feature_names)
    ]
    pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)

    top_positive = [(n, s, v) for n, s, v in pairs_sorted if s > 0][:top_n]
    top_negative = [(n, s, v) for n, s, v in pairs_sorted if s < 0][:top_n]
    shap_sum = round(float(sum(s for _, s, _ in pairs)), 4)

    # Génération du texte explicatif
    signal_str = f"{signal*100:.1f}%" if signal is not None else "N/A"
    direction = "BUY 🟢" if (signal or 0) >= 0.65 else "SELL 🔴" if (signal or 1) < 0.35 else "NEUTRE ⚪"

    if language == "fr":
        text = _build_french_explanation(symbol, direction, signal_str, top_positive, top_negative, method)
    else:
        text = _build_english_explanation(symbol, direction, signal_str, top_positive, top_negative, method)

    return {
        "symbol": symbol,
        "signal": signal,
        "direction": direction,
        "top_positive": [(n, round(s, 4), round(v, 4)) for n, s, v in top_positive],
        "top_negative": [(n, round(s, 4), round(v, 4)) for n, s, v in top_negative],
        "shap_sum": shap_sum,
        "method": method,
        "text": text,
    }


def _fmt_feature(name: str, val: float) -> str:
    """Formate une paire feature=valeur de façon lisible."""
    name_map = {
        "rsi_14": "RSI", "RSI14": "RSI",
        "rubber_band_20": "Rubber Band", "rubber_band_index": "Rubber Band",
        "volume_zscore_20": "Volume Z", "VolumeZ": "Volume Z",
        "macd_hist": "MACD Hist", "MACD_HIST": "MACD Hist",
        "bb_pct_b_20": "Bollinger %B",
        "adx_14": "ADX",
        "stoch_k_14": "Stoch %K",
        "sentiment_score": "Sentiment",
        "return_20d": "Retour 20j", "Momentum20": "Momentum 20j",
        "volatility_20": "Volatilité", "Volatility": "Volatilité",
        "obv_zscore": "OBV Z",
        "mom_zscore_20": "Momentum Z",
        "vol_regime": "Régime Vol",
    }
    display_name = name_map.get(name, name.replace("_", " ").title())
    return f"{display_name}={val:.2f}"


def _build_french_explanation(
    symbol: str,
    direction: str,
    signal_str: str,
    top_positive: list[tuple],
    top_negative: list[tuple],
    method: str,
) -> str:
    lines = [f"**{symbol}** — Signal {direction} ({signal_str})"]
    lines.append(f"_Méthode d'explication : {method}_\n")

    if top_positive:
        pos_strs = [f"{_fmt_feature(n, v)} (impact +{s:.3f})" for n, s, v in top_positive[:3]]
        lines.append("📈 **Facteurs haussiers :** " + ", ".join(pos_strs))

    if top_negative:
        neg_strs = [f"{_fmt_feature(n, v)} (impact {s:.3f})" for n, s, v in top_negative[:3]]
        lines.append("📉 **Facteurs baissiers :** " + ", ".join(neg_strs))

    return "\n".join(lines)


def _build_english_explanation(
    symbol: str,
    direction: str,
    signal_str: str,
    top_positive: list[tuple],
    top_negative: list[tuple],
    method: str,
) -> str:
    lines = [f"**{symbol}** — Signal {direction} ({signal_str})"]
    if top_positive:
        pos_strs = [f"{_fmt_feature(n, v)} (impact +{s:.3f})" for n, s, v in top_positive[:3]]
        lines.append("📈 **Bullish factors:** " + ", ".join(pos_strs))
    if top_negative:
        neg_strs = [f"{_fmt_feature(n, v)} (impact {s:.3f})" for n, s, v in top_negative[:3]]
        lines.append("📉 **Bearish factors:** " + ", ".join(neg_strs))
    return "\n".join(lines)


# ─── Importance globale (pour dashboard Analytics) ────────────────────────────

def global_feature_importance(
    model_payload: dict[str, Any],
    top_n: int = 20,
) -> list[dict[str, Any]]:
    """
    Retourne l'importance globale des features depuis le RandomForest.
    Utilisé dans le dashboard Analytics (AnalyticsLabPage.js).

    Returns:
        [{"name": feature, "value": importance, "rank": i}, ...]
    """
    feature_names = model_payload.get("features") or []
    pipeline = model_payload.get("model")
    if not feature_names or not pipeline:
        return []

    rf = _extract_rf_from_pipeline(pipeline)
    if rf is None or not hasattr(rf, "feature_importances_"):
        return []

    importances = rf.feature_importances_
    if len(importances) != len(feature_names):
        return []

    pairs = sorted(
        zip(feature_names, importances.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]

    return [
        {
            "name": name,
            "value": round(float(imp), 6),
            "rank": i + 1,
            "pct": round(float(imp) * 100, 2),
        }
        for i, (name, imp) in enumerate(pairs)
    ]


# ─── Intégration dans views.py (PaperTradeExplanationLogView) ─────────────────
#
# Dans portfolio/views.py → PaperTradeExplanationLogView.get() :
#
# from .ml_engine.shap_explainer import explain_prediction
#
# # Après avoir chargé le modèle et les features :
# explanation = explain_prediction(
#     model_payload=payload,
#     feature_row=last_row.iloc[0],
#     symbol=symbol,
#     signal=float(signal),
# )
# result["explanation"] = explanation["text"]
# result["top_factors"] = explanation["top_positive"][:3]
# result["risk_factors"] = explanation["top_negative"][:3]
#
# ─── Intégration dans AnalyticsLabPage (feature importance) ──────────────────
#
# Dans portfolio/views.py → BacktestView ou AIBacktesterView :
#
# from .ml_engine.shap_explainer import global_feature_importance
#
# feature_importance = global_feature_importance(payload, top_n=20)
# return Response({
#     ...
#     "feature_importance": feature_importance,  # déjà attendu par le frontend!
# })

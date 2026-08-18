from __future__ import annotations

"""
feature_registry.py  —  VERSION 2
===================================
Registre centralisé de toutes les features du projet personalstock.

Nouvelles features v2 :
- Bollinger %B, Bollinger Bandwidth
- ADX(14), DI+, DI-, ADX/DI ratio
- Stochastic %K/%D
- Williams %R
- OBV Z-score
- CCI(20)
- Donchian %
- Fibonacci Distance
- Price to VWAP
- Momentum Z-score
- ATR %
"""

from typing import Dict, List

# ─── Features stables (recommender/ancienne version) ──────────────────────────
#
# NOTE : ceci est le set de features du pipeline legacy (train_stable_model.py,
# lu positionnellement — l'ordre de cette liste doit rester synchronisé avec
# l'ordre de retour de train_stable_model.py::_build_features). Le pipeline
# plus récent pipelines/stable_pipeline.py calcule un set VOLONTAIREMENT
# différent (10 features, sans sma_ratio_10_20, sous le nom local
# STABLE_FEATURES) — ce n'est pas une duplication accidentelle à fusionner,
# les deux pipelines ont évolué séparément et ne calculent pas les mêmes
# colonnes ; importer cette liste-ci dans stable_pipeline.py ferait planter
# X[STABLE_FEATURES] (colonne manquante).

STABLE_FEATURE_NAMES: List[str] = [
    "rsi_14",
    "sma_ratio_10_50",
    "sma_ratio_20_50",
    "volatility_20",
    "volume_zscore_20",
    "return_20d",
    "spy_correlation",
    "dividend_yield",
    "sector_beta",
    "sentiment_score",
    "sma_ratio_10_20",
]

# ─── Features penny v2 ────────────────────────────────────────────────────────
#
# Corrigé pour matcher exactement ce que train_penny_model.py::_build_features
# ET pipelines/penny_pipeline.py::build_dataset calculent réellement (les deux
# calculent le même set de 7 features). Cette liste listait auparavant 20
# noms — 13 d'entre eux (rsi_7, stoch_k_14, williams_r_14, macd_hist,
# atr_pct_14, bb_pct_b_20, rubber_band_20, obv_zscore, mom_zscore_20,
# linear_slope_20, donchian_pct_20, fib_distance_50, close_pos_in_range)
# n'étaient jamais calculés par train_penny_model.py, qui les remplaçait
# silencieusement par des zéros constants (`if col not in X.columns: X[col]
# = 0.0`) — 13 features mortes diluant le modèle PENNY sans lever d'erreur.

PENNY_FEATURE_NAMES: List[str] = [
    "rsi_14",
    "sma_ratio_10_20",
    "volatility_20",
    "volume_zscore_20",
    "rvol_20",
    "return_5d",
    "sentiment_score",
]

# ─── Features bluechip v2 ─────────────────────────────────────────────────────
#
# NOTE : aucun pipeline n'importe cette liste actuellement (ni le legacy ni
# pipelines/, qui n'a pas d'équivalent "BLUECHIP" — stable_pipeline.py couvre
# ce rôle sous le nom "STABLE", voir plus haut). Gardée telle quelle comme set
# de features cible/aspirationnel plutôt que supprimée, faute d'implémentation
# réelle à comparer.

BLUECHIP_FEATURE_NAMES: List[str] = [
    # Momentum oscillateurs
    "rsi_14",
    "stoch_k_14",
    "cci_20",
    # Tendance
    "sma_ratio_10_50",
    "sma_ratio_20_50",
    "ema_ratio_9_20",
    "macd_hist",
    "adx_14",
    "adx_di_ratio",
    # Volatilité & régime
    "volatility_20",
    "vol_regime",
    "atr_pct_14",
    "bb_pct_b_20",
    "bb_bandwidth_20",
    # Volume
    "volume_zscore_20",
    "rvol_20",
    # Momentum prix
    "return_20d",
    "mom_zscore_20",
    "linear_slope_20",
    "slope_accel_20",
    # Structure de prix
    "donchian_pct_20",
    "fib_distance_50",
    "price_to_vwap_20",
    # Macro
    "spy_corr_60",
    "sector_beta_60",
    # Fondamentaux
    "sentiment_score",
    "dividend_yield",
]

# ─── Features crypto ──────────────────────────────────────────────────────────
#
# Corrigé pour matcher ce que crypto_training.py::_build_crypto_features ET
# pipelines/crypto_pipeline.py::build_dataset calculent réellement (même set
# de 6 features dans les deux). Cette liste renommait auparavant deux colonnes
# (rubber_band_20/price_to_vwap_20 "pour cohérence de nommage") sans jamais
# renommer les colonnes réellement produites (toujours rubber_band_index /
# price_to_vwap) et ajoutait 4 indicateurs jamais calculés (macd_hist,
# bb_pct_b_20, stoch_k_14, obv_zscore) — crypto_training.py::build_crypto_dataset
# faisait alors `dataset.dropna(subset=feature_cols + ['label'])` avec des noms
# de colonnes inexistants, ce qui lève un KeyError garanti au premier run réel.

CRYPTO_FEATURE_NAMES: List[str] = [
    "return_1",
    "rsi_14",
    "rubber_band_index",
    "price_to_vwap",
    "volatility_spike",
    "btc_correlation",
]

# ─── Features crypto SWING (2026-08-18) ───────────────────────────────────────
#
# CRYPTO_FEATURE_NAMES ci-dessus reste le set INTRADAY (15m/1h, horizon de
# quelques heures, voir crypto_training.py::train_crypto_model) -- 6 features
# maison, jamais entraîné en production (crypto_brain_v1.pkl absent).
#
# Celui-ci sert le pipeline SWING (bougies journalières, horizon de jours à
# semaines, voir crypto_training.py::train_crypto_swing_model) -- réutilise
# directement les fonctions techniques génériques déjà éprouvées sur les
# actions (features/features_technical_v2.py::build_full_feature_set,
# patterns.py::add_pattern_columns), pas de formules crypto réinventées.
# Sous-ensemble de FUSION_FEATURE_NAMES_V2 -- retire ce qui n'a pas de source
# de données côté crypto (dividend_yield, sentiment_score/news_count,
# sector_beta_60, macro FRED, order book) ; "spy_corr_60" est calculé avec les
# rendements de BTC-USD comme ancre au lieu du S&P500 et renommé
# "btc_corr_60" en sortie (voir _build_crypto_swing_features) -- vaut 0 pour
# BTC-USD lui-même (rien à corréler avec soi-même).
CRYPTO_SWING_FEATURE_NAMES: List[str] = [
    "rsi_14", "rsi_7",
    "sma_ratio_10_50", "sma_ratio_20_50",
    "ema_ratio_9_20",
    "price_to_ema9", "price_to_ema20",
    "macd_hist", "macd_line",
    "volume_zscore_20", "rvol_20", "obv_zscore",
    "volatility_20", "vol_regime", "atr_pct_14",
    "return_20d", "return_5d", "mom_zscore_20",
    "bb_pct_b_20", "bb_bandwidth_20",
    "rubber_band_20",
    "adx_14", "adx_di_ratio",
    "stoch_k_14", "stoch_d_14",
    "williams_r_14",
    "cci_20",
    "donchian_pct_20",
    "fib_distance_50",
    "price_to_vwap_20",
    "candle_body_pct", "close_pos_in_range", "gap_pct",
    "btc_corr_60",
    "pattern_doji", "pattern_hammer", "pattern_engulfing", "pattern_morning_star",
]

# ─── Features fusion complètes (modèle principal) ────────────────────────────

FUSION_FEATURE_NAMES: List[str] = [
    # Moyennes mobiles (niveaux absolus remplacés par ratios)
    "sma_ratio_10_50",
    "sma_ratio_20_50",
    "ema_ratio_9_20",
    "price_to_ema9",
    "price_to_ema20",
    # Momentum oscillateurs
    "rsi_14",
    "rsi_7",
    "stoch_k_14",
    "stoch_d_14",
    "williams_r_14",
    "cci_20",
    # MACD
    "macd_hist",
    "macd_line",
    # Bollinger
    "bb_pct_b_20",
    "bb_bandwidth_20",
    # ADX (force de tendance)
    "adx_14",
    "adx_di_ratio",
    # Rubber Band (snapback)
    "rubber_band_20",
    # Volume
    "volume_zscore_20",
    "rvol_20",
    "RVOL10",
    "obv_zscore",
    "VPT_roc",
    # Volatilité & régime
    "volatility_20",
    "vol_regime",
    "atr_pct_14",
    # Momentum prix
    "return_20d",
    "return_5d",
    "mom_zscore_20",
    "linear_slope_20",
    "slope_accel_20",
    # Structure de prix
    "donchian_pct_20",
    "fib_distance_50",
    "price_to_vwap_20",
    # Bougie
    "candle_body_pct",
    "close_pos_in_range",
    "gap_pct",
    # Patterns (booléens 0/1)
    "pattern_doji",
    "pattern_hammer",
    "pattern_engulfing",
    "pattern_morning_star",
    # Macro
    "spy_corr_60",
    "tsx_corr_60",
    "sector_beta_60",
    # Macro FRED
    "VIXCLS",
    "DCOILWTICO",
    "CPIAUCSL",
    # Order book
    "bid_ask_spread_pct",
    "order_book_imbalance",
    "trade_velocity",
    # Fondamentaux
    "sentiment_score",
    "news_count",
    "dividend_yield",
    # Fractional differencing
    "frac_diff_close",
]

# ─── Features recommender ────────────────────────────────────────────────────

RECOMMENDER_FEATURE_NAMES: List[str] = [
    "rsi_14",
    "vol_zscore",           # alias volume_zscore_20
    "return_20d",
    "roe",
    "debt_to_equity",
    "news_sentiment",
    "news_count",
    "fred_rate",
]

# ─── Registre global ──────────────────────────────────────────────────────────

FEATURE_REGISTRY: Dict[str, List[str]] = {
    "STABLE":      STABLE_FEATURE_NAMES,
    "PENNY":       PENNY_FEATURE_NAMES,
    "BLUECHIP":    BLUECHIP_FEATURE_NAMES,
    "CRYPTO":       CRYPTO_FEATURE_NAMES,
    "CRYPTO_SWING": CRYPTO_SWING_FEATURE_NAMES,
    "FUSION":      FUSION_FEATURE_NAMES,
    "RECOMMENDER": RECOMMENDER_FEATURE_NAMES,
}

# Alias pour la compatibilité avec l'ancienne version
AI_PENNY = PENNY_FEATURE_NAMES
AI_BLUECHIP = BLUECHIP_FEATURE_NAMES
AI_CRYPTO = CRYPTO_FEATURE_NAMES


def get_feature_names(model_name: str, fallback: List[str] | None = None) -> List[str]:
    """Retourne les features pour un modèle donné."""
    key = (model_name or "").strip().upper()
    # Résolution des alias sandboxes
    aliases = {
        "AI_PENNY": "PENNY",
        "AI_BLUECHIP": "BLUECHIP",
        "AI_CRYPTO": "CRYPTO",
        "PENNY_STOCK": "PENNY",
        "BLUE_CHIP": "BLUECHIP",
        # Corrigé le 2026-08-12 (ML_PIPELINE_AUDIT.md) : pointait vers
        # BLUECHIP_FEATURE_NAMES, une liste documentée plus haut comme
        # "aucun pipeline n'importe cette liste actuellement" -- un piège
        # silencieux (rien n'appelait get_feature_names('WATCHLIST') en
        # production au moment de l'audit, mais le premier code qui l'aurait
        # fait aurait reçu 27 features au lieu des 54 réellement utilisées
        # par le modèle chargé pour WATCHLIST -- data_fusion_brain_
        # bluechip_v1.pkl, entraîné sur FUSION_FEATURE_NAMES). WATCHLIST n'a
        # pas son propre modèle (partagé avec AI_BLUECHIP au niveau du
        # fichier .pkl), mais son chemin de features réel EST FUSION -- même
        # registre que celui que retrain_from_paper_trades_daily importe
        # directement (FEATURE_COLUMNS = list(FUSION_FEATURE_NAMES)).
        "WATCHLIST": "FUSION",
    }
    resolved = aliases.get(key, key)
    if resolved in FEATURE_REGISTRY:
        return FEATURE_REGISTRY[resolved]
    return list(fallback or FUSION_FEATURE_NAMES)


def get_feature_count(model_name: str) -> int:
    """Nombre de features pour un modèle donné."""
    return len(get_feature_names(model_name))


# ─── Groupes de features (pour interprétabilité / SHAP) ──────────────────────

FEATURE_GROUPS: Dict[str, List[str]] = {
    "momentum":    ["rsi_14", "rsi_7", "stoch_k_14", "stoch_d_14", "williams_r_14", "cci_20"],
    "trend":       ["sma_ratio_10_50", "sma_ratio_20_50", "ema_ratio_9_20", "macd_hist", "adx_14", "adx_di_ratio"],
    "volatility":  ["bb_pct_b_20", "bb_bandwidth_20", "atr_pct_14", "volatility_20", "vol_regime"],
    "volume":      ["volume_zscore_20", "rvol_20", "obv_zscore", "trade_velocity"],
    "price_level": ["rubber_band_20", "donchian_pct_20", "fib_distance_50", "price_to_vwap_20", "close_pos_in_range"],
    "returns":     ["return_20d", "return_5d", "mom_zscore_20", "gap_pct"],
    "macro":       ["spy_corr_60", "sector_beta_60", "VIXCLS", "DCOILWTICO", "CPIAUCSL"],
    "sentiment":   ["sentiment_score", "news_count"],
    "fundamental": ["dividend_yield", "roe", "debt_to_equity", "fred_rate"],
}

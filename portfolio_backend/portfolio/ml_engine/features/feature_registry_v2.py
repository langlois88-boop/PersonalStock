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

STABLE_FEATURE_NAMES: List[str] = [
    "rsi_14",
    "sma_ratio_10_50",
    "sma_ratio_20_50",
    "volatility_20",
    "volume_zscore_20",
    "return_20d",
    "spy_corr_60",          # renommé depuis spy_correlation
    "dividend_yield",
    "sector_beta_60",       # renommé depuis sector_beta
    "sentiment_score",
]

# ─── Features penny v2 ────────────────────────────────────────────────────────

PENNY_FEATURE_NAMES: List[str] = [
    # Momentum oscillateurs
    "rsi_14",
    "rsi_7",
    "stoch_k_14",
    "williams_r_14",
    # Tendance
    "sma_ratio_10_20",
    "macd_hist",
    # Volatilité & range
    "volatility_20",
    "atr_pct_14",
    "bb_pct_b_20",
    "rubber_band_20",        # ← clé pour snapback penny
    # Volume
    "volume_zscore_20",
    "rvol_20",
    "obv_zscore",
    # Momentum prix
    "return_5d",
    "mom_zscore_20",
    # Structure de prix
    "donchian_pct_20",
    "fib_distance_50",
    "close_pos_in_range",
    # Contexte
    "sentiment_score",
]

# ─── Features bluechip v2 ─────────────────────────────────────────────────────

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

CRYPTO_FEATURE_NAMES: List[str] = [
    "return_1",
    "rsi_14",
    "rubber_band_20",        # remplace rubber_band_index pour cohérence de nommage
    "price_to_vwap_20",
    "volume_zscore_20",      # remplace volatility_spike pour plus de robustesse
    "btc_correlation",
    "macd_hist",
    "bb_pct_b_20",
    "stoch_k_14",
    "obv_zscore",
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
    "CRYPTO":      CRYPTO_FEATURE_NAMES,
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
        "WATCHLIST": "BLUECHIP",
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

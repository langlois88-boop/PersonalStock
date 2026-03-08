from __future__ import annotations

from typing import Dict, List

STABLE_FEATURE_NAMES: List[str] = [
    'log_ret_20',
    'vol_60',
    'beta',
    'rel_volume_200',
    'dividend_yield',
    'sector_strength',
    'macro_sp500_close',
    'macro_vix_index',
    'macro_rate_10y',
    'macro_inflation',
    'macro_oil_price',
]

PENNY_FEATURE_NAMES: List[str] = [
    'close',
    'sma_10',
    'sma_20',
    'sma_50',
    'volatility_20',
    'volume_change_10',
    'volume_zscore_20',
    'rvol_20',
    'rsi_14',
]

CRYPTO_FEATURE_NAMES: List[str] = [
    'return_1',
    'rsi_14',
    'rubber_band_index',
    'price_to_vwap',
    'volatility_spike',
    'btc_correlation',
]

FUSION_FEATURE_NAMES: List[str] = [
    "MA20",
    "MA50",
    "MA200",
    "EMA9",
    "EMA20",
    "price_to_ema9",
    "price_to_ema20",
    "vol_regime",
    "Volatility",
    "Momentum20",
    "VolumeZ",
    "RVOL10",
    "VPT",
    "VPT_roc",
    "RSI14",
    "MACD_HIST",
    "CandleBodyPct",
    "CandleRangePct",
    "gap_pct",
    "intraday_range_pct",
    "close_pos_in_range",
    "pattern_doji",
    "pattern_hammer",
    "pattern_engulfing",
    "pattern_morning_star",
    "sentiment_score",
    "news_count",
    "VIXCLS",
    "DCOILWTICO",
    "CPIAUCSL",
    "spy_corr_60",
    "tsx_corr_60",
    "bid_ask_spread_pct",
    "order_book_imbalance",
    "trade_velocity",
    "frac_diff_close",
]

RECOMMENDER_FEATURE_NAMES: List[str] = [
    "rsi_14",
    "vol_zscore",
    "return_20d",
    "roe",
    "debt_to_equity",
    "news_sentiment",
    "news_count",
    "fred_rate",
]

FEATURE_REGISTRY: Dict[str, List[str]] = {
    "STABLE": STABLE_FEATURE_NAMES,
    "PENNY": PENNY_FEATURE_NAMES,
    "CRYPTO": CRYPTO_FEATURE_NAMES,
    "FUSION": FUSION_FEATURE_NAMES,
    "RECOMMENDER": RECOMMENDER_FEATURE_NAMES,
}


def get_feature_names(model_name: str, fallback: List[str] | None = None) -> List[str]:
    key = (model_name or '').strip().upper()
    if key in FEATURE_REGISTRY:
        return FEATURE_REGISTRY[key]
    return list(fallback or [])

from __future__ import annotations

"""
features/technical.py  —  VERSION AMÉLIORÉE
=============================================
Indicateurs techniques complets pour personnalstock.

Ajouts par rapport à l'ancienne version :
- Bollinger Bands (position %B, largeur)
- ADX + DI+/DI- (force de tendance)
- Stochastic %K / %D (momentum oscillateur)
- OBV (On Balance Volume) + OBV Z-score
- Williams %R
- Rubber Band Index (déjà présent, amélioré)
- CCI (Commodity Channel Index)
- Fibonacci Retracement Levels (nearest level distance)
- Donchian Channel %
- VWAP ratio (intraday context)

Chaque fonction retourne une pd.Series avec un nom descriptif.
"""

import numpy as np
import pandas as pd


# ─── Existants améliorés ──────────────────────────────────────────────────────

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI via Wilder EMA smoothing."""
    if len(close) < period + 1:
        return pd.Series(np.nan, index=close.index, name=f"rsi_{period}")
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename(f"rsi_{period}")


def sma_ratio(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """Ratio SMA fast / SMA slow."""
    sma_f = close.rolling(fast).mean()
    sma_s = close.rolling(slow).mean().replace(0, np.nan)
    return (sma_f / sma_s).fillna(1.0).rename(f"sma_ratio_{fast}_{slow}")


def ema_ratio(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """Ratio EMA fast / EMA slow."""
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean().replace(0, np.nan)
    return (ema_f / ema_s).fillna(1.0).rename(f"ema_ratio_{fast}_{slow}")


def volume_zscore(volume: pd.Series, period: int = 20) -> pd.Series:
    """Volume Z-score clippé [-3, 3]."""
    mean = volume.rolling(period).mean()
    std = volume.rolling(period).std().replace(0, np.nan)
    return ((volume - mean) / std).clip(-3, 3).fillna(0).rename(f"volume_zscore_{period}")


def rvol(volume: pd.Series, period: int = 20) -> pd.Series:
    """Relative Volume clippé [0, 10]."""
    mean = volume.rolling(period).mean().replace(0, np.nan)
    return (volume / mean).clip(0, 10).fillna(1.0).rename(f"rvol_{period}")


def volatility(close: pd.Series, period: int = 20) -> pd.Series:
    """Volatilité historique (écart-type des rendements)."""
    return close.pct_change().rolling(period).std().rename(f"volatility_{period}")


def rubber_band_index(close: pd.Series, period: int = 20) -> pd.Series:
    """
    Distance normalisée de la SMA par 2*écart-type.
    Valeur < -1.0 = oversold fort (élastique tendu vers le bas).
    Valeur > +1.0 = overbought fort.
    """
    sma = close.rolling(period).mean()
    std = close.rolling(period).std().replace(0, np.nan)
    return ((close - sma) / (2 * std)).fillna(0).rename(f"rubber_band_{period}")


def macd_signal(close: pd.Series) -> pd.Series:
    """MACD line − Signal line (histogramme)."""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return (macd_line - signal_line).rename("macd_hist")


def macd_line(close: pd.Series) -> pd.Series:
    """MACD line seule."""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    return (ema12 - ema26).rename("macd_line")


# ─── Nouveaux indicateurs ─────────────────────────────────────────────────────

def bollinger_pct_b(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """
    Bollinger %B : position du prix dans les bandes.
    0.0 = sur la bande basse, 1.0 = sur la bande haute, 0.5 = milieu.
    < 0 = sous la bande basse (oversold extrême).
    """
    sma = close.rolling(period).mean()
    std = close.rolling(period).std().replace(0, np.nan)
    upper = sma + num_std * std
    lower = sma - num_std * std
    band_width = (upper - lower).replace(0, np.nan)
    pct_b = (close - lower) / band_width
    return pct_b.fillna(0.5).rename(f"bb_pct_b_{period}")


def bollinger_bandwidth(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """
    Bollinger Bandwidth normalisé par la SMA.
    Valeur faible = squeeze (breakout potentiel).
    """
    sma = close.rolling(period).mean().replace(0, np.nan)
    std = close.rolling(period).std()
    bandwidth = (4 * num_std * std) / sma
    return bandwidth.fillna(0).rename(f"bb_bandwidth_{period}")


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    """
    ADX + DI+ + DI- (Average Directional Index).

    ADX > 25 = tendance forte.
    DI+ > DI- = tendance haussière.
    DI- > DI+ = tendance baissière.

    Retourne un DataFrame avec colonnes : adx_14, di_plus_14, di_minus_14.
    """
    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Directional movements
    up_move = high.diff()
    down_move = -low.diff()

    dm_plus = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    dm_minus = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    # Wilder smoothing
    atr_s = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    di_plus_raw = dm_plus.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    di_minus_raw = dm_minus.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    di_plus = (di_plus_raw / atr_s.replace(0, np.nan) * 100).fillna(0)
    di_minus = (di_minus_raw / atr_s.replace(0, np.nan) * 100).fillna(0)

    dx = (abs(di_plus - di_minus) / (di_plus + di_minus).replace(0, np.nan) * 100).fillna(0)
    adx_val = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    return pd.DataFrame({
        f"adx_{period}": adx_val,
        f"di_plus_{period}": di_plus,
        f"di_minus_{period}": di_minus,
    }, index=close.index)


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    """
    Stochastic Oscillator %K et %D.

    %K < 20 = oversold.
    %K > 80 = overbought.
    Croisement %K/%D = signal.
    """
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    k = ((close - lowest_low) / denom * 100).fillna(50)
    d = k.rolling(d_period).mean().fillna(50)
    return pd.DataFrame({
        f"stoch_k_{k_period}": k,
        f"stoch_d_{k_period}": d,
    }, index=close.index)


def williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Williams %R.
    -100 à 0 : -80 à -100 = oversold, -0 à -20 = overbought.
    """
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    wr = ((highest_high - close) / denom * -100).fillna(-50)
    return wr.rename(f"williams_r_{period}")


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On Balance Volume (OBV).
    Accumulation = OBV monte quand le prix monte avec volume.
    """
    direction = np.sign(close.diff().fillna(0))
    obv_series = (volume * direction).cumsum()
    return obv_series.rename("obv")


def obv_zscore(close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """OBV normalisé par rolling Z-score pour la stationnarité."""
    obv_s = obv(close, volume)
    mean = obv_s.rolling(period).mean()
    std = obv_s.rolling(period).std().replace(0, np.nan)
    return ((obv_s - mean) / std).clip(-3, 3).fillna(0).rename(f"obv_zscore_{period}")


def cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Commodity Channel Index.
    > 100 = overbought, < -100 = oversold.
    """
    typical = (high + low + close) / 3.0
    sma_tp = typical.rolling(period).mean()
    mad = typical.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci_val = (typical - sma_tp) / (0.015 * mad.replace(0, np.nan))
    return cci_val.fillna(0).rename(f"cci_{period}")


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_val = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr_val.rename(f"atr_{period}")


def atr_pct(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """ATR normalisé par le prix (pour comparer entre tickers)."""
    atr_v = atr(high, low, close, period)
    return (atr_v / close.replace(0, np.nan)).fillna(0).rename(f"atr_pct_{period}")


def donchian_pct(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Position du prix dans le canal de Donchian.
    0.0 = sur le plus bas sur N jours, 1.0 = sur le plus haut.
    """
    highest = high.rolling(period).max()
    lowest = low.rolling(period).min()
    denom = (highest - lowest).replace(0, np.nan)
    return ((close - lowest) / denom).fillna(0.5).rename(f"donchian_pct_{period}")


def fibonacci_distance(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    lookback: int = 50,
) -> pd.Series:
    """
    Distance au niveau de Fibonacci le plus proche (retracement 38.2%, 50%, 61.8%).
    Normalise par l'ATR pour être dimensionless.

    Valeur proche de 0 = prix sur un niveau de Fibonacci (zone clé).
    """
    fib_levels = [0.236, 0.382, 0.500, 0.618, 0.786]
    results = []
    prices = close.values
    highs = high.values
    lows = low.values
    atr_arr = atr(high, low, close, 14).values

    for i in range(len(prices)):
        if i < lookback:
            results.append(np.nan)
            continue
        window_high = np.nanmax(highs[i - lookback:i + 1])
        window_low = np.nanmin(lows[i - lookback:i + 1])
        swing = window_high - window_low
        if swing < 1e-8:
            results.append(0.0)
            continue

        # Retracements (depuis le haut)
        levels = [window_high - fib * swing for fib in fib_levels]
        price = prices[i]
        atv = atr_arr[i] if not np.isnan(atr_arr[i]) and atr_arr[i] > 0 else swing * 0.02
        min_dist = min(abs(price - level) for level in levels) / atv
        results.append(round(float(min_dist), 4))

    return pd.Series(results, index=close.index, name=f"fib_distance_{lookback}").fillna(1.0)


def price_to_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Ratio prix / VWAP rolling sur N jours.
    > 1.0 = prix au-dessus du VWAP (momentum positif).
    < 1.0 = prix sous le VWAP (sous-évalué à court terme).
    """
    typical = (high + low + close) / 3.0
    pv = typical * volume
    rolling_vwap = pv.rolling(period).sum() / volume.rolling(period).sum().replace(0, np.nan)
    ratio = (close / rolling_vwap).fillna(1.0)
    return ratio.clip(0.5, 2.0).rename(f"price_to_vwap_{period}")


def momentum_zscore(close: pd.Series, period: int = 20, lookback: int = 60) -> pd.Series:
    """
    Z-score du momentum sur N jours, normalisé sur les derniers lookback jours.
    Mesure si le momentum actuel est extrême par rapport à son historique récent.
    """
    mom = close.pct_change(period)
    mean = mom.rolling(lookback).mean()
    std = mom.rolling(lookback).std().replace(0, np.nan)
    return ((mom - mean) / std).clip(-3, 3).fillna(0).rename(f"mom_zscore_{period}")


def gap_pct(open_price: pd.Series, prev_close: pd.Series) -> pd.Series:
    """Gap d'ouverture en % par rapport à la clôture précédente."""
    return ((open_price - prev_close) / prev_close.replace(0, np.nan)).fillna(0).rename("gap_pct")


def candle_body_pct(open_price: pd.Series, close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
    """Taille du corps de bougie normalisée par le range."""
    body = (close - open_price).abs()
    rng = (high - low).replace(0, np.nan)
    return (body / rng).fillna(0).clip(0, 1).rename("candle_body_pct")


def close_position_in_range(close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
    """
    Position de la clôture dans le range haut-bas de la journée.
    1.0 = clôture au plus haut (bullish), 0.0 = clôture au plus bas (bearish).
    """
    rng = (high - low).replace(0, np.nan)
    return ((close - low) / rng).fillna(0.5).clip(0, 1).rename("close_pos_in_range")


def vol_regime(close: pd.Series, short_period: int = 21, long_period: int = 252) -> pd.Series:
    """
    Régime de volatilité : vol courte / vol longue.
    > 1.5 = régime haute volatilité.
    < 0.7 = régime basse volatilité (potentiel squeeze/breakout).
    """
    vol_short = close.pct_change().rolling(short_period, min_periods=10).std()
    vol_long = close.pct_change().rolling(long_period, min_periods=60).std().replace(0, np.nan)
    return (vol_short / vol_long).fillna(1.0).clip(0, 5).rename("vol_regime")


def spy_correlation(returns: pd.Series, spy_returns: pd.Series, period: int = 60) -> pd.Series:
    """Corrélation rolling avec SPY/TSX."""
    return returns.rolling(period).corr(spy_returns).fillna(0).rename(f"spy_corr_{period}")


def sector_beta(
    returns: pd.Series,
    sector_returns: pd.Series,
    period: int = 60,
) -> pd.Series:
    """Beta sectoriel rolling."""
    cov = returns.rolling(period).cov(sector_returns)
    var = sector_returns.rolling(period).var().replace(0, np.nan)
    return (cov / var).fillna(1.0).clip(-3, 3).rename(f"sector_beta_{period}")


# ─── Feature set builder ──────────────────────────────────────────────────────

def build_full_feature_set(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    open_price: pd.Series | None = None,
    spy_returns: pd.Series | None = None,
    sector_returns: pd.Series | None = None,
    sentiment_score: float = 0.0,
    dividend_yield: float = 0.0,
) -> pd.DataFrame:
    """
    Construit le DataFrame complet de features pour un ticker.
    Utilisé pour l'entraînement et l'inférence.

    Couvre tous les indicateurs du feature_registry FUSION + nouveaux.
    """
    features = pd.DataFrame(index=close.index)
    ret = close.pct_change()

    # ── Prix & tendance
    features["rsi_14"]         = rsi(close, 14)
    features["rsi_7"]          = rsi(close, 7)
    features["sma_ratio_10_50"]= sma_ratio(close, 10, 50)
    features["sma_ratio_20_50"]= sma_ratio(close, 20, 50)
    features["sma_ratio_10_20"]= sma_ratio(close, 10, 20)
    features["ema_ratio_9_20"] = ema_ratio(close, 9, 20)

    # ── MA levels (pour DipAnalysis)
    features["MA20"]    = close.rolling(20).mean()
    features["MA50"]    = close.rolling(50).mean()
    features["MA200"]   = close.rolling(200, min_periods=60).mean()
    features["EMA9"]    = close.ewm(span=9, adjust=False).mean()
    features["EMA20"]   = close.ewm(span=20, adjust=False).mean()
    features["price_to_ema9"]  = (close / features["EMA9"].replace(0, np.nan)).fillna(1.0)
    features["price_to_ema20"] = (close / features["EMA20"].replace(0, np.nan)).fillna(1.0)

    # ── MACD
    features["macd_hist"]  = macd_signal(close)
    features["macd_line"]  = macd_line(close)

    # ── Volume
    features["volume_zscore_20"] = volume_zscore(volume, 20)
    features["VolumeZ"]          = features["volume_zscore_20"]   # alias
    features["rvol_20"]          = rvol(volume, 20)
    features["RVOL10"]           = rvol(volume, 10)
    features["obv_zscore"]       = obv_zscore(close, volume, 20)

    # ── Volatilité
    features["volatility_20"] = volatility(close, 20)
    features["Volatility"]     = features["volatility_20"]
    features["vol_regime"]     = vol_regime(close)

    # ── Momentum
    features["return_20d"]  = close.pct_change(20)
    features["return_5d"]   = close.pct_change(5)
    features["Momentum20"]  = features["return_20d"]
    features["mom_zscore_20"] = momentum_zscore(close, 20, 60)

    # ── Bollinger
    features["bb_pct_b_20"]     = bollinger_pct_b(close, 20)
    features["bb_bandwidth_20"] = bollinger_bandwidth(close, 20)

    # ── Rubber Band
    features["rubber_band_20"] = rubber_band_index(close, 20)

    # ── ATR & prix dérivés
    atr_14 = atr(high, low, close, 14)
    features["atr_14"]     = atr_14
    features["atr_pct_14"] = atr_pct(high, low, close, 14)

    # ── ADX (force de tendance)
    adx_df = adx(high, low, close, 14)
    features["adx_14"]      = adx_df["adx_14"]
    features["di_plus_14"]  = adx_df["di_plus_14"]
    features["di_minus_14"] = adx_df["di_minus_14"]
    features["adx_di_ratio"] = (
        (adx_df["di_plus_14"] - adx_df["di_minus_14"])
        / (adx_df["di_plus_14"] + adx_df["di_minus_14"]).replace(0, np.nan)
    ).fillna(0).clip(-1, 1)

    # ── Stochastic
    stoch_df = stochastic(high, low, close, 14, 3)
    features["stoch_k_14"] = stoch_df["stoch_k_14"]
    features["stoch_d_14"] = stoch_df["stoch_d_14"]

    # ── Williams %R
    features["williams_r_14"] = williams_r(high, low, close, 14)

    # ── CCI
    features["cci_20"] = cci(high, low, close, 20)

    # ── Donchian
    features["donchian_pct_20"] = donchian_pct(high, low, close, 20)

    # ── Fibonacci
    features["fib_distance_50"] = fibonacci_distance(close, high, low, 50)

    # ── VWAP
    features["price_to_vwap_20"] = price_to_vwap(high, low, close, volume, 20)

    # ── Bougie
    if open_price is not None:
        features["candle_body_pct"]    = candle_body_pct(open_price, close, high, low)
        features["gap_pct"]            = gap_pct(open_price, close.shift(1))
        features["close_pos_in_range"] = close_position_in_range(close, high, low)
    else:
        features["close_pos_in_range"] = close_position_in_range(close, high, low)
        features["candle_body_pct"]    = 0.5
        features["gap_pct"]            = 0.0

    # ── Contexte macro
    if spy_returns is not None:
        features["spy_corr_60"] = spy_correlation(ret, spy_returns, 60)
    else:
        features["spy_corr_60"] = 0.0

    if sector_returns is not None:
        features["sector_beta_60"] = sector_beta(ret, sector_returns, 60)
    else:
        features["sector_beta_60"] = 1.0

    # ── Fondamentaux statiques
    features["sentiment_score"]  = float(sentiment_score)
    features["dividend_yield"]   = float(dividend_yield)

    return features.replace([np.inf, -np.inf], np.nan)


# ─── Feature registry mis à jour ─────────────────────────────────────────────

FUSION_FEATURE_NAMES_V2 = [
    # Tendance
    "rsi_14", "rsi_7",
    "sma_ratio_10_50", "sma_ratio_20_50",
    "ema_ratio_9_20",
    "price_to_ema9", "price_to_ema20",
    # MACD
    "macd_hist", "macd_line",
    # Volume
    "volume_zscore_20", "rvol_20", "obv_zscore",
    # Volatilité & régime
    "volatility_20", "vol_regime", "atr_pct_14",
    # Momentum
    "return_20d", "return_5d", "mom_zscore_20",
    # Bollinger
    "bb_pct_b_20", "bb_bandwidth_20",
    # Rubber Band
    "rubber_band_20",
    # ADX (force de tendance)
    "adx_14", "adx_di_ratio",
    # Stochastic
    "stoch_k_14", "stoch_d_14",
    # Williams %R
    "williams_r_14",
    # CCI
    "cci_20",
    # Donchian
    "donchian_pct_20",
    # Fibonacci
    "fib_distance_50",
    # VWAP
    "price_to_vwap_20",
    # Bougie
    "candle_body_pct", "close_pos_in_range", "gap_pct",
    # Macro
    "spy_corr_60", "sector_beta_60",
    # Fondamentaux
    "sentiment_score", "dividend_yield",
]

PENNY_FEATURE_NAMES_V2 = [
    "rsi_14", "rsi_7",
    "sma_ratio_10_20",
    "macd_hist",
    "volatility_20", "atr_pct_14",
    "volume_zscore_20", "rvol_20", "obv_zscore",
    "return_5d", "mom_zscore_20",
    "bb_pct_b_20",
    "rubber_band_20",          # ← clé pour le snapback
    "stoch_k_14",
    "williams_r_14",
    "donchian_pct_20",
    "fib_distance_50",
    "close_pos_in_range",
    "sentiment_score",
]

BLUECHIP_FEATURE_NAMES_V2 = [
    "rsi_14",
    "sma_ratio_10_50", "sma_ratio_20_50",
    "ema_ratio_9_20",
    "macd_hist",
    "volatility_20", "vol_regime", "atr_pct_14",
    "volume_zscore_20", "rvol_20",
    "return_20d", "mom_zscore_20",
    "bb_pct_b_20", "bb_bandwidth_20",
    "adx_14", "adx_di_ratio",
    "stoch_k_14",
    "cci_20",
    "donchian_pct_20",
    "fib_distance_50",
    "price_to_vwap_20",
    "spy_corr_60", "sector_beta_60",
    "sentiment_score", "dividend_yield",
]

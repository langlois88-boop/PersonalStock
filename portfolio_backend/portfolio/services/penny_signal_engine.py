from __future__ import annotations

"""
penny_signal_engine.py  — FIX #1 + #12 + #13
==============================================
Remplace generate_penny_signals() dans tasks.py.

Corrections critiques :
- Seuil prix configurable (AI_PENNY_MAX_PRICE au lieu de 0.50 hardcodé)
- Score basé sur rubber_band_index, RSI, ATR, volume, patterns, Altman Z
- Altman Z intégré dans le score final (était calculé mais ignoré)
- Patterns hammer/engulfing utilisés comme filtre boost
- Score combiné pondéré et calibré (était simpliste et buggé)
"""

import os
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Configuration ────────────────────────────────────────────────────────────

def _cfg_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


def _cfg_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


MAX_PRICE          = _cfg_float("AI_PENNY_MAX_PRICE", 5.0)       # était 0.50 hardcodé !
MIN_VOLUME         = _cfg_float("AI_PENNY_MIN_VOLUME", 100_000)
MIN_ALTMAN_Z       = _cfg_float("AI_PENNY_MIN_ALTMAN_Z", 2.0)
SCAN_WORKERS       = _cfg_int("PENNY_SCAN_WORKERS", 8)
DAYS_LOOKBACK      = 30
TRIPLE_UP_PCT      = _cfg_float("PENNY_TRIPLE_BARRIER_UP_PCT", 0.05)   # était 15% irréaliste !
TRIPLE_DOWN_PCT    = _cfg_float("PENNY_TRIPLE_BARRIER_DOWN_PCT", 0.03)
TRIPLE_MAX_DAYS    = _cfg_int("PENNY_TRIPLE_BARRIER_MAX_DAYS", 5)


# ─── Indicateurs techniques purs ──────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> float | None:
    if len(close) < period + 1:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    last = rsi.iloc[-1]
    return None if pd.isna(last) else float(last)


def _rubber_band(close: pd.Series, period: int = 20) -> float | None:
    """Distance normalisée de la SMA20 par 2*std. < -1.0 = élastique tendu → snapback."""
    if len(close) < period:
        return None
    sma = close.rolling(period).mean()
    std = close.rolling(period).std().replace(0, np.nan)
    rbi = ((close - sma) / (2 * std)).iloc[-1]
    return None if pd.isna(rbi) else float(rbi)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float | None:
    if len(close) < period:
        return None
    prev = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev).abs(),
        (low - prev).abs(),
    ], axis=1).max(axis=1)
    atr_val = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean().iloc[-1]
    return None if pd.isna(atr_val) else float(atr_val)


def _volume_zscore(volume: pd.Series, period: int = 20) -> float:
    if len(volume) < period:
        return 0.0
    mean = volume.rolling(period).mean().iloc[-1]
    std = volume.rolling(period).std().iloc[-1]
    if pd.isna(mean) or pd.isna(std) or std == 0:
        return 0.0
    z = (float(volume.iloc[-1]) - float(mean)) / float(std)
    return float(np.clip(z, -3, 3))


def _detect_hammer(open_p: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> bool:
    """Détecte un pattern hammer/bullish engulfing sur les 2 dernières bougies."""
    if len(close) < 2:
        return False
    # Hammer : mèche basse longue, petit corps, mèche haute courte
    body = abs(float(close.iloc[-1]) - float(open_p.iloc[-1]))
    lower_wick = float(min(open_p.iloc[-1], close.iloc[-1])) - float(low.iloc[-1])
    upper_wick = float(high.iloc[-1]) - float(max(open_p.iloc[-1], close.iloc[-1]))
    total_range = float(high.iloc[-1]) - float(low.iloc[-1])
    if total_range < 1e-8:
        return False
    is_hammer = (lower_wick / total_range > 0.5) and (upper_wick / total_range < 0.25)
    # Bullish engulfing
    prev_body = float(close.iloc[-2]) - float(open_p.iloc[-2])
    curr_body = float(close.iloc[-1]) - float(open_p.iloc[-1])
    is_engulfing = prev_body < 0 and curr_body > 0 and abs(curr_body) > abs(prev_body)
    return bool(is_hammer or is_engulfing)


def _altman_z(symbol: str) -> float | None:
    """Récupère l'Altman Z depuis yfinance pour filtrer les penny à risque de faillite."""
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info or {}
        total_assets = info.get("totalAssets")
        total_liab = info.get("totalLiabilities") or info.get("totalLiab")
        current_assets = info.get("currentAssets") or info.get("totalCurrentAssets")
        current_liab = info.get("currentLiabilities") or info.get("totalCurrentLiabilities")
        retained = info.get("retainedEarnings")
        ebit = info.get("ebit") or info.get("operatingIncome")
        sales = info.get("totalRevenue")
        market_cap = info.get("marketCap")
        if not all([total_assets, total_liab, current_assets, current_liab,
                    retained is not None, ebit is not None, sales, market_cap]):
            return None
        working_cap = float(current_assets) - float(current_liab)
        z = (
            1.2 * (working_cap / float(total_assets))
            + 1.4 * (float(retained) / float(total_assets))
            + 3.3 * (float(ebit) / float(total_assets))
            + 0.6 * (float(market_cap) / float(total_liab))
            + 1.0 * (float(sales) / float(total_assets))
        )
        return round(float(z), 3)
    except Exception:
        return None


# ─── Score composite amélioré ─────────────────────────────────────────────────

def _penny_composite_score(
    rsi: float | None,
    rubber_band: float | None,
    volume_z: float,
    sentiment_score: float,
    hype_score: float,
    has_pattern: bool,
    gap_pct: float,
    altman_z: float | None,
    rvol: float,
) -> float:
    """
    Score composite 0.0–1.0 pour un penny stock.

    Pondérations :
    - Rubber Band (snapback) : 30%  ← NOUVEAU clé
    - RSI oversold            : 20%
    - Volume/RVOL             : 15%
    - Sentiment news          : 15%
    - Hype social             : 10%
    - Pattern bougie          :  5%  ← NOUVEAU
    - Gap momentum            :  5%

    Malus Altman Z < seuil : ×0.5
    """
    score = 0.0

    # Rubber Band : plus c'est négatif, mieux c'est (élastique tendu)
    if rubber_band is not None:
        # -2.0 = max oversold → score 1.0 | 0.0 = à la SMA → 0.3 | +1.0 = overbought → 0.0
        rbi_score = max(0.0, min(1.0, (-rubber_band + 0.5) / 2.5))
        score += rbi_score * 0.30

    # RSI : oversold = bullish
    if rsi is not None:
        rsi_score = max(0.0, min(1.0, (50 - rsi) / 30 + 0.3))
        score += rsi_score * 0.20

    # Volume Z + RVOL
    vol_score = max(0.0, min(1.0, (volume_z + 3) / 6))
    rvol_score = max(0.0, min(1.0, rvol / 5.0))
    score += ((vol_score + rvol_score) / 2) * 0.15

    # Sentiment
    sent_score = max(0.0, min(1.0, (sentiment_score + 1) / 2))
    score += sent_score * 0.15

    # Hype social
    score += min(1.0, hype_score) * 0.10

    # Pattern bougie
    if has_pattern:
        score += 1.0 * 0.05

    # Gap momentum (positif mais pas trop — éviter les pump)
    gap_score = max(0.0, min(1.0, gap_pct * 3 + 0.5)) if abs(gap_pct) < 0.20 else 0.0
    score += gap_score * 0.05

    # Malus si Altman Z trop bas (risque de faillite)
    if altman_z is not None and altman_z < MIN_ALTMAN_Z:
        score *= 0.5

    return round(float(np.clip(score, 0.0, 1.0)), 4)


# ─── Analyse d'un symbole ────────────────────────────────────────────────────

def analyze_penny_candidate(
    symbol: str,
    hype_stats: dict[str, float],
    days: int = DAYS_LOOKBACK,
) -> dict[str, Any] | None:
    """
    Analyse complète d'un penny stock candidat.
    Retourne None si le ticker ne passe pas les filtres.
    """
    try:
        import yfinance as yf
        data = yf.Ticker(symbol).history(period=f"{days}d")
    except Exception:
        return None

    if data is None or data.empty:
        return None
    required_cols = {"Close", "Volume", "High", "Low", "Open"}
    if not required_cols.issubset(data.columns):
        return None

    close = data["Close"].dropna()
    volume = data["Volume"].dropna()
    high = data["High"].dropna()
    low = data["Low"].dropna()
    open_p = data["Open"].dropna()

    if len(close) < 10:
        return None

    last_price = float(close.iloc[-1])
    if last_price <= 0:
        return None

    # ── Filtre prix (configurable, pas hardcodé à 0.50$!)
    max_price = _cfg_float("AI_PENNY_MAX_PRICE", MAX_PRICE)
    min_price = _cfg_float("AI_PENNY_MIN_PRICE", 0.01)
    if not (min_price <= last_price <= max_price):
        return None

    # ── Filtre volume moyen
    avg_vol = float(volume.rolling(20).mean().iloc[-1]) if len(volume) >= 20 else float(volume.mean())
    if avg_vol < MIN_VOLUME:
        return None

    last_vol = float(volume.iloc[-1])
    prev_close = float(close.iloc[-2]) if len(close) > 1 else last_price
    gap = (last_price - prev_close) / prev_close if prev_close > 0 else 0.0

    # ── Indicateurs
    rsi_val = _rsi(close, 14)
    rbi_val = _rubber_band(close, 20)
    atr_val = _atr(high, low, close, 14)
    vol_z = _volume_zscore(volume, 20)
    rvol = float(np.clip(last_vol / max(avg_vol, 1), 0, 10))
    has_pattern = _detect_hammer(open_p, high, low, close)

    # ── Altman Z (optionnel, lent → timeout court)
    altman_z = None
    if os.getenv("PENNY_CHECK_ALTMAN_Z", "false").lower() in {"1", "true", "yes"}:
        altman_z = _altman_z(symbol)

    # ── Sentiment depuis les stats hype (Reddit/news)
    mentions = hype_stats.get("mentions", 0)
    raw_sentiment = hype_stats.get("sentiment", 0.0)
    hype = hype_stats.get("hype", 0.0)
    sentiment_score = raw_sentiment / max(mentions, 1) if mentions > 0 else 0.0
    hype_score = min(1.0, hype / 10_000)
    liquidity_score = min(1.0, avg_vol / 1_000_000)

    # ── Score composite
    combined = _penny_composite_score(
        rsi=rsi_val,
        rubber_band=rbi_val,
        volume_z=vol_z,
        sentiment_score=sentiment_score,
        hype_score=hype_score,
        has_pattern=has_pattern,
        gap_pct=gap,
        altman_z=altman_z,
        rvol=rvol,
    )

    # ── Niveaux swing de base
    atr_fallback = last_price * 0.02
    atr_used = atr_val if atr_val else atr_fallback
    stop_loss = round(last_price - 1.5 * atr_used, 4)
    target_price = round(last_price + 3.0 * atr_used, 4)
    risk_reward = round((target_price - last_price) / max(last_price - stop_loss, 1e-6), 2)

    return {
        "symbol": symbol,
        "last_price": round(last_price, 4),
        "avg_volume": round(avg_vol, 0),
        "rvol": round(rvol, 2),
        "rsi": round(rsi_val, 1) if rsi_val is not None else None,
        "rubber_band": round(rbi_val, 3) if rbi_val is not None else None,
        "atr": round(atr_used, 4),
        "volume_z": round(vol_z, 2),
        "gap_pct": round(gap * 100, 2),
        "has_pattern": has_pattern,
        "altman_z": round(altman_z, 2) if altman_z is not None else None,
        "sentiment_score": round(sentiment_score, 3),
        "hype_score": round(hype_score, 3),
        "liquidity_score": round(liquidity_score, 3),
        "combined_score": combined,
        "stop_loss": stop_loss,
        "target_price": target_price,
        "risk_reward": risk_reward,
        "mentions": int(mentions),
    }


# ─── Task principale (drop-in replacement pour generate_penny_signals) ────────

def generate_penny_signals_v2(
    mention_map: dict[str, dict],
    days: int = 7,
    max_symbols: int | None = None,
) -> dict[str, int]:
    """
    Replacement pour generate_penny_signals() dans tasks.py.

    Args:
        mention_map : dict {symbol: {mentions, sentiment, hype}} depuis Reddit/news
        days        : lookback pour l'historique prix
        max_symbols : limite (env PENNY_MAX_SYMBOLS ou 30)

    Usage dans tasks.py :
        from .penny_signal_engine import generate_penny_signals_v2
        ...
        result = generate_penny_signals_v2(mention_map, days=days)
    """
    from django.utils import timezone
    # Import conditionnel pour ne pas casser si Django pas dispo
    try:
        from portfolio.models import PennySignal
        django_available = True
    except ImportError:
        django_available = False

    max_sym = max_symbols or _cfg_int("PENNY_MAX_SYMBOLS", 30)
    today = timezone.now().date() if django_available else None

    # Tri par mentions + hype
    ranked = sorted(
        mention_map.items(),
        key=lambda x: (x[1].get("mentions", 0), x[1].get("hype", 0)),
        reverse=True,
    )[:max_sym]

    created = 0
    seen = 0

    def _process(item: tuple[str, dict]) -> dict[str, Any] | None:
        symbol, stats = item
        return analyze_penny_candidate(symbol, stats, days=days)

    with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as executor:
        for result in executor.map(_process, ranked):
            seen += 1
            if result is None:
                continue
            if not django_available or today is None:
                created += 1
                continue
            _, was_created = PennySignal.objects.update_or_create(
                symbol=result["symbol"],
                as_of=today,
                defaults={
                    "pattern_score": float(result.get("has_pattern", False)),
                    "sentiment_score": result["sentiment_score"],
                    "hype_score": result["hype_score"],
                    "liquidity_score": result["liquidity_score"],
                    "combined_score": result["combined_score"],
                    "last_price": result["last_price"],
                    "avg_volume": result["avg_volume"],
                    "mentions": result["mentions"],
                    "data": {
                        "gap_pct": result["gap_pct"],
                        "volume_z": result["volume_z"],
                        "rsi": result["rsi"],
                        "rubber_band": result["rubber_band"],
                        "atr": result["atr"],
                        "rvol": result["rvol"],
                        "has_pattern": result["has_pattern"],
                        "altman_z": result["altman_z"],
                        "stop_loss": result["stop_loss"],
                        "target_price": result["target_price"],
                        "risk_reward": result["risk_reward"],
                    },
                },
            )
            if was_created:
                created += 1

    return {"created": created, "seen": seen}

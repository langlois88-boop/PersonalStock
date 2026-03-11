from __future__ import annotations

"""
signal_engine_patches.py  — FIX #3, #6, #7, #10
================================================

FIX #3  : DanasEngine URL hardcodée → variable d'env
FIX #6  : Feature linear_slope (tendance qui accélère ou ralentit)
FIX #7  : Confirmation multi-timeframe (5m + 15m + daily)
FIX #10 : Cache ATR/features pour éviter les refusions à chaque call
"""

import os
import logging
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# FIX #3 — DanasEngine : URL hardcodée → utiliser les env vars
# ══════════════════════════════════════════════════════════════════════════════
#
# AVANT (dans services/ai_engine.py) :
#   self.client = openai.OpenAI(
#       base_url="http://100.88.73.110:8001/v1",   ← IP hardcodée en production !
#       api_key="danas-local-key",
#   )
#
# APRÈS — remplacer la classe DanasEngine.__init__ par :

DANAS_URL_PATCH = '''
# Dans portfolio/services/ai_engine.py — remplacer __init__ :

def __init__(self) -> None:
    base_url = (
        os.getenv("DANAS_BASE_URL")
        or os.getenv("OLLAMA_CHAT_BASE_URL")
        or os.getenv("OLLAMA_BASE_URL")
        or "http://localhost:8001"
    ).strip().rstrip("/")
    if "/v1" not in base_url:
        base_url = f"{base_url}/v1"
    
    api_key = os.getenv("DANAS_API_KEY", "danas-local-key")
    
    self.client = openai.OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    self.model_name = os.getenv("DANAS_MODEL", os.getenv("OLLAMA_MODEL", "deepseek-r1"))
    
    # Chargement des modèles avec chemin configurable
    penny_path  = os.getenv("PENNY_MODEL_PATH",  "/app/portfolio/ml_engine/models/data_fusion_brain_penny_v1.pkl")
    bluechip_path = os.getenv("BLUECHIP_MODEL_PATH", "/app/portfolio/ml_engine/models/data_fusion_brain_bluechip_v1.pkl")
    
    self.models = {}
    for name, path in [("penny", penny_path), ("bluechip", bluechip_path)]:
        try:
            self.models[name] = joblib.load(path)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Could not load %s model from %s: %s", name, path, exc)

# Ajouter dans .env.example :
# DANAS_BASE_URL=http://100.88.73.110:8001
# DANAS_API_KEY=danas-local-key
# DANAS_MODEL=deepseek-r1
'''


# ══════════════════════════════════════════════════════════════════════════════
# FIX #6 — Feature : linear_slope (tendance qui accélère ou ralentit)
# ══════════════════════════════════════════════════════════════════════════════

def linear_slope(series: pd.Series, period: int = 20) -> pd.Series:
    """
    Pente de régression linéaire normalisée sur N jours.

    Valeur > 0 : tendance haussière, plus grand = accélère
    Valeur < 0 : tendance baissière
    Normalisé par le prix moyen pour être comparables entre tickers.

    Exemple : slope=0.005 = le prix monte d'env 0.5% par jour
    """
    if len(series) < period:
        return pd.Series(np.nan, index=series.index, name=f"linear_slope_{period}")

    slopes = np.full(len(series), np.nan)
    x = np.arange(period, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    for i in range(period - 1, len(series)):
        y = series.values[i - period + 1:i + 1].astype(float)
        if np.any(np.isnan(y)):
            continue
        y_mean = y.mean()
        slope = ((x - x_mean) * (y - y_mean)).sum() / x_var
        # Normalise par le prix moyen
        slopes[i] = slope / max(abs(y_mean), 1e-8)

    return pd.Series(slopes, index=series.index, name=f"linear_slope_{period}").fillna(0)


def slope_acceleration(series: pd.Series, period: int = 20) -> pd.Series:
    """
    Accélération de la tendance = slope(t) - slope(t-5).
    Positif = tendance qui s'accélère (momentum croissant).
    """
    slope = linear_slope(series, period)
    return (slope - slope.shift(5)).fillna(0).rename(f"slope_accel_{period}")


def price_channel_position(close: pd.Series, high: pd.Series, low: pd.Series, period: int = 20) -> pd.Series:
    """
    Position du prix dans son canal de tendance (0=bas, 1=haut).
    Combine Donchian avec la tendance linéaire.
    """
    highest = high.rolling(period).max()
    lowest = low.rolling(period).min()
    channel = (highest - lowest).replace(0, np.nan)
    return ((close - lowest) / channel).clip(0, 1).fillna(0.5).rename(f"channel_pos_{period}")


# ══════════════════════════════════════════════════════════════════════════════
# FIX #7 — Confirmation multi-timeframe
# ══════════════════════════════════════════════════════════════════════════════

def compute_multitimeframe_signal(
    symbol: str,
    daily_ml_score: float,
    ctx_5m: dict[str, Any] | None = None,
    ctx_15m: dict[str, Any] | None = None,
    min_timeframes_aligned: int = 2,
) -> dict[str, Any]:
    """
    Combine les signaux de 3 timeframes pour un verdict consolidé.

    Règle : un signal est "confirmé" si au moins min_timeframes_aligned
    timeframes pointent dans la même direction.

    Args:
        symbol                : ticker
        daily_ml_score        : score du modèle daily (0.0–1.0)
        ctx_5m                : contexte intraday 5 min (depuis _intraday_context_for_timeframe)
        ctx_15m               : contexte intraday 15 min
        min_timeframes_aligned: nb de timeframes qui doivent confirmer

    Returns:
        {
            "confirmed": bool,
            "alignment_count": int,
            "daily_signal": str,    # "BULL" | "BEAR" | "NEUTRAL"
            "tf_5m_signal": str,
            "tf_15m_signal": str,
            "composite_score": float,
            "reason": str,
        }
    """
    buy_threshold = float(os.getenv("PAPER_BUY_THRESHOLD", "0.82"))
    sell_threshold = float(os.getenv("PAPER_SELL_THRESHOLD", "0.40"))

    def _classify(score: float) -> str:
        if score >= buy_threshold * 0.85:
            return "BULL"
        if score <= sell_threshold * 1.15:
            return "BEAR"
        return "NEUTRAL"

    def _intraday_score(ctx: dict[str, Any] | None) -> float:
        """Convertit un contexte intraday en score 0.0–1.0."""
        if not ctx:
            return 0.5
        score = 0.5

        # RSI
        rsi = float(ctx.get("rsi14") or ctx.get("rsi") or 50)
        rsi_score = 1.0 - min(max((rsi - 30) / 40, 0), 1)
        score += (rsi_score - 0.5) * 0.30

        # RVOL (volume inhabituel = confirme le mouvement)
        rvol = float(ctx.get("rvol") or 1.0)
        rvol_boost = min(max((rvol - 1.0) / 3.0, 0), 0.15)
        score += rvol_boost

        # Pattern signal
        pattern = float(ctx.get("pattern_signal") or 0)
        score += pattern * 0.10

        # Price vs VWAP
        p2v = float(ctx.get("price_to_vwap") or 1.0)
        vwap_signal = (p2v - 1.0) * 0.15
        score += vwap_signal

        return float(np.clip(score, 0, 1))

    # Calcul des scores
    daily_score = daily_ml_score
    score_5m = _intraday_score(ctx_5m)
    score_15m = _intraday_score(ctx_15m)

    daily_signal = _classify(daily_score)
    tf_5m_signal = _classify(score_5m) if ctx_5m else "N/A"
    tf_15m_signal = _classify(score_15m) if ctx_15m else "N/A"

    signals = [daily_signal]
    if ctx_5m:
        signals.append(tf_5m_signal)
    if ctx_15m:
        signals.append(tf_15m_signal)

    bull_count = signals.count("BULL")
    bear_count = signals.count("BEAR")
    alignment_count = max(bull_count, bear_count)
    dominant = "BULL" if bull_count >= bear_count else "BEAR"
    confirmed = alignment_count >= min(min_timeframes_aligned, len(signals))

    # Score composite pondéré : daily 50%, 15m 30%, 5m 20%
    weights = []
    scores = []
    weights.append(0.50); scores.append(daily_score)
    if ctx_15m:
        weights.append(0.30); scores.append(score_15m)
    if ctx_5m:
        weights.append(0.20); scores.append(score_5m)
    total_w = sum(weights)
    composite = sum(w * s for w, s in zip(weights, scores)) / total_w if total_w > 0 else daily_score

    # Raison textuelle
    tf_summary = f"Daily {daily_signal}"
    if ctx_15m:
        tf_summary += f" | 15m {tf_15m_signal}"
    if ctx_5m:
        tf_summary += f" | 5m {tf_5m_signal}"

    if confirmed and dominant == "BULL":
        reason = f"✅ Signal BULL confirmé {alignment_count}/{len(signals)} timeframes ({tf_summary})"
    elif confirmed and dominant == "BEAR":
        reason = f"🔴 Signal BEAR confirmé {alignment_count}/{len(signals)} timeframes ({tf_summary})"
    else:
        reason = f"⚠️ Signaux divergents ({tf_summary}) — attendre confirmation"

    return {
        "symbol": symbol,
        "confirmed": confirmed,
        "dominant_signal": dominant,
        "alignment_count": alignment_count,
        "total_timeframes": len(signals),
        "daily_signal": daily_signal,
        "daily_score": round(daily_score, 4),
        "tf_15m_signal": tf_15m_signal,
        "tf_15m_score": round(score_15m, 4) if ctx_15m else None,
        "tf_5m_signal": tf_5m_signal,
        "tf_5m_score": round(score_5m, 4) if ctx_5m else None,
        "composite_score": round(composite, 4),
        "reason": reason,
    }


def should_trade_with_mtf(
    symbol: str,
    daily_score: float,
    ctx_5m: dict | None,
    ctx_15m: dict | None,
    universe: str = "BLUECHIP",
) -> tuple[bool, dict[str, Any]]:
    """
    Décision finale de trading basée sur la confirmation multi-timeframe.

    Logique :
    - Bluechip : daily + 15m suffisent (marché moins volatile)
    - Penny    : daily + 15m + 5m requis (momentum rapide nécessaire)

    Returns:
        (should_trade: bool, mtf_context: dict)
    """
    min_aligned = 3 if universe in {"PENNY", "AI_PENNY"} else 2
    mtf = compute_multitimeframe_signal(
        symbol=symbol,
        daily_ml_score=daily_score,
        ctx_5m=ctx_5m,
        ctx_15m=ctx_15m,
        min_timeframes_aligned=min_aligned,
    )
    # Override env
    mtf_enabled = os.getenv("MTF_CONFIRMATION_ENABLED", "true").lower() in {"1", "true", "yes", "y"}
    if not mtf_enabled:
        return True, mtf

    return mtf["confirmed"] and mtf["dominant_signal"] == "BULL", mtf


# ══════════════════════════════════════════════════════════════════════════════
# FIX #10 — Cache ATR/features pour éviter les refusions répétées
# ══════════════════════════════════════════════════════════════════════════════

# Cache Django (si dispo) ou dict en mémoire
_MEMORY_CACHE: dict[str, tuple[float, Any]] = {}


def _get_cache_backend():
    try:
        from django.core.cache import cache
        return cache
    except Exception:
        return None


def cache_get(key: str) -> Any | None:
    cache_backend = _get_cache_backend()
    if cache_backend:
        return cache_backend.get(key)
    # Mémoire avec TTL simple
    entry = _MEMORY_CACHE.get(key)
    if entry:
        import time
        expiry, value = entry
        if time.time() < expiry:
            return value
        del _MEMORY_CACHE[key]
    return None


def cache_set(key: str, value: Any, timeout: int = 300) -> None:
    cache_backend = _get_cache_backend()
    if cache_backend:
        cache_backend.set(key, value, timeout)
    else:
        import time
        _MEMORY_CACHE[key] = (time.time() + timeout, value)


def get_cached_atr(
    symbol: str,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    period: int = 14,
    ttl: int = 300,  # 5 min
) -> float:
    """
    Retourne l'ATR depuis le cache (5 min) ou le calcule.
    Évite le recalcul à chaque appel de dashboard / views.

    Usage dans views.py :
        from .signal_engine_patches import get_cached_atr
        atr = get_cached_atr(symbol, close, high, low)
    """
    key = f"atr:{symbol}:{period}"
    cached = cache_get(key)
    if cached is not None:
        return float(cached)

    prev = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev).abs(),
        (low - prev).abs(),
    ], axis=1).max(axis=1)
    atr_series = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    atr_val = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else float(close.iloc[-1]) * 0.02

    cache_set(key, atr_val, timeout=ttl)
    return atr_val


def get_cached_fusion_frame(
    symbol: str,
    ttl: int = 300,
) -> Any | None:
    """
    Cache du DataFusionEngine.fuse_all() result.
    Évite de refusionner les données à chaque call de dashboard.

    Usage dans views.py (remplace les appels directs) :
        from .signal_engine_patches import get_cached_fusion_frame, set_cached_fusion_frame
        frame = get_cached_fusion_frame(symbol)
        if frame is None:
            engine = DataFusionEngine(symbol)
            frame = engine.fuse_all()
            set_cached_fusion_frame(symbol, frame)
    """
    key = f"fusion_frame:{symbol}"
    return cache_get(key)


def set_cached_fusion_frame(symbol: str, frame: Any, ttl: int = 300) -> None:
    key = f"fusion_frame:{symbol}"
    cache_set(key, frame, timeout=ttl)


def get_cached_ai_score(symbol: str, universe: str) -> float | None:
    key = f"ai_score:{symbol}:{universe}"
    return cache_get(key)


def set_cached_ai_score(symbol: str, universe: str, score: float, ttl: int = 300) -> None:
    key = f"ai_score:{symbol}:{universe}"
    cache_set(key, score, timeout=ttl)


# ── Décorateur pour cacher les méthodes de views lentes ──────────────────────

def cached_view_result(ttl: int = 300, key_prefix: str = "view"):
    """
    Décorateur pour cacher le résultat d'une méthode de view.

    Usage :
        @cached_view_result(ttl=300, key_prefix="atr_value")
        def _atr_value(self, symbol: str, window: int = 14):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Construire une clé depuis les arguments
            key_parts = [key_prefix] + [str(a) for a in args[1:]] + [f"{k}={v}" for k, v in kwargs.items()]
            cache_key = ":".join(key_parts)[:200]
            cached = cache_get(cache_key)
            if cached is not None:
                return cached
            result = func(*args, **kwargs)
            if result is not None:
                cache_set(cache_key, result, timeout=ttl)
            return result
        return wrapper
    return decorator

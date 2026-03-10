from __future__ import annotations

"""
pipeline_fixes.py  — FIX #4 + #9
===================================

FIX #4 : Labels penny — seuils triple-barrier irréalistes
    AVANT : up_pct=0.15, down_pct=0.10, max_days=10
    → 15% en 10 jours pour un penny stock : quasi jamais atteint
    → Résultat : ~5% de labels positifs → modèle biaisé "SELL"

FIX #9 : _check_bluechip_health non réutilisé dans bluechip_dip_scanner
    Le scanner dip calcule le score et le RSI mais skip la vérification
    de santé fondamentale (PE ratio, cash flow, debt ratio) → faux positifs
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# FIX #4 — Labels réalistes pour le penny pipeline
# ══════════════════════════════════════════════════════════════════════════════

def _cfg(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


def triple_barrier_labels_adaptive(
    close: pd.Series,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    universe: str = "PENNY",
    up_pct: float | None = None,
    down_pct: float | None = None,
    max_days: int | None = None,
    use_atr_barriers: bool = True,
    atr_period: int = 14,
) -> pd.Series:
    """
    Labels triple-barrier adaptatifs selon l'univers.

    PENNY (réaliste) :
        up_pct=0.05, down_pct=0.03, max_days=5   ← ATR-based ou configurable
        AVANT : up_pct=0.15, down_pct=0.10, max_days=10 → labels trop rares

    BLUECHIP :
        up_pct=0.04, down_pct=0.025, max_days=15

    CRYPTO :
        up_pct=0.08, down_pct=0.05, max_days=7

    Si use_atr_barriers=True, les barrières sont calculées depuis l'ATR
    pour s'adapter à la volatilité réelle du ticker.

    Returns:
        pd.Series avec labels 1 (UP), -1 (DOWN), 0 (NEUTRAL/TIMEOUT)
        NaN pour les périodes sans données suffisantes
    """
    # Seuils par univers depuis les env vars ou valeurs par défaut réalistes
    DEFAULTS = {
        "PENNY": {
            "up_pct": _cfg("PENNY_TRIPLE_BARRIER_UP_PCT", 0.05),      # était 0.15 !
            "down_pct": _cfg("PENNY_TRIPLE_BARRIER_DOWN_PCT", 0.03),   # était 0.10 !
            "max_days": int(_cfg("PENNY_TRIPLE_BARRIER_MAX_DAYS", 5)), # était 10 !
            "atr_mult_up": _cfg("PENNY_BARRIER_ATR_MULT_UP", 2.0),
            "atr_mult_down": _cfg("PENNY_BARRIER_ATR_MULT_DOWN", 1.5),
        },
        "BLUECHIP": {
            "up_pct": _cfg("BLUECHIP_TRIPLE_BARRIER_UP_PCT", 0.04),
            "down_pct": _cfg("BLUECHIP_TRIPLE_BARRIER_DOWN_PCT", 0.025),
            "max_days": int(_cfg("BLUECHIP_TRIPLE_BARRIER_MAX_DAYS", 15)),
            "atr_mult_up": _cfg("BLUECHIP_BARRIER_ATR_MULT_UP", 2.5),
            "atr_mult_down": _cfg("BLUECHIP_BARRIER_ATR_MULT_DOWN", 1.5),
        },
        "CRYPTO": {
            "up_pct": _cfg("CRYPTO_TRIPLE_BARRIER_UP_PCT", 0.08),
            "down_pct": _cfg("CRYPTO_TRIPLE_BARRIER_DOWN_PCT", 0.05),
            "max_days": int(_cfg("CRYPTO_TRIPLE_BARRIER_MAX_DAYS", 7)),
            "atr_mult_up": _cfg("CRYPTO_BARRIER_ATR_MULT_UP", 2.0),
            "atr_mult_down": _cfg("CRYPTO_BARRIER_ATR_MULT_DOWN", 1.5),
        },
    }

    cfg = DEFAULTS.get(universe.upper(), DEFAULTS["PENNY"])
    final_up = up_pct if up_pct is not None else cfg["up_pct"]
    final_down = down_pct if down_pct is not None else cfg["down_pct"]
    final_max_days = max_days if max_days is not None else cfg["max_days"]

    close_vals = close.values.astype(float)
    n = len(close_vals)
    labels = np.full(n, np.nan)

    # Calcul ATR si demandé
    atr_vals = None
    if use_atr_barriers and high is not None and low is not None:
        try:
            prev = close.shift(1)
            tr = pd.concat([
                (high - low).abs(),
                (high - prev).abs(),
                (low - prev).abs(),
            ], axis=1).max(axis=1)
            atr_series = tr.ewm(alpha=1 / atr_period, min_periods=atr_period, adjust=False).mean()
            atr_vals = atr_series.values.astype(float)
        except Exception:
            atr_vals = None

    for i in range(n - final_max_days):
        entry = close_vals[i]
        if np.isnan(entry) or entry <= 0:
            continue

        if atr_vals is not None and not np.isnan(atr_vals[i]):
            atr = atr_vals[i]
            up_barrier = entry + cfg["atr_mult_up"] * atr
            down_barrier = entry - cfg["atr_mult_down"] * atr
        else:
            up_barrier = entry * (1 + final_up)
            down_barrier = entry * (1 - final_down)

        label = 0  # timeout par défaut
        for j in range(i + 1, min(i + final_max_days + 1, n)):
            future_price = close_vals[j]
            if np.isnan(future_price):
                continue
            if future_price >= up_barrier:
                label = 1
                break
            if future_price <= down_barrier:
                label = -1
                break

        labels[i] = label

    result = pd.Series(labels, index=close.index, name="label")

    # Log de la distribution pour débogage
    valid = result.dropna()
    if len(valid) > 10:
        pos_pct = (valid == 1).sum() / len(valid) * 100
        neg_pct = (valid == -1).sum() / len(valid) * 100
        neu_pct = (valid == 0).sum() / len(valid) * 100
        logger.debug(
            "Labels %s: pos=%.1f%% neg=%.1f%% neutral=%.1f%% (n=%d)",
            universe, pos_pct, neg_pct, neu_pct, len(valid)
        )
        if pos_pct < 10:
            logger.warning(
                "⚠️ Labels très déséquilibrés pour %s: seulement %.1f%% positifs. "
                "Considérer d'ajuster PENNY_TRIPLE_BARRIER_UP_PCT ou d'utiliser "
                "use_atr_barriers=True.", universe, pos_pct
            )

    return result


def validate_label_distribution(labels: pd.Series, universe: str = "PENNY") -> dict[str, Any]:
    """
    Valide que la distribution des labels est raisonnable pour l'entraînement.
    Utile avant de lancer un retrain.

    Returns:
        {ok: bool, positive_pct, negative_pct, neutral_pct, warning: str|None}
    """
    valid = labels.dropna()
    if len(valid) < 50:
        return {"ok": False, "warning": "Pas assez de labels (< 50)."}

    pos = int((valid == 1).sum())
    neg = int((valid == -1).sum())
    neu = int((valid == 0).sum())
    total = len(valid)

    pos_pct = pos / total * 100
    neg_pct = neg / total * 100

    warning = None
    ok = True

    if pos_pct < 5:
        warning = f"Labels très déséquilibrés: seulement {pos_pct:.1f}% positifs. Revoir les barrières."
        ok = False
    elif pos_pct < 15:
        warning = f"Labels déséquilibrés: {pos_pct:.1f}% positifs. class_weight='balanced' requis."
    elif pos_pct > 60:
        warning = f"Trop de labels positifs: {pos_pct:.1f}%. Barrières trop larges ?"

    return {
        "ok": ok,
        "n_total": total,
        "n_positive": pos,
        "n_negative": neg,
        "n_neutral": neu,
        "positive_pct": round(pos_pct, 1),
        "negative_pct": round(neg_pct, 1),
        "neutral_pct": round(100 - pos_pct - neg_pct, 1),
        "warning": warning,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FIX #9 — _check_bluechip_health dans bluechip_dip_scanner
# ══════════════════════════════════════════════════════════════════════════════

def check_dip_health(
    symbol: str,
    price: float,
    score: float,
    min_score: float,
    use_health_check: bool | None = None,
) -> tuple[bool, dict[str, Any]]:
    """
    Vérifie la santé fondamentale d'un candidat dip.

    À appeler dans bluechip_dip_scanner() après _mean_reversion_score() :

        score, rsi, lower = _mean_reversion_score(symbol)
        score = _canadian_boost(score, symbol)
        if score < min_score:
            continue
        
        # FIX #9 : ajouter ce bloc
        passes_health, health_meta = check_dip_health(symbol, price, score, min_score)
        if not passes_health:
            continue

    Args:
        symbol        : ticker
        price         : prix actuel
        score         : score mean-reversion
        min_score     : seuil minimum
        use_health_check : None = utiliser env var BLUECHIP_DIP_HEALTH_CHECK

    Returns:
        (passes: bool, metadata: dict)
    """
    enabled = use_health_check
    if enabled is None:
        enabled = os.getenv("BLUECHIP_DIP_HEALTH_CHECK", "true").lower() in {"1", "true", "yes", "y"}

    if not enabled:
        return True, {}

    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info or {}
    except Exception:
        return True, {}  # Fail open — ne pas bloquer si yfinance indisponible

    health_meta = {}

    # ── 1. Ratio dette/capitaux propres
    total_debt = info.get("totalDebt") or 0
    total_equity = info.get("totalStockholderEquity") or info.get("stockholdersEquity") or 0
    if total_equity and total_equity > 0:
        debt_ratio = float(total_debt) / float(total_equity)
        health_meta["debt_equity_ratio"] = round(debt_ratio, 2)
        max_debt = float(os.getenv("BLUECHIP_MAX_DEBT_EQUITY", "3.0"))
        if debt_ratio > max_debt:
            return False, {**health_meta, "reject_reason": f"Debt/Equity {debt_ratio:.1f} > {max_debt}"}

    # ── 2. Free Cash Flow positif
    fcf = info.get("freeCashflow")
    if fcf is not None:
        fcf = float(fcf)
        health_meta["free_cashflow"] = fcf
        if fcf < 0:
            if os.getenv("BLUECHIP_REQUIRE_POSITIVE_FCF", "false").lower() in {"1", "true", "yes"}:
                return False, {**health_meta, "reject_reason": f"FCF négatif: {fcf:,.0f}"}

    # ── 3. Current Ratio (liquidité court terme)
    current_ratio = info.get("currentRatio")
    if current_ratio is not None:
        current_ratio = float(current_ratio)
        health_meta["current_ratio"] = round(current_ratio, 2)
        min_cr = float(os.getenv("BLUECHIP_MIN_CURRENT_RATIO", "0.8"))
        if current_ratio < min_cr:
            return False, {**health_meta, "reject_reason": f"Current ratio {current_ratio:.2f} < {min_cr}"}

    # ── 4. Revenue growth positif sur 12 mois
    revenue_growth = info.get("revenueGrowth")
    if revenue_growth is not None:
        revenue_growth = float(revenue_growth)
        health_meta["revenue_growth"] = round(revenue_growth * 100, 1)
        min_rev_growth = float(os.getenv("BLUECHIP_MIN_REVENUE_GROWTH", "-0.20"))
        if revenue_growth < min_rev_growth:
            return False, {**health_meta, "reject_reason": f"Revenue growth {revenue_growth*100:.1f}% < {min_rev_growth*100:.0f}%"}

    # ── 5. Market cap minimum (éviter les micro-caps qui se font passer pour du bluechip)
    market_cap = info.get("marketCap")
    if market_cap is not None:
        health_meta["market_cap"] = market_cap
        min_cap = float(os.getenv("BLUECHIP_MIN_MARKET_CAP", "1e9"))
        if float(market_cap) < min_cap:
            return False, {**health_meta, "reject_reason": f"Market cap {market_cap:,.0f} < {min_cap:,.0f}"}

    # ── 6. Altman Z (si configuré)
    if os.getenv("BLUECHIP_CHECK_ALTMAN_Z", "false").lower() in {"1", "true", "yes"}:
        try:
            total_assets = info.get("totalAssets")
            total_liab = info.get("totalLiabilities") or info.get("totalLiab")
            current_assets = info.get("totalCurrentAssets")
            current_liab = info.get("totalCurrentLiabilities")
            retained = info.get("retainedEarnings")
            ebit = info.get("ebit") or info.get("operatingIncome")
            sales = info.get("totalRevenue")
            cap = info.get("marketCap")
            if all(v is not None for v in [total_assets, total_liab, current_assets, current_liab, retained, ebit, sales, cap]):
                working_cap = float(current_assets) - float(current_liab)
                z = (
                    1.2 * (working_cap / float(total_assets))
                    + 1.4 * (float(retained) / float(total_assets))
                    + 3.3 * (float(ebit) / float(total_assets))
                    + 0.6 * (float(cap) / float(total_liab))
                    + 1.0 * (float(sales) / float(total_assets))
                )
                health_meta["altman_z"] = round(float(z), 2)
                min_z = float(os.getenv("BLUECHIP_MIN_ALTMAN_Z", "1.5"))
                if z < min_z:
                    return False, {**health_meta, "reject_reason": f"Altman Z {z:.2f} < {min_z}"}
        except Exception:
            pass

    return True, health_meta


def enrich_dip_candidate(
    candidate: dict[str, Any],
    health_meta: dict[str, Any],
) -> dict[str, Any]:
    """
    Ajoute les métriques de santé fondamentale au candidat dip.
    À utiliser juste avant d'append() dans la liste candidates.
    """
    enriched = dict(candidate)
    enriched.update({
        "debt_equity_ratio": health_meta.get("debt_equity_ratio"),
        "current_ratio": health_meta.get("current_ratio"),
        "revenue_growth": health_meta.get("revenue_growth"),
        "free_cashflow": health_meta.get("free_cashflow"),
        "market_cap": health_meta.get("market_cap"),
        "altman_z": health_meta.get("altman_z"),
    })
    return enriched


# ─── Patch complet pour bluechip_dip_scanner dans tasks.py ────────────────────
#
# Remplacer (après la ligne score = _canadian_boost(score, symbol)):
#
#   if score < min_score:
#       continue
#   candidates.append({
#       'symbol': symbol,
#       'price': round(price, 2),
#       'score': round(score, 4),
#       'rsi': ...,
#       'lower_band': ...,
#   })
#
# Par :
#
#   if score < min_score:
#       continue
#
#   from .pipeline_fixes import check_dip_health, enrich_dip_candidate
#   passes, health_meta = check_dip_health(symbol, price, score, min_score)
#   if not passes:
#       continue
#
#   candidate = {
#       'symbol': symbol,
#       'price': round(price, 2),
#       'score': round(score, 4),
#       'rsi': None if rsi is None else round(float(rsi), 2),
#       'lower_band': None if lower is None else round(float(lower), 2),
#   }
#   candidates.append(enrich_dip_candidate(candidate, health_meta))

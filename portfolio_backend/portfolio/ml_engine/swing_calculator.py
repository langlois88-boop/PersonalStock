from __future__ import annotations

"""
swing_calculator.py
===================
Calcul précis des niveaux d'entrée, cible et stop-loss pour le swing trading.

Utilisé par :
- Penny stock snapback (rubber band tendu → entrée, cible, stop)
- Bluechip dip injustifié (zone d'achat, cible, stop)
- Portfolio paper trading signals

Formules :
- Stop-loss : ATR × multiplicateur en dessous du support
- Cible    : Niveau de résistance suivant ou ATR × 3 (R:R ≥ 2.0)
- Entrée   : Idéalement au support ou légèrement au-dessus
- Position size : % risk per trade / (entrée - stop)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


# ─── Configuration (overridable par env) ─────────────────────────────────────

def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


ATR_STOP_MULT       = _env_float("SWING_ATR_STOP_MULT", 1.5)
ATR_TARGET_MULT     = _env_float("SWING_ATR_TARGET_MULT", 3.0)
MIN_RISK_REWARD     = _env_float("SWING_MIN_RR", 2.0)
RISK_PCT_PER_TRADE  = _env_float("PAPER_RISK_PCT", 0.015)   # 1.5% du capital par défaut
SWING_MAX_LOSS_PCT  = _env_float("SWING_MAX_LOSS_PCT", 0.08)  # stop max 8% sous l'entrée


# ─── Structures de données ────────────────────────────────────────────────────

@dataclass
class TradeLevels:
    """Niveaux complets d'un trade swing."""
    symbol: str
    entry: float
    stop_loss: float
    target_1: float        # Premier objectif
    target_2: float        # Second objectif (si R:R permet)
    risk_reward_1: float
    risk_reward_2: float
    risk_per_share: float
    atr: float
    universe: str          # "PENNY" | "BLUECHIP" | "ETF" | "CRYPTO"
    trade_type: str        # "SWING" | "POSITION" | "DIP_BUY"
    horizon_days: int = 5

    # Position sizing
    position_size_shares: int = 0
    position_size_pct: float = 0.0

    # Validation
    is_valid: bool = True
    invalid_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "entry": round(self.entry, 4),
            "stop_loss": round(self.stop_loss, 4),
            "target_1": round(self.target_1, 4),
            "target_2": round(self.target_2, 4),
            "risk_reward_1": round(self.risk_reward_1, 2),
            "risk_reward_2": round(self.risk_reward_2, 2),
            "risk_per_share": round(self.risk_per_share, 4),
            "atr": round(self.atr, 4),
            "universe": self.universe,
            "trade_type": self.trade_type,
            "horizon_days": self.horizon_days,
            "position_size_shares": self.position_size_shares,
            "position_size_pct": round(self.position_size_pct, 4),
            "is_valid": self.is_valid,
            "invalid_reason": self.invalid_reason,
        }

    @property
    def stop_distance_pct(self) -> float:
        if self.entry == 0:
            return 0.0
        return abs(self.entry - self.stop_loss) / self.entry

    @property
    def target_distance_pct(self) -> float:
        if self.entry == 0:
            return 0.0
        return abs(self.target_1 - self.entry) / self.entry


# ─── Détection supports / résistances ────────────────────────────────────────

def find_support_resistance(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    lookback: int = 50,
    min_touches: int = 2,
    tolerance_pct: float = 0.015,
) -> tuple[list[float], list[float]]:
    """
    Détecte les niveaux de support et résistance par clustering de pivots.

    Algorithme :
    1. Trouve les pivots locaux (hauts/bas sur 3 bougies)
    2. Groupe les niveaux proches (dans tolerance_pct)
    3. Retourne les niveaux les plus touchés

    Returns:
        (supports, resistances) — listes triées
    """
    if len(close) < lookback:
        return [], []

    recent_close = close.iloc[-lookback:]
    recent_high = high.iloc[-lookback:]
    recent_low = low.iloc[-lookback:]

    pivot_highs: list[float] = []
    pivot_lows: list[float] = []

    for i in range(1, len(recent_high) - 1):
        if recent_high.iloc[i] > recent_high.iloc[i-1] and recent_high.iloc[i] > recent_high.iloc[i+1]:
            pivot_highs.append(float(recent_high.iloc[i]))
        if recent_low.iloc[i] < recent_low.iloc[i-1] and recent_low.iloc[i] < recent_low.iloc[i+1]:
            pivot_lows.append(float(recent_low.iloc[i]))

    current_price = float(recent_close.iloc[-1])

    def _cluster(levels: list[float]) -> list[float]:
        if not levels:
            return []
        sorted_levels = sorted(levels)
        clusters: list[list[float]] = []
        current_cluster = [sorted_levels[0]]
        for level in sorted_levels[1:]:
            ref = current_cluster[0]
            if abs(level - ref) / max(abs(ref), 1e-8) <= tolerance_pct:
                current_cluster.append(level)
            else:
                clusters.append(current_cluster)
                current_cluster = [level]
        clusters.append(current_cluster)
        # Retourne le centroïde de chaque cluster avec au moins min_touches contacts
        result = []
        for c in clusters:
            if len(c) >= min_touches:
                result.append(float(np.mean(c)))
        return sorted(result)

    supports = [s for s in _cluster(pivot_lows) if s < current_price]
    resistances = [r for r in _cluster(pivot_highs) if r > current_price]

    return supports, resistances


# ─── Calcul des niveaux de trade ──────────────────────────────────────────────

def calculate_swing_levels(
    symbol: str,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    universe: str = "PENNY",
    capital: float = 10_000.0,
    atr_stop_mult: float | None = None,
    atr_target_mult: float | None = None,
    rubber_band: float | None = None,  # pour déterminer l'urgence du snapback
) -> TradeLevels:
    """
    Calcule les niveaux complets d'un trade swing.

    Logique :
    1. Calcule ATR(14) pour le sizing du stop
    2. Détecte supports/résistances
    3. Entrée = prix courant (ou légèrement au-dessus du support)
    4. Stop = support le plus proche - 1.5 ATR (ou entrée - 2 ATR min)
    5. Cible 1 = résistance la plus proche (si R:R ≥ 2)
    6. Cible 2 = résistance suivante ou entrée + 3 ATR
    7. Position sizing = capital × RISK_PCT / risk_per_share
    """
    atr_mult_s = atr_stop_mult if atr_stop_mult is not None else ATR_STOP_MULT
    atr_mult_t = atr_target_mult if atr_target_mult is not None else ATR_TARGET_MULT

    # ATR
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_val = float(tr.ewm(alpha=1/14, min_periods=10, adjust=False).mean().iloc[-1])
    if np.isnan(atr_val) or atr_val <= 0:
        atr_val = float(close.iloc[-1]) * 0.02   # fallback 2%

    current_price = float(close.iloc[-1])

    # Supports / résistances
    supports, resistances = find_support_resistance(close, high, low, lookback=60)

    # Entrée
    # Pour penny snapback : si rubber_band < -1.5, entrée agressive au prix courant
    # Sinon : entrée au-dessus du support le plus proche
    if supports and rubber_band is not None and rubber_band < -1.5:
        # Très oversold → entrée immédiate
        entry = current_price
    elif supports:
        near_support = max([s for s in supports if s <= current_price], default=current_price * 0.97)
        entry = round(near_support * 1.002, 4)  # 0.2% au-dessus du support
        entry = max(entry, current_price * 0.98)  # pas trop loin du prix courant
    else:
        entry = current_price

    # Stop-loss
    if supports:
        near_support = max([s for s in supports if s < entry], default=entry - atr_val * 2)
        stop = round(near_support - atr_mult_s * atr_val, 4)
    else:
        stop = round(entry - atr_mult_s * atr_val, 4)

    # Contrainte : stop max SWING_MAX_LOSS_PCT sous l'entrée
    max_stop_distance = entry * SWING_MAX_LOSS_PCT
    if (entry - stop) > max_stop_distance:
        stop = round(entry - max_stop_distance, 4)

    risk_per_share = max(entry - stop, 0.0001)

    # Cibles
    # Cible 1 : première résistance si R:R ≥ 2.0
    min_target_1 = entry + MIN_RISK_REWARD * risk_per_share

    if resistances:
        near_resist = min([r for r in resistances if r > entry], default=entry + atr_mult_t * atr_val)
        target_1 = max(near_resist, min_target_1)
    else:
        target_1 = round(entry + atr_mult_t * atr_val, 4)

    target_1 = max(target_1, min_target_1)

    # Cible 2 : résistance suivante ou cible 1 + 50%
    if len(resistances) >= 2:
        farther_resists = [r for r in resistances if r > target_1 * 1.005]
        target_2 = min(farther_resists, default=target_1 * 1.05)
    else:
        target_2 = round(target_1 * 1.05, 4)

    rr1 = round((target_1 - entry) / risk_per_share, 2)
    rr2 = round((target_2 - entry) / risk_per_share, 2)

    # Position sizing
    risk_capital = capital * RISK_PCT_PER_TRADE
    shares = int(risk_capital / risk_per_share)
    position_value = shares * entry
    position_pct = position_value / capital if capital > 0 else 0.0

    # Validation
    is_valid = True
    invalid_reason = ""

    if rr1 < MIN_RISK_REWARD:
        is_valid = False
        invalid_reason = f"R:R {rr1} < minimum {MIN_RISK_REWARD}"
    elif entry <= 0:
        is_valid = False
        invalid_reason = "Prix d'entrée invalide"
    elif stop <= 0:
        is_valid = False
        invalid_reason = "Stop-loss invalide"

    # Horizon selon l'univers
    horizon = 3 if universe in {"PENNY", "AI_PENNY"} else 10

    return TradeLevels(
        symbol=symbol,
        entry=round(entry, 4),
        stop_loss=round(stop, 4),
        target_1=round(target_1, 4),
        target_2=round(target_2, 4),
        risk_reward_1=rr1,
        risk_reward_2=rr2,
        risk_per_share=round(risk_per_share, 4),
        atr=round(atr_val, 4),
        universe=universe,
        trade_type="SWING" if universe in {"PENNY", "AI_PENNY"} else "DIP_BUY",
        horizon_days=horizon,
        position_size_shares=shares,
        position_size_pct=round(position_pct, 4),
        is_valid=is_valid,
        invalid_reason=invalid_reason,
    )


def calculate_dip_buy_levels(
    symbol: str,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    dip_pct: float,            # ex: -0.20 = dip de 20%
    price_52w_high: float,
    capital: float = 10_000.0,
) -> TradeLevels:
    """
    Calcule les niveaux d'achat pour un dip bluechip.

    Logique spécifique au dip bluechip :
    - Entrée dans la zone de support (SMA200 ou Fibonacci 61.8%)
    - Stop sous le plus bas récent - 1 ATR
    - Cible : récupération vers SMA50 puis SMA20
    """
    current = float(close.iloc[-1])

    # Fibonacci retracements depuis le 52w high
    swing_down = price_52w_high - close.rolling(252, min_periods=20).min().iloc[-1]
    fib_382 = price_52w_high - 0.382 * swing_down
    fib_500 = price_52w_high - 0.500 * swing_down
    fib_618 = price_52w_high - 0.618 * swing_down

    # Trouver le niveau fib le plus proche du prix actuel
    fib_levels = sorted([fib_382, fib_500, fib_618], key=lambda x: abs(x - current))
    nearest_fib = fib_levels[0]

    # L'entrée est au niveau fib le plus proche (si le prix y est, sinon au prix actuel)
    entry = min(current, nearest_fib * 1.01)  # achète au fib ou au prix courant si déjà dessous

    # Utiliser calculate_swing_levels pour le reste
    levels = calculate_swing_levels(
        symbol=symbol,
        close=close,
        high=high,
        low=low,
        volume=volume,
        universe="BLUECHIP",
        capital=capital,
        atr_stop_mult=1.5,
        atr_target_mult=4.0,   # cible plus large pour bluechip
    )

    # Override avec les niveaux fib
    levels.entry = round(entry, 4)
    levels.trade_type = "DIP_BUY"

    # Cibles = SMA50 et SMA20 (tendance de récupération naturelle)
    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])

    if sma50 > entry:
        levels.target_1 = round(sma50, 4)
    if sma20 > entry:
        levels.target_2 = round(sma20, 4)

    # Recalcul R:R
    rps = max(entry - levels.stop_loss, 0.0001)
    levels.risk_per_share = round(rps, 4)
    if levels.target_1 > entry:
        levels.risk_reward_1 = round((levels.target_1 - entry) / rps, 2)
    if levels.target_2 > entry:
        levels.risk_reward_2 = round((levels.target_2 - entry) / rps, 2)

    return levels


def trailing_stop_update(
    entry: float,
    current_price: float,
    highest_since_entry: float,
    atr: float,
    trail_pct: float | None = None,
    trail_atr_mult: float = 1.5,
    profit_trigger_pct: float = 0.15,
) -> dict[str, float]:
    """
    Met à jour un trailing stop dynamique.

    Logique :
    - Si profit < profit_trigger_pct : stop fixe ATR
    - Si profit ≥ profit_trigger_pct : trailing stop à trail_pct (ou trail_atr_mult × ATR)
      calculé depuis le plus haut depuis l'entrée.

    Returns:
        {"stop_loss": <float>, "mode": "FIXED"|"TRAILING", "profit_pct": <float>}
    """
    if trail_pct is None:
        trail_pct = float(os.getenv("PAPER_TRAIL_PCT", "0.04"))
    profit_trigger = float(os.getenv("PAPER_TRAIL_PROFIT_TRIGGER_PCT", str(profit_trigger_pct)))

    profit_pct = (current_price - entry) / entry if entry > 0 else 0.0

    if profit_pct < profit_trigger:
        # Stop fixe : entrée - 1.5 ATR
        stop = round(entry - 1.5 * atr, 4)
        mode = "FIXED"
    else:
        # Trailing depuis le plus haut
        trail_distance = max(trail_pct * highest_since_entry, trail_atr_mult * atr)
        stop = round(highest_since_entry - trail_distance, 4)
        mode = "TRAILING"

    return {
        "stop_loss": stop,
        "mode": mode,
        "profit_pct": round(profit_pct, 4),
        "highest_since_entry": round(highest_since_entry, 4),
    }


def score_trade_quality(levels: TradeLevels) -> dict[str, object]:
    """
    Score de qualité du trade (0–100) basé sur R:R, distance stop, position size.
    Utilisé pour filtrer les trades de basse qualité.
    """
    score = 0.0
    reasons = []

    # R:R
    if levels.risk_reward_1 >= 3.0:
        score += 30
        reasons.append("R:R excellent (≥3.0)")
    elif levels.risk_reward_1 >= 2.0:
        score += 20
        reasons.append("R:R bon (≥2.0)")
    elif levels.risk_reward_1 >= 1.5:
        score += 10
        reasons.append("R:R acceptable (≥1.5)")
    else:
        reasons.append(f"R:R faible ({levels.risk_reward_1})")

    # Stop distance
    stop_dist = levels.stop_distance_pct
    if stop_dist <= 0.03:
        score += 25
        reasons.append("Stop serré (≤3%)")
    elif stop_dist <= 0.05:
        score += 15
        reasons.append("Stop normal (≤5%)")
    elif stop_dist <= 0.08:
        score += 5
        reasons.append("Stop large (≤8%)")
    else:
        reasons.append(f"Stop trop large ({stop_dist*100:.1f}%)")

    # Position size raisonnable
    pos_pct = levels.position_size_pct
    if 0.02 <= pos_pct <= 0.10:
        score += 25
        reasons.append(f"Taille position raisonnable ({pos_pct*100:.1f}%)")
    elif pos_pct < 0.02:
        score += 10
        reasons.append("Position très petite")
    else:
        reasons.append(f"Position trop grande ({pos_pct*100:.1f}%)")

    # ATR valide
    if levels.atr > 0:
        score += 10
        reasons.append("ATR valide")

    # Validité globale
    if levels.is_valid:
        score += 10

    grade = (
        "A" if score >= 80
        else "B" if score >= 60
        else "C" if score >= 40
        else "D"
    )

    return {
        "score": round(score),
        "grade": grade,
        "reasons": reasons,
        "summary": f"Grade {grade} — Score {score}/100",
    }

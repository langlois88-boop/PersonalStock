from __future__ import annotations

"""
correlation_guard.py  — FIX #8
================================
Analyse de corrélation inter-actifs et garde-fous de concentration.

Problème actuel : l'optimizer ne calcule pas la matrice de corrélation.
Tu peux avoir 5 positions TECH corrélées à 0.95 sans aucune alerte.

Fonctionnalités :
- Matrice de corrélation rolling du portfolio
- Détection de clusters à haute corrélation (> seuil)
- Score de diversification (0–100)
- Alertes si ajout d'un titre augmente la corrélation globale
- Poids optimaux Markowitz simplifié (min-variance)
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CORR_ALERT_THRESHOLD = float(os.getenv("CORR_ALERT_THRESHOLD", "0.75"))
CORR_HIGH_THRESHOLD  = float(os.getenv("CORR_HIGH_THRESHOLD", "0.90"))
MAX_SECTOR_WEIGHT    = float(os.getenv("MAX_SECTOR_WEIGHT_PCT", "40")) / 100  # ex: 40%
LOOKBACK_DAYS        = int(os.getenv("CORR_LOOKBACK_DAYS", "60"))


# ─── Fetching des prix historiques ───────────────────────────────────────────

def _fetch_returns(symbols: list[str], days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Retourne un DataFrame de rendements journaliers pour chaque symbole."""
    try:
        import yfinance as yf
        data = yf.download(
            " ".join(symbols),
            period=f"{days}d",
            interval="1d",
            group_by="ticker",
            threads=True,
            auto_adjust=True,
            progress=False,
        )
        if data.empty:
            return pd.DataFrame()

        frames = {}
        for sym in symbols:
            try:
                if len(symbols) == 1:
                    close = data["Close"] if "Close" in data.columns else data.iloc[:, 0]
                elif isinstance(data.columns, pd.MultiIndex):
                    close = data[sym]["Close"] if sym in data else None
                else:
                    close = data["Close"].get(sym)
                if close is not None and not close.empty:
                    frames[sym] = close.pct_change().dropna()
            except Exception:
                continue

        if not frames:
            return pd.DataFrame()

        return pd.DataFrame(frames).dropna(how="all")
    except Exception as exc:
        logger.warning("_fetch_returns failed: %s", exc)
        return pd.DataFrame()


# ─── Matrice de corrélation ───────────────────────────────────────────────────

def compute_correlation_matrix(
    symbols: list[str],
    returns_df: pd.DataFrame | None = None,
    days: int = LOOKBACK_DAYS,
) -> dict[str, Any]:
    """
    Calcule la matrice de corrélation du portfolio.

    Returns:
        {
            "matrix": pd.DataFrame (corrélation),
            "symbols": list[str],
            "high_corr_pairs": [(sym1, sym2, corr), ...],   # > seuil d'alerte
            "clusters": [{symbols: [], avg_corr: float}, ...],
            "avg_portfolio_corr": float,
            "diversification_score": int (0–100),
            "alerts": [str],
        }
    """
    if len(symbols) < 2:
        return {
            "matrix": pd.DataFrame(),
            "symbols": symbols,
            "high_corr_pairs": [],
            "clusters": [],
            "avg_portfolio_corr": 0.0,
            "diversification_score": 100,
            "alerts": [],
        }

    if returns_df is None or returns_df.empty:
        returns_df = _fetch_returns(symbols, days)

    available = [s for s in symbols if s in returns_df.columns]
    if len(available) < 2:
        return {
            "matrix": pd.DataFrame(),
            "symbols": available,
            "high_corr_pairs": [],
            "clusters": [],
            "avg_portfolio_corr": 0.0,
            "diversification_score": 80,
            "alerts": ["Données de prix insuffisantes pour le calcul de corrélation."],
        }

    corr_matrix = returns_df[available].corr()

    # Paires à haute corrélation
    high_corr_pairs = []
    n = len(available)
    for i in range(n):
        for j in range(i + 1, n):
            c = corr_matrix.iloc[i, j]
            if abs(c) >= CORR_ALERT_THRESHOLD:
                high_corr_pairs.append((available[i], available[j], round(float(c), 3)))

    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    # Clustering simple (greedy)
    clusters = _build_correlation_clusters(corr_matrix, available, threshold=CORR_ALERT_THRESHOLD)

    # Corrélation moyenne du portfolio (off-diagonal)
    triu = np.triu(corr_matrix.values, k=1)
    n_pairs = n * (n - 1) / 2
    avg_corr = float(np.abs(triu).sum() / max(n_pairs, 1))

    # Score de diversification : 100 = parfaitement diversifié, 0 = tout corrélé
    diversity_score = max(0, min(100, int((1 - avg_corr) * 100)))

    # Alertes
    alerts = []
    if avg_corr > CORR_HIGH_THRESHOLD:
        alerts.append(f"🔴 Corrélation moyenne très élevée ({avg_corr:.2f}). Portfolio très concentré.")
    elif avg_corr > CORR_ALERT_THRESHOLD:
        alerts.append(f"🟡 Corrélation moyenne élevée ({avg_corr:.2f}). Diversification insuffisante.")

    for s1, s2, c in high_corr_pairs[:3]:
        if abs(c) >= CORR_HIGH_THRESHOLD:
            alerts.append(f"⚠️ {s1} ↔ {s2} : corrélation {c:.2f} (quasi-redondant)")
        elif abs(c) >= CORR_ALERT_THRESHOLD:
            alerts.append(f"📌 {s1} ↔ {s2} : corrélation élevée {c:.2f}")

    for cluster in clusters:
        if len(cluster["symbols"]) >= 3:
            alerts.append(
                f"🔗 Cluster haute corrélation: {', '.join(cluster['symbols'])} "
                f"(corr moy {cluster['avg_corr']:.2f})"
            )

    return {
        "matrix": corr_matrix,
        "matrix_dict": corr_matrix.round(3).to_dict(),
        "symbols": available,
        "high_corr_pairs": high_corr_pairs,
        "clusters": clusters,
        "avg_portfolio_corr": round(avg_corr, 3),
        "diversification_score": diversity_score,
        "alerts": alerts,
    }


def _build_correlation_clusters(
    corr_matrix: pd.DataFrame,
    symbols: list[str],
    threshold: float = CORR_ALERT_THRESHOLD,
) -> list[dict[str, Any]]:
    """Détecte les groupes de titres fortement corrélés (greedy clustering)."""
    visited = set()
    clusters = []

    for sym in symbols:
        if sym in visited:
            continue
        group = {sym}
        for other in symbols:
            if other == sym or other in visited:
                continue
            if abs(corr_matrix.loc[sym, other]) >= threshold:
                group.add(other)
        if len(group) >= 2:
            group_list = sorted(group)
            sub = corr_matrix.loc[group_list, group_list]
            triu = np.triu(sub.values, k=1)
            n = len(group_list)
            n_pairs = n * (n - 1) / 2
            avg_c = float(np.abs(triu).sum() / max(n_pairs, 1))
            clusters.append({"symbols": group_list, "avg_corr": round(avg_c, 3)})
            visited.update(group_list)

    return sorted(clusters, key=lambda x: -x["avg_corr"])


# ─── Impact d'un ajout sur la corrélation ────────────────────────────────────

def marginal_correlation_impact(
    new_symbol: str,
    existing_symbols: list[str],
    days: int = LOOKBACK_DAYS,
) -> dict[str, Any]:
    """
    Calcule l'impact d'ajouter new_symbol sur la corrélation du portfolio.

    Returns:
        {
            "new_symbol": str,
            "avg_corr_with_portfolio": float,
            "max_corr_with_single": float,
            "most_correlated_with": str,
            "recommendation": "ADD" | "CAUTION" | "AVOID",
            "reason": str,
        }
    """
    all_symbols = existing_symbols + [new_symbol]
    returns_df = _fetch_returns(all_symbols, days)

    if new_symbol not in returns_df.columns or len(existing_symbols) == 0:
        return {
            "new_symbol": new_symbol,
            "avg_corr_with_portfolio": 0.0,
            "max_corr_with_single": 0.0,
            "most_correlated_with": None,
            "recommendation": "ADD",
            "reason": "Pas de données suffisantes.",
        }

    new_returns = returns_df[new_symbol]
    corrs = {}
    for sym in existing_symbols:
        if sym in returns_df.columns:
            c = float(new_returns.corr(returns_df[sym]))
            if not np.isnan(c):
                corrs[sym] = c

    if not corrs:
        return {"new_symbol": new_symbol, "recommendation": "ADD", "reason": "Corrélation non calculable."}

    avg_corr = float(np.mean(list(corrs.values())))
    max_corr = max(corrs.values(), key=abs)
    most_corr_sym = max(corrs, key=lambda x: abs(corrs[x]))

    if abs(max_corr) >= CORR_HIGH_THRESHOLD:
        recommendation = "AVOID"
        reason = f"Corrélation très haute avec {most_corr_sym} ({max_corr:.2f}). Ajout redondant."
    elif abs(avg_corr) >= CORR_ALERT_THRESHOLD:
        recommendation = "CAUTION"
        reason = f"Corrélation moyenne élevée avec le portfolio ({avg_corr:.2f}). Diversification limitée."
    else:
        recommendation = "ADD"
        reason = f"Corrélation acceptable ({avg_corr:.2f}). Améliore la diversification."

    return {
        "new_symbol": new_symbol,
        "avg_corr_with_portfolio": round(avg_corr, 3),
        "max_corr_with_single": round(max_corr, 3),
        "most_correlated_with": most_corr_sym,
        "all_correlations": {k: round(v, 3) for k, v in sorted(corrs.items(), key=lambda x: -abs(x[1]))},
        "recommendation": recommendation,
        "reason": reason,
    }


# ─── Poids optimaux (min-variance Markowitz simplifié) ───────────────────────

def compute_min_variance_weights(
    symbols: list[str],
    returns_df: pd.DataFrame | None = None,
    days: int = LOOKBACK_DAYS,
    max_weight: float = 0.30,
    min_weight: float = 0.02,
) -> dict[str, float]:
    """
    Calcule les poids minimum-variance pour le portfolio.

    Algorithme : inverse-volatility weighting avec contrainte de corrélation.
    (Markowitz complet nécessite scipy.optimize, pas toujours dispo)

    Returns:
        {symbol: weight} — somme = 1.0
    """
    if len(symbols) < 2:
        return {s: 1.0 / len(symbols) for s in symbols}

    if returns_df is None or returns_df.empty:
        returns_df = _fetch_returns(symbols, days)

    available = [s for s in symbols if s in returns_df.columns]
    if not available:
        eq_weight = 1.0 / len(symbols)
        return {s: eq_weight for s in symbols}

    # Volatilité par symbole
    vols = {}
    for sym in available:
        std = float(returns_df[sym].std())
        vols[sym] = std if std > 0 else 0.01

    # Inverse-vol weighting (proxy min-variance)
    inv_vols = {s: 1.0 / v for s, v in vols.items()}
    total_inv = sum(inv_vols.values())
    raw_weights = {s: v / total_inv for s, v in inv_vols.items()}

    # Appliquer contraintes min/max
    weights = {}
    for sym, w in raw_weights.items():
        weights[sym] = max(min_weight, min(max_weight, w))

    # Renormaliser pour que la somme = 1.0
    total_w = sum(weights.values())
    weights = {s: round(w / total_w, 4) for s, w in weights.items()}

    # Symbols non disponibles = poids minimum
    for sym in symbols:
        if sym not in weights:
            weights[sym] = min_weight

    return weights


# ─── Rapport de diversification complet ──────────────────────────────────────

def portfolio_diversification_report(
    holdings: list[dict[str, Any]],   # [{symbol, sector, weight_pct, value}]
    days: int = LOOKBACK_DAYS,
) -> dict[str, Any]:
    """
    Rapport complet de diversification pour le portfolio optimizer.

    Ajouter dans PortfolioOptimizerEndpointView.post() :
        from .correlation_guard import portfolio_diversification_report
        div_report = portfolio_diversification_report(holdings_list)
        response_data["diversification"] = div_report
    """
    symbols = [h["symbol"] for h in holdings if h.get("symbol")]
    sectors = {}
    for h in holdings:
        sector = h.get("sector") or "Autre"
        val = float(h.get("value") or h.get("weight_pct") or 0)
        sectors[sector] = sectors.get(sector, 0) + val

    total_val = sum(sectors.values()) or 1
    sector_weights = {s: round(v / total_val, 3) for s, v in sectors.items()}

    # Alertes sectorielles
    sector_alerts = []
    for sector, w in sector_weights.items():
        if w >= MAX_SECTOR_WEIGHT:
            sector_alerts.append(
                f"⚠️ Concentration {sector}: {w*100:.1f}% (max recommandé {MAX_SECTOR_WEIGHT*100:.0f}%)"
            )

    # Matrice de corrélation
    corr_result = compute_correlation_matrix(symbols, days=days)

    # Score global
    sector_penalty = sum(max(0, w - MAX_SECTOR_WEIGHT) for w in sector_weights.values())
    diversity_score = max(0, corr_result["diversification_score"] - int(sector_penalty * 100))

    # Poids optimaux suggérés
    optimal_weights = compute_min_variance_weights(symbols, days=days)

    all_alerts = sector_alerts + corr_result["alerts"]

    return {
        "diversity_score": diversity_score,
        "sector_weights": sector_weights,
        "sector_alerts": sector_alerts,
        "correlation_matrix": corr_result["matrix_dict"],
        "high_corr_pairs": corr_result["high_corr_pairs"][:5],
        "corr_clusters": corr_result["clusters"],
        "avg_portfolio_corr": corr_result["avg_portfolio_corr"],
        "optimal_weights": optimal_weights,
        "all_alerts": all_alerts,
        "n_symbols": len(symbols),
        "n_sectors": len(sectors),
    }

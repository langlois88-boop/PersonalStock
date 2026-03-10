from __future__ import annotations

"""
drift_monitor_v2.py  — FIX #2
==============================
Remplace le calcul PSI dans compute_continuous_evaluation_daily().

Problème actuel : seulement 3 features surveillées sur ~40 du modèle.
    features = ['Volatility', 'Momentum20', 'RSI14']  ← BUGUÉ

Solution : surveillance de toutes les features disponibles dans les trades,
avec priorité sur les features critiques connues pour drifter.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ─── Features à surveiller par priorité ──────────────────────────────────────

# Features critiques — drift ici = problème immédiat
CRITICAL_FEATURES = [
    "RSI14", "rsi_14",
    "Volatility", "volatility_20",
    "Momentum20", "return_20d",
    "MACD_HIST", "macd_hist",
    "VolumeZ", "volume_zscore_20",
]

# Features secondaires — drift détecté mais seuil plus tolérant
SECONDARY_FEATURES = [
    "bb_pct_b_20",
    "rubber_band_20",
    "adx_14",
    "stoch_k_14",
    "obv_zscore",
    "cci_20",
    "sentiment_score",
    "mom_zscore_20",
    "vol_regime",
    "fib_distance_50",
]

# Features macro — drift lent mais structurel
MACRO_FEATURES = [
    "spy_corr_60",
    "sector_beta_60",
    "VIXCLS",
    "DCOILWTICO",
]

ALL_MONITORED_FEATURES = CRITICAL_FEATURES + SECONDARY_FEATURES + MACRO_FEATURES


def _psi(expected: list[float], actual: list[float], bins: int = 10) -> float:
    """
    Population Stability Index (PSI).
    PSI < 0.1  : stable
    PSI 0.1–0.2: drift modéré → surveiller
    PSI > 0.2  : drift significatif → retrain
    """
    if not expected or not actual:
        return 0.0
    expected_arr = np.array([x for x in expected if x is not None and not np.isnan(x)])
    actual_arr = np.array([x for x in actual if x is not None and not np.isnan(x)])
    if len(expected_arr) < 5 or len(actual_arr) < 5:
        return 0.0
    try:
        all_vals = np.concatenate([expected_arr, actual_arr])
        edges = np.percentile(all_vals, np.linspace(0, 100, bins + 1))
        edges = np.unique(edges)
        if len(edges) < 2:
            return 0.0

        exp_counts, _ = np.histogram(expected_arr, bins=edges)
        act_counts, _ = np.histogram(actual_arr, bins=edges)

        exp_pct = (exp_counts / max(exp_counts.sum(), 1)) + 1e-8
        act_pct = (act_counts / max(act_counts.sum(), 1)) + 1e-8

        psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
        return round(abs(psi), 6)
    except Exception:
        return 0.0


def _extract_feature_values(trades: list[Any], feature_name: str) -> list[float]:
    """Extrait les valeurs d'une feature depuis entry_features des trades."""
    values = []
    for t in trades:
        features = getattr(t, "entry_features", None) or {}
        if not isinstance(features, dict):
            continue
        val = features.get(feature_name)
        if val is None:
            # Essayer aussi la version CamelCase / lowercase
            alt_keys = [feature_name.lower(), feature_name.upper()]
            for k in alt_keys:
                val = features.get(k)
                if val is not None:
                    break
        if val is not None:
            try:
                f = float(val)
                if not np.isnan(f) and not np.isinf(f):
                    values.append(f)
            except (TypeError, ValueError):
                continue
    return values


def compute_full_drift(
    baseline_trades: list[Any],
    current_trades: list[Any],
    model_name: str = "BLUECHIP",
    critical_threshold: float | None = None,
    secondary_threshold: float | None = None,
) -> dict[str, Any]:
    """
    Calcule le PSI sur TOUTES les features disponibles.

    Args:
        baseline_trades : trades de la période de référence (ex: 30 derniers jours)
        current_trades  : trades récents (ex: 7 derniers jours)
        model_name      : "BLUECHIP" | "PENNY"
        critical_threshold  : seuil PSI pour features critiques (défaut: DRIFT_PSI_THRESHOLD)
        secondary_threshold : seuil PSI pour features secondaires (plus tolérant)

    Returns:
        {
            psi_metrics: {feature: psi_value},
            feature_stats: {feature: {baseline_mean, current_mean, drift_direction}},
            max_psi: float,
            critical_drifts: [feature],
            secondary_drifts: [feature],
            total_features_monitored: int,
            should_retrain: bool,
            drift_summary: str,
        }
    """
    env_prefix = model_name.upper()
    crit_thresh = critical_threshold or float(
        os.getenv(f"{env_prefix}_DRIFT_PSI_THRESHOLD",
        os.getenv("DRIFT_PSI_THRESHOLD", "0.2"))
    )
    sec_thresh = secondary_threshold or crit_thresh * 1.5  # features secondaires : seuil +50%

    psi_metrics: dict[str, float] = {}
    feature_stats: dict[str, dict] = {}
    critical_drifts: list[str] = []
    secondary_drifts: list[str] = []

    # Découvrir toutes les features disponibles dans les trades
    available_features = set(ALL_MONITORED_FEATURES)
    for t in (baseline_trades + current_trades)[:10]:
        features_dict = getattr(t, "entry_features", None) or {}
        if isinstance(features_dict, dict):
            available_features.update(features_dict.keys())

    monitored = 0
    for feat in sorted(available_features):
        baseline_vals = _extract_feature_values(baseline_trades, feat)
        current_vals = _extract_feature_values(current_trades, feat)

        if len(baseline_vals) < 5 or len(current_vals) < 5:
            continue

        psi_val = _psi(baseline_vals, current_vals)
        psi_metrics[feat] = psi_val
        monitored += 1

        baseline_mean = float(np.mean(baseline_vals))
        current_mean = float(np.mean(current_vals))
        drift_direction = "↑" if current_mean > baseline_mean * 1.05 else "↓" if current_mean < baseline_mean * 0.95 else "→"

        feature_stats[feat] = {
            "baseline_mean": round(baseline_mean, 4),
            "current_mean": round(current_mean, 4),
            "drift_direction": drift_direction,
            "psi": psi_val,
            "n_baseline": len(baseline_vals),
            "n_current": len(current_vals),
        }

        is_critical = any(feat == cf or feat.lower() == cf.lower() for cf in CRITICAL_FEATURES)
        threshold = crit_thresh if is_critical else sec_thresh

        if psi_val >= threshold:
            if is_critical:
                critical_drifts.append(feat)
            else:
                secondary_drifts.append(feat)

    max_psi = max(psi_metrics.values(), default=0.0)
    should_retrain = len(critical_drifts) > 0 or max_psi >= crit_thresh

    # Résumé textuel
    if critical_drifts:
        drift_summary = f"CRITIQUE: drift sur {', '.join(critical_drifts[:3])}"
    elif secondary_drifts:
        drift_summary = f"Drift modéré sur {len(secondary_drifts)} features secondaires"
    elif max_psi > 0:
        drift_summary = f"Stable (PSI max {max_psi:.3f})"
    else:
        drift_summary = "Pas de données drift"

    return {
        "psi_metrics": psi_metrics,
        "feature_stats": feature_stats,
        "max_psi": round(max_psi, 6),
        "critical_drifts": critical_drifts,
        "secondary_drifts": secondary_drifts,
        "total_features_monitored": monitored,
        "should_retrain": should_retrain,
        "drift_summary": drift_summary,
    }


# ─── Patch pour compute_continuous_evaluation_daily dans tasks.py ─────────────
#
# Remplacer :
#   features = ['Volatility', 'Momentum20', 'RSI14']
#   psi_metrics = {}
#   for feat in features:
#       expected = [...]
#       actual = [...]
#       psi_metrics[feat] = _psi(expected, actual)
#
# Par :
#   from .drift_monitor_v2 import compute_full_drift
#   drift_result = compute_full_drift(
#       baseline_trades=list(baseline),
#       current_trades=list(trades),
#       model_name=model_name,
#   )
#   psi_metrics = drift_result["psi_metrics"]
#   feature_stats = drift_result["feature_stats"]
#
# Et mettre à jour ModelDriftDaily :
#   ModelDriftDaily.objects.update_or_create(
#       ...,
#       defaults={
#           'psi': psi_metrics,
#           'feature_stats': feature_stats,
#           # NOUVEAUX champs (migration requise) :
#           'critical_drifts': drift_result['critical_drifts'],
#           'total_features_monitored': drift_result['total_features_monitored'],
#           'should_retrain': drift_result['should_retrain'],
#           'drift_summary': drift_result['drift_summary'],
#       },
#   )
#
# Et dans auto_retrain_on_drift_daily :
#   max_psi = drift_result["max_psi"]
#   should_retrain = drift_result["should_retrain"]

from __future__ import annotations

"""Penny model pipeline orchestrator."""

import logging
import os
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from portfolio.ml_engine.config import config
from portfolio.ml_engine.data.market import fetch_history
from portfolio.ml_engine.export.onnx_exporter import OnnxExporter
from portfolio.ml_engine.export.push_model import push_model
from portfolio.ml_engine.feature_registry import PENNY_FEATURE_NAMES
from portfolio.ml_engine.features.technical import rsi, sma_ratio, volatility, volume_zscore, rvol
from portfolio.ml_engine.pipeline_fixes import validate_label_distribution
from portfolio.ml_engine.training.labeling import triple_barrier_events_adaptive, universe_max_days
from portfolio.ml_engine.training.trainer import Trainer
# DeepSeek sample weighting — kept for reference, not used by default (see
# `_resolve_sample_weights` below). Left intact and importable so someone
# can still call it directly if they need to.
from portfolio.ml_engine.training.deepseek_weighter import build_sample_weights  # noqa: F401
from portfolio.ml_engine.training.sample_weights import build_sample_weights_uniqueness, max_observed_horizon
from portfolio.ml_engine.registry.model_registry import LocalModelRegistry

# Single source of truth: feature_registry.py::PENNY_FEATURE_NAMES — also
# used by train_penny_model.py, and matches exactly what build_dataset()
# below computes.
PENNY_FEATURES = list(PENNY_FEATURE_NAMES)

# A reduced subset for PENNY_FEATURE_SET=core — pipeline-specific selection,
# not part of the shared registry.
PENNY_FEATURES_CORE = [
    "volume_zscore_20",
    "rvol_20",
    "return_5d",
]


def _ensure_django() -> None:
    """Ensure Django settings loaded."""
    import os
    if not os.getenv("DJANGO_SETTINGS_MODULE"):
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "portfolio_backend.settings")
    import django
    django.setup()


def _setup_logging() -> None:
    """Configure file logging for training."""
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"training_PENNY_{datetime.utcnow().date().isoformat()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )


def _pipeline_factory() -> Pipeline:
    """Create the sklearn pipeline."""
    return Pipeline([
        ("scaler", RobustScaler()),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=20,
                max_features="sqrt",
                random_state=42,
                class_weight="balanced_subsample",
            ),
        ),
    ])


def _select_features_by_importance(
    X: pd.DataFrame,
    y: pd.Series,
    min_importance: float,
    pipeline_factory: callable,
) -> tuple[pd.DataFrame, list[str]]:
    if min_importance <= 0:
        return X, list(X.columns)

    pipe = pipeline_factory()
    pipe.fit(X, y)
    model = pipe.named_steps.get("classifier")
    if not hasattr(model, "feature_importances_"):
        return X, list(X.columns)

    importances = np.array(model.feature_importances_, dtype=float)
    feature_names = list(X.columns)
    pairs = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)
    selected = [name for name, score in pairs if score >= min_importance]
    if len(selected) < 3:
        selected = [name for name, _ in pairs[:3]]

    dropped = [name for name in feature_names if name not in selected]
    logging.info("Feature importances (top 5): %s", pairs[:5])
    if dropped:
        logging.info("Dropped low-importance features: %s", dropped)

    return X[selected], selected


def _write_feature_importance(universe: str, model: object, feature_names: list[str]) -> None:
    if not hasattr(model, "feature_importances_"):
        return
    importances = getattr(model, "feature_importances_", None)
    if importances is None or len(importances) != len(feature_names):
        return
    payload = {
        "universe": universe,
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "features": [
            {"name": feature_names[i], "value": float(importances[i]) * 100}
            for i in range(len(feature_names))
        ],
    }
    path = Path(os.getenv("FEATURE_IMPORTANCE_PATH", "/app/logs/feature_importance.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if path.exists():
            existing = json.loads(path.read_text())
        else:
            existing = {}
    except Exception:
        existing = {}
    existing[universe] = payload
    path.write_text(json.dumps(existing, indent=2))


def _get_penny_symbols() -> list[str]:
    _ensure_django()
    from portfolio.models import SandboxWatchlist, Stock

    watchlist = SandboxWatchlist.objects.filter(sandbox="AI_PENNY").first()
    if watchlist and watchlist.symbols:
        return [s for s in watchlist.symbols if s]

    qs = Stock.objects.filter(latest_price__isnull=False, latest_price__gt=0, latest_price__lte=5)
    symbols = [s.symbol for s in qs.order_by("symbol") if s.symbol]
    if symbols:
        return symbols

    return [s.symbol for s in Stock.objects.all().order_by("symbol") if s.symbol]


def build_dataset(
    symbols: list[str],
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str], list, list, dict[str, tuple[pd.Index, pd.Series]]]:
    """Build feature matrix and labels for penny universe.

    Returns X, y, sample_symbols, sample_dates, sample_t0, sample_t1,
    symbol_context. ``sample_t0[i]``/``sample_t1[i]`` are the triple-barrier
    window start/end for row i (real index values, unlike the stringified
    `sample_dates`), and ``symbol_context[symbol] = (close_index,
    log_returns)`` — both needed by `build_sample_weights_uniqueness` for
    overlap-based sample weighting, computed per symbol (concurrency is a
    single-instrument concept) then concatenated in the same row order as
    X/y in `run()`.
    """
    rows: list[dict[str, float]] = []
    labels: list[int] = []
    sample_symbols: list[str] = []
    sample_dates: list[str] = []
    sample_t0: list = []
    sample_t1: list = []
    symbol_context: dict[str, tuple[pd.Index, pd.Series]] = {}

    _ensure_django()
    from portfolio.ml_engine.collectors.news_rss import fetch_news_sentiment

    for symbol in symbols:
        hist = fetch_history(symbol, period="2y", interval="1d")
        if hist is None or hist.empty or "Close" not in hist.columns:
            continue
        close = hist["Close"].astype(float)
        volume = hist.get("Volume", pd.Series(0.0, index=close.index)).astype(float)
        high = hist.get("High", close)
        low = hist.get("Low", close)

        if len(close) < 40:
            continue
        rsi_14 = rsi(close, 14)
        sma_10_20 = sma_ratio(close, 10, 20)
        vol_20 = volatility(close, 20)
        vol_z = volume_zscore(volume, 20)
        rvol_20 = rvol(volume, 20)
        ret_5 = close.pct_change(5)
        sentiment_score = float(fetch_news_sentiment(symbol).get("news_sentiment") or 0.0)
        log_returns = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        symbol_context[symbol] = (close.index, log_returns)

        use_atr = os.getenv("PENNY_USE_ATR_BARRIERS", "true").lower() in {"1", "true", "yes", "y"}
        events = triple_barrier_events_adaptive(
            close,
            high=high,
            low=low,
            universe="PENNY",
            use_atr_barriers=use_atr,
        )
        label_series = events["label"]
        t1_series = events["t1"]

        valid_mask = (
            ~rsi_14.isna()
            & ~sma_10_20.isna()
            & ~vol_20.isna()
            & ~vol_z.isna()
            & ~rvol_20.isna()
            & ~ret_5.isna()
        )

        for idx in range(len(close)):
            if not valid_mask.iloc[idx]:
                continue
            feat_row = {
                "rsi_14": float(rsi_14.iloc[idx]) if not pd.isna(rsi_14.iloc[idx]) else np.nan,
                "sma_ratio_10_20": float(sma_10_20.iloc[idx]) if not pd.isna(sma_10_20.iloc[idx]) else 1.0,
                "volatility_20": float(vol_20.iloc[idx]) if not pd.isna(vol_20.iloc[idx]) else np.nan,
                "volume_zscore_20": float(vol_z.iloc[idx]) if not pd.isna(vol_z.iloc[idx]) else 0.0,
                "rvol_20": float(rvol_20.iloc[idx]) if not pd.isna(rvol_20.iloc[idx]) else 1.0,
                "return_5d": float(ret_5.iloc[idx]) if not pd.isna(ret_5.iloc[idx]) else np.nan,
                "sentiment_score": sentiment_score,
            }
            label_val = label_series.iloc[idx]
            if pd.isna(label_val):
                continue
            if any(pd.isna(v) for v in feat_row.values()):
                continue
            rows.append(feat_row)
            labels.append(int(label_val))
            sample_symbols.append(symbol)
            sample_dates.append(str(close.index[idx].date()))
            sample_t0.append(close.index[idx])
            sample_t1.append(t1_series.iloc[idx])

    X = pd.DataFrame(rows)
    X = X[PENNY_FEATURES]
    y = pd.Series(labels)
    return X, y, sample_symbols, sample_dates, sample_t0, sample_t1, symbol_context


def _deepseek_weighting_enabled() -> bool:
    return (os.getenv("DEEPSEEK_WEIGHTING_ENABLED") or "").lower() in {"1", "true", "yes", "y"}


def _uniqueness_weighting_enabled() -> bool:
    return (os.getenv("PENNY_UNIQUENESS_WEIGHTING_ENABLED", "true") or "").lower() in {"1", "true", "yes", "y"}


def _resolve_sample_weights(
    sample_symbols: list[str],
    sample_dates: list[str],
    sample_t0: list,
    sample_t1: list,
    symbol_context: dict[str, tuple[pd.Index, pd.Series]],
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[np.ndarray, int | None]:
    """Resolve sample weights and, when uniqueness weighting drives them,
    the purge window derived from the labels' actually-observed horizon.

    Two mutually-exclusive strategies — never combine them silently:
    - Uniqueness weighting (default): Lopez de Prado average-uniqueness
      from triple-barrier label overlap, computed per symbol then
      concatenated in row order (concurrency is single-instrument).
    - DeepSeek confidence weighting (opt-in via DEEPSEEK_WEIGHTING_ENABLED):
      kept intact in training/deepseek_weighter.py for reference, off by
      default (returns uniform 1.0 weights when disabled, as before).

    Returns:
        (weights, purge_window) — `purge_window` is the max horizon (in
        bars) actually observed across all labels' t1 when uniqueness
        weighting computed them, or None to fall back to the configured
        `PENNY_TRIPLE_BARRIER_MAX_DAYS` constant.
    """
    deepseek_enabled = _deepseek_weighting_enabled()
    uniqueness_enabled = _uniqueness_weighting_enabled()
    if deepseek_enabled and uniqueness_enabled:
        raise RuntimeError(
            "Both DEEPSEEK_WEIGHTING_ENABLED and uniqueness weighting "
            "(PENNY_UNIQUENESS_WEIGHTING_ENABLED) are active. These are two "
            "different, mutually-exclusive sample-weighting strategies — "
            "combining them silently would double-weight in ways neither "
            "was designed for. Disable one: unset DEEPSEEK_WEIGHTING_ENABLED, "
            "or set PENNY_UNIQUENESS_WEIGHTING_ENABLED=false."
        )

    if not uniqueness_enabled:
        weights = build_sample_weights(sample_symbols, sample_dates, X.to_dict("records"), y.tolist())
        return np.array(weights), None

    rows_by_symbol: dict[str, list[int]] = {}
    for row_pos, symbol in enumerate(sample_symbols):
        rows_by_symbol.setdefault(symbol, []).append(row_pos)

    weights_arr = np.ones(len(sample_symbols), dtype=float)
    max_horizon_bars = 0
    for symbol, row_positions in rows_by_symbol.items():
        close_index, log_returns = symbol_context[symbol]
        t0_vals = [sample_t0[p] for p in row_positions]
        t1_vals = [sample_t1[p] for p in row_positions]
        t1_series = pd.Series(t1_vals, index=pd.Index(t0_vals))

        sym_weights = build_sample_weights_uniqueness(
            close_index=close_index,
            t1=t1_series,
            returns=log_returns,
            use_return_attribution=True,
        )
        max_horizon_bars = max(max_horizon_bars, max_observed_horizon(t1_series, close_index))
        for row_pos, t0 in zip(row_positions, t0_vals):
            w = sym_weights.loc[t0]
            weights_arr[row_pos] = float(w) if not pd.isna(w) else 1.0

    return weights_arr, (max_horizon_bars or None)


def run() -> None:
    """Execute penny training pipeline."""
    _setup_logging()
    logging.info("Training started")

    symbols = _get_penny_symbols()
    X, y, sample_symbols, sample_dates, sample_t0, sample_t1, symbol_context = build_dataset(symbols)
    if X.empty:
        raise RuntimeError("No training samples available for penny pipeline")

    feature_set = os.getenv("PENNY_FEATURE_SET", "full").strip().lower()
    base_features = PENNY_FEATURES_CORE if feature_set == "core" else PENNY_FEATURES
    X = X[base_features]

    label_validation = validate_label_distribution(y, universe="PENNY")
    if label_validation.get("warning"):
        logging.warning("Label distribution warning: %s", label_validation["warning"])
    if not label_validation.get("ok", True):
        raise RuntimeError("Label distribution not suitable for training")

    min_importance = float(os.getenv("FEATURE_IMPORTANCE_MIN_PENNY", "0"))
    X, selected_features = _select_features_by_importance(X, y, min_importance, _pipeline_factory)
    if selected_features != base_features:
        logging.info("Using %s features after selection", len(selected_features))

    weights, observed_purge_window = _resolve_sample_weights(
        sample_symbols, sample_dates, sample_t0, sample_t1, symbol_context, X, y,
    )
    # Purge window matches the triple-barrier label horizon so no training
    # fold ends within a label's forward-looking window of the test fold.
    # Prefer the horizon actually observed in this run's labels (some
    # resolve before timeout) over the configured constant, when available.
    if observed_purge_window is not None:
        purge_window = observed_purge_window
        logging.info("Purge window derived from observed label t1: %s bars", purge_window)
    else:
        purge_window = universe_max_days("PENNY")
    trainer = Trainer(_pipeline_factory, purge_window=purge_window)
    min_cv = float(config.model.min_cv_mean)
    min_f1 = float(config.model.min_wf_f1)
    if "MIN_CV_MEAN_PENNY" in os.environ:
        min_cv = float(os.getenv("MIN_CV_MEAN_PENNY", min_cv))
    if "MIN_WF_F1_PENNY" in os.environ:
        min_f1 = float(os.getenv("MIN_WF_F1_PENNY", min_f1))
    result = trainer.fit(X, y, sample_weight=weights, min_cv_mean=min_cv, min_wf_f1=min_f1)
    logging.info("cv_mean=%.4f wf_f1=%.4f passed_gate=%s", result.cv_mean, result.wf_f1, result.passed_gate)

    _write_feature_importance("PENNY", result.model.named_steps.get("classifier"), selected_features)

    exporter = OnnxExporter()
    output_path = Path("/app/portfolio/ml_engine/models/scout_brain_v1.onnx")
    onnx_path = exporter.export(result, output_path, "PENNY", expected_feature_count=len(selected_features))

    registry = LocalModelRegistry(Path("/app/portfolio/ml_engine/models/registry"))
    meta = json.loads(onnx_path.with_suffix(".json").read_text())
    registry.register("penny", onnx_path, meta)

    if config.integration.auto_push:
        push_model("penny", onnx_path, meta)

    logging.info("Training completed")


if __name__ == "__main__":
    run()

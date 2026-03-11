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
from portfolio.ml_engine.features.technical import rsi, sma_ratio, volatility, volume_zscore, rvol
from portfolio.ml_engine.pipeline_fixes import triple_barrier_labels_adaptive, validate_label_distribution
from portfolio.ml_engine.training.trainer import Trainer
from portfolio.ml_engine.training.deepseek_weighter import build_sample_weights
from portfolio.ml_engine.registry.model_registry import LocalModelRegistry

PENNY_FEATURES = [
    "rsi_14",
    "sma_ratio_10_20",
    "volatility_20",
    "volume_zscore_20",
    "rvol_20",
    "return_5d",
    "sentiment_score",
]

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


def build_dataset(symbols: list[str]) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Build feature matrix and labels for penny universe."""
    rows: list[dict[str, float]] = []
    labels: list[int] = []
    sample_symbols: list[str] = []
    sample_dates: list[str] = []

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

        use_atr = os.getenv("PENNY_USE_ATR_BARRIERS", "true").lower() in {"1", "true", "yes", "y"}
        label_series = triple_barrier_labels_adaptive(
            close,
            high=high,
            low=low,
            universe="PENNY",
            use_atr_barriers=use_atr,
        )

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

    X = pd.DataFrame(rows)
    X = X[PENNY_FEATURES]
    y = pd.Series(labels)
    return X, y, sample_symbols, sample_dates


def run() -> None:
    """Execute penny training pipeline."""
    _setup_logging()
    logging.info("Training started")

    symbols = _get_penny_symbols()
    X, y, sample_symbols, sample_dates = build_dataset(symbols)
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

    trainer = Trainer(_pipeline_factory)
    weights = build_sample_weights(sample_symbols, sample_dates, X.to_dict("records"), y.tolist())
    min_cv = float(config.model.min_cv_mean)
    min_f1 = float(config.model.min_wf_f1)
    if "MIN_CV_MEAN_PENNY" in os.environ:
        min_cv = float(os.getenv("MIN_CV_MEAN_PENNY", min_cv))
    if "MIN_WF_F1_PENNY" in os.environ:
        min_f1 = float(os.getenv("MIN_WF_F1_PENNY", min_f1))
    result = trainer.fit(X, y, sample_weight=np.array(weights), min_cv_mean=min_cv, min_wf_f1=min_f1)

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

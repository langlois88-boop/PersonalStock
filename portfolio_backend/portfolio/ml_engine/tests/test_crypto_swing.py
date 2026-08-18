"""Tests pour le pipeline crypto SWING (2026-08-18, crypto_training.py) --
bougies journalières, horizon de jours/semaines, réutilise build_full_
feature_set et add_pattern_columns (déjà éprouvés sur les actions) au lieu
de réinventer des formules crypto maison. Distinct du pipeline intraday
existant (CRYPTO_FEATURE_NAMES, 15m/1h, jamais entraîné en prod)."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from portfolio.ml_engine.feature_registry import CRYPTO_SWING_FEATURE_NAMES


def _synthetic_daily_ohlcv(n: int = 400, seed: int = 0, trend: float = 0.0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    close = 100 + np.cumsum(rng.randn(n) * 1.5 + trend)
    close = np.clip(close, 1.0, None)  # jamais négatif ou nul
    return pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=n, freq="D"),
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": rng.uniform(1_000_000, 5_000_000, n),
    })


def test_crypto_swing_feature_names_match_what_swing_training_computes() -> None:
    """Même classe de bug que test_crypto_feature_names_match_what_crypto_
    training_computes (features listées dans le registre mais jamais
    réellement produites -> KeyError garanti sur dropna) -- vérifié ici pour
    le nouveau pipeline swing avant qu'il ne tourne jamais en production."""
    from portfolio.ml_engine.crypto_training import build_crypto_swing_dataset

    df = _synthetic_daily_ohlcv(n=400)
    with patch("portfolio.ml_engine.crypto_training.fetch_crypto_history_daily", return_value=df):
        dataset, labels, feature_cols = build_crypto_swing_dataset(["BTC-USD"], years=2)

    assert feature_cols == list(CRYPTO_SWING_FEATURE_NAMES)
    assert len(feature_cols) > 30  # sanity : pas le stub 6-features intraday
    assert not dataset.empty
    for col in feature_cols:
        assert col in dataset.columns, f"{col} listé dans CRYPTO_SWING_FEATURE_NAMES mais absent du dataset"
        assert dataset[col].notna().all(), f"{col} contient des NaN après dropna"


def test_pattern_columns_are_0_or_1_not_nan() -> None:
    """Les 4 features de patterns de bougies (pattern_doji, pattern_hammer,
    pattern_engulfing, pattern_morning_star) doivent être des booléens 0/1
    -- pas des NaN qui se feraient droper silencieusement avec le reste des
    lignes valides par le dropna() de build_crypto_swing_dataset."""
    from portfolio.ml_engine.crypto_training import _build_crypto_swing_features

    df = _synthetic_daily_ohlcv(n=250)
    features = _build_crypto_swing_features(df)
    for col in ("pattern_doji", "pattern_hammer", "pattern_engulfing", "pattern_morning_star"):
        assert set(features[col].dropna().unique()) <= {0, 1}


def test_btc_corr_60_is_zero_for_btc_itself() -> None:
    """btc_corr_60 -- rendements de BTC-USD comme ancre au lieu du S&P500
    (réutilisation de build_full_feature_set via spy_returns) -- doit valoir
    0 pour BTC-USD lui-même (rien à corréler avec soi-même), pas un artefact
    de corrélation à 1.0 qui donnerait un faux signal 'toujours aligné'."""
    from portfolio.ml_engine.crypto_training import _build_crypto_swing_features

    df = _synthetic_daily_ohlcv(n=250, seed=1)
    features_no_btc = _build_crypto_swing_features(df, btc_df=None)
    assert (features_no_btc["btc_corr_60"] == 0.0).all()


def test_label_swing_targets_flags_a_real_10_percent_rally() -> None:
    from portfolio.ml_engine.crypto_training import _label_swing_targets

    # 20 jours plats, puis un rally de +15% sur les 10 jours suivants -- le
    # jour 0 doit être labellisé "gagnant" (horizon=10, target_pct=0.08).
    flat = [100.0] * 20
    rally = list(np.linspace(100.0, 115.0, 10))
    close = pd.Series(flat[:10] + rally + [115.0] * 10)  # aligne les longueurs
    labels = _label_swing_targets(close, horizon_days=10, target_pct=0.08)

    assert labels.iloc[9] == 1  # jour juste avant le rally : +15% en 10j, capture le seuil de 8%
    assert labels.iloc[0] == 0  # jour plat suivi de 10 jours plats : aucun mouvement


def test_train_crypto_swing_model_raises_on_insufficient_history() -> None:
    """Non-régression : un univers avec un historique trop court pour
    calculer les features rolling (ex. MA200) doit échouer proprement
    (ValueError explicite), pas planter plus loin avec un KeyError obscur."""
    from portfolio.ml_engine.crypto_training import train_crypto_swing_model

    tiny_df = _synthetic_daily_ohlcv(n=5)
    with patch("portfolio.ml_engine.crypto_training.fetch_crypto_history_daily", return_value=tiny_df):
        with pytest.raises(ValueError, match="No crypto swing dataset available"):
            train_crypto_swing_model(["BTC-USD"], years=1)


def test_save_crypto_swing_model_rejects_below_cv_quality_gate(tmp_path) -> None:
    from portfolio.ml_engine.crypto_training import save_crypto_swing_model

    weak_payload = {
        "model": object(),
        "features": list(CRYPTO_SWING_FEATURE_NAMES),
        "cv_scores": [0.40, 0.42, 0.38],
        "cv_mean": 0.40,  # < CRYPTO_SWING_CV_MIN (0.55 par défaut)
        "walk_forward": [{"f1": 0.60}],
        "n_samples": 500,
        "label_balance": 0.3,
    }
    with pytest.raises(ValueError, match="CV mean"):
        save_crypto_swing_model(weak_payload, output_path=tmp_path / "test_swing_model.pkl")
    assert not (tmp_path / "test_swing_model.pkl").exists()


def test_save_crypto_swing_model_rejects_below_walk_forward_gate(tmp_path) -> None:
    from portfolio.ml_engine.crypto_training import save_crypto_swing_model

    weak_payload = {
        "model": object(),
        "features": list(CRYPTO_SWING_FEATURE_NAMES),
        "cv_scores": [0.60, 0.62, 0.58],
        "cv_mean": 0.60,  # passe le seuil CV...
        "walk_forward": [{"f1": 0.20}, {"f1": 0.25}],  # ...mais pas walk-forward (< 0.50)
        "n_samples": 500,
        "label_balance": 0.3,
    }
    with pytest.raises(ValueError, match="Walk-forward"):
        save_crypto_swing_model(weak_payload, output_path=tmp_path / "test_swing_model.pkl")


def test_swing_and_intraday_pipelines_use_separate_model_paths() -> None:
    """Non-régression : le nouveau pipeline swing ne doit jamais pouvoir
    écraser crypto_brain_v1.pkl (intraday) ni partager sa clé d'env --
    les deux systèmes doivent pouvoir coexister sans interférence."""
    from portfolio.ml_engine.crypto_training import CRYPTO_MODEL_PATH, CRYPTO_SWING_MODEL_PATH

    assert CRYPTO_MODEL_PATH != CRYPTO_SWING_MODEL_PATH
    assert CRYPTO_MODEL_PATH.name == "crypto_brain_v1.pkl"
    assert CRYPTO_SWING_MODEL_PATH.name == "crypto_swing_brain_v1.pkl"

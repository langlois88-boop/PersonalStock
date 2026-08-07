import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import portfolio.ml_engine.intraday_training as intraday_training
from portfolio.ml_engine.intraday_training import PreFittedVotingEnsemble, train_voting_ensemble


def _synthetic_dataset(offset: float, n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    rng = np.random.RandomState(seed)
    values = rng.randn(n) + offset
    df = pd.DataFrame({"f": values})
    labels = pd.Series((values > offset).astype(int))
    return df, labels, ["f"]


def test_prefitted_voting_ensemble_blends_without_refitting() -> None:
    """The ensemble must only blend predict_proba of already-fit estimators —
    it must never re-fit an estimator on another estimator's data."""
    X_blue, y_blue, _ = _synthetic_dataset(offset=0.0, seed=1)
    X_penny, y_penny, _ = _synthetic_dataset(offset=5.0, seed=2)

    blue_model = RandomForestClassifier(n_estimators=20, random_state=42).fit(X_blue, y_blue)
    penny_model = RandomForestClassifier(n_estimators=20, random_state=42).fit(X_penny, y_penny)

    ensemble = PreFittedVotingEnsemble(
        estimators=[("bluechip_expert", blue_model), ("penny_expert", penny_model)],
        weights=[0.7, 0.3],
    )

    probe = X_blue.iloc[:10]
    probs = ensemble.predict_proba(probe)
    expected = blue_model.predict_proba(probe) * 0.7 + penny_model.predict_proba(probe) * 0.3
    assert np.allclose(probs, expected)

    # The two estimators must still disagree — proof neither was silently
    # re-fit on the combined data, which would make their outputs converge.
    assert not np.allclose(
        blue_model.predict_proba(X_penny.iloc[:10]),
        penny_model.predict_proba(X_penny.iloc[:10]),
    )


def test_train_voting_ensemble_specializes_each_expert(monkeypatch) -> None:
    """Regression test for the leakage bug: train_voting_ensemble used to fit
    the whole VotingClassifier on the bluechip+penny data concatenated
    together (``pipeline.fit(X_all, y_all)``), which silently re-fit both
    ``bluechip_expert`` and ``penny_expert`` on the same combined data and
    erased the intended per-universe specialization. Each expert must be fit
    only on its own universe's data."""

    def _fake_build_dataset(symbols, market_symbol="QQQ", days=365):
        symbols = list(symbols)
        if symbols and symbols[0].startswith("BLUE"):
            return _synthetic_dataset(offset=0.0, seed=10)
        return _synthetic_dataset(offset=5.0, seed=20)

    monkeypatch.setattr(intraday_training, "build_dataset", _fake_build_dataset)
    monkeypatch.setenv("INTRADAY_ENSEMBLE_WEIGHTS", "0.5,0.5")

    result = train_voting_ensemble(["BLUE1"], ["PENNY1"])
    assert isinstance(result.model, PreFittedVotingEnsemble)
    experts = dict(result.model.estimators)
    assert set(experts) == {"bluechip_expert", "penny_expert"}

    # f=3.0 sits deep in the bluechip "positive" region (label = f > 0) but
    # deep in the penny "negative" region (label = f > 5). If either expert
    # had leaked the other universe's data into its own fit, this split
    # would blur.
    probe = pd.DataFrame({"f": [3.0]})
    assert experts["bluechip_expert"].predict(probe)[0] == 1
    assert experts["penny_expert"].predict(probe)[0] == 0

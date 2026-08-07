import numpy as np
import pandas as pd
import pytest

from portfolio.ml_engine.training.labeling import triple_barrier_events
from portfolio.ml_engine.training.sample_weights import (
    build_sample_weights_uniqueness,
    get_average_uniqueness,
    get_concurrency_per_bar,
    get_sample_weights,
    max_observed_horizon,
)


def test_concurrency_hand_worked_example() -> None:
    """bar_index = 0..5. Label A: [0,2], label B: [1,3] (overlaps A on bars
    1-2), label C: [4,5] (isolated). Expected concurrency per bar:
    [1, 2, 2, 1, 1, 1]."""
    bar_index = pd.RangeIndex(6)
    t1 = pd.Series({0: 2, 1: 3, 4: 5})

    concurrency = get_concurrency_per_bar(bar_index, t1)
    assert concurrency.tolist() == [1, 2, 2, 1, 1, 1]

    uniqueness = get_average_uniqueness(t1, concurrency)
    # A (bars 0-2): mean(1/1, 1/2, 1/2) = 2/3 ; B (bars 1-3): mean(1/2, 1/2, 1/1) = 2/3
    assert uniqueness.loc[0] == pytest.approx(2 / 3)
    assert uniqueness.loc[1] == pytest.approx(2 / 3)
    # C (bars 4-5): mean(1/1, 1/1) = 1.0 — isolated, no overlap
    assert uniqueness.loc[4] == pytest.approx(1.0)


def test_isolated_labels_get_full_weight_overlapping_get_less() -> None:
    bar_index = pd.RangeIndex(10)
    # Three labels all overlapping the same bars 0-4, one isolated at 8-9.
    t1 = pd.Series({0: 4, 1: 4, 2: 4, 8: 9})
    weights = get_sample_weights(t1, bar_index, normalize=False)
    assert weights.loc[8] > weights.loc[0]
    assert weights.loc[8] == pytest.approx(1.0)


def test_get_average_uniqueness_rejects_duplicate_index() -> None:
    t1 = pd.Series([2, 3], index=[0, 0])  # duplicate t0
    concurrency = pd.Series(1, index=pd.RangeIndex(5))
    with pytest.raises(ValueError):
        get_average_uniqueness(t1, concurrency)


def test_sample_weights_from_triple_barrier_events_are_positive_and_normalized() -> None:
    rng = np.random.RandomState(0)
    n = 200
    close = pd.Series(100 + np.cumsum(rng.randn(n) * 0.5))
    events = triple_barrier_events(close, up_pct=0.05, down_pct=0.03, max_days=10)

    weights = get_sample_weights(events["t1"], close.index, normalize=True)
    valid = weights.dropna()
    assert len(valid) > 0
    assert (valid > 0).all()
    assert valid.mean() == pytest.approx(1.0, abs=1e-6)


def test_build_sample_weights_uniqueness_sums_to_valid_count() -> None:
    rng = np.random.RandomState(1)
    n = 200
    close = pd.Series(100 + np.cumsum(rng.randn(n) * 0.5))
    events = triple_barrier_events(close, up_pct=0.05, down_pct=0.03, max_days=10)

    weights = build_sample_weights_uniqueness(close.index, events["t1"], use_return_attribution=False)
    valid = weights.dropna()
    assert len(valid) > 0
    assert (valid > 0).all()
    assert valid.sum() == pytest.approx(len(valid), abs=1e-6)


def test_build_sample_weights_uniqueness_with_return_attribution() -> None:
    """A unique label whose window saw a big move should end up weighted
    higher than an equally-unique label whose window was flat."""
    bar_index = pd.RangeIndex(20)
    # Two non-overlapping (equally unique) windows: [0,3] and [10,13].
    t1 = pd.Series({0: 3, 10: 13})
    # Flat returns everywhere except a big move inside the second window.
    returns = pd.Series(0.0, index=bar_index)
    returns.loc[11] = 0.5
    returns.loc[12] = -0.3

    weights = build_sample_weights_uniqueness(bar_index, t1, returns=returns, use_return_attribution=True)
    assert weights.loc[10] > weights.loc[0]


def test_max_observed_horizon() -> None:
    bar_index = pd.RangeIndex(10)
    t1 = pd.Series({0: 2, 3: 9, 5: float("nan")})
    assert max_observed_horizon(t1, bar_index) == 6  # label at t0=3 -> t1=9

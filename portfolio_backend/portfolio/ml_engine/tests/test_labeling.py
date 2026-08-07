import numpy as np
import pandas as pd
import pytest

from portfolio.ml_engine.training.labeling import (
    triple_barrier_labels,
    triple_barrier_labels_adaptive,
    universe_max_days,
)


def test_labels_are_binary() -> None:
    close = pd.Series(100 + np.random.randn(200).cumsum())
    labels = triple_barrier_labels(close, up_pct=0.10, down_pct=0.05, max_days=10)
    valid = labels.dropna()
    assert set(valid.unique()).issubset({0, 1})


def test_adaptive_labels_are_binary_not_ternary() -> None:
    """Regression test: pipeline_fixes.py's old triple_barrier_labels_adaptive
    returned ternary (-1/0/1) labels, inconsistent with every other training
    pathway's binary classifiers. The unified version must always be
    binary (0/1)."""
    close = pd.Series(100 + np.random.randn(300).cumsum())
    for universe in ("PENNY", "BLUECHIP", "CRYPTO"):
        labels = triple_barrier_labels_adaptive(close, universe=universe)
        valid = labels.dropna()
        assert set(valid.unique()).issubset({0, 1}), f"{universe} produced non-binary labels"


def test_adaptive_labels_use_distinct_per_universe_defaults() -> None:
    """Each universe must keep its own (previously duplicated-and-divergent)
    up_pct/down_pct/max_days rather than silently collapsing to one set."""
    assert universe_max_days("PENNY") == pytest.approx(5)
    assert universe_max_days("CRYPTO") == pytest.approx(7)
    assert universe_max_days("BLUECHIP") == pytest.approx(15)
    assert universe_max_days("STABLE") == universe_max_days("BLUECHIP")  # same universe, two names


def test_adaptive_labels_unknown_universe_falls_back_to_bluechip() -> None:
    assert universe_max_days("SOME_UNKNOWN_UNIVERSE") == universe_max_days("BLUECHIP")

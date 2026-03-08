from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class PurgedTimeSeriesSplit:
    n_splits: int = 5
    purge_window: int = 5
    embargo_pct: float = 0.01

    def split(self, X: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        if n_samples == 0 or self.n_splits <= 1:
            return
        embargo = int(max(0, np.floor(n_samples * self.embargo_pct)))
        fold_size = n_samples // (self.n_splits + 1)
        if fold_size <= 0:
            return
        for split_idx in range(1, self.n_splits + 1):
            test_start = split_idx * fold_size
            test_end = min(n_samples, test_start + fold_size)
            train_end = max(0, test_start - self.purge_window)
            train_idx = np.arange(0, train_end)
            embargo_end = min(n_samples, test_end + embargo)
            test_idx = np.arange(test_start, test_end)
            if embargo_end < n_samples:
                train_idx = np.concatenate([train_idx, np.arange(embargo_end, n_samples)])
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            yield train_idx, test_idx

    def get_n_splits(self, X: np.ndarray | None = None, y: np.ndarray | None = None, groups: np.ndarray | None = None) -> int:
        return self.n_splits

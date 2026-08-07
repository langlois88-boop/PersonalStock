from __future__ import annotations

"""Sample weighting for overlapping triple-barrier labels.

Triple-barrier labels are not i.i.d.: a label at t0 with resolution window
[t0, t1] shares the same underlying price moves with any other label whose
window overlaps it. Every trainer in this codebase defaults to uniform
sample weighting, which treats a run of heavily-overlapping labels as if
they were independent evidence — overweighting redundant samples relative
to isolated ones.

This module computes a per-sample weight inversely proportional to how many
other label windows were concurrently active over the same bars (Lopez de
Prado, "Advances in Financial Machine Learning", ch. 4: concurrency ->
average uniqueness -> sample weight). Feed the result into any of this
codebase's trainers via their existing `sample_weight` parameter
(`Trainer.fit(..., sample_weight=...)`, `pipeline.fit(..., model__sample_weight=...)`).

Usage (per symbol — concurrency is a single-instrument concept; for a
multi-symbol dataset, compute weights per symbol on that symbol's own price
index, then concatenate):

    events = triple_barrier_events_adaptive(close, high, low, universe="PENNY")
    weights = get_sample_weights(events["t1"], bar_index=close.index)
"""

import numpy as np
import pandas as pd


def get_concurrency_per_bar(bar_index: pd.Index, t1: pd.Series) -> pd.Series:
    """Number of label windows [t0_i, t1_i] active at each bar in `bar_index`.

    O(n_bars + n_labels) via a +1/-1 delta swept with cumsum, rather than
    building the O(n_bars * n_labels) dense indicator matrix explicitly.

    Args:
        bar_index: Full index of available bars (e.g. `close.index`) — must
            be the same index the `t1` values were computed against.
        t1: Series indexed by each label's t0, valued at its t1 (NaN/NaT
            entries — labels with no valid outcome — are ignored). This is
            exactly the ``t1`` column `triple_barrier_events`/`_adaptive`
            produces.

    Returns:
        Series of integer concurrency counts, indexed by `bar_index`.
    """
    concurrency = pd.Series(0, index=bar_index, dtype=int)
    valid = t1.dropna()
    if valid.empty:
        return concurrency

    pos = {ts: i for i, ts in enumerate(bar_index)}
    delta = np.zeros(len(bar_index) + 1, dtype=int)
    for t_start, t_end in valid.items():
        start_pos = pos.get(t_start)
        end_pos = pos.get(t_end)
        if start_pos is None or end_pos is None or end_pos < start_pos:
            continue
        delta[start_pos] += 1
        delta[end_pos + 1] -= 1  # in-bounds: delta has len(bar_index) + 1 slots

    concurrency[:] = np.cumsum(delta[:-1])
    return concurrency


def get_average_uniqueness(t1: pd.Series, concurrency: pd.Series) -> pd.Series:
    """For each label, the average (over its [t0, t1] window) of
    1/concurrency at each bar in that window.

    A label whose window rarely overlaps any other label's stays close to
    1.0 (unique evidence); one that's constantly co-active with many others
    is pulled toward 0 (redundant evidence).

    Args:
        t1: Same as in `get_concurrency_per_bar` — must have a unique index
            (one row per label; per-symbol only, see module docstring).
        concurrency: Output of `get_concurrency_per_bar`, over the same
            bar index `t1`'s values were computed against.

    Returns:
        Series of average uniqueness in (0, 1], indexed like `t1`, NaN
        where `t1` is NaN.
    """
    if not t1.index.is_unique:
        raise ValueError(
            "get_average_uniqueness: t1 must have a unique index (one row "
            "per label) — concurrency is a single-instrument concept; "
            "compute it per symbol, then concatenate the resulting weights."
        )
    inv_concurrency = (1.0 / concurrency.replace(0, np.nan)).fillna(0.0)
    pos = {ts: i for i, ts in enumerate(concurrency.index)}
    values = inv_concurrency.values

    out = pd.Series(np.nan, index=t1.index, dtype=float)
    for t_start, t_end in t1.items():
        if pd.isna(t_end):
            continue
        start_pos = pos.get(t_start)
        end_pos = pos.get(t_end)
        if start_pos is None or end_pos is None or end_pos < start_pos:
            continue
        out.loc[t_start] = float(np.mean(values[start_pos:end_pos + 1]))
    return out


def get_sample_weights(t1: pd.Series, bar_index: pd.Index, *, normalize: bool = True) -> pd.Series:
    """End-to-end: concurrency -> average uniqueness -> sample weights.

    Args:
        t1: Series indexed by each label's t0, valued at its t1 — the
            ``t1`` column of `triple_barrier_events`/`_adaptive`, for one
            symbol.
        bar_index: The full price index (e.g. `close.index`) the `t1`
            values were computed against — required, since concurrency has
            to be counted over every bar in each window, not just its
            endpoints.
        normalize: If True, rescale weights to mean 1.0, so they act as
            relative multipliers on top of any other sample weighting
            already applied (e.g. DeepSeek confidence) rather than shifting
            the overall scale of `sample_weight`.

    Returns:
        Series of weights indexed like `t1`, NaN where `t1` is NaN.
    """
    concurrency = get_concurrency_per_bar(bar_index, t1)
    uniqueness = get_average_uniqueness(t1, concurrency)
    if normalize:
        mean_u = uniqueness.mean(skipna=True)
        if mean_u and not np.isnan(mean_u) and mean_u > 0:
            uniqueness = uniqueness / mean_u
    return uniqueness


def build_sample_weights_uniqueness(
    close_index: pd.Index,
    t1: pd.Series,
    returns: pd.Series | None = None,
    *,
    use_return_attribution: bool = True,
) -> pd.Series:
    """Sample weights for one symbol, from its triple-barrier label overlap.

    Replaces uniform or LLM-derived sample weighting with Lopez de Prado's
    average-uniqueness weighting ("Advances in Financial Machine Learning",
    ch. 4), optionally combined with return attribution (ch. 4.5): a
    "unique" label whose window saw a big price move counts for more than
    an equally-unique label whose window was flat.

    Args:
        close_index: Full price index for this symbol — the same index
            `t1` was computed against (concurrency has to be counted over
            every bar in each window, not just its endpoints).
        t1: Series indexed by each label's t0, valued at its t1 — the
            ``t1`` column of `triple_barrier_events`/`_adaptive`, for this
            symbol only. Concurrency is a single-instrument concept: call
            this once per symbol, then concatenate the resulting weights in
            the same row order as that symbol's X/y.
        returns: Optional per-bar log returns over `close_index`, used for
            return attribution when `use_return_attribution=True`.
        use_return_attribution: If True and `returns` is given, weight each
            label by the absolute sum of returns over its window too.

    Returns:
        Series of weights indexed like `t1` (NaN where `t1` is NaN),
        normalized so the valid weights sum to their own count — matches
        the scale of uniform ``sample_weight=1.0`` per sample.
    """
    concurrency = get_concurrency_per_bar(close_index, t1)
    uniqueness = get_average_uniqueness(t1, concurrency)

    weights = uniqueness
    if use_return_attribution and returns is not None:
        pos = {ts: i for i, ts in enumerate(close_index)}
        returns_vals = returns.reindex(close_index).fillna(0.0).values
        magnitude = pd.Series(np.nan, index=t1.index, dtype=float)
        for t_start, t_end in t1.items():
            if pd.isna(t_end):
                continue
            start_pos = pos.get(t_start)
            end_pos = pos.get(t_end)
            if start_pos is None or end_pos is None or end_pos < start_pos:
                continue
            magnitude.loc[t_start] = abs(float(np.sum(returns_vals[start_pos:end_pos + 1])))
        weights = uniqueness * magnitude

    # Normalize so *valid* weights sum to their own count (not
    # `len(weights)`, which would also count NaN placeholders and skew the
    # scale down when some labels have no valid t1) — keeps the scale
    # comparable to uniform sample_weight=1.0 per sample.
    valid_count = weights.notna().sum()
    valid_sum = weights.sum(skipna=True)
    if valid_count and valid_sum:
        weights = weights * (valid_count / valid_sum)
    return weights


def max_observed_horizon(t1: pd.Series, bar_index: pd.Index) -> int:
    """Actual max label horizon (in bars) observed in `t1`, rather than the
    universe's configured max_days.

    Some labels resolve before timeout (barrier hit early), so the true
    worst case can only be read off the real data that was actually
    trained on. Use this as `Trainer`'s `purge_window` for internal
    coherence with the labels the model is actually trained on, instead of
    a fixed constant that may not match what was really computed for this
    run (e.g. under ATR-widened barriers, or a different env var value than
    when the dataset was built).
    """
    pos = {ts: i for i, ts in enumerate(bar_index)}
    max_h = 0
    for t_start, t_end in t1.items():
        if pd.isna(t_end):
            continue
        start_pos = pos.get(t_start)
        end_pos = pos.get(t_end)
        if start_pos is None or end_pos is None:
            continue
        max_h = max(max_h, end_pos - start_pos)
    return max_h

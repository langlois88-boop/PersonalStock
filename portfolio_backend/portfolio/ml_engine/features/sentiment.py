from __future__ import annotations

"""Sentiment aggregation helpers."""

import pandas as pd


def aggregate_sentiment(df_news: pd.DataFrame) -> pd.Series:
    """Aggregate sentiment by date for a symbol.

    Args:
        df_news: Dataframe with columns ['date', 'score'].

    Returns:
        Series indexed by date with mean sentiment.
    """
    if df_news is None or df_news.empty:
        return pd.Series(dtype=float)
    if 'date' not in df_news.columns or 'score' not in df_news.columns:
        return pd.Series(dtype=float)
    grouped = df_news.groupby('date')['score'].mean().fillna(0.0)
    return grouped.rename('sentiment_score')

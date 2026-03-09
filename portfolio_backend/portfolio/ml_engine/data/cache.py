from __future__ import annotations

"""Disk cache helpers for dataframes."""

from pathlib import Path
import pandas as pd


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write dataframe to parquet atomically.

    Args:
        df: Dataframe to store.
        path: Destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp_path)
    tmp_path.replace(path)

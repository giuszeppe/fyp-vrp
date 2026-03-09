"""Reference baseline helpers."""

from __future__ import annotations

import pandas as pd


def best_reference(df: pd.DataFrame, group_cols: list[str], target_col: str = "total_distance") -> pd.DataFrame:
    out = df.groupby(group_cols, as_index=False)[target_col].min()
    return out.rename(columns={target_col: "reference_distance"})

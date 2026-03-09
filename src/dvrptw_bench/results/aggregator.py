"""Aggregates benchmark records."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class Aggregator:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir

    def summarize(self, parquet_path: Path) -> pd.DataFrame:
        df = pd.read_parquet(parquet_path)
        grouped = (
            df.groupby(["strategy", "budget_s", "epsilon"], as_index=False)
            .agg(
                total_distance_mean=("total_distance", "mean"),
                feasible_rate=("feasible", "mean"),
                runtime_mean=("compute_times", "count"),
                n=("instance_id", "count"),
            )
            .sort_values(["strategy", "budget_s", "epsilon"])
        )
        out = self.run_dir / "summary.parquet"
        grouped.to_parquet(out, index=False)
        grouped.to_csv(self.run_dir / "summary.csv", index=False)
        return grouped

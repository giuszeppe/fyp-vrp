"""Coverage and completeness diagnostics for saved evaluation results."""

from __future__ import annotations

from itertools import product

import pandas as pd


def _expected_non_oracle_count(df: pd.DataFrame, evaluation_size: int) -> int:
    subset = df[df["evaluation_size"].eq(evaluation_size) & df["modality_group"].ne("oracle")]
    if subset.empty:
        return 0
    return (
        subset["instance_id"].dropna().nunique()
        * subset["seed"].dropna().nunique()
        * subset["degree_of_dynamicity"].dropna().nunique()
        * subset["cutoff_time"].dropna().nunique()
    )


def compute_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize result completeness by modality, model, and evaluation size."""

    if df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    group_cols = ["modality_group", "model_name", "evaluation_size"]
    grouped = df.groupby(group_cols, dropna=False, as_index=False)
    for _, group in grouped:
        modality = str(group["modality_group"].iloc[0])
        model_name = str(group["model_name"].iloc[0])
        evaluation_size = int(group["evaluation_size"].iloc[0])

        if modality == "oracle":
            expected = group["instance_id"].dropna().nunique()
        else:
            expected = _expected_non_oracle_count(df, evaluation_size)

        observed = len(group)
        rows.append(
            {
                "modality_group": modality,
                "model_name": model_name,
                "evaluation_size": evaluation_size,
                "observed_runs": observed,
                "expected_runs": expected,
                "missing_runs": max(expected - observed, 0),
                "coverage_ratio": (observed / expected) if expected else 0.0,
                "instance_count": group["instance_id"].dropna().nunique(),
                "seed_count": group["seed"].dropna().nunique(),
                "dod_count": group["degree_of_dynamicity"].dropna().nunique(),
                "cutoff_count": group["cutoff_time"].dropna().nunique(),
                "oracle_gap_coverage_ratio": group["has_oracle_gap"].mean(),
                "feasible_rate": group["feasible"].mean(),
            }
        )

    return pd.DataFrame(rows).sort_values(group_cols, kind="stable").reset_index(drop=True)


def missing_oracle_gap_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize where oracle-gap values are unavailable."""

    if df.empty:
        return pd.DataFrame()

    subset = df[df["modality_group"].ne("oracle")]
    grouped = subset.groupby(["model_name", "evaluation_size"], as_index=False).agg(
        runs=("model_name", "count"),
        missing_oracle_gap=("has_oracle_gap", lambda values: int((~values).sum())),
        available_oracle_gap=("has_oracle_gap", "sum"),
    )
    grouped["availability_ratio"] = grouped["available_oracle_gap"] / grouped["runs"]
    return grouped.sort_values(["model_name", "evaluation_size"], kind="stable").reset_index(drop=True)


def expected_non_oracle_grid(df: pd.DataFrame, evaluation_size: int) -> pd.DataFrame:
    """Materialize the expected instance/seed/DoD/cutoff grid for one evaluation size."""

    subset = df[df["evaluation_size"].eq(evaluation_size) & df["modality_group"].ne("oracle")]
    if subset.empty:
        return pd.DataFrame(columns=["instance_id", "seed", "degree_of_dynamicity", "cutoff_time"])

    universe = list(
        product(
            sorted(subset["instance_id"].dropna().unique()),
            sorted(subset["seed"].dropna().unique()),
            sorted(subset["degree_of_dynamicity"].dropna().unique()),
            sorted(subset["cutoff_time"].dropna().unique()),
        )
    )
    return pd.DataFrame(
        universe,
        columns=["instance_id", "seed", "degree_of_dynamicity", "cutoff_time"],
    )

"""Grouped summaries used by the evaluation notebooks."""

from __future__ import annotations

import pandas as pd


DEFAULT_METRICS = [
    "total_distance",
    "num_vehicles",
    "average_lateness",
    "customers_rejected",
    "rejection_rate",
    "service_level",
    "computational_time",
    "total_cost",
    "oracle_gap",
]


def summarize_results(df: pd.DataFrame, group_by: list[str], metrics: list[str] | None = None) -> pd.DataFrame:
    """Aggregate key metrics across arbitrary grouping columns."""

    if df.empty:
        return pd.DataFrame()

    metrics = metrics or DEFAULT_METRICS
    agg_spec: dict[str, tuple[str, str]] = {
        "runs": ("model_name", "count"),
        "feasible_rate": ("feasible", "mean"),
    }
    for metric in metrics:
        if metric in df.columns:
            agg_spec[f"{metric}_mean"] = (metric, "mean")
            agg_spec[f"{metric}_median"] = (metric, "median")

    summary = df.groupby(group_by, dropna=False, as_index=False).agg(**agg_spec)
    if "total_cost_mean" in summary.columns:
        summary = summary.sort_values([*group_by, "total_cost_mean"], kind="stable")
    return summary.reset_index(drop=True)


def family_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate by model and Solomon family."""

    return summarize_results(
        df,
        group_by=["model_display_name", "instance_family", "evaluation_size"],
        metrics=DEFAULT_METRICS,
    )


def oracle_comparison_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate only rows that have a valid oracle comparison."""

    subset = df[df["has_oracle_gap"]].copy()
    return summarize_results(
        subset,
        group_by=["model_display_name", "evaluation_size"],
        metrics=["oracle_gap", "total_cost", "total_distance", "service_level"],
    )


def ai_ablation_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate RouteFinder variants for the ablation section."""

    subset = df[df["modality_group"].eq("ai")].copy()
    return summarize_results(
        subset,
        group_by=["model_display_name", "model_variant", "evaluation_size"],
        metrics=["total_cost", "total_distance", "average_lateness", "service_level", "oracle_gap"],
    )

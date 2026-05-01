"""Matplotlib helpers shared by the thesis and exploratory notebooks."""

from __future__ import annotations

import math
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from .style import METRIC_LABELS, MODALITY_COLORS


def _label_for_metric(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def plot_metric_bars(
    summary_df: pd.DataFrame,
    *,
    metric: str,
    label_col: str = "model_display_name",
    color_col: str = "modality_group",
    ax=None,
):
    """Bar chart for grouped summaries with `<metric>_mean` columns."""

    value_col = metric if metric in summary_df.columns else f"{metric}_mean"
    if value_col not in summary_df.columns:
        raise KeyError(f"{value_col} not found in summary DataFrame")

    plot_df = summary_df.sort_values(value_col, kind="stable")
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    colors = [MODALITY_COLORS.get(value, "#4c566a") for value in plot_df.get(color_col, [])]
    ax.bar(plot_df[label_col], plot_df[value_col], color=colors if colors else "#4c566a")
    ax.set_ylabel(_label_for_metric(metric))
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.2)
    return ax


def plot_metric_by_dynamicity(
    df: pd.DataFrame,
    *,
    metric: str,
    hue: str = "model_display_name",
    ax=None,
):
    """Line chart across degree of dynamicity."""

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    grouped = (
        df.groupby(["degree_of_dynamicity", hue], as_index=False)[metric]
        .mean()
        .sort_values(["degree_of_dynamicity", hue], kind="stable")
    )
    for label, part in grouped.groupby(hue):
        ax.plot(part["degree_of_dynamicity"], part[metric], marker="o", label=label)
    ax.set_xlabel("Degree of Dynamism")
    ax.set_ylabel(_label_for_metric(metric))
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    return ax


def plot_tradeoff_scatter(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    label_col: str = "model_display_name",
    color_col: str = "modality_group",
    ax=None,
):
    """Mean trade-off scatter plot for two metrics."""

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    grouped = df.groupby([label_col, color_col], as_index=False)[[x, y]].mean()
    for _, row in grouped.iterrows():
        ax.scatter(row[x], row[y], s=80, color=MODALITY_COLORS.get(row[color_col], "#4c566a"))
        ax.text(row[x], row[y], row[label_col], fontsize=8, ha="left", va="bottom")
    ax.set_xlabel(_label_for_metric(x))
    ax.set_ylabel(_label_for_metric(y))
    ax.grid(alpha=0.2)
    return ax


def plot_family_heatmaps(
    summary_df: pd.DataFrame,
    *,
    metric: str,
    family_col: str = "instance_family",
    size_col: str = "evaluation_size",
    value_suffix: str = "_mean",
):
    """Return a figure with one heatmap per Solomon family."""

    value_col = metric if metric in summary_df.columns else f"{metric}{value_suffix}"
    if value_col not in summary_df.columns:
        raise KeyError(f"{value_col} not found in summary DataFrame")

    families = [family for family in summary_df[family_col].dropna().unique()]
    ncols = min(3, max(1, len(families)))
    nrows = int(math.ceil(len(families) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)

    for ax, family in zip(axes.flat, families):
        part = summary_df[summary_df[family_col].eq(family)]
        piv = part.pivot_table(index="model_display_name", columns=size_col, values=value_col, aggfunc="mean")
        im = ax.imshow(piv.fillna(0).values, aspect="auto")
        ax.set_title(str(family))
        ax.set_xticks(range(len(piv.columns)), [str(value) for value in piv.columns])
        ax.set_yticks(range(len(piv.index)), [str(value) for value in piv.index])
        fig.colorbar(im, ax=ax, shrink=0.85)

    for ax in axes.flat[len(families):]:
        ax.axis("off")

    fig.tight_layout()
    return fig, axes

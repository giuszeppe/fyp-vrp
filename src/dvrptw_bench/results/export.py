"""Export helpers for reports and tabular outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_report(run_dir: Path, summary_df: pd.DataFrame) -> Path:
    lines = ["# DVRPTW Benchmark Report", "", "## Summary", "", summary_df.to_markdown(index=False)]
    out = run_dir / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out

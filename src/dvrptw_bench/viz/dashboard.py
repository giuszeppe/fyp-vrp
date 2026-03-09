"""Result dashboard plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def render_dashboard(run_dir: Path) -> list[Path]:
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        return []

    df = pd.read_csv(summary_path)
    outs: list[Path] = []

    if not df.empty:
        piv = df.pivot_table(index="epsilon", columns="budget_s", values="total_distance_mean", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(piv.fillna(0).values, aspect="auto")
        ax.set_title("Gap proxy heatmap (distance mean)")
        ax.set_xticks(range(len(piv.columns)), [str(c) for c in piv.columns])
        ax.set_yticks(range(len(piv.index)), [str(i) for i in piv.index])
        fig.colorbar(im, ax=ax)
        out = run_dir / "heatmap_distance.png"
        fig.tight_layout()
        fig.savefig(out, dpi=180)
        plt.close(fig)
        outs.append(out)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        for strategy, part in df.groupby("strategy"):
            ax2.plot(part["epsilon"], part["feasible_rate"], marker="o", label=strategy)
        ax2.set_title("Feasibility trends")
        ax2.set_xlabel("epsilon")
        ax2.set_ylabel("feasible_rate")
        ax2.grid(alpha=0.2)
        ax2.legend(fontsize=8)
        out2 = run_dir / "feasibility_trends.png"
        fig2.tight_layout()
        fig2.savefig(out2, dpi=180)
        plt.close(fig2)
        outs.append(out2)

    return outs

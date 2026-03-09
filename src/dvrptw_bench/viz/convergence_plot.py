"""Convergence plots for iterative methods."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_convergence(points: list[tuple[float, float]], out_path: Path = None, title: str = "Convergence") -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    if points:
        xs, ys = zip(*points, strict=False)
        ax.plot(xs, ys, color="tab:green", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("best objective")
    ax.grid(alpha=0.2)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
    else:
        plt.show()
    plt.close(fig)
    return out_path

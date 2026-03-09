"""Route visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from dvrptw_bench.common.typing import Solution, VRPTWInstance


def plot_routes(instance: VRPTWInstance, solution: Solution, out_path: Path = None) -> Path:
    nodes = {n.id: n for n in instance.all_nodes}
    colors = plt.cm.tab20.colors
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter([instance.depot.x], [instance.depot.y], c="red", marker="s", s=80, label="Depot")
    ax.scatter([c.x for c in instance.customers], [c.y for c in instance.customers], c="gray", s=20, alpha=0.8)

    for i, route in enumerate(solution.routes):
        path = [instance.depot.id, *route.node_ids, instance.depot.id]
        xs = [nodes[n].x for n in path if n in nodes]
        ys = [nodes[n].y for n in path if n in nodes]
        ax.plot(xs, ys, color=colors[i % len(colors)], linewidth=1.8, label=f"V{route.vehicle_id}")

    ax.set_title(f"{instance.instance_id} - {solution.strategy}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180)
        fig.savefig(out_path.with_suffix(".svg"))
        plt.close(fig)
        plt.show()
        return None
    return out_path

"""Timeline visualization for time-window service."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from dvrptw_bench.common.typing import Solution, VRPTWInstance


def plot_timeline(instance: VRPTWInstance, solution: Solution, out_path: Path = None) -> Path:
    nodes = {n.id: n for n in instance.all_nodes}
    fig, ax = plt.subplots(figsize=(10, 5))
    y = 0
    for route in solution.routes:
        t = instance.depot.ready_time
        prev = instance.depot.id
        for nid in route.node_ids:
            if nid not in nodes:
                continue
            c = nodes[nid]
            t += instance.distance_matrix[prev][nid] if prev < len(instance.distance_matrix) and nid < len(instance.distance_matrix) else 0.0
            start = max(t, c.ready_time)
            end = start + c.service_time
            ax.hlines(y, c.ready_time, c.due_time, color="lightgray", linewidth=6)
            ax.hlines(y, start, end, color="tab:blue", linewidth=4)
            y += 1
            t = end
            prev = nid

    ax.set_title(f"Time windows timeline: {instance.instance_id}")
    ax.set_xlabel("time")
    ax.set_ylabel("service order")
    ax.grid(alpha=0.2)
    if out_path is None:
        plt.show()
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path

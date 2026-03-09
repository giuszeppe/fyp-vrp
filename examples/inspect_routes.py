"""Example: inspect static and dynamic route execution."""

from __future__ import annotations

from pathlib import Path

from dvrptw_bench.data.instance_filters import find_rc_instances
from dvrptw_bench.data.solomon_parser import parse_solomon
from dvrptw_bench.dynamic.simulator import DynamicSimulator
from dvrptw_bench.heuristics.gls import GLSSolver
from dvrptw_bench.viz.inspector import inspect_dynamic, inspect_static


def main() -> None:
    dataset_root = Path("./dataset")
    files = find_rc_instances(dataset_root)
    if not files:
        raise RuntimeError("No RC instances found under ./dataset")

    instance = parse_solomon(files[0])

    static_solver = GLSSolver(seed=1)
    static_solution = static_solver.solve(instance, time_limit_s=5)
    inspect_static(instance, static_solution, title=f"{instance.instance_id} static")

    simulator = DynamicSimulator(instance)
    inspect_dynamic(
        instance,
        simulator,
        lambda snap_inst, budget: GLSSolver(seed=1).solve(snap_inst, budget),
        epsilon=0.5,
        budget_s=5.0,
        seed=1,
        title=f"{instance.instance_id} dynamic",
    )


if __name__ == "__main__":
    main()

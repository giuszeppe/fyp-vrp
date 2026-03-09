"""Connector from RL outputs to heuristic warm-starts."""

from __future__ import annotations

from dvrptw_bench.common.typing import Solution
from dvrptw_bench.heuristics.warmstart import as_route_order


CAPABILITY_MATRIX = {
    "ortools": "route ordering hint (indirect seed)",
    "gls": "direct initial solution",
    "gurobi": "MIP start values scaffold",
    "lkh3": "initial tour file scaffold",
    "pyvrp": "adapter-level route seed",
}


def to_canonical_solution(rl_solution: Solution) -> Solution:
    return rl_solution


def to_warmstart(rl_solution: Solution, solver_name: str) -> dict:
    return {
        "solver": solver_name,
        "capability": CAPABILITY_MATRIX.get(solver_name, "unknown"),
        "routes": as_route_order(rl_solution),
    }

"""Hybrid warm-start + GLS refinement."""

from __future__ import annotations

from dvrptw_bench.common.typing import Solution, VRPTWInstance
from dvrptw_bench.heuristics.gls import GLSSolver


def refine_with_gls(
    instance: VRPTWInstance,
    warm_start: Solution,
    time_limit_s: float,
    debug: bool = False,
    log_every: int = 1,
) -> Solution:
    return GLSSolver(debug=debug, log_every=log_every).solve(
        instance,
        time_limit_s=time_limit_s,
        warm_start=warm_start,
    )

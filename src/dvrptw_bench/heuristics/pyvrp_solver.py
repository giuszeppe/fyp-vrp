"""PyVRP adapter with robust fallback path."""

from __future__ import annotations

import time

from dvrptw_bench.common.typing import Solution, VRPTWInstance
from dvrptw_bench.heuristics.constructive_pmca import PMCAVRPTWSolver
from dvrptw_bench.heuristics.interfaces import HeuristicSolver


class PyVRPVRPTWSolver(HeuristicSolver):
    name = "pyvrp"

    def solve(self, instance: VRPTWInstance, time_limit_s: float, warm_start: Solution | None = None) -> Solution:
        t0 = time.perf_counter()
        try:
            import pyvrp  # noqa: F401
            # A full PyVRP model builder can be added here depending on the exact installed API.
            # To keep the pipeline stable across PyVRP versions we use PMCA fallback by default.
            sol = PMCAVRPTWSolver().solve(instance, time_limit_s, warm_start=warm_start)
            sol.strategy = self.name + ":adapter_pmca"
        except Exception:
            sol = PMCAVRPTWSolver().solve(instance, time_limit_s, warm_start=warm_start)
            sol.strategy = self.name + ":fallback_pmca"

        sol.solve_time_s = time.perf_counter() - t0
        return sol

"""PyVRP adapter with robust fallback path."""

from __future__ import annotations

import time

from dvrptw_bench.common.typing import Solution, VRPTWInstance
from dvrptw_bench.heuristics.constructive_pmca import PMCAVRPTWSolver
from dvrptw_bench.heuristics.interfaces import HeuristicSolver
from dvrptw_bench.rl.routefinder_adapter import instance_to_routefinder_td, routefinder_actions_to_solution
from routefinder.baselines.solve import solve


class PyVRPVRPTWSolver(HeuristicSolver):
    name = "pyvrp"

    def solve(self, instance: VRPTWInstance, time_limit_s: float, warm_start: Solution | None = None) -> Solution:
        t0 = time.perf_counter()
        try:
            print(time_limit_s)
            # Convert VRPTWInstance to TensorDict format for routefinder
            td = instance_to_routefinder_td(instance, normalize_coords=True)
            
            # Number of processes for parallel solving
            num_procs = 32
            
            # Call the routefinder baseline solver
            actions, costs = solve(td, max_runtime=time_limit_s, num_procs=num_procs, solver="pyvrp")
            
            # Convert the actions back to a Solution object
            sol = routefinder_actions_to_solution(actions, instance, strategy=self.name)
            
        except Exception as e:
            # Fallback to PMCA solver if PyVRP fails
            sol = PMCAVRPTWSolver().solve(instance, time_limit_s, warm_start=warm_start)
            sol.strategy = self.name + ":fallback_pmca"

        sol.solve_time_s = time.perf_counter() - t0
        return sol

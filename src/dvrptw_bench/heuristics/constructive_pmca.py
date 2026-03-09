"""Constraint-aware insertion baseline (PMCA)."""

from __future__ import annotations

import time

from dvrptw_bench.common.typing import Route, Solution, VRPTWInstance
from dvrptw_bench.dynamic.feasibility import verify_solution
from dvrptw_bench.heuristics.interfaces import HeuristicSolver
from dvrptw_bench.metrics.objective import total_distance


class PMCAVRPTWSolver(HeuristicSolver):
    name = "pmca"

    def solve(self, instance: VRPTWInstance, time_limit_s: float, warm_start: Solution | None = None) -> Solution:
        t0 = time.perf_counter()
        customers = sorted(instance.customers, key=lambda c: (c.due_time, c.ready_time))
        routes = [Route(vehicle_id=i, node_ids=[]) for i in range(instance.vehicle_count)]
        node_by_id = {n.id: n for n in instance.customers}

        for c in customers:
            best_route: Route | None = None
            best_distance = float("inf")
            for r in routes:
                trial = Solution(strategy=self.name, routes=[rt.model_copy(deep=True) for rt in routes])
                trial.routes[r.vehicle_id].node_ids.append(c.id)
                rep = verify_solution(instance, trial)
                if rep.capacity_violation <= 1e-9 and rep.time_violation <= 1e-9 and c.id not in rep.unserved_customers:
                    trial_distance = total_distance(instance, trial)
                    if trial_distance < best_distance:
                        best_distance = trial_distance
                        best_route = r
            if best_route is not None:
                best_route.node_ids.append(c.id)
                continue

            # Ensure complete assignment even when no fully-feasible insertion is found.
            # This gives GLS a complete solution to repair/refine instead of a permanently
            # infeasible partial plan with unserved customers.
            least_load_route = min(
                routes,
                key=lambda r: sum(node_by_id[nid].demand for nid in r.node_ids if nid in node_by_id),
            )
            least_load_route.node_ids.append(c.id)

        sol = Solution(strategy=self.name, routes=routes)
        rep = verify_solution(instance, sol)
        sol.feasible = rep.feasible
        sol.violations = {"capacity": rep.capacity_violation, "time": rep.time_violation, "unserved": float(len(rep.unserved_customers))}
        sol.total_distance = total_distance(instance, sol)
        sol.solve_time_s = time.perf_counter() - t0
        return sol

"""Feasibility verifier and repair/fallback modes."""

from __future__ import annotations

from dvrptw_bench.common.errors import InfeasibleSolutionError
from dvrptw_bench.common.typing import FeasibilityReport, Solution, VRPTWInstance
from dvrptw_bench.dynamic.feasibility import verify_solution
from dvrptw_bench.heuristics.constructive_pmca import PMCAVRPTWSolver


class FeasibilityLayer:
    def __init__(self, mode: str = "repair", fallback_heuristic: str = "pmca"):
        self.mode = mode
        self.fallback_heuristic = fallback_heuristic

    def verify(self, solution: Solution, instance: VRPTWInstance) -> FeasibilityReport:
        return verify_solution(instance, solution)

    def enforce(self, solution: Solution, instance: VRPTWInstance) -> Solution:
        rep = self.verify(solution, instance)
        if rep.feasible:
            return solution

        if self.mode == "strict":
            raise InfeasibleSolutionError(f"Infeasible proposal: {rep}")

        if self.mode == "repair":
            repaired = self._simple_repair(solution, instance)
            rep2 = self.verify(repaired, instance)
            if rep2.feasible:
                return repaired

        return PMCAVRPTWSolver().solve(instance, time_limit_s=2.0)

    def _simple_repair(self, solution: Solution, instance: VRPTWInstance) -> Solution:
        all_ids = {c.id for c in instance.customers}
        seen: set[int] = set()
        for r in solution.routes:
            dedup = []
            for nid in r.node_ids:
                if nid in all_ids and nid not in seen:
                    dedup.append(nid)
                    seen.add(nid)
            r.node_ids = dedup

        missing = sorted(all_ids - seen)
        for m in missing:
            if solution.routes:
                solution.routes[0].node_ids.append(m)
        return solution

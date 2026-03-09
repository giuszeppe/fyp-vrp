"""Guided Local Search (GLS) for VRPTW.

This module implements a lightweight GLS metaheuristic over a route-based
solution representation. The solver supports:
- warm starts,
- strict wall-clock budgeting,
- feasibility-aware search from infeasible seeds,
- per-edge penalties for diversification,
- optional iteration-level debug logging.
"""

from __future__ import annotations

import logging
import random
import time
from collections import defaultdict

from dvrptw_bench.common.time_budget import BudgetTimer
from dvrptw_bench.common.typing import Solution, VRPTWInstance
from dvrptw_bench.dynamic.feasibility import verify_solution
from dvrptw_bench.heuristics.constructive_pmca import PMCAVRPTWSolver
from dvrptw_bench.heuristics.interfaces import HeuristicSolver
from dvrptw_bench.heuristics.local_search_ops import cross_exchange, relocate, swap, two_opt
from dvrptw_bench.metrics.objective import total_distance

logger = logging.getLogger(__name__)


class GLSSolver(HeuristicSolver):
    """Guided Local Search solver for VRPTW.

    The optimization target is an augmented objective:

    `distance + violation_weight * infeasibility + lambda_penalty * edge_penalties`

    where infeasibility is the sum of capacity violation, time-window violation,
    and the number of unserved customers.
    """

    name = "gls"

    def __init__(
        self,
        lambda_penalty: float = 0.1,
        violation_weight: float = 1000.0,
        seed: int | None = None,
        debug: bool = False,
        log_every: int = 1,
    ):
        """Initialize GLS hyperparameters.

        Args:
            lambda_penalty: Weight for accumulated edge penalties.
            violation_weight: Weight applied to infeasibility magnitude.
            seed: RNG seed for deterministic move sampling.
            debug: Enable debug logs for progress diagnostics.
            log_every: Emit one debug line every N iterations when debug is on.
        """
        self.lambda_penalty = lambda_penalty
        self.violation_weight = violation_weight
        self._rng = random.Random(seed)
        self.debug = debug
        self.log_every = max(1, log_every)
        if self.debug:
            logger.setLevel(logging.DEBUG)

    def _edges(self, solution: Solution, depot_id: int) -> list[tuple[int, int]]:
        """Return all directed edges in a solution including depot legs."""
        edges: list[tuple[int, int]] = []
        for r in solution.routes:
            path = [depot_id, *r.node_ids, depot_id]
            edges.extend((a, b) for a, b in zip(path[:-1], path[1:], strict=False))
        return edges

    def _violation_score(self, instance: VRPTWInstance, sol: Solution) -> tuple[float, bool]:
        """Compute scalar infeasibility and feasibility boolean for a solution."""
        rep = verify_solution(instance, sol)
        v = rep.capacity_violation + rep.time_violation + float(len(rep.unserved_customers))
        return v, rep.feasible

    def _augmented_score(
        self,
        instance: VRPTWInstance,
        sol: Solution,
        penalties: dict[tuple[int, int], int],
    ) -> tuple[float, float, bool]:
        """Compute augmented GLS score, pure distance, and feasibility flag."""
        dist = total_distance(instance, sol)
        viol, feasible = self._violation_score(instance, sol)
        p = sum(penalties[e] for e in self._edges(sol, instance.depot.id))
        score = dist + self.violation_weight * viol + self.lambda_penalty * p
        return score, dist, feasible

    @staticmethod
    def _edge_cost(instance: VRPTWInstance, i: int, j: int) -> float:
        """Return edge travel cost with safe fallback for sparse/snapshot matrices."""
        if i < len(instance.distance_matrix) and j < len(instance.distance_matrix):
            return instance.distance_matrix[i][j]
        return 1.0

    def solve(self, instance: VRPTWInstance, time_limit_s: float, warm_start: Solution | None = None) -> Solution:
        """Run GLS within a fixed wall-clock budget.

        Args:
            instance: VRPTW instance to optimize.
            time_limit_s: Maximum runtime in seconds.
            warm_start: Optional initial solution. If omitted, PMCA is used.

        Returns:
            The best feasible solution found; if no feasible solution is found,
            returns the best solution by augmented score.
        """
        timer = BudgetTimer(time_limit_s)
        start = time.perf_counter()

        current = (
            warm_start.model_copy(deep=True)
            if warm_start is not None
            else PMCAVRPTWSolver().solve(instance, min(1.0, time_limit_s))
        )
        best_any = current.model_copy(deep=True)
        best_feasible: Solution | None = (
            current.model_copy(deep=True) if verify_solution(instance, current).feasible else None
        )
        best_feasible_dist = total_distance(instance, best_feasible) if best_feasible is not None else float("inf")

        penalties: dict[tuple[int, int], int] = defaultdict(int)
        convergence: list[tuple[float, float]] = []
        ops = [relocate, swap, two_opt, cross_exchange]
        op_idx = 0
        iteration = 0

        current_score, _, _ = self._augmented_score(instance, current, penalties)
        best_any_score = current_score
        if self.debug:
            logger.debug(
                "GLS start: dist=%.3f feasible=%s score=%.3f",
                total_distance(instance, current),
                verify_solution(instance, current).feasible,
                current_score,
            )

        while not timer.expired:
            iteration += 1
            op = ops[op_idx % len(ops)]
            op_idx += 1
            cand = op(current, self._rng)

            cand_score, cand_dist, cand_feasible = self._augmented_score(instance, cand, penalties)
            if cand_score <= current_score:
                current = cand
                current_score = cand_score
            else:
                # Penalize one high-utility edge to diversify search.
                utilities: list[tuple[float, tuple[int, int]]] = []
                for e in self._edges(cand, instance.depot.id):
                    i, j = e
                    c = self._edge_cost(instance, i, j)
                    utilities.append((c / (1 + penalties[e]), e))
                if utilities:
                    _, e_star = max(utilities, key=lambda x: x[0])
                    penalties[e_star] += 1

            if cand_score < best_any_score:
                best_any = cand.model_copy(deep=True)
                best_any_score = cand_score

            if cand_feasible and cand_dist < best_feasible_dist:
                best_feasible = cand.model_copy(deep=True)
                best_feasible_dist = cand_dist

            if self.debug and iteration % self.log_every == 0:
                rep = verify_solution(instance, cand)
                logger.debug(
                    "iter=%d op=%s cand_dist=%.3f cand_score=%.3f feasible=%s cap_v=%.3f tw_v=%.3f unserved=%d",
                    iteration,
                    op.__name__,
                    cand_dist,
                    cand_score,
                    cand_feasible,
                    rep.capacity_violation,
                    rep.time_violation,
                    len(rep.unserved_customers),
                )

            tracked = best_feasible if best_feasible is not None else best_any
            convergence.append((timer.elapsed_s, total_distance(instance, tracked)))

        best = best_feasible if best_feasible is not None else best_any
        rep = verify_solution(instance, best)
        best.total_distance = total_distance(instance, best)
        best.feasible = rep.feasible
        best.violations = {
            "capacity": rep.capacity_violation,
            "time": rep.time_violation,
            "unserved": float(len(rep.unserved_customers)),
        }
        best.solve_time_s = time.perf_counter() - start
        best.details["convergence"] = convergence
        if self.debug:
            logger.debug(
                "GLS end: iters=%d best_dist=%.3f feasible=%s",
                iteration,
                best.total_distance,
                best.feasible,
            )
        return best

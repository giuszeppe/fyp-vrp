"""Hybrid runner: RL inference -> feasibility layer -> GLS refinement."""

from __future__ import annotations

import time

from dvrptw_bench.common.typing import Solution, VRPTWInstance
from dvrptw_bench.heuristics.ortools_solver import ORToolsVRPTWSolver
from dvrptw_bench.hybrid.feasibility_layer import FeasibilityLayer
from dvrptw_bench.hybrid.warmstart_then_gls import refine_with_gls
from dvrptw_bench.rl.policies import build_policy


def run_hybrid(
    instance: VRPTWInstance,
    policy: object,
    policy_name: str,
    budget_s: float,
    gls_time_share: float = 0.9,
    feasibility_mode: str = "repair",
    gls_debug: bool = False,
    gls_log_every: int = 1,
) -> tuple[Solution, dict[str, float]]:

    from dvrptw_bench.dynamic.snapshot import SnapshotState

    snapshot = SnapshotState(
        time=0.0,
        remaining_customers=instance.customers,
        active_customer_ids={c.id for c in instance.customers},
        served_customer_ids=set(),
        vehicles=[],
    )

    t0 = time.perf_counter()
    proposal = policy.infer_solution(snapshot)
    t1 = time.perf_counter()

    layer = FeasibilityLayer(mode=feasibility_mode)
    feasible_sol = layer.enforce(proposal, instance)

    remaining = max(0.0, budget_s - (t1 - t0))
    gls_budget = remaining * gls_time_share
    ortSolver = ORToolsVRPTWSolver()
    print(f"Refining with ORTools for up to {gls_budget:.1f}s...")
    refined =  ortSolver.solve(instance, gls_budget, warm_start=feasible_sol)
    # refined = refine_with_gls(
    #     instance,
    #     feasible_sol,
    #     gls_budget,
    #     debug=gls_debug,
    #     log_every=gls_log_every,
    # )

    timings = {"inference_s": t1 - t0, "local_search_s": gls_budget, "total_s": time.perf_counter() - t0}
    return refined, timings

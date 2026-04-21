"""Hybrid runner: RL inference -> feasibility layer -> GLS refinement."""

from __future__ import annotations

import time
from typing import Any, Callable

from dvrptw_bench.common.typing import Solution, VRPTWInstance
from dvrptw_bench.dynamic.dynamic_instance import DynamicInstance
from dvrptw_bench.dynamic.snapshot import SnapshotState
from dvrptw_bench.heuristics.ortools_dynamic import ORToolsDVRPTWSolver
from dvrptw_bench.heuristics.ortools_solver import ORToolsVRPTWSolver
from dvrptw_bench.hybrid.feasibility_layer import FeasibilityLayer
from dvrptw_bench.hybrid.warmstart_then_gls import refine_with_gls
from dvrptw_bench.rl.policies import build_policy


def _invoke_solver_fn(
    solver_fn: Callable,
    instance: DynamicInstance,
    time_limit_s: float,
    warm_start: Solution | None = None,
):
    """Normalize solver function signatures to a common call pattern."""
    try:
        return solver_fn(instance=instance, time_limit_s=time_limit_s, warm_start=warm_start)
    except TypeError:
        try:
            return solver_fn(instance, time_limit_s, warm_start)
        except TypeError:
            try:
                return solver_fn(instance, time_limit_s)
            except TypeError:
                return solver_fn(instance)


def run_hybrid_with_solver_fn(
    instance: DynamicInstance,
    solver_fn: Callable,
    budget_s: float,
    gls_time_share: float = 0.9,
    feasibility_mode: str = "repair",
    gls_debug: bool = False,
    gls_log_every: int = 1,
) -> tuple[Solution, dict[str, float]]:
    """Hybrid pipeline using a solver function for the proposal stage."""
    _ = (gls_debug, gls_log_every)

    t0 = time.perf_counter()
    proposal = _invoke_solver_fn(solver_fn, instance, budget_s, warm_start=None)
    t1 = time.perf_counter()

    layer = FeasibilityLayer(mode=feasibility_mode)
    feasible_sol = layer.enforce(proposal, instance)

    remaining = max(0.0, budget_s - (t1 - t0))
    gls_budget = remaining * gls_time_share
    ort_solver = ORToolsDVRPTWSolver()


    print(f"Refining with ORTools for up to {gls_budget:.1f}s...")
    refined = ort_solver.solve(instance, gls_budget, warm_start=feasible_sol)
    if refined is None:
        refined = feasible_sol

    timings = {
        "inference_s": t1 - t0,
        "local_search_s": gls_budget,
        "total_s": time.perf_counter() - t0,
    }
    return refined, timings


def run_hybrid(
    instance: DynamicInstance,
    *,
    budget_s: float,
    policy: object | None = None,
    policy_name: str | None = None,
    gls_time_share: float = 0.9,
    feasibility_mode: str = "repair",
    gls_debug: bool = False,
    gls_log_every: int = 1,
) -> tuple[Solution, dict[str, float]]:
    """Backward-compatible policy-based hybrid runner."""

    from dvrptw_bench.dynamic.snapshot import SnapshotState

    if policy is None:
        if policy_name is None:
            raise ValueError("policy_name is required when policy is not provided")
        policy = build_policy(policy_name)
    policy_obj: Any = policy

    def policy_solver_fn(instance, time_limit_s, warm_start=None):
        _ = (time_limit_s, warm_start)
        snapshot = SnapshotState(
            time=0,
            remaining_customers=instance.customers,
            active_customer_ids={c.id for c in instance.customers},
            served_customer_ids=set(),
            vehicles=[],
        )

        if hasattr(policy_obj, "infer_instance"):
            return policy_obj.infer_instance(instance)
        if hasattr(policy_obj, "infer_solution"):
            return policy_obj.infer_solution(instance)
        return policy_obj.infer(snapshot)

    return run_hybrid_with_solver_fn(
        instance=instance,
        solver_fn=policy_solver_fn,
        budget_s=budget_s,
        gls_time_share=gls_time_share,
        feasibility_mode=feasibility_mode,
        gls_debug=gls_debug,
        gls_log_every=gls_log_every,
    )

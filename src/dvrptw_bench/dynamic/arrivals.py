"""Dynamic customer revelation policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dvrptw_bench.common.typing import Node, VRPTWInstance


@dataclass
class DynamicScenario:
    instance: VRPTWInstance
    reveal_times: dict[int, float]
    dynamic_customer_ids: set[int]
    feasible: bool
    dropped_reason: str | None = None


def build_dynamic_scenario(
    instance: VRPTWInstance,
    epsilon: float,
    seed: int,
    cutoff_ratio: float = 0.8,
    discard_infeasible: bool = True,
    end_time_closeness: float|None = None,
) -> DynamicScenario:
    rng = np.random.default_rng(seed)
    horizon = instance.depot.due_time
    cutoff = horizon * cutoff_ratio
    n = instance.n_customers
    n_dyn = int(round(epsilon * n))
    ids = [c.id for c in instance.customers]
    dynamic_ids = set(rng.choice(ids, size=n_dyn, replace=False).tolist()) if n_dyn > 0 else set()

    reveal_times: dict[int, float] = {}
    adjusted: list[Node] = []
    feasible = True
    reason = None

    # print(f"CUstomers length: {len(instance.customers)}, n_dyn: {n_dyn}, dynamic_ids: {sorted(dynamic_ids)}")
    for c in instance.customers:
        if c.id not in dynamic_ids:
            adjusted.append(c)
            continue
        upper_bound = min(c.due_time, cutoff)
        if end_time_closeness is not None and end_time_closeness > 0:
            closeness = rng.uniform(upper_bound * end_time_closeness, upper_bound)
            reveal_time = float(rng.uniform(closeness, upper_bound))
        else:
            reveal_time = float(rng.uniform(0.0, upper_bound))
        reveal_times[c.id] = reveal_time
        ready = max(c.ready_time, reveal_time)
        due = min(c.due_time, horizon)
        if ready > due:
            feasible = False
            reason = f"ready({ready}) > due({due}) for customer {c.id}"
            if discard_infeasible:
                break
        adjusted.append(c.model_copy(update={"ready_time": ready, "due_time": due}))

    # print(f"Dynamic scenario with {n_dyn} dynamic customers (epsilon={epsilon:.2f}), cutoff at {cutoff:.1f}s ({cutoff_ratio:.2%} of horizon)")
    # print(f"Dynamic customer IDs: {sorted(dynamic_ids)}")
    new_instance = instance.model_copy(update={"customers": adjusted})
    if not feasible and discard_infeasible:
        return DynamicScenario(instance=instance, reveal_times={}, dynamic_customer_ids=dynamic_ids, feasible=False, dropped_reason=reason)
    return DynamicScenario(instance=new_instance, reveal_times=reveal_times, dynamic_customer_ids=dynamic_ids, feasible=feasible, dropped_reason=reason)

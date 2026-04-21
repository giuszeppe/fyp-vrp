"""Feasibility checks for VRPTW solutions."""

from __future__ import annotations

import math

from dvrptw_bench.common.typing import FeasibilityReport, Solution, VRPTWInstance
from dvrptw_bench.dynamic.dynamic_instance import DynamicInstance


def _travel(instance: DynamicInstance, nodes: dict[int, object], from_id: int, to_id: int) -> float:
    if 0 <= from_id < len(instance.distance_matrix) and 0 <= to_id < len(instance.distance_matrix):
        return instance.distance_matrix[from_id][to_id]
    a = nodes[from_id]
    b = nodes[to_id]
    return math.hypot(a.x - b.x, a.y - b.y)


def verify_solution(instance: DynamicInstance, solution: Solution) -> FeasibilityReport:
    nodes = {n.id: n for n in instance.all_nodes}

    cap_violation = 0.0
    time_violation = 0.0
    served: set[int] = set()

    for route in solution.routes:
        load = 0.0
        time = instance.depot.ready_time
        prev = instance.depot.id
        for nid in route.node_ids:
            if nid == instance.depot.id:
                continue
            if nid not in nodes:
                continue
            c = nodes[nid]
            served.add(nid)
            load += c.demand
            if load > instance.vehicle_capacity:
                cap_violation += load - instance.vehicle_capacity
            travel = _travel(instance, nodes, prev, nid)
            time += travel
            if time < c.ready_time:
                time = c.ready_time
            if time > c.due_time:
                time_violation += time - c.due_time
            time += c.service_time
            prev = nid

    unserved = [c.id for c in instance.customers if c.id not in served]
    feasible = cap_violation <= 1e-9 and time_violation <= 1e-9 and not unserved
    return FeasibilityReport(
        feasible=feasible,
        capacity_violation=cap_violation,
        time_violation=time_violation,
        unserved_customers=unserved,
    )

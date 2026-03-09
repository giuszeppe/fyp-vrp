"""Objective computations."""

from __future__ import annotations

import math

from dvrptw_bench.common.typing import Solution, VRPTWInstance


def _travel(instance: VRPTWInstance, from_id: int, to_id: int) -> float:
    if 0 <= from_id < len(instance.distance_matrix) and 0 <= to_id < len(instance.distance_matrix):
        return instance.distance_matrix[from_id][to_id]
    nodes = {n.id: n for n in instance.all_nodes}
    a = nodes[from_id]
    b = nodes[to_id]
    return math.hypot(a.x - b.x, a.y - b.y)


def route_distance(instance: VRPTWInstance, node_ids: list[int]) -> float:
    dep = instance.depot.id
    path = [dep, *[n for n in node_ids if n != dep], dep]
    return sum(_travel(instance, a, b) for a, b in zip(path[:-1], path[1:], strict=False))


def total_distance(instance: VRPTWInstance, solution: Solution) -> float:
    return sum(route_distance(instance, r.node_ids) for r in solution.routes)


def optimality_gap(value: float, reference: float) -> float:
    if reference <= 0:
        return 0.0
    return 100.0 * (value - reference) / reference

"""Decoding helpers from permutations to routes."""

from __future__ import annotations

from dvrptw_bench.common.typing import Route, Solution, VRPTWInstance


def split_permutation(instance: VRPTWInstance, perm: list[int], strategy: str) -> Solution:
    routes: list[Route] = []
    current: list[int] = []
    load = 0.0
    routeId = 0
    demand = {c.id: c.demand for c in instance.customers}
    for cid in perm:
        d = demand.get(cid, 0.0)
        if load + d > instance.vehicle_capacity and current:
            routes.append(Route(vehicle_id=routeId, node_ids=current))
            routeId += 1
            current = []
            load = 0.0
        current.append(cid)
        load += d
    if current:
        routes.append(Route(vehicle_id=routeId, node_ids=current))
    return Solution(strategy=strategy, routes=routes)

"""Adapters between benchmark instances and RouteFinder tensors/actions."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from dvrptw_bench.common.typing import Route, Solution, VRPTWInstance


def _normalize_coord(coord: torch.Tensor) -> torch.Tensor:
    x, y = coord[:, 0], coord[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_range = x_max - x_min
    y_range = y_max - y_min

    if torch.isclose(x_range, torch.tensor(0.0, device=coord.device)):
        x_scaled = torch.zeros_like(x)
    else:
        x_scaled = (x - x_min) / x_range

    if torch.isclose(y_range, torch.tensor(0.0, device=coord.device)):
        y_scaled = torch.zeros_like(y)
    else:
        y_scaled = (y - y_min) / y_range

    return torch.stack([x_scaled, y_scaled], dim=1)


def instance_to_routefinder_td(
    instance: VRPTWInstance,
    normalize_coords: bool = True,
) -> TensorDict:
    coords = torch.tensor(
        [[instance.depot.x, instance.depot.y], *[[c.x, c.y] for c in instance.customers]],
        dtype=torch.float32,
    )
    if normalize_coords:
        coords = _normalize_coord(coords)

    demand_linehaul = torch.tensor(
        [c.demand for c in instance.customers],
        dtype=torch.float32,
    )
    capacity = float(instance.vehicle_capacity)

    service_time = torch.tensor(
        [0.0, *[c.service_time for c in instance.customers]],
        dtype=torch.float32,
    )
    time_windows = torch.tensor(
        [
            [instance.depot.ready_time, instance.depot.due_time],
            *[[c.ready_time, c.due_time] for c in instance.customers],
        ],
        dtype=torch.float32,
    )

    td = TensorDict(
        {
            "locs": coords,
            "demand_linehaul": demand_linehaul / capacity,
            "demand_backhaul": torch.zeros_like(demand_linehaul),
            "distance_limit": torch.tensor([float("inf")], dtype=torch.float32),
            "service_time": service_time,
            "open_route": torch.tensor([False], dtype=torch.bool),
            "time_windows": time_windows,
            "vehicle_capacity": torch.tensor([1.0], dtype=torch.float32),
            "capacity_original": torch.tensor([capacity], dtype=torch.float32),
            "speed": torch.tensor([1.0], dtype=torch.float32),
        },
        batch_size=[],
    )
    return td.unsqueeze(0)


def decode_routefinder_actions(
    actions: torch.Tensor,
    instance: VRPTWInstance,
) -> list[Route]:
    normalized_actions = actions.detach().cpu()
    while normalized_actions.ndim > 2:
        normalized_actions = normalized_actions[0]
    if normalized_actions.ndim == 1:
        normalized_actions = normalized_actions.unsqueeze(0)

    action_seq = normalized_actions[0].tolist()
    idx_to_customer_id = {idx + 1: c.id for idx, c in enumerate(instance.customers)}

    routes: list[Route] = []
    current: list[int] = []
    seen: set[int] = set()

    for token in action_seq:
        token = int(token)
        if token == 0:
            if current:
                routes.append(Route(vehicle_id=len(routes), node_ids=current))
                current = []
            continue

        customer_id = idx_to_customer_id.get(token)
        if customer_id is None or customer_id in seen:
            continue

        seen.add(customer_id)
        current.append(customer_id)

    if current:
        routes.append(Route(vehicle_id=len(routes), node_ids=current))

    if not routes:
        routes = [Route(vehicle_id=0, node_ids=[])]
    return routes


def routefinder_actions_to_solution(
    actions: torch.Tensor,
    instance: VRPTWInstance,
    strategy: str = "routefinder",
) -> Solution:
    return Solution(strategy=strategy, routes=decode_routefinder_actions(actions, instance))

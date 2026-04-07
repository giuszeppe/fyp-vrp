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

def scale_times_to_normalized_coords(
    coord_raw: torch.Tensor,         # [N+1, 2], original Solomon coords
    coord_norm: torch.Tensor,        # [N+1, 2], after _normalize_coord
    time_windows_raw: torch.Tensor,  # [N+1, 2]
    service_time_raw: torch.Tensor,  # [N+1]
    max_time_raw: float | torch.Tensor,
    eps: float = 1e-8,
):
    depot_raw = coord_raw[0:1]
    depot_norm = coord_norm[0:1]

    d0_raw = torch.norm(coord_raw[1:] - depot_raw, dim=-1)
    d0_norm = torch.norm(coord_norm[1:] - depot_norm, dim=-1)

    # robust instance-level scale
    gamma = d0_norm.median() / (d0_raw.median() + eps)

    time_windows_norm = time_windows_raw * gamma
    service_time_norm = service_time_raw * gamma
    max_time_norm = max_time_raw * gamma

    return time_windows_norm, service_time_norm, max_time_norm, gamma

def pairwise_depot_dist(coords):
    # coords: [B, N+1, 2]
    return torch.norm(coords[:, 1:] - coords[:, 0:1], dim=-1)

def scale_times_batch(coord_raw, coord_norm, time_windows_raw, service_time_raw, max_time_raw, eps=1e-8):
    d0_raw = pairwise_depot_dist(coord_raw)    # [B, N]
    d0_norm = pairwise_depot_dist(coord_norm)  # [B, N]

    gamma = d0_norm.median(dim=1).values / (d0_raw.median(dim=1).values + eps)  # [B]

    time_windows_norm = time_windows_raw * gamma[:, None, None]
    service_time_norm = service_time_raw * gamma[:, None]

    if torch.is_tensor(max_time_raw):
        max_time_norm = max_time_raw * gamma
    else:
        max_time_norm = gamma * max_time_raw

    return time_windows_norm, service_time_norm, max_time_norm, gamma

def scale_times_to_max_time(
    time_windows: torch.Tensor,   # [N+1, 2]
    service_time: torch.Tensor,   # [N+1]
    target_max_time: float = 4.6,
    eps: float = 1e-8,
):
    depot_due = time_windows[0, 1].item()
    alpha = target_max_time / max(depot_due, eps)
    print(f"Scaling times to target max time {target_max_time} with depot due time {depot_due} and epsilon {eps}")

    time_windows_scaled = time_windows * alpha
    service_time_scaled = service_time * alpha

    # Make depot exactly [0, target_max_time]
    time_windows_scaled[0, 0] = 0.0
    time_windows_scaled[0, 1] = target_max_time

    return time_windows_scaled, service_time_scaled, alpha

def instance_to_routefinder_td(
    instance: VRPTWInstance,
    normalize_coords: bool = True,
) -> TensorDict:
    coords_raw = torch.tensor(
        [[instance.depot.x, instance.depot.y], *[[c.x, c.y] for c in instance.customers]],
        dtype=torch.float32,
    )
    coords = coords_raw if not normalize_coords else _normalize_coord(coords_raw)


    demand_linehaul = torch.tensor(
        [c.demand for c in instance.customers],
        dtype=torch.float32,
    )
    capacity = float(instance.vehicle_capacity)

    service_time = torch.tensor(
        [0.0, *[c.service_time for c in instance.customers]],
        dtype=torch.float32,
    )
    if normalize_coords:
        time_windows, service_time, max_time, gamma = scale_times_to_normalized_coords(
            coord_raw=coords_raw,
            coord_norm=coords,
            time_windows_raw=torch.tensor(
                [
                    [instance.depot.ready_time, instance.depot.due_time],
                    *[[c.ready_time, c.due_time] for c in instance.customers],
                ],
                dtype=torch.float32,
            ),
            service_time_raw=service_time,
            max_time_raw=float(instance.depot.due_time),
        )
        time_windows, service_time, alpha = scale_times_to_max_time(time_windows, service_time, target_max_time=4.6)
        # time_windows, service_time, max_time, gamma = scale_times_batch(
        #     coord_raw=torch.tensor(
        #         [[instance.depot.x, instance.depot.y], *[[c.x, c.y] for c in instance.customers]],
        #         dtype=torch.float32,
        #     ),
        #     coord_norm=coords,
        #     time_windows_raw=torch.tensor(
        #         [
        #             [instance.depot.ready_time, instance.depot.due_time],
        #             *[[c.ready_time, c.due_time] for c in instance.customers],
        #         ],
        #         dtype=torch.float32,
        #     ),
        #     service_time_raw=service_time,
        #     max_time_raw=float(instance.depot.due_time),
        # )

    else: 
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

from __future__ import annotations

from typing import Optional, Union

import torch
from rl4co.utils.ops import gather_by_index, get_distance
from tensordict.tensordict import TensorDict

from routefinder.envs.mtvrp import MTVRPEnv, MTVRPGenerator


class MTVRPDynamicEnv(MTVRPEnv):
    """Dynamic RouteFinder env driven by per-customer reveal times."""

    name = "mtvrp"

    def __init__(
        self,
        generator: MTVRPGenerator = None,
        generator_params: dict = {},
        select_start_nodes_fn: Union[str, callable] = "all",
        check_solution: bool = False,
        load_solutions: bool = True,
        solution_fname: str = "_sol_pyvrp.npz",
        dod: float = 0.3,
        cutoff_time: float = 0.8,
        hide_unrevealed_customers: bool = True,
        dynamic_seed: int | None = None,
        **kwargs,
    ):
        super().__init__(
            generator=generator,
            generator_params=generator_params,
            select_start_nodes_fn=select_start_nodes_fn,
            check_solution=check_solution,
            load_solutions=load_solutions,
            solution_fname=solution_fname,
            **kwargs,
        )
        self.dod = float(dod)
        self.cutoff_time = float(cutoff_time)
        self.hide_unrevealed_customers = bool(hide_unrevealed_customers)
        self.dynamic_seed = dynamic_seed

    @staticmethod
    def _normalize_reveal_times(
        reveal_times: Optional[torch.Tensor],
        customer_count: int,
        batch_size: torch.Size | tuple[int, ...],
        device: torch.device,
    ) -> torch.Tensor:
        if reveal_times is None:
            result = torch.zeros((*batch_size, customer_count + 1), dtype=torch.float32, device=device)
        else:
            if reveal_times.dim() == 1:
                reveal_times = reveal_times.unsqueeze(0)
            if reveal_times.size(-1) == customer_count:
                reveal_times = torch.cat(
                    [torch.zeros_like(reveal_times[..., :1]), reveal_times], dim=-1
                )
            if reveal_times.size(-1) != customer_count + 1:
                raise ValueError("reveal_times must have shape [B, N] or [B, N+1]")
            result = reveal_times.to(device=device, dtype=torch.float32)
        result[..., 0] = 0.0
        return result

    @staticmethod
    def _normalize_is_dynamic(
        is_dynamic: Optional[torch.Tensor],
        reveal_times: torch.Tensor,
        customer_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        if is_dynamic is None:
            dynamic = reveal_times > 0
        else:
            if is_dynamic.dim() == 1:
                is_dynamic = is_dynamic.unsqueeze(0)
            if is_dynamic.size(-1) == customer_count:
                is_dynamic = torch.cat(
                    [torch.zeros_like(is_dynamic[..., :1]), is_dynamic], dim=-1
                )
            if is_dynamic.size(-1) != customer_count + 1:
                raise ValueError("is_dynamic must have shape [B, N] or [B, N+1]")
            dynamic = is_dynamic.to(device=device, dtype=torch.bool)
        dynamic[..., 0] = False
        return dynamic

    @staticmethod
    def _compute_revealed(td: TensorDict) -> torch.Tensor:
        revealed = td["reveal_times"] <= td["elapsed_time"]
        revealed[..., 0] = True
        return revealed

    def _refresh_dynamic_observation(self, td: TensorDict) -> TensorDict:
        revealed = self._compute_revealed(td)
        td.set("revealed", revealed)

        if not self.hide_unrevealed_customers:
            td.update(
                {
                    "locs": td["actual_locs"],
                    "demand_linehaul": td["actual_demand_linehaul"],
                    "demand_backhaul": td["actual_demand_backhaul"],
                    "service_time": td["actual_service_time"],
                    "time_windows": td["actual_time_windows"],
                }
            )
            return td

        hidden = ~revealed
        depot_xy = td["actual_locs"][..., :1, :].expand_as(td["actual_locs"])
        depot_tw = td["actual_time_windows"][..., :1, :].expand_as(td["actual_time_windows"])
        td.update(
            {
                "locs": torch.where(hidden.unsqueeze(-1), depot_xy, td["actual_locs"]),
                "demand_linehaul": torch.where(
                    hidden, torch.zeros_like(td["actual_demand_linehaul"]), td["actual_demand_linehaul"]
                ),
                "demand_backhaul": torch.where(
                    hidden, torch.zeros_like(td["actual_demand_backhaul"]), td["actual_demand_backhaul"]
                ),
                "service_time": torch.where(
                    hidden, torch.zeros_like(td["actual_service_time"]), td["actual_service_time"]
                ),
                "time_windows": torch.where(
                    hidden.unsqueeze(-1), depot_tw, td["actual_time_windows"]
                ),
            }
        )
        return td

    @staticmethod
    def _advance_to_next_reveal(td: TensorDict) -> TensorDict:
        all_customers_visited = td["visited"][..., 1:].all(dim=-1, keepdim=True)
        can_visit = MTVRPDynamicEnv._compute_feasibility(td)
        has_revealed_customer = can_visit[..., 1:].any(dim=-1, keepdim=True)
        future_unvisited = (~td["revealed"]) & (~td["visited"])
        future_unvisited[..., 0] = False
        need_wait = (~all_customers_visited) & (~has_revealed_customer) & future_unvisited.any(
            dim=-1, keepdim=True
        )

        if need_wait.any():
            inf = torch.full_like(td["reveal_times"], float("inf"))
            next_reveal = torch.where(future_unvisited, td["reveal_times"], inf).min(
                dim=-1, keepdim=True
            ).values
            td.set("elapsed_time", torch.where(need_wait, next_reveal, td["elapsed_time"]))
        return td

    @staticmethod
    def _compute_feasibility(td: TensorDict) -> torch.Tensor:
        curr_node = td["current_node"]
        locs = td["actual_locs"]
        d_ij = get_distance(gather_by_index(locs, curr_node)[..., None, :], locs)
        d_j0 = get_distance(locs, locs[..., 0:1, :])

        early_tw = td["actual_time_windows"][..., 0]
        late_tw = td["actual_time_windows"][..., 1]
        arrival_time = td["current_time"] + (d_ij / td["speed"])
        can_reach_customer = arrival_time < late_tw
        can_reach_depot = (
            torch.max(arrival_time, early_tw) + td["actual_service_time"] + (d_j0 / td["speed"])
        ) * ~td["open_route"] < late_tw[..., 0:1]

        exceeds_dist_limit = (
            td["current_route_length"] + d_ij + (d_j0 * ~td["open_route"]) > td["distance_limit"]
        )
        exceeds_cap_linehaul = (
            td["actual_demand_linehaul"] + td["used_capacity_linehaul"] > td["vehicle_capacity"]
        )
        exceeds_cap_backhaul = (
            td["actual_demand_backhaul"] + td["used_capacity_backhaul"] > td["vehicle_capacity"]
        )

        linehauls_missing = ((td["actual_demand_linehaul"] * ~td["visited"]).sum(-1) > 0)[..., None]
        is_carrying_backhaul = (
            gather_by_index(
                src=td["actual_demand_backhaul"], idx=curr_node, dim=1, squeeze=False
            )
            > 0
        )
        meets_demand_constraint_backhaul_1 = (
            linehauls_missing
            & ~exceeds_cap_linehaul
            & ~is_carrying_backhaul
            & (td["actual_demand_linehaul"] > 0)
        ) | (~exceeds_cap_backhaul & (td["actual_demand_backhaul"] > 0))

        cannot_serve_linehaul = (
            td["actual_demand_linehaul"] > td["vehicle_capacity"] - td["used_capacity_backhaul"]
        )
        meets_demand_constraint_backhaul_2 = (
            ~exceeds_cap_linehaul & ~exceeds_cap_backhaul & ~cannot_serve_linehaul
        )
        meets_demand_constraint = (
            (td["backhaul_class"] == 1) & meets_demand_constraint_backhaul_1
        ) | ((td["backhaul_class"] == 2) & meets_demand_constraint_backhaul_2)

        return (
            td["revealed"]
            & can_reach_customer
            & can_reach_depot
            & meets_demand_constraint
            & ~exceeds_dist_limit
            & ~td["visited"]
        )

    def _reset(
        self,
        td: Optional[TensorDict],
        batch_size: Optional[list] = None,
    ) -> TensorDict:
        if td is None:
            raise ValueError("MTVRPDynamicEnv._reset requires an input TensorDict")
        if batch_size is None:
            return self._reset(td, batch_size=td.batch_size)

        device = td.device

        demand_linehaul = torch.cat(
            [torch.zeros_like(td["demand_linehaul"][..., :1]), td["demand_linehaul"]], dim=1
        )
        demand_backhaul = td.get("demand_backhaul", torch.zeros_like(td["demand_linehaul"]))
        demand_backhaul = torch.cat(
            [torch.zeros_like(td["demand_linehaul"][..., :1]), demand_backhaul], dim=1
        )
        backhaul_class = td.get(
            "backhaul_class",
            torch.full((*batch_size, 1), 1, dtype=torch.int32, device=device),
        )

        time_windows = td.get("time_windows", None)
        if time_windows is None:
            time_windows = torch.zeros_like(td["locs"])
            time_windows[..., 1] = float("inf")
        service_time = td.get("service_time", torch.zeros_like(demand_linehaul))
        open_route = td.get(
            "open_route", torch.zeros_like(demand_linehaul[..., :1], dtype=torch.bool)
        )
        distance_limit = td.get(
            "distance_limit", torch.full_like(demand_linehaul[..., :1], float("inf"))
        )

        customer_count = td["locs"].shape[-2] - 1
        reveal_times = self._normalize_reveal_times(
            td.get("reveal_times", None), customer_count, batch_size, device
        )
        is_dynamic = self._normalize_is_dynamic(
            td.get("is_dynamic", None), reveal_times, customer_count, device
        )

        td_reset = TensorDict(
            {
                "locs": td["locs"],
                "actual_locs": td["locs"],
                "demand_backhaul": demand_backhaul,
                "actual_demand_backhaul": demand_backhaul,
                "demand_linehaul": demand_linehaul,
                "actual_demand_linehaul": demand_linehaul,
                "backhaul_class": backhaul_class,
                "distance_limit": distance_limit,
                "service_time": service_time,
                "actual_service_time": service_time,
                "open_route": open_route,
                "time_windows": time_windows,
                "actual_time_windows": time_windows,
                "reveal_times": reveal_times,
                "is_dynamic": is_dynamic,
                "revealed": torch.zeros_like(reveal_times, dtype=torch.bool),
                "speed": td.get("speed", torch.ones_like(demand_linehaul[..., :1])),
                "vehicle_capacity": td.get(
                    "vehicle_capacity", torch.ones_like(demand_linehaul[..., :1])
                ),
                "capacity_original": td.get(
                    "capacity_original", torch.ones_like(demand_linehaul[..., :1])
                ),
                "current_node": torch.zeros((*batch_size,), dtype=torch.long, device=device),
                "current_route_length": torch.zeros((*batch_size, 1), dtype=torch.float32, device=device),
                "current_time": torch.zeros((*batch_size, 1), dtype=torch.float32, device=device),
                "elapsed_time": torch.zeros((*batch_size, 1), dtype=torch.float32, device=device),
                "used_capacity_backhaul": torch.zeros((*batch_size, 1), device=device),
                "used_capacity_linehaul": torch.zeros((*batch_size, 1), device=device),
                "visited": torch.zeros(
                    (*batch_size, td["locs"].shape[-2]), dtype=torch.bool, device=device
                ),
                "done": torch.zeros((*batch_size,), dtype=torch.bool, device=device),
                "reward": torch.zeros((*batch_size,), dtype=torch.float32, device=device),
            },
            batch_size=batch_size,
            device=device,
        )
        td_reset = self._refresh_dynamic_observation(td_reset)
        td_reset = self._advance_to_next_reveal(td_reset)
        td_reset = self._refresh_dynamic_observation(td_reset)
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    def _step(self, td: TensorDict) -> TensorDict:
        prev_node, curr_node = td["current_node"], td["action"]
        prev_loc = gather_by_index(td["actual_locs"], prev_node)
        curr_loc = gather_by_index(td["actual_locs"], curr_node)
        distance = get_distance(prev_loc, curr_loc)[..., None]

        service_time = gather_by_index(
            src=td["actual_service_time"], idx=curr_node, dim=1, squeeze=False
        )
        start_times = gather_by_index(
            src=td["actual_time_windows"], idx=curr_node, dim=1, squeeze=False
        )[..., 0]
        arrival_time_route = td["current_time"] + distance / td["speed"]
        service_start = torch.max(arrival_time_route, start_times)
        curr_time = torch.where(
            curr_node[:, None] != 0,
            service_start + service_time,
            torch.zeros_like(td["current_time"]),
        )
        arrival_time_elapsed = td["elapsed_time"] + distance / td["speed"]
        elapsed_time = torch.where(
            curr_node[:, None] != 0,
            torch.max(arrival_time_elapsed, start_times) + service_time,
            arrival_time_elapsed,
        )
        curr_route_length = (curr_node[:, None] != 0) * (td["current_route_length"] + distance)

        selected_demand_linehaul = gather_by_index(
            td["actual_demand_linehaul"], curr_node, dim=1, squeeze=False
        )
        selected_demand_backhaul = gather_by_index(
            td["actual_demand_backhaul"], curr_node, dim=1, squeeze=False
        )
        used_capacity_linehaul = (curr_node[:, None] != 0) * (
            td["used_capacity_linehaul"] + selected_demand_linehaul
        )
        used_capacity_backhaul = (curr_node[:, None] != 0) * (
            td["used_capacity_backhaul"] + selected_demand_backhaul
        )

        visited = td["visited"].scatter(-1, curr_node[..., None], True)
        done = visited[..., 1:].all(dim=-1)

        td.update(
            {
                "current_node": curr_node,
                "current_route_length": curr_route_length,
                "current_time": curr_time,
                "elapsed_time": elapsed_time,
                "done": done,
                "reward": torch.zeros_like(done).float(),
                "used_capacity_linehaul": used_capacity_linehaul,
                "used_capacity_backhaul": used_capacity_backhaul,
                "visited": visited,
            }
        )
        td = self._refresh_dynamic_observation(td)
        td = self._advance_to_next_reveal(td)
        td = self._refresh_dynamic_observation(td)
        td.set("action_mask", self.get_action_mask(td))
        return td

    def _get_reward(self, td: TensorDict, actions: TensorDict) -> TensorDict:
        go_from = torch.cat((torch.zeros_like(actions[:, :1]), actions), dim=1)
        go_to = torch.roll(go_from, -1, dims=1)
        loc_from = gather_by_index(td.get("actual_locs", td["locs"]), go_from)
        loc_to = gather_by_index(td.get("actual_locs", td["locs"]), go_to)
        distances = get_distance(loc_from, loc_to)
        tour_length = (distances * ~((go_to == 0) & td["open_route"])).sum(-1)
        return -tour_length

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        can_visit = MTVRPDynamicEnv._compute_feasibility(td)
        curr_node = td["current_node"]
        can_visit[:, 0] = ~((curr_node == 0) & (can_visit[:, 1:].sum(-1) > 0))
        return can_visit

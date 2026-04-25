import os

from typing import List, Optional, Union

import numpy as np
import torch

from rl4co.data.utils import load_npz_to_tensordict
from rl4co.utils.ops import gather_by_index, get_distance
from rl4co.utils.pylogger import get_pylogger
from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, UnboundedContinuous, UnboundedDiscrete

from routefinder.envs.mtvrp import MTVRPEnv, MTVRPGenerator

log = get_pylogger(__name__)


class MTVRPDynamicEnv(MTVRPEnv):
    """RouteFinder environment with dynamic customer revelation."""

    name = "mtvrp"

    def __init__(
        self,
        generator: MTVRPGenerator = None,
        generator_params: dict = {},
        select_start_nodes_fn: Union[str, callable] = "all",
        check_solution: bool = False,
        load_solutions: bool = True,
        solution_fname: str = "_sol_pyvrp.npz",
        allow_late_customers: bool = True,
        lateness_penalty: float = 0.0,
        allow_reject_customers: bool = False,
        reject_penalty: float = 0.0,
        hard_depot_time_window: bool = True,
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
        self.allow_late_customers = allow_late_customers
        self.lateness_penalty = float(lateness_penalty)
        self.allow_reject_customers = allow_reject_customers
        self.reject_penalty = float(reject_penalty)
        self.hard_depot_time_window = hard_depot_time_window
        self.dod = float(dod)
        self.cutoff_time = float(cutoff_time)
        self.hide_unrevealed_customers = bool(hide_unrevealed_customers)
        self.dynamic_seed = dynamic_seed

    @staticmethod
    def _ensure_reveal_times_with_depot(
        reveal_times: torch.Tensor,
        customer_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        if reveal_times.dim() == 1:
            reveal_times = reveal_times.unsqueeze(0)
        if reveal_times.size(-1) == customer_count:
            reveal_times = torch.cat(
                [torch.zeros_like(reveal_times[..., :1]), reveal_times], dim=-1
            )
        if reveal_times.size(-1) != customer_count + 1:
            raise ValueError("reveal_times must have shape [B, N] or [B, N+1]")
        reveal_times = reveal_times.to(device=device, dtype=torch.float32)
        reveal_times[..., 0] = 0.0
        return reveal_times

    def _sample_reveal_times(
        self,
        time_windows: torch.Tensor,
        customer_count: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = tuple(time_windows.shape[:-2])
        reveal_times = torch.zeros(
            *batch_size, customer_count + 1, dtype=torch.float32, device=device
        )
        is_dynamic = torch.zeros(
            *batch_size, customer_count + 1, dtype=torch.bool, device=device
        )
        if customer_count == 0 or self.dod <= 0:
            return reveal_times, is_dynamic

        horizon = time_windows[..., 0, 1]
        cutoff_abs = horizon * self.cutoff_time
        customer_due = time_windows[..., 1:, 1]

        n_dyn = int(round(self.dod * customer_count))
        n_dyn = max(0, min(customer_count, n_dyn))
        if n_dyn == 0:
            return reveal_times, is_dynamic

        flat_dynamic = is_dynamic[..., 1:].reshape(-1, customer_count)
        flat_reveal = reveal_times[..., 1:].reshape(-1, customer_count)
        flat_due = customer_due.reshape(-1, customer_count)
        flat_cutoff = cutoff_abs.reshape(-1)

        generator = None
        if self.dynamic_seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.dynamic_seed)

        for row in range(flat_dynamic.size(0)):
            perm = torch.randperm(customer_count, generator=generator, device=device)
            selected = perm[:n_dyn]
            flat_dynamic[row, selected] = True
            upper_bounds = torch.minimum(flat_due[row, selected], flat_cutoff[row])
            sampled = torch.rand(selected.numel(), generator=generator, device=device)
            flat_reveal[row, selected] = sampled * torch.clamp(upper_bounds, min=0.0)

        return reveal_times, is_dynamic

    @staticmethod
    def _apply_reveal_to_time_windows(
        time_windows: torch.Tensor,
        reveal_times: torch.Tensor,
        is_dynamic: torch.Tensor,
    ) -> torch.Tensor:
        adjusted = time_windows.clone()
        adjusted[..., 1:, 0] = torch.where(
            is_dynamic[..., 1:],
            torch.maximum(adjusted[..., 1:, 0], reveal_times[..., 1:]),
            adjusted[..., 1:, 0],
        )
        return adjusted

    def _refresh_dynamic_observation(self, td: TensorDict) -> TensorDict:
        revealed = td["reveal_times"] <= td["current_time"]
        revealed[..., 0] = True
        td.set("revealed", revealed)

        if not td["hide_unrevealed_customers"].any():
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
        zeros_like_linehaul = torch.zeros_like(td["actual_demand_linehaul"])
        zeros_like_backhaul = torch.zeros_like(td["actual_demand_backhaul"])
        zeros_like_service = torch.zeros_like(td["actual_service_time"])

        td.update(
            {
                "locs": torch.where(hidden.unsqueeze(-1), depot_xy, td["actual_locs"]),
                "demand_linehaul": torch.where(
                    hidden, zeros_like_linehaul, td["actual_demand_linehaul"]
                ),
                "demand_backhaul": torch.where(
                    hidden, zeros_like_backhaul, td["actual_demand_backhaul"]
                ),
                "service_time": torch.where(
                    hidden, zeros_like_service, td["actual_service_time"]
                ),
                "time_windows": torch.where(
                    hidden.unsqueeze(-1), depot_tw, td["actual_time_windows"]
                ),
            }
        )
        return td

    @staticmethod
    def _compute_customer_feasibility(td: TensorDict) -> torch.Tensor:
        curr_node = td["current_node"]
        locs = td["actual_locs"]
        d_ij = get_distance(gather_by_index(locs, curr_node)[..., None, :], locs)
        d_j0 = get_distance(locs, locs[..., 0:1, :])
        early_tw = td["actual_time_windows"][..., 0]
        late_tw = td["actual_time_windows"][..., 1]
        arrival_time = td["current_time"] + (d_ij / td["speed"])

        if td["allow_late_customers"].any():
            can_reach_customer = torch.ones_like(arrival_time, dtype=torch.bool)
        else:
            can_reach_customer = arrival_time < late_tw

        if td["hard_depot_time_window"].any():
            can_reach_depot = (
                torch.max(arrival_time, early_tw)
                + td["actual_service_time"]
                + (d_j0 / td["speed"])
            ) * ~td["open_route"] < late_tw[..., 0:1]
        else:
            can_reach_depot = torch.ones_like(arrival_time, dtype=torch.bool)

        exceeds_dist_limit = (
            td["current_route_length"] + d_ij + (d_j0 * ~td["open_route"])
            > td["distance_limit"]
        )
        exceeds_cap_linehaul = (
            td["actual_demand_linehaul"] + td["used_capacity_linehaul"] > td["vehicle_capacity"]
        )
        exceeds_cap_backhaul = (
            td["actual_demand_backhaul"] + td["used_capacity_backhaul"] > td["vehicle_capacity"]
        )

        linehauls_missing = (
            (td["actual_demand_linehaul"] * ~td["visited"]).sum(-1) > 0
        )[..., None]
        is_carrying_backhaul = (
            gather_by_index(
                src=td["actual_demand_backhaul"],
                idx=curr_node,
                dim=1,
                squeeze=False,
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
            td["actual_demand_linehaul"]
            > td["vehicle_capacity"] - td["used_capacity_backhaul"]
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

    def _advance_to_next_reveal(self, td: TensorDict) -> TensorDict:
        all_customers_visited = td["visited"][..., 1:].all(dim=-1, keepdim=True)
        can_visit = self._compute_customer_feasibility(td)
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
            td.set("current_time", torch.where(need_wait, next_reveal, td["current_time"]))
            td = self._refresh_dynamic_observation(td)
        return td

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
        late_times = gather_by_index(
            src=td["actual_time_windows"], idx=curr_node, dim=1, squeeze=False
        )[..., 1]

        arrival_time = td["current_time"] + distance / td["speed"]
        service_start = torch.max(arrival_time, start_times)
        curr_time = service_start + service_time

        step_tardiness = (curr_node[:, None] != 0).float() * torch.clamp(
            service_start - late_times, min=0.0
        )
        total_tardiness = td["total_tardiness"] + step_tardiness

        curr_route_length = torch.where(
            curr_node[:, None] != 0,
            td["current_route_length"] + distance,
            torch.zeros_like(td["current_route_length"]),
        )

        selected_demand_linehaul = gather_by_index(
            td["actual_demand_linehaul"], curr_node, dim=1, squeeze=False
        )
        selected_demand_backhaul = gather_by_index(
            td["actual_demand_backhaul"], curr_node, dim=1, squeeze=False
        )
        used_capacity_linehaul = torch.where(
            curr_node[:, None] != 0,
            td["used_capacity_linehaul"] + selected_demand_linehaul,
            torch.zeros_like(td["used_capacity_linehaul"]),
        )
        used_capacity_backhaul = torch.where(
            curr_node[:, None] != 0,
            td["used_capacity_backhaul"] + selected_demand_backhaul,
            torch.zeros_like(td["used_capacity_backhaul"]),
        )

        visited = td["visited"].scatter(-1, curr_node[..., None], True)
        all_customers_visited = visited[..., 1:].all(dim=-1)
        early_stop = (
            td["allow_reject_customers"].squeeze(-1) & (prev_node == 0) & (curr_node == 0)
        )
        done = all_customers_visited | early_stop

        td.update(
            {
                "current_node": curr_node,
                "current_route_length": curr_route_length,
                "current_time": curr_time,
                "done": done,
                "reward": torch.zeros_like(done).float(),
                "used_capacity_linehaul": used_capacity_linehaul,
                "used_capacity_backhaul": used_capacity_backhaul,
                "visited": visited,
                "total_tardiness": total_tardiness,
            }
        )
        td = self._refresh_dynamic_observation(td)
        td = self._advance_to_next_reveal(td)
        td.set("action_mask", self.get_action_mask(td))
        return td

    def _reset(
        self,
        td: Optional[TensorDict],
        batch_size: Optional[list] = None,
    ) -> TensorDict:
        device = td.device

        demand_linehaul = torch.cat(
            [torch.zeros_like(td["demand_linehaul"][..., :1]), td["demand_linehaul"]],
            dim=1,
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
        reveal_times = td.get("reveal_times", None)
        is_dynamic = td.get("is_dynamic", None)
        if reveal_times is not None:
            reveal_times = self._ensure_reveal_times_with_depot(
                reveal_times, customer_count, device
            )
            if is_dynamic is None:
                is_dynamic = reveal_times > 0
        else:
            reveal_times, is_dynamic = self._sample_reveal_times(
                time_windows, customer_count, device
            )

        if is_dynamic.dim() == 1:
            is_dynamic = is_dynamic.unsqueeze(0)
        if is_dynamic.size(-1) == customer_count:
            is_dynamic = torch.cat(
                [torch.zeros_like(is_dynamic[..., :1]), is_dynamic], dim=-1
            )
        is_dynamic = is_dynamic.to(device=device, dtype=torch.bool)
        is_dynamic[..., 0] = False
        reveal_times = torch.where(is_dynamic, reveal_times, torch.zeros_like(reveal_times))
        actual_time_windows = self._apply_reveal_to_time_windows(time_windows, reveal_times, is_dynamic)

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
                "time_windows": actual_time_windows,
                "actual_time_windows": actual_time_windows,
                "reveal_times": reveal_times,
                "is_dynamic": is_dynamic,
                "revealed": reveal_times <= 0,
                "speed": td.get("speed", torch.ones_like(demand_linehaul[..., :1])),
                "vehicle_capacity": td.get(
                    "vehicle_capacity", torch.ones_like(demand_linehaul[..., :1])
                ),
                "capacity_original": td.get(
                    "capacity_original", torch.ones_like(demand_linehaul[..., :1])
                ),
                "current_node": torch.zeros((*batch_size,), dtype=torch.long, device=device),
                "current_route_length": torch.zeros(
                    (*batch_size, 1), dtype=torch.float32, device=device
                ),
                "current_time": torch.zeros((*batch_size, 1), dtype=torch.float32, device=device),
                "used_capacity_backhaul": torch.zeros((*batch_size, 1), device=device),
                "used_capacity_linehaul": torch.zeros((*batch_size, 1), device=device),
                "visited": torch.zeros(
                    (*batch_size, td["locs"].shape[-2]), dtype=torch.bool, device=device
                ),
                "total_tardiness": torch.zeros(
                    (*batch_size, 1), dtype=torch.float32, device=device
                ),
                "allow_late_customers": torch.full(
                    (*batch_size, 1), self.allow_late_customers, dtype=torch.bool, device=device
                ),
                "lateness_penalty": torch.full(
                    (*batch_size, 1), self.lateness_penalty, dtype=torch.float32, device=device
                ),
                "allow_reject_customers": torch.full(
                    (*batch_size, 1), self.allow_reject_customers, dtype=torch.bool, device=device
                ),
                "reject_penalty": torch.full(
                    (*batch_size, 1), self.reject_penalty, dtype=torch.float32, device=device
                ),
                "hard_depot_time_window": torch.full(
                    (*batch_size, 1), self.hard_depot_time_window, dtype=torch.bool, device=device
                ),
                "dod": torch.full((*batch_size, 1), self.dod, dtype=torch.float32, device=device),
                "cutoff_time": torch.full(
                    (*batch_size, 1), self.cutoff_time, dtype=torch.float32, device=device
                ),
                "hide_unrevealed_customers": torch.full(
                    (*batch_size, 1),
                    self.hide_unrevealed_customers,
                    dtype=torch.bool,
                    device=device,
                ),
            },
            batch_size=batch_size,
            device=device,
        )
        td_reset = self._refresh_dynamic_observation(td_reset)
        td_reset = self._advance_to_next_reveal(td_reset)
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:  # type: ignore
        can_visit = MTVRPDynamicEnv._compute_customer_feasibility(td)
        curr_node = td["current_node"]

        if td["allow_reject_customers"].any():
            can_visit[:, 0] = True
        else:
            can_visit[:, 0] = ~((curr_node == 0) & (can_visit[:, 1:].sum(-1) > 0))

        return can_visit

    def _get_reward(self, td: TensorDict, actions: TensorDict) -> TensorDict:
        go_from = torch.cat((torch.zeros_like(actions[:, :1]), actions), dim=1)
        go_to = torch.roll(go_from, -1, dims=1)
        loc_from = gather_by_index(td.get("actual_locs", td["locs"]), go_from)
        loc_to = gather_by_index(td.get("actual_locs", td["locs"]), go_to)

        distances = get_distance(loc_from, loc_to)
        tour_length = (distances * ~((go_to == 0) & td["open_route"])).sum(-1)

        total_tardiness = td.get("total_tardiness", torch.zeros_like(tour_length[..., None])).squeeze(-1)
        lateness_cost = td.get(
            "lateness_penalty", torch.zeros_like(total_tardiness[..., None])
        ).squeeze(-1) * total_tardiness

        unserved = (~td["visited"][..., 1:]).sum(-1).float()
        reject_cost = td.get("reject_penalty", torch.zeros_like(unserved[..., None])).squeeze(
            -1
        ) * unserved * td.get(
            "allow_reject_customers",
            torch.zeros_like(unserved[..., None], dtype=torch.bool),
        ).squeeze(-1).float()

        return -(tour_length + lateness_cost + reject_cost)

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor):
        batch_size, n_loc = td["actual_demand_linehaul"].size()
        n_loc -= 1
        allow_reject = td.get(
            "allow_reject_customers",
            torch.zeros((batch_size, 1), dtype=torch.bool, device=td.device),
        ).squeeze(-1)
        allow_late = td.get(
            "allow_late_customers",
            torch.zeros((batch_size, 1), dtype=torch.bool, device=td.device),
        ).squeeze(-1)

        sorted_pi = actions.data.sort(1)[0]
        if not allow_reject.any():
            assert (
                torch.arange(1, n_loc + 1, out=sorted_pi.data.new())
                .view(1, -1)
                .expand(batch_size, n_loc)
                == sorted_pi[:, -n_loc:]
            ).all() and (sorted_pi[:, :-n_loc] == 0).all(), "Invalid tour"
        else:
            nonzero = actions.clone()
            nonzero[nonzero == 0] = n_loc + 1
            sorted_nonzero = nonzero.sort(1)[0]
            served = sorted_nonzero[:, :-1]
            nxt = sorted_nonzero[:, 1:]
            assert ((served == n_loc + 1) | (served != nxt)).all(), "A customer was served more than once"

        locs = td.get("actual_locs", td["locs"])
        time_windows = td.get("actual_time_windows", td["time_windows"])
        service_time = td.get("actual_service_time", td["service_time"])
        demand_l = td.get("actual_demand_linehaul", td["demand_linehaul"])
        demand_b = td.get("actual_demand_backhaul", td["demand_backhaul"])
        reveal_times = td.get("reveal_times", torch.zeros_like(demand_l))

        current_time = torch.zeros(batch_size, dtype=torch.float32, device=td.device)
        curr_node = torch.zeros(batch_size, dtype=torch.int64, device=td.device)
        current_route_length = torch.zeros(batch_size, dtype=torch.float32, device=td.device)
        visited = torch.zeros((batch_size, n_loc + 1), dtype=torch.bool, device=td.device)
        used_cap_l = torch.zeros(batch_size, dtype=torch.float32, device=td.device)
        used_cap_b = torch.zeros(batch_size, dtype=torch.float32, device=td.device)

        for ii in range(actions.size(1)):
            next_node = actions[:, ii]
            curr_loc = gather_by_index(locs, curr_node)
            next_loc = gather_by_index(locs, next_node)
            dist = get_distance(curr_loc, next_loc)
            arrival = current_time + dist
            start_tw = gather_by_index(time_windows, next_node)[..., 0]
            end_tw = gather_by_index(time_windows, next_node)[..., 1]
            served_at = torch.max(arrival, start_tw)

            customer_mask = next_node != 0
            assert torch.all(
                (~customer_mask) | (served_at >= gather_by_index(reveal_times, next_node))
            ), "Visited hidden customer before reveal"

            hard_customer_check = (~allow_late) & customer_mask
            assert torch.all((~hard_customer_check) | (served_at <= end_tw)), "Late customer visit"

            current_time = served_at + gather_by_index(service_time, next_node)
            current_route_length = torch.where(
                customer_mask,
                current_route_length + dist,
                torch.zeros_like(current_route_length),
            )
            used_cap_l = torch.where(
                customer_mask,
                used_cap_l + gather_by_index(demand_l, next_node),
                torch.zeros_like(used_cap_l),
            )
            used_cap_b = torch.where(
                customer_mask,
                used_cap_b + gather_by_index(demand_b, next_node),
                torch.zeros_like(used_cap_b),
            )
            curr_node = next_node
            repeated_customer = customer_mask & visited.gather(1, next_node[:, None]).squeeze(1)
            assert (~repeated_customer).all(), "A customer was visited more than once"
            visited.scatter_(1, next_node[:, None], True)

            tmp = TensorDict(
                {
                    "actual_locs": locs,
                    "actual_time_windows": time_windows,
                    "actual_service_time": service_time,
                    "actual_demand_linehaul": demand_l,
                    "actual_demand_backhaul": demand_b,
                    "current_node": curr_node,
                    "current_time": current_time[:, None],
                    "current_route_length": current_route_length[:, None],
                    "speed": td["speed"],
                    "open_route": td["open_route"],
                    "distance_limit": td["distance_limit"],
                    "vehicle_capacity": td["vehicle_capacity"],
                    "used_capacity_linehaul": used_cap_l[:, None],
                    "used_capacity_backhaul": used_cap_b[:, None],
                    "backhaul_class": td["backhaul_class"],
                    "visited": visited,
                    "allow_late_customers": td["allow_late_customers"],
                    "hard_depot_time_window": td["hard_depot_time_window"],
                    "revealed": reveal_times <= current_time[:, None],
                    "reveal_times": reveal_times,
                },
                batch_size=td.batch_size,
                device=td.device,
            )
            tmp["revealed"][..., 0] = True
            can_visit = MTVRPDynamicEnv._compute_customer_feasibility(tmp)
            has_revealed_customer = can_visit[..., 1:].any(dim=-1)
            future_unvisited = (~tmp["revealed"]) & (~visited)
            future_unvisited[..., 0] = False
            need_wait = (~visited[..., 1:].all(dim=-1)) & (~has_revealed_customer) & future_unvisited.any(dim=-1)
            if need_wait.any():
                inf = torch.full_like(reveal_times, float("inf"))
                next_reveal = torch.where(future_unvisited, reveal_times, inf).min(dim=-1).values
                current_time = torch.where(need_wait, next_reveal, current_time)

    def _make_spec(self, td_params: TensorDict):
        self.observation_spec = Composite(
            locs=Bounded(
                low=self.generator.min_loc,
                high=self.generator.max_loc,
                shape=(self.generator.num_loc + 1, 2),
                dtype=torch.float32,
                device=self.device,
            ),
            current_node=UnboundedDiscrete(shape=(1), dtype=torch.int64, device=self.device),
            demand_linehaul=Bounded(
                low=-self.generator.capacity,
                high=self.generator.max_demand,
                shape=(self.generator.num_loc, 1),
                dtype=torch.float32,
                device=self.device,
            ),
            demand_backhaul=Bounded(
                low=-self.generator.capacity,
                high=self.generator.max_demand,
                shape=(self.generator.num_loc, 1),
                dtype=torch.float32,
                device=self.device,
            ),
            action_mask=UnboundedDiscrete(
                shape=(self.generator.num_loc + 1, 1),
                dtype=torch.bool,
                device=self.device,
            ),
            shape=(),
        )
        self.action_spec = Bounded(
            low=0,
            high=self.generator.num_loc + 1,
            shape=(1,),
            dtype=torch.int64,
            device=self.device,
        )
        self.reward_spec = UnboundedContinuous(shape=(1,), dtype=torch.float32, device=self.device)
        self.done_spec = UnboundedDiscrete(shape=(1,), dtype=torch.bool, device=self.device)

    def load_data(self, fpath, batch_size=[]):
        td = load_npz_to_tensordict(fpath)
        if self.load_solutions:
            solution_fpath = fpath.replace(".npz", self.solution_fname)
            if os.path.exists(solution_fpath):
                sol = np.load(solution_fpath)
                sol_dict = {}
                for key, value in sol.items():
                    if isinstance(value, np.ndarray) and len(value.shape) > 0:
                        if value.shape[0] == td.batch_size[0]:
                            key = "costs_bks" if key == "costs" else key
                            key = "actions_bks" if key == "actions" else key
                            sol_dict[key] = torch.tensor(value)
                td.update(sol_dict)
            else:
                log.warning(f"No solution file found at {solution_fpath}")
        return td


MTVRPFlexibleEnv = MTVRPDynamicEnv

import os

from typing import List, Optional, Union

import numpy as np
import torch

from rl4co.data.utils import load_npz_to_tensordict
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index, get_distance
from rl4co.utils.pylogger import get_pylogger
from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, UnboundedContinuous, UnboundedDiscrete

from routefinder.envs.mtvrp import MTVRPGenerator
from routefinder.envs.mtvrp import MTVRPEnv
from routefinder.envs.mtdvrp.selectstartnodes import get_select_start_nodes_fn

log = get_pylogger(__name__)


class MTVRPFlexibleEnv(MTVRPEnv):
    r"""MTVRPEnv is a Multi-Task VRP environment which can take any combination of the following constraints:

    Features:

    - *Capacity (C)*
        - Each vehicle has a maximum capacity :math:`Q`, restricting the total load that can be in the vehicle at any point of the route.
        - The route must be planned such that the sum of demands and pickups for all customers visited does not exceed this capacity.
    - *Time Windows (TW)*
        - Every node :math:`i` has an associated time window :math:`[e_i, l_i]` during which service must commence.
        - Additionally, each node has a service time :math:`s_i`. Vehicles must reach node :math:`i` within its time window; early arrivals must wait at the node location until time :math:`e_i`.
    - *Open Routes (O)*
        - Vehicles are not required to return to the depot after serving all customers.
        - Note that this does not need to be counted as a constraint since it can be modelled by setting zero costs on arcs returning to the depot :math:`c_{i0} = 0` from any customer :math:`i \in C`, and not counting the return arc as part of the route.
    - *Backhauls (B)*
        - Backhauls generalize demand to also account for return shipments. Customers are either linehaul or backhaul customers.
        - Linehaul customers require delivery of a demand :math:`q_i > 0` that needs to be transported from the depot to the customer, whereas backhaul customers need a pickup of an amount :math:`p_i > 0` that is transported from the client back to the depot.
        - It is possible for vehicles to serve a combination of linehaul and backhaul customers in a single route, but then any linehaul customers must precede the backhaul customers in the route.
    - *Duration Limits (L)*
        - Imposes a limit on the total travel duration (or length) of each route, ensuring a balanced workload across vehicles.
    - *Mixed (M) Backhaul (M)*
        - This is a variant of the backhaul constraint where the vehicle can pick up and deliver linehaul customers in any order.
        - However, we need to ensure that the vehicle has enough capacity to deliver the linehaul customers and that the vehicle can pick up backhaul customers only if it has enough capacity to deliver the linehaul customers.

    The environment covers the following 16 variants depending on the data generation:

    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | VRP Variant  || Capacity (C) | Open Route (O) | Backhaul (B) | Duration Limit (L) | Time Window (TW) |
    +==============++==============+================+==============+====================+==================+
    | CVRP         || ✔            |                |              |                    |                  |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | OVRP         || ✔            | ✔              |              |                    |                  |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | VRPB         || ✔            |                | ✔            |                    |                  |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | VRPL         || ✔            |                |              | ✔                  |                  |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | VRPTW        || ✔            |                |              |                    | ✔                |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | OVRPTW       || ✔            | ✔              |              |                    | ✔                |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | OVRPB        || ✔            | ✔              | ✔            |                    |                  |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | OVRPL        || ✔            | ✔              |              | ✔                  |                  |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | VRPBL        || ✔            |                | ✔            | ✔                  |                  |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | VRPBTW       || ✔            |                | ✔            |                    | ✔                |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | VRPLTW       || ✔            |                |              | ✔                  | ✔                |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | OVRPBL       || ✔            | ✔              | ✔            | ✔                  |                  |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | OVRPBTW      || ✔            | ✔              | ✔            |                    | ✔                |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | OVRPLTW      || ✔            | ✔              |              | ✔                  | ✔                |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | VRPBLTW      || ✔            |                | ✔            | ✔                  | ✔                |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | OVRPBLTW     || ✔            | ✔              | ✔            | ✔                  | ✔                |
    +--------------++--------------+----------------+--------------+--------------------+------------------+

    Additionally, with the mixed backhaul (M) variant, we obtain 24 variants.

    You may also check out `"Multi-Task Learning for Routing Problem with Cross-Problem Zero-Shot Generalization" (Liu et al., 2024) <https://arxiv.org/abs/2402.16891>`_
    and `"MVMoE: Multi-Task Vehicle Routing Solver with Mixture-of-Experts" (Zhou et al, 2024) <https://arxiv.org/abs/2405.01029>`_.


    Note:
        Have a look at https://pyvrp.org/ for more information about VRP and its variants and their solutions. Kudos to their help and great job!

    Args:
        generator: Generator for the environment, see :class:`MTVRPGenerator`.
        generator_params: Parameters for the generator.
    """

    name = "mtvrp"

    def __init__(
        self,
        generator: MTVRPGenerator = None,
        generator_params: dict = {},
        select_start_nodes_fn: Union[str, callable] = "all",
        check_solution: bool = False,
        load_solutions: bool = True,
        solution_fname: str = "_sol_pyvrp.npz",
        allow_late_customers: bool = False,
        lateness_penalty: float = 0.0,
        allow_reject_customers: bool = False,
        reject_penalty: float = 0.0,
        hard_depot_time_window: bool = True,
        **kwargs,
    ):
        super().__init__(check_solution=check_solution, **kwargs)
        if generator is None:
            generator = MTVRPGenerator(**generator_params)

        if check_solution:
            log.warning(
                "Solution checking is enabled. This may slow down the environment."
                " We recommend disabling this for training by passing `check_solution=False`."
            )

        self.generator = generator
        print(select_start_nodes_fn)
        # if isinstance(select_start_nodes_fn, str):
        #     self.select_start_nodes_fn = get_select_start_nodes_fn(select_start_nodes_fn)
        # else:
        #     self.select_start_nodes_fn = select_start_nodes_fn

        self.solution_fname = solution_fname
        self.load_solutions = load_solutions

        # New flexibility knobs:
        # - allow_late_customers: time windows become soft for customers
        # - allow_reject_customers: depot->depot while at the depot ends the episode early
        #   and all still-unserved customers receive reject_penalty
        # - hard_depot_time_window: keep the depot closing time as a hard constraint


        self.allow_late_customers = allow_late_customers
        self.lateness_penalty = float(lateness_penalty)
        self.allow_reject_customers = allow_reject_customers
        self.reject_penalty = float(reject_penalty)
        self.hard_depot_time_window = hard_depot_time_window

        self._make_spec(self.generator)

    def _step(self, td: TensorDict) -> TensorDict:
        # Get locations and distance
        prev_node, curr_node = td["current_node"], td["action"]
        prev_loc = gather_by_index(td["locs"], prev_node)
        curr_loc = gather_by_index(td["locs"], curr_node)
        distance = get_distance(prev_loc, curr_loc)[..., None]

        # Update current time
        service_time = gather_by_index(
            src=td["service_time"], idx=curr_node, dim=1, squeeze=False
        )
        start_times = gather_by_index(
            src=td["time_windows"], idx=curr_node, dim=1, squeeze=False
        )[..., 0]
        late_times = gather_by_index(
            src=td["time_windows"], idx=curr_node, dim=1, squeeze=False
        )[..., 1]

        arrival_time = td["current_time"] + distance / td["speed"]
        service_start = torch.max(arrival_time, start_times)

        # we cannot start before we arrive and we should start at least at start times
        curr_time = (curr_node[:, None] != 0) * (service_start + service_time)

        # Accumulate tardiness only for customer visits
        step_tardiness = (curr_node[:, None] != 0).float() * torch.clamp(
            service_start - late_times, min=0.0
        )
        total_tardiness = td["total_tardiness"] + step_tardiness

        # Update current route length (reset at depot)
        curr_route_length = (curr_node[:, None] != 0) * (
            td["current_route_length"] + distance
        )

        # Linehaul (delivery) demands
        selected_demand_linehaul = gather_by_index(
            td["demand_linehaul"], curr_node, dim=1, squeeze=False
        )
        selected_demand_backhaul = gather_by_index(
            td["demand_backhaul"], curr_node, dim=1, squeeze=False
        )

        # Backhaul (pickup) demands
        # this holds for backhaul_classes 0, 1, and 2:
        used_capacity_linehaul = (curr_node[:, None] != 0) * (
            td["used_capacity_linehaul"] + selected_demand_linehaul
        )
        used_capacity_backhaul = (curr_node[:, None] != 0) * (
            td["used_capacity_backhaul"] + selected_demand_backhaul
        )

        visited = td["visited"].scatter(-1, curr_node[..., None], True)
        all_customers_visited = visited[..., 1:].all(dim=-1)

        # Optional-customer mode:
        # depot->depot while already at the depot ends the episode early.
        early_stop = td["allow_reject_customers"].squeeze(-1) & (prev_node == 0) & (curr_node == 0)
        done = all_customers_visited | early_stop

        reward = torch.zeros_like(done).float()

        td.update(
            {
                "current_node": curr_node,
                "current_route_length": curr_route_length,
                "current_time": curr_time,
                "done": done,
                "reward": reward,
                "used_capacity_linehaul": used_capacity_linehaul,
                "used_capacity_backhaul": used_capacity_backhaul,
                "visited": visited,
                "total_tardiness": total_tardiness,
            }
        )
        td.set("action_mask", self.get_action_mask(td))
        return td


    def _reset(
        self,
        td: Optional[TensorDict],
        batch_size: Optional[list] = None,
    ) -> TensorDict:
        # if td is None or batch_size is None:
        #     return TensorDict({}, batch_size=batch_size, device=self.device)
        device = td.device

        # Demands: linehaul (C) and backhaul (B). Backhaul defaults to 0
        demand_linehaul = torch.cat(
            [torch.zeros_like(td["demand_linehaul"][..., :1]), td["demand_linehaul"]],
            dim=1,
        )
        demand_backhaul = td.get(
            "demand_backhaul",
            torch.zeros_like(td["demand_linehaul"]),
        )
        demand_backhaul = torch.cat(
            [torch.zeros_like(td["demand_linehaul"][..., :1]), demand_backhaul], dim=1
        )
        # Backhaul class (MB). 1 is the default backhaul class
        backhaul_class = td.get(
            "backhaul_class",
            torch.full((*batch_size, 1), 1, dtype=torch.int32),
        )

        # Time windows (TW). Defaults to [0, inf] and service time to 0
        time_windows = td.get("time_windows", None)
        if time_windows is None:
            time_windows = torch.zeros_like(td["locs"])
            time_windows[..., 1] = float("inf")
        service_time = td.get("service_time", torch.zeros_like(demand_linehaul))

        # Open (O) route. Defaults to 0
        open_route = td.get(
            "open_route", torch.zeros_like(demand_linehaul[..., :1], dtype=torch.bool)
        )

        # Distance limit (L). Defaults to inf
        distance_limit = td.get(
            "distance_limit", torch.full_like(demand_linehaul[..., :1], float("inf"))
        )

        td_reset = TensorDict(
            {
                "locs": td["locs"],
                "demand_backhaul": demand_backhaul,
                "demand_linehaul": demand_linehaul,
                "backhaul_class": backhaul_class,
                "distance_limit": distance_limit,
                "service_time": service_time,
                "open_route": open_route,
                "time_windows": time_windows,
                "speed": td.get("speed", torch.ones_like(demand_linehaul[..., :1])),
                "vehicle_capacity": td.get(
                    "vehicle_capacity", torch.ones_like(demand_linehaul[..., :1])
                ),
                "capacity_original": td.get(
                    "capacity_original", torch.ones_like(demand_linehaul[..., :1])
                ),
                "current_node": torch.zeros(
                    (*batch_size,), dtype=torch.long, device=device
                ),
                "current_route_length": torch.zeros(
                    (*batch_size, 1), dtype=torch.float32, device=device
                ),
                "current_time": torch.zeros(
                    (*batch_size, 1), dtype=torch.float32, device=device
                ),
                "used_capacity_backhaul": torch.zeros(
                    (*batch_size, 1), device=device
                ),
                "used_capacity_linehaul": torch.zeros(
                    (*batch_size, 1), device=device
                ),
                "visited": torch.zeros(
                    (*batch_size, td["locs"].shape[-2]),
                    dtype=torch.bool,
                    device=device,
                ),
                "total_tardiness": torch.zeros(
                    (*batch_size, 1), dtype=torch.float32, device=device
                ),
                "allow_late_customers": torch.full(
                    (*batch_size, 1),
                    self.allow_late_customers,
                    dtype=torch.bool,
                    device=device,
                ),
                "lateness_penalty": torch.full(
                    (*batch_size, 1),
                    self.lateness_penalty,
                    dtype=torch.float32,
                    device=device,
                ),
                "allow_reject_customers": torch.full(
                    (*batch_size, 1),
                    self.allow_reject_customers,
                    dtype=torch.bool,
                    device=device,
                ),
                "reject_penalty": torch.full(
                    (*batch_size, 1),
                    self.reject_penalty,
                    dtype=torch.float32,
                    device=device,
                ),
                "hard_depot_time_window": torch.full(
                    (*batch_size, 1),
                    self.hard_depot_time_window,
                    dtype=torch.bool,
                    device=device,
                ),
            },
            batch_size=batch_size,
            device=device,
        )
        # print("td_reset keys:", td_reset.keys())
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset


    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor: # type: ignore
        curr_node = td["current_node"]  # note that this was just updated!
        locs = td["locs"]
        d_ij = get_distance(
            gather_by_index(locs, curr_node)[..., None, :], locs
        )  # i (current) -> j (next)
        d_j0 = get_distance(locs, locs[..., 0:1, :])  # j (next) -> 0 (depot)

        early_tw, late_tw = (
            td["time_windows"][..., 0],
            td["time_windows"][..., 1],
        )
        arrival_time = td["current_time"] + (d_ij / td["speed"])

        # Customer time windows can optionally be made soft.
        if td["allow_late_customers"].any():
            can_reach_customer = torch.ones_like(arrival_time, dtype=torch.bool)
        else:
            can_reach_customer = arrival_time < late_tw

        # Keep the depot-closing horizon hard unless explicitly disabled.
        if td["hard_depot_time_window"].any():
            can_reach_depot = (
                torch.max(arrival_time, early_tw) + td["service_time"] + (d_j0 / td["speed"])
            ) * ~td["open_route"] < late_tw[..., 0:1]
        else:
            can_reach_depot = torch.ones_like(arrival_time, dtype=torch.bool)

        # Distance limit (L): do not add distance to depot if open route (O)
        exceeds_dist_limit = (
            td["current_route_length"] + d_ij + (d_j0 * ~td["open_route"])
            > td["distance_limit"]
        )

        # Capacity constraints linehaul (C) and backhaul (B)
        exceeds_cap_linehaul = (
            td["demand_linehaul"] + td["used_capacity_linehaul"] > td["vehicle_capacity"]
        )
        exceeds_cap_backhaul = (
            td["demand_backhaul"] + td["used_capacity_backhaul"] > td["vehicle_capacity"]
        )

        # Backhaul class 1 (classical backhaul) (B)
        linehauls_missing = ((td["demand_linehaul"] * ~td["visited"]).sum(-1) > 0)[
            ..., None
        ]
        is_carrying_backhaul = (
            gather_by_index(
                src=td["demand_backhaul"],
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
            & (td["demand_linehaul"] > 0)
        ) | (~exceeds_cap_backhaul & (td["demand_backhaul"] > 0))

        # Backhaul class 2 (mixed pickup and delivery / mixed backhaul) (MB)
        cannot_serve_linehaul = (
            td["demand_linehaul"] > td["vehicle_capacity"] - td["used_capacity_backhaul"]
        )
        meets_demand_constraint_backhaul_2 = (
            ~exceeds_cap_linehaul & ~exceeds_cap_backhaul & ~cannot_serve_linehaul
        )

        meets_demand_constraint = (
            (td["backhaul_class"] == 1) & meets_demand_constraint_backhaul_1
        ) | ((td["backhaul_class"] == 2) & meets_demand_constraint_backhaul_2)

        can_visit = (
            can_reach_customer
            & can_reach_depot
            & meets_demand_constraint
            & ~exceeds_dist_limit
            & ~td["visited"]
        )

        # Depot handling:
        # - classical mode: same as upstream
        # - optional-customer mode: depot->depot is allowed as an explicit end-episode action
        if td["allow_reject_customers"].any():
            can_visit[:, 0] = True
        else:
            can_visit[:, 0] = ~((curr_node == 0) & (can_visit[:, 1:].sum(-1) > 0))

        return can_visit


    def _get_reward(self, td: TensorDict, actions: TensorDict) -> TensorDict:
        # Append depot to actions and get sequence of locations
        go_from = torch.cat((torch.zeros_like(actions[:, :1]), actions), dim=1)
        go_to = torch.roll(go_from, -1, dims=1)
        loc_from = gather_by_index(td["locs"], go_from)
        loc_to = gather_by_index(td["locs"], go_to)

        distances = get_distance(loc_from, loc_to)
        tour_length = (distances * ~((go_to == 0) & td["open_route"])).sum(-1)

        total_tardiness = td.get(
            "total_tardiness",
            torch.zeros_like(tour_length[..., None]),
        ).squeeze(-1)
        lateness_cost = td.get(
            "lateness_penalty",
            torch.zeros_like(total_tardiness[..., None]),
        ).squeeze(-1) * total_tardiness

        unserved = (~td["visited"][..., 1:]).sum(-1).float()
        reject_cost = td.get(
            "reject_penalty",
            torch.zeros_like(unserved[..., None]),
        ).squeeze(-1) * unserved * td.get(
            "allow_reject_customers",
            torch.zeros_like(unserved[..., None], dtype=torch.bool),
        ).squeeze(-1).float()

        return -(tour_length + lateness_cost + reject_cost)


    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor):
        batch_size, n_loc = td["demand_linehaul"].size()
        locs = td["locs"]
        n_loc -= 1  # exclude depot

        allow_reject = td.get(
            "allow_reject_customers",
            torch.zeros((batch_size, 1), dtype=torch.bool, device=td.device),
        ).squeeze(-1)
        allow_late = td.get(
            "allow_late_customers",
            torch.zeros((batch_size, 1), dtype=torch.bool, device=td.device),
        ).squeeze(-1)
        hard_depot_time_window = td.get(
            "hard_depot_time_window",
            torch.ones((batch_size, 1), dtype=torch.bool, device=td.device),
        ).squeeze(-1)

        # Customer coverage:
        # - classic mode: every customer exactly once
        # - reject mode: each served customer at most once; some may be absent
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

        # Distance limits (L)
        assert (td["distance_limit"] >= 0).all(), "Distance limits must be non-negative."

        # Time windows (TW)
        d_j0 = get_distance(locs, locs[..., 0:1, :])  # j (next) -> 0 (depot)
        assert torch.all(td["time_windows"] >= 0.0), "Time windows must be non-negative."
        assert torch.all(td["service_time"] >= 0.0), "Service time must be non-negative."
        assert torch.all(
            td["time_windows"][..., 0] < td["time_windows"][..., 1]
        ), "there are unfeasible time windows"

        if hard_depot_time_window.any():
            assert torch.all(
                td["time_windows"][..., :, 0] + d_j0 + td["service_time"]
                <= td["time_windows"][..., 0, 1, None]
            ), "vehicle cannot perform service and get back to depot in time."

        curr_time = torch.zeros(batch_size, dtype=torch.float32, device=td.device)
        curr_node = torch.zeros(batch_size, dtype=torch.int64, device=td.device)
        curr_length = torch.zeros(batch_size, dtype=torch.float32, device=td.device)
        seen = torch.zeros((batch_size, n_loc + 1), dtype=torch.bool, device=td.device)
        seen[:, 0] = True

        for ii in range(actions.size(1)):
            next_node = actions[:, ii]
            curr_loc = gather_by_index(td["locs"], curr_node)
            next_loc = gather_by_index(td["locs"], next_node)
            dist = get_distance(curr_loc, next_loc)

            curr_length = curr_length + dist * ~(
                td["open_route"].squeeze(-1) & (next_node == 0)
            )
            assert torch.all(
                curr_length <= td["distance_limit"].squeeze(-1)
            ), "Route exceeds distance limit"
            curr_length[next_node == 0] = 0.0

            start_tw = gather_by_index(td["time_windows"], next_node)[..., 0]
            end_tw = gather_by_index(td["time_windows"], next_node)[..., 1]
            curr_time = torch.max(curr_time + dist, start_tw)

            hard_customer_check = (~allow_late) & (next_node != 0)
            assert torch.all(
                (~hard_customer_check) | (curr_time <= end_tw)
            ), "vehicle cannot start customer service before deadline"

            hard_depot_check = hard_depot_time_window & (next_node == 0)
            assert torch.all(
                (~hard_depot_check) | (curr_time <= end_tw)
            ), "vehicle cannot return to depot before depot deadline"

            curr_time = curr_time + gather_by_index(td["service_time"], next_node)
            curr_node = next_node
            curr_time[curr_node == 0] = 0.0

            repeated_customer = (next_node != 0) & seen.gather(1, next_node[:, None]).squeeze(1)
            assert (~repeated_customer).all(), "A customer was visited more than once"
            seen.scatter_(1, next_node[:, None], True)

        demand_l = td["demand_linehaul"].gather(dim=1, index=actions)
        demand_b = td["demand_backhaul"].gather(dim=1, index=actions)
        used_cap_l = torch.zeros_like(td["demand_linehaul"][:, 0])
        used_cap_b = torch.zeros_like(td["demand_backhaul"][:, 0])
        for ii in range(actions.size(1)):
            used_cap_l = used_cap_l * (actions[:, ii] != 0)
            used_cap_b = used_cap_b * (actions[:, ii] != 0)
            used_cap_l += demand_l[:, ii]
            used_cap_b += demand_b[:, ii]

            assert (
                (td["backhaul_class"] == 2)
                | (used_cap_b == 0)
                | ((td["backhaul_class"] == 1) & ~(demand_l[:, ii] > 0))
            ).all(), "Cannot pick up linehaul while carrying backhaul due to precedence constraints"

            assert (
                (td["backhaul_class"] == 1)
                | (used_cap_b == 0)
                | (
                    (td["backhaul_class"] == 2)
                    & (used_cap_b + demand_l[:, ii] <= td["vehicle_capacity"])
                )
            ).all(), "Cannot deliver linehaul, not enough load"

            assert (
                used_cap_l <= td["vehicle_capacity"]
            ).all(), "Used more linehaul than capacity: {} / {}".format(
                used_cap_l, td["vehicle_capacity"]
            )
            assert (
                used_cap_b <= td["vehicle_capacity"]
            ).all(), "Used more backhaul than capacity: {} / {}".format(
                used_cap_b, td["vehicle_capacity"]
            )


    def get_num_starts(self, td):
        return self.select_start_nodes_fn.get_num_starts(td)

    def select_start_nodes(self, td, num_starts):
        return self.select_start_nodes_fn(td, num_starts, self.get_num_starts(td))

    @staticmethod
    def render(*args, **kwargs):
        """Simple wrapper for render function"""
        from routefinder.envs.mtvrp.render import render

        return render(*args, **kwargs)

    def _make_spec(self, td_params: TensorDict):
        # TODO: include extra vars (but we don't really need them for now)
        """Make the observation and action specs from the parameters."""
        self.observation_spec = Composite(
            locs=Bounded(
                low=self.generator.min_loc,
                high=self.generator.max_loc,
                shape=(self.generator.num_loc + 1, 2),
                dtype=torch.float32,
                device=self.device,
            ),
            current_node=UnboundedDiscrete(
                shape=(1),
                dtype=torch.int64,
                device=self.device,
            ),
            demand_linehaul=Bounded(
                low=-self.generator.capacity,
                high=self.generator.max_demand,
                shape=(self.generator.num_loc, 1),  # demand is only for customers
                dtype=torch.float32,
                device=self.device,
            ),
            demand_backhaul=Bounded(
                low=-self.generator.capacity,
                high=self.generator.max_demand,
                shape=(self.generator.num_loc, 1),  # demand is only for customers
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
        self.reward_spec = UnboundedContinuous(
            shape=(1,), dtype=torch.float32, device=self.device
        )
        self.done_spec = UnboundedDiscrete(
            shape=(1,), dtype=torch.bool, device=self.device
        )

    @staticmethod
    def check_variants(td):
        """Check if the problem has the variants"""
        has_open = td["open_route"].squeeze(-1)
        has_tw = (td["time_windows"][..., 1] != float("inf")).any(-1)
        has_limit = (td["distance_limit"] != float("inf")).squeeze(-1)
        has_backhaul = (td["demand_backhaul"] != 0).any(-1)
        backhaul_class = td.get("backhaul_class", torch.full_like(has_open, 1))
        return has_open, has_tw, has_limit, has_backhaul, backhaul_class

    @staticmethod
    def get_variant_names(td: TensorDict) -> Union[str, List[str]]:
        (
            has_open,
            has_time_window,
            has_duration_limit,
            has_backhaul,
            backhaul_class,
        ) = MTVRPEnv.check_variants(td)

        def _name(o, b, bc, l_, tw):
            if not o and not b and not l_ and not tw:
                instance_name = "CVRP"
            else:
                instance_name = "VRP"
                if o:
                    instance_name = "O" + instance_name
                if b:
                    if bc == 2:  # mixed backhaul
                        instance_name += "M"
                    instance_name += "B"
                if l_:
                    instance_name += "L"
                if tw:
                    instance_name += "TW"
            return instance_name

        if len(has_open.shape) == 0:
            return _name(
                has_open,
                has_backhaul,
                backhaul_class,
                has_duration_limit,
                has_time_window,
            )
        else:
            return [
                _name(o, b, bc, l_, tw)
                for o, b, bc, l_, tw in zip(
                    has_open,
                    has_backhaul,
                    backhaul_class,
                    has_duration_limit,
                    has_time_window,
                )
            ]

    def print_presets(self):
        self.generator.print_presets()

    def available_variants(self):
        return self.generator.available_variants()

    def load_data(self, fpath, batch_size=[]):
        """Dataset loading from file"""
        td = load_npz_to_tensordict(fpath)
        if self.load_solutions:
            # Load solutions if they exist depending on the file name
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

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from tensordict.tensordict import TensorDict
from torch import Tensor
from torchrl.data import Bounded, Composite, Unbounded

from rl4co.envs.common import Generator, RL4COEnvBase


@dataclass
class DCVRPGeneratorParams:
    num_loc: int = 20
    min_loc: float = 0.0
    max_loc: float = 1.0
    min_demand: int = 1
    max_demand: int = 9
    vehicle_capacity: int = 30
    dynamic_ratio: float = 0.5
    max_disclosure_time: float = 6.4


class DCVRPGenerator(Generator):
    """Random DCVRP instance generator.

    Output keys:
      - depot: (B, 1, 2)
      - locs: (B, N, 2)
      - demands: (B, N) normalized to [0, 1]
      - disclosure_times: (B, N)
      - capacity: (B, 1) fixed to 1.0 after normalization
    """

    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        min_demand: int = 1,
        max_demand: int = 9,
        vehicle_capacity: int = 30,
        dynamic_ratio: float = 0.5,
        max_disclosure_time: float = 6.4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.min_demand = min_demand
        self.max_demand = max_demand
        self.vehicle_capacity = vehicle_capacity
        self.dynamic_ratio = dynamic_ratio
        self.max_disclosure_time = max_disclosure_time

    def _generate(self, batch_size, **kwargs) -> TensorDict:
        device = kwargs.get("device", None)
        bs = [batch_size] if isinstance(batch_size, int) else list(batch_size)

        depot = torch.rand(*bs, 1, 2, device=device)
        depot = depot * (self.max_loc - self.min_loc) + self.min_loc

        locs = torch.rand(*bs, self.num_loc, 2, device=device)
        locs = locs * (self.max_loc - self.min_loc) + self.min_loc

        demands = torch.randint(
            low=self.min_demand,
            high=self.max_demand + 1,
            size=(*bs, self.num_loc),
            device=device,
        ).float()
        demands = demands / float(self.vehicle_capacity)

        is_dynamic = torch.rand(*bs, self.num_loc, device=device) < self.dynamic_ratio
        sampled_t = torch.rand(*bs, self.num_loc, device=device) * self.max_disclosure_time
        disclosure_times = torch.where(is_dynamic, sampled_t, torch.zeros_like(sampled_t))

        return TensorDict(
            {
                "depot": depot,
                "locs": locs,
                "demands": demands,
                "disclosure_times": disclosure_times,
                "capacity": torch.ones(*bs, 1, device=device),
            },
            batch_size=bs,
        )


class DCVRPEnv(RL4COEnvBase):
    """Dynamic CVRP environment approximating the OpenReview DCVRP paper setup.

    Design choices:
      - one active vehicle that may revisit the depot to refill capacity;
      - some customers are hidden until disclosure time;
      - if no disclosed feasible customer exists, time advances to the next disclosure;
      - action 0 is the depot; actions 1..N are customers.
    """

    name = "dcvrp"

    def __init__(self, generator: Optional[Generator] = None, generator_params: Optional[dict] = None, **kwargs):
        super().__init__(**kwargs)
        if generator is None:
            generator_params = generator_params or {}
            generator = DCVRPGenerator(**generator_params)
        self.generator = generator
        self._make_spec(self.generator)

    def _make_spec(self, generator: DCVRPGenerator):
        n = generator.num_loc
        self.observation_spec = Composite(
            depot=Bounded(
                low=generator.min_loc,
                high=generator.max_loc,
                shape=(1, 2),
                dtype=torch.float32,
            ),
            locs=Bounded(
                low=generator.min_loc,
                high=generator.max_loc,
                shape=(n, 2),
                dtype=torch.float32,
            ),
            demands=Bounded(
                low=0.0,
                high=1.0,
                shape=(n,),
                dtype=torch.float32,
            ),
            disclosure_times=Unbounded(shape=(n,), dtype=torch.float32),
            capacity=Bounded(low=1.0, high=1.0, shape=(1,), dtype=torch.float32),
            current_node=Unbounded(shape=(1,), dtype=torch.int64),
            current_time=Unbounded(shape=(1,), dtype=torch.float32),
            used_capacity=Bounded(low=0.0, high=1.0, shape=(1,), dtype=torch.float32),
            visited=Unbounded(shape=(n,), dtype=torch.bool),
            done=Unbounded(shape=(1,), dtype=torch.bool),
            i=Unbounded(shape=(1,), dtype=torch.int64),
            action_mask=Unbounded(shape=(n + 1,), dtype=torch.bool),
            reward=Unbounded(shape=(1,), dtype=torch.float32),
            shape=(),
        )
        self.action_spec = Bounded(shape=(1,), dtype=torch.int64, low=0, high=n + 1)
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32)
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool)

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        if td is None:
            td = self.generator(batch_size=batch_size)

        bs = td.batch_size
        device = td.device
        out = TensorDict(
            {
                **td,
                "current_node": torch.zeros(*bs, 1, dtype=torch.int64, device=device),
                "current_time": torch.zeros(*bs, 1, dtype=torch.float32, device=device),
                "used_capacity": torch.zeros(*bs, 1, dtype=torch.float32, device=device),
                "visited": torch.zeros(*bs, self.generator.num_loc, dtype=torch.bool, device=device),
                "done": torch.zeros(*bs, 1, dtype=torch.bool, device=device),
                "i": torch.zeros(*bs, 1, dtype=torch.int64, device=device),
                "reward": torch.zeros(*bs, 1, dtype=torch.float32, device=device),
            },
            batch_size=bs,
        )
        out.set("action_mask", self.get_action_mask(out))
        return out

    def _step(self, td: TensorDict) -> TensorDict:
        action = td["action"]
        if action.dim() == td["current_node"].dim() - 1:
            action = action.unsqueeze(-1)

        depot = td["depot"]
        locs = td["locs"]
        demands = td["demands"]
        disclosure_times = td["disclosure_times"]
        current_node = td["current_node"]
        current_time = td["current_time"]
        used_capacity = td["used_capacity"]
        visited = td["visited"]
        step_idx = td["i"]

        all_nodes = torch.cat([depot, locs], dim=1)
        cur_xy = all_nodes.gather(1, current_node[..., None].expand(-1, -1, 2)).squeeze(1)
        nxt_xy = all_nodes.gather(1, action[..., None].expand(-1, -1, 2)).squeeze(1)
        travel = torch.norm(nxt_xy - cur_xy, dim=-1, keepdim=True)

        new_time = current_time + travel
        new_used_capacity = used_capacity.clone()
        new_visited = visited.clone()

        is_depot = action.squeeze(-1) == 0
        cust_idx = action.squeeze(-1) - 1
        customer_mask = ~is_depot
        if customer_mask.any():
            batch_ids = torch.arange(action.size(0), device=action.device)[customer_mask]
            selected = cust_idx[customer_mask]
            selected_demand = demands[batch_ids, selected].unsqueeze(-1)
            new_used_capacity[customer_mask] = new_used_capacity[customer_mask] + selected_demand
            new_visited[batch_ids, selected] = True

        new_used_capacity[is_depot] = 0.0

        disclosed = disclosure_times <= new_time
        feasible_cap = demands <= (1.0 - new_used_capacity)
        available = disclosed & (~new_visited) & feasible_cap

        all_served = new_visited.all(dim=-1, keepdim=True)
        none_available = ~available.any(dim=-1, keepdim=True)
        need_wait = (~all_served) & none_available
        if need_wait.any():
            inf = torch.full_like(disclosure_times, float("inf"))
            future_mask = (~disclosed) & (~new_visited)
            next_disclosure = torch.where(future_mask, disclosure_times, inf).min(dim=-1, keepdim=True).values
            next_disclosure = torch.where(torch.isinf(next_disclosure), new_time, next_disclosure)
            new_time = torch.where(need_wait, next_disclosure, new_time)

        done = all_served & (action == 0)
        # If all served immediately after a customer, force one final depot return in the mask.
        # This keeps decoding consistent with CVRP-style route closure.

        next_td = td.clone()
        next_td.update(
            {
                "current_node": action,
                "current_time": new_time,
                "used_capacity": new_used_capacity,
                "visited": new_visited,
                "done": done,
                "i": step_idx + 1,
                "reward": torch.zeros_like(td["reward"]),
            }
        )
        next_td.set("action_mask", self.get_action_mask(next_td))
        return next_td

    def get_action_mask(self, td: TensorDict) -> Tensor:
        demands = td["demands"]
        disclosure_times = td["disclosure_times"]
        current_time = td["current_time"]
        used_capacity = td["used_capacity"]
        visited = td["visited"]
        current_node = td["current_node"].squeeze(-1)

        disclosed = disclosure_times <= current_time
        cap_ok = demands <= (1.0 - used_capacity)
        customer_ok = disclosed & (~visited) & cap_ok

        all_served = visited.all(dim=-1, keepdim=True)
        any_customer_ok = customer_ok.any(dim=-1, keepdim=True)

        # Depot is allowed unless we are already at depot while there is an available customer.
        depot_ok = (~((current_node == 0).unsqueeze(-1) & any_customer_ok))
        # Once all customers are served, only the depot should be available.
        depot_ok = depot_ok | all_served
        customer_ok = torch.where(all_served, torch.zeros_like(customer_ok), customer_ok)

        mask = torch.cat([depot_ok, customer_ok], dim=-1)
        return mask

    def check_solution_validity(self, td: TensorDict, actions: torch.Tensor):
        n = td["locs"].size(1)
        customers = actions[actions > 0] - 1
        # This method is mostly informative; strict batched per-instance checking is done in _get_reward.
        assert customers.numel() >= n, "Decoded sequence is too short to cover all customers."

    def _get_reward(self, td: TensorDict, actions: Tensor) -> Tensor:
        """Simulate the decoded route and return negative travel length.

        `actions` is expected to contain depot/customer indices in [0, N].
        """
        depot = td["depot"]
        locs = td["locs"]
        demands = td["demands"]
        disclosure_times = td["disclosure_times"]
        device = locs.device

        if actions.dim() == 1:
            actions = actions.unsqueeze(0)

        batch_size, seq_len = actions.shape
        all_nodes = torch.cat([depot, locs], dim=1)

        total_length = torch.zeros(batch_size, device=device)
        current_node = torch.zeros(batch_size, dtype=torch.long, device=device)
        current_time = torch.zeros(batch_size, device=device)
        used_capacity = torch.zeros(batch_size, device=device)
        visited = torch.zeros(batch_size, locs.size(1), dtype=torch.bool, device=device)

        for t in range(seq_len):
            a = actions[:, t]
            cur_xy = all_nodes[torch.arange(batch_size, device=device), current_node]
            nxt_xy = all_nodes[torch.arange(batch_size, device=device), a]
            travel = torch.norm(nxt_xy - cur_xy, dim=-1)
            total_length = total_length + travel
            current_time = current_time + travel

            is_customer = a > 0
            if is_customer.any():
                cidx = a[is_customer] - 1
                bidx = torch.arange(batch_size, device=device)[is_customer]

                # Feasibility checks
                assert (~visited[bidx, cidx]).all(), "Visited the same customer more than once"
                assert (disclosure_times[bidx, cidx] <= current_time[bidx]).all(), "Visited undisclosed customer"
                assert (used_capacity[bidx] + demands[bidx, cidx] <= 1.0 + 1e-6).all(), "Capacity exceeded"

                visited[bidx, cidx] = True
                used_capacity[bidx] = used_capacity[bidx] + demands[bidx, cidx]

            is_depot = a == 0
            used_capacity[is_depot] = 0.0
            current_node = a

            # Waiting semantics: if not all served and no feasible disclosed customer exists, time jumps.
            disclosed = disclosure_times <= current_time[:, None]
            feasible_cap = demands <= (1.0 - used_capacity[:, None])
            available = disclosed & (~visited) & feasible_cap
            all_served = visited.all(dim=-1)
            need_wait = (~all_served) & (~available.any(dim=-1))
            if need_wait.any():
                inf = torch.full_like(disclosure_times, float("inf"))
                future_mask = (~disclosed) & (~visited)
                next_disclosure = torch.where(future_mask, disclosure_times, inf).min(dim=-1).values
                current_time = torch.where(need_wait, next_disclosure, current_time)

        assert visited.all(), "Not all customers were visited"
        assert (current_node == 0).all(), "Route must end at depot"

        return -total_length.unsqueeze(-1)

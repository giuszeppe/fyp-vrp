from __future__ import annotations

import numpy as np
import torch
from routefinder.envs.mtvrp import MTVRPGenerator
from tensordict import TensorDict


class DynamicGenerator(MTVRPGenerator):
    def __init__(
        self,
        solomon_instances,
        num_loc: int = 20,
        dod: float = 0.3,
        cutoff_time: float = 0.8,
        dynamic_seed: int | None = None,
        discard_infeasible: bool = True,
        **kwargs,
    ):
        super().__init__(num_loc=num_loc, **kwargs)
        if not 0.0 <= dod <= 1.0:
            raise ValueError(f"dod must be in [0, 1], got {dod}")
        if not 0.0 <= cutoff_time <= 1.0:
            raise ValueError(f"cutoff_time must be in [0, 1], got {cutoff_time}")

        self.subsample = False
        self.dod = float(dod)
        self.cutoff_time = float(cutoff_time)
        self.dynamic_seed = dynamic_seed
        self.discard_infeasible = discard_infeasible
        self._rng = np.random.default_rng(dynamic_seed)

        self.all_locs = torch.stack([x["locs"][: self.num_loc + 1] for x in solomon_instances])
        self.all_time_windows = torch.stack([x["time_windows"][: self.num_loc + 1] for x in solomon_instances])
        self.all_service_time = torch.stack([x["service_time"][: self.num_loc + 1] for x in solomon_instances])
        self.all_demand_linehaul = torch.stack([x["demand_linehaul"][: self.num_loc].float() for x in solomon_instances])
        self.all_vehicle_capacity = torch.tensor(
            [[float(x["vehicle_capacity"])] for x in solomon_instances],
            dtype=torch.float32,
        )

        self.num_instances = self.all_locs.shape[0]

    def perturb_time_windows(self, locs, time_windows, service_time):
        tw = time_windows.clone()
        st = service_time.clone()

        start = tw[:, 1:, 0]
        end = tw[:, 1:, 1]
        width = end - start

        width_factor = (1.0 + 0.03 * torch.randn_like(width)).clamp(0.8, 1.2)
        service_factor = (1.0 + 0.03 * torch.randn_like(st[:, 1:])).clamp(0.8, 1.2)

        new_width = (width * width_factor).clamp_min(0.05)
        new_service = (st[:, 1:] * service_factor).clamp_min(0.0)

        center = 0.5 * (start + end)
        new_start = center - 0.5 * new_width
        new_end = center + 0.5 * new_width

        depot_due = tw[:, 0, 1].unsqueeze(-1)
        new_start = new_start.clamp_min(0.0)
        new_end = torch.minimum(new_end, depot_due)
        new_end = torch.maximum(new_end, new_start + 0.05)

        tw[:, 1:, 0] = new_start
        tw[:, 1:, 1] = new_end
        st[:, 1:] = new_service

        tw[:, 0, 0] = 0.0
        tw[:, 0, 1] = self.max_time
        st[:, 0] = 0.0

        return tw, st

    def _sample_disclosing_times(
        self, time_windows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[float]]]:
        batch_size, num_nodes, _ = time_windows.shape
        customer_count = num_nodes - 1

        adjusted_time_windows = time_windows.clone()
        reveal_times = torch.zeros((batch_size, num_nodes), dtype=torch.float32, device=time_windows.device)
        is_dynamic = torch.zeros((batch_size, num_nodes), dtype=torch.bool, device=time_windows.device)
        disclosing_time: list[list[float]] = []

        for batch_idx in range(batch_size):
            horizon = float(time_windows[batch_idx, 0, 1].item())
            cutoff = horizon * self.cutoff_time
            n_dyn = int(round(self.dod * customer_count))

            while True:
                dynamic_customer_pos = (
                    np.sort(self._rng.choice(customer_count, size=n_dyn, replace=False)).tolist()
                    if n_dyn > 0
                    else []
                )

                scenario_reveal = [0.0 for _ in range(num_nodes)]
                scenario_dynamic = set()
                feasible = True
                candidate_time_windows = time_windows[batch_idx].clone()

                for customer_pos in dynamic_customer_pos:
                    node_idx = customer_pos + 1
                    scenario_dynamic.add(node_idx)
                    due_time = float(time_windows[batch_idx, node_idx, 1].item())
                    upper_bound = min(due_time, cutoff)
                    reveal_time = (
                        float(self._rng.uniform(0.0, upper_bound)) if upper_bound > 0.0 else 0.0
                    )

                    ready_time = float(time_windows[batch_idx, node_idx, 0].item())
                    adjusted_ready = max(ready_time, reveal_time)
                    adjusted_due = min(due_time, horizon)
                    if adjusted_ready > adjusted_due:
                        feasible = False
                        break

                    scenario_reveal[node_idx] = reveal_time
                    candidate_time_windows[node_idx, 0] = adjusted_ready
                    candidate_time_windows[node_idx, 1] = adjusted_due

                if feasible or not self.discard_infeasible:
                    adjusted_time_windows[batch_idx] = candidate_time_windows
                    for customer_id, reveal_time in enumerate(scenario_reveal):
                        reveal_times[batch_idx, customer_id] = reveal_time
                        is_dynamic[batch_idx, customer_id] = customer_id in scenario_dynamic
                    disclosing_time.append(scenario_reveal)
                    break

        return adjusted_time_windows, reveal_times, is_dynamic, disclosing_time

    def _generate(self, batch_size):
        if isinstance(batch_size, (tuple, list, torch.Size)):
            B = batch_size[0]
        else:
            B = batch_size

        idx = torch.randint(0, self.num_instances, (B,))

        locs = self.all_locs[idx].clone()
        time_windows = self.all_time_windows[idx].clone()
        service_time = self.all_service_time[idx].clone()
        demand_linehaul = self.all_demand_linehaul[idx].clone()
        vehicle_capacity = self.all_vehicle_capacity[idx].clone()
        capacity_original = vehicle_capacity.clone()

        # optional: small perturbation here, fully vectorized
        # time_windows, service_time = self.perturb_time_windows(locs, time_windows, service_time)

        time_windows, reveal_times, is_dynamic, disclosing_time = self._sample_disclosing_times(time_windows)

        demand_backhaul = torch.zeros_like(demand_linehaul)
        backhaul_class = self.generate_backhaul_class(shape=(B, 1), sample=self.sample_backhaul_class)
        open_route = self.generate_open_route(shape=(B, 1))
        speed = self.generate_speed(shape=(B, 1))
        distance_limit = self.generate_distance_limit(shape=(B, 1), locs=locs)

        if self.scale_demand:
            demand_linehaul = demand_linehaul / vehicle_capacity
            demand_backhaul = demand_backhaul / vehicle_capacity
            vehicle_capacity = vehicle_capacity / vehicle_capacity

        td = TensorDict(
            {
                "locs": locs,
                "demand_backhaul": demand_backhaul,
                "demand_linehaul": demand_linehaul,
                "backhaul_class": backhaul_class,
                "distance_limit": distance_limit,
                "time_windows": time_windows,
                "service_time": service_time,
                "vehicle_capacity": vehicle_capacity,
                "capacity_original": capacity_original,
                "open_route": open_route,
                "speed": speed,
                "reveal_times": reveal_times,
                "is_dynamic": is_dynamic,
                "disclosing_time": disclosing_time,
            },
            batch_size=[B],
        )

        if self.subsample:
            td = self.subsample_problems(td)

        return td

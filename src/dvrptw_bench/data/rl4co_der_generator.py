from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from tensordict import TensorDict

from dvrptw_bench.data.der_solomon_generator import DERTimeWindowGenerator
from rl4co.envs.routing.cvrptw.generator import CVRPTWGenerator

@dataclass
class FamilySpec:
    name: str
    density_levels: Sequence[float] = (1.0, 0.75, 0.5, 0.25)
    weight: float = 1.0


class DERSolomonCVRPTWGenerator(CVRPTWGenerator):
    """
    RL4CO-compatible generator that plugs DER-Solomon instances
    into CVRPTWEnv by overriding _generate().
    """

    def __init__(
        self,
        der_generator: DERTimeWindowGenerator,
        family_specs: Sequence[FamilySpec],
        seed: int = 1234,
        normalize_coords: bool = True,
        num_loc: int = 100,
        min_loc: float = 0.0,
        max_loc: float = 100.0,
        vehicle_capacity: float = 1.0,
        max_time: float = 1000.0,
        scale: bool = False,
        min_customer_window_size: float = 1e-3,
        feasibility_margin: float = 1e-2,
        max_sampling_attempts: int = 64,
        **kwargs,
    ):
        super().__init__(
            num_loc=num_loc,
            min_loc=min_loc,
            max_loc=max_loc,
            vehicle_capacity=vehicle_capacity,
            max_time=max_time,
            scale=scale,
            **kwargs,
        )
        self.der_generator = der_generator
        self.family_specs = list(family_specs)
        self.normalize_coords = normalize_coords
        self.min_customer_window_size = float(min_customer_window_size)
        self.feasibility_margin = float(feasibility_margin)
        self.max_sampling_attempts = int(max_sampling_attempts)

        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

        if not self.family_specs:
            raise ValueError("family_specs must contain at least one family.")
        weights = torch.tensor([fs.weight for fs in self.family_specs], dtype=torch.float32)
        self.family_probs = weights / weights.sum()

    def _choose_family_and_density(self) -> tuple[str, float]:
        idx = torch.multinomial(self.family_probs, num_samples=1, generator=self.rng).item()
        spec = self.family_specs[idx]
        density_idx = torch.randint(
            low=0,
            high=len(spec.density_levels),
            size=(1,),
            generator=self.rng,
        ).item()
        return spec.name, float(spec.density_levels[density_idx])

    @staticmethod
    def _normalize_coord(coord: torch.Tensor) -> torch.Tensor:
        x, y = coord[:, 0], coord[:, 1]
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        x_range = x_max - x_min
        y_range = y_max - y_min

        if float(x_range) == 0.0:
            x_scaled = torch.zeros_like(x)
        else:
            x_scaled = (x - x_min) / x_range

        if float(y_range) == 0.0:
            y_scaled = torch.zeros_like(y)
        else:
            y_scaled = (y - y_min) / y_range

        return torch.stack([x_scaled, y_scaled], dim=1)

    def _project_customer_windows(self, instance) -> tuple[torch.Tensor, torch.Tensor]:
        n = len(instance.customers)
        durations = torch.zeros(n + 1, dtype=torch.float32)
        time_windows = torch.zeros(n + 1, 2, dtype=torch.float32)

        depot = instance.depot
        depot_open = float(self.min_time)
        depot_close = float(self.max_time)
        eps = self.min_customer_window_size
        time_windows[0, 0] = depot_open
        time_windows[0, 1] = depot_close

        depot_xy = (float(depot.x), float(depot.y))
        for idx, customer in enumerate(instance.customers, start=1):
            service = float(customer.service_time)
            durations[idx] = service
            dist0i = math.hypot(float(customer.x) - depot_xy[0], float(customer.y) - depot_xy[1])

            # Hard-feasible bounds used by RL4CO CVRPTW semantics.
            latest_start = depot_close - dist0i - service - self.feasibility_margin
            min_due = dist0i + eps
            max_due = latest_start

            start = float(customer.ready_time)
            end = float(customer.due_time)
            start = min(max(start, depot_open), latest_start)
            end = min(max(end, min_due), max_due)

            if end <= start + eps:
                end = min(max(min_due, start + eps), max_due)
            if end <= start + eps:
                start = max(depot_open, min(latest_start, end - eps))
            if end <= start + eps:
                start = depot_open
                end = min(max_due, max(min_due, start + eps))

            time_windows[idx, 0] = float(start)
            time_windows[idx, 1] = float(end)

        return durations, time_windows

    def _instance_to_tensordict(self, instance) -> TensorDict:
        coords = torch.tensor(
            [[instance.depot.x, instance.depot.y], *[[c.x, c.y] for c in instance.customers]],
            dtype=torch.float32,
        )

        if self.normalize_coords:
            coords = self._normalize_coord(coords)

        capacity = float(instance.vehicle_capacity)
        demand = torch.tensor([c.demand for c in instance.customers], dtype=torch.float32)
        durations, time_windows = self._project_customer_windows(instance)

        if self.scale:
            coords = coords / self.max_time
            durations = durations / self.max_time
            time_windows = time_windows / self.max_time

        td = TensorDict(
            {
                "depot": coords[0],
                "locs": coords[1:],
                "demand": demand / capacity,
                "capacity": torch.tensor(1.0, dtype=torch.float32),
                "durations": durations,
                "time_windows": time_windows,
            },
            batch_size=[],
        )
        return td

    def _is_hard_feasible(self, td: TensorDict) -> bool:
        if td["locs"].shape[0] != self.num_loc:
            return False

        if not torch.isfinite(td["depot"]).all() or not torch.isfinite(td["locs"]).all():
            return False

        if not torch.isfinite(td["demand"]).all() or not torch.isfinite(td["capacity"]).all():
            return False

        if not torch.isfinite(td["durations"]).all() or not torch.isfinite(td["time_windows"]).all():
            return False

        if td["durations"].shape[0] != self.num_loc + 1:
            return False

        if td["time_windows"].shape != (self.num_loc + 1, 2):
            return False

        if (td["demand"] < 0).any():
            return False

        if (td["demand"] > self.vehicle_capacity + 1e-6).any():
            return False

        if (td["durations"] < 0).any():
            return False

        widths = td["time_windows"][:, 1] - td["time_windows"][:, 0]
        if (widths <= 0).any():
            return False

        depot = td["depot"]
        locs = td["locs"]
        dist0 = torch.norm(locs - depot[None, :], dim=-1)
        latest_start = td["time_windows"][0, 1] - dist0 - td["durations"][1:] - self.feasibility_margin
        starts = td["time_windows"][1:, 0]
        due = td["time_windows"][1:, 1]
        if (starts - 1e-6 > latest_start).any():
            return False
        if (due - 1e-6 > latest_start).any():
            return False
        if (due + 1e-6 < dist0).any():
            return False

        return True

    def _generate(self, batch_size) -> TensorDict:
        if isinstance(batch_size, int):
            n = batch_size
            batch_dims = [batch_size]
        else:
            if len(batch_size) != 1:
                raise ValueError(f"Expected 1D batch size, got {batch_size}")
            n = int(batch_size[0])
            batch_dims = list(batch_size)

        items = []
        for _ in range(n):
            accepted = None
            for _attempt in range(self.max_sampling_attempts):
                family, density = self._choose_family_and_density()
                py_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=self.rng).item())
                inst = self.der_generator.sample_instance(
                    family=family,
                    density=density,
                    seed=py_seed,
                )
                td = self._instance_to_tensordict(inst)
                if self._is_hard_feasible(td):
                    accepted = td
                    break
            if accepted is None:
                raise RuntimeError(
                    "Could not sample a hard-feasible DER-Solomon CVRPTW instance "
                    f"in {self.max_sampling_attempts} attempts."
                )
            items.append(accepted)

        td = torch.stack(items, dim=0)
        td.batch_size = torch.Size(batch_dims)
        return td

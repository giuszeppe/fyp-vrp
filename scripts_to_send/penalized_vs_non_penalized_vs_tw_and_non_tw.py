from __future__ import annotations
from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
from scipy import stats
import torch
from rl4co.envs import CVRPTWEnv
from rl4co.envs.routing.cvrptw.generator import CVRPTWGenerator
from rl4co.models.rl import REINFORCE
from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.utils.trainer import RL4COTrainer
from tensordict import TensorDict


Density = Literal[0.25, 0.50, 0.75, 1.00]
Family = Literal["C1", "C2", "R1", "R2", "RC1", "RC2"]


@dataclass(frozen=True)
class Node:
    id: int
    x: float
    y: float
    demand: float = 0.0
    ready_time: float = 0.0
    due_time: float = 1e9
    service_time: float = 0.0


@dataclass(frozen=True)
class VRPTWInstance:
    instance_id: str
    depot: Node
    customers: list[Node]
    vehicle_capacity: float
    vehicle_count: int
    distance_matrix: list[list[float]]

    @property
    def n_customers(self) -> int:
        return len(self.customers)


def euclidean_nodes(a: Node, b: Node, scale: float = 1.0) -> float:
    return scale * math.hypot(a.x - b.x, a.y - b.y)


def distance_matrix(nodes: list[Node], scale: float = 1.0) -> list[list[float]]:
    return [[euclidean_nodes(i, j, scale=scale) for j in nodes] for i in nodes]


def _extract_vehicle_info(lines: list[str]) -> tuple[int, float]:
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("VEHICLE"):
            for j in range(i, min(len(lines), i + 8)):
                parts = lines[j].split()
                if len(parts) >= 2 and parts[0].isdigit():
                    return int(parts[0]), float(parts[1])
    return 25, 200.0


def _find_customer_section(lines: list[str]) -> int:
    for i, line in enumerate(lines):
        if "CUSTOMER" in line.upper():
            return i + 1
    raise ValueError("Could not find CUSTOMER section in Solomon file.")


def parse_solomon(path: Path, distance_scale: float = 1.0, max_customers: int | None = None) -> VRPTWInstance:
    raw_lines = [ln.rstrip() for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines()]
    lines = [ln for ln in raw_lines if ln.strip()]
    if not lines:
        raise ValueError(f"Empty file: {path}")
    if max_customers is not None and max_customers < 0:
        raise ValueError("max_customers must be >= 0")

    vehicle_count, capacity = _extract_vehicle_info(lines)
    start = _find_customer_section(lines)

    rows: list[list[float]] = []
    for line in lines[start:]:
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            rows.append([float(v) for v in parts[:7]])
        except ValueError:
            continue

    if not rows:
        raise ValueError(f"No customer rows parsed from {path}")

    depot_row, *customer_rows = rows
    depot = Node(
        id=int(depot_row[0]),
        x=depot_row[1],
        y=depot_row[2],
        demand=depot_row[3],
        ready_time=depot_row[4],
        due_time=depot_row[5],
        service_time=depot_row[6],
    )
    customers = [
        Node(
            id=int(row[0]),
            x=row[1],
            y=row[2],
            demand=row[3],
            ready_time=row[4],
            due_time=row[5],
            service_time=row[6],
        )
        for row in customer_rows
    ]
    if max_customers is not None:
        customers = customers[:max_customers]

    dmat = distance_matrix([depot, *customers], scale=distance_scale)
    return VRPTWInstance(
        instance_id=path.name,
        depot=depot,
        customers=customers,
        vehicle_capacity=capacity,
        vehicle_count=vehicle_count,
        distance_matrix=dmat,
    )


@dataclass(frozen=True)
class Customer:
    id: int
    x: float
    y: float
    demand: float
    ready_time: float
    due_time: float
    service_time: float


@dataclass(frozen=True)
class Depot:
    x: float
    y: float
    ready_time: float
    due_time: float
    service_time: float = 0.0


@dataclass(frozen=True)
class SolomonSeriesTemplate:
    family: Family
    vehicle_capacity: float
    depot: Depot
    customers: tuple[Customer, ...]

    @property
    def n_customers(self) -> int:
        return len(self.customers)


@dataclass(frozen=True)
class GeneratedInstance:
    family: Family
    density: Density
    vehicle_capacity: float
    depot: Depot
    customers: tuple[Customer, ...]
    constrained_mask: tuple[bool, ...]
    seed: int | None = None


def template_dict_from_vrptw_instance(instance: VRPTWInstance, family: str) -> dict:
    valid_families = {"C1", "C2", "R1", "R2", "RC1", "RC2"}
    if family not in valid_families:
        raise ValueError(f"Unsupported family '{family}'. Expected one of {sorted(valid_families)}.")

    return {
        "family": family,
        "vehicle_capacity": float(instance.vehicle_capacity),
        "depot": {
            "x": float(instance.depot.x),
            "y": float(instance.depot.y),
            "ready_time": float(instance.depot.ready_time),
            "due_time": float(instance.depot.due_time),
            "service_time": float(instance.depot.service_time),
        },
        "customers": [
            {
                "id": int(c.id),
                "x": float(c.x),
                "y": float(c.y),
                "demand": float(c.demand),
                "ready_time": float(c.ready_time),
                "due_time": float(c.due_time),
                "service_time": float(c.service_time),
            }
            for c in instance.customers
        ],
    }


def template_from_dict(data: dict) -> SolomonSeriesTemplate:
    depot_data = data["depot"]
    customers_data = data["customers"]
    return SolomonSeriesTemplate(
        family=data["family"],
        vehicle_capacity=float(data["vehicle_capacity"]),
        depot=Depot(
            x=float(depot_data["x"]),
            y=float(depot_data["y"]),
            ready_time=float(depot_data["ready_time"]),
            due_time=float(depot_data["due_time"]),
            service_time=float(depot_data.get("service_time", 0.0)),
        ),
        customers=tuple(
            Customer(
                id=int(c["id"]),
                x=float(c["x"]),
                y=float(c["y"]),
                demand=float(c["demand"]),
                ready_time=float(c["ready_time"]),
                due_time=float(c["due_time"]),
                service_time=float(c["service_time"]),
            )
            for c in customers_data
        ),
    )


def euclidean_points(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class DistSampler:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def constant(self, value: float, size: int) -> np.ndarray:
        return np.full(size, float(value), dtype=float)

    def beta(self, a: float, b: float, loc: float, scale: float, size: int) -> np.ndarray:
        return stats.beta.rvs(a, b, loc=loc, scale=scale, size=size, random_state=self.rng)

    def gamma(self, shape: float, loc: float, scale: float, size: int) -> np.ndarray:
        return stats.gamma.rvs(shape, loc=loc, scale=scale, size=size, random_state=self.rng)

    def gev(self, c: float, loc: float, scale: float, size: int) -> np.ndarray:
        return stats.genextreme.rvs(c, loc=loc, scale=scale, size=size, random_state=self.rng)

    def symmetric_weibull(self, c: float, loc: float, scale: float, size: int) -> np.ndarray:
        widths = self.rng.weibull(c, size=size)
        signs = self.rng.choice(np.array([-1.0, 1.0]), size=size)
        return loc + scale * signs * widths

    def truncated(
        self,
        sample_fn: Callable[[int], np.ndarray],
        size: int,
        low: float | None = None,
        high: float | None = None,
        max_rounds: int = 10_000,
    ) -> np.ndarray:
        out = np.empty(size, dtype=float)
        filled = 0
        rounds = 0
        while filled < size:
            rounds += 1
            if rounds > max_rounds:
                raise RuntimeError("Exceeded max_rounds while truncation-sampling.")
            need = size - filled
            vals = sample_fn(max(need * 2, 32))
            if low is not None:
                vals = vals[vals >= low]
            if high is not None:
                vals = vals[vals <= high]
            take = min(need, len(vals))
            if take > 0:
                out[filled : filled + take] = vals[:take]
                filled += take
        return out


class FamilyRule:
    def sample(self, n: int, sampler: DistSampler) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class AtomicRule(FamilyRule):
    kind: Literal["const", "beta", "gamma", "gev", "weibull"]
    params: tuple[float, ...]
    low: float | None = None
    high: float | None = None

    def sample(self, n: int, sampler: DistSampler) -> np.ndarray:
        if self.kind == "const":
            return sampler.constant(self.params[0], n)
        if self.kind == "beta":
            a, b, loc, scale = self.params
            return sampler.truncated(
                lambda m: sampler.beta(a, b, loc, scale, m),
                size=n,
                low=self.low,
                high=self.high,
            )
        if self.kind == "gamma":
            shape, loc, scale = self.params
            return sampler.truncated(
                lambda m: sampler.gamma(shape, loc, scale, m),
                size=n,
                low=self.low,
                high=self.high,
            )
        if self.kind == "gev":
            c, loc, scale = self.params
            return sampler.truncated(
                lambda m: sampler.gev(c, loc, scale, m),
                size=n,
                low=self.low,
                high=self.high,
            )
        if self.kind == "weibull":
            c, loc, scale = self.params
            return sampler.truncated(
                lambda m: sampler.symmetric_weibull(c, loc, scale, m),
                size=n,
                low=self.low,
                high=self.high,
            )
        raise ValueError(f"Unsupported rule kind: {self.kind}")


@dataclass(frozen=True)
class MixtureRule(FamilyRule):
    rules: tuple[FamilyRule, ...]
    weights: tuple[float, ...]

    def sample(self, n: int, sampler: DistSampler) -> np.ndarray:
        weights = np.asarray(self.weights, dtype=float)
        weights = weights / weights.sum()
        choices = sampler.rng.choice(len(self.rules), size=n, p=weights)
        out = np.empty(n, dtype=float)
        for idx, rule in enumerate(self.rules):
            mask = choices == idx
            k = int(mask.sum())
            if k:
                out[mask] = rule.sample(k, sampler)
        return out


@dataclass(frozen=True)
class NestedWithinInstanceRule(FamilyRule):
    subrules: tuple[FamilyRule, ...]
    proportions: tuple[float, ...]

    def sample(self, n: int, sampler: DistSampler) -> np.ndarray:
        return MixtureRule(self.subrules, self.proportions).sample(n, sampler)


def _const(v: float) -> AtomicRule:
    return AtomicRule("const", (v,))


def _beta(a: float, b: float, loc: float, scale: float, low: float, high: float) -> AtomicRule:
    return AtomicRule("beta", (a, b, loc, scale), low, high)


def _gamma(shape: float, loc: float, scale: float, low: float, high: float) -> AtomicRule:
    return AtomicRule("gamma", (shape, loc, scale), low, high)


def _gev(c: float, loc: float, scale: float, low: float, high: float) -> AtomicRule:
    return AtomicRule("gev", (c, loc, scale), low, high)


def _weibull(c: float, loc: float, scale: float, low: float, high: float) -> AtomicRule:
    return AtomicRule("weibull", (c, loc, scale), low, high)


FAMILY_RULES: dict[Family, FamilyRule] = {
    "C1": MixtureRule(
        rules=(
            _beta(4.06, 5.95, 16.05, 35.34, 22.17, 39.39),
            _beta(3.66, 5.33, 33.51, 67.03, 44.49, 78.79),
            _gamma(1.52, 12.49, 43.03, 20.38, 182.43),
            _beta(3.73, 5.23, 66.20, 133.38, 88.87, 157.46),
            _const(90.0),
            _const(180.0),
        ),
        weights=(3 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8),
    ),
    "C2": MixtureRule(
        rules=(
            _const(80.0),
            _const(160.0),
            _const(320.0),
            _beta(3.67, 5.20, 133.29, 266.06, 177.81, 315.13),
            _beta(0.86, 1.41, 88.50, 547.94, 100.08, 561.94),
        ),
        weights=(1 / 2, 1 / 8, 1 / 8, 1 / 8, 1 / 8),
    ),
    "R1": MixtureRule(
        rules=(
            _const(5.0),
            _const(15.0),
            _gev(0.23, 27.77, 4.35, 22.35, 37.11),
            _beta(1.23, 1.82, 11.32, 79.54, 15.38, 77.68),
            _beta(0.77, 1.25, 9.50, 88.05, 10.91, 87.51),
            _gev(0.24, 55.60, 8.57, 44.82, 73.69),
        ),
        weights=(1 / 4, 1 / 4, 1 / 8, 1 / 8, 1 / 8, 1 / 8),
    ),
    "R2": MixtureRule(
        rules=(
            _gev(0.22, 51.24, 17.33, 29.68, 88.83),
            _const(120.0),
            _beta(1.30, 2.27, 44.52, 359.15, 61.63, 322.55),
            _beta(0.90, 1.76, 36.50, 457.83, 45.57, 403.76),
            _gev(0.22, 222.49, 34.74, 179.26, 297.74),
        ),
        weights=(3 / 8, 1 / 4, 1 / 8, 1 / 8, 1 / 8),
    ),
    "RC1": MixtureRule(
        rules=(
            _const(15.0),
            NestedWithinInstanceRule(
                subrules=(
                    _const(5.0),
                    _const(60.0),
                    _beta(1.94, 87.21, 8.89, 663.77, 22.52, 39.25),
                ),
                proportions=(1 / 4, 1 / 4, 1 / 2),
            ),
            _const(30.0),
            NestedWithinInstanceRule(
                subrules=(
                    _beta(2.88, 8.24, 19.28, 40.81, 22.52, 39.25),
                    _beta(12.26, 10.26, 16.42, 78.39, 45.65, 72.19),
                ),
                proportions=(1 / 2, 1 / 2),
            ),
            _beta(9.90, 5.49, -27.18, 129.57, 29.53, 79.97),
        ),
        weights=(1 / 2, 1 / 8, 1 / 8, 1 / 8, 1 / 8),
    ),
    "RC2": MixtureRule(
        rules=(
            _const(60.0),
            NestedWithinInstanceRule(
                subrules=(
                    _const(30.0),
                    _const(240.0),
                    _weibull(2.05, 92.65, 31.63, 45.15, 140.14),
                ),
                proportions=(1 / 4, 1 / 4, 1 / 2),
            ),
            _const(120.0),
            _beta(1.30, 2.27, 44.52, 359.15, 61.63, 322.55),
            _gev(0.22, 222.48, 34.73, 179.26, 297.74),
        ),
        weights=(1 / 2, 1 / 8, 1 / 8, 1 / 8, 1 / 8),
    ),
}


class DERTimeWindowGenerator:
    def __init__(
        self,
        templates: dict[Family, SolomonSeriesTemplate],
        seed: int | None = None,
        c_center_provider: Callable[[SolomonSeriesTemplate, np.random.Generator], np.ndarray] | None = None,
    ):
        self.templates = templates
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.c_center_provider = c_center_provider

    def sample_instance(self, family: Family, density: Density = 1.00, seed: int | None = None) -> GeneratedInstance:
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        sampler = DistSampler(rng)
        template = self.templates[family]

        half_widths = self._sample_half_widths(family, template.n_customers, sampler)
        centers = self._sample_centers(template, rng)
        starts, ends = self._build_time_windows(template, centers, half_widths)
        constrained_mask = self._sample_density_mask(template.n_customers, density, rng)

        customers: list[Customer] = []
        for i, cust in enumerate(template.customers):
            if constrained_mask[i]:
                ready, due = starts[i], ends[i]
            else:
                ready, due = template.depot.ready_time, template.depot.due_time
            customers.append(replace(cust, ready_time=float(ready), due_time=float(due)))

        return GeneratedInstance(
            family=family,
            density=density,
            vehicle_capacity=template.vehicle_capacity,
            depot=template.depot,
            customers=tuple(customers),
            constrained_mask=tuple(bool(x) for x in constrained_mask.tolist()),
            seed=seed if seed is not None else self.seed,
        )

    def _sample_density_mask(self, n: int, density: Density, rng: np.random.Generator) -> np.ndarray:
        k = int(round(n * float(density)))
        mask = np.zeros(n, dtype=bool)
        idx = rng.choice(n, size=k, replace=False)
        mask[idx] = True
        return mask

    def _sample_centers(self, template: SolomonSeriesTemplate, rng: np.random.Generator) -> np.ndarray:
        if template.family.startswith("C"):
            if self.c_center_provider is not None:
                centers = np.asarray(self.c_center_provider(template, rng), dtype=float)
            else:
                centers = np.asarray(
                    [0.5 * (c.ready_time + c.due_time) for c in template.customers],
                    dtype=float,
                )
            if centers.shape != (template.n_customers,):
                raise ValueError(f"Expected {template.n_customers} centers, got {centers.shape}.")
            return centers

        depot = template.depot
        centers = np.empty(template.n_customers, dtype=float)
        for i, cust in enumerate(template.customers):
            dist0i = euclidean_points((depot.x, depot.y), (cust.x, cust.y))
            low = depot.ready_time + dist0i
            high = depot.due_time - dist0i - cust.service_time
            if high < low:
                raise ValueError(
                    f"Infeasible center range for customer {cust.id} in family {template.family}: [{low}, {high}]"
                )
            centers[i] = rng.uniform(low, high)
        return centers

    def _build_time_windows(
        self,
        template: SolomonSeriesTemplate,
        centers: np.ndarray,
        half_widths: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        depot = template.depot
        starts = np.maximum(centers - half_widths, depot.ready_time)
        ends = np.minimum(centers + half_widths, depot.due_time)
        bad = ends < starts
        if np.any(bad):
            mids = centers[bad]
            starts[bad] = np.clip(mids, depot.ready_time, depot.due_time)
            ends[bad] = starts[bad]
        return starts, ends

    def _sample_half_widths(self, family: Family, n: int, sampler: DistSampler) -> np.ndarray:
        return FAMILY_RULES[family].sample(n=n, sampler=sampler)


@dataclass
class FamilySpec:
    name: str
    density_levels: Sequence[float] = (1.0, 0.75, 0.5, 0.25)
    weight: float = 1.0


class DERSolomonCVRPTWGenerator(CVRPTWGenerator):
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
        density_idx = torch.randint(low=0, high=len(spec.density_levels), size=(1,), generator=self.rng).item()
        return spec.name, float(spec.density_levels[density_idx])

    @staticmethod
    def _normalize_coord(coord: torch.Tensor) -> torch.Tensor:
        x, y = coord[:, 0], coord[:, 1]
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()

        x_scaled = torch.zeros_like(x) if float(x_range) == 0.0 else (x - x.min()) / x_range
        y_scaled = torch.zeros_like(y) if float(y_range) == 0.0 else (y - y.min()) / y_range
        return torch.stack([x_scaled, y_scaled], dim=1)

    def _project_customer_windows(self, instance: GeneratedInstance) -> tuple[torch.Tensor, torch.Tensor]:
        n = len(instance.customers)
        durations = torch.zeros(n + 1, dtype=torch.float32)
        time_windows = torch.zeros(n + 1, 2, dtype=torch.float32)

        depot_open = float(self.min_time)
        depot_close = float(self.max_time)
        eps = self.min_customer_window_size
        time_windows[0, 0] = depot_open
        time_windows[0, 1] = depot_close

        depot_xy = (float(instance.depot.x), float(instance.depot.y))
        for idx, customer in enumerate(instance.customers, start=1):
            service = float(customer.service_time)
            durations[idx] = service
            dist0i = math.hypot(float(customer.x) - depot_xy[0], float(customer.y) - depot_xy[1])

            latest_start = depot_close - dist0i - service - self.feasibility_margin
            min_due = dist0i + eps
            max_due = latest_start

            start = min(max(float(customer.ready_time), depot_open), latest_start)
            end = min(max(float(customer.due_time), min_due), max_due)

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

    def _instance_to_tensordict(self, instance: GeneratedInstance) -> TensorDict:
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

        return TensorDict(
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
                inst = self.der_generator.sample_instance(family=family, density=density, seed=py_seed)
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


class PenalizedCVRPTWEnv(CVRPTWEnv):
    def __init__(self, vehicle_penalty: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.vehicle_penalty = float(vehicle_penalty)

    @staticmethod
    def count_routes_from_actions(actions: torch.Tensor) -> torch.Tensor:
        batch_routes = []
        for seq in actions.tolist():
            in_route = False
            routes = 0
            for action in seq:
                if action == 0:
                    in_route = False
                elif not in_route:
                    routes += 1
                    in_route = True
            batch_routes.append(routes)
        return torch.tensor(batch_routes, device=actions.device, dtype=torch.float32)

    def _get_reward(self, td, actions):
        base_reward = super()._get_reward(td, actions)
        num_routes = self.count_routes_from_actions(actions)
        return base_reward - self.vehicle_penalty * num_routes


class RLModel:
    def __init__(
        self,
        device=None,
        env=None,
        policy=None,
        model=None,
        max_epochs: int = 100,
        batch_size: int = 512,
        train_data_size: int = 100_000,
        val_data_size: int = 10_000,
        lr: float = 1e-4,
        num_loc: int = 100,
    ):
        self.device = device
        self.env = env
        self.policy = policy
        self.model = model

        if self.device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

        if self.env is None:
            self.env = CVRPTWEnv(generator_params={"num_loc": num_loc})

        if self.policy is None:
            self.policy = AttentionModelPolicy(env_name=self.env.name).to(self.device)

        if self.model is None:
            self.model = REINFORCE(
                self.env,
                self.policy,
                baseline="rollout",
                batch_size=batch_size,
                train_data_size=train_data_size,
                val_data_size=val_data_size,
                optimizer_kwargs={"lr": lr},
            )

        accelerator = "cpu"
        devices = 1
        if self.device.type == "cuda":
            accelerator = "gpu"
        elif self.device.type == "mps":
            accelerator = "mps"

        self.trainer = RL4COTrainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            logger=None,
        )

    def train(self) -> None:
        self.trainer.fit(self.model)

    def save(self, path: str | Path) -> None:
        self.trainer.save_checkpoint(str(path))


def build_attention_model(
    *,
    device=None,
    max_epochs: int = 100,
    batch_size: int = 512,
    train_data_size: int = 100_000,
    val_data_size: int = 10_000,
    lr: float = 1e-4,
    normalize_coords: bool = True,
    der_templates: dict[Family, SolomonSeriesTemplate] | None = None,
    family_specs: Sequence[FamilySpec] | None = None,
    der_seed: int = 123,
    vehicle_penalty: float = 50.0,
    penalized: bool = True,
):
    env = None
    num_loc = 100

    if der_templates is not None:
        if not der_templates:
            raise ValueError("der_templates was provided but is empty.")

        if family_specs is None:
            family_specs = [FamilySpec(name) for name in sorted(der_templates.keys())]
        else:
            family_specs = [fs for fs in family_specs if fs.name in der_templates]
            if not family_specs:
                available = ", ".join(sorted(der_templates.keys()))
                raise ValueError(
                    f"No family_specs match the provided der_templates. Available template families: {available}"
                )

        sample_template = next(iter(der_templates.values()))
        num_loc = sample_template.n_customers
        max_loc = max(
            max(float(t.depot.x), float(t.depot.y), *(max(float(c.x), float(c.y)) for c in t.customers))
            for t in der_templates.values()
        )
        max_time = max(float(t.depot.due_time) for t in der_templates.values())

        der_gen = DERTimeWindowGenerator(der_templates, seed=der_seed)
        generator = DERSolomonCVRPTWGenerator(
            der_generator=der_gen,
            family_specs=family_specs,
            seed=der_seed,
            num_loc=num_loc,
            max_loc=max_loc,
            max_time=max_time,
            normalize_coords=normalize_coords,
        )
        env_cls = PenalizedCVRPTWEnv if penalized else CVRPTWEnv
        env_kwargs = {"generator": generator}
        if penalized:
            env_kwargs["vehicle_penalty"] = vehicle_penalty
        env = env_cls(**env_kwargs)

    return RLModel(
        device=device,
        env=env,
        max_epochs=max_epochs,
        batch_size=batch_size,
        train_data_size=train_data_size,
        val_data_size=val_data_size,
        lr=lr,
        num_loc=num_loc,
    )


def train_one_configuration(
    dataset_path: Path,
    output_dir: Path,
    num_customers: int,
    learning_rate: float,
    penalty: float,
    seed: int,
    max_epochs: int,
    batch_size: int,
    train_data_size: int,
    val_data_size: int,
    family: Family,
    penalized: bool,
) -> Path:
    base_instance = parse_solomon(dataset_path, max_customers=num_customers)
    templates = {
        family: template_from_dict(template_dict_from_vrptw_instance(base_instance, family)),
    }

    model = build_attention_model(
        max_epochs=max_epochs,
        batch_size=batch_size,
        train_data_size=train_data_size,
        val_data_size=val_data_size,
        lr=learning_rate,
        normalize_coords=True,
        der_templates=templates,
        family_specs=[FamilySpec(family)],
        der_seed=seed,
        vehicle_penalty=penalty,
        penalized=penalized,
    )
    model.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "penalized" if penalized else "non_penalized"
    ckpt_path = output_dir / (
        f"attention-model-{suffix}-{family}-customers-{num_customers}-"
        f"penalty-{penalty}-lr-{learning_rate}-seed-{seed}.ckpt"
    )
    model.save(ckpt_path)
    return ckpt_path

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = (SCRIPT_DIR /  "RC201.txt").resolve()
OUTPUT_DIR = (SCRIPT_DIR / "trained_models").resolve()

NUM_CUSTOMERS = [10,25,50,100]
LEARNING_RATES = [1e-4, 5e-4, 1e-3]
PENALTIES = [0,10,50,100,500,1000]
SEEDS = [6238116, 7960100, 7454245]
FAMILY: Family = "RC1"
MAX_EPOCHS = 100
BATCH_SIZE = 512
TRAIN_DATA_SIZE = 100_000
VAL_DATA_SIZE = 10_000
RUN_MODES = [True]


for num_customers in NUM_CUSTOMERS:
    for learning_rate in LEARNING_RATES:
        for penalty in PENALTIES:
            for seed in SEEDS:
                for penalized in RUN_MODES:
                    effective_penalty = penalty if penalized else 0.0
                    label = "penalized" if penalized else "non-penalized"
                    print(
                        f"Training {label} model: customers={num_customers}, lr={learning_rate}, "
                        f"penalty={effective_penalty}, seed={seed}, family={FAMILY}"
                    )
                    ckpt_path = train_one_configuration(
                        dataset_path=DATASET_PATH,
                        output_dir=OUTPUT_DIR,
                        num_customers=num_customers,
                        learning_rate=learning_rate,
                        penalty=effective_penalty,
                        seed=seed,
                        max_epochs=MAX_EPOCHS,
                        batch_size=BATCH_SIZE,
                        train_data_size=TRAIN_DATA_SIZE,
                        val_data_size=VAL_DATA_SIZE,
                        family=FAMILY,
                        penalized=penalized,
                    )
                    print(f"Saved checkpoint to {ckpt_path}")

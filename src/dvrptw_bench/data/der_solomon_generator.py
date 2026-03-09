from __future__ import annotations

"""DER-Solomon style instance generator.

This module implements a practical generator for Solomon-like CVRPTW training data,
following the series-level rules in Table 4 of the DER-Solomon paper.

Important limitation:
- For the C families, the paper says time-window centers come from solving a CVRP via
  a 3-opt route method. The paper does not provide enough implementation detail to
  reproduce that center-generation step exactly. By default this implementation uses
  center values taken from a reference/template instance for the chosen family. You can
  override that with a custom center provider.

What this generator does preserve:
- Same coordinates, demands, service times, depot horizon, and capacity as the base series
  template.
- Density levels 25%, 50%, 75%, and 100%.
- Family-specific half-width distributions from Table 4.
- R / RC center generation using the feasible-range uniform rule stated in the paper.
"""

from dataclasses import dataclass, replace
from typing import Callable, Iterable, Literal, Sequence
import math

import numpy as np

try:
    from scipy import stats
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "der_solomon_generator.py requires scipy. Install scipy to use the GE/Beta/Gamma samplers."
    ) from exc

Density = Literal[0.25, 0.50, 0.75, 1.00]
Family = Literal["C1", "C2", "R1", "R2", "RC1", "RC2"]
CenterStrategy = Literal["paper", "reference"]


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


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class DistSampler:
    """Sampler utilities with truncation/rejection support."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def constant(self, value: float, size: int) -> np.ndarray:
        return np.full(size, float(value), dtype=float)

    def beta(self, a: float, b: float, loc: float, scale: float, size: int) -> np.ndarray:
        return stats.beta.rvs(a, b, loc=loc, scale=scale, size=size, random_state=self.rng)

    def gamma(self, shape: float, loc: float, scale: float, size: int) -> np.ndarray:
        return stats.gamma.rvs(shape, loc=loc, scale=scale, size=size, random_state=self.rng)

    def gev(self, c: float, loc: float, scale: float, size: int) -> np.ndarray:
        # Matches the GE functional form used in the paper.
        return stats.genextreme.rvs(c, loc=loc, scale=scale, size=size, random_state=self.rng)

    def symmetric_weibull(self, c: float, loc: float, scale: float, size: int) -> np.ndarray:
        # The paper's Weibull density is symmetric around loc after the y=(x-loc)/scale transform.
        w = self.rng.weibull(c, size=size)
        signs = self.rng.choice(np.array([-1.0, 1.0]), size=size)
        return loc + scale * signs * w

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


class DERTimeWindowGenerator:
    """Generate Solomon-like CVRPTW instances following DER-Solomon Table 4.

    Parameters
    ----------
    templates:
        Mapping from family name to a base template that provides depot, coordinates,
        demands, service times, and a reference time window layout.
    seed:
        Base seed.
    c_center_provider:
        Optional callback for C-family centers. Signature:
            (template: SolomonSeriesTemplate, rng: np.random.Generator) -> np.ndarray
        It must return one center per customer.
        If omitted, centers are taken from the template's current windows.
    """

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

    def reseed(self, seed: int | None) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def sample_instance(
        self,
        family: Family,
        density: Density = 1.00,
        seed: int | None = None,
    ) -> GeneratedInstance:
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
            customers.append(
                replace(cust, ready_time=float(ready), due_time=float(due))
            )

        return GeneratedInstance(
            family=family,
            density=density,
            vehicle_capacity=template.vehicle_capacity,
            depot=template.depot,
            customers=tuple(customers),
            constrained_mask=tuple(bool(x) for x in constrained_mask.tolist()),
            seed=seed if seed is not None else self.seed,
        )

    def sample_density_bundle(
        self,
        family: Family,
        seed: int | None = None,
        densities: Sequence[Density] = (1.00, 0.75, 0.50, 0.25),
    ) -> dict[Density, GeneratedInstance]:
        """Generate nested-density variants from one shared 100% base.

        This is closer to the Solomon description where lower-density instances reuse
        the same underlying windows as the 100% instance and relax a subset of nodes.
        """
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        sampler = DistSampler(rng)
        template = self.templates[family]

        half_widths = self._sample_half_widths(family, template.n_customers, sampler)
        centers = self._sample_centers(template, rng)
        starts, ends = self._build_time_windows(template, centers, half_widths)

        order = rng.permutation(template.n_customers)
        results: dict[Density, GeneratedInstance] = {}
        for density in sorted(densities, reverse=True):
            k = int(round(template.n_customers * density))
            constrained_mask = np.zeros(template.n_customers, dtype=bool)
            constrained_mask[order[:k]] = True
            customers: list[Customer] = []
            for i, cust in enumerate(template.customers):
                if constrained_mask[i]:
                    ready, due = starts[i], ends[i]
                else:
                    ready, due = template.depot.ready_time, template.depot.due_time
                customers.append(replace(cust, ready_time=float(ready), due_time=float(due)))
            results[density] = GeneratedInstance(
                family=family,
                density=density,
                vehicle_capacity=template.vehicle_capacity,
                depot=template.depot,
                customers=tuple(customers),
                constrained_mask=tuple(bool(x) for x in constrained_mask.tolist()),
                seed=seed if seed is not None else self.seed,
            )
        return results

    def sample_many(
        self,
        family: Family,
        n: int,
        densities: Sequence[Density] = (1.00,),
        base_seed: int | None = None,
    ) -> list[GeneratedInstance]:
        root = np.random.default_rng(base_seed if base_seed is not None else self.seed)
        out: list[GeneratedInstance] = []
        density_list = tuple(densities)
        for _ in range(n):
            density = float(root.choice(np.array(density_list, dtype=float)))
            seed = int(root.integers(0, 2**63 - 1))
            out.append(self.sample_instance(family=family, density=density, seed=seed))
        return out

    def _sample_density_mask(self, n: int, density: Density, rng: np.random.Generator) -> np.ndarray:
        k = int(round(n * float(density)))
        mask = np.zeros(n, dtype=bool)
        idx = rng.choice(n, size=k, replace=False)
        mask[idx] = True
        return mask

    def _sample_centers(self, template: SolomonSeriesTemplate, rng: np.random.Generator) -> np.ndarray:
        family = template.family
        if family.startswith("C"):
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
            dist0i = euclidean((depot.x, depot.y), (cust.x, cust.y))
            low = depot.ready_time + dist0i
            high = depot.due_time - dist0i - cust.service_time
            if high < low:
                raise ValueError(
                    f"Infeasible center range for customer {cust.id} in family {family}: [{low}, {high}]"
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
        starts = centers - half_widths
        ends = centers + half_widths

        # Keep windows within depot horizon.
        starts = np.maximum(starts, depot.ready_time)
        ends = np.minimum(ends, depot.due_time)

        # Guarantee nonnegative width after clipping.
        bad = ends < starts
        if np.any(bad):
            mids = centers[bad]
            starts[bad] = np.clip(mids, depot.ready_time, depot.due_time)
            ends[bad] = starts[bad]

        return starts, ends

    def _sample_half_widths(self, family: Family, n: int, sampler: DistSampler) -> np.ndarray:
        rule = FAMILY_RULES[family]
        return rule.sample(n=n, sampler=sampler)


class FamilyRule:
    def sample(self, n: int, sampler: DistSampler) -> np.ndarray:  # pragma: no cover
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
    """Rule for the 'Within instances, generated proportionally' rows.

    This means one top-level mixture component, and inside that component another fixed
    mixture across customers of the same generated instance.
    """

    subrules: tuple[FamilyRule, ...]
    proportions: tuple[float, ...]

    def sample(self, n: int, sampler: DistSampler) -> np.ndarray:
        mix = MixtureRule(self.subrules, self.proportions)
        return mix.sample(n, sampler)


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


# Table 4 rules from the DER-Solomon paper.
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


def instance_to_dict(inst: GeneratedInstance) -> dict:
    return {
        "family": inst.family,
        "density": inst.density,
        "vehicle_capacity": inst.vehicle_capacity,
        "seed": inst.seed,
        "depot": {
            "x": inst.depot.x,
            "y": inst.depot.y,
            "ready_time": inst.depot.ready_time,
            "due_time": inst.depot.due_time,
            "service_time": inst.depot.service_time,
        },
        "customers": [
            {
                "id": c.id,
                "x": c.x,
                "y": c.y,
                "demand": c.demand,
                "ready_time": c.ready_time,
                "due_time": c.due_time,
                "service_time": c.service_time,
                "constrained": constrained,
            }
            for c, constrained in zip(inst.customers, inst.constrained_mask)
        ],
    }


__all__ = [
    "Customer",
    "Depot",
    "SolomonSeriesTemplate",
    "GeneratedInstance",
    "DERTimeWindowGenerator",
    "template_from_dict",
    "instance_to_dict",
]

"""Compatibility layer anchored to scripts/benchmark_fixed.py metrics and model specs."""

from __future__ import annotations

import os
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from dvrptw_bench.heuristics.constructive_pmca import PMCAVRPTWSolver
from dvrptw_bench.heuristics.ortools_dynamic import ORToolsDVRPTWSolver
from dvrptw_bench.heuristics.ortools_solver import ORToolsVRPTWSolver
from dvrptw_bench.heuristics.pyvrp_solver import PyVRPVRPTWSolver
from dvrptw_bench.hybrid.hybrid_runner import run_hybrid_with_solver_fn


AI_MODEL_SPECS: dict[str, dict[str, Any]] = {
    "routefinder_solomon_generated_50": {
        "checkpoint_candidates": [
            "trained_models/solomon_generated_50_working.ckpt",
            "good_runs/solomon_generated_50_working.pt",
        ],
        "num_customers": 50,
        "variant": "solomon_generated",
    },
}


@dataclass
class BenchmarkResult:
    instance_id: str
    evaluation_size: int
    model_name: str
    model_type: str
    degree_of_dynamicity: float
    cutoff_time: float
    allow_rejection: bool
    total_distance: float
    num_vehicles: int
    computational_time: float
    customers_served: int
    customers_rejected: int
    rejection_rate: float
    average_lateness: float
    service_level: float
    total_cost: float
    routes: list[dict[str, Any]]
    decode_type: str | None = None
    num_samples: int | None = None
    num_starts: int | None = None
    num_augment: int | None = None
    select_best: bool | None = None
    oracle_gap: float | None = None
    run_id: int = 0
    timestamp: str = ""
    feasible: bool = True
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MetricsCalculator:
    """Mirror the benchmark_fixed.py metric contract."""

    @staticmethod
    def calculate_metrics(instance, solution, oracle_result: float | None = None) -> dict[str, Any]:
        metrics = {
            "total_distance": 0.0,
            "num_vehicles": 0,
            "computational_time": solution.solve_time_s if solution else 0.0,
            "customers_served": 0,
            "customers_rejected": 0,
            "rejection_rate": 0.0,
            "average_lateness": 0.0,
            "service_level": 0.0,
            "total_cost": 0.0,
            "oracle_gap": None,
            "feasible": True,
        }
        if not solution or not solution.routes:
            metrics["feasible"] = False
            metrics["customers_rejected"] = instance.n_customers
            metrics["rejection_rate"] = 1.0
            return metrics

        metrics["total_distance"] = solution.total_distance
        metrics["num_vehicles"] = len([route for route in solution.routes if route.node_ids])
        served_customers: set[int] = set()
        for route in solution.routes:
            served_customers.update(route.node_ids)
        metrics["customers_served"] = len(served_customers)
        metrics["customers_rejected"] = instance.n_customers - len(served_customers)
        metrics["rejection_rate"] = metrics["customers_rejected"] / max(1, instance.n_customers)
        if getattr(solution, "violations", None):
            violations = solution.violations
            metrics["average_lateness"] = violations.get("late_sum", 0.0) / max(1, metrics["customers_served"])
            late_count = violations.get("late_count", 0.0)
            metrics["service_level"] = 1.0 - (late_count / max(1, metrics["customers_served"]))
        else:
            metrics["service_level"] = 1.0 if solution.feasible else 0.0
        metrics["total_cost"] = metrics["total_distance"] + (metrics["num_vehicles"] * 100.0)
        if oracle_result is not None:
            metrics["oracle_gap"] = ((metrics["total_cost"] - oracle_result) / oracle_result) * 100
        metrics["feasible"] = bool(getattr(solution, "feasible", True))
        return metrics


def _invoke_solver(model, instance, time_limit_s: float, warm_start=None):
    if hasattr(model, "solve"):
        try:
            return model.solve(instance=instance, time_limit_s=time_limit_s, warm_start=warm_start)
        except TypeError:
            try:
                return model.solve(instance, time_limit_s, warm_start)
            except TypeError:
                try:
                    return model.solve(instance, time_limit_s)
                except TypeError:
                    return model.solve(instance)
    raise TypeError(f"Model {type(model).__name__} does not expose a supported solve() API")


def _resolve_torch_device(device: str | torch.device | None) -> torch.device:
    if isinstance(device, torch.device):
        return device

    requested = device or os.getenv("DVRPTW_EVAL_DEVICE", "auto")
    normalized = str(requested).strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if normalized in {"cpu", "cuda", "mps"}:
        return torch.device(normalized)
    raise ValueError("Unsupported evaluation device. Expected one of: auto, cpu, cuda, mps.")


class RouteFinderBenchmarkSolver:
    """Checkpoint-backed RouteFinder solver wrapper for benchmark use."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        num_customers: int,
        variant: str,
        decode_type: str = "greedy",
        num_samples: int = 1,
        num_starts: int | None = None,
        select_best: bool = True,
        num_augment: int = 8,
        device: str | torch.device | None = None,
    ):
        from dvrptw_bench.rl.routefinder_policy import RouteFinderAdapterPolicy

        self.checkpoint_path = checkpoint_path
        self.variant = variant
        self.num_customers = num_customers
        self.decode_type = decode_type
        self.num_samples = num_samples
        self.num_starts = num_starts
        self.select_best = select_best
        self.num_augment = num_augment
        self.device = _resolve_torch_device(device)
        self.policy = RouteFinderAdapterPolicy()
        self.policy.load(checkpoint_path, device=self.device, num_loc=num_customers)

    def solve(self, instance, time_limit_s=None, warm_start=None):
        _ = (time_limit_s, warm_start)
        try:
            solution = self.policy.infer_instance(
                instance,
                decode_type=self.decode_type,
                num_samples=self.num_samples,
                num_starts=self.num_starts,
                select_best=self.select_best,
                num_augment=self.num_augment,
            )
            solution.details.update(
                {
                    "variant": self.variant,
                    "checkpoint_path": str(self.checkpoint_path),
                }
            )
            return solution
        except Exception as exc:
            raise RuntimeError(
                "RouteFinderBenchmarkSolver.solve failed "
                f"(instance={getattr(instance, 'instance_id', 'unknown')}, "
                f"checkpoint={self.checkpoint_path}, "
                f"variant={self.variant}, "
                f"num_customers={self.num_customers})\n"
                f"Traceback:\n{traceback.format_exc()}"
            ) from exc


class OracleSolver:
    def __init__(self):
        self.solver = ORToolsVRPTWSolver()

    def solve(self, instance, time_limit_s, warm_start=None):
        return self.solver.solve(instance, time_limit_s=time_limit_s, warm_start=warm_start)


def create_model(
    model_name: str,
    project_root: Path,
    model_registry: dict[str, dict[str, Any]],
    *,
    decode_type: str = "greedy",
    num_samples: int = 1,
    num_starts: int | None = None,
    select_best: bool = True,
    num_augment: int = 8,
    device: str | torch.device | None = None,
):
    if model_name == "oracle":
        return "oracle", OracleSolver()
    if model_name == "ortools":
        return "heuristic", ORToolsDVRPTWSolver(soft_time_windows=True)
    if model_name == "pyvrp":
        return "heuristic", PyVRPVRPTWSolver()
    if model_name == "nearest_neighbor":
        return "heuristic", PMCAVRPTWSolver()
    if model_name.startswith("hybrid:"):
        ai_name = model_name.split(":", 1)[1]
        _kind, ai_solver = create_model(
            ai_name,
            project_root,
            model_registry,
            decode_type=decode_type,
            num_samples=num_samples,
            num_starts=num_starts,
            select_best=select_best,
            num_augment=num_augment,
        )

        class HybridSolver:
            def solve(self, instance, time_limit_s, warm_start=None):
                _ = warm_start
                solution, timings = run_hybrid_with_solver_fn(
                    instance=instance,
                    solver_fn=ai_solver.solve,
                    budget_s=time_limit_s,
                )
                solution.details.update(
                    {
                        "decode_type": decode_type,
                        "num_samples": num_samples,
                        "num_starts": num_starts,
                        "select_best": select_best,
                        "num_augment": num_augment,
                    }
                )
                solution.solve_time_s = timings.get("total_s", 0.0)
                return solution

        return "hybrid", HybridSolver()

    spec = model_registry.get(model_name)
    if spec is None:
        raise ValueError(f"Unknown model: {model_name}")
    checkpoint = None
    for candidate in spec.get("checkpoint_candidates", []):
        candidate_path = (project_root / candidate).resolve()
        if candidate_path.exists():
            checkpoint = candidate_path
            break
    if checkpoint is None:
        raise FileNotFoundError(
            f"Checkpoint for {model_name} not found. Candidates: {spec.get('checkpoint_candidates', [])}"
        )
    return "ai", RouteFinderBenchmarkSolver(
        checkpoint_path=checkpoint,
        num_customers=int(spec["num_customers"]),
        variant=str(spec["variant"]),
        decode_type=decode_type,
        num_samples=num_samples,
        num_starts=num_starts,
        select_best=select_best,
        num_augment=num_augment,
        device=device,
    )


def create_static_model(
    model_name: str,
    project_root: Path,
    model_registry: dict[str, dict[str, Any]],
    *,
    decode_type: str = "greedy",
    num_samples: int = 1,
    num_starts: int | None = None,
    select_best: bool = True,
    num_augment: int = 8,
    device: str | torch.device | None = None,
):
    if model_name == "ortools":
        return "heuristic", ORToolsVRPTWSolver()
    return create_model(
        model_name,
        project_root,
        model_registry,
        decode_type=decode_type,
        num_samples=num_samples,
        num_starts=num_starts,
        select_best=select_best,
        num_augment=num_augment,
        device=device,
    )


def result_from_solution(
    *,
    instance,
    model_name: str,
    model_type: str,
    evaluation_size: int,
    degree_of_dynamicity: float,
    cutoff_time: float,
    run_id: int,
    solution,
    oracle_cost: float | None,
    decode_type: str | None = None,
    num_samples: int | None = None,
    num_starts: int | None = None,
    num_augment: int | None = None,
    select_best: bool | None = None,
    error_message: str | None = None,
) -> BenchmarkResult:
    metrics = MetricsCalculator.calculate_metrics(instance, solution, oracle_cost)
    routes = [route.model_dump(mode="json") for route in getattr(solution, "routes", [])]
    details = getattr(solution, "details", {}) or {}
    return BenchmarkResult(
        instance_id=instance.instance_id,
        evaluation_size=evaluation_size,
        model_name=model_name,
        model_type=model_type,
        degree_of_dynamicity=degree_of_dynamicity,
        cutoff_time=cutoff_time,
        allow_rejection=False,
        total_distance=metrics["total_distance"],
        num_vehicles=metrics["num_vehicles"],
        computational_time=metrics["computational_time"],
        customers_served=metrics["customers_served"],
        customers_rejected=metrics["customers_rejected"],
        rejection_rate=metrics["rejection_rate"],
        average_lateness=metrics["average_lateness"],
        service_level=metrics["service_level"],
        total_cost=metrics["total_cost"],
        routes=routes,
        decode_type=details.get("decode_type", decode_type),
        num_samples=details.get("num_samples", num_samples),
        num_starts=details.get("num_starts", num_starts),
        num_augment=details.get("num_augment", num_augment),
        select_best=details.get("select_best", select_best),
        oracle_gap=metrics["oracle_gap"],
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        feasible=metrics["feasible"],
        error_message=error_message,
    )

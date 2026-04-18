#!/usr/bin/env python3
"""
Fixed benchmark script for Dynamic Vehicle Routing Problem with Time Windows (DVRPTW).
This script runs multiple models on Solomon instances and reports comparative results.
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dvrptw_bench.data.solomon_parser import parse_solomon
from dvrptw_bench.data.instance_filters import find_rc_instances, find_c_instances
from dvrptw_bench.heuristics.ortools_solver import ORToolsVRPTWSolver
from dvrptw_bench.heuristics.ortools_dynamic import ORToolsDVRPTWSolver
from dvrptw_bench.heuristics.pyvrp_solver import PyVRPVRPTWSolver
from dvrptw_bench.heuristics.constructive_pmca import PMCAVRPTWSolver
from dvrptw_bench.dynamic.simulator import DynamicSimulator

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


AI_MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "routefinder_solomon_generated_50": {
        "checkpoint_candidates": [
            "good_runs/solomon_generated_50_working.pt",
        ],
        "num_customers": 50,
        "variant": "solomon_generated",
    },
    "routefinder_solomon_generated_75": {
        "checkpoint_candidates": [
            "good_runs/solomon_generated_75_working.pt",
        ],
        "num_customers": 75,
        "variant": "solomon_generated",
    },
    "routefinder_general_50": {
        "checkpoint_candidates": [
            "good_runs/general_50_working.pt",
        ],
        "num_customers": 50,
        "variant": "general",
    },
    "routefinder_general_75": {
        "checkpoint_candidates": [],
        "num_customers": 75,
        "variant": "general",
    },
    "routefinder_with_lateness_50": {
        "checkpoint_candidates": [
            "good_runs/routefinder_with_lateness_50_cust_100_epochs.pt",
        ],
        "num_customers": 50,
        "variant": "with_lateness",
    },
    "routefinder_with_lateness_75": {
        "checkpoint_candidates": [
            "good_runs/routefinder_with_lateness_75_cust_100_epochs.pt",
        ],
        "num_customers": 75,
        "variant": "with_lateness",
    },
}

# =====================================================================
# Configuration
# =====================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    dataset_dir: str = "../dataset"
    evaluation_sizes: List[int] = None

    # Dynamic parameters
    degrees_of_dynamicity: List[float] = None
    cutoff_times: List[float] = None
    allow_rejection: bool = False
    
    # Execution parameters
    num_runs: int = 3
    wipe_results: bool = False
    overwrite_existing: bool = False
    random_seed: int = 42
    
    # Models to test
    heuristics: List[str] = None
    ai_models: List[str] = None
    enable_hybrid: bool = False
    enable_oracle: bool = False
    
    # Output configuration
    results_dir: str = "results"
    output_format: str = "json"  # json, csv, or both
    
    # Resource limits
    max_solve_time: float = 5.0  
    oracle_solve_time: float = 300.0  # 5 minutes for oracle
    heuristic_workers: Optional[int] = None
    
    def __post_init__(self):
        if self.evaluation_sizes is None:
            self.evaluation_sizes = [50, 75]
        if self.degrees_of_dynamicity is None:
            self.degrees_of_dynamicity = [0.3, 0.5, 0.7]
        if self.cutoff_times is None:
            self.cutoff_times = [0.5, 0.7, 0.9]
        if self.heuristics is None:
            self.heuristics = ["ortools"]
        if self.ai_models is None:
            self.ai_models = list(AI_MODEL_SPECS.keys())

# =====================================================================
# Result Data Structure
# =====================================================================

@dataclass
class BenchmarkResult:
    """Single benchmark run result."""
    instance_id: str
    evaluation_size: int
    model_name: str
    model_type: str  # heuristic, ai, hybrid, oracle
    
    # Dynamic parameters
    degree_of_dynamicity: float
    cutoff_time: float
    allow_rejection: bool
    
    # Performance metrics
    total_distance: float
    num_vehicles: int
    computational_time: float
    
    # Service metrics
    customers_served: int
    customers_rejected: int
    rejection_rate: float
    average_lateness: float
    service_level: float  # % served within time window
    
    # Cost metrics
    total_cost: float
    oracle_gap: Optional[float] = None  # % difference from oracle
    
    # Metadata
    run_id: int = 0
    timestamp: str = ""
    feasible: bool = True
    error_message: Optional[str] = None


@dataclass(frozen=True)
class HeuristicRunTask:
    """Serializable payload for heuristic benchmark worker processes."""
    instance_path: str
    instance_id: str
    evaluation_size: int
    model_name: str
    degree_of_dynamicity: float
    cutoff_time: float
    run_id: int
    oracle_cost: Optional[float]
    allow_rejection: bool
    max_solve_time: float
    random_seed: int


def _iter_solomon_instance_files(dataset_dir: Path) -> List[Path]:
    """Return all Solomon instance files under the dataset directory."""
    instance_files: List[Path] = []
    for path in dataset_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() != ".txt":
            continue
        stem = path.stem.upper()
        if stem.startswith("C") or stem.startswith("RC"):
            instance_files.append(path)
    return sorted(instance_files)


def _parse_instance_for_size(instance_path: Path, evaluation_size: int):
    return parse_solomon(instance_path, max_customers=evaluation_size)


def _build_solomon_training_item(instance, num_customers: int, normalize_coords: bool = True) -> Dict[str, Any]:
    from dvrptw_bench.rl.routefinder_adapter import _normalize_coord_and_get_scale
    import torch

    customers = instance.customers[:num_customers]
    coords = [(instance.depot.x, instance.depot.y)] + [(c.x, c.y) for c in customers]
    locs_raw = torch.tensor(coords, dtype=torch.float32)

    if normalize_coords:
        locs, coord_scale_factor = _normalize_coord_and_get_scale(locs_raw)
    else:
        locs = locs_raw
        coord_scale_factor = 1.0

    ready = torch.tensor(
        [instance.depot.ready_time] + [c.ready_time for c in customers],
        dtype=torch.float32,
    )
    due = torch.tensor(
        [instance.depot.due_time] + [c.due_time for c in customers],
        dtype=torch.float32,
    )
    service_raw = torch.tensor(
        [0.0] + [c.service_time for c in customers],
        dtype=torch.float32,
    )
    demand = torch.tensor(
        [c.demand for c in customers],
        dtype=torch.float32,
    )

    time_windows_raw = torch.stack([ready, due], dim=-1)

    if normalize_coords:
        time_windows = time_windows_raw / coord_scale_factor
        service = service_raw / coord_scale_factor
    else:
        time_windows = time_windows_raw
        service = service_raw

    return {
        "locs": locs,
        "time_windows": time_windows,
        "service_time": service,
        "demand_linehaul": demand,
        "vehicle_capacity": float(instance.vehicle_capacity),
        "instance_id": instance.instance_id,
        "coord_scale_factor": coord_scale_factor,
    }


def _resolve_instance_selection(dataset_dir: Path, names: List[str]) -> List[Path]:
    available = _iter_solomon_instance_files(dataset_dir)
    by_name = {path.name: path for path in available}
    by_stem = {path.stem: path for path in available}
    selected: List[Path] = []
    for name in names:
        direct = dataset_dir / name
        if direct.exists():
            selected.append(direct)
            continue
        match = by_name.get(name) or by_stem.get(name)
        if match is None:
            raise FileNotFoundError(f"Instance '{name}' was not found under {dataset_dir}")
        selected.append(match)
    return selected


def _invoke_solver(model, instance, time_limit_s: float, warm_start=None):
    """Normalize different solver APIs to a common benchmark call."""
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


class RouteFinderBenchmarkSolver:
    """Checkpoint-backed RouteFinder solver wrapper for benchmark use."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        dataset_dir: Path,
        num_customers: int,
        variant: str,
        decode_type: str = "sampling",
        num_samples: int = 100,
        select_best: bool = True,
        num_augment: int = 8,
        normalize_coords: bool = True,
    ):
        import torch
        from dvrptw_bench.rl.env_flexible import MTVRPFlexibleEnv
        from dvrptw_bench.rl.routefinder_adapter import (
            instance_to_routefinder_td,
            routefinder_actions_to_solution,
        )
        from dvrptw_bench.rl.mtvrp_solomon_generator import SolomonMTVRPGenerator
        from routefinder.envs.mtvrp import MTVRPEnv, MTVRPGenerator
        from routefinder.models import RouteFinderBase, RouteFinderPolicy
        from routefinder.utils import evaluate as evaluate_routefinder

        self.instance_to_routefinder_td = instance_to_routefinder_td
        self.routefinder_actions_to_solution = routefinder_actions_to_solution
        self.evaluate_routefinder = evaluate_routefinder
        self.decode_type = decode_type
        self.num_samples = num_samples
        self.select_best = select_best
        self.num_augment = num_augment
        self.normalize_coords = normalize_coords

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        if variant == "solomon_generated":
            pool: List[Dict[str, Any]] = []
            for path in _iter_solomon_instance_files(dataset_dir):
                instance = parse_solomon(path, max_customers=num_customers)
                pool.append(
                    _build_solomon_training_item(
                        instance,
                        num_customers=num_customers,
                        normalize_coords=normalize_coords,
                    )
                )
            generator = SolomonMTVRPGenerator(
                num_loc=num_customers,
                variant_preset="vrptw",
                solomon_instances=pool,
            )
            env = MTVRPEnv(generator, check_solution=False)
        elif variant == "with_lateness":
            generator = MTVRPGenerator(
                num_loc=num_customers,
                max_time=10.11,
                variant_preset="vrptw",
            )
            env = MTVRPFlexibleEnv(generator, check_solution=False)
        else:
            generator = MTVRPGenerator(
                num_loc=num_customers,
                max_time=10.11,
                variant_preset="vrptw",
            )
            env = MTVRPEnv(generator, check_solution=False)

        policy = RouteFinderPolicy(env_name=env.name).to(self.device)
        model = RouteFinderBase.load_from_checkpoint(
            str(checkpoint_path),
            env=env,
            policy=policy,
            map_location=torch.device('cpu'),
            weights_only=False,
        )
        self.env = env
        self.model = model.to(self.device).eval()
        self.policy = self.model.policy.to(self.device).eval()

    def solve(self, instance, time_limit_s=None, warm_start=None):
        import time as _time
        import torch

        _ = (time_limit_s, warm_start)
        t0 = _time.perf_counter()
        td = self.instance_to_routefinder_td(
            instance,
            normalize_coords=self.normalize_coords,
        ).to(self.device)
        td_reset = self.env.reset(td)

        with torch.inference_mode():
            if self.num_augment > 1:
                out = self.evaluate_routefinder(
                    self.model,
                    td_reset.clone(),
                    num_augment=self.num_augment,
                )
                actions = out.get(
                    "best_aug_actions",
                    out.get("best_multistart_actions", out.get("actions")),
                )
            elif self.decode_type == "sampling":
                out = self.policy(
                    td_reset.clone(),
                    self.env,
                    phase="test",
                    decode_type="sampling",
                    num_samples=self.num_samples,
                    select_best=self.select_best,
                    return_actions=True,
                )
                actions = out["actions"]
            else:
                out = self.policy(
                    td_reset.clone(),
                    self.env,
                    phase="test",
                    decode_type="greedy",
                    return_actions=True,
                )
                actions = out["actions"]

        solution = self.routefinder_actions_to_solution(
            actions,
            instance,
            strategy="routefinder",
        )
        solution.solve_time_s = _time.perf_counter() - t0
        solution.details.update(
            {
                "num_augment": self.num_augment,
                "decode_type": self.decode_type,
                "normalize_coords": self.normalize_coords,
            }
        )
        return solution


def _create_heuristic_model(model_name: str):
    """Create a heuristic model inside a worker process."""
    if model_name == "ortools":
        return ORToolsDVRPTWSolver(soft_time_windows=True)
    if model_name == "pyvrp":
        return PyVRPVRPTWSolver()
    if model_name == "nearest_neighbor":
        return PMCAVRPTWSolver()
    raise ValueError(f"Unsupported heuristic model for multiprocessing: {model_name}")


def _run_heuristic_task(task: HeuristicRunTask) -> BenchmarkResult:
    """Execute one heuristic benchmark task in a worker process."""
    instance = _parse_instance_for_size(Path(task.instance_path), task.evaluation_size)
    model = _create_heuristic_model(task.model_name)
    simulator = DynamicSimulator(instance)

    try:
        def solver_fn(instance, time_limit_s, warm_start=None):
            return _invoke_solver(model, instance, time_limit_s, warm_start)

        seed = task.random_seed + task.run_id
        final_solution, event_logs, scenario = simulator.run(
            solver_fn=solver_fn,
            epsilon=task.degree_of_dynamicity,
            budget_s=task.max_solve_time,
            seed=seed,
            cutoff_ratio=task.cutoff_time
        )

        if final_solution is None:
            raise ValueError("Simulation returned no solution")

        metrics = MetricsCalculator.calculate_metrics(
            instance, final_solution, task.oracle_cost
        )

        return BenchmarkResult(
            instance_id=task.instance_id,
            evaluation_size=task.evaluation_size,
            model_name=task.model_name,
            model_type="heuristic",
            degree_of_dynamicity=task.degree_of_dynamicity,
            cutoff_time=task.cutoff_time,
            allow_rejection=task.allow_rejection,
            total_distance=metrics['total_distance'],
            num_vehicles=metrics['num_vehicles'],
            computational_time=metrics['computational_time'],
            customers_served=metrics['customers_served'],
            customers_rejected=metrics['customers_rejected'],
            rejection_rate=metrics['rejection_rate'],
            average_lateness=metrics['average_lateness'],
            service_level=metrics['service_level'],
            total_cost=metrics['total_cost'],
            oracle_gap=metrics['oracle_gap'],
            run_id=task.run_id,
            timestamp=datetime.now().isoformat(),
            feasible=metrics['feasible']
        )
    except Exception as e:
        logger.error(f"    Error running {task.model_name}: {str(e)}")
        return BenchmarkResult(
            instance_id=task.instance_id,
            evaluation_size=task.evaluation_size,
            model_name=task.model_name,
            model_type="heuristic",
            degree_of_dynamicity=task.degree_of_dynamicity,
            cutoff_time=task.cutoff_time,
            allow_rejection=task.allow_rejection,
            total_distance=0,
            num_vehicles=0,
            computational_time=0,
            customers_served=0,
            customers_rejected=instance.n_customers,
            rejection_rate=1.0,
            average_lateness=0,
            service_level=0,
            total_cost=-1,
            oracle_gap=None,
            run_id=task.run_id,
            timestamp=datetime.now().isoformat(),
            feasible=False,
            error_message=str(e)
        )

# =====================================================================
# Model Loading
# =====================================================================

class ModelRegistry:
    """Registry for loading and managing different solver models."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """Load all specified models."""
        # Load heuristics
        for heuristic in self.config.heuristics:
            if heuristic == "ortools":
                self.models["ortools"] = ("heuristic", self._create_ortools_solver)
            elif heuristic == "pyvrp":
                self.models["pyvrp"] = ("heuristic", self._create_pyvrp_solver)
            elif heuristic == "nearest_neighbor":
                self.models["nearest_neighbor"] = ("heuristic", self._create_nn_solver)
        
        # Load AI models
        for ai_model in self.config.ai_models:
            if ai_model in AI_MODEL_SPECS:
                self.models[ai_model] = ("ai", lambda model_name=ai_model: self._create_routefinder_solver(model_name))
        
        # Add hybrid combinations if enabled
        if self.config.enable_hybrid:
            for heuristic in self.config.heuristics:
                for ai_model in self.config.ai_models:
                    hybrid_name = f"{ai_model}+{heuristic}"
                    self.models[hybrid_name] = ("hybrid", lambda h=heuristic, a=ai_model: 
                                                self._create_hybrid_solver(a, h))
        
        # Add oracle if enabled
        if self.config.enable_oracle:
            self.models["oracle"] = ("oracle", self._create_oracle_solver)
    
    def _create_ortools_solver(self):
        """Create ORTools solver instance."""
        return ORToolsDVRPTWSolver(
            soft_time_windows=True
        )
    
    def _create_pyvrp_solver(self):
        """Create PyVRP solver instance."""
        return PyVRPVRPTWSolver()
    
    def _create_nn_solver(self):
        """Create nearest neighbor solver instance."""
        return PMCAVRPTWSolver()  # Using PMCA as simple heuristic
    
    def _create_routefinder_solver(self, model_name: str):
        """Create RouteFinder solver instance."""
        spec = AI_MODEL_SPECS.get(model_name)
        if spec is None:
            raise ValueError(f"Unknown AI model: {model_name}")

        project_root = Path(__file__).resolve().parent.parent
        checkpoint = None
        for candidate in spec.get("checkpoint_candidates", []):
            candidate_path = project_root / candidate
            if candidate_path.exists():
                checkpoint = candidate_path
                break
        if checkpoint is None:
            raise FileNotFoundError(
                f"Checkpoint for {model_name} not found. Candidates: {spec.get('checkpoint_candidates', [])}"
            )

        logger.info(f"Loading RouteFinder model from {checkpoint}")
        dataset_dir = (Path(__file__).resolve().parent / self.config.dataset_dir).resolve()
        return RouteFinderBenchmarkSolver(
            checkpoint_path=checkpoint,
            dataset_dir=dataset_dir,
            num_customers=spec["num_customers"],
            variant=spec["variant"],
        )
    
    def _create_hybrid_solver(self, ai_model: str, heuristic: str):
        """Create hybrid solver combining AI and heuristic."""
        from dvrptw_bench.hybrid.hybrid_runner import run_hybrid

        ai_solver = self._create_routefinder_solver(ai_model)

        class HybridBenchmarkSolver:
            def solve(self, instance, time_limit_s, warm_start=None):
                _ = warm_start
                solution, timings = run_hybrid(
                    instance=instance,
                    policy=ai_solver.policy,
                    policy_name=ai_model,
                    budget_s=time_limit_s,
                )
                solution.solve_time_s = timings.get("total_s", 0.0)
                return solution

        return HybridBenchmarkSolver()
    
    def _create_oracle_solver(self):
        """Create oracle solver with full future knowledge."""
        return OracleSolver(time_limit=self.config.oracle_solve_time)
    
    def get_model(self, name: str):
        """Get a model instance by name."""
        if name not in self.models:
            raise ValueError(f"Unknown model: {name}")
        
        model_type, factory = self.models[name]
        return model_type, factory()

# =====================================================================
# Oracle Solver
# =====================================================================

class OracleSolver:
    """Oracle solver with full knowledge of future customers."""
    
    def __init__(self, time_limit: float = 300.0):
        self.time_limit = time_limit
        self.solver = ORToolsVRPTWSolver()
    
    def solve(self, instance, time_limit_s, warm_start=None):
        """Solve with full knowledge of all customers."""
        # Simply solve the full instance as it already has all customers
        return self.solver.solve(instance, time_limit_s=time_limit_s, warm_start=warm_start)

# =====================================================================
# Metrics Calculation
# =====================================================================

class MetricsCalculator:
    """Calculate comprehensive metrics for benchmark results."""
    
    @staticmethod
    def calculate_metrics(instance, solution, oracle_result: Optional[float] = None) -> Dict[str, Any]:
        """Calculate all metrics for a solution."""
        metrics = {
            'total_distance': 0.0,
            'num_vehicles': 0,
            'computational_time': solution.solve_time_s if solution else 0.0,
            'customers_served': 0,
            'customers_rejected': 0,
            'rejection_rate': 0.0,
            'average_lateness': 0.0,
            'service_level': 0.0,
            'total_cost': 0.0,
            'oracle_gap': None,
            'feasible': True
        }
        
        if not solution or not solution.routes:
            metrics['feasible'] = False
            metrics['customers_rejected'] = instance.n_customers
            metrics['rejection_rate'] = 1.0
            return metrics
        
        # Calculate basic metrics from solution object
        metrics['total_distance'] = solution.total_distance
        metrics['num_vehicles'] = len([r for r in solution.routes if r.node_ids])
        
        # Count served customers
        served_customers = set()
        for route in solution.routes:
            served_customers.update(route.node_ids)
        metrics['customers_served'] = len(served_customers)
        
        total_customers = instance.n_customers
        metrics['customers_rejected'] = total_customers - len(served_customers)
        metrics['rejection_rate'] = metrics['customers_rejected'] / max(1, total_customers)
        
        # Get violation metrics if available
        if hasattr(solution, 'violations') and solution.violations:
            violations = solution.violations
            metrics['average_lateness'] = violations.get('late_sum', 0.0) / max(1, metrics['customers_served'])
            late_count = violations.get('late_count', 0.0)
            metrics['service_level'] = 1.0 - (late_count / max(1, metrics['customers_served']))
        else:
            metrics['service_level'] = 1.0 if solution.feasible else 0.0
        
        # Calculate total cost (distance + vehicle fixed cost)
        vehicle_cost = 100.0  # Fixed cost per vehicle
        metrics['total_cost'] = metrics['total_distance'] + (metrics['num_vehicles'] * vehicle_cost)
        
        # Calculate oracle gap if available
        if oracle_result is not None:
            metrics['oracle_gap'] = ((metrics['total_cost'] - oracle_result) / oracle_result) * 100
        
        # Check feasibility
        metrics['feasible'] = solution.feasible if hasattr(solution, 'feasible') else True
        
        return metrics

# =====================================================================
# Dynamic Simulation Runner
# =====================================================================

class DynamicBenchmarkRunner:
    """Run dynamic VRPTW simulations with various models."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.model_registry = ModelRegistry(config)
        self.metrics_calculator = MetricsCalculator()
        self.results = []

    @staticmethod
    def _persist_partial_progress(
        result_manager: Optional["ResultManager"],
        instance_results: List[BenchmarkResult],
    ) -> None:
        if result_manager is None or not instance_results:
            return
        partial_results = [*result_manager.existing_results, *instance_results]
        result_manager.save_partial_results(partial_results)
    
    def run_instance(self, instance_path: str, instance_id: str,
                     result_manager: Optional["ResultManager"] = None) -> List[BenchmarkResult]:
        """Run benchmark on a single instance."""
        logger.info(f"Running benchmark on instance: {instance_id}")
        instance_results = []

        for evaluation_size in self.config.evaluation_sizes:
            instance = _parse_instance_for_size(Path(instance_path), evaluation_size)
            logger.info(f"  Evaluation size: {evaluation_size}")

            for dod in self.config.degrees_of_dynamicity:
                for cutoff in self.config.cutoff_times:
                    logger.info(f"    DOD: {dod:.1%}, Cutoff: {cutoff:.1%}")

                    oracle_cost = None
                    if self.config.enable_oracle:
                        oracle_cost = self._run_oracle(instance)

                    heuristic_tasks: List[HeuristicRunTask] = []
                    other_models: List[Tuple[str, str]] = []

                    for model_name in self.model_registry.models.keys():
                        if model_name == "oracle":
                            continue

                        model_type, _ = self.model_registry.models[model_name]
                        if model_type == "ai":
                            model_size = AI_MODEL_SPECS[model_name]["num_customers"]
                            if model_size != evaluation_size:
                                continue

                        if model_type == "heuristic":
                            for run_id in range(self.config.num_runs):
                                if result_manager and result_manager.should_skip(
                                    instance_id, evaluation_size, model_name, dod, cutoff, run_id
                                ):
                                    continue
                                heuristic_tasks.append(HeuristicRunTask(
                                    instance_path=instance_path,
                                    instance_id=instance_id,
                                    evaluation_size=evaluation_size,
                                    model_name=model_name,
                                    degree_of_dynamicity=dod,
                                    cutoff_time=cutoff,
                                    run_id=run_id,
                                    oracle_cost=oracle_cost,
                                    allow_rejection=self.config.allow_rejection,
                                    max_solve_time=self.config.max_solve_time,
                                    random_seed=self.config.random_seed
                                ))
                        else:
                            other_models.append((model_name, model_type))

                    instance_results.extend(self._run_heuristics_parallel(heuristic_tasks))
                    self._persist_partial_progress(result_manager, instance_results)

                    for model_name, model_type in other_models:
                        for run_id in range(self.config.num_runs):
                            if result_manager and result_manager.should_skip(
                                instance_id, evaluation_size, model_name, dod, cutoff, run_id
                            ):
                                continue
                            result = self._run_single_model(
                                instance_id, evaluation_size, instance,
                                model_name, model_type, dod, cutoff,
                                run_id, oracle_cost
                            )
                            instance_results.append(result)
                            self._persist_partial_progress(result_manager, instance_results)
        
        return instance_results

    def _run_heuristics_parallel(self, tasks: List[HeuristicRunTask]) -> List[BenchmarkResult]:
        """Execute heuristic runs in separate processes."""
        if not tasks:
            return []

        max_workers = self.config.heuristic_workers or os.cpu_count() or 1
        max_workers = max(1, min(max_workers, len(tasks)))

        logger.info(
            f"    Running {len(tasks)} heuristic task(s) with {max_workers} worker process(es)"
        )

        if max_workers == 1:
            return [_run_heuristic_task(task) for task in tasks]

        results: List[BenchmarkResult] = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(_run_heuristic_task, task): task
                for task in tasks
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    logger.error(f"    Heuristic worker crashed for {task.model_name}: {exc}")
                    results.append(BenchmarkResult(
                        instance_id=task.instance_id,
                        evaluation_size=task.evaluation_size,
                        model_name=task.model_name,
                        model_type="heuristic",
                        degree_of_dynamicity=task.degree_of_dynamicity,
                        cutoff_time=task.cutoff_time,
                        allow_rejection=task.allow_rejection,
                        total_distance=0,
                        num_vehicles=0,
                        computational_time=0,
                        customers_served=0,
                        customers_rejected=0,
                        rejection_rate=1.0,
                        average_lateness=0,
                        service_level=0,
                        total_cost=float('inf'),
                        oracle_gap=None,
                        run_id=task.run_id,
                        timestamp=datetime.now().isoformat(),
                        feasible=False,
                        error_message=str(exc)
                    ))

        return results
    
    def _run_oracle(self, instance):
        """Run oracle solver with full knowledge."""
        logger.info("    Running oracle solver...")
        try:
            oracle = self.model_registry.get_model("oracle")[1]
            # Oracle sees the full instance
            solution = _invoke_solver(oracle, instance, self.config.oracle_solve_time, warm_start=None)
            
            if solution:
                metrics = self.metrics_calculator.calculate_metrics(instance, solution)
                return metrics['total_cost']
        except Exception as e:
            logger.error(f"Oracle failed: {e}")
        return None
    
    def _run_single_model(self, instance_id: str, evaluation_size: int, instance,
                         model_name: str, model_type: str, dod: float, cutoff: float,
                         run_id: int, oracle_cost: Optional[float]) -> BenchmarkResult:
        """Run a single model on the instance."""
        logger.info(f"    Running {model_name} (run {run_id + 1}/{self.config.num_runs})")
        
        try:
            # Get model
            _, model = self.model_registry.get_model(model_name)
            if model is None:
                raise ValueError(f"Failed to load model {model_name}")
            
            # Create simulator with the proper instance
            simulator = DynamicSimulator(instance)
            
            # Create solver function with proper signature
            def solver_fn(instance, time_limit_s, warm_start=None):
                return _invoke_solver(model, instance, time_limit_s, warm_start)
            
            # Run simulation with correct parameters
            seed = self.config.random_seed + run_id  # Different seed for each run
            final_solution, event_logs, scenario = simulator.run(
                solver_fn=solver_fn,
                epsilon=dod,
                budget_s=self.config.max_solve_time,
                seed=seed,
                cutoff_ratio=cutoff
            )
            
            if final_solution is None:
                raise ValueError("Simulation returned no solution")
            
            # Calculate metrics
            metrics = self.metrics_calculator.calculate_metrics(
                instance, final_solution, oracle_cost
            )
            
            # Create result
            result = BenchmarkResult(
                instance_id=instance_id,
                evaluation_size=evaluation_size,
                model_name=model_name,
                model_type=model_type,
                degree_of_dynamicity=dod,
                cutoff_time=cutoff,
                allow_rejection=self.config.allow_rejection,
                total_distance=metrics['total_distance'],
                num_vehicles=metrics['num_vehicles'],
                computational_time=metrics['computational_time'],
                customers_served=metrics['customers_served'],
                customers_rejected=metrics['customers_rejected'],
                rejection_rate=metrics['rejection_rate'],
                average_lateness=metrics['average_lateness'],
                service_level=metrics['service_level'],
                total_cost=metrics['total_cost'],
                oracle_gap=metrics['oracle_gap'],
                run_id=run_id,
                timestamp=datetime.now().isoformat(),
                feasible=metrics['feasible']
            )
            
        except Exception as e:
            logger.error(f"    Error running {model_name}: {str(e)}")
            # Create error result
            result = BenchmarkResult(
                instance_id=instance_id,
                evaluation_size=evaluation_size,
                model_name=model_name,
                model_type=model_type,
                degree_of_dynamicity=dod,
                cutoff_time=cutoff,
                allow_rejection=self.config.allow_rejection,
                total_distance=0,
                num_vehicles=0,
                computational_time=0,
                customers_served=0,
                customers_rejected=instance.n_customers,
                rejection_rate=1.0,
                average_lateness=0,
                service_level=0,
                total_cost=float('inf'),
                oracle_gap=None,
                run_id=run_id,
                timestamp=datetime.now().isoformat(),
                feasible=False,
                error_message=str(e)
            )
        
        return result

# =====================================================================
# Result Storage and Loading
# =====================================================================

class ResultManager:
    """Manage saving and loading of benchmark results."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Define paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_path = self.results_dir / f"benchmark_results_{timestamp}.json"
        self.csv_path = self.results_dir / f"benchmark_results_{timestamp}.csv"
        self.checkpoint_path = self.results_dir / "checkpoint.json"
        
        # Load existing results if not wiping
        self.existing_results = []
        if not config.wipe_results:
            self.existing_results = self.load_checkpoint()
    
    def load_checkpoint(self) -> List[BenchmarkResult]:
        """Load existing results from checkpoint."""
        if not self.checkpoint_path.exists():
            return []
        
        try:
            with open(self.checkpoint_path, 'r') as f:
                data = json.load(f)
            
            results = []
            for item in data:
                results.append(BenchmarkResult(**item))
            
            logger.info(f"Loaded {len(results)} existing results from checkpoint")
            return results
        
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return []
    
    def save_checkpoint(self, results: List[BenchmarkResult]):
        """Save current results to checkpoint."""
        data = [asdict(r) for r in results]
        
        with open(self.checkpoint_path, 'w') as f:
            json.dump(data, f, indent=2)

    def save_partial_results(self, results: List[BenchmarkResult]):
        """Persist partial benchmark output during long-running executions."""
        if not results:
            return

        partial_json = self.results_dir / "partial_results.json"
        partial_csv = self.results_dir / "partial_results.csv"
        data = [asdict(r) for r in results]

        with open(partial_json, 'w') as f:
            json.dump(data, f, indent=2)

        df = pd.DataFrame(data)
        df.to_csv(partial_csv, index=False)

        self.save_checkpoint(results)
    
    def should_skip(self, instance_id: str, evaluation_size: int, model_name: str, dod: float,
                    cutoff: float, run_id: int) -> bool:
        """Check if a specific run should be skipped."""
        if self.config.overwrite_existing:
            return False
        
        for result in self.existing_results:
            if (result.instance_id == instance_id and
                result.evaluation_size == evaluation_size and
                result.model_name == model_name and
                abs(result.degree_of_dynamicity - dod) < 0.01 and
                abs(result.cutoff_time - cutoff) < 0.01 and
                result.run_id == run_id):
                return True
        
        return False
    
    def save_results(self, results: List[BenchmarkResult]):
        """Save final results to files."""
        # Save JSON
        if self.config.output_format in ["json", "both"]:
            data = [asdict(r) for r in results]
            with open(self.json_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved results to {self.json_path}")
        
        # Save CSV
        if self.config.output_format in ["csv", "both"]:
            df = pd.DataFrame([asdict(r) for r in results])
            df.to_csv(self.csv_path, index=False)
            logger.info(f"Saved results to {self.csv_path}")
        
        # Update checkpoint
        self.save_checkpoint(results)
    
    def generate_summary(self, results: List[BenchmarkResult]):
        """Generate summary statistics."""
        df = pd.DataFrame([asdict(r) for r in results])
        
        # Group by model and parameters
        summary = df.groupby(['evaluation_size', 'model_name', 'model_type', 'degree_of_dynamicity', 'cutoff_time']).agg({
            'total_distance': ['mean', 'std'],
            'num_vehicles': ['mean', 'std'],
            'computational_time': ['mean', 'std'],
            'service_level': ['mean', 'std'],
            'rejection_rate': ['mean', 'std'],
            'oracle_gap': ['mean', 'std']
        }).round(2)
        
        # Save summary
        summary_path = self.results_dir / "summary_statistics.csv"
        summary.to_csv(summary_path)
        logger.info(f"Saved summary to {summary_path}")
        
        return summary

# =====================================================================
# Main Benchmark Execution
# =====================================================================

def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description="DVRPTW Benchmark Script")

    parser.add_argument("--sizes", type=int, nargs="+",
                       default=[50, 75],
                       help="Customer counts to evaluate on truncated Solomon instances")
    
    # Dynamic parameters
    parser.add_argument("--dod", type=float, nargs="+", 
                       default=[0.3, 0.5, 0.7],
                       help="Degrees of dynamicity to test")
    parser.add_argument("--cutoff", type=float, nargs="+",
                       default=[0.5, 0.7, 0.9],
                       help="Cutoff times to test")
    parser.add_argument("--allow-rejection", action="store_true",
                       help="Allow customer rejection")
    
    # Execution parameters
    parser.add_argument("--num-runs", type=int, default=3,
                       help="Number of runs per configuration")
    parser.add_argument("--wipe-results", action="store_true",
                       help="Wipe existing results before running")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Model selection
    parser.add_argument("--heuristics", nargs="+",
                       default=["ortools"],
                       help="Heuristic models to test")
    parser.add_argument("--ai-models", nargs="+",
                       default=list(AI_MODEL_SPECS.keys()),
                       help="AI models to test")
    parser.add_argument("--no-hybrid", action="store_true",
                       help="Disable hybrid models")
    parser.add_argument("--no-oracle", action="store_true",
                       help="Disable oracle solver")
    
    # Output configuration
    parser.add_argument("--results-dir", default="results",
                       help="Directory for results")
    parser.add_argument("--output-format", choices=["json", "csv", "both"],
                       default="both",
                       help="Output format")
    
    # Resource limits
    parser.add_argument("--max-time", type=float, default=5.0,
                       help="Max solve time per model (seconds)")
    parser.add_argument("--oracle-time", type=float, default=60.0,
                       help="Max solve time for oracle (seconds)")
    parser.add_argument("--heuristic-workers", type=int, default=None,
                       help="Worker processes for heuristic models (default: CPU count)")
    
    # Instance selection
    parser.add_argument("--instances", nargs="+",
                       help="Specific instances to test (default: all)")
    parser.add_argument("--dataset-dir", default="../dataset",
                       help="Dataset directory path")
    
    args = parser.parse_args()
    
    # Create configuration
    config = BenchmarkConfig(
        dataset_dir=args.dataset_dir,
        evaluation_sizes=args.sizes,
        degrees_of_dynamicity=args.dod,
        cutoff_times=args.cutoff,
        allow_rejection=args.allow_rejection,
        num_runs=args.num_runs,
        wipe_results=args.wipe_results,
        overwrite_existing=args.overwrite,
        random_seed=args.seed,
        heuristics=args.heuristics,
        ai_models=args.ai_models,
        enable_hybrid=not args.no_hybrid,
        enable_oracle=not args.no_oracle,
        results_dir=args.results_dir,
        output_format=args.output_format,
        max_solve_time=args.max_time,
        oracle_solve_time=args.oracle_time,
        heuristic_workers=args.heuristic_workers
    )
    
    # Set random seed
    np.random.seed(config.random_seed)
    
    # Initialize components
    runner = DynamicBenchmarkRunner(config)
    result_manager = ResultManager(config)
    
    # Get instances to test
    dataset_dir = Path(args.dataset_dir)
    if args.instances:
        instances = _resolve_instance_selection(dataset_dir, args.instances)
    else:
        instances = _iter_solomon_instance_files(dataset_dir)
    
    if not instances:
        logger.error(f"No instances found in {dataset_dir}")
        return
    
    logger.info(f"Testing {len(instances)} instances")
    logger.info(f"Models: {list(runner.model_registry.models.keys())}")
    logger.info(f"Evaluation sizes: {config.evaluation_sizes}")
    logger.info(f"DOD values: {config.degrees_of_dynamicity}")
    logger.info(f"Cutoff times: {config.cutoff_times}")
    
    # Run benchmark
    all_results = result_manager.existing_results.copy()
    
    for instance_path in instances:
        instance_id = Path(instance_path).stem
        
        # Check if we should skip this instance entirely
        skip_instance = True
        for evaluation_size in config.evaluation_sizes:
            for model_name in runner.model_registry.models.keys():
                if model_name == "oracle":
                    continue
                model_type, _ = runner.model_registry.models[model_name]
                if model_type == "ai" and AI_MODEL_SPECS[model_name]["num_customers"] != evaluation_size:
                    continue
                for dod in config.degrees_of_dynamicity:
                    for cutoff in config.cutoff_times:
                        for run_id in range(config.num_runs):
                            if not result_manager.should_skip(
                                instance_id, evaluation_size, model_name, dod, cutoff, run_id
                            ):
                                skip_instance = False
                                break
        
        if skip_instance:
            logger.info(f"Skipping instance {instance_id} (already complete)")
            continue
        
        # Run instance
        try:
            instance_results = runner.run_instance(
                str(instance_path), instance_id, result_manager=result_manager
            )
            all_results.extend(instance_results)
            
            # Save partial outputs after each completed instance.
            result_manager.save_partial_results(all_results)
            
        except Exception as e:
            logger.error(f"Failed to process instance {instance_id}: {e}")
            continue
    
    # Save final results
    logger.info("Saving final results...")
    result_manager.save_results(all_results)
    
    # Generate summary
    logger.info("Generating summary statistics...")
    summary = result_manager.generate_summary(all_results)
    print("\n=== Summary Statistics ===")
    print(summary)
    
    logger.info("Benchmark complete!")

if __name__ == "__main__":
    main()

"""Heuristics CLI."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console

from dvrptw_bench.cli.common_options import ensure_run_dir, git_hash, now_run_id
from dvrptw_bench.common.rng import set_seed
from dvrptw_bench.common.typing import RecordTimes, ResultRecord
from dvrptw_bench.data.instance_filters import find_rc_instances
from dvrptw_bench.data.solomon_parser import parse_solomon
from dvrptw_bench.dynamic.simulator import DynamicSimulator
from dvrptw_bench.heuristics.constructive_pmca import PMCAVRPTWSolver
from dvrptw_bench.heuristics.gls import GLSSolver
from dvrptw_bench.heuristics.gurobi_solver import GurobiVRPTWSolver
from dvrptw_bench.heuristics.lkh3_solver import LKH3VRPTWSolver
from dvrptw_bench.heuristics.ortools_solver import ORToolsVRPTWSolver
from dvrptw_bench.heuristics.pyvrp_solver import PyVRPVRPTWSolver
from dvrptw_bench.results.aggregator import Aggregator
from dvrptw_bench.results.export import write_report
from dvrptw_bench.results.recorder import Recorder
from dvrptw_bench.viz.convergence_plot import plot_convergence
from dvrptw_bench.viz.route_plot import plot_routes
from dvrptw_bench.viz.timeline_plot import plot_timeline

app = typer.Typer(help="Heuristics pipeline")
console = Console()


def _build_solver(
    name: str,
    lkh3_binary: str | None = None,
    debug_gls: bool = False,
    gls_log_every: int = 1,
    gls_seed: int | None = None,
):
    key = name.lower()
    if key == "pmca":
        return PMCAVRPTWSolver()
    if key == "gls":
        return GLSSolver(debug=debug_gls, log_every=gls_log_every, seed=gls_seed)
    if key == "ortools":
        return ORToolsVRPTWSolver()
    if key == "pyvrp":
        return PyVRPVRPTWSolver()
    if key == "gurobi":
        return GurobiVRPTWSolver()
    if key == "lkh3":
        return LKH3VRPTWSolver(binary_path=lkh3_binary)
    raise typer.BadParameter(f"Unknown solver {name}")


def _find_instance(dataset_root: Path, instance: str) -> Path:
    matches = [p for p in find_rc_instances(dataset_root) if p.name == instance or p.stem == instance]
    if not matches:
        raise typer.BadParameter(f"Instance not found: {instance}")
    return matches[0]


@app.command("list-instances")
def list_instances(dataset_root: Path = Path("./dataset")):
    files = find_rc_instances(dataset_root)
    for p in files:
        console.print(str(p))
    console.print(f"Found {len(files)} RC instances")


@app.command("solve-static")
def solve_static(
    solver: str = typer.Option("ortools"),
    instance: str = typer.Option(...),
    time_limit: float = typer.Option(10.0),
    dataset_root: Path = typer.Option(Path("./dataset")),
    output_root: Path = typer.Option(Path("./outputs")),
    seed: int = typer.Option(1),
    lkh3_binary: str | None = typer.Option(None),
    debug_gls: bool = typer.Option(False, "--debug-gls", help="Enable per-iteration GLS debug logs."),
    gls_log_every: int = typer.Option(1, "--gls-log-every", help="Log every N iterations when GLS debug is enabled."),
):
    set_seed(seed)
    inst_path = _find_instance(dataset_root, instance)
    inst = parse_solomon(inst_path)
    solver_obj = _build_solver(
        solver,
        lkh3_binary=lkh3_binary,
        debug_gls=debug_gls,
        gls_log_every=gls_log_every,
        gls_seed=seed,
    )

    run_id = now_run_id("heur_static")
    run_dir = ensure_run_dir(output_root, run_id)
    recorder = Recorder(run_dir)

    sol = solver_obj.solve(inst, time_limit)
    route_path = plot_routes(inst, sol, run_dir / "route.png")
    timeline_path = plot_timeline(inst, sol, run_dir / "timeline.png")
    if "convergence" in sol.details:
        plot_convergence(sol.details["convergence"], run_dir / "convergence.png")

    rec = ResultRecord(
        run_id=run_id,
        timestamp=datetime.now(),
        git_hash=git_hash(),
        strategy=sol.strategy,
        solver_details=solver,
        budget_s=time_limit,
        epsilon=0.0,
        seed=seed,
        instance_id=inst.instance_id,
        n_customers=inst.n_customers,
        total_distance=sol.total_distance,
        feasible=sol.feasible,
        violations=sol.violations,
        compute_times=RecordTimes(total_s=sol.solve_time_s),
        artifacts={"route": str(route_path), "timeline": str(timeline_path)},
        mode="static",
    )
    recorder.log(rec)
    pq = recorder.flush_parquet()
    summary = Aggregator(run_dir).summarize(pq)
    write_report(run_dir, summary)
    console.print(f"Done. Output: {run_dir}")


@app.command("solve-dynamic")
def solve_dynamic(
    solver: str = typer.Option("ortools"),
    instance: str = typer.Option(...),
    epsilon: float = typer.Option(0.5),
    budget: float = typer.Option(10.0),
    dataset_root: Path = typer.Option(Path("./dataset")),
    output_root: Path = typer.Option(Path("./outputs")),
    seed: int = typer.Option(1),
    lkh3_binary: str | None = typer.Option(None),
    debug_gls: bool = typer.Option(False, "--debug-gls", help="Enable per-iteration GLS debug logs."),
    gls_log_every: int = typer.Option(1, "--gls-log-every", help="Log every N iterations when GLS debug is enabled."),
):
    set_seed(seed)
    inst_path = _find_instance(dataset_root, instance)
    inst = parse_solomon(inst_path)
    solver_obj = _build_solver(
        solver,
        lkh3_binary=lkh3_binary,
        debug_gls=debug_gls,
        gls_log_every=gls_log_every,
        gls_seed=seed,
    )

    sim = DynamicSimulator(inst)
    run_id = now_run_id("heur_dyn")
    run_dir = ensure_run_dir(output_root, run_id)
    recorder = Recorder(run_dir)

    sol, events, scenario = sim.run(lambda snap_inst, t: solver_obj.solve(snap_inst, t), epsilon=epsilon, budget_s=budget, seed=seed)
    if sol is None:
        console.print(f"Scenario infeasible and discarded: {scenario.dropped_reason}")
        raise typer.Exit(2)

    route_path = plot_routes(scenario.instance, sol, run_dir / "route.png")
    rec = ResultRecord(
        run_id=run_id,
        timestamp=datetime.now(),
        git_hash=git_hash(),
        strategy=sol.strategy,
        solver_details=solver,
        budget_s=budget,
        epsilon=epsilon,
        seed=seed,
        instance_id=inst.instance_id,
        n_customers=inst.n_customers,
        total_distance=sol.total_distance,
        feasible=sol.feasible,
        violations=sol.violations,
        compute_times=RecordTimes(total_s=sol.solve_time_s),
        dynamic_events=events,
        artifacts={"route": str(route_path)},
        mode="dynamic",
    )
    recorder.log(rec)
    pq = recorder.flush_parquet()
    summary = Aggregator(run_dir).summarize(pq)
    write_report(run_dir, summary)
    console.print(f"Done. Output: {run_dir}")


@app.command("batch")
def batch(
    solver: str = typer.Option("pmca"),
    epsilon_grid: str = typer.Option("0,0.25,0.5,0.75"),
    budgets: str = typer.Option("5,10,40,120"),
    dataset_root: Path = typer.Option(Path("./dataset")),
    output_root: Path = typer.Option(Path("./outputs")),
    seed: int = typer.Option(1),
    debug_gls: bool = typer.Option(False, "--debug-gls", help="Enable per-iteration GLS debug logs."),
    gls_log_every: int = typer.Option(1, "--gls-log-every", help="Log every N iterations when GLS debug is enabled."),
):
    eps = [float(x) for x in epsilon_grid.split(",") if x]
    bgs = [float(x) for x in budgets.split(",") if x]
    files = find_rc_instances(dataset_root)
    run_id = now_run_id("heur_batch")
    run_dir = ensure_run_dir(output_root, run_id)
    recorder = Recorder(run_dir)

    solver_obj = _build_solver(
        solver,
        debug_gls=debug_gls,
        gls_log_every=gls_log_every,
        gls_seed=seed,
    )
    for p in files:
        inst = parse_solomon(p)
        for e in eps:
            for b in bgs:
                sim = DynamicSimulator(inst)
                sol, events, scenario = sim.run(lambda snap_inst, t: solver_obj.solve(snap_inst, t), epsilon=e, budget_s=b, seed=seed)
                if sol is None:
                    continue
                recorder.log(
                    ResultRecord(
                        run_id=run_id,
                        timestamp=datetime.now(),
                        git_hash=git_hash(),
                        strategy=sol.strategy,
                        solver_details=solver,
                        budget_s=b,
                        epsilon=e,
                        seed=seed,
                        instance_id=inst.instance_id,
                        n_customers=inst.n_customers,
                        total_distance=sol.total_distance,
                        feasible=sol.feasible,
                        violations=sol.violations,
                        compute_times=RecordTimes(total_s=sol.solve_time_s),
                        dynamic_events=events,
                        mode="dynamic",
                    )
                )

    pq = recorder.flush_parquet()
    summary = Aggregator(run_dir).summarize(pq)
    write_report(run_dir, summary)
    console.print(f"Done. Output: {run_dir}")


if __name__ == "__main__":
    app()

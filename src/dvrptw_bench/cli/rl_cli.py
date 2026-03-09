"""RL CLI."""

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
from dvrptw_bench.metrics.objective import total_distance
from dvrptw_bench.results.aggregator import Aggregator
from dvrptw_bench.results.export import write_report
from dvrptw_bench.results.recorder import Recorder
from dvrptw_bench.rl.ga_baseline import GAPolicy
from dvrptw_bench.rl.policies import build_policy
from dvrptw_bench.rl.qlearning_baseline import QLearningPolicy
from dvrptw_bench.rl.rl4co_policy import RL4COPolicy
from dvrptw_bench.viz.convergence_plot import plot_convergence
from dvrptw_bench.viz.route_plot import plot_routes

app = typer.Typer(help="RL pipeline")
console = Console()


def _instance(dataset_root: Path, name: str):
    for p in find_rc_instances(dataset_root):
        if p.name == name or p.stem == name:
            return parse_solomon(p)
    raise typer.BadParameter(f"Instance not found: {name}")


@app.command("train-rl4co")
def train_rl4co(epochs: int = 1, train_size: int = 128, val_size: int = 32, device: str = "cpu"):
    policy = RL4COPolicy()
    out = policy.train(epochs=epochs, train_size=train_size, val_size=val_size, device=device)
    console.print(out)


@app.command("train-ga")
def train_ga(instance: str, dataset_root: Path = Path("./dataset"), time_limit: float = 10.0):
    inst = _instance(dataset_root, instance)
    ga = GAPolicy()
    out = ga.train(inst, time_limit_s=time_limit)
    console.print({"status": "ok", "points": len(out.get("curve", []))})


@app.command("train-qlearning")
def train_qlearning(episodes: int = 200):
    q = QLearningPolicy()
    q.train(episodes=episodes)
    console.print({"status": "ok", "episodes": episodes})


@app.command("eval-static")
def eval_static(
    policy: str = typer.Option("ga"),
    instance: str = typer.Option(...),
    dataset_root: Path = typer.Option(Path("./dataset")),
    output_root: Path = typer.Option(Path("./outputs")),
    seed: int = 1,
):
    set_seed(seed)
    inst = _instance(dataset_root, instance)
    pol = build_policy(policy)
    if hasattr(pol, "infer_solution"):
        sol = pol.infer_solution(inst)
    elif hasattr(pol, "infer_instance"):
        sol = pol.infer_instance(inst)
    else:
        from dvrptw_bench.dynamic.snapshot import SnapshotState

        snap = SnapshotState(time=0.0, remaining_customers=inst.customers, active_customer_ids={c.id for c in inst.customers}, served_customer_ids=set(), vehicles=[])
        sol = pol.infer(snap)
    sol.total_distance = total_distance(inst, sol)

    run_id = now_run_id("rl_static")
    run_dir = ensure_run_dir(output_root, run_id)
    recorder = Recorder(run_dir)
    route = plot_routes(inst, sol, run_dir / "route.png")
    if "convergence" in sol.details:
        plot_convergence(sol.details["convergence"], run_dir / "convergence.png")

    recorder.log(
        ResultRecord(
            run_id=run_id,
            timestamp=datetime.now(),
            git_hash=git_hash(),
            strategy=sol.strategy,
            solver_details=policy,
            budget_s=0.0,
            epsilon=0.0,
            seed=seed,
            instance_id=inst.instance_id,
            n_customers=inst.n_customers,
            total_distance=sol.total_distance,
            feasible=sol.feasible,
            violations=sol.violations,
            compute_times=RecordTimes(total_s=sol.solve_time_s),
            artifacts={"route": str(route)},
            mode="static",
        )
    )
    pq = recorder.flush_parquet()
    summary = Aggregator(run_dir).summarize(pq)
    write_report(run_dir, summary)
    console.print(f"Done. Output: {run_dir}")


@app.command("eval-dynamic")
def eval_dynamic(
    policy: str = typer.Option("ga"),
    instance: str = typer.Option(...),
    epsilon: float = 0.5,
    budget: float = 10.0,
    dataset_root: Path = Path("./dataset"),
    output_root: Path = Path("./outputs"),
    seed: int = 1,
):
    set_seed(seed)
    inst = _instance(dataset_root, instance)
    pol = build_policy(policy)

    from dvrptw_bench.dynamic.snapshot import SnapshotState

    def solve_fn(snap_inst, _budget):
        snap = SnapshotState(time=0.0, remaining_customers=snap_inst.customers, active_customer_ids={c.id for c in snap_inst.customers}, served_customer_ids=set(), vehicles=[])
        out = pol.infer(snap)
        out.total_distance = total_distance(snap_inst, out)
        return out

    sim = DynamicSimulator(inst)
    sol, events, scenario = sim.run(solve_fn, epsilon=epsilon, budget_s=budget, seed=seed)
    if sol is None:
        raise typer.Exit(2)

    run_id = now_run_id("rl_dynamic")
    run_dir = ensure_run_dir(output_root, run_id)
    recorder = Recorder(run_dir)
    recorder.log(
        ResultRecord(
            run_id=run_id,
            timestamp=datetime.now(),
            git_hash=git_hash(),
            strategy=sol.strategy,
            solver_details=policy,
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
            mode="dynamic",
        )
    )
    pq = recorder.flush_parquet()
    summary = Aggregator(run_dir).summarize(pq)
    write_report(run_dir, summary)
    console.print(f"Done. Output: {run_dir}")


if __name__ == "__main__":
    app()

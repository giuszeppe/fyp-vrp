"""Hybrid CLI."""

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
from dvrptw_bench.hybrid.hybrid_runner import run_hybrid
from dvrptw_bench.results.aggregator import Aggregator
from dvrptw_bench.results.export import write_report
from dvrptw_bench.results.recorder import Recorder

app = typer.Typer(help="Hybrid pipeline")
console = Console()


def _instance(dataset_root: Path, name: str):
    for p in find_rc_instances(dataset_root):
        if p.name == name or p.stem == name:
            return parse_solomon(p)
    raise typer.BadParameter(f"Instance not found: {name}")


@app.command("eval-hybrid")
def eval_hybrid(
    policy: str = typer.Option("ga"),
    instance: str = typer.Option(...),
    epsilon: float = typer.Option(0.5),
    budget: float = typer.Option(10.0),
    gls_time_share: float = typer.Option(0.9),
    feasibility: str = typer.Option("repair"),
    fallback_heuristic: str = typer.Option("pmca"),
    dataset_root: Path = typer.Option(Path("./dataset")),
    output_root: Path = typer.Option(Path("./outputs")),
    seed: int = typer.Option(1),
    debug_gls: bool = typer.Option(False, "--debug-gls", help="Enable per-iteration GLS debug logs."),
    gls_log_every: int = typer.Option(1, "--gls-log-every", help="Log every N iterations when GLS debug is enabled."),
):
    set_seed(seed)
    inst = _instance(dataset_root, instance)

    def solve_fn(snap_inst, t):
        sol, timings = run_hybrid(
            snap_inst,
            policy_name=policy,
            budget_s=t,
            gls_time_share=gls_time_share,
            feasibility_mode=feasibility,
            gls_debug=debug_gls,
            gls_log_every=gls_log_every,
        )
        sol.details["timings"] = timings
        return sol

    sim = DynamicSimulator(inst)
    sol, events, scenario = sim.run(solve_fn, epsilon=epsilon, budget_s=budget, seed=seed)
    if sol is None:
        console.print(f"Discarded scenario: {scenario.dropped_reason}")
        raise typer.Exit(2)

    run_id = now_run_id("hybrid")
    run_dir = ensure_run_dir(output_root, run_id)
    recorder = Recorder(run_dir)
    timings = sol.details.get("timings", {"total_s": sol.solve_time_s, "inference_s": 0.0, "local_search_s": 0.0})
    recorder.log(
        ResultRecord(
            run_id=run_id,
            timestamp=datetime.now(),
            git_hash=git_hash(),
            strategy=f"hybrid/{policy}_gls",
            solver_details=f"mode={feasibility},fallback={fallback_heuristic}",
            budget_s=budget,
            epsilon=epsilon,
            seed=seed,
            instance_id=inst.instance_id,
            n_customers=inst.n_customers,
            total_distance=sol.total_distance,
            feasible=sol.feasible,
            violations=sol.violations,
            compute_times=RecordTimes(**{k: float(v) for k, v in timings.items()}),
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

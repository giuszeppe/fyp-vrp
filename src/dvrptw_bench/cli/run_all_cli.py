"""Run-all benchmark CLI."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import hydra
import typer
from omegaconf import DictConfig
from rich.console import Console

from dvrptw_bench.cli.common_options import ensure_run_dir, git_hash, now_run_id
from dvrptw_bench.common.rng import set_seed
from dvrptw_bench.common.typing import RecordTimes, ResultRecord
from dvrptw_bench.data.instance_filters import find_rc_instances
from dvrptw_bench.data.solomon_parser import parse_solomon
from dvrptw_bench.dynamic.simulator import DynamicSimulator
from dvrptw_bench.heuristics.constructive_pmca import PMCAVRPTWSolver
from dvrptw_bench.heuristics.gls import GLSSolver
from dvrptw_bench.heuristics.ortools_solver import ORToolsVRPTWSolver
from dvrptw_bench.hybrid.hybrid_runner import run_hybrid
from dvrptw_bench.results.aggregator import Aggregator
from dvrptw_bench.results.export import write_report
from dvrptw_bench.results.recorder import Recorder
from dvrptw_bench.rl.policies import build_policy
from dvrptw_bench.viz.dashboard import render_dashboard

app = typer.Typer(help="Run all experiments")
console = Console()


STRATEGY_BUILDERS = {
    "heuristics/pmca": lambda: PMCAVRPTWSolver(),
    "heuristics/ortools": lambda: ORToolsVRPTWSolver(),
    "heuristics/gls": lambda: GLSSolver(),
}


def _solve_with_strategy(strategy: str, instance, budget, epsilon, seed):
    if strategy.startswith("heuristics/"):
        solver = STRATEGY_BUILDERS[strategy]()
        sim = DynamicSimulator(instance)
        return sim.run(lambda snap_inst, t: solver.solve(snap_inst, t), epsilon=epsilon, budget_s=budget, seed=seed)
    if strategy.startswith("rl/"):
        policy_name = strategy.split("/", 1)[1]
        policy = build_policy(policy_name)
        from dvrptw_bench.dynamic.snapshot import SnapshotState
        from dvrptw_bench.metrics.objective import total_distance

        def solve_fn(snap_inst, _t):
            snap = SnapshotState(time=0.0, remaining_customers=snap_inst.customers, active_customer_ids={c.id for c in snap_inst.customers}, served_customer_ids=set(), vehicles=[])
            sol = policy.infer(snap)
            sol.total_distance = total_distance(snap_inst, sol)
            return sol

        sim = DynamicSimulator(instance)
        return sim.run(solve_fn, epsilon=epsilon, budget_s=budget, seed=seed)
    if strategy.startswith("hybrid/"):
        policy_name = strategy.split("/", 1)[1].replace("_gls", "")

        def solve_fn(snap_inst, t):
            sol, timings = run_hybrid(snap_inst, policy_name=policy_name, budget_s=t)
            sol.details["timings"] = timings
            return sol

        sim = DynamicSimulator(instance)
        return sim.run(solve_fn, epsilon=epsilon, budget_s=budget, seed=seed)
    raise ValueError(strategy)


@hydra.main(version_base=None, config_path="../config", config_name="main")
def _hydra_run(cfg: DictConfig) -> None:
    dataset_root = Path(cfg.dataset_root)
    output_root = Path(cfg.output_root)
    run_id = now_run_id("full")
    run_dir = ensure_run_dir(output_root, run_id)
    recorder = Recorder(run_dir)

    instances = find_rc_instances(dataset_root)
    set_seed(int(cfg.seed))

    for inst_path in instances:
        inst = parse_solomon(inst_path)
        for strategy in cfg.strategies:
            for epsilon in cfg.epsilons:
                for budget in cfg.budgets:
                    sol, events, scenario = _solve_with_strategy(strategy, inst, float(budget), float(epsilon), int(cfg.seed))
                    if sol is None:
                        continue
                    timings = sol.details.get("timings", {"total_s": sol.solve_time_s, "inference_s": 0.0, "local_search_s": 0.0})
                    recorder.log(
                        ResultRecord(
                            run_id=run_id,
                            timestamp=datetime.now(),
                            git_hash=git_hash(),
                            strategy=strategy,
                            solver_details=strategy,
                            budget_s=float(budget),
                            epsilon=float(epsilon),
                            seed=int(cfg.seed),
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
    render_dashboard(run_dir)
    console.print(f"Done. Output: {run_dir}")


@app.command("run")
def run(dataset_root: Path = Path("./dataset"), output_root: Path = Path("./outputs"), verbose: bool = False):
    # Hydra loads defaults from config/main.yaml; CLI root args remain convenience placeholders.
    _ = (dataset_root, output_root, verbose)
    _hydra_run()


if __name__ == "__main__":
    app()

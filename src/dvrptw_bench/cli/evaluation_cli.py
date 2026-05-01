"""Resumable final evaluation CLI."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from dvrptw_bench.evaluation.runner import (
    build_paths,
    generate_scenarios,
    init_workspace,
    project_root_from,
    run_modality,
    status_summary,
)

app = typer.Typer(help="Final evaluation workflow")
console = Console()


def _default_dataset_root() -> Path:
    return project_root_from(Path.cwd()) / "dataset"


@app.command("init")
def init(
    data_root: Path = typer.Option(Path("./data")),
    dataset_root: Path | None = typer.Option(None),
    seeds: str = typer.Option("1,2,3"),
    sizes: str = typer.Option("50"),
    dod: str = typer.Option("0.0"),
    cutoff: str = typer.Option("0.5"),
):
    dataset = dataset_root or _default_dataset_root()
    manifest = init_workspace(
        data_root=data_root,
        dataset_root=dataset,
        seeds=[int(value) for value in seeds.split(",") if value],
        evaluation_sizes=[int(value) for value in sizes.split(",") if value],
        degrees_of_dynamicity=[float(value) for value in dod.split(",") if value],
        cutoff_times=[float(value) for value in cutoff.split(",") if value],
    )
    console.print(f"Initialized workspace at {build_paths(data_root).root}")
    console.print(f"Found {len(manifest.instances)} base instances")


@app.command("generate-scenarios")
def generate(
    data_root: Path = typer.Option(Path("./data")),
    seeds: str | None = typer.Option(None),
):
    seed_values = [int(value) for value in seeds.split(",") if value] if seeds else None
    records = generate_scenarios(data_root, seeds=seed_values)
    console.print(f"Generated {len(records)} scenario artifacts")


@app.command("run")
def run(
    modality: str = typer.Argument(..., help="oracle | heuristic | ai | hybrid | static"),
    data_root: Path = typer.Option(Path("./data")),
    limit: int | None = typer.Option(None),
    workers: int | None = typer.Option(None, help="Process count for oracle/heuristic; defaults to auto there and 1 elsewhere."),
    decode_type: str = typer.Option("greedy", help="AI decode type: greedy | sampling | multistart"),
    num_samples: int = typer.Option(1, help="Number of samples for sampling decode."),
    num_starts: int | None = typer.Option(None, help="Number of starts for multistart decode."),
    num_augment: int = typer.Option(8, help="Number of augmentations for greedy/multistart evaluation."),
    select_best: bool = typer.Option(True, help="Whether to select the best sampled/augmented solution."),
):
    completed, attempted = run_modality(
        data_root,
        modality,
        limit=limit,
        workers=workers,
        decode_type=decode_type,
        num_samples=num_samples,
        num_starts=num_starts,
        num_augment=num_augment,
        select_best=select_best,
    )
    console.print(f"Attempted {attempted} work units, completed {completed}")


@app.command("status")
def status(
    data_root: Path = typer.Option(Path("./data")),
):
    summary = status_summary(data_root)
    for modality, counts in summary.items():
        console.print(f"{modality}: {counts}")


if __name__ == "__main__":
    app()

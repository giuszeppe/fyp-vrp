import sys
import time
import re
import shutil
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from IPython.display import display
from rl4co.utils.trainer import RL4COTrainer

try:
    from lightning.pytorch.callbacks import ModelCheckpoint
except ImportError:
    from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append(str(Path("..").resolve() / "src"))

from dvrptw_bench.data.instance_filters import find_rc_instances
from dvrptw_bench.data.solomon_parser import parse_solomon
from dvrptw_bench.heuristics.ortools_solver import ORToolsVRPTWSolver
from dvrptw_bench.metrics.objective import total_distance
from dvrptw_bench.rl.routefinder_adapter import (
    _normalize_coord_and_get_scale,
    instance_to_routefinder_td,
    routefinder_actions_to_solution,
)
from dvrptw_bench.rl.mtvrp_solomon_generator import SolomonMTVRPGenerator
from dvrptw_bench.viz.route_plot import plot_routes
from routefinder.envs.mtvrp import MTVRPEnv
from routefinder.models import RouteFinderBase, RouteFinderPolicy
from routefinder.utils import evaluate as evaluate_routefinder


def parse_args():
    project_root_default = Path("..").resolve()
    output_root_default = (
        project_root_default / "outputs/notebook_routefinder_solomon_generated"
    )
    checkpoint_dir_default = output_root_default / "checkpoints"

    parser = argparse.ArgumentParser(
        description="Train and evaluate RouteFinder on Solomon VRPTW instances."
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root_default,
        help="Project root directory.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_root_default / "dataset/solomon_rc100",
        help="Dataset root used for evaluation RC instances.",
    )
    parser.add_argument(
        "--rc-dataset-root",
        type=Path,
        default=project_root_default / "dataset/solomon_rc100",
        help="Dataset root for RC training instances.",
    )
    parser.add_argument(
        "--c-dataset-root",
        type=Path,
        default=project_root_default / "dataset/solomon_c100",
        help="Dataset root for C training instances.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=output_root_default,
        help="Output directory.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=checkpoint_dir_default,
        help="Checkpoint directory.",
    )
    parser.add_argument(
        "--num-customers",
        type=int,
        default=100,
        help="Number of customers per instance.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=150,
        help="Total number of training epochs.",
    )
    parser.add_argument(
        "--checkpoint-every-n-epochs",
        type=int,
        default=2,
        help="Checkpoint save frequency in epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size.",
    )
    parser.add_argument(
        "--train-data-size",
        type=int,
        default=100_000,
        help="Number of generated training samples per epoch.",
    )
    parser.add_argument(
        "--val-data-size",
        type=int,
        default=10_000,
        help="Number of generated validation samples.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-6,
        help="Optimizer weight decay.",
    )
    parser.add_argument(
        "--num-augment",
        type=int,
        default=8,
        help="Number of augmentations during RouteFinder evaluation.",
    )
    parser.add_argument(
        "--ortools-time-limit-s",
        type=float,
        default=0.1,
        help="OR-Tools time limit in seconds.",
    )
    parser.add_argument(
        "--max-eval-instances",
        type=int,
        default=5,
        help="Maximum number of evaluation instances. Use a negative value to evaluate all.",
    )
    parser.add_argument(
        "--normalize-coords",
        dest="normalize_coords",
        action="store_true",
        default=True,
        help="Enable coordinate normalization.",
    )
    parser.add_argument(
        "--no-normalize-coords",
        dest="normalize_coords",
        action="store_false",
        help="Disable coordinate normalization.",
    )
    parser.add_argument(
        "--final-checkpoint-path",
        type=Path,
        default=None,
        help="Optional explicit final checkpoint path. Defaults to OUTPUT_ROOT/routefinder_<N>cust_<E>epochs.ckpt",
    )

    return parser.parse_args()


def _extract_epoch_from_checkpoint_name(path: Path) -> int:
    matches = re.findall(r"epoch[-_](\d+)", path.stem)
    if not matches:
        return -1
    return int(matches[-1])


def find_resume_checkpoint(checkpoint_dir: Path) -> Path | None:
    candidates = []

    interrupted_checkpoint = checkpoint_dir / "interrupted.ckpt"
    if interrupted_checkpoint.exists():
        candidates.append(interrupted_checkpoint)

    last_checkpoint = checkpoint_dir / "last.ckpt"
    if last_checkpoint.exists():
        candidates.append(last_checkpoint)

    candidates.extend(
        path for path in checkpoint_dir.glob("epoch-*.ckpt") if path.is_file()
    )

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda path: (
            path.name == "interrupted.ckpt",
            path.name == "last.ckpt",
            _extract_epoch_from_checkpoint_name(path),
            path.stat().st_mtime,
        ),
    )


def get_completed_epochs(checkpoint_path: Path | None) -> int:
    if checkpoint_path is None:
        return 0

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    return max(int(checkpoint.get("epoch", -1)) + 1, 0)


def build_solomon_training_item(
    instance,
    num_customers,
    normalize_coords=True,
):
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


def main():
    args = parse_args()

    interrupted_checkpoint_path = args.checkpoint_dir / "interrupted.ckpt"
    final_checkpoint_path = args.final_checkpoint_path or (
        args.output_root
        / f"routefinder_{args.num_customers}cust_{args.num_epochs}epochs.ckpt"
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        accelerator = "gpu"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
        accelerator = "mps"
    else:
        device = torch.device("cpu")
        accelerator = "cpu"

    print("Device:", device)
    print("Dataset root:", args.dataset_root.resolve())
    print("Output root:", args.output_root.resolve())

    rc_instances = [
        parse_solomon(instance, max_customers=args.num_customers)
        for instance in find_rc_instances(args.rc_dataset_root)
    ]
    c_instances = [
        parse_solomon(instance, max_customers=args.num_customers)
        for instance in find_rc_instances(args.c_dataset_root)
    ]

    generator_solomon_instances = (
        [
            build_solomon_training_item(
                instance,
                num_customers=args.num_customers,
                normalize_coords=args.normalize_coords,
            )
            for instance in rc_instances
        ]
        + [
            build_solomon_training_item(
                instance,
                num_customers=args.num_customers,
                normalize_coords=args.normalize_coords,
            )
            for instance in c_instances
        ]
    )

    print(f"Training pool size: {len(generator_solomon_instances)}")
    print(generator_solomon_instances[:1])

    generator = SolomonMTVRPGenerator(
        num_loc=args.num_customers,
        variant_preset="vrptw",
        solomon_instances=generator_solomon_instances,
    )
    env = MTVRPEnv(generator, check_solution=False)

    policy = RouteFinderPolicy(env_name=env.name).to(device)
    model = RouteFinderBase(
        env,
        policy,
        batch_size=args.batch_size,
        train_data_size=args.train_data_size,
        val_data_size=args.val_data_size,
        optimizer_kwargs={
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="epoch-{epoch:03d}",
        save_top_k=-1,
        save_last=True,
        every_n_epochs=args.checkpoint_every_n_epochs,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
    )

    trainer = RL4COTrainer(
        max_epochs=args.num_epochs,
        accelerator=accelerator,
        devices=1,
        logger=None,
        num_sanity_val_steps=0,
        precision="32-true",
        callbacks=[checkpoint_callback],
    )

    resume_checkpoint = find_resume_checkpoint(args.checkpoint_dir)
    completed_epochs = get_completed_epochs(resume_checkpoint)
    remaining_epochs = max(args.num_epochs - completed_epochs, 0)

    if resume_checkpoint is not None:
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
        print(
            f"Completed epochs: {completed_epochs}/{args.num_epochs}. "
            f"Remaining epochs: {remaining_epochs}"
        )

    if remaining_epochs == 0:
        if (
            resume_checkpoint is not None
            and resume_checkpoint.resolve() != final_checkpoint_path.resolve()
        ):
            shutil.copy2(resume_checkpoint, final_checkpoint_path)
        print(
            f"Training already complete. Final checkpoint available at {final_checkpoint_path}"
        )
    else:
        try:
            super(RL4COTrainer, trainer).fit(
                model,
                ckpt_path=str(resume_checkpoint) if resume_checkpoint is not None else None,
                weights_only=False,
            )
        except KeyboardInterrupt:
            trainer.save_checkpoint(interrupted_checkpoint_path)
            print(
                f"Training interrupted. Saved resume checkpoint to {interrupted_checkpoint_path}"
            )
            raise

        trainer.save_checkpoint(final_checkpoint_path)
        if interrupted_checkpoint_path.exists():
            interrupted_checkpoint_path.unlink()

        print(f"Final checkpoint saved to {final_checkpoint_path}")

    try:
        model = RouteFinderBase.load_from_checkpoint(
            str(final_checkpoint_path),
            env=env,
            policy=policy,
            map_location=device,
            weights_only=False,
        )
        model = model.to(device)
        print(f"Reloaded model from {final_checkpoint_path}")
    except Exception as e:
        print(
            f"Could not reload from final checkpoint, using in-memory model instead: {e}"
        )
        model = model.to(device)

    model.eval()

    instance_paths = find_rc_instances(args.dataset_root)
    assert instance_paths, f"No RC instances found under {args.dataset_root.resolve()}"

    if args.max_eval_instances is not None and args.max_eval_instances >= 0:
        instance_paths = instance_paths[: args.max_eval_instances]

    instances = [
        parse_solomon(path, max_customers=args.num_customers) for path in instance_paths
    ]
    print("Loaded instances:", [instance.instance_id for instance in instances])

    def solve_with_routefinder(instance, num_augment=args.num_augment):
        t0 = time.perf_counter()
        td = instance_to_routefinder_td(
            instance, normalize_coords=args.normalize_coords
        ).to(device)
        td_reset = env.reset(td)

        model.to(device).eval()
        with torch.inference_mode():
            if num_augment > 1:
                out = evaluate_routefinder(
                    model, td_reset.clone(), num_augment=num_augment
                )
                actions = out.get(
                    "best_aug_actions",
                    out.get("best_multistart_actions", out.get("actions")),
                )
            else:
                out = model.policy(
                    td_reset.clone(),
                    env,
                    phase="test",
                    decode_type="greedy",
                    return_actions=True,
                )
                actions = out["actions"]

        solution = routefinder_actions_to_solution(
            actions, instance, strategy="routefinder"
        )
        solution.total_distance = total_distance(instance, solution)
        solution.solve_time_s = time.perf_counter() - t0
        solution.details.update({"num_augment": num_augment})
        return solution

    ortools = ORToolsVRPTWSolver()
    routefinder_solutions = {}
    ortools_solutions = {}
    rows = []

    for instance in instances:
        rf_solution = solve_with_routefinder(instance)
        or_solution = ortools.solve(instance, time_limit_s=args.ortools_time_limit_s)

        routefinder_solutions[instance.instance_id] = rf_solution
        ortools_solutions[instance.instance_id] = or_solution

        gap_pct = (
            100.0
            * (rf_solution.total_distance - or_solution.total_distance)
            / or_solution.total_distance
        )

        rows.append(
            {
                "instance_id": instance.instance_id,
                "n_customers": instance.n_customers,
                "routefinder_distance": rf_solution.total_distance,
                "ortools_distance": or_solution.total_distance,
                "gap_to_ortools_pct": gap_pct,
                "routefinder_routes": len(
                    [r for r in rf_solution.routes if r.node_ids]
                ),
                "ortools_routes": len(
                    [r for r in or_solution.routes if r.node_ids]
                ),
                "routefinder_time_s": rf_solution.solve_time_s,
                "ortools_time_s": or_solution.solve_time_s,
            }
        )

    results_df = pd.DataFrame(rows).sort_values("gap_to_ortools_pct").reset_index(
        drop=True
    )
    results_df[
        [
            "routefinder_distance",
            "ortools_distance",
            "gap_to_ortools_pct",
            "routefinder_time_s",
            "ortools_time_s",
        ]
    ] = results_df[
        [
            "routefinder_distance",
            "ortools_distance",
            "gap_to_ortools_pct",
            "routefinder_time_s",
            "ortools_time_s",
        ]
    ].round(3)

    display(results_df)

    results_csv_path = args.output_root / "routefinder_vs_ortools.csv"
    results_df.to_csv(results_csv_path, index=False)
    print("Saved table to", results_csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    plot_df = results_df.set_index("instance_id")
    plot_df[["routefinder_distance", "ortools_distance"]].plot(
        kind="bar", ax=axes[0]
    )
    axes[0].set_title("RouteFinder vs OR-Tools Distance")
    axes[0].set_ylabel("Total distance")
    axes[0].grid(axis="y", alpha=0.3)

    plot_df["gap_to_ortools_pct"].plot(kind="bar", ax=axes[1], color="tab:orange")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_title("RouteFinder Gap vs OR-Tools")
    axes[1].set_ylabel("Gap (%)")
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    plot_path = args.output_root / "routefinder_vs_ortools_summary.png"
    fig.savefig(plot_path, dpi=180)
    plt.show()
    print("Saved summary plot to", plot_path)

    for instance in instances[: min(3, len(instances))]:
        print(f"\n=== {instance.instance_id} ===")
        print(
            "RouteFinder distance:",
            routefinder_solutions[instance.instance_id].total_distance,
        )
        print(
            "OR-Tools distance:",
            ortools_solutions[instance.instance_id].total_distance,
        )
        plot_routes(instance, routefinder_solutions[instance.instance_id])
        plot_routes(instance, ortools_solutions[instance.instance_id])


if __name__ == "__main__":
    main()
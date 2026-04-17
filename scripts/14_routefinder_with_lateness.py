import sys
import time
import re
import shutil
import argparse
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import pandas as pd
import torch
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
from dvrptw_bench.rl.env_flexible import MTVRPFlexibleEnv
from dvrptw_bench.rl.routefinder_adapter import (
    instance_to_routefinder_td,
    routefinder_actions_to_solution,
)
from routefinder.envs.mtvrp import MTVRPGenerator
from routefinder.models import RouteFinderBase, RouteFinderPolicy
from routefinder.utils import evaluate as evaluate_routefinder
from routefinder.utils import rollout, greedy_policy, evaluate
from routefinder.baselines.solve import solve


def optional_int(value: str):
    lowered = str(value).strip().lower()
    if lowered in {"none", "all"}:
        return None
    return int(value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RouteFinder on generated MTVRP instances and evaluate on Solomon RC instances."
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../dataset/solomon_rc100"),
        help="Root directory for Solomon RC dataset.",
    )
    parser.add_argument(
        "--rc-dataset-root",
        type=Path,
        default=Path("../dataset/solomon_rc100"),
        help="RC dataset root (kept for parity with notebook constants).",
    )
    parser.add_argument(
        "--c-dataset-root",
        type=Path,
        default=Path("../dataset/solomon_c100"),
        help="C dataset root (kept for parity with notebook constants).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("../outputs/routefinder_with_lateness"),
        help="Directory where CSV/plots/checkpoints are saved.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Checkpoint directory. Defaults to <output-root>/checkpoints",
    )
    parser.add_argument(
        "--final-checkpoint-path",
        type=Path,
        default=None,
        help="Optional explicit final checkpoint path. Defaults to <output-root>/routefinder_<N>cust_<E>epochs.ckpt",
    )
    parser.add_argument(
        "--checkpoint-every-n-epochs",
        type=int,
        default=2,
        help="Checkpoint save frequency in epochs.",
    )

    parser.add_argument(
        "--num-customers",
        type=int,
        default=10,
        help="Number of customers per instance.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
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
        default=1000,
        help="Generated training dataset size.",
    )
    parser.add_argument(
        "--val-data-size",
        type=int,
        default=100,
        help="Generated validation dataset size.",
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
        type=optional_int,
        default=5,
        help="Maximum number of RC instances to evaluate. Use 'none' or 'all' for no limit.",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=10.11,
        help="MTVRP generator max_time parameter.",
    )
    parser.add_argument(
        "--num-rollout-test-instances",
        type=int,
        default=50,
        help="Number of generated instances for the rollout/evaluate demo block.",
    )
    parser.add_argument(
        "--pyvrp-max-runtime",
        type=float,
        default=3.0,
        help="Max runtime for PyVRP baseline in demo block.",
    )
    parser.add_argument(
        "--pyvrp-num-procs",
        type=int,
        default=32,
        help="Number of processes for PyVRP baseline in demo block.",
    )

    parser.add_argument(
        "--normalize-coords",
        dest="normalize_coords",
        action="store_true",
        default=True,
        help="Normalize coordinates to [0, 1] when creating RouteFinder tensors.",
    )
    parser.add_argument(
        "--no-normalize-coords",
        dest="normalize_coords",
        action="store_false",
        help="Disable coordinate normalization.",
    )

    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and assume model is already initialized.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip Solomon evaluation block.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip saving/showing matplotlib plots.",
    )
    parser.add_argument(
        "--skip-demo",
        action="store_true",
        help="Skip rollout/evaluate/PyVRP demo block.",
    )

    return parser.parse_args()


def gap(sol, bks):
    return ((sol - bks) / bks).mean() * 100


def summarize_td(td, name):
    tw = td["time_windows"][0].cpu()
    st = td["service_time"][0].cpu()

    starts = tw[:, 0]
    ends = tw[:, 1]
    widths = ends - starts

    print(f"\n{name}")
    print("depot due:", ends[0].item())
    print("start  min/mean/max:", starts.min().item(), starts.mean().item(), starts.max().item())
    print("end    min/mean/max:", ends.min().item(), ends.mean().item(), ends.max().item())
    print("width  min/mean/max:", widths.min().item(), widths.mean().item(), widths.max().item())
    print("service min/mean/max:", st.min().item(), st.mean().item(), st.max().item())


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


def main():
    args = parse_args()

    dataset_root = args.dataset_root
    rc_dataset_root = args.rc_dataset_root
    c_dataset_root = args.c_dataset_root
    output_root = args.output_root
    checkpoint_dir = args.checkpoint_dir or (output_root / "checkpoints")
    interrupted_checkpoint_path = checkpoint_dir / "interrupted.ckpt"
    final_checkpoint_path = args.final_checkpoint_path or (
        output_root / f"routefinder_{args.num_customers}_cust_{args.num_epochs}_epochs.ckpt"
    )

    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
    print("Dataset root:", dataset_root.resolve())
    print("RC dataset root:", rc_dataset_root.resolve())
    print("C dataset root:", c_dataset_root.resolve())
    print("Output root:", output_root.resolve())
    print("Checkpoint dir:", checkpoint_dir.resolve())

    generator = MTVRPGenerator(
        num_loc=args.num_customers,
        max_time=args.max_time,
        variant_preset="vrptw",
    )
    env = MTVRPFlexibleEnv(generator, check_solution=False)
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
        dirpath=checkpoint_dir,
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
        callbacks=[cast(Any, checkpoint_callback)],
    )

    resume_checkpoint = find_resume_checkpoint(checkpoint_dir)
    completed_epochs = get_completed_epochs(resume_checkpoint)
    remaining_epochs = max(args.num_epochs - completed_epochs, 0)

    if resume_checkpoint is not None:
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
        print(
            f"Completed epochs: {completed_epochs}/{args.num_epochs}. "
            f"Remaining epochs: {remaining_epochs}"
        )

    if not args.skip_training:
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


if __name__ == "__main__":
    main()
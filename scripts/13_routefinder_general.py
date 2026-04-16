
import sys
import time
import re
import shutil
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from tensordict import TensorDict
import pandas as pd
import torch
from IPython.display import display
from rl4co.utils.trainer import RL4COTrainer

try:
    from lightning.pytorch.callbacks import ModelCheckpoint
except ImportError:
    from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append(str(Path("..").resolve() / "src"))

from dvrptw_bench.rl.mtvrp_solomon_generator import MTVRPGenerator
from routefinder.envs.mtvrp import MTVRPEnv, MTVRPGenerator
from routefinder.models import RouteFinderBase, RouteFinderPolicy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate RouteFinder with configurable CLI parameters."
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("../outputs/notebook_routefinder_solomon_generated"),
        help="Output directory.",
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
        default=100,
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
        default=100_000,
        help="Training data size.",
    )
    parser.add_argument(
        "--val-data-size",
        type=int,
        default=10_000,
        help="Validation data size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-6,
        help="Weight decay.",
    )
    parser.add_argument(
        "--num-augment",
        type=int,
        default=8,
        help="Number of augmentations during RouteFinder evaluation.",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=10.11,
        help="MTVRP generator max_time parameter.",
    )

    parser.add_argument(
        "--normalize-coords",
        dest="normalize_coords",
        action="store_true",
        default=True,
        help="Normalize coordinates to [0, 1].",
    )
    parser.add_argument(
        "--no-normalize-coords",
        dest="normalize_coords",
        action="store_false",
        help="Disable coordinate normalization.",
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


def main():
    args = parse_args()

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
    print("Output root:", output_root.resolve())
    print("Checkpoint dir:", checkpoint_dir.resolve())

    generator = MTVRPGenerator(
        num_loc=args.num_customers,
        max_time=args.max_time,
        variant_preset="vrptw",
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
        callbacks=[checkpoint_callback],
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
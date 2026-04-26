import argparse
import re
import shutil
import sys
from pathlib import Path

import torch
from rl4co.utils.trainer import RL4COTrainer

from dvrptw_bench.paper_dynamic_routefinder.generator import DynamicGenerator

try:
    from lightning.pytorch.callbacks import ModelCheckpoint
except ImportError:
    from pytorch_lightning.callbacks import ModelCheckpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from dvrptw_bench.common.rng import set_seed
from dvrptw_bench.data.solomon_parser import parse_solomon
from dvrptw_bench.rl.env_dynamic import MTVRPDynamicEnv
from dvrptw_bench.rl.mtvrp_solomon_generator import SolomonMTVRPGenerator
from dvrptw_bench.rl.routefinder_adapter import _normalize_coord_and_get_scale
from routefinder.models import RouteFinderBase, RouteFinderPolicy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RouteFinder on the dynamic environment using Solomon instances."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT/ "dataset/",
        help="Root directory containing Solomon instance files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "15_dynamic_training",
        help="Root output directory.",
    )
    parser.add_argument(
        "--num-customers",
        type=int,
        nargs="+",
        default=[50, 75],
        help="Customer counts to train. Default trains both 50 and 75.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Total number of training epochs per customer count.",
    )
    parser.add_argument(
        "--checkpoint-every-n-epochs",
        type=int,
        default=1,
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
        help="Generated training samples per epoch.",
    )
    parser.add_argument(
        "--val-data-size",
        type=int,
        default=10_000,
        help="Generated validation samples.",
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
        "--max-time",
        type=float,
        default=10.11,
        help="Generator max_time parameter.",
    )
    parser.add_argument(
        "--dod",
        type=float,
        default=0.3,
        help="Degree of dynamism for the training environment.",
    )
    parser.add_argument(
        "--cutoff-ratio",
        type=float,
        default=0.8,
        help="Cutoff ratio for the training environment.",
    )
    parser.add_argument(
        "--lateness-penalty",
        type=float,
        default=100.0,
        help="Soft time-window lateness penalty.",
    )
    parser.add_argument(
        "--reject-penalty",
        type=float,
        default=1_000.0,
        help="Customer rejection penalty.",
    )
    parser.add_argument(
        "--dynamic-seed",
        type=int,
        default=1234,
        help="Seed for dynamic customer revelation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Global random seed.",
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


class RollingEpochCheckpoint(ModelCheckpoint):
    def _prune_old_epoch_checkpoints(self, current_path: Path) -> None:
        checkpoint_root = Path(self.dirpath)
        epoch_checkpoints = [
            path
            for path in checkpoint_root.glob("epoch-*.ckpt")
            if path.is_file() and path.resolve() != current_path.resolve()
        ]
        for path in epoch_checkpoints:
            path.unlink(missing_ok=True)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        current_path = getattr(self, "_last_checkpoint_saved", None)
        if current_path is None:
            return
        self._prune_old_epoch_checkpoints(Path(current_path))


def discover_solomon_instances(dataset_root: Path) -> list[Path]:
    instance_paths = sorted(path for path in dataset_root.rglob("*.txt") if path.is_file())
    if not instance_paths:
        raise FileNotFoundError(
            f"No Solomon .txt instances found under {dataset_root.resolve()}"
        )
    return instance_paths


def build_solomon_training_item(instance, num_customers: int, normalize_coords: bool):
    customers = instance.customers[:num_customers]
    if len(customers) != num_customers:
        raise ValueError(
            f"Instance {instance.instance_id} has only {len(customers)} customers; "
            f"cannot build training item for {num_customers} customers."
        )

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


def load_training_pool(
    dataset_root: Path,
    num_customers: int,
    normalize_coords: bool,
) -> list[dict]:
    instance_paths = discover_solomon_instances(dataset_root)
    instances = [
        parse_solomon(path, max_customers=num_customers)
        for path in instance_paths
    ]
    training_items = [
        build_solomon_training_item(
            instance,
            num_customers=num_customers,
            normalize_coords=normalize_coords,
        )
        for instance in instances
    ]
    return training_items


def get_device_and_accelerator() -> tuple[torch.device, str]:
    if torch.cuda.is_available():
        return torch.device("cuda"), "gpu"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


def train_for_customer_count(args, num_customers: int, device: torch.device, accelerator: str) -> None:
    set_seed(args.seed)

    run_root = args.output_root / f"{num_customers}_customers"
    checkpoint_dir = run_root / "checkpoints"
    interrupted_checkpoint_path = checkpoint_dir / "interrupted.ckpt"
    final_checkpoint_path = (
        run_root / f"routefinder_dynamic_solomon_{num_customers}cust_{args.num_epochs}epochs.ckpt"
    )

    run_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Training {num_customers} customers ===")
    print("Dataset root:", args.dataset_root.resolve())
    print("Run output root:", run_root.resolve())
    print("Checkpoint dir:", checkpoint_dir.resolve())
    print("Seed:", args.seed)
    print("Dynamic seed:", args.dynamic_seed)

    training_items = load_training_pool(
        dataset_root=args.dataset_root,
        num_customers=num_customers,
        normalize_coords=args.normalize_coords,
    )
    print(f"Loaded {len(training_items)} Solomon instances for training")

    generator = DynamicGenerator(
        solomon_instances=training_items,
        num_loc=num_customers,
        dod=args.dod,
        cutoff_time=args.cutoff_ratio,
        max_time=args.max_time,
        variant_preset="vrptw",
    )
    env = MTVRPDynamicEnv(
        generator=generator,
        check_solution=False,
        allow_late_customers=True,
        lateness_penalty=args.lateness_penalty,
        allow_reject_customers=True,
        reject_penalty=args.reject_penalty,
        dynamic_seed=args.dynamic_seed,
    )
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

    checkpoint_callback = RollingEpochCheckpoint(
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
        return

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


def main():
    args = parse_args()
    device, accelerator = get_device_and_accelerator()

    print("Device:", device)
    print("Requested customer counts:", args.num_customers)

    for num_customers in args.num_customers:
        train_for_customer_count(args, num_customers, device, accelerator)


if __name__ == "__main__":
    main()

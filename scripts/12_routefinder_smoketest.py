
import sys
import time
import re
import shutil
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
    _normalize_coord,
    instance_to_routefinder_td,
    routefinder_actions_to_solution,
)
from dvrptw_bench.rl.mtvrp_solomon_generator import SolomonMTVRPGenerator
from dvrptw_bench.viz.route_plot import plot_routes
from routefinder.envs.mtvrp import MTVRPEnv
from routefinder.models import RouteFinderBase, RouteFinderPolicy
from routefinder.utils import evaluate as evaluate_routefinder


PROJECT_ROOT = Path("..").resolve()
DATASET_ROOT = PROJECT_ROOT / "dataset/solomon_rc100"
RC_DATASET_ROOT = PROJECT_ROOT / "dataset/solomon_rc100"
C_DATASET_ROOT = PROJECT_ROOT / "dataset/solomon_c100"

OUTPUT_ROOT = PROJECT_ROOT / "outputs/notebook_routefinder_solomon_generated"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
INTERRUPTED_CHECKPOINT_PATH = CHECKPOINT_DIR / "interrupted.ckpt"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CUSTOMERS = 10
TARGET_EPOCHS = 3
CHECKPOINT_EVERY_N_EPOCHS = 3

BATCH_SIZE = 256
TRAIN_DATA_SIZE = 100_000
VAL_DATA_SIZE = 10_000
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-6

NUM_AUGMENT = 8
ORTOOLS_TIME_LIMIT_S = 0.1
MAX_EVAL_INSTANCES = 5  # set to None to evaluate all RC instances
NORMALIZE_COORDS = False
FINAL_CHECKPOINT_PATH = OUTPUT_ROOT / f"routefinder_{NUM_CUSTOMERS}cust_{TARGET_EPOCHS}epochs.ckpt"

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
print("Dataset root:", DATASET_ROOT.resolve())
print("Output root:", OUTPUT_ROOT.resolve())


def normalize_coord(coord: torch.Tensor) -> torch.Tensor:
    return _normalize_coord(coord)


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
    target_max_time=4.6,
):
    customers = instance.customers[:num_customers]

    coords = [(instance.depot.x, instance.depot.y)] + [(c.x, c.y) for c in customers]
    locs = torch.tensor(coords, dtype=torch.float32)

    if normalize_coords:
        locs = normalize_coord(locs)

    ready = torch.tensor(
        [instance.depot.ready_time] + [c.ready_time for c in customers],
        dtype=torch.float32,
    )
    due = torch.tensor(
        [instance.depot.due_time] + [c.due_time for c in customers],
        dtype=torch.float32,
    )
    service = torch.tensor(
        [0.0] + [c.service_time for c in customers],
        dtype=torch.float32,
    )
    demand = torch.tensor(
        [c.demand for c in customers],
        dtype=torch.float32,
    )

    alpha = 1.0
    time_windows = torch.stack([ready, due], dim=-1)

    # Optional time scaling block left disabled intentionally
    # depot_due = time_windows[0, 1].item()
    # alpha = target_max_time / max(depot_due, 1e-8)
    # time_windows = time_windows * alpha
    # service = service * alpha
    # time_windows[0, 0] = 0.0
    # time_windows[0, 1] = target_max_time

    service[0] = 0.0

    return {
        "locs": locs,                         # [N+1, 2]
        "time_windows": time_windows,         # [N+1, 2]
        "service_time": service,              # [N+1]
        "demand_linehaul": demand,            # [N]
        "vehicle_capacity": float(instance.vehicle_capacity),
        "instance_id": instance.instance_id,
        "alpha": alpha,
    }



rc_instances = [
    parse_solomon(instance, max_customers=NUM_CUSTOMERS)
    for instance in find_rc_instances(RC_DATASET_ROOT)
]
c_instances = [
    parse_solomon(instance, max_customers=NUM_CUSTOMERS)
    for instance in find_rc_instances(C_DATASET_ROOT)
]

generator_solomon_instances = (
    [
        build_solomon_training_item(
            instance,
            num_customers=NUM_CUSTOMERS,
            normalize_coords=NORMALIZE_COORDS,
            target_max_time=4.6,
        )
        for instance in rc_instances
    ]
    + [
        build_solomon_training_item(
            instance,
            num_customers=NUM_CUSTOMERS,
            normalize_coords=NORMALIZE_COORDS,
            target_max_time=4.6,
        )
        for instance in c_instances
    ]
)

print(f"Training pool size: {len(generator_solomon_instances)}")
print(generator_solomon_instances[:1])


generator = SolomonMTVRPGenerator(
    num_loc=NUM_CUSTOMERS,
    variant_preset="vrptw",
    solomon_instances=generator_solomon_instances,
)
env = MTVRPEnv(generator, check_solution=False)

policy = RouteFinderPolicy(env_name=env.name).to(device)
model = RouteFinderBase(
    env,
    policy,
    batch_size=BATCH_SIZE,
    train_data_size=TRAIN_DATA_SIZE,
    val_data_size=VAL_DATA_SIZE,
    optimizer_kwargs={"lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY},
)

checkpoint_callback = ModelCheckpoint(
    dirpath=CHECKPOINT_DIR,
    filename="epoch-{epoch:03d}",
    save_top_k=-1,
    save_last=True,
    every_n_epochs=CHECKPOINT_EVERY_N_EPOCHS,
    save_on_train_epoch_end=True,
    auto_insert_metric_name=False,
)

trainer = RL4COTrainer(
    max_epochs=TARGET_EPOCHS,
    accelerator=accelerator,
    devices=1,
    logger=None,
    num_sanity_val_steps=0,
    precision="32-true",
    callbacks=[checkpoint_callback],
)


resume_checkpoint = find_resume_checkpoint(CHECKPOINT_DIR)
completed_epochs = get_completed_epochs(resume_checkpoint)
remaining_epochs = max(TARGET_EPOCHS - completed_epochs, 0)

if resume_checkpoint is not None:
    print(f"Resuming training from checkpoint: {resume_checkpoint}")
    print(
        f"Completed epochs: {completed_epochs}/{TARGET_EPOCHS}. "
        f"Remaining epochs: {remaining_epochs}"
    )

if remaining_epochs == 0:
    if resume_checkpoint is not None and resume_checkpoint.resolve() != FINAL_CHECKPOINT_PATH.resolve():
        shutil.copy2(resume_checkpoint, FINAL_CHECKPOINT_PATH)
    print(f"Training already complete. Final checkpoint available at {FINAL_CHECKPOINT_PATH}")
else:
    try:
        super(RL4COTrainer, trainer).fit(
            model,
            ckpt_path=str(resume_checkpoint) if resume_checkpoint is not None else None,
            weights_only=False,
        )
    except KeyboardInterrupt:
        trainer.save_checkpoint(INTERRUPTED_CHECKPOINT_PATH)
        print(f"Training interrupted. Saved resume checkpoint to {INTERRUPTED_CHECKPOINT_PATH}")
        raise

    trainer.save_checkpoint(FINAL_CHECKPOINT_PATH)
    if INTERRUPTED_CHECKPOINT_PATH.exists():
        INTERRUPTED_CHECKPOINT_PATH.unlink()

    print(f"Final checkpoint saved to {FINAL_CHECKPOINT_PATH}")


# Optional: reload explicitly from the final checkpoint for clean evaluation
try:
    model = RouteFinderBase.load_from_checkpoint(
        str(FINAL_CHECKPOINT_PATH),
        env=env,
        policy=policy,
        map_location=device,
        weights_only=False,
    )
    model = model.to(device)
    print(f"Reloaded model from {FINAL_CHECKPOINT_PATH}")
except Exception as e:
    print(f"Could not reload from final checkpoint, using in-memory model instead: {e}")
    model = model.to(device)

model.eval()


instance_paths = find_rc_instances(DATASET_ROOT)
assert instance_paths, f"No RC instances found under {DATASET_ROOT.resolve()}"

if MAX_EVAL_INSTANCES is not None:
    instance_paths = instance_paths[:MAX_EVAL_INSTANCES]

instances = [parse_solomon(path, max_customers=NUM_CUSTOMERS) for path in instance_paths]
print("Loaded instances:", [instance.instance_id for instance in instances])


def solve_with_routefinder(instance, num_augment=NUM_AUGMENT):
    t0 = time.perf_counter()
    td = instance_to_routefinder_td(instance, normalize_coords=NORMALIZE_COORDS).to(device)
    td_reset = env.reset(td)

    model.to(device).eval()
    with torch.inference_mode():
        if num_augment > 1:
            out = evaluate_routefinder(model, td_reset.clone(), num_augment=num_augment)
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

    solution = routefinder_actions_to_solution(actions, instance, strategy="routefinder")
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
    or_solution = ortools.solve(instance, time_limit_s=ORTOOLS_TIME_LIMIT_S)

    routefinder_solutions[instance.instance_id] = rf_solution
    ortools_solutions[instance.instance_id] = or_solution

    gap_pct = 100.0 * (rf_solution.total_distance - or_solution.total_distance) / or_solution.total_distance

    rows.append(
        {
            "instance_id": instance.instance_id,
            "n_customers": instance.n_customers,
            "routefinder_distance": rf_solution.total_distance,
            "ortools_distance": or_solution.total_distance,
            "gap_to_ortools_pct": gap_pct,
            "routefinder_routes": len([r for r in rf_solution.routes if r.node_ids]),
            "ortools_routes": len([r for r in or_solution.routes if r.node_ids]),
            "routefinder_time_s": rf_solution.solve_time_s,
            "ortools_time_s": or_solution.solve_time_s,
        }
    )

results_df = pd.DataFrame(rows).sort_values("gap_to_ortools_pct").reset_index(drop=True)
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

results_csv_path = OUTPUT_ROOT / "routefinder_vs_ortools.csv"
results_df.to_csv(results_csv_path, index=False)
print("Saved table to", results_csv_path)


fig, axes = plt.subplots(1, 2, figsize=(16, 5))

plot_df = results_df.set_index("instance_id")
plot_df[["routefinder_distance", "ortools_distance"]].plot(kind="bar", ax=axes[0])
axes[0].set_title("RouteFinder vs OR-Tools Distance")
axes[0].set_ylabel("Total distance")
axes[0].grid(axis="y", alpha=0.3)

plot_df["gap_to_ortools_pct"].plot(kind="bar", ax=axes[1], color="tab:orange")
axes[1].axhline(0.0, color="black", linewidth=1)
axes[1].set_title("RouteFinder Gap vs OR-Tools")
axes[1].set_ylabel("Gap (%)")
axes[1].grid(axis="y", alpha=0.3)

fig.tight_layout()
plot_path = OUTPUT_ROOT / "routefinder_vs_ortools_summary.png"
fig.savefig(plot_path, dpi=180)
plt.show()
print("Saved summary plot to", plot_path)


for instance in instances[: min(3, len(instances))]:
    print(f"\n=== {instance.instance_id} ===")
    print("RouteFinder distance:", routefinder_solutions[instance.instance_id].total_distance)
    print("OR-Tools distance:", ortools_solutions[instance.instance_id].total_distance)
    plot_routes(instance, routefinder_solutions[instance.instance_id])
    plot_routes(instance, ortools_solutions[instance.instance_id])
import sys
import time
from pathlib import Path
import re
import shutil

import matplotlib.pyplot as plt
from tensordict import TensorDict
import pandas as pd
import torch
from torchrl.data.tensor_specs import BoundedContinuous, Composite, UnboundedContinuous
from IPython.display import display
from rl4co.utils.trainer import RL4COTrainer

try:
    from lightning.pytorch.callbacks import ModelCheckpoint
except ImportError:
    from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append(str(Path('..').resolve() / 'src'))

from dvrptw_bench.data.instance_filters import find_rc_instances
from dvrptw_bench.data.solomon_parser import parse_solomon
from dvrptw_bench.heuristics.ortools_solver import ORToolsVRPTWSolver
from dvrptw_bench.metrics.objective import total_distance
from dvrptw_bench.rl.routefinder_adapter import instance_to_routefinder_td, routefinder_actions_to_solution
from dvrptw_bench.rl.mtvrp_solomon_generator import SolomonMTVRPGenerator
from dvrptw_bench.rl.mtvrp_solomon_generator import MTVRPGenerator
from dvrptw_bench.viz.route_plot import plot_routes
from dvrptw_bench.common.typing import VRPTWInstance
from routefinder.envs.mtvrp import MTVRPEnv, MTVRPGenerator
from routefinder.models import RouteFinderBase, RouteFinderPolicy
from routefinder.utils import evaluate as evaluate_routefinder
from dvrptw_bench.rl.routefinder_adapter import _normalize_coord
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / Path('dataset/solomon_rc100')
RC_DATASET_ROOT = PROJECT_ROOT / Path('dataset/solomon_rc100')
C_DATASET_ROOT = PROJECT_ROOT / Path('dataset/solomon_c100')
OUTPUT_ROOT = PROJECT_ROOT / Path('outputs/notebook_routefinder')
CHECKPOINT_DIR = OUTPUT_ROOT / 'checkpoints'
INTERRUPTED_CHECKPOINT_PATH = CHECKPOINT_DIR / 'interrupted.ckpt'
CHECKPOINT_EVERY_N_EPOCHS = 2

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CUSTOMERS = 50
TARGET_EPOCHS = 200
BATCH_SIZE = 256
TRAIN_DATA_SIZE = 100_000
VAL_DATA_SIZE = 10_000
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-6
NUM_AUGMENT = 8
max_runtime = 3
num_procs = 4
ORTOOLS_TIME_LIMIT_S = 3
MAX_EVAL_INSTANCES = 5  # set to None to evaluate all RC instances
NORMALIZE_COORDS = True # whether to normalize coordinates to [0, 1] when creating RouteFinder training data
FINAL_CHECKPOINT_PATH = OUTPUT_ROOT / f'50_customers_routefinder_{TARGET_EPOCHS}_epochs.ckpt'

if torch.cuda.is_available():
    device = torch.device('cuda')
    accelerator = 'gpu'
elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
    device = torch.device('mps')
    accelerator = 'mps'
else:
    device = torch.device('cpu')
    accelerator = 'cpu'

print('Device:', device)
print('Dataset root:', DATASET_ROOT.resolve())
print('Output root:', OUTPUT_ROOT.resolve())

def _extract_epoch_from_checkpoint_name(path: Path) -> int:
    matches = re.findall(r'epoch[-_](\d+)', path.stem)
    if not matches:
        return -1
    return int(matches[-1])


def find_resume_checkpoint(checkpoint_dir: Path) -> Path | None:
    candidates = []

    interrupted_checkpoint = checkpoint_dir / 'interrupted.ckpt'
    if interrupted_checkpoint.exists():
        candidates.append(interrupted_checkpoint)

    last_checkpoint = checkpoint_dir / 'last.ckpt'
    if last_checkpoint.exists():
        candidates.append(last_checkpoint)

    candidates.extend(
        path
        for path in checkpoint_dir.glob('epoch-*.ckpt')
        if path.is_file()
    )

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda path: (
            path.name == 'interrupted.ckpt',
            path.name == 'last.ckpt',
            _extract_epoch_from_checkpoint_name(path),
            path.stat().st_mtime,
        ),
    )


def get_completed_epochs(checkpoint_path: Path | None) -> int:
    if checkpoint_path is None:
        return 0

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

    return max(int(checkpoint.get('epoch', -1)) + 1, 0)




def _normalize_coord(coord: torch.Tensor) -> torch.Tensor:
    x, y = coord[:, 0], coord[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_range = x_max - x_min
    y_range = y_max - y_min

    if torch.isclose(x_range, torch.tensor(0.0, device=coord.device)):
        x_scaled = torch.zeros_like(x)
    else:
        x_scaled = (x - x_min) / x_range

    if torch.isclose(y_range, torch.tensor(0.0, device=coord.device)):
        y_scaled = torch.zeros_like(y)
    else:
        y_scaled = (y - y_min) / y_range

    return torch.stack([x_scaled, y_scaled], dim=1)



def instance_to_generator_td(instance, num_customers):
    coords = [(instance.depot.x, instance.depot.y)] + [
        (c.x, c.y) for c in instance.customers
    ]

    locs = torch.tensor(coords, dtype=torch.float32)

    # enforce size
    assert locs.shape[0] >= num_customers + 1
    locs = locs[: num_customers + 1]

    # normalize
    if NORMALIZE_COORDS:
        locs = _normalize_coord(locs)

    return locs
rcInstances = [parse_solomon(instance, max_customers=NUM_CUSTOMERS) for instance in find_rc_instances(RC_DATASET_ROOT)]
cInstances = [parse_solomon(instance, max_customers=NUM_CUSTOMERS) for instance in find_rc_instances(C_DATASET_ROOT)]
generatorSolomonInstances = [instance_to_generator_td(instance, NUM_CUSTOMERS) for instance in rcInstances] + [instance_to_generator_td(instance, NUM_CUSTOMERS) for instance in cInstances]

generator = SolomonMTVRPGenerator(num_loc=NUM_CUSTOMERS, variant_preset='vrptw', solomon_instances=generatorSolomonInstances)
env = MTVRPEnv(generator, check_solution=False)
policy = RouteFinderPolicy(env_name=env.name).to(device)
model = RouteFinderBase(
    env,
    policy,
    batch_size=BATCH_SIZE,
    train_data_size=TRAIN_DATA_SIZE,
    val_data_size=VAL_DATA_SIZE,
    optimizer_kwargs={'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY},
)

checkpoint_callback = ModelCheckpoint(
    dirpath=CHECKPOINT_DIR,
    filename='epoch-{epoch:03d}',
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
    precision='32-true',
    callbacks=[checkpoint_callback],
)

resume_checkpoint = find_resume_checkpoint(CHECKPOINT_DIR)
completed_epochs = get_completed_epochs(resume_checkpoint)
remaining_epochs = max(TARGET_EPOCHS - completed_epochs, 0)

if resume_checkpoint is not None:
    print(f'Resuming training from checkpoint: {resume_checkpoint}')
    print(f'Completed epochs: {completed_epochs}/{TARGET_EPOCHS}. Remaining epochs: {remaining_epochs}')

if remaining_epochs == 0:
    if resume_checkpoint is not None and resume_checkpoint.resolve() != FINAL_CHECKPOINT_PATH.resolve():
        shutil.copy2(resume_checkpoint, FINAL_CHECKPOINT_PATH)
    print(f'Training already complete. Final checkpoint available at {FINAL_CHECKPOINT_PATH}')
    sys.exit(0)

try:
    super(RL4COTrainer, trainer).fit(
        model,
        ckpt_path=str(resume_checkpoint) if resume_checkpoint is not None else None,
        weights_only=False,
    )
except KeyboardInterrupt:
    trainer.save_checkpoint(INTERRUPTED_CHECKPOINT_PATH)
    print(f'Training interrupted. Saved resume checkpoint to {INTERRUPTED_CHECKPOINT_PATH}')
    raise

trainer.save_checkpoint(FINAL_CHECKPOINT_PATH)
if INTERRUPTED_CHECKPOINT_PATH.exists():
    INTERRUPTED_CHECKPOINT_PATH.unlink()

print(f'Final checkpoint saved to {FINAL_CHECKPOINT_PATH}')

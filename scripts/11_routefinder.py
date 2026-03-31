import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from tensordict import TensorDict
import pandas as pd
import torch
from IPython.display import display
from rl4co.utils.trainer import RL4COTrainer

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

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

NUM_CUSTOMERS = 50
NUM_EPOCHS = 200
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

trainer = RL4COTrainer(
    max_epochs=NUM_EPOCHS,
    accelerator=accelerator,
    devices=1,
    logger=None,
    precision='32-true',
)

trainer.fit(model) 
trainer.save_checkpoint(OUTPUT_ROOT / '50_customers_routefinder_200_epochs.pt')
# model = model.load_from_checkpoint('end_of_training.pt', weights_only=False)
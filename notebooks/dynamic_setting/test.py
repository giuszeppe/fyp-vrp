from matplotlib import pyplot as plt
from tensordict import TensorDict
import torch

from dvrptw_bench.dynamic.simulator import DynamicSimulator
from dvrptw_bench.rl.rl4co_policy import RL4COPolicy
from dvrptw_bench.rl.rl_model import RLModel, build_attention_model
from dvrptw_bench.viz.inspector import inspect_dynamic
from dvrptw_bench.heuristics.ortools_dynamic import ORToolsDVRPTWSolver
from rl4co.envs.routing import CVRPTWEnv
from rl4co.models.zoo.am import AttentionModel, AttentionModelPolicy
from rl4co.models.rl import REINFORCE
from rl4co.utils.trainer import RL4COTrainer

from pathlib import Path

from pyarrow import dataset

from dvrptw_bench.data.instance_filters import find_rc_instances
from dvrptw_bench.data.solomon_parser import parse_solomon
from dvrptw_bench.viz.route_plot import plot_routes
MAX_EPOCHS = 100
BATCH_SIZE = 512
TRAIN_DATA_SIZE = 100_000
VAL_DATA_SIZE = 10_000

dataset_path = Path("../../dataset/solomon_rc100")
instances = [parse_solomon(instance, max_customers=25, distance_scale=100) for instance in find_rc_instances(dataset_path)]
dod = 0
cutoff = 0.99
budget_s = 0.1
end_time_closeness = 0.9
soft_time_windows = True
# dynamic_instances = [build_dynamic_scenario(instance, epsilon=dod, seed=42) for instance in instances]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RL4CO env based on TorchRL
env = CVRPTWEnv(generator_params={'num_loc': 25})

# Policy: neural network, in this case with encoder-decoder architecture
policy = AttentionModelPolicy(env_name=env.name).to(device)

# RL Model: REINFORCE and greedy rollout baseline
model = AttentionModel(env, 
                    policy,
                    baseline="rollout",
                    batch_size=512,
                    train_data_size=100_000,
                    val_data_size=10_000,
                    optimizer_kwargs={"lr": 1e-4},
                    ) 


trainer = RL4COTrainer(
    max_epochs=3,
    accelerator="gpu",
    devices=1,
    logger=None,
)

trainer.fit(model)


def solve_function(am, instance, time_limit_s, warm_start = None):
    policy = RL4COPolicy(am)
    warm_start =policy.infer_instance(instance)
    plot_routes(instance, warm_start)
    plt.show()
    solver = ORToolsDVRPTWSolver(soft_time_windows=soft_time_windows)
    return solver.solve(instance, time_limit_s, warm_start)

model_weights = [f for f in Path("./model_weights/").glob("*.ckpt")][1]
print("Model weights found:", model_weights)
sim = DynamicSimulator(instances[0])
sols = {}
for model_weight in [model_weights]:
    am = build_attention_model(normalize_coords=True, num_loc=25, 
                               train_data_size=TRAIN_DATA_SIZE, 
                               val_data_size=VAL_DATA_SIZE, 
                               lr=1e-4, 
                               max_epochs=0, 
                               batch_size=BATCH_SIZE
                               )
    am.model= model
    am.policy = policy
    am.trainer = trainer
    am.env = env
    # am.env = CVRPTWEnv(generator_params={"num_loc": 25})
    # am.load(model_weight)
    
    res = sim.run(lambda instance, time_limit_s, warm_start: solve_function(am, instance, time_limit_s, warm_start), budget_s=budget_s, epsilon=dod, seed=15, cutoff_ratio=cutoff)
    sols[model_weight.stem] = res[0]
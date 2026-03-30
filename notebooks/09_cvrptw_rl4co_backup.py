# Aggregated from notebooks/09_cvrptw_rl4co.ipynb

import os
import sys
import torch
import torchrl
import rl4co
import vrplib
from tensordict import TensorDict

from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.models.rl import REINFORCE
from rl4co.utils.trainer import RL4COTrainer
from pathlib import Path

from tqdm import tqdm

import dvrptw_bench
from dvrptw_bench.common.typing import VRPTWInstance
from dvrptw_bench.data.solomon_parser import parse_solomon
from dvrptw_bench.heuristics.ortools_solver import ORToolsVRPTWSolver
from dvrptw_bench.rl.cvrptwenv import CVRPTWFixed


# ===== Cell 2 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_customers = 10

# RL4CO env based on TorchRL
env = CVRPTWFixed(
    generator_params={
        "num_loc": n_customers,
        "scale": True,
        "min_service_time": 10.0,
        "max_service_time": 10.0,
    }
)

# Policy: neural network, in this case with encoder-decoder architecture
policy = AttentionModelPolicy(env_name=env.name).to(device)

# RL Model: REINFORCE and greedy rollout baseline
model = REINFORCE(
    env,
    policy,
    baseline="rollout",
    batch_size=512,
    train_data_size=100_000,
    val_data_size=10_000,
    optimizer_kwargs={"lr": 1e-4},
)


# ===== Cell 3 =====
def scale_vrptw_features(
    coords: torch.Tensor,
    time_windows: torch.Tensor,
    durations: torch.Tensor,
    *,
    scale: bool,
    max_time: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not scale:
        return coords, time_windows, durations
    scale_factor = float(max_time)
    return coords / scale_factor, time_windows / scale_factor, durations / scale_factor


def adjusted_capacity(capacity: float, selected_customers: int, total_customers: int) -> float:
    if selected_customers >= total_customers:
        return float(capacity)
    return max(1.0, float(capacity) * (selected_customers / total_customers))


def instance_to_td(
    instance: VRPTWInstance,
    scale: bool = True,
    total_customers: int | None = None,
) -> TensorDict:
    coords = torch.tensor(
        [[instance.depot.x, instance.depot.y], *[[c.x, c.y] for c in instance.customers]],
        dtype=torch.float32,
    )

    demand = torch.tensor([c.demand for c in instance.customers], dtype=torch.float32)
    selected_customers = len(instance.customers)
    total_customers = selected_customers if total_customers is None else total_customers
    capacity = adjusted_capacity(instance.vehicle_capacity, selected_customers, total_customers)

    durations = torch.tensor(
        [0.0, *[c.service_time for c in instance.customers]],
        dtype=torch.float32,
    )
    time_windows = torch.tensor(
        [
            [instance.depot.ready_time, instance.depot.due_time],
            *[[c.ready_time, c.due_time] for c in instance.customers],
        ],
        dtype=torch.float32,
    )
    coords, time_windows, durations = scale_vrptw_features(
        coords,
        time_windows,
        durations,
        scale=scale,
        max_time=env.generator.max_time,
    )

    td = TensorDict(
        {
            "depot": coords[0],
            "locs": coords[1:],
            # Keep this convention internally consistent
            "demand": demand / capacity,
            "capacity": torch.tensor(1.0, dtype=torch.float32),
            "time_windows": time_windows,
            "durations": durations,
        },
        batch_size=[],
    )
    return td.unsqueeze(0)

def vrplib_to_td(problem, scale=True, max_customers=None):
    coords = torch.tensor(problem['node_coord']).float()
    demand = torch.tensor(problem['demand'][1:]).float()
    total_customers = len(problem['demand']) - 1
    durations = torch.tensor(problem['service_time']).float()
    time_windows = torch.tensor(problem['time_window']).float()
    n = coords.shape[0]
    if max_customers is not None:
        n = min(n, max_customers + 1)  # +1 for depot
    capacity = adjusted_capacity(problem['capacity'], n - 1, total_customers)
    coords, time_windows, durations = scale_vrptw_features(
        coords[:n],
        time_windows[:n],
        durations[:n],
        scale=scale,
        max_time=env.generator.max_time,
    )
    td = TensorDict({
        'depot': coords[0,:],
        'locs': coords[1:,:],
        'demand': demand[:n-1] / capacity, # normalized demand
        'capacity': torch.tensor(1.0, dtype=torch.float32),
        "time_windows": time_windows,
        "durations": durations,
    })
    td = td[None] # add batch dimension, in this case just 1
    return td

# ===== Cell 4 =====
datasets_path = Path("../dataset/solomon_rc100/")
instances = [[f.stem for f in datasets_path.glob("*.txt")][0]]


# ===== Cell 6 (Test Untrained) =====
tds, actions = [], []
for instance in instances:
    # Inference
    print(f"Processing instance: {instance}")
    full_problem = parse_solomon(datasets_path / (instance + ".txt"))
    problem = parse_solomon(datasets_path / (instance + ".txt"), max_customers=n_customers)
    td_reset = env.reset(
        instance_to_td(
            problem,
            scale=env.generator.scale,
            total_customers=len(full_problem.customers),
        ).to(device)
    )
    with torch.inference_mode():
        out = policy(td_reset.clone(), env, decode_type="greedy", num_samples=128, select_best=True)
        unscaled_td = env.reset(
            instance_to_td(
                problem,
                scale=False,
                total_customers=len(full_problem.customers),
            ).to(device)
        )
        cost = -env.get_reward(unscaled_td, out["actions"]).int().item()

    # Load the optimal cost
    solution = ORToolsVRPTWSolver().solve(parse_solomon(datasets_path / (instance + ".txt"), max_customers=n_customers), time_limit_s=10)
    optimal_cost = solution.total_distance
    # solution = vrplib.read_solution(datasets_path / (instance + ".sol"))
    # optimal_cost = solution["cost"]

    tds.append(td_reset)
    actions.append(out["actions"])

    # Calculate the gap and print
    gap = (cost - optimal_cost) / optimal_cost
    stringUntrained = f"Problem: {instance:<15} Cost: {cost:<8} BKS: {optimal_cost:<8}\t Gap: {gap:.2%}"
    print(f"Problem: {instance:<15} Cost: {cost:<8} BKS: {optimal_cost:<8}\t Gap: {gap:.2%}")


# ===== Cell 7 =====
# Plot some instances
# env.render(tds[0], actions[0].cpu())
# env.render(tds[-2], actions[-2].cpu())
# env.render(tds[-1], actions[-1].cpu())


# ===== Cell 9 (Train) =====
trainer = RL4COTrainer(
    max_epochs=10,
    accelerator="gpu",
    devices=1,
    logger=None,
)

trainer.fit(model)
trainer.save_checkpoint("cvrptw_am_10_epoches_checkpoint.ckpt")
# model = model.load_from_checkpoint("cvrptw_am_checkpoint.ckpt", weights_only=False, load_baseline=True)
#clear terminal 
# os.system('cls' if os.name == 'nt' else 'clear')


# ===== Cell 11 (Test trained) =====
policy = model.policy.to(device).eval()  # trained policy

tds, actions = [], []
for instance in instances:
    # Inference
    vrplib_problem = vrplib.read_instance(os.path.join(datasets_path, instance + ".txt"), "solomon")

    td_reset = env.reset(
        vrplib_to_td(
            vrplib_problem,
            scale=env.generator.scale,
            max_customers=n_customers,
        ).to(device)
    )

    # td_reset = env.reset(instance_to_td(problem).to(device))
    with torch.inference_mode():
        out = policy(td_reset.clone(), env, decode_type="greedy", num_samples=128, select_best=True)
        unscaled_td = env.reset(
            vrplib_to_td(
                vrplib_problem,
                scale=False,
                max_customers=n_customers,
            ).to(device)
        )
        cost = -env.get_reward(unscaled_td, out["actions"]).int().item()

    # Load the optimal cost
    # solution = vrplib.read_solution(os.path.join(datasets_path, instance + ".sol"))
    solution = ORToolsVRPTWSolver().solve(parse_solomon(datasets_path / (instance + ".txt"), max_customers=n_customers), time_limit_s=10)
    optimal_cost = solution.total_distance

    tds.append(td_reset)
    actions.append(out["actions"])

    # Calculate the gap and print
    gap = (cost - optimal_cost) / optimal_cost
    print(stringUntrained)
    print(f"Problem: {instance:<15} Cost: {cost:<8} BKS: {optimal_cost:<8}\t Gap: {gap:.2%}")

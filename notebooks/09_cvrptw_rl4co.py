# Aggregated from notebooks/09_cvrptw_rl4co.ipynb

import torch
import matplotlib.pyplot as plt
from tensordict import TensorDict

from lightning import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from rl4co.envs.routing.mtvrp.env import MTVRPEnv
from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.models.rl import REINFORCE
from rl4co.utils.trainer import RL4COTrainer
from pathlib import Path

from dvrptw_bench.common.typing import VRPTWInstance
from dvrptw_bench.data.solomon_parser import parse_solomon
from dvrptw_bench.heuristics.ortools_solver import ORToolsVRPTWSolver


# ===== Cell 2 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_customers = 10

# RL4CO env based on TorchRL
env = MTVRPEnv(
    generator_params={
        "num_loc": n_customers,
        "variant_preset": "vrptw",
        "subsample": True,
    }
)


# ===== Cell 3 =====
def scale_solomon_instance(
    coords: torch.Tensor,
    time_windows: torch.Tensor,
    service_time: torch.Tensor,
    *,
    scale: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not scale:
        return coords, time_windows, service_time

    mins = coords.min(dim=0).values
    span = (coords.max(dim=0).values - mins).max().clamp_min(1.0)
    coords_scaled = (coords - mins) / span
    return coords_scaled, time_windows / span, service_time / span


def solomon_to_mtvrp_td(
    instance: VRPTWInstance,
    scale: bool = True,
) -> TensorDict:
    coords = torch.tensor(
        [[instance.depot.x, instance.depot.y], *[[c.x, c.y] for c in instance.customers]],
        dtype=torch.float32,
    )
    capacity = float(instance.vehicle_capacity)
    demand_linehaul = torch.tensor(
        [0.0, *[c.demand for c in instance.customers]],
        dtype=torch.float32,
    )
    service_time = torch.tensor(
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
    coords, time_windows, service_time = scale_solomon_instance(
        coords,
        time_windows,
        service_time,
        scale=scale,
    )

    td = TensorDict(
        {
            "locs": coords,
            "demand_linehaul": demand_linehaul / capacity,
            "demand_backhaul": torch.zeros_like(demand_linehaul),
            "distance_limit": torch.tensor([float("inf")], dtype=torch.float32),
            "service_time": service_time,
            "open_route": torch.tensor([False], dtype=torch.bool),
            "time_windows": time_windows,
            "vehicle_capacity": torch.tensor([1.0], dtype=torch.float32),
            "capacity_original": torch.tensor([capacity], dtype=torch.float32),
            "speed": torch.tensor([1.0], dtype=torch.float32),
        },
        batch_size=[],
    )
    return td.unsqueeze(0)


def count_routes(actions: torch.Tensor) -> int:
    action_seq = actions.squeeze(0).tolist()
    route_count = 0
    in_route = False
    for node in action_seq:
        if node == 0:
            if in_route:
                route_count += 1
                in_route = False
            continue
        in_route = True
    if in_route:
        route_count += 1
    return route_count


def evaluate_policy_on_instance(
    policy: AttentionModelPolicy,
    problem: VRPTWInstance,
    label: str,
) -> dict:
    td_scaled = env.reset(solomon_to_mtvrp_td(problem, scale=True).to(device))
    td_raw = env.reset(solomon_to_mtvrp_td(problem, scale=False).to(device))
    with torch.inference_mode():
        out = policy(td_scaled.clone(), env, decode_type="greedy")
        cost = -env.get_reward(td_raw, out["actions"]).float().item()
    routes = count_routes(out["actions"])
    summary = f"{label:<10} Cost: {cost:>8.2f} Routes: {routes}"
    return {
        "label": label,
        "summary": summary,
        "cost": cost,
        "routes": routes,
        "td": td_scaled,
        "actions": out["actions"],
    }


def policy_distance(reference_policy: AttentionModelPolicy, other_policy: AttentionModelPolicy) -> float:
    squared_norm = 0.0
    for reference_param, other_param in zip(reference_policy.parameters(), other_policy.parameters()):
        delta = (other_param.detach() - reference_param.detach()).float()
        squared_norm += torch.sum(delta * delta).item()
    return squared_norm ** 0.5


def solve_reference(problem: VRPTWInstance, time_limit_s: int = 10) -> float:
    solution = ORToolsVRPTWSolver().solve(problem, time_limit_s=time_limit_s)
    return solution.total_distance


def print_comparison(
    instance_name: str,
    result: dict,
    reference_cost: float,
    *,
    baseline_actions: torch.Tensor | None = None,
) -> None:
    gap = (result["cost"] - reference_cost) / reference_cost
    diagnostics = []
    if baseline_actions is not None and result["label"] != "Untrained":
        same_actions = torch.equal(result["actions"].cpu(), baseline_actions.cpu())
        diagnostics.append(f"SameActions: {same_actions}")
        diagnostics.append(f"ParamDelta: {result['param_delta']:.4f}")
    diagnostics_str = f" {' | '.join(diagnostics)}" if diagnostics else ""
    print(
        f"Problem: {instance_name:<15} "
        f"{result['summary']} OR-Tools: {reference_cost:.2f} Gap: {gap:.2%}{diagnostics_str}"
    )


def build_model() -> REINFORCE:
    policy = AttentionModelPolicy(env_name=env.name).to(device)
    return REINFORCE(
        env,
        policy,
        baseline="rollout",
        batch_size=512,
        train_data_size=100_000,
        val_data_size=10_000,
        optimizer_kwargs={"lr": 1e-4},
    )


def train_single_run_with_checkpoints(
    max_epochs: int,
    checkpoint_dir: Path,
    seed: int,
) -> tuple[REINFORCE, dict[int, Path]]:
    seed_everything(seed, workers=True)
    model = build_model()
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch-{epoch:02d}",
        save_top_k=-1,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
    )
    trainer = RL4COTrainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=1,
        logger=None,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model)
    checkpoint_paths = {
        epoch: checkpoint_dir / f"epoch-{epoch - 1:02d}.ckpt"
        for epoch in range(1, max_epochs + 1)
    }
    return model, checkpoint_paths


def benchmark_instance(
    problem: VRPTWInstance,
    epoch_options: list[int],
    checkpoint_dir: Path,
    seed: int,
) -> list[dict]:
    results = []

    seed_everything(seed, workers=True)
    untrained_model = build_model()
    untrained_policy = untrained_model.policy.to(device).eval()
    untrained_result = evaluate_policy_on_instance(untrained_policy, problem, label="Untrained")
    untrained_result["param_delta"] = 0.0
    results.append(untrained_result)

    max_epochs = max(epoch_options)
    # trained_model, checkpoint_paths = train_single_run_with_checkpoints(
    #     max_epochs=max_epochs,
    #     checkpoint_dir=checkpoint_dir,
    #     seed=seed,
    # )
    # # max_epoch_policy = trained_model.policy.to(device).eval()
    # # max_epoch_result = evaluate_policy_on_instance(
    # #     max_epoch_policy,
    # #     problem,
    # #     label=f"{max_epochs} epochs",
    # # )
    # # max_epoch_result["param_delta"] = policy_distance(untrained_policy, max_epoch_policy)
    # results.append(max_epoch_result)
    checkpoint_paths = {
        10 : checkpoint_dir / f"epoch-10.ckpt",
        20 : checkpoint_dir / f"epoch-20.ckpt",
        # 30 : checkpoint_dir / f"epoch-30.ckpt",
        40 : checkpoint_dir / f"epoch-40.ckpt",
        49 : checkpoint_dir / f"epoch-49.ckpt",
        # epoch : checkpoint_dir / f"epoch-{epoch - 1:02d}.ckpt"

        # for epoch in range(10, max_epochs + 1,10)
    }

    for epochs in epoch_options:
        if epochs == max_epochs:
            continue
        checkpoint_path = checkpoint_paths[epochs]
        trained_model = REINFORCE.load_from_checkpoint(
            checkpoint_path,
            env=env,
            policy=AttentionModelPolicy(env_name=env.name).to(device),
            load_baseline=True,
            weights_only=False,
            map_location='cpu',
        )
        trained_policy = trained_model.policy.to(device).eval()
        result = evaluate_policy_on_instance(
            trained_policy,
            problem,
            label=f"{epochs} epochs",
        )
        result["param_delta"] = policy_distance(untrained_policy, trained_policy)
        result["routes"] = result["actions"].cpu().tolist()
        results.append(result)

    order = ["Untrained", *[f"{epochs} epochs" for epochs in epoch_options]]
    return sorted(results, key=lambda result: order.index(result["label"]))


def plot_benchmark_results(results: list[dict], reference_cost: float, instance_name: str) -> None:
    labels = [result["label"] for result in results]
    costs = [result["cost"] for result in results]
    gaps = [100.0 * (cost - reference_cost) / reference_cost for cost in costs]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(labels, costs, color="#4C78A8")
    axes[0].axhline(reference_cost, color="#F58518", linestyle="--", label="OR-Tools")
    axes[0].set_title(f"{instance_name} Cost Comparison")
    axes[0].set_ylabel("Cost")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend()

    axes[1].bar(labels, gaps, color="#54A24B")
    axes[1].axhline(0.0, color="black", linestyle=":")
    axes[1].set_title(f"{instance_name} Gap vs OR-Tools")
    axes[1].set_ylabel("Gap (%)")
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.show()

# ===== Cell 4 =====
datasets_path = Path("../dataset/solomon_rc100/")
instances = [[f.stem for f in datasets_path.glob("*.txt")][0]]

# ===== Cell 6 (Benchmark Epochs) =====
epoch_options = [10,20,40,49]
seed = 1234
checkpoint_root = Path("./epoch_benchmark_checkpoints")
benchmark_results = {}

for instance in instances:
    print(f"Processing instance: {instance}")
    problem = parse_solomon(datasets_path / (instance + ".txt"), max_customers=n_customers)
    reference_cost = solve_reference(problem, time_limit_s=10)
    checkpoint_dir = checkpoint_root / instance
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results = benchmark_instance(
        problem,
        epoch_options=epoch_options,
        checkpoint_dir=checkpoint_dir,
        seed=seed,
    )
    benchmark_results[instance] = {
        "reference_cost": reference_cost,
        "results": results,
    }

    for result in results:
        print_comparison(instance, result, reference_cost)


# ===== Cell 7 (Plot Benchmark) =====
for instance, benchmark in benchmark_results.items():
    plot_benchmark_results(
        benchmark["results"],
        benchmark["reference_cost"],
        instance_name=instance,
    )

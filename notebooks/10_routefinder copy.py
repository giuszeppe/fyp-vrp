"""Aggregated RouteFinder quickstart script.

This is a plain Python version of the upstream notebook at
`routefinder/examples/1.quickstart.ipynb`.
"""

from __future__ import annotations

import argparse

from tensordict import TensorDict
import torch
from rl4co.utils.trainer import RL4COTrainer

from routefinder.envs.mtvrp import MTVRPEnv, MTVRPGenerator
from routefinder.models import RouteFinderBase, RouteFinderPolicy
from routefinder.utils import evaluate, greedy_policy, rollout

try:
    from routefinder.baselines.solve import solve as solve_baseline
except ImportError:
    solve_baseline = None


def build_env(batch_size: int) -> tuple[MTVRPEnv, TensorDict, list[str]|str]:
    generator = MTVRPGenerator(num_loc=50, variant_preset="vrptw")
    env = MTVRPEnv(generator, check_solution=False)
    td_data = env.generator(batch_size)
    variant_names = env.get_variant_names(td_data)
    td_test = env.reset(td_data)
    return env, td_test, variant_names


def run_greedy_demo(env: MTVRPEnv, td_test: TensorDict, variant_names: list[str]|str) -> torch.Tensor:
    actions = rollout(env, td_test.clone(), greedy_policy)
    rewards = env.get_reward(td_test, actions)

    print(f"Greedy average cost: {-rewards.mean():.3f}")
    for idx in range(min(3, td_test.batch_size[0])):
        env.render(td_test[idx], actions[idx])
        print(f"Greedy instance {idx} | Cost: {-rewards[idx].item():.3f} | Problem: {variant_names[idx]}")
    return rewards


def build_model(env: MTVRPEnv) -> RouteFinderBase:
    policy = RouteFinderPolicy()
    return RouteFinderBase(
        env,
        policy,
        batch_size=256,
        train_data_size=100_000,
        val_data_size=10_000,
        optimizer_kwargs={"lr": 3e-4, "weight_decay": 1e-6},
    )


def run_untrained_demo(
    env: MTVRPEnv,
    model: RouteFinderBase,
    td_test: TensorDict,
    device: torch.device,
) -> None:
    policy = model.policy.to(device)
    out = policy(
        td_test.clone().to(device),
        env,
        phase="test",
        decode_type="greedy",
        return_actions=True,
    )

    actions = out["actions"].cpu().detach()
    rewards = out["reward"].cpu().detach()

    for idx in range(min(3, td_test.batch_size[0])):
        print(f"Untrained instance {idx} | Cost: {-rewards[idx]:.3f}")
        env.render(td_test[idx], actions[idx])

    
def train_model(
    model: RouteFinderBase,
    epochs: int,
    device: torch.device,
) -> None:
    accelerator = "gpu" if device.type == "cuda" else "cpu"
    devices = 1

    trainer = RL4COTrainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices,
        logger=None,
    )
    trainer.fit(model)


def run_evaluation(
    env: MTVRPEnv,
    model: RouteFinderBase,
    td_test: TensorDict,
    device: torch.device,
) -> torch.Tensor:
    td_eval = td_test.to(device)
    model = model.to(device)
    out = evaluate(model, td_eval.clone())

    actions = out["best_aug_actions"]
    rewards = env.get_reward(td_eval, actions)

    print(f"RouteFinder average cost: {-rewards.mean():.3f}")
    for idx in range(min(3, td_test.batch_size[0])):
        print(f"Trained instance {idx} | Cost: {-rewards[idx]:.3f}")
        print("Variant", env.get_variant_names(td_eval[idx]))
        env.render(td_eval[idx].cpu(), actions[idx].cpu())

    return rewards.cpu()


def run_pyvrp_baseline(
    env: MTVRPEnv,
    td_test: TensorDict,
    num_procs: int,
) -> torch.Tensor | None:
    if solve_baseline is None:
        print("Skipping PyVRP baseline: routefinder baseline solvers are not installed.")
        return None

    actions_pyvrp, _ = solve_baseline(
        td_test.cpu(),
        max_runtime=10.0,
        num_procs=num_procs,
        solver="pyvrp",
    )
    rewards_pyvrp = env.get_reward(td_test.clone().cpu(), actions_pyvrp)
    print(f"PyVRP average cost: {-rewards_pyvrp.mean():.3f}")
    return rewards_pyvrp.cpu()


def gap(sol: torch.Tensor, bks: torch.Tensor) -> torch.Tensor:
    return ((sol - bks) / bks).mean() * 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RouteFinder quickstart flow.")
    parser.add_argument("--batch-size", type=int, default=32, help="Number of synthetic VRPTW instances to generate.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs for the demo.")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Build the model and run only greedy plus untrained inference.",
    )
    parser.add_argument(
        "--num-procs",
        type=int,
        default=4,
        help="Number of processes for the optional PyVRP baseline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env, td_test, variant_names = build_env(args.batch_size)
    rewards_greedy = run_greedy_demo(env, td_test, variant_names)

    model = build_model(env)
    run_untrained_demo(env, model, td_test, device)

    if args.skip_training:
        return

    train_model(model, args.epochs, device)
    rewards_model = run_evaluation(env, model, td_test, device)

    rewards_pyvrp = run_pyvrp_baseline(env, td_test, args.num_procs)
    if rewards_pyvrp is not None:
        print(f"Nearest Neighbor gap to HGS-PyVRP: {gap(rewards_greedy, rewards_pyvrp):.3f}%")
        print(f"RouteFinder gap to HGS-PyVRP: {gap(rewards_model, rewards_pyvrp):.3f}%")


if __name__ == "__main__":
    main()

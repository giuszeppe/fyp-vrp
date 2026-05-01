"""RouteFinder model builder and solver wrapper."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from rl4co.utils.trainer import RL4COTrainer

from dvrptw_bench.common.typing import Solution, VRPTWInstance
from dvrptw_bench.metrics.objective import total_distance
from dvrptw_bench.rl.routefinder_adapter import (
    instance_to_routefinder_td,
    routefinder_actions_to_solution,
)
from routefinder.envs.mtvrp import MTVRPEnv, MTVRPGenerator
from routefinder.models import RouteFinderBase, RouteFinderPolicy
from routefinder.utils import evaluate as evaluate_routefinder


class RouteFinderModel:
    def __init__(
        self,
        device=None,
        env=None,
        policy=None,
        model=None,
        max_epochs: int = 100,
        batch_size: int = 256,
        train_data_size: int = 100_000,
        val_data_size: int = 10_000,
        lr: float = 3e-4,
        weight_decay: float = 1e-6,
        num_loc: int = 100,
        normalize_coords: bool = True,
        variant_preset: str = "vrptw",
    ):
        self.device = device
        self.env = env
        self.policy = policy
        self.model = model
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.train_data_size = train_data_size
        self.val_data_size = val_data_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_loc = num_loc
        self.normalize_coords = normalize_coords
        self.variant_preset = variant_preset

        if self.device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

        if self.env is None:
            generator = MTVRPGenerator(num_loc=num_loc, variant_preset=variant_preset)
            self.env = MTVRPEnv(generator, check_solution=False)

        if self.policy is None:
            self.policy = RouteFinderPolicy(env_name=self.env.name).to(self.device)

        if self.model is None:
            self.model = RouteFinderBase(
                self.env,
                self.policy,
                batch_size=self.batch_size,
                train_data_size=self.train_data_size,
                val_data_size=self.val_data_size,
                optimizer_kwargs={"lr": self.lr, "weight_decay": self.weight_decay},
            )

        accelerator = "cpu"
        devices = 1
        if self.device.type == "cuda":
            accelerator = "gpu"
        elif self.device.type == "mps":
            accelerator = "mps"

        self.trainer = RL4COTrainer(
            max_epochs=self.max_epochs,
            accelerator=accelerator,
            devices=devices,
            logger=None,
        )

    def train(self) -> None:
        self.trainer.fit(self.model)

    def save(self, path: str | Path) -> None:
        self.trainer.save_checkpoint(str(path))

    def load(self, path: str | Path) -> None:
        self.model = RouteFinderBase.load_from_checkpoint(
            str(path),
            map_location=self.device,
            env=self.env,
            policy=self.policy,
            strict=False,
            weights_only=False,
        )
        self.model.to(self.device)
        self.model.eval()
        self.policy = self.model.policy

    def solve(
        self,
        instance: VRPTWInstance,
        decode_type: str = "greedy",
        num_samples: int = 1,
        num_starts: int | None = None,
        select_best: bool = True,
        num_augment: int = 8,
    ) -> Solution:
        t0 = time.perf_counter()
        td = instance_to_routefinder_td(
            instance,
            normalize_coords=self.normalize_coords,
        ).to(self.device)
        td_reset = self.env.reset(td)

        self.model.to(self.device).eval()
        policy = self.model.policy.to(self.device).eval()

        with torch.inference_mode():
            if decode_type == "multistart":
                out = evaluate_routefinder(
                    self.model,
                    td_reset.clone(),
                    num_augment=max(1, num_augment),
                    num_starts=max(2, num_starts or 2),
                )
                actions = out.get(
                    "best_aug_actions",
                    out.get("best_multistart_actions", out.get("actions")),
                )
            elif decode_type == "greedy" and select_best and num_augment > 1:
                out = evaluate_routefinder(
                    self.model,
                    td_reset.clone(),
                    num_augment=max(1, num_augment),
                )
                actions = out.get(
                    "best_aug_actions",
                    out.get("best_multistart_actions", out.get("actions")),
                )
            elif decode_type == "sampling":
                out = policy(
                    td_reset.clone(),
                    self.env,
                    phase="test",
                    decode_type="sampling",
                    num_samples=num_samples,
                    select_best=select_best,
                    return_actions=True,
                )
                actions = out["actions"]
            else:
                out = policy(
                    td_reset.clone(),
                    self.env,
                    phase="test",
                    decode_type="greedy",
                    return_actions=True,
                )
                actions = out["actions"]

        solution = routefinder_actions_to_solution(actions, instance)
        solution.total_distance = total_distance(instance, solution)
        solution.solve_time_s = time.perf_counter() - t0
        solution.details.update(
            {
                "decode_type": decode_type,
                "num_samples": num_samples,
                "num_starts": num_starts,
                "select_best": select_best,
                "num_augment": num_augment,
            }
        )
        return solution


def build_routefinder_model(
    *,
    device=None,
    checkpoint_path: str | Path | None = None,
    max_epochs: int = 100,
    batch_size: int = 256,
    train_data_size: int = 100_000,
    val_data_size: int = 10_000,
    lr: float = 3e-4,
    weight_decay: float = 1e-6,
    num_loc: int = 100,
    normalize_coords: bool = True,
    variant_preset: str = "vrptw",
) -> RouteFinderModel:
    model = RouteFinderModel(
        device=device,
        max_epochs=max_epochs,
        batch_size=batch_size,
        train_data_size=train_data_size,
        val_data_size=val_data_size,
        lr=lr,
        weight_decay=weight_decay,
        num_loc=num_loc,
        normalize_coords=normalize_coords,
        variant_preset=variant_preset,
    )
    if checkpoint_path is not None:
        model.load(checkpoint_path)
    return model

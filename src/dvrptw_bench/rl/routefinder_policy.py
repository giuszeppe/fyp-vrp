"""RouteFinder training and inference adapter."""

from __future__ import annotations

from pathlib import Path

import torch

from dvrptw_bench.common.typing import Solution, VRPTWInstance
from dvrptw_bench.rl.routefinder_model import RouteFinderModel, build_routefinder_model


class RouteFinderAdapterPolicy:
    name = "routefinder"

    def __init__(self, model: RouteFinderModel | None = None):
        self.model = model

    def train(
        self,
        epochs: int = 1,
        train_size: int = 100_000,
        val_size: int = 10_000,
        batch_size: int = 256,
        lr: float = 3e-4,
        weight_decay: float = 1e-6,
        device: torch.device | None = None,
        normalize_coords: bool = True,
        variant_preset: str = "vrptw",
        num_loc: int = 100,
    ) -> dict:
        model = build_routefinder_model(
            device=device,
            max_epochs=epochs,
            batch_size=batch_size,
            train_data_size=train_size,
            val_data_size=val_size,
            lr=lr,
            weight_decay=weight_decay,
            num_loc=num_loc,
            normalize_coords=normalize_coords,
            variant_preset=variant_preset,
        )
        model.train()
        self.model = model
        return {"status": "ok"}

    def load(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | None = None,
        normalize_coords: bool = True,
        variant_preset: str = "vrptw",
        num_loc: int = 100,
    ) -> None:
        self.model = build_routefinder_model(
            device=device,
            checkpoint_path=checkpoint_path,
            normalize_coords=normalize_coords,
            variant_preset=variant_preset,
            num_loc=num_loc,
        )

    def infer_instance(
        self,
        instance: VRPTWInstance,
        decode_type: str = "greedy",
        num_samples: int = 1,
        num_starts: int | None = None,
        select_best: bool = True,
        num_augment: int = 8,
    ) -> Solution:
        if self.model is None:
            self.model = build_routefinder_model(
                num_loc=max(1, instance.n_customers),
                normalize_coords=True,
                variant_preset="vrptw",
            )
        return self.model.solve(
            instance,
            decode_type=decode_type,
            num_samples=num_samples,
            num_starts=num_starts,
            select_best=select_best,
            num_augment=num_augment,
        )

    def infer_solution(self, instance: VRPTWInstance) -> Solution:
        return self.infer_instance(instance)

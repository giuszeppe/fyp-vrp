"""RL4CO training/inference adapter with graceful fallback."""

from __future__ import annotations

from dvrptw_bench.common.typing import Solution, VRPTWInstance
from dvrptw_bench.rl.rl_model import build_attention_model


class RL4COPolicy:
    name = "rl4co"

    def __init__(self, model=None):
        self.model = model

    def train(
        self,
        epochs: int = 1,
        train_size: int = 100_000,
        val_size: int = 10_000,
        batch_size: int = 512,
        lr: float = 1e-4,
        device: str | None = None,
        normalize_coords: bool = True,
        der_templates: dict | None = None,
        family_specs=None,
        der_seed: int = 123,
    ) -> dict:
        am = build_attention_model(
            device=device,
            max_epochs=epochs,
            batch_size=batch_size,
            train_data_size=train_size,
            val_data_size=val_size,
            lr=lr,
            normalize_coords=normalize_coords,
            der_templates=der_templates,
            family_specs=family_specs,
            der_seed=der_seed,
        )
        if am is None:
            return {"status": "rl4co_unavailable"}

        self.model = am
        self.model.train()
        return {"status": "ok"}

    def infer_instance(self, instance: VRPTWInstance) -> Solution:
        solution = self.model.solve(instance)
        return solution

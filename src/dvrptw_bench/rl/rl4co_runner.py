"""RL4CO training/inference adapter with graceful fallback."""

from __future__ import annotations

import random

from dvrptw_bench.common.typing import Route, Solution, VRPTWInstance
from dvrptw_bench.rl.rl4co_model_zoo import build_attention_model


class RL4COPolicy:
    name = "rl4co"

    def __init__(self, model=None):
        self.model = model

    def train(self, epochs: int = 1, train_size: int = 128, val_size: int = 32, device: str = "cpu") -> dict:
        am = build_attention_model()
        if am is None:
            return {"status": "rl4co_unavailable", "epochs": 0}
        self.model = am
        self.model.train()
        return {"status": "ok", "epochs": epochs, "train_size": train_size, "val_size": val_size, "device": device}

    def infer_instance(self, instance: VRPTWInstance) -> Solution:
        if self.model is None:
            return None
        return self.model.solve(instance)

    def infer(self, snapshot_state):
        if self.model is None:
            return {"status": "rl4co_unavailable", "solution": None}
        solution = self.infer_instance(snapshot_state.instance)
        return {"status": "ok", "solution": solution}

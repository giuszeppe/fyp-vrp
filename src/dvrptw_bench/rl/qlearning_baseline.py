"""Didactic tabular Q-learning baseline."""

from __future__ import annotations

import random
from collections import defaultdict

from dvrptw_bench.common.typing import Route, Solution


class QLearningPolicy:
    name = "qlearning"

    def __init__(self):
        self.q = defaultdict(float)

    def train(self, episodes: int = 200, alpha: float = 0.1, gamma: float = 0.95, eps: float = 0.2) -> None:
        for _ in range(episodes):
            state = (4, 4, 4)
            for _step in range(30):
                actions = [0, 1, 2, 3]
                if random.random() < eps:
                    a = random.choice(actions)
                else:
                    a = max(actions, key=lambda x: self.q[(state, x)])
                reward = -float(a)
                next_state = (max(0, state[0] - 1), state[1], max(0, state[2] - 1))
                best_next = max(self.q[(next_state, na)] for na in actions)
                self.q[(state, a)] += alpha * (reward + gamma * best_next - self.q[(state, a)])
                state = next_state

    def infer(self, snapshot_state):
        if not self.q:
            self.train()
        ids = [n.id for n in snapshot_state.remaining_customers]
        ids_sorted = sorted(ids)
        return Solution(strategy=self.name, routes=[Route(vehicle_id=0, node_ids=ids_sorted)])

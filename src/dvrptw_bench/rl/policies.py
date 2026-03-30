"""Policy registry and helpers."""

from __future__ import annotations

from dvrptw_bench.rl.ga_baseline import GAPolicy
from dvrptw_bench.rl.qlearning_baseline import QLearningPolicy
from dvrptw_bench.rl.routefinder_policy import RouteFinderAdapterPolicy
from dvrptw_bench.rl.rl4co_policy import RL4COPolicy


def build_policy(name: str):
    key = name.lower()
    if key == "ga":
        return GAPolicy()
    if key == "qlearning":
        return QLearningPolicy()
    if key == "rl4co":
        return RL4COPolicy()
    if key == "routefinder":
        return RouteFinderAdapterPolicy()
    raise ValueError(f"Unknown policy: {name}")

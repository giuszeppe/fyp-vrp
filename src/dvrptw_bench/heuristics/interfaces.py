"""Interfaces for heuristic and OR solvers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from dvrptw_bench.common.typing import Solution, VRPTWInstance


class HeuristicSolver(ABC):
    name: str

    @abstractmethod
    def solve(self, instance: VRPTWInstance, time_limit_s: float, warm_start: Solution | None = None) -> Solution:
        pass

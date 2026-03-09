"""Interfaces for RL policies."""

from __future__ import annotations

from abc import ABC, abstractmethod

from dvrptw_bench.common.typing import Solution
from dvrptw_bench.dynamic.snapshot import SnapshotState


class RLPolicy(ABC):
    name: str

    @abstractmethod
    def infer(self, snapshot_state: SnapshotState) -> Solution:
        pass

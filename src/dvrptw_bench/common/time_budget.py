"""Time budget helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class BudgetTimer:
    """Simple wall-clock budget timer."""

    budget_s: float
    start_s: float = 0.0

    def __post_init__(self) -> None:
        self.start_s = time.perf_counter()

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self.start_s

    @property
    def remaining_s(self) -> float:
        return max(0.0, self.budget_s - self.elapsed_s)

    @property
    def expired(self) -> bool:
        return self.elapsed_s >= self.budget_s

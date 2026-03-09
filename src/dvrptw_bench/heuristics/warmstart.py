"""Warm-start utility methods."""

from __future__ import annotations

from dvrptw_bench.common.typing import Solution


def as_route_order(solution: Solution) -> list[list[int]]:
    return [r.node_ids[:] for r in solution.routes]

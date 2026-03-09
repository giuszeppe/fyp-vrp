"""Distance and coordinate normalization utilities."""

from __future__ import annotations

import math

from dvrptw_bench.common.typing import Node


def euclidean(a: Node, b: Node, scale: float = 1.0) -> float:
    return scale * math.hypot(a.x - b.x, a.y - b.y)


def distance_matrix(nodes: list[Node], scale: float = 1.0) -> list[list[float]]:
    return [[euclidean(i, j, scale=scale) for j in nodes] for i in nodes]

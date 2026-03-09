"""Local search operators for route-based VRPTW solutions."""

from __future__ import annotations

from copy import deepcopy
import random

from dvrptw_bench.common.typing import Solution


def relocate(solution: Solution, rng: random.Random | None = None) -> Solution:
    new = deepcopy(solution)
    rng = rng or random
    candidates = [r for r in new.routes if len(r.node_ids) > 1]
    if candidates:
        r = rng.choice(candidates)
        i = rng.randrange(len(r.node_ids))
        j = rng.randrange(len(r.node_ids))
        node = r.node_ids.pop(i)
        r.node_ids.insert(j, node)
    return new


def swap(solution: Solution, rng: random.Random | None = None) -> Solution:
    new = deepcopy(solution)
    rng = rng or random
    non_empty = [r for r in new.routes if r.node_ids]
    if len(non_empty) >= 2:
        a, b = rng.sample(non_empty, 2)
        ia = rng.randrange(len(a.node_ids))
        ib = rng.randrange(len(b.node_ids))
        a.node_ids[ia], b.node_ids[ib] = b.node_ids[ib], a.node_ids[ia]
    return new


def two_opt(solution: Solution, rng: random.Random | None = None) -> Solution:
    new = deepcopy(solution)
    rng = rng or random
    candidates = [r for r in new.routes if len(r.node_ids) >= 4]
    if candidates:
        r = rng.choice(candidates)
        i, j = sorted(rng.sample(range(len(r.node_ids)), 2))
        if i != j:
            r.node_ids[i : j + 1] = list(reversed(r.node_ids[i : j + 1]))
    return new


def cross_exchange(solution: Solution, rng: random.Random | None = None) -> Solution:
    new = deepcopy(solution)
    rng = rng or random
    non_empty = [r for r in new.routes if r.node_ids]
    if len(non_empty) >= 2:
        a, b = rng.sample(non_empty, 2)
        ia = rng.randrange(len(a.node_ids))
        ib = rng.randrange(len(b.node_ids))
        a.node_ids[ia], b.node_ids[ib] = b.node_ids[ib], a.node_ids[ia]
    return new

from __future__ import annotations

import pytest
import torch

from dvrptw_bench.common.typing import Node, VRPTWInstance
from dvrptw_bench.rl.policies import build_policy
from dvrptw_bench.rl.routefinder_adapter import (
    decode_routefinder_actions,
    instance_to_routefinder_td,
)

try:
    import routefinder  # noqa: F401

    HAS_ROUTEFINDER = True
except ImportError:
    HAS_ROUTEFINDER = False


def _tiny_instance() -> VRPTWInstance:
    depot = Node(id=0, x=0, y=0, ready_time=0, due_time=100, service_time=0)
    c1 = Node(id=1, x=1, y=0, demand=1, ready_time=0, due_time=100, service_time=0)
    c2 = Node(id=2, x=0, y=1, demand=1, ready_time=0, due_time=100, service_time=0)
    return VRPTWInstance(
        instance_id="tiny-rf",
        depot=depot,
        customers=[c1, c2],
        vehicle_capacity=10,
        vehicle_count=2,
        distance_matrix=[
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.41],
            [1.0, 1.41, 0.0],
        ],
    )


def test_instance_to_routefinder_td_shapes() -> None:
    td = instance_to_routefinder_td(_tiny_instance())

    assert td["locs"].shape == torch.Size([1, 3, 2])
    assert td["demand_linehaul"].shape == torch.Size([1, 2])
    assert td["service_time"].shape == torch.Size([1, 3])
    assert td["time_windows"].shape == torch.Size([1, 3, 2])


def test_decode_routefinder_actions_routes() -> None:
    routes = decode_routefinder_actions(torch.tensor([[1, 2, 0]]), _tiny_instance())
    assert [route.node_ids for route in routes] == [[1, 2]]


def test_build_policy_routefinder() -> None:
    policy = build_policy("routefinder")
    assert policy.name == "routefinder"


@pytest.mark.skipif(not HAS_ROUTEFINDER, reason="routefinder is not installed")
def test_routefinder_policy_infer_instance_smoke() -> None:
    policy = build_policy("routefinder")
    solution = policy.infer_instance(_tiny_instance(), num_augment=1)

    served = sorted(node_id for route in solution.routes for node_id in route.node_ids)
    assert solution.strategy == "routefinder"
    assert served == [1, 2]

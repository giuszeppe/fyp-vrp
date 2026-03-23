from dvrptw_bench.common.typing import Node, VRPTWInstance
from dvrptw_bench.data.normalization import distance_matrix
from dvrptw_bench.dynamic.dynamic_instance import DynamicInstance
from dvrptw_bench.dynamic.snapshot import SnapshotState, VehicleState
from dvrptw_bench.metrics.objective import total_distance
from dvrptw_bench.common.typing import Route, Solution


def _base_instance() -> VRPTWInstance:
    depot = Node(id=0, x=0, y=0, demand=0, ready_time=0, due_time=200, service_time=0)
    customers = [
        Node(id=10, x=3, y=0, demand=1, ready_time=0, due_time=200, service_time=1),
        Node(id=20, x=6, y=0, demand=1, ready_time=0, due_time=200, service_time=1),
    ]
    return VRPTWInstance(
        instance_id="dynamic_indexing",
        depot=depot,
        customers=customers,
        vehicle_capacity=2,
        vehicle_count=2,
        distance_matrix=distance_matrix([depot, *customers]),
    )


def test_dynamic_instance_starts_are_manager_node_positions():
    base = _base_instance()
    snapshot = SnapshotState(
        time=5,
        remaining_customers=base.customers,
        active_customer_ids={10, 20},
        served_customer_ids=set(),
        vehicles=[
            VehicleState(
                vehicle_id=0,
                x=0,
                y=0,
                remaining_capacity=2,
                elapsed_time=5,
                planned_route=[],
            ),
            VehicleState(
                vehicle_id=1,
                x=4,
                y=0,
                remaining_capacity=2,
                elapsed_time=5,
                planned_route=[],
            ),
        ],
    )

    inst = DynamicInstance(snapshot, base)

    assert inst.starts == [0, 3]
    assert inst.ends == [0, 0]
    assert inst.node_pos_for_customer(10) == 1
    assert inst.node_pos_for_customer(20) == 2
    assert inst.customer_id_for_node_pos(1) == 10
    assert inst.customer_id_for_node_pos(2) == 20


def test_dynamic_total_distance_uses_customer_mapping_not_customer_ids_as_matrix_indices():
    base = _base_instance()
    snapshot = SnapshotState(
        time=0,
        remaining_customers=base.customers,
        active_customer_ids={10, 20},
        served_customer_ids=set(),
        vehicles=[
            VehicleState(
                vehicle_id=0,
                x=0,
                y=0,
                remaining_capacity=2,
                elapsed_time=0,
                planned_route=[],
            ),
            VehicleState(
                vehicle_id=1,
                x=4,
                y=0,
                remaining_capacity=2,
                elapsed_time=0,
                planned_route=[],
            ),
        ],
    )

    inst = DynamicInstance(snapshot, base)
    solution = Solution(
        strategy="test",
        routes=[
            Route(vehicle_id=0, node_ids=[10]),
            Route(vehicle_id=1, node_ids=[20]),
        ],
    )

    assert total_distance(inst, solution) == 11.0

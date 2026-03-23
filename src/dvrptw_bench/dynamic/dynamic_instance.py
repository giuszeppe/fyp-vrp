from __future__ import annotations

from dataclasses import dataclass

from dvrptw_bench.common.typing import Node, VRPTWInstance
from dvrptw_bench.data.normalization import distance_matrix as compute_distance_matrix
from dvrptw_bench.dynamic.snapshot import SnapshotState, VehicleState


@dataclass(frozen=True)
class DynamicNodeMeta:
    kind: str
    customer_id: int | None = None
    vehicle_id: int | None = None


class DynamicInstance:
    """Residual snapshot instance indexed in OR-Tools manager node space."""

    depot: Node
    starts: list[int]
    ends: list[int]
    distance_matrix: list[list[float]]
    customers: list[Node]
    vehicles: list[VehicleState]
    instance_id: str
    vehicle_count: int
    vehicle_capacity: float
    all_nodes: list[Node]
    node_meta: list[DynamicNodeMeta]
    node_pos_by_customer_id: dict[int, int]
    start_node_pos_by_vehicle_id: dict[int, int]
    customer_id_by_node_pos: dict[int, int]
    n_customers: int

    def __init__(self, snapshot: SnapshotState, base_instance: VRPTWInstance):
        self.instance_id = base_instance.instance_id
        self.depot = base_instance.depot
        self.vehicle_count = base_instance.vehicle_count
        self.vehicle_capacity = base_instance.vehicle_capacity
        self.vehicles = snapshot.vehicles
        self.customers = list(snapshot.remaining_customers)

        self.all_nodes = [self.depot, *self.customers]
        self.node_meta = [DynamicNodeMeta(kind="depot")] + [
            DynamicNodeMeta(kind="customer", customer_id=customer.id) for customer in self.customers
        ]
        self.node_pos_by_customer_id = {
            customer.id: node_pos for node_pos, customer in enumerate(self.all_nodes[1:], start=1)
        }
        self.customer_id_by_node_pos = {
            node_pos: customer_id for customer_id, node_pos in self.node_pos_by_customer_id.items()
        }
        self.n_customers = len(self.customers)

        self.starts = []
        self.ends = [0] * self.vehicle_count
        self.start_node_pos_by_vehicle_id = {}

        for vehicle in snapshot.vehicles:
            ready_time = vehicle.elapsed_time + vehicle.remaining_service_time
            is_at_depot = vehicle.x == int(self.depot.x) and vehicle.y == int(self.depot.y)
            use_depot_start = is_at_depot and vehicle.remaining_service_time == 0
            if use_depot_start:
                start_node_pos = 0
            else:
                start_node_pos = len(self.all_nodes)
                start_node = Node(
                    id=start_node_pos,
                    x=vehicle.x,
                    y=vehicle.y,
                    demand=0.0,
                    ready_time=ready_time,
                    due_time=base_instance.depot.due_time,
                    service_time=0.0,
                )
                self.all_nodes.append(start_node)
                self.node_meta.append(DynamicNodeMeta(kind="vehicle_start", vehicle_id=vehicle.vehicle_id))
            self.starts.append(start_node_pos)
            self.start_node_pos_by_vehicle_id[vehicle.vehicle_id] = start_node_pos

        self.distance_matrix = compute_distance_matrix(self.all_nodes)

    def node_pos_for_customer(self, customer_id: int) -> int | None:
        return self.node_pos_by_customer_id.get(customer_id)

    def customer_id_for_node_pos(self, node_pos: int) -> int | None:
        return self.customer_id_by_node_pos.get(node_pos)

    def find_node_pos_by_coordinates(self, x: int, y: int) -> int | None:
        for node_pos, node in enumerate(self.all_nodes):
            if int(node.x) == x and int(node.y) == y:
                return node_pos
        return None

    def vehicle_node_pos(self, vehicle: VehicleState) -> int | None:
        start_pos = self.start_node_pos_by_vehicle_id.get(vehicle.vehicle_id)
        if start_pos is not None:
            start_node = self.all_nodes[start_pos]
            if int(start_node.x) == vehicle.x and int(start_node.y) == vehicle.y:
                return start_pos
        return self.find_node_pos_by_coordinates(vehicle.x, vehicle.y)

    def travel_between_positions(self, from_pos: int, to_pos: int) -> int:
        return int(self.distance_matrix[from_pos][to_pos])

    def route_distance(self, vehicle_id: int, customer_ids: list[int]) -> float:
        if vehicle_id in self.start_node_pos_by_vehicle_id:
            current_pos = self.start_node_pos_by_vehicle_id[vehicle_id]
        else:
            current_pos = 0
        total = 0
        for customer_id in customer_ids:
            customer_pos = self.node_pos_for_customer(customer_id)
            if customer_pos is None:
                continue
            total += self.travel_between_positions(current_pos, customer_pos)
            current_pos = customer_pos
        total += self.travel_between_positions(current_pos, 0)
        return float(total)

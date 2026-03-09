"""Snapshot representation during simulation."""

from __future__ import annotations

from pydantic import BaseModel, Field

from dvrptw_bench.common.typing import Node, VRPTWInstance


class VehicleState(BaseModel):
    vehicle_id: int
    x: float
    y: float
    remaining_capacity: float
    elapsed_time: float
    planned_route: list[int] = Field(default_factory=list)
    traveled_distance: float = 0.0
    served_sequence: list[int] = Field(default_factory=list)
    current_service_customer_id: int | None = None
    remaining_service_time: float = 0.0


class SnapshotState(BaseModel):
    time: float
    remaining_customers: list[Node]
    active_customer_ids: set[int]
    served_customer_ids: set[int]
    vehicles: list[VehicleState]


def snapshot_to_instance(base: VRPTWInstance, snapshot: SnapshotState, instance_id: str | None = None) -> VRPTWInstance:
    return VRPTWInstance(
        instance_id=instance_id or f"{base.instance_id}@t{snapshot.time:.2f}",
        depot=base.depot,
        customers=snapshot.remaining_customers,
        vehicle_capacity=base.vehicle_capacity,
        vehicle_count=base.vehicle_count,
        distance_matrix=base.distance_matrix,
    )

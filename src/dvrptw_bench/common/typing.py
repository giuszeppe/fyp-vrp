"""Core typed models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Node(BaseModel):
    id: int
    x: float
    y: float
    demand: float = 0.0
    ready_time: float = 0.0
    due_time: float = 1e9
    service_time: float = 0.0


class VRPTWInstance(BaseModel):
    instance_id: str
    depot: Node
    customers: list[Node]
    vehicle_capacity: float
    vehicle_count: int
    distance_matrix: list[list[float]]

    @property
    def n_customers(self) -> int:
        return len(self.customers)

    @property
    def all_nodes(self) -> list[Node]:
        return [self.depot, *self.customers]


class Route(BaseModel):
    vehicle_id: int
    node_ids: list[int]


class Solution(BaseModel):
    strategy: str
    routes: list[Route]
    total_distance: float = 0.0
    feasible: bool = True
    violations: dict[str, float] = Field(default_factory=dict)
    solve_time_s: float = 0.0
    details: dict = Field(default_factory=dict)


class EventLog(BaseModel):
    event_idx: int
    event_time: float
    remaining_customers: int
    reopt_time_s: float
    objective_after: float


class FeasibilityReport(BaseModel):
    feasible: bool
    capacity_violation: float = 0.0
    time_violation: float = 0.0
    unserved_customers: list[int] = Field(default_factory=list)


class RecordTimes(BaseModel):
    total_s: float
    inference_s: float = 0.0
    local_search_s: float = 0.0


class ResultRecord(BaseModel):
    run_id: str
    timestamp: datetime
    git_hash: str | None
    strategy: str
    solver_details: str
    budget_s: float
    epsilon: float
    seed: int
    instance_id: str
    n_customers: int
    total_distance: float
    feasible: bool
    violations: dict[str, float]
    compute_times: RecordTimes
    dynamic_events: list[EventLog] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    mode: Literal["static", "dynamic"] = "static"

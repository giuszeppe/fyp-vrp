"""Dynamic simulator with stateful execution and snapshot re-optimization."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

from dvrptw_bench.common.typing import EventLog, Node, Route, Solution, VRPTWInstance
from dvrptw_bench.data.normalization import distance_matrix
from dvrptw_bench.dynamic.arrivals import build_dynamic_scenario
from dvrptw_bench.dynamic.snapshot import SnapshotState, VehicleState


class DynamicSimulator:
    """Event-driven simulator for customer revelation and state-aware re-optimization."""

    def __init__(self, base_instance: VRPTWInstance):
        self.base = base_instance

    @staticmethod
    def _euclid_xy(x1: float, y1: float, x2: float, y2: float) -> float:
        return math.hypot(x1 - x2, y1 - y2)

    @staticmethod
    def _matrix_travel(instance: VRPTWInstance, from_id: int, to_id: int) -> float:
        if 0 <= from_id < len(instance.distance_matrix) and 0 <= to_id < len(instance.distance_matrix):
            return float(instance.distance_matrix[from_id][to_id])
        return 0.0

    def _lookup(self, instance: VRPTWInstance) -> dict[int, Node]:
        return {n.id: n for n in instance.all_nodes}

    def _travel_to_customer(self, instance: VRPTWInstance, vehicle: VehicleState, customer: Node) -> float:
        """Prefer instance matrix when vehicle is at a known node, else Euclidean."""
        nodes = self._lookup(instance)
        from_id: int | None = None
        for nid, n in nodes.items():
            if abs(vehicle.x - n.x) <= 1e-9 and abs(vehicle.y - n.y) <= 1e-9:
                from_id = nid
                break
        if from_id is not None:
            m = self._matrix_travel(instance, from_id, customer.id)
            if m > 0.0:
                return m
        return self._euclid_xy(vehicle.x, vehicle.y, customer.x, customer.y)

    def _complete_ongoing_service(
        self,
        vehicle: VehicleState,
        target_time: float,
        nodes: dict[int, Node],
        served: set[int],
    ) -> bool:
        """Complete non-preemptive service if currently in service.

        Returns True when target_time is reached before service completion.
        """
        sid = vehicle.current_service_customer_id
        if sid is None:
            return False
        if sid not in nodes:
            vehicle.current_service_customer_id = None
            vehicle.remaining_service_time = 0.0
            return False

        if vehicle.elapsed_time + vehicle.remaining_service_time > target_time:
            vehicle.remaining_service_time -= max(0.0, target_time - vehicle.elapsed_time)
            vehicle.elapsed_time = target_time
            return True

        vehicle.elapsed_time += vehicle.remaining_service_time
        vehicle.remaining_service_time = 0.0
        vehicle.current_service_customer_id = None
        served.add(sid)
        vehicle.served_sequence.append(sid)
        if vehicle.planned_route and vehicle.planned_route[0] == sid:
            vehicle.planned_route.pop(0)
        return False

    def _advance_vehicle_to_time(
        self,
        vehicle: VehicleState,
        target_time: float,
        instance: VRPTWInstance,
        served: set[int],
        violations: dict[str, float],
    ) -> None:
        """Advance one vehicle from its current state up to target_time."""
        nodes = self._lookup(instance)

        # Non-preemptive service lock.
        if self._complete_ongoing_service(vehicle, target_time, nodes, served):
            return

        while vehicle.elapsed_time < target_time and vehicle.planned_route:
            nid = vehicle.planned_route[0]
            if nid in served:
                vehicle.planned_route.pop(0)
                continue
            if nid not in nodes:
                vehicle.planned_route.pop(0)
                continue

            c = nodes[nid]
            travel = self._travel_to_customer(instance, vehicle, c)

            # Partial travel if event occurs mid-edge.
            if vehicle.elapsed_time + travel > target_time and travel > 1e-12:
                dt = target_time - vehicle.elapsed_time
                frac = dt / travel
                vehicle.x += frac * (c.x - vehicle.x)
                vehicle.y += frac * (c.y - vehicle.y)
                vehicle.elapsed_time = target_time
                vehicle.traveled_distance += dt
                return

            # Reach customer.
            vehicle.elapsed_time += travel
            vehicle.traveled_distance += travel
            vehicle.x, vehicle.y = c.x, c.y
            arrival_time = vehicle.elapsed_time

            service_start = max(arrival_time, c.ready_time)
            completion_time = service_start + c.service_time

            # Time-window enforcement at execution time.
            if service_start > c.due_time or completion_time > c.due_time:
                lateness = max(0.0, service_start - c.due_time, completion_time - c.due_time)
                violations["late_count"] += 1.0
                violations["late_sum"] += lateness
                vehicle.planned_route.pop(0)
                continue

            # Wait if early.
            if service_start > target_time:
                vehicle.elapsed_time = target_time
                return
            vehicle.elapsed_time = service_start

            # Capacity enforcement at execution time.
            if c.demand > vehicle.remaining_capacity + 1e-9:
                violations["capacity"] += max(0.0, c.demand - vehicle.remaining_capacity)
                vehicle.planned_route.pop(0)
                continue

            # Service may cross event time: keep non-preemptive lock.
            if completion_time > target_time:
                vehicle.current_service_customer_id = c.id
                vehicle.remaining_service_time = completion_time - target_time
                vehicle.elapsed_time = target_time
                return

            # Service completed.
            vehicle.elapsed_time = completion_time
            vehicle.remaining_capacity = max(0.0, vehicle.remaining_capacity - c.demand)
            served.add(nid)
            vehicle.served_sequence.append(nid)
            vehicle.planned_route.pop(0)

        if vehicle.elapsed_time < target_time:
            vehicle.elapsed_time = target_time

    def _advance_all_to_time(
        self,
        vehicles: list[VehicleState],
        target_time: float,
        instance: VRPTWInstance,
        served: set[int],
        violations: dict[str, float],
    ) -> None:
        for v in vehicles:
            self._advance_vehicle_to_time(v, target_time, instance, served, violations)

    def _finish_execution(
        self,
        vehicles: list[VehicleState],
        instance: VRPTWInstance,
        served: set[int],
        violations: dict[str, float],
    ) -> None:
        """Execute all remaining planned routes and return to depot."""
        for v in vehicles:
            self._advance_vehicle_to_time(v, float("inf"), instance, served, violations)
            back = self._euclid_xy(v.x, v.y, instance.depot.x, instance.depot.y)
            v.traveled_distance += back
            v.elapsed_time += back
            v.x, v.y = instance.depot.x, instance.depot.y

    def _build_snapshot(
        self,
        now_t: float,
        active_ids: set[int],
        served_ids: set[int],
        vehicles: list[VehicleState],
        instance: VRPTWInstance,
    ) -> SnapshotState:
        locked = {v.current_service_customer_id for v in vehicles if v.current_service_customer_id is not None}
        remaining = [
            c
            for c in instance.customers
            if c.id in active_ids and c.id not in served_ids and c.id not in locked
        ]
        return SnapshotState(
            time=now_t,
            remaining_customers=remaining,
            active_customer_ids=set(active_ids),
            served_customer_ids=set(served_ids),
            vehicles=[v.model_copy(deep=True) for v in vehicles],
        )

    def _assign_customers_to_vehicles(self, snapshot: SnapshotState) -> dict[int, list[Node]]:
        """Greedy assignment by nearest current vehicle position with capacity preference."""
        assign: dict[int, list[Node]] = {v.vehicle_id: [] for v in snapshot.vehicles}
        by_due = sorted(snapshot.remaining_customers, key=lambda c: (c.due_time, c.ready_time))
        vehicles_by_id = {v.vehicle_id: v for v in snapshot.vehicles}

        for c in by_due:
            feasible = [v for v in snapshot.vehicles if v.remaining_capacity + 1e-9 >= c.demand]
            candidates = feasible if feasible else snapshot.vehicles
            best_v = min(candidates, key=lambda v: self._euclid_xy(v.x, v.y, c.x, c.y))
            assign[best_v.vehicle_id].append(c)
            vehicles_by_id[best_v.vehicle_id].remaining_capacity = max(
                0.0,
                vehicles_by_id[best_v.vehicle_id].remaining_capacity - c.demand,
            )
        return assign

    def _build_vehicle_subinstance(
        self,
        base_instance: VRPTWInstance,
        vehicle: VehicleState,
        customers: list[Node],
    ) -> tuple[VRPTWInstance, dict[int, int]]:
        """Build a 1-vehicle residual problem starting from current vehicle position."""
        depot = Node(
            id=0,
            x=vehicle.x,
            y=vehicle.y,
            demand=0.0,
            ready_time=vehicle.elapsed_time,
            due_time=base_instance.depot.due_time,
            service_time=0.0,
        )
        local_customers: list[Node] = []
        local_to_global: dict[int, int] = {}
        for i, c in enumerate(customers, start=1):
            local_customers.append(
                Node(
                    id=i,
                    x=c.x,
                    y=c.y,
                    demand=c.demand,
                    ready_time=c.ready_time,
                    due_time=c.due_time,
                    service_time=c.service_time,
                )
            )
            local_to_global[i] = c.id

        dmat = distance_matrix([depot, *local_customers])
        sub = VRPTWInstance(
            instance_id=f"{base_instance.instance_id}@t{vehicle.elapsed_time:.2f}_v{vehicle.vehicle_id}",
            depot=depot,
            customers=local_customers,
            vehicle_capacity=vehicle.remaining_capacity,
            vehicle_count=1,
            distance_matrix=dmat,
        )
        return sub, local_to_global

    def _reoptimize_snapshot(
        self,
        solver_fn,
        snapshot: SnapshotState,
        base_instance: VRPTWInstance,
        budget_s: float,
    ) -> Solution:
        """Solve the residual snapshot using per-vehicle subproblems."""
        assigned = self._assign_customers_to_vehicles(snapshot)
        routes: list[Route] = []
        strategy = "dynamic/subproblem"

        n_active = max(1, len(snapshot.remaining_customers))
        for v in snapshot.vehicles:
            customers = assigned.get(v.vehicle_id, [])
            if not customers:
                routes.append(Route(vehicle_id=v.vehicle_id, node_ids=[]))
                continue
            subinst, local_to_global = self._build_vehicle_subinstance(base_instance, v, customers)
            sub_budget = max(0.1, budget_s * (len(customers) / n_active))
            sub_sol = solver_fn(subinst, sub_budget)
            local_route = sub_sol.routes[0].node_ids if sub_sol.routes else []
            mapped = [local_to_global[n] for n in local_route if n in local_to_global]
            routes.append(Route(vehicle_id=v.vehicle_id, node_ids=mapped))
            strategy = sub_sol.strategy

        return Solution(strategy=strategy, routes=routes)

    def _apply_plan_to_vehicles(self, vehicles: list[VehicleState], solution: Solution) -> None:
        route_by_vehicle = {r.vehicle_id: r.node_ids[:] for r in solution.routes}
        for v in vehicles:
            # Preserve non-preemptive service lock customer at route head.
            locked = [v.current_service_customer_id] if v.current_service_customer_id is not None else []
            planned = route_by_vehicle.get(v.vehicle_id, [])
            planned = [nid for nid in planned if nid not in locked]
            v.planned_route = [*locked, *planned]

    def _predicted_remaining_distance(self, vehicles: list[VehicleState], instance: VRPTWInstance) -> float:
        nodes = self._lookup(instance)
        total = 0.0
        for v in vehicles:
            x, y = v.x, v.y
            for nid in v.planned_route:
                if nid not in nodes:
                    continue
                c = nodes[nid]
                total += self._euclid_xy(x, y, c.x, c.y)
                x, y = c.x, c.y
            total += self._euclid_xy(x, y, instance.depot.x, instance.depot.y)
        return total

    def _emit_snapshot(
        self,
        on_snapshot: Callable[[SnapshotState, Solution | None, list[VehicleState], set[int], dict[str, float] | None, int | None], None] | None,
        snapshot: SnapshotState,
        current_plan: Solution | None,
        vehicles: list[VehicleState],
        served: set[int],
        violations: dict[str, float] | None,
        event_idx: int | None,
        instance: VRPTWInstance,
        phase: str,
    ) -> None:
        if on_snapshot is None:
            return
        metrics = dict(violations or {})
        metrics["traveled_distance_total"] = float(sum(v.traveled_distance for v in vehicles))
        metrics["predicted_remaining_distance"] = float(self._predicted_remaining_distance(vehicles, instance))
        metrics["phase"] = phase
        on_snapshot(
            snapshot.model_copy(deep=True),
            current_plan.model_copy(deep=True) if current_plan is not None else None,
            [v.model_copy(deep=True) for v in vehicles],
            set(served),
            metrics,
            event_idx,
        )

    def run(
        self,
        solver_fn,
        epsilon: float,
        budget_s: float,
        seed: int,
        cutoff_ratio: float = 0.8,
        on_snapshot: Callable[
            [SnapshotState, Solution | None, list[VehicleState], set[int], dict[str, float] | None, int | None],
            None,
        ]
        | None = None,
    ):
        """Run stateful dynamic simulation and return executed-result solution."""
        scenario = build_dynamic_scenario(self.base, epsilon=epsilon, seed=seed, cutoff_ratio=cutoff_ratio)
        if not scenario.feasible:
            return None, [], scenario

        static_ids = [c.id for c in scenario.instance.customers if c.id not in scenario.dynamic_customer_ids]
        reveal_events = sorted(scenario.reveal_times.items(), key=lambda x: x[1])

        active_ids = set(static_ids)
        served: set[int] = set()
        violations: dict[str, float] = {"late_count": 0.0, "late_sum": 0.0, "capacity": 0.0}
        vehicles = [
            VehicleState(
                vehicle_id=v,
                x=self.base.depot.x,
                y=self.base.depot.y,
                remaining_capacity=self.base.vehicle_capacity,
                elapsed_time=0.0,
                planned_route=[],
                traveled_distance=0.0,
                served_sequence=[],
                current_service_customer_id=None,
                remaining_service_time=0.0,
            )
            for v in range(self.base.vehicle_count)
        ]

        snap0 = self._build_snapshot(0.0, active_ids, served, vehicles, scenario.instance)
        current_plan = self._reoptimize_snapshot(solver_fn, snap0, scenario.instance, budget_s)
        self._apply_plan_to_vehicles(vehicles, current_plan)
        self._emit_snapshot(
            on_snapshot,
            snap0,
            current_plan,
            vehicles,
            served,
            violations,
            None,
            scenario.instance,
            "initial_plan",
        )

        event_logs: list[EventLog] = []
        for i, (cid, evt_t) in enumerate(reveal_events):
            self._advance_all_to_time(vehicles, float(evt_t), scenario.instance, served, violations)
            active_ids.add(cid)

            snapshot = self._build_snapshot(float(evt_t), active_ids, served, vehicles, scenario.instance)
            self._emit_snapshot(
                on_snapshot,
                snapshot,
                current_plan,
                vehicles,
                served,
                violations,
                i,
                scenario.instance,
                "before_reopt",
            )
            t0 = time.perf_counter()
            current_plan = self._reoptimize_snapshot(solver_fn, snapshot, scenario.instance, budget_s)
            reopt_elapsed = time.perf_counter() - t0
            self._apply_plan_to_vehicles(vehicles, current_plan)
            self._emit_snapshot(
                on_snapshot,
                snapshot,
                current_plan,
                vehicles,
                served,
                violations,
                i,
                scenario.instance,
                "after_reopt",
            )

            objective_after = sum(v.traveled_distance for v in vehicles) + self._predicted_remaining_distance(
                vehicles, scenario.instance
            )
            event_logs.append(
                EventLog(
                    event_idx=i,
                    event_time=float(evt_t),
                    remaining_customers=len(snapshot.remaining_customers),
                    reopt_time_s=reopt_elapsed,
                    objective_after=objective_after,
                )
            )

        self._finish_execution(vehicles, scenario.instance, served, violations)
        final_snapshot = self._build_snapshot(
            max((v.elapsed_time for v in vehicles), default=0.0),
            active_ids,
            served,
            vehicles,
            scenario.instance,
        )
        self._emit_snapshot(
            on_snapshot,
            final_snapshot,
            current_plan,
            vehicles,
            served,
            violations,
            None,
            scenario.instance,
            "finished",
        )
        final_routes = [Route(vehicle_id=v.vehicle_id, node_ids=v.served_sequence) for v in vehicles if v.served_sequence]
        total_traveled = sum(v.traveled_distance for v in vehicles)
        all_customers = {c.id for c in scenario.instance.customers}
        unserved_count = len(all_customers - served)
        feasible = (
            unserved_count == 0
            and violations["late_count"] <= 1e-9
            and violations["capacity"] <= 1e-9
        )

        final_solution = Solution(
            strategy=f"{current_plan.strategy}:executed",
            routes=final_routes,
            total_distance=total_traveled,
            feasible=feasible,
            violations={
                "unserved": float(unserved_count),
                "late_count": violations["late_count"],
                "late_sum": violations["late_sum"],
                "capacity": violations["capacity"],
            },
            solve_time_s=0.0,
            details={"served_count": len(served)},
        )
        return final_solution, event_logs, scenario

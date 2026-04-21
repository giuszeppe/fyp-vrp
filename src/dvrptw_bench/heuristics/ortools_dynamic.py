"""OR-Tools VRPTW solver adapter."""

from __future__ import annotations

import time

from dvrptw_bench.common.typing import Route, Solution
from dvrptw_bench.dynamic.dynamic_instance import DynamicInstance
from dvrptw_bench.metrics.objective import total_distance


class ORToolsDVRPTWSolver:
    name = "ortools-dynamic"


    def __init__(self, soft_time_windows: bool = True):
        self.soft_time_windows = soft_time_windows

    def _warm_start_routes_for_assignment(
        self,
        instance: DynamicInstance,
        warm_start: Solution,
    ) -> list[list[int]]:
        """Convert solution customer IDs to OR-Tools manager node positions."""
        routes = [[] for _ in range(instance.vehicle_count)]
        seen_customer_ids: set[int] = set()

        for route in warm_start.routes:
            if route.vehicle_id < 0 or route.vehicle_id >= instance.vehicle_count:
                continue

            node_positions: list[int] = []
            for customer_id in route.node_ids:
                if customer_id in seen_customer_ids:
                    continue

                node_pos = instance.node_pos_for_customer(customer_id)
                if node_pos is None:
                    continue

                seen_customer_ids.add(customer_id)
                node_positions.append(node_pos)

            routes[route.vehicle_id] = node_positions

        return routes

    def solve(self, instance: DynamicInstance, time_limit_s: float, warm_start: Solution | None = None) -> Solution | None:
        t0 = time.perf_counter()
        try:
            from ortools.constraint_solver import pywrapcp, routing_enums_pb2

            starts = instance.starts
            ends = instance.ends
            manager = pywrapcp.RoutingIndexManager(len(instance.all_nodes), instance.vehicle_count, starts, ends)
            routing = pywrapcp.RoutingModel(manager)
            initial_assignment = None

            def demand_callback(from_idx: int) -> int:
                node_pos = manager.IndexToNode(from_idx)
                return int(instance.all_nodes[node_pos].demand)

            demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)

            routing.AddDimensionWithVehicleCapacity(
                demand_idx,
                0,  # no slack
                [int(v.remaining_capacity) for v in instance.vehicles],  # vehicle capacities
                True,  # start load at zero
                "Capacity",
            )

            def distance_callback(from_idx: int, to_idx: int) -> int:
                from_pos = manager.IndexToNode(from_idx)
                to_pos = manager.IndexToNode(to_idx)
                return instance.travel_between_positions(from_pos, to_pos)

            def time_callback(from_idx: int, to_idx: int) -> int:
                from_pos = manager.IndexToNode(from_idx)
                to_pos = manager.IndexToNode(to_idx)
                travel = instance.travel_between_positions(from_pos, to_pos)
                service = int(instance.all_nodes[from_pos].service_time)
                return service + travel

            transit_idx = routing.RegisterTransitCallback(distance_callback)
            time_idx = routing.RegisterTransitCallback(time_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

            routing.AddDimension(time_idx, 10_000, 10_000_000, False, "Time")
            time_dim = routing.GetDimensionOrDie("Time")

            late_penalty = 50
            for node_pos, node in enumerate(instance.all_nodes):
                idx = manager.NodeToIndex(node_pos)
                if idx < 0:
                    continue
                if self.soft_time_windows:
                    # Hard lower bound: cannot start before ready time
                    time_dim.CumulVar(idx).SetMin(int(node.ready_time))

                    # Soft upper bound: lateness allowed, but penalized
                    time_dim.SetCumulVarSoftUpperBound(
                        idx,
                        int(node.due_time),
                        late_penalty,
                    )
                else:
                    time_dim.CumulVar(idx).SetRange(
                        int(node.ready_time),
                        int(node.due_time),
                    )

            for v in range(instance.vehicle_count):
                start_idx = routing.Start(v)
                vehicle = instance.vehicles[v]
                start_ready = int(vehicle.elapsed_time + vehicle.remaining_service_time)
                start_due = int(instance.depot.due_time)
                time_dim.CumulVar(start_idx).SetRange(
                    start_ready,
                    start_due,
                )
                end_idx = routing.End(v)
                end_node = manager.IndexToNode(end_idx)
                time_dim.CumulVar(end_idx).SetRange(
                    int(instance.all_nodes[end_node].ready_time),
                    int(instance.all_nodes[end_node].due_time),
                )

            params = pywrapcp.DefaultRoutingSearchParameters()
            params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            params.time_limit.seconds = max(1, int(time_limit_s))
            # params.log_search = True

            if warm_start is not None:
                initial_routes = self._warm_start_routes_for_assignment(instance, warm_start)
                if any(initial_routes):
                    initial_assignment = routing.ReadAssignmentFromRoutes(
                        initial_routes,
                        True,
                    )

            if initial_assignment is not None:
                assignment = routing.SolveFromAssignmentWithParameters(initial_assignment, params)
            else:
                assignment = routing.SolveWithParameters(params)
            routes: list[Route] = []
            if assignment:
                for v in range(instance.vehicle_count):
                    idx = routing.Start(v)
                    node_ids: list[int] = []
                    while not routing.IsEnd(idx):
                        node_pos = manager.IndexToNode(idx)
                        customer_id = instance.customer_id_for_node_pos(node_pos)
                        if customer_id is not None:
                            node_ids.append(customer_id)
                        idx = assignment.Value(routing.NextVar(idx))
                    routes.append(Route(vehicle_id=v, node_ids=node_ids))
                sol = Solution(strategy=self.name, routes=routes)
            else:
                return None
        except Exception as e:
            print("ORTools exception:", repr(e))
            return None

        sol.total_distance = total_distance(instance, sol)
        sol.solve_time_s = time.perf_counter() - t0
        return sol

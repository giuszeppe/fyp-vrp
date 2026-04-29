"""OR-Tools VRPTW solver adapter."""

from __future__ import annotations

import time

from dvrptw_bench.common.typing import Route, Solution, VRPTWInstance
from dvrptw_bench.heuristics.constructive_pmca import PMCAVRPTWSolver
from dvrptw_bench.heuristics.interfaces import HeuristicSolver
from dvrptw_bench.metrics.objective import total_distance


class ORToolsVRPTWSolver(HeuristicSolver):
    name = "ortools"

    def solve(self, instance: VRPTWInstance, time_limit_s: float, warm_start: Solution | None = None) -> Solution:
        t0 = time.perf_counter()
        try:
            from ortools.constraint_solver import pywrapcp, routing_enums_pb2

            manager = pywrapcp.RoutingIndexManager(len(instance.all_nodes), instance.vehicle_count, 0)
            routing = pywrapcp.RoutingModel(manager)
            # routing.SetFixedCostOfAllVehicles(10_000)   # tune this

            def demand_callback(from_idx: int) -> int:
                node = manager.IndexToNode(from_idx)
                return 0 if node == 0 else int(instance.all_nodes[node].demand)

            demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)

            routing.AddDimensionWithVehicleCapacity(
                demand_idx,
                0,  # no slack
                [int(instance.vehicle_capacity)] * instance.vehicle_count,
                True,  # start load at zero
                "Capacity",
            )

            def distance_callback(from_idx: int, to_idx: int) -> int:
                f = manager.IndexToNode(from_idx)
                t = manager.IndexToNode(to_idx)
                return int(instance.distance_matrix[f][t])

            def time_callback(from_idx: int, to_idx: int) -> int:
                f = manager.IndexToNode(from_idx)
                t = manager.IndexToNode(to_idx)
                travel = int(instance.distance_matrix[f][t])
                service = int(instance.all_nodes[f].service_time)
                return service + travel
            transit_idx = routing.RegisterTransitCallback(distance_callback)
            time_idx = routing.RegisterTransitCallback(time_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

            routing.AddDimension(time_idx, 10_000, 10_000_000, False, "Time")
            time_dim = routing.GetDimensionOrDie("Time")

            for node_pos, node in enumerate(instance.all_nodes):
                idx = manager.NodeToIndex(node_pos)
                time_dim.CumulVar(idx).SetRange(int(node.ready_time), int(node.due_time))

            params = pywrapcp.DefaultRoutingSearchParameters()
            params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            params.time_limit.seconds = max(1, int(time_limit_s))

            assignment = routing.SolveWithParameters(params)
            routes: list[Route] = []
            if assignment:
                for v in range(instance.vehicle_count):
                    idx = routing.Start(v)
                    node_ids: list[int] = []
                    while not routing.IsEnd(idx):
                        node = manager.IndexToNode(idx)
                        if node != 0:
                            node_ids.append(instance.all_nodes[node].id)
                        idx = assignment.Value(routing.NextVar(idx))
                    routes.append(Route(vehicle_id=v, node_ids=node_ids))
                sol = Solution(strategy=self.name, routes=routes)
            else:
                sol = PMCAVRPTWSolver().solve(instance, time_limit_s, warm_start=warm_start)
                sol.strategy = self.name + ":fallback_pmca"
        except Exception:
            sol = PMCAVRPTWSolver().solve(instance, time_limit_s, warm_start=warm_start)
            sol.strategy = self.name + ":fallback_pmca"

        sol.total_distance = total_distance(instance, sol)
        sol.solve_time_s = time.perf_counter() - t0
        return sol

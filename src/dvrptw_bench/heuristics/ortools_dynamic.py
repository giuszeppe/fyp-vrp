"""OR-Tools VRPTW solver adapter."""

from __future__ import annotations

import time

from dvrptw_bench.common.typing import Route, Solution, VRPTWInstance
from dvrptw_bench.dynamic.dynamic_instance import DynamicInstance
from dvrptw_bench.heuristics.constructive_pmca import PMCAVRPTWSolver
from dvrptw_bench.heuristics.interfaces import HeuristicSolver
from dvrptw_bench.metrics.objective import total_distance


class ORToolsDVRPTWSolver:
    name = "ortools-dynamic"

    def solve(self, instance: DynamicInstance, time_limit_s: float, warm_start: Solution | None = None) -> Solution| None:
        t0 = time.perf_counter()
        try:
            from ortools.constraint_solver import pywrapcp, routing_enums_pb2

            starts = [n.id for n in instance.starts]
            ends = [n.id for n in instance.ends]
            manager = pywrapcp.RoutingIndexManager(len(instance.all_nodes), instance.vehicle_count, starts, ends)
            routing = pywrapcp.RoutingModel(manager)
            # routing.SetFixedCostOfAllVehicles(10_000)   # tune this

            def demand_callback(from_idx: int) -> int:
                node = manager.IndexToNode(from_idx)
                return 0 if node == 0 else int(instance.all_nodes[node].demand)

            demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)

            routing.AddDimensionWithVehicleCapacity(
                demand_idx,
                0,  # no slack
                [int(v.remaining_capacity) for v in instance.vehicles],  # vehicle capacities
                True,  # start load at zero
                "Capacity",
            )
            skip_penalty = 0  # tune this

            for node_pos, node in enumerate(instance.all_nodes):
                if node.id not in instance.ignored_customers:  # depot
                    continue
                routing.AddDisjunction([manager.NodeToIndex(node_pos)], skip_penalty)

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

            late_penalty = 10 # tune this
            for node_pos, node in enumerate(instance.all_nodes):
                idx = manager.NodeToIndex(node_pos)
                time_dim.CumulVar(idx).SetRange(
                    int(node.ready_time),
                    int(node.due_time),
                )
                # Hard lower bound: cannot start before ready time
                # time_dim.CumulVar(idx).SetMin(int(node.ready_time))

                # Soft upper bound: lateness allowed, but penalized
                # time_dim.SetCumulVarSoftUpperBound(
                    # idx,
                    # int(node.due_time),
                    # late_penalty,
                # )
            
            for v in range(instance.vehicle_count):
                start_idx = routing.Start(v)
                start_node = manager.IndexToNode(start_idx)
                time_dim.CumulVar(start_idx).SetRange(
                    int(instance.all_nodes[start_node].ready_time),
                    int(instance.all_nodes[start_node].due_time),
                )

            params = pywrapcp.DefaultRoutingSearchParameters()
            params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            params.time_limit.seconds = max(1, int(time_limit_s))
            # params.log_search = True

            assignment = routing.SolveWithParameters(params)
            routes: list[Route] = []
            if assignment:
                print("Solution found by OR-Tools!")
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
                return None
                # sol = PMCAVRPTWSolver().solve(instance, time_limit_s, warm_start=warm_start)
                # sol.strategy = self.name + ":fallback_pmca"
        except Exception as e:
            print("ORTools exception:", repr(e))
            # sol = PMCAVRPTWSolver().solve(instance, time_limit_s, warm_start=warm_start)
            # sol.strategy = self.name + ":fallback_pmca"
            return None

        sol.total_distance = total_distance(instance, sol)
        sol.solve_time_s = time.perf_counter() - t0
        return sol

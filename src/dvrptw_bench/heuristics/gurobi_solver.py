"""Optional Gurobi VRPTW scaffold."""

from __future__ import annotations

import time

from dvrptw_bench.common.errors import SolverUnavailableError
from dvrptw_bench.common.typing import Route, Solution, VRPTWInstance
from dvrptw_bench.heuristics.constructive_pmca import PMCAVRPTWSolver
from dvrptw_bench.heuristics.interfaces import HeuristicSolver
from dvrptw_bench.metrics.objective import total_distance


class GurobiVRPTWSolver(HeuristicSolver):
    name = "gurobi"

    def solve(self, instance: VRPTWInstance, time_limit_s: float, warm_start: Solution | None = None) -> Solution:
        t0 = time.perf_counter()
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except Exception as exc:
            raise SolverUnavailableError("gurobipy not installed. Use: pip install -e '.[gurobi]'") from exc

        try:
            model = gp.Model("vrptw_scaffold")
            model.Params.TimeLimit = float(time_limit_s)
            model.Params.OutputFlag = 0

            n = len(instance.all_nodes)
            K = instance.vehicle_count
            x = model.addVars(n, n, K, vtype=GRB.BINARY, name="x")
            t = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0.0, name="t")
            load = model.addVars(n, K, vtype=GRB.CONTINUOUS, lb=0.0, name="load")

            # Reduced single-commodity flow skeleton (scalable limitations documented).
            model.setObjective(
                gp.quicksum(instance.distance_matrix[i][j] * x[i, j, k] for i in range(n) for j in range(n) for k in range(K) if i != j),
                GRB.MINIMIZE,
            )

            for c in range(1, n):
                model.addConstr(gp.quicksum(x[i, c, k] for i in range(n) for k in range(K) if i != c) == 1)
                model.addConstr(gp.quicksum(x[c, j, k] for j in range(n) for k in range(K) if j != c) == 1)

            for k in range(K):
                model.addConstr(gp.quicksum(x[0, j, k] for j in range(1, n)) <= 1)
                model.addConstr(gp.quicksum(x[i, 0, k] for i in range(1, n)) <= 1)

            M = 1e6
            for i, node_i in enumerate(instance.all_nodes):
                model.addConstr(t[i] >= node_i.ready_time)
                model.addConstr(t[i] <= node_i.due_time)
                for j, node_j in enumerate(instance.all_nodes):
                    if i == j:
                        continue
                    for k in range(K):
                        model.addConstr(t[j] >= t[i] + node_i.service_time + instance.distance_matrix[i][j] - M * (1 - x[i, j, k]))
                        model.addConstr(load[j, k] >= load[i, k] + node_j.demand - M * (1 - x[i, j, k]))
                        model.addConstr(load[j, k] <= instance.vehicle_capacity)

            model.optimize()

            if model.SolCount <= 0:
                sol = PMCAVRPTWSolver().solve(instance, time_limit_s, warm_start=warm_start)
                sol.strategy = self.name + ":fallback_pmca"
                return sol

            routes = [Route(vehicle_id=k, node_ids=[]) for k in range(K)]
            for k in range(K):
                current = 0
                visited = set()
                while True:
                    nxt = None
                    for j in range(n):
                        if j == current:
                            continue
                        if x[current, j, k].X > 0.5:
                            nxt = j
                            break
                    if nxt is None or nxt == 0 or nxt in visited:
                        break
                    visited.add(nxt)
                    routes[k].node_ids.append(instance.all_nodes[nxt].id)
                    current = nxt

            sol = Solution(strategy=self.name, routes=routes)
            sol.total_distance = total_distance(instance, sol)
            sol.solve_time_s = time.perf_counter() - t0
            return sol
        except Exception:
            sol = PMCAVRPTWSolver().solve(instance, time_limit_s, warm_start=warm_start)
            sol.strategy = self.name + ":license_or_model_fallback_pmca"
            sol.solve_time_s = time.perf_counter() - t0
            return sol

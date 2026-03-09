"""From-scratch genetic algorithm baseline for VRPTW."""

from __future__ import annotations

import random
import time

from dvrptw_bench.common.typing import Solution, VRPTWInstance
from dvrptw_bench.dynamic.feasibility import verify_solution
from dvrptw_bench.metrics.objective import total_distance
from dvrptw_bench.rl.decoding import split_permutation


class GAPolicy:
    name = "ga"
    """
    Simple genetic algorithm baseline for VRPTW. Operates on permutations of customer IDs, decoded via the split procedure.
    Uses a permutation of customers as the "chromosome" representation, and applies ordered crossover and mutation to evolve the population. 
    The fitness function is the total distance plus a large penalty for any constraint violations. 
    The best solution found within the time limit is returned.
    A permutation of the customers is used because it is a route representation that can be easily manipulated
    """
    def train(self, instance: VRPTWInstance, time_limit_s: float = 20.0, pop_size: int = 40) -> dict:
        t0 = time.perf_counter()
        ids = [c.id for c in instance.customers] # IDs of customers to be permuted
        pop = [random.sample(ids, len(ids)) for _ in range(pop_size)] # Random initial population of permutations of customers
        best = None
        curve = []


        while time.perf_counter() - t0 < time_limit_s:
            scored = []
            for chrom in pop:
                sol = split_permutation(instance, chrom, self.name)
                rep = verify_solution(instance, sol) # Check feasibility and get details of any violations
                dist = total_distance(instance, sol)
                route_imbalance = max(len(r.node_ids) for r in sol.routes) - min(len(r.node_ids) for r in sol.routes) if sol.routes else 0
                vehicleIdlesPenalty = 1000.0 * (instance.vehicle_count - len(sol.routes))
                print(f"Distance: {dist}, Capacity Violation: {rep.capacity_violation}, Time Violation: {rep.time_violation}, Unserved Customers: {len(rep.unserved_customers)}, Route Imbalance: {route_imbalance}, Idle Vehicles Penalty: {vehicleIdlesPenalty}")
                penalty = 1000.0 * (rep.capacity_violation + rep.time_violation + len(rep.unserved_customers) +route_imbalance + vehicleIdlesPenalty)
                scored.append((dist + penalty, chrom, dist))
            scored.sort(key=lambda x: x[0])
            if best is None or scored[0][0] < best[0]:
                best = scored[0]
            curve.append((time.perf_counter() - t0, best[2]))

            elites = [s[1] for s in scored[: max(2, pop_size // 10)]]
            new_pop = elites[:]
            while len(new_pop) < pop_size:
                p1 = random.choice(scored[: pop_size // 2])[1]
                p2 = random.choice(scored[: pop_size // 2])[1]
                child = self._ordered_crossover(p1, p2)
                child = self._mutate(child)
                new_pop.append(child)
            pop = new_pop

        self._last = {"best_perm": best[1] if best else ids, "curve": curve}
        return self._last

    def infer_solution(self, instance: VRPTWInstance) -> Solution:
        if not hasattr(self, "_last"):
            self.train(instance, time_limit_s=2.0)
        sol = split_permutation(instance, self._last["best_perm"], self.name)
        sol.total_distance = total_distance(instance, sol)
        sol.details["convergence"] = self._last.get("curve", [])
        return sol

    def infer(self, snapshot_state):
        # Snapshot-level route proposal; greedy by nearest x+y proxy.
        sorted_ids = [n.id for n in sorted(snapshot_state.remaining_customers, key=lambda n: n.x + n.y)]
        from dvrptw_bench.common.typing import Route, Solution

        return Solution(strategy=self.name, routes=[Route(vehicle_id=0, node_ids=sorted_ids)])

    def _ordered_crossover(self, a: list[int], b: list[int]) -> list[int]:
        i, j = sorted(random.sample(range(len(a)), 2))
        seg = a[i:j]
        rest = [x for x in b if x not in seg]
        return rest[:i] + seg + rest[i:]

    def _mutate(self, arr: list[int], p: float = 0.2) -> list[int]:
        out = arr[:]
        if random.random() < p:
            i, j = sorted(random.sample(range(len(out)), 2))
            out[i:j] = reversed(out[i:j])
        if random.random() < p:
            i, j = random.sample(range(len(out)), 2)
            out[i], out[j] = out[j], out[i]
        return out

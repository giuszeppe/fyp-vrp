"""Optional LKH-3 external binary wrapper scaffold."""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

from dvrptw_bench.common.errors import SolverUnavailableError
from dvrptw_bench.common.typing import Solution, VRPTWInstance
from dvrptw_bench.heuristics.constructive_pmca import PMCAVRPTWSolver
from dvrptw_bench.heuristics.interfaces import HeuristicSolver


class LKH3VRPTWSolver(HeuristicSolver):
    name = "lkh3"

    def __init__(self, binary_path: str | None = None):
        self.binary_path = binary_path

    def solve(self, instance: VRPTWInstance, time_limit_s: float, warm_start: Solution | None = None) -> Solution:
        t0 = time.perf_counter()
        if not self.binary_path:
            raise SolverUnavailableError("LKH-3 binary missing. Provide --lkh3-binary /path/to/LKH.")

        bin_path = Path(self.binary_path)
        if not bin_path.exists():
            raise SolverUnavailableError(f"LKH-3 binary not found at {bin_path}. See https://webhotel4.ruc.dk/~keld/research/LKH-3/")

        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmpdir = Path(tmp)
                tsp = tmpdir / "problem.vrp"
                par = tmpdir / "params.par"
                out = tmpdir / "tour.out"

                self._write_vrplib(instance, tsp)
                par.write_text(
                    "\n".join([
                        f"PROBLEM_FILE = {tsp}",
                        f"OUTPUT_TOUR_FILE = {out}",
                        "RUNS = 1",
                        f"TIME_LIMIT = {max(1, int(time_limit_s))}",
                    ]),
                    encoding="utf-8",
                )
                subprocess.run([str(bin_path), str(par)], check=True, capture_output=True, text=True, timeout=max(1, int(time_limit_s) + 2))

                # Parsing omitted for portability; fallback keeps pipeline runnable.
                sol = PMCAVRPTWSolver().solve(instance, time_limit_s, warm_start=warm_start)
                sol.strategy = self.name + ":scaffold_pmca"
                sol.solve_time_s = time.perf_counter() - t0
                sol.details["tour_file"] = str(out)
                return sol
        except Exception:
            sol = PMCAVRPTWSolver().solve(instance, time_limit_s, warm_start=warm_start)
            sol.strategy = self.name + ":fallback_pmca"
            sol.solve_time_s = time.perf_counter() - t0
            return sol

    def _write_vrplib(self, instance: VRPTWInstance, path: Path) -> None:
        lines = [
            f"NAME : {instance.instance_id}",
            "TYPE : CVRPTW",
            f"DIMENSION : {len(instance.all_nodes)}",
            f"CAPACITY : {int(instance.vehicle_capacity)}",
            "EDGE_WEIGHT_TYPE : EUC_2D",
            "NODE_COORD_SECTION",
        ]
        for n in instance.all_nodes:
            lines.append(f"{n.id + 1} {n.x:.3f} {n.y:.3f}")
        lines.append("DEMAND_SECTION")
        for n in instance.all_nodes:
            lines.append(f"{n.id + 1} {int(n.demand)}")
        lines.append("DEPOT_SECTION")
        lines.append("1")
        lines.append("-1")
        lines.append("EOF")
        path.write_text("\n".join(lines), encoding="utf-8")

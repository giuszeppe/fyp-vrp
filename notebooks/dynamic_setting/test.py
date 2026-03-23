from dvrptw_bench.dynamic.simulator import DynamicSimulator
from dvrptw_bench.heuristics.ortools_dynamic import ORToolsDVRPTWSolver

from pathlib import Path

from pyarrow import dataset

from dvrptw_bench.data.instance_filters import find_rc_instances
from dvrptw_bench.data.solomon_parser import parse_solomon

from dvrptw_bench.dynamic.arrivals import build_dynamic_scenario

dataset_path = Path("../../dataset/solomon_rc100")
instances = [parse_solomon(instance, max_customers=100) for instance in find_rc_instances(dataset_path)]
dod = .5
dynamic_instances = [build_dynamic_scenario(instance, epsilon=dod, seed=42) for instance in instances]

sim = DynamicSimulator(instances[0])
res = sim.run(lambda instance, time_limit_s, warm_start = None: ORToolsDVRPTWSolver().solve(instance, time_limit_s, warm_start), budget_s=1, epsilon=dod, seed=15, cutoff_ratio=.1)
from pathlib import Path

import pytest

from dvrptw_bench.common.typing import Node, Route, Solution, VRPTWInstance
from dvrptw_bench.dynamic.feasibility import verify_solution
from dvrptw_bench.metrics.objective import total_distance


def _tiny_instance() -> VRPTWInstance:
    depot = Node(id=0, x=0, y=0, demand=0, ready_time=0, due_time=100, service_time=0)
    c1 = Node(id=1, x=1, y=0, demand=1, ready_time=0, due_time=100, service_time=0)
    c2 = Node(id=2, x=2, y=0, demand=1, ready_time=0, due_time=100, service_time=0)
    dmat = [
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 1.0],
        [2.0, 1.0, 0.0],
    ]
    return VRPTWInstance(instance_id="tiny", depot=depot, customers=[c1, c2], vehicle_capacity=10, vehicle_count=2, distance_matrix=dmat)


def test_total_distance_and_feasibility():
    inst = _tiny_instance()
    sol = Solution(strategy="t", routes=[Route(vehicle_id=0, node_ids=[1, 2])])
    dist = total_distance(inst, sol)
    rep = verify_solution(inst, sol)
    assert abs(dist - 4.0) < 1e-9
    assert rep.feasible


def test_solomon_parser_smoke(tmp_path: Path):
    content = """RC_TEST
VEHICLE
NUMBER     CAPACITY
  25         200
CUSTOMER
CUST NO.  XCOORD.    YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE TIME
  0      40         50          0        0          1236       0
  1      45         68          10       10         200        10
  2      45         70          20       20         220        10
"""
    f = tmp_path / "RC_TEST.txt"
    f.write_text(content)
    from dvrptw_bench.data.solomon_parser import parse_solomon

    inst = parse_solomon(f)
    assert inst.instance_id == "RC_TEST.txt"
    assert inst.n_customers == 2


def test_solomon_parser_max_customers(tmp_path: Path):
    content = """RC_TEST
VEHICLE
NUMBER     CAPACITY
  25         200
CUSTOMER
CUST NO.  XCOORD.    YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE TIME
  0      40         50          0        0          1236       0
  1      45         68          10       10         200        10
  2      45         70          20       20         220        10
  3      45         72          30       30         240        10
"""
    f = tmp_path / "RC_TEST.txt"
    f.write_text(content)
    from dvrptw_bench.data.solomon_parser import parse_solomon

    inst_limited = parse_solomon(f, max_customers=2)
    assert inst_limited.n_customers == 2
    assert [c.id for c in inst_limited.customers] == [1, 2]

    inst_zero = parse_solomon(f, max_customers=0)
    assert inst_zero.n_customers == 0

    with pytest.raises(ValueError, match="max_customers"):
        parse_solomon(f, max_customers=-1)

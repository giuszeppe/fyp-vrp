from dvrptw_bench.common.typing import Node, Route, Solution, VRPTWInstance
from dvrptw_bench.data.normalization import distance_matrix
from dvrptw_bench.dynamic.simulator import DynamicSimulator
from dvrptw_bench.heuristics.constructive_pmca import PMCAVRPTWSolver
from dvrptw_bench.viz.inspector import build_dynamic_frames, build_static_frames


def _tiny_instance() -> VRPTWInstance:
    depot = Node(id=0, x=0, y=0, demand=0, ready_time=0, due_time=200, service_time=0)
    customers = [
        Node(id=1, x=1, y=0, demand=1, ready_time=0, due_time=200, service_time=1),
        Node(id=2, x=2, y=0, demand=1, ready_time=0, due_time=200, service_time=1),
        Node(id=3, x=3, y=0, demand=1, ready_time=0, due_time=200, service_time=1),
        Node(id=4, x=4, y=0, demand=1, ready_time=0, due_time=200, service_time=1),
    ]
    dmat = distance_matrix([depot, *customers])
    return VRPTWInstance(
        instance_id="tiny_inspector",
        depot=depot,
        customers=customers,
        vehicle_capacity=3,
        vehicle_count=2,
        distance_matrix=dmat,
    )


def test_static_frames_properties():
    inst = _tiny_instance()
    sol = Solution(strategy="static-test", routes=[Route(vehicle_id=0, node_ids=[1, 2]), Route(vehicle_id=1, node_ids=[3, 4])])
    frames = build_static_frames(inst, sol)

    assert len(frames) > 0
    times = [f.time for f in frames]
    assert all(t2 >= t1 for t1, t2 in zip(times[:-1], times[1:], strict=False))

    prev_served = set()
    prev_dist = {}
    for f in frames:
        assert prev_served.issubset(f.served_ids)
        prev_served = set(f.served_ids)
        for v in f.vehicles:
            old = prev_dist.get(v.vehicle_id, 0.0)
            assert v.traveled_distance >= old - 1e-9
            prev_dist[v.vehicle_id] = v.traveled_distance


def test_dynamic_frames_properties():
    inst = _tiny_instance()
    sim = DynamicSimulator(inst)
    solver = PMCAVRPTWSolver()

    frames, final_solution, _event_logs = build_dynamic_frames(
        inst,
        sim,
        lambda snap_inst, budget: solver.solve(snap_inst, budget),
        epsilon=0.5,
        budget_s=1.0,
        seed=1,
    )

    assert len(frames) > 0
    assert final_solution is not None
    times = [f.time for f in frames]
    assert all(t2 >= t1 for t1, t2 in zip(times[:-1], times[1:], strict=False))

    prev_served = set()
    prev_dist = {}
    for f in frames:
        assert prev_served.issubset(f.served_ids)
        prev_served = set(f.served_ids)
        for v in f.vehicles:
            old = prev_dist.get(v.vehicle_id, 0.0)
            assert v.traveled_distance >= old - 1e-9
            prev_dist[v.vehicle_id] = v.traveled_distance

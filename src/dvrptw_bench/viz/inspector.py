"""Interactive route inspector for static and dynamic VRPTW runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import math

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

from dvrptw_bench.common.typing import EventLog, Node, Route, Solution, VRPTWInstance
from dvrptw_bench.dynamic.simulator import DynamicSimulator
from dvrptw_bench.dynamic.snapshot import SnapshotState, VehicleState


@dataclass
class InspectorFrame:
    """Serializable frame used by the inspector renderer."""

    mode: str
    time: float
    step_type: str
    active_ids: set[int]
    served_ids: set[int]
    vehicles: list[VehicleState]
    current_plan: Solution
    metrics: dict[str, Any] = field(default_factory=dict)
    event_idx: int | None = None
    event_time: float | None = None
    revealed_customer_id: int | None = None
    reopt_time_s: float | None = None
    objective_after: float | None = None
    late_customer_ids: set[int] = field(default_factory=set)
    rejected_customer_ids: set[int] = field(default_factory=set)


def _node_map(instance: VRPTWInstance) -> dict[int, Node]:
    return {n.id: n for n in instance.all_nodes}


def _travel(instance: VRPTWInstance, from_id: int, to_id: int) -> float:
    if 0 <= from_id < len(instance.distance_matrix) and 0 <= to_id < len(instance.distance_matrix):
        return float(instance.distance_matrix[from_id][to_id])
    nodes = _node_map(instance)
    a = nodes[from_id]
    b = nodes[to_id]
    return math.hypot(a.x - b.x, a.y - b.y)


def _predicted_remaining_distance(instance: VRPTWInstance, vehicles: list[VehicleState]) -> float:
    nodes = _node_map(instance)
    dep = instance.depot
    total = 0.0
    for v in vehicles:
        x, y = v.x, v.y
        for nid in v.planned_route:
            if nid not in nodes:
                continue
            n = nodes[nid]
            total += math.hypot(x - n.x, y - n.y)
            x, y = n.x, n.y
        total += math.hypot(x - dep.x, y - dep.y)
    return total


def _solution_from_vehicles(strategy: str, vehicles: list[VehicleState]) -> Solution:
    routes = [Route(vehicle_id=v.vehicle_id, node_ids=v.planned_route[:]) for v in vehicles]
    return Solution(strategy=strategy, routes=routes)


def build_static_frames(instance: VRPTWInstance, solution: Solution) -> list[InspectorFrame]:
    """Build static inspection frames at edge/service granularity."""
    nodes = _node_map(instance)
    route_by_vehicle = {r.vehicle_id: r.node_ids[:] for r in solution.routes}
    vehicle_ids = sorted(route_by_vehicle.keys())
    vehicles: dict[int, VehicleState] = {
        vid: VehicleState(
            vehicle_id=vid,
            x=instance.depot.x,
            y=instance.depot.y,
            remaining_capacity=instance.vehicle_capacity,
            elapsed_time=instance.depot.ready_time,
            planned_route=route_by_vehicle[vid][:],
            traveled_distance=0.0,
            served_sequence=[],
            current_service_customer_id=None,
            remaining_service_time=0.0,
        )
        for vid in vehicle_ids
    }

    active = {c.id for c in instance.customers}
    served: set[int] = set()

    frames: list[InspectorFrame] = [
        InspectorFrame(
            mode="static",
            time=0.0,
            step_type="initial_plan",
            active_ids=set(active),
            served_ids=set(),
            vehicles=[v.model_copy(deep=True) for v in vehicles.values()],
            current_plan=_solution_from_vehicles(solution.strategy, list(vehicles.values())),
            metrics={
                "traveled_distance_total": 0.0,
                "predicted_remaining_distance": _predicted_remaining_distance(instance, list(vehicles.values())),
            },
        )
    ]

    events: list[tuple[float, int, int, str, int]] = []
    # (time, priority, vehicle_id, event_type, customer_id)
    for vid in vehicle_ids:
        t = instance.depot.ready_time
        prev = instance.depot.id
        for nid in route_by_vehicle[vid]:
            if nid not in nodes:
                continue
            c = nodes[nid]
            travel = _travel(instance, prev, nid)
            arrival = t + travel
            start = max(arrival, c.ready_time)
            end = start + c.service_time
            events.append((arrival, 0, vid, "edge_end", nid))
            events.append((start, 1, vid, "service_start", nid))
            events.append((end, 2, vid, "service_end", nid))
            t = end
            prev = nid

    events.sort(key=lambda x: (x[0], x[1], x[2]))

    for t, _priority, vid, etype, nid in events:
        v = vehicles[vid]
        n = nodes[nid]
        if etype == "edge_end":
            dist = math.hypot(v.x - n.x, v.y - n.y)
            v.traveled_distance += dist
            v.x, v.y = n.x, n.y
            v.elapsed_time = t
        elif etype == "service_start":
            v.current_service_customer_id = nid
            v.elapsed_time = t
        elif etype == "service_end":
            v.current_service_customer_id = None
            v.elapsed_time = t
            if v.planned_route and v.planned_route[0] == nid:
                v.planned_route.pop(0)
            if nid in nodes:
                v.remaining_capacity = max(0.0, v.remaining_capacity - nodes[nid].demand)
            v.served_sequence.append(nid)
            served.add(nid)

        vlist = [vv.model_copy(deep=True) for vv in vehicles.values()]
        frames.append(
            InspectorFrame(
                mode="static",
                time=t,
                step_type=etype,
                active_ids=set(active),
                served_ids=set(served),
                vehicles=vlist,
                current_plan=_solution_from_vehicles(solution.strategy, vlist),
                metrics={
                    "traveled_distance_total": float(sum(vv.traveled_distance for vv in vlist)),
                    "predicted_remaining_distance": _predicted_remaining_distance(instance, vlist),
                },
            )
        )

    return frames


def build_dynamic_frames(
    instance: VRPTWInstance,
    simulator: DynamicSimulator,
    solver_fn,
    *,
    epsilon: float,
    budget_s: float,
    seed: int,
    cutoff_ratio: float = 0.8,
    end_time_closeness: float|None = None,
) -> tuple[list[InspectorFrame], Solution | None, list[EventLog]]:
    """Run dynamic simulation and collect snapshot frames through callback hook."""
    frames: list[InspectorFrame] = []

    def _on_snapshot(
        snapshot: SnapshotState,
        current_plan: Solution | None,
        vehicles: list[VehicleState],
        served: set[int],
        metrics: dict[str, Any] | None,
        event_idx: int | None,
    ) -> None:
        metric_map = dict(metrics or {})
        plan = current_plan if current_plan is not None else Solution(strategy="snapshot", routes=[])
        frames.append(
            InspectorFrame(
                mode="dynamic",
                time=float(snapshot.time),
                step_type=str(metric_map.get("phase", "snapshot")),
                active_ids=set(snapshot.active_customer_ids),
                served_ids=set(served),
                vehicles=[v.model_copy(deep=True) for v in vehicles],
                current_plan=plan.model_copy(deep=True),
                metrics=metric_map,
                event_idx=event_idx,
                event_time=float(snapshot.time),
                late_customer_ids=set(metric_map.get("late_customer_ids", [])),
                rejected_customer_ids=set(metric_map.get("rejected_customer_ids", [])),
            )
        )

    final_solution, event_logs, scenario = simulator.run(
        solver_fn,
        epsilon=epsilon,
        budget_s=budget_s,
        seed=seed,
        cutoff_ratio=cutoff_ratio,
        end_time_closeness=end_time_closeness,
        on_snapshot=_on_snapshot,
    )

    if final_solution is None:
        return frames, None, []

    reveal_map = {i: cid for i, (cid, _evt) in enumerate(sorted(scenario.reveal_times.items(), key=lambda x: x[1]))}
    log_map = {log.event_idx: log for log in event_logs}

    for frame in frames:
        if frame.event_idx is not None:
            frame.revealed_customer_id = reveal_map.get(frame.event_idx)
            if frame.step_type == "after_reopt" and frame.event_idx in log_map:
                elog = log_map[frame.event_idx]
                frame.reopt_time_s = elog.reopt_time_s
                frame.objective_after = elog.objective_after

    return frames, final_solution, event_logs


class _MatplotlibInspector:
    """Matplotlib interactive inspector with keyboard and button controls."""

    def __init__(self, instance: VRPTWInstance, frames: list[InspectorFrame], title: str):
        self.instance = instance
        self.frames = frames
        self.title = title
        self.idx = 0
        self.playing = False
        self.show_labels = True
        self.event_to_step: dict[int, int] = {}
        for i, f in enumerate(frames):
            if f.event_idx is not None and f.event_idx not in self.event_to_step:
                self.event_to_step[f.event_idx] = i

        self.fig = plt.figure(figsize=(13, 8))
        self.ax = self.fig.add_axes([0.05, 0.14, 0.62, 0.8])
        self.side = self.fig.add_axes([0.7, 0.14, 0.28, 0.8])
        self.side.axis("off")

        self._btn_prev = Button(self.fig.add_axes([0.05, 0.03, 0.08, 0.05]), "Prev")
        self._btn_next = Button(self.fig.add_axes([0.14, 0.03, 0.08, 0.05]), "Next")
        self._btn_reset = Button(self.fig.add_axes([0.23, 0.03, 0.08, 0.05]), "Reset")
        self._btn_play = Button(self.fig.add_axes([0.32, 0.03, 0.08, 0.05]), "Play")
        self._btn_pause = Button(self.fig.add_axes([0.41, 0.03, 0.08, 0.05]), "Pause")
        self._jump_box = TextBox(self.fig.add_axes([0.52, 0.03, 0.12, 0.05]), "Event #", initial="0")
        self._btn_jump = Button(self.fig.add_axes([0.65, 0.03, 0.08, 0.05]), "Jump")

        self._btn_prev.on_clicked(lambda _e: self.prev_step())
        self._btn_next.on_clicked(lambda _e: self.next_step())
        self._btn_reset.on_clicked(lambda _e: self.reset())
        self._btn_play.on_clicked(lambda _e: self.play())
        self._btn_pause.on_clicked(lambda _e: self.pause())
        self._btn_jump.on_clicked(lambda _e: self.jump_event())

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.timer = self.fig.canvas.new_timer(interval=450)
        self.timer.add_callback(self._on_timer)

        self.render()

    def _on_timer(self) -> None:
        if self.playing:
            if self.idx < len(self.frames) - 1:
                self.idx += 1
                self.render()
            else:
                self.playing = False
                self.timer.stop()

    def play(self) -> None:
        self.playing = True
        self.timer.start()

    def pause(self) -> None:
        self.playing = False
        self.timer.stop()

    def reset(self) -> None:
        self.idx = 0
        self.render()

    def next_step(self) -> None:
        self.idx = min(len(self.frames) - 1, self.idx + 1)
        self.render()

    def prev_step(self) -> None:
        self.idx = max(0, self.idx - 1)
        self.render()

    def jump_event(self) -> None:
        raw = self._jump_box.text.strip()
        if not raw:
            return
        try:
            evt = int(raw)
        except ValueError:
            return
        if evt in self.event_to_step:
            self.idx = self.event_to_step[evt]
            self.render()

    def on_key(self, event: Any) -> None:
        if event.key == "right":
            self.next_step()
        elif event.key == "left":
            self.prev_step()
        elif event.key == "home":
            self.idx = 0
            self.render()
        elif event.key == "end":
            self.idx = len(self.frames) - 1
            self.render()
        elif event.key == " ":
            if self.playing:
                self.pause()
            else:
                self.play()
        elif event.key and event.key.lower() == "l":
            self.show_labels = not self.show_labels
            self.render()

    def _draw_paths(self, frame: InspectorFrame, nodes: dict[int, Node]) -> None:
        colors = plt.cm.tab20.colors
        for i, v in enumerate(sorted(frame.vehicles, key=lambda vv: vv.vehicle_id)):
            remaining_route = v.planned_route[:]
            if not remaining_route:
                continue
            path_xy = [(v.x, v.y)]
            for nid in remaining_route:
                if nid in nodes:
                    n = nodes[nid]
                    path_xy.append((n.x, n.y))
            path_xy.append((self.instance.depot.x, self.instance.depot.y))
            if len(path_xy) >= 2:
                xs, ys = zip(*path_xy, strict=False)
                self.ax.plot(xs, ys, color=colors[i % len(colors)], linestyle="--", linewidth=1.4, alpha=0.8)

        for i, v in enumerate(sorted(frame.vehicles, key=lambda vv: vv.vehicle_id)):
            seq = [nid for nid in v.served_sequence if nid in nodes]
            if not seq and (v.x, v.y) == (self.instance.depot.x, self.instance.depot.y):
                continue
            pts = [(self.instance.depot.x, self.instance.depot.y)] + [(nodes[n].x, nodes[n].y) for n in seq]
            current_xy = (v.x, v.y)
            if pts[-1] != current_xy:
                pts.append(current_xy)
            xs, ys = zip(*pts, strict=False)
            self.ax.plot(xs, ys, color=colors[i % len(colors)], linewidth=2.2, alpha=0.9)

    def render(self) -> None:
        frame = self.frames[self.idx]
        nodes = _node_map(self.instance)

        self.ax.clear()
        self.side.clear()
        self.side.axis("off")

        all_ids = {c.id for c in self.instance.customers}
        active = set(frame.active_ids)
        served = set(frame.served_ids)
        late_ids = set(frame.late_customer_ids)
        rejected_ids = set(frame.rejected_customer_ids)
        unrevealed = all_ids - active - served - rejected_ids
        active_unserved = active - served - late_ids
        served_on_time = served - late_ids

        self.ax.scatter([self.instance.depot.x], [self.instance.depot.y], c="red", marker="s", s=120, label="Depot")
        if unrevealed:
            xs = [nodes[i].x for i in sorted(unrevealed)]
            ys = [nodes[i].y for i in sorted(unrevealed)]
            self.ax.scatter(xs, ys, c="lightgray", s=22, alpha=0.45, label="Not revealed")
        if active_unserved:
            xs = [nodes[i].x for i in sorted(active_unserved)]
            ys = [nodes[i].y for i in sorted(active_unserved)]
            self.ax.scatter(xs, ys, c="tab:blue", s=28, alpha=0.85, label="Active unserved")
        if served_on_time:
            xs = [nodes[i].x for i in sorted(served_on_time)]
            ys = [nodes[i].y for i in sorted(served_on_time)]
            self.ax.scatter(xs, ys, c="tab:green", marker="x", s=30, alpha=0.9, label="Served")
        if late_ids:
            xs = [nodes[i].x for i in sorted(late_ids) if i in nodes]
            ys = [nodes[i].y for i in sorted(late_ids) if i in nodes]
            if xs and ys:
                self.ax.scatter(xs, ys, c="tab:orange", marker="X", s=70, alpha=0.95, label="Late")
        if rejected_ids:
            xs = [nodes[i].x for i in sorted(rejected_ids) if i in nodes]
            ys = [nodes[i].y for i in sorted(rejected_ids) if i in nodes]
            if xs and ys:
                self.ax.scatter(xs, ys, facecolors="none", edgecolors="crimson", marker="o", s=110, linewidths=1.8, label="Rejected")

        self._draw_paths(frame, nodes)

        for v in frame.vehicles:
            self.ax.scatter([v.x], [v.y], c="black", s=55, marker="o")
            self.ax.text(v.x, v.y, f"V{v.vehicle_id}", fontsize=8)
            if v.current_service_customer_id is not None and v.current_service_customer_id in nodes:
                c = nodes[v.current_service_customer_id]
                self.ax.scatter([c.x], [c.y], facecolors="none", edgecolors="orange", s=220, linewidths=2.0)

        if self.show_labels:
            for cid in sorted(all_ids):
                n = nodes[cid]
                self.ax.text(n.x + 0.15, n.y + 0.15, str(cid), fontsize=7, alpha=0.85)

        traveled = frame.metrics.get("traveled_distance_total", sum(v.traveled_distance for v in frame.vehicles))
        rem = frame.metrics.get("predicted_remaining_distance", _predicted_remaining_distance(self.instance, frame.vehicles))
        top = f"{self.title} | {frame.mode.upper()} | Step {self.idx + 1}/{len(self.frames)} | t={frame.time:.2f}"
        self.ax.set_title(top)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.grid(alpha=0.2)
        self.ax.legend(loc="best", fontsize=8)

        lines = [
            f"step_type: {frame.step_type}",
            f"time: {frame.time:.2f}",
            f"event_idx: {frame.event_idx}",
            f"revealed_customer_id: {frame.revealed_customer_id}",
            f"reopt_time_s: {frame.reopt_time_s}",
            f"objective_after: {frame.objective_after}",
            f"traveled_distance: {traveled:.3f}",
            f"pred_remaining_distance: {rem:.3f}",
            f"late_customers: {sorted(frame.late_customer_ids)}",
            f"rejected_customers: {sorted(frame.rejected_customer_ids)}",
            "",
            "Vehicles:",
        ]
        for v in sorted(frame.vehicles, key=lambda x: x.vehicle_id):
            head = v.planned_route[0] if v.planned_route else None
            lines.append(
                f"V{v.vehicle_id}: cap={v.remaining_capacity:.1f}, t={v.elapsed_time:.2f}, "
                f"head={head}, lock={v.current_service_customer_id}, served={v.served_sequence}"
            )

        self.side.text(0.0, 1.0, "\n".join(lines), va="top", fontsize=9, family="monospace")
        self.fig.canvas.draw_idle()



def inspect_static(instance: VRPTWInstance, solution: Solution, *, title: str | None = None) -> None:
    """Open interactive static inspector."""
    frames = build_static_frames(instance, solution)
    if not frames:
        return
    t = title or f"{instance.instance_id}"
    _MatplotlibInspector(instance, frames, t)
    plt.show()



def inspect_dynamic(
    instance: VRPTWInstance,
    simulator: DynamicSimulator,
    solver_fn,
    *,
    epsilon: float,
    budget_s: float,
    seed: int,
    cutoff_ratio: float = 0.8,
    title: str | None = None,
    end_time_closeness: float|None = None,
) -> None:
    """Open interactive dynamic inspector built from simulator snapshot frames."""
    frames, final_solution, _events = build_dynamic_frames(
        instance,
        simulator,
        solver_fn,
        epsilon=epsilon,
        budget_s=budget_s,
        seed=seed,
        cutoff_ratio=cutoff_ratio,
        end_time_closeness=end_time_closeness,
    )
    if not frames:
        return
    t = title or f"{instance.instance_id}"
    if final_solution is not None:
        t = f"{t} | executed={final_solution.total_distance:.2f}"
    _MatplotlibInspector(instance, frames, t)
    plt.show()

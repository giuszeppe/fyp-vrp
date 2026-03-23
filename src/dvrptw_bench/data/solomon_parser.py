"""Parser for Solomon VRPTW instance files."""

from __future__ import annotations

from pathlib import Path

from dvrptw_bench.common.errors import DatasetError
from dvrptw_bench.common.typing import Node, VRPTWInstance
from dvrptw_bench.data.normalization import distance_matrix


def _extract_vehicle_info(lines: list[str]) -> tuple[int, float]:
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("VEHICLE"):
            for j in range(i, min(len(lines), i + 8)):
                parts = lines[j].split()
                if len(parts) >= 2 and parts[0].isdigit():
                    return int(parts[0]), float(parts[1])
    return 25, 200.0


def _find_customer_section(lines: list[str]) -> int:
    for i, line in enumerate(lines):
        if "CUSTOMER" in line.upper():
            return i + 1
    raise DatasetError("Could not find CUSTOMER section in Solomon file.")


def parse_solomon(path: Path, distance_scale: float = 1.0, max_customers: int | None = None) -> VRPTWInstance:
    raw_lines = [ln.rstrip() for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines()]
    lines = [ln for ln in raw_lines if ln.strip()]
    if not lines:
        raise DatasetError(f"Empty file: {path}")
    if max_customers is not None and max_customers < 0:
        raise ValueError("max_customers must be >= 0")

    vehicle_count, capacity = _extract_vehicle_info(lines)
    start = _find_customer_section(lines)

    rows: list[list[float]] = []
    for ln in lines[start:]:
        parts = ln.split()
        if len(parts) < 7:
            continue
        try:
            rows.append([float(v) for v in parts[:7]])
        except ValueError:
            continue

    if not rows:
        raise DatasetError(f"No customer rows parsed from {path}")

    depot_row, *customer_rows = rows
    depot = Node(
        id=int(depot_row[0]),
        x=depot_row[1] * distance_scale,
        y=depot_row[2] * distance_scale,
        demand=depot_row[3],
        ready_time=depot_row[4] * distance_scale,
        due_time=depot_row[5] * distance_scale,
        service_time=depot_row[6] * distance_scale,
    )
    customers = [
        Node(
            id=int(r[0]),
            x=r[1] * distance_scale,
            y=r[2] * distance_scale,
            demand=r[3],
            ready_time=r[4] * distance_scale,
            due_time=r[5] * distance_scale,
            service_time=r[6] * distance_scale,
        )
        for r in customer_rows
    ]
    if max_customers is not None:
        customers = customers[:max_customers]

    dmat = distance_matrix([depot, *customers])
    return VRPTWInstance(
        instance_id=path.name,
        depot=depot,
        customers=customers,
        vehicle_capacity=capacity,
        vehicle_count=vehicle_count,
        distance_matrix=dmat,
    )

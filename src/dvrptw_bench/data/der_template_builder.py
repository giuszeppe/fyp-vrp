from __future__ import annotations

from dvrptw_bench.common.typing import VRPTWInstance
from dvrptw_bench.data.der_solomon_generator import SolomonSeriesTemplate, template_from_dict

VALID_FAMILIES = {"C1", "C2", "R1", "R2", "RC1", "RC2"}


def template_dict_from_vrptw_instance(instance: VRPTWInstance, family: str) -> dict:
    if family not in VALID_FAMILIES:
        raise ValueError(f"Unsupported family '{family}'. Expected one of {sorted(VALID_FAMILIES)}.")

    return {
        "family": family,
        "vehicle_capacity": float(instance.vehicle_capacity),
        "depot": {
            "x": float(instance.depot.x),
            "y": float(instance.depot.y),
            "ready_time": float(instance.depot.ready_time),
            "due_time": float(instance.depot.due_time),
            "service_time": float(instance.depot.service_time),
        },
        "customers": [
            {
                "id": int(c.id),
                "x": float(c.x),
                "y": float(c.y),
                "demand": float(c.demand),
                "ready_time": float(c.ready_time),
                "due_time": float(c.due_time),
                "service_time": float(c.service_time),
            }
            for c in instance.customers
        ],
    }


def template_from_vrptw_instance(instance: VRPTWInstance, family: str) -> SolomonSeriesTemplate:
    return template_from_dict(template_dict_from_vrptw_instance(instance, family))


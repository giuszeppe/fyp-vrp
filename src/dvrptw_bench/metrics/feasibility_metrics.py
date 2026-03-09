"""Feasibility metrics."""

from dvrptw_bench.common.typing import FeasibilityReport


def fulfillment_ratio(report: FeasibilityReport, n_customers: int) -> float:
    if n_customers <= 0:
        return 1.0
    served = n_customers - len(report.unserved_customers)
    return served / n_customers

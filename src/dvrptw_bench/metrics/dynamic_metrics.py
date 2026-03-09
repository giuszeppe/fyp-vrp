"""Dynamic metrics."""

from dvrptw_bench.common.typing import EventLog


def responsiveness_ok(event_logs: list[EventLog], budget_s: float) -> float:
    if not event_logs:
        return 1.0
    ok = sum(1 for e in event_logs if e.reopt_time_s <= budget_s)
    return ok / len(event_logs)

"""Shared labels, ordering, and defaults for notebook outputs."""

from __future__ import annotations

from pathlib import Path

DEFAULT_RESULTS_ROOT = (Path(__file__).resolve().parents[1] / "data" / "results").resolve()

CANONICAL_MODALITIES = ("oracle", "heuristic", "ai", "hybrid", "static")
FAMILY_ORDER = ["C", "R", "RC"]
INSTANCE_GROUP_ORDER = ["C1", "C2", "R1", "R2", "RC1", "RC2"]

MODEL_DISPLAY_NAMES = {
    "oracle": "Oracle",
    "ortools": "OR-Tools",
    "general_50": "RouteFinder General 50",
    "routefinder_solomon_generated_50": "RouteFinder Solomon 50",
    "routefinder_solomon_generated_75": "RouteFinder Solomon 75",
    "routefinder_with_lateness_50": "RouteFinder Late 50",
    "routefinder_with_lateness_75": "RouteFinder Late 75",
}

MODALITY_DISPLAY_NAMES = {
    "oracle": "Oracle",
    "heuristic": "Heuristic",
    "ai": "AI",
    "hybrid": "Hybrid",
    "static": "Static",
}

MODALITY_COLORS = {
    "oracle": "#1b4332",
    "heuristic": "#005f73",
    "ai": "#bb3e03",
    "hybrid": "#6a4c93",
    "static": "#3a5a40",
}

METRIC_LABELS = {
    "total_distance": "Total Distance",
    "num_vehicles": "Vehicles Used",
    "average_lateness": "Average Lateness",
    "customers_rejected": "Rejected Customers",
    "rejection_rate": "Rejection Rate",
    "service_level": "Service Level",
    "computational_time": "Runtime (s)",
    "total_cost": "Total Cost",
    "oracle_gap": "Gap to Oracle (%)",
}


def display_name_for_model(model_name: str) -> str:
    """Return a stable human-readable model label."""

    return MODEL_DISPLAY_NAMES.get(model_name, model_name.replace("_", " ").title())

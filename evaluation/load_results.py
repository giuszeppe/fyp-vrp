"""Load JSON benchmark results into analysis-friendly pandas DataFrames."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .style import CANONICAL_MODALITIES, DEFAULT_RESULTS_ROOT, display_name_for_model

LEGACY_MODALITY_MAP = {
    "heuristic_fisso": "heuristic",
}

_INSTANCE_PATTERN = re.compile(r"^(RC|R|C)(\d)(\d{2})$")
_NUMERIC_COLUMNS = [
    "evaluation_size",
    "degree_of_dynamicity",
    "cutoff_time",
    "num_samples",
    "num_starts",
    "num_augment",
    "total_distance",
    "num_vehicles",
    "computational_time",
    "customers_served",
    "customers_rejected",
    "rejection_rate",
    "average_lateness",
    "service_level",
    "total_cost",
    "oracle_gap",
]
_BOOL_COLUMNS = ["allow_rejection", "feasible", "has_oracle_gap", "is_complete_run", "select_best"]
_REQUIRED_COLUMNS = [
    "instance_id",
    "model_name",
    "evaluation_size",
    "degree_of_dynamicity",
    "cutoff_time",
]


def discover_result_dirs(results_root: Path | str = DEFAULT_RESULTS_ROOT) -> dict[str, Path]:
    """Resolve canonical modality names to the folders that should back them."""

    root = Path(results_root)
    discovered: dict[str, Path] = {}
    for modality in CANONICAL_MODALITIES:
        candidate = root / modality
        if candidate.exists() and any(candidate.glob("*.json")):
            discovered[modality] = candidate

    legacy_heuristic = root / "heuristic_fisso"
    if "heuristic" not in discovered and legacy_heuristic.exists() and any(legacy_heuristic.glob("*.json")):
        discovered["heuristic"] = legacy_heuristic

    return discovered


def _load_json_records(result_dir: Path, canonical_modality: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(result_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["source_file"] = str(path.resolve())
        payload["source_dir"] = result_dir.name
        payload["modality_group"] = canonical_modality
        payload["modality_folder"] = result_dir.name
        rows.append(payload)
    return rows


def _instance_metadata(instance_id: str) -> dict[str, Any]:
    stem = Path(instance_id).stem.upper()
    match = _INSTANCE_PATTERN.match(stem)
    if not match:
        return {
            "instance_stem": stem,
            "instance_family": None,
            "instance_type": None,
            "instance_group": None,
        }

    family, type_digit, _ = match.groups()
    return {
        "instance_stem": stem,
        "instance_family": family,
        "instance_type": int(type_digit),
        "instance_group": f"{family}{type_digit}",
    }


def _variant_metadata(model_name: str, modality_group: str) -> dict[str, Any]:
    if modality_group == "oracle":
        model_variant = "oracle"
    elif modality_group == "heuristic":
        model_variant = "heuristic"
    elif modality_group == "hybrid":
        model_variant = "hybrid"
    elif "solomon" in model_name:
        model_variant = "solomon_generated"
    elif "lateness" in model_name:
        model_variant = "with_lateness"
    elif "general" in model_name:
        model_variant = "general"
    else:
        model_variant = "other"

    return {
        "model_variant": model_variant,
        "model_display_name": display_name_for_model(model_name),
    }


def _normalize_modalities(df: pd.DataFrame) -> pd.DataFrame:
    if "model_type" in df.columns:
        df["model_type"] = df["model_type"].fillna(df["modality_group"])
    else:
        df["model_type"] = df["modality_group"]

    df["modality_group"] = df["modality_group"].replace(LEGACY_MODALITY_MAP).fillna(df["model_type"])
    return df


def _apply_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    meta = df["instance_id"].map(_instance_metadata).apply(pd.Series)
    df = pd.concat([df, meta], axis=1)

    variant_meta = df.apply(
        lambda row: _variant_metadata(str(row["model_name"]), str(row["modality_group"])),
        axis=1,
    ).apply(pd.Series)
    df = pd.concat([df, variant_meta], axis=1)

    df["seed"] = df["source_file"].map(_seed_from_path)
    df["has_oracle_gap"] = df["oracle_gap"].notna()
    df["is_complete_run"] = df[_REQUIRED_COLUMNS].notna().all(axis=1)
    df["oracle_subset"] = df["evaluation_size"].eq(50)
    df["coverage_note"] = ""
    df.loc[df["modality_group"].eq("oracle"), "coverage_note"] = "Oracle baseline is static and size-50 only."
    df.loc[
        df["modality_group"].ne("oracle") & df["oracle_gap"].isna() & df["evaluation_size"].eq(75),
        "coverage_note",
    ] = "No oracle baseline available for size 75."
    return df


def _seed_from_path(source_file: str) -> int | None:
    match = re.search(r"__seed(\d+)__", source_file)
    if match:
        return int(match.group(1))
    return None


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for column in _NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in _BOOL_COLUMNS:
        if column in df.columns:
            df[column] = df[column].fillna(False).astype(bool)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def _ordered_columns(df: pd.DataFrame) -> pd.DataFrame:
    front = [
        "modality_group",
        "model_name",
        "model_display_name",
        "model_variant",
        "instance_id",
        "instance_family",
        "instance_type",
        "instance_group",
        "evaluation_size",
        "seed",
        "degree_of_dynamicity",
        "cutoff_time",
        "decode_type",
        "num_samples",
        "num_starts",
        "num_augment",
        "select_best",
    ]
    ordered = [column for column in front if column in df.columns]
    remaining = [column for column in df.columns if column not in ordered]
    return df[ordered + remaining]


def load_results(
    results_root: Path | str = DEFAULT_RESULTS_ROOT,
    *,
    include_partial: bool = True,
    modalities: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load benchmark result JSONs into a normalized DataFrame."""

    discovered = discover_result_dirs(results_root)
    if modalities is not None:
        requested = set(modalities)
        discovered = {name: path for name, path in discovered.items() if name in requested}

    rows: list[dict[str, Any]] = []
    for modality, result_dir in discovered.items():
        rows.extend(_load_json_records(result_dir, modality))

    if not rows:
        return pd.DataFrame(columns=_REQUIRED_COLUMNS + ["modality_group", "source_file"])

    df = pd.DataFrame.from_records(rows)
    df = _normalize_modalities(df)
    df = _coerce_dtypes(df)
    df = _apply_derived_columns(df)
    if not include_partial:
        df = df[df["is_complete_run"]].copy()
    df = _ordered_columns(df)
    return df.sort_values(
        ["modality_group", "model_name", "instance_id", "evaluation_size", "degree_of_dynamicity", "cutoff_time", "seed"],
        kind="stable",
    ).reset_index(drop=True)

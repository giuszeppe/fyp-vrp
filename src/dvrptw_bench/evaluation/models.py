"""Pydantic models for the evaluation workflow."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    model_id: str
    enabled: bool = True
    checkpoint_candidates: list[str] = Field(default_factory=list)
    num_customers: int
    variant: str


class Manifest(BaseModel):
    version: int = 1
    dataset_root: str
    seeds: list[int]
    evaluation_sizes: list[int] = Field(default_factory=lambda: [50])
    degrees_of_dynamicity: list[float] = Field(default_factory=lambda: [0.0])
    cutoff_times: list[float] = Field(default_factory=lambda: [0.5])
    instances: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ScenarioArtifact(BaseModel):
    scenario_id: str
    instance_name: str
    evaluation_size: int
    seed: int
    degree_of_dynamicity: float
    cutoff_time: float
    source_path: str
    dynamic_customer_ids: list[int] = Field(default_factory=list)
    reveal_times: dict[int, float] = Field(default_factory=dict)
    feasible: bool
    dropped_reason: str | None = None
    instance: dict[str, Any]


class WorkUnit(BaseModel):
    work_id: str
    modality: Literal["oracle", "heuristic", "ai", "hybrid"]
    instance_name: str
    evaluation_size: int
    seed: int | None = None
    degree_of_dynamicity: float | None = None
    cutoff_time: float | None = None
    model_name: str
    scenario_id: str | None = None
    result_path: str


class WorkState(BaseModel):
    work_id: str
    status: Literal["pending", "in_progress", "completed", "failed"] = "pending"
    attempts: int = 0
    started_at: str | None = None
    completed_at: str | None = None
    output_path: str | None = None
    error_message: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class Ledger(BaseModel):
    modality: Literal["oracle", "heuristic", "ai", "hybrid"]
    updated_at: str | None = None
    items: dict[str, WorkState] = Field(default_factory=dict)


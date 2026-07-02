from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from app.schemas.common import APIModel


class TrainCandidateRequest(APIModel):
    dataset_id: str
    model_name: str = "weather-bracket-model"
    requested_version: str | None = None
    random_state: int = 42
    max_iter: int = Field(default=300, ge=50, le=2000)
    learning_rate: float = Field(default=0.05, gt=0, le=1)
    max_leaf_nodes: int = Field(default=31, ge=4, le=255)
    min_samples_leaf: int = Field(default=20, ge=2, le=1000)
    l2_regularization: float = Field(default=0.1, ge=0, le=100)
    calibration_method: str = Field(default="isotonic", pattern="^(isotonic|identity)$")


class TrainingJobRead(APIModel):
    id: str
    status: str
    dataset_id: str
    model_name: str
    requested_version: str | None
    configuration: dict[str, Any]
    result_model_version_id: str | None
    progress: float
    logs: list[dict[str, Any]]
    error_message: str | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime


class ModelComparisonRequest(APIModel):
    champion_version: str
    challenger_version: str
    dataset_id: str
    minimum_brier_improvement: float = 0.0
    maximum_mae_degradation: float = 0.0


class ModelComparisonResponse(APIModel):
    champion_version: str
    challenger_version: str
    dataset_id: str
    champion_metrics: dict[str, Any]
    challenger_metrics: dict[str, Any]
    metric_deltas: dict[str, float]
    recommendation: str
    reasons: list[str]

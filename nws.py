from __future__ import annotations

from datetime import date, datetime
from typing import Any

from app.schemas.common import APIModel


class ModelVersionRead(APIModel):
    id: str
    model_name: str
    version: str
    status: str
    artifact_uri: str
    artifact_sha256: str
    feature_schema_version: str
    training_dataset_id: str | None
    training_start_date: date | None
    training_end_date: date | None
    metrics: dict[str, Any]
    hyperparameters: dict[str, Any]
    metadata: dict[str, Any]
    promoted_at: datetime | None
    retired_at: datetime | None
    created_at: datetime


class ModelHealthResponse(APIModel):
    service_status: str
    model_status: str
    model_version: str | None
    model_id: str | None
    feature_schema_version: str | None
    artifact_verified: bool
    supported_stations: list[str]
    supported_market_types: list[str]
    metrics: dict[str, Any]
    warnings: list[str]
    timestamp: datetime


class ModelPromotionRequest(APIModel):
    model_version: str
    reason: str


class ModelRollbackRequest(APIModel):
    model_version: str
    reason: str

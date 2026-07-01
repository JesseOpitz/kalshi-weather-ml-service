from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class APIModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class HealthResponse(APIModel):
    status: str
    service: str
    environment: str
    timestamp: datetime
    version: str


class ErrorDetail(APIModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


class Pagination(APIModel):
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)

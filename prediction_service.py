from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from app.schemas.common import APIModel


class OutcomeCreate(APIModel):
    market_id: str
    station_code: str
    official_value: float
    official_source: str
    settled_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class OutcomeRead(APIModel):
    id: str
    market_id: str
    station_code: str
    official_value: float
    official_source: str
    settled_at: datetime
    metadata: dict[str, Any]
    created_at: datetime
    evaluations_created: int = 0

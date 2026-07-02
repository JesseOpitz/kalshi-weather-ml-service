from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from app.schemas.common import APIModel


class StationCreate(APIModel):
    station_code: str = Field(min_length=2, max_length=32)
    station_name: str = Field(min_length=2, max_length=255)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    elevation_m: float | None = None
    timezone: str = "UTC"
    official_source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    active: bool = True


class StationRead(StationCreate):
    id: str
    created_at: datetime
    updated_at: datetime

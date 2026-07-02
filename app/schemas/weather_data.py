from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import Field

from app.schemas.common import APIModel


class DatasetRead(APIModel):
    id: str
    name: str
    source_type: str
    station_code: str | None
    file_uri: str
    checksum_sha256: str
    row_count: int
    schema_version: str
    date_start: date | None
    date_end: date | None
    quality_report: dict[str, Any]
    created_at: datetime


class OpenMeteoDatasetRequest(APIModel):
    name: str
    station_code: str
    start_date: date
    end_date: date
    market_type: str = "daily_high_temperature"
    assumed_lead_hours: int = Field(default=24, ge=1, le=240)

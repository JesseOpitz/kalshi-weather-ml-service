from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from app.schemas.common import APIModel


class BacktestRequest(APIModel):
    dataset_id: str
    model_version: str = "champion"
    minimum_net_edge: float = Field(default=0.03, ge=-1, le=1)
    fee_multiplier: float = Field(default=0.07, ge=0, le=1)
    slippage_per_contract: float = Field(default=0.01, ge=0, le=1)
    uncertainty_buffer: float = Field(default=0.01, ge=0, le=1)
    contracts_per_trade: int = Field(default=1, ge=1, le=100000)
    maximum_total_exposure: float = Field(default=1000.0, gt=0)


class BacktestRunRead(APIModel):
    id: str
    status: str
    dataset_id: str
    model_version_id: str
    configuration: dict[str, Any]
    metrics: dict[str, Any]
    trades: list[dict[str, Any]]
    error_message: str | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime

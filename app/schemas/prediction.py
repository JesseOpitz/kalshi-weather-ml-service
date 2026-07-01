from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field, model_validator

from app.schemas.common import APIModel


class OutcomeBracket(APIModel):
    label: str
    lower_bound: float | None = None
    upper_bound: float | None = None

    @model_validator(mode="after")
    def ensure_some_bound(self) -> OutcomeBracket:
        if self.lower_bound is None and self.upper_bound is None:
            raise ValueError("At least one bracket bound is required.")
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound >= self.upper_bound
        ):
            raise ValueError("lower_bound must be less than upper_bound.")
        return self


class PredictionRequest(APIModel):
    market_id: str
    market_ticker: str
    station_code: str
    market_type: str
    target_time: datetime
    lower_bound: float | None = None
    upper_bound: float | None = None
    unit: str = "F"
    prediction_time: datetime | None = None
    weather_observations: list[dict[str, Any]] = Field(default_factory=list)
    weather_forecasts: list[dict[str, Any]] = Field(default_factory=list)
    market_snapshot: dict[str, Any] = Field(default_factory=dict)
    order_book: dict[str, Any] = Field(default_factory=dict)
    feature_overrides: dict[str, Any] = Field(default_factory=dict)
    brackets: list[OutcomeBracket] = Field(default_factory=list)
    requested_model_version: str = "champion"

    @model_validator(mode="after")
    def validate_bounds(self) -> PredictionRequest:
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound >= self.upper_bound
        ):
            raise ValueError("lower_bound must be less than upper_bound.")
        return self


class Driver(APIModel):
    feature: str
    contribution: float
    current_value: Any
    reference_value: Any
    direction: str


class BracketProbability(APIModel):
    label: str
    lower_bound: float | None
    upper_bound: float | None
    raw_probability: float
    calibrated_probability: float


class PredictionResponse(APIModel):
    prediction_id: str
    model_version: str
    model_status: str
    predicted_value: float
    raw_probability: float | None
    calibrated_probability: float | None
    probability_lower_bound: float | None
    probability_upper_bound: float | None
    prediction_distribution: dict[str, Any]
    bracket_probabilities: list[BracketProbability]
    data_quality_score: float
    data_freshness_seconds: int | None
    feature_schema_version: str
    top_positive_drivers: list[Driver]
    top_negative_drivers: list[Driver]
    historical_analogues: list[dict[str, Any]]
    warnings: list[str]
    trading_eligible: bool


class ExplanationResponse(APIModel):
    model_version: str
    predicted_value: float
    top_positive_drivers: list[Driver]
    top_negative_drivers: list[Driver]
    model_disagreement: float | None
    station_bias: float | None
    historical_analogues: list[dict[str, Any]]
    warnings: list[str]

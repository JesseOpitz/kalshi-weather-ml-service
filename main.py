import os
import secrets
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field


app = FastAPI(
    title="Kalshi Weather ML Service",
    version="0.1.0",
    description="Prediction and model-management API for weather markets.",
)

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)

EXPECTED_API_KEY = os.getenv("ML_SERVICE_API_KEY")
MODEL_VERSION = os.getenv("MODEL_VERSION", "not-trained")
MODEL_MODE = os.getenv("MODEL_MODE", "unavailable")


def require_api_key(
    provided_api_key: str | None = Security(api_key_header),
) -> str:
    if not EXPECTED_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML_SERVICE_API_KEY is not configured on the server.",
        )

    if not provided_api_key or not secrets.compare_digest(
        provided_api_key,
        EXPECTED_API_KEY,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )

    return provided_api_key


class PredictionRequest(BaseModel):
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

    requested_model_version: str = "champion"


class PredictionResponse(BaseModel):
    model_version: str
    model_status: str
    predicted_value: float
    raw_probability: float
    calibrated_probability: float
    probability_lower_bound: float
    probability_upper_bound: float
    prediction_distribution: dict[str, Any]
    data_quality_score: float
    feature_schema_version: str
    top_positive_drivers: list[dict[str, Any]]
    top_negative_drivers: list[dict[str, Any]]
    historical_analogues: list[dict[str, Any]]
    warnings: list[str]


@app.get("/health")
def health() -> dict[str, Any]:
    """
    Public health endpoint. It intentionally exposes no credentials,
    model internals, market data, or user data.
    """
    return {
        "status": "online",
        "service": "kalshi-weather-ml-service",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/v1/model-health")
def model_health(
    _: str = Security(require_api_key),
) -> dict[str, Any]:
    return {
        "service_status": "healthy",
        "model_status": MODEL_MODE,
        "model_version": MODEL_VERSION,
        "feature_schema_version": "weather-features-v1",
        "supported_market_types": [
            "daily_high_temperature",
            "daily_low_temperature",
            "precipitation",
            "snowfall",
        ],
        "warnings": (
            ["No trained production model is currently loaded."]
            if MODEL_MODE != "production"
            else []
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post(
    "/v1/predict-market",
    response_model=PredictionResponse,
)
def predict_market(
    request: PredictionRequest,
    _: str = Security(require_api_key),
) -> PredictionResponse:
    """
    This endpoint deliberately refuses production predictions until
    a trained model is installed. Do not replace this with fabricated
    probabilities.
    """

    if MODEL_MODE != "production":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "MODEL_NOT_READY",
                "message": "No trained production model is loaded.",
                "market_id": request.market_id,
                "production_trading_allowed": False,
            },
        )

    # Later, replace this section with:
    #
    # features = feature_pipeline.transform(request)
    # raw_prediction = weather_model.predict_distribution(features)
    # calibrated = probability_calibrator.transform(raw_prediction)
    # explanation = explainer.explain(features)
    #
    # Never use placeholder probabilities in production.

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Production inference implementation has not been installed.",
    )


@app.post("/v1/explain-prediction")
def explain_prediction(
    payload: dict[str, Any],
    _: str = Security(require_api_key),
) -> dict[str, Any]:
    if MODEL_MODE != "production":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No trained production model is loaded.",
        )

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Prediction explanation is not yet implemented.",
    )


@app.post("/v1/backtest-strategy")
def backtest_strategy(
    payload: dict[str, Any],
    _: str = Security(require_api_key),
) -> dict[str, Any]:
    return {
        "accepted": False,
        "status": "not_configured",
        "message": "Connect this endpoint to the training/backtesting worker.",
        "production_trading_allowed": False,
    )

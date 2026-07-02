from __future__ import annotations

import hashlib
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from orjson import orjson
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.middleware import RequestContextMiddleware, SimpleRateLimitMiddleware
from app.core.security import ApiPrincipal, require_admin_key, require_api_key
from app.db.base import Base
from app.db.models import (
    BacktestRun,
    DatasetManifest,
    ModelVersion,
    PredictionRecord,
    Station,
    TrainingJob,
)
from app.db.session import engine, get_db
from app.ml.backtest import run_backtest
from app.ml.features import build_feature_row, normalize_training_frame
from app.ml.model_store import get_model_store
from app.ml.trainer import train_model_bundle
from app.schemas.learning import TrainCandidateRequest
from app.schemas.model import ModelPromotionRequest
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.schemas.station import StationCreate
from app.schemas.weather_data import OpenMeteoDatasetRequest

settings = get_settings()
configure_logging()


class BacktestRequest(BaseModel):
    dataset_id: str
    model_version: str = "champion"
    minimum_net_edge: float = Field(default=0.03, ge=-1, le=1)
    fee_multiplier: float = Field(default=0.07, ge=0, le=1)
    slippage_per_contract: float = Field(default=0.01, ge=0, le=1)
    uncertainty_buffer: float = Field(default=0.01, ge=0, le=1)
    contracts_per_trade: int = Field(default=1, ge=1, le=100000)
    maximum_total_exposure: float = Field(default=1000.0, gt=0)


def _ensure_directories() -> None:
    Path("data/datasets").mkdir(parents=True, exist_ok=True)
    settings.model_store_path.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _ensure_directories()
    if settings.auto_create_tables:
        Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Weather-model training, calibrated probability inference, registry, and evaluation API.",
    lifespan=lifespan,
)
app.add_middleware(RequestContextMiddleware)
app.add_middleware(SimpleRateLimitMiddleware)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.trusted_hosts)
if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key", "X-Admin-Key", "X-Request-ID"],
    )


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_dataset(manifest: DatasetManifest) -> pd.DataFrame:
    path = Path(manifest.file_uri)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Dataset file is unavailable")
    if _checksum(path) != manifest.checksum_sha256:
        raise HTTPException(status_code=409, detail="Dataset checksum validation failed")
    frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    return normalize_training_frame(frame)


def _model_record(db: Session, requested: str) -> ModelVersion:
    query = db.query(ModelVersion)
    if requested == "champion":
        record = query.filter(ModelVersion.status == "champion").order_by(ModelVersion.promoted_at.desc()).first()
    else:
        record = query.filter(ModelVersion.version == requested).one_or_none()
    if record is None:
        raise HTTPException(status_code=404, detail=f"Model '{requested}' is unavailable")
    return record


def _load_bundle(record: ModelVersion):
    return get_model_store().load_bundle(record.artifact_uri, record.artifact_sha256)


def _station_values(station: Station) -> dict[str, Any]:
    return {
        "station_code": station.station_code,
        "station_name": station.station_name,
        "latitude": station.latitude,
        "longitude": station.longitude,
        "elevation_m": station.elevation_m,
        "timezone": station.timezone,
    }


def _model_payload(record: ModelVersion) -> dict[str, Any]:
    return {
        "id": record.id,
        "model_name": record.model_name,
        "version": record.version,
        "status": record.status,
        "feature_schema_version": record.feature_schema_version,
        "training_dataset_id": record.training_dataset_id,
        "training_start_date": record.training_start_date,
        "training_end_date": record.training_end_date,
        "metrics": record.metrics,
        "hyperparameters": record.hyperparameters,
        "metadata": record.metadata_json,
        "promoted_at": record.promoted_at,
        "retired_at": record.retired_at,
        "created_at": record.created_at,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "online",
        "service": settings.app_name,
        "environment": settings.app_env,
        "version": "1.0.0",
        "timestamp": datetime.now(UTC),
    }


@app.get("/ready")
def ready(db: Session = Depends(get_db)) -> dict[str, Any]:
    db.execute(text("SELECT 1"))
    return {"status": "ready", "timestamp": datetime.now(UTC)}


@app.get("/v1/stations")
def list_stations(
    db: Session = Depends(get_db),
    _: ApiPrincipal = Depends(require_api_key),
) -> list[dict[str, Any]]:
    return [
        {
            "id": item.id,
            "station_code": item.station_code,
            "station_name": item.station_name,
            "latitude": item.latitude,
            "longitude": item.longitude,
            "elevation_m": item.elevation_m,
            "timezone": item.timezone,
            "official_source": item.official_source,
            "metadata": item.metadata_json,
            "active": item.active,
        }
        for item in db.query(Station).order_by(Station.station_code).all()
    ]


@app.post("/v1/stations", status_code=status.HTTP_201_CREATED)
def save_station(
    payload: StationCreate,
    db: Session = Depends(get_db),
    _: ApiPrincipal = Depends(require_admin_key),
) -> dict[str, Any]:
    item = db.query(Station).filter(Station.station_code == payload.station_code).one_or_none()
    if item is None:
        item = Station(
            station_code=payload.station_code,
            station_name=payload.station_name,
            latitude=payload.latitude,
            longitude=payload.longitude,
        )
        db.add(item)
    item.station_name = payload.station_name
    item.latitude = payload.latitude
    item.longitude = payload.longitude
    item.elevation_m = payload.elevation_m
    item.timezone = payload.timezone
    item.official_source = payload.official_source
    item.metadata_json = payload.metadata
    item.active = payload.active
    db.commit()
    db.refresh(item)
    return {"id": item.id, "station_code": item.station_code, "station_name": item.station_name}


@app.get("/v1/datasets")
def list_datasets(
    db: Session = Depends(get_db),
    _: ApiPrincipal = Depends(require_api_key),
) -> list[dict[str, Any]]:
    return [
        {
            "id": item.id,
            "name": item.name,
            "source_type": item.source_type,
            "station_code": item.station_code,
            "row_count": item.row_count,
            "date_start": item.date_start,
            "date_end": item.date_end,
            "quality_report": item.quality_report,
            "created_at": item.created_at,
        }
        for item in db.query(DatasetManifest).order_by(DatasetManifest.created_at.desc()).all()
    ]


@app.post("/v1/weather-data/open-meteo", status_code=status.HTTP_201_CREATED)
def build_open_meteo_dataset(
    payload: OpenMeteoDatasetRequest,
    db: Session = Depends(get_db),
    _: ApiPrincipal = Depends(require_admin_key),
) -> dict[str, Any]:
    station = db.query(Station).filter(Station.station_code == payload.station_code).one_or_none()
    if station is None:
        raise HTTPException(status_code=404, detail="Station is not registered")

    common = {
        "latitude": station.latitude,
        "longitude": station.longitude,
        "start_date": payload.start_date.isoformat(),
        "end_date": payload.end_date.isoformat(),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "UTC",
    }
    headers = {"User-Agent": settings.http_user_agent, "Accept": "application/json"}
    with httpx.Client(timeout=settings.http_timeout_seconds, headers=headers) as client:
        archive_response = client.get(
            settings.open_meteo_archive_url,
            params={
                **common,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,shortwave_radiation_sum",
            },
        )
        archive_response.raise_for_status()
        forecast_response = client.get(
            settings.open_meteo_historical_forecast_url,
            params={
                **common,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,wind_speed_10m_max",
            },
        )
        forecast_response.raise_for_status()

    actual = pd.DataFrame(archive_response.json().get("daily") or {}).rename(
        columns={"time": "target_time", "temperature_2m_max": "actual_value"}
    )
    forecast = pd.DataFrame(forecast_response.json().get("daily") or {}).rename(
        columns={
            "time": "target_time",
            "temperature_2m_max": "forecast_mean",
            "temperature_2m_min": "forecast_low",
            "precipitation_probability_max": "precipitation_probability",
            "wind_speed_10m_max": "wind_speed",
        }
    )
    if actual.empty or forecast.empty:
        raise HTTPException(status_code=502, detail="Weather provider returned no usable records")

    frame = forecast.merge(actual[["target_time", "actual_value"]], on="target_time", how="inner")
    frame["target_time"] = pd.to_datetime(frame["target_time"], utc=True)
    frame["station_code"] = station.station_code
    frame["market_type"] = payload.market_type
    frame["latitude"] = station.latitude
    frame["longitude"] = station.longitude
    frame["elevation_m"] = station.elevation_m or 0.0
    frame["lead_hours"] = payload.assumed_lead_hours
    frame["forecast_median"] = frame["forecast_mean"]
    frame["forecast_min"] = frame["forecast_mean"] - 1.5
    frame["forecast_max"] = frame["forecast_mean"] + 1.5
    frame["forecast_std"] = 1.5
    frame["model_count"] = 1
    frame["data_age_seconds"] = payload.assumed_lead_hours * 3600
    normalized = normalize_training_frame(frame)
    if len(normalized) < settings.min_training_rows:
        raise HTTPException(
            status_code=422,
            detail=f"Only {len(normalized)} valid rows were produced; at least {settings.min_training_rows} are required",
        )

    path = Path("data/datasets") / f"{uuid.uuid4()}.csv"
    normalized.to_csv(path, index=False)
    manifest = DatasetManifest(
        name=payload.name,
        source_type="open_meteo_bootstrap",
        station_code=station.station_code,
        file_uri=str(path),
        checksum_sha256=_checksum(path),
        row_count=len(normalized),
        date_start=normalized["target_time"].min().date(),
        date_end=normalized["target_time"].max().date(),
        quality_report={
            "valid_rows": len(normalized),
            "station_count": int(normalized["station_code"].nunique()),
            "warning": "Bootstrap gridded data should be replaced or augmented with exact official-station history before production use.",
        },
    )
    db.add(manifest)
    db.commit()
    db.refresh(manifest)
    return {"id": manifest.id, "row_count": manifest.row_count, "quality_report": manifest.quality_report}


@app.post("/v1/train-candidate", status_code=status.HTTP_201_CREATED)
def train_candidate(
    payload: TrainCandidateRequest,
    db: Session = Depends(get_db),
    _: ApiPrincipal = Depends(require_admin_key),
) -> dict[str, Any]:
    manifest = db.get(DatasetManifest, payload.dataset_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="Dataset is unavailable")
    frame = _load_dataset(manifest)
    version = payload.requested_version or f"{payload.model_name}-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    config = payload.model_dump(exclude={"dataset_id", "model_name", "requested_version"})
    job = TrainingJob(
        dataset_id=manifest.id,
        model_name=payload.model_name,
        requested_version=version,
        configuration=config,
        status="running",
        progress=0.1,
        started_at=datetime.now(UTC),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    try:
        result = train_model_bundle(frame, version=version, config=config)
        artifact_uri, artifact_hash = get_model_store().save_bundle(result.bundle)
        model = ModelVersion(
            model_name=payload.model_name,
            version=version,
            status="candidate",
            artifact_uri=artifact_uri,
            artifact_sha256=artifact_hash,
            feature_schema_version=result.bundle.feature_schema_version,
            training_dataset_id=manifest.id,
            training_start_date=result.date_start.date(),
            training_end_date=result.date_end.date(),
            metrics=result.metrics,
            hyperparameters=config,
            metadata_json=result.bundle.metadata,
        )
        db.add(model)
        db.flush()
        job.status = "completed"
        job.progress = 1.0
        job.result_model_version_id = model.id
        job.completed_at = datetime.now(UTC)
        db.commit()
        db.refresh(model)
        return _model_payload(model)
    except Exception as exc:
        db.rollback()
        job = db.get(TrainingJob, job.id)
        if job is not None:
            job.status = "failed"
            job.error_message = str(exc)
            job.completed_at = datetime.now(UTC)
            db.commit()
        raise HTTPException(status_code=422, detail=f"Training failed: {exc}") from exc


@app.get("/v1/model-versions")
def list_models(
    db: Session = Depends(get_db),
    _: ApiPrincipal = Depends(require_api_key),
) -> list[dict[str, Any]]:
    return [_model_payload(item) for item in db.query(ModelVersion).order_by(ModelVersion.created_at.desc()).all()]


@app.get("/v1/model-health")
def model_health(
    db: Session = Depends(get_db),
    _: ApiPrincipal = Depends(require_api_key),
) -> dict[str, Any]:
    try:
        record = _model_record(db, "champion")
        bundle = _load_bundle(record)
    except Exception as exc:
        return {
            "service_status": "healthy",
            "model_status": "unavailable",
            "model_version": None,
            "artifact_verified": False,
            "supported_stations": [],
            "supported_market_types": [],
            "warnings": [str(exc)],
            "timestamp": datetime.now(UTC),
        }
    return {
        "service_status": "healthy",
        "model_status": record.status,
        "model_version": record.version,
        "model_id": record.id,
        "feature_schema_version": record.feature_schema_version,
        "artifact_verified": True,
        "supported_stations": bundle.supported_stations,
        "supported_market_types": bundle.supported_market_types,
        "metrics": record.metrics,
        "warnings": [],
        "timestamp": datetime.now(UTC),
    }


@app.post("/v1/promote-model")
def promote_model(
    payload: ModelPromotionRequest,
    db: Session = Depends(get_db),
    _: ApiPrincipal = Depends(require_admin_key),
) -> dict[str, Any]:
    record = _model_record(db, payload.model_version)
    _load_bundle(record)
    now = datetime.now(UTC)
    for champion in db.query(ModelVersion).filter(ModelVersion.status == "champion").all():
        if champion.id != record.id:
            champion.status = "retired"
            champion.retired_at = now
    record.status = "champion"
    record.promoted_at = now
    record.retired_at = None
    db.commit()
    db.refresh(record)
    return _model_payload(record)


@app.post("/v1/predict-market", response_model=PredictionResponse)
def predict_market(
    payload: PredictionRequest,
    db: Session = Depends(get_db),
    _: ApiPrincipal = Depends(require_api_key),
) -> PredictionResponse:
    station = db.query(Station).filter(Station.station_code == payload.station_code).one_or_none()
    if station is None:
        raise HTTPException(status_code=404, detail="Station is not registered")
    record = _model_record(db, payload.requested_model_version)
    bundle = _load_bundle(record)
    prediction_time = payload.prediction_time or datetime.now(UTC)
    row, diagnostics = build_feature_row(
        station_code=payload.station_code,
        market_type=payload.market_type,
        target_time=payload.target_time,
        prediction_time=prediction_time,
        station=_station_values(station),
        observations=payload.weather_observations,
        forecasts=payload.weather_forecasts,
        overrides=payload.feature_overrides,
    )
    predicted_value, interval_low, interval_high, sigma = bundle.predict_value(row)
    raw_probability = None
    calibrated_probability = None
    probability_low = None
    probability_high = None
    if payload.lower_bound is not None or payload.upper_bound is not None:
        raw_probability, calibrated_probability = bundle.predict_event(
            row, payload.lower_bound, payload.upper_bound
        )
        candidates = [
            bundle.calibrate(
                bundle.raw_event_probability_with_sigma(
                    predicted_value, sigma * factor, payload.lower_bound, payload.upper_bound
                )
            )
            for factor in (0.8, 1.2)
        ]
        probability_low, probability_high = min(candidates), max(candidates)

    bracket_probabilities = []
    for bracket in payload.brackets:
        raw, calibrated = bundle.predict_event(row, bracket.lower_bound, bracket.upper_bound)
        bracket_probabilities.append(
            {
                "label": bracket.label,
                "lower_bound": bracket.lower_bound,
                "upper_bound": bracket.upper_bound,
                "raw_probability": raw,
                "calibrated_probability": calibrated,
            }
        )

    drivers = bundle.explain(row, limit=12)
    positive = [item for item in drivers if item["contribution"] > 0][:6]
    negative = [item for item in drivers if item["contribution"] < 0][:6]
    data_age = diagnostics.get("data_age_seconds")
    warnings: list[str] = []
    if diagnostics["missing_critical_features"]:
        warnings.append("Missing critical features: " + ", ".join(diagnostics["missing_critical_features"]))
    if data_age is not None and data_age > settings.max_input_data_age_seconds:
        warnings.append("Latest observation is stale")
    if payload.station_code not in bundle.supported_stations:
        warnings.append("Station was not represented during model training")
    if payload.market_type not in bundle.supported_market_types:
        warnings.append("Market type was not represented during model training")
    if record.status != "champion":
        warnings.append("A non-champion model was requested")

    eligible = (
        record.status == "champion"
        and diagnostics["data_quality_score"] >= settings.min_data_quality_score
        and not diagnostics["missing_critical_features"]
        and (data_age is None or data_age <= settings.max_input_data_age_seconds)
        and payload.station_code in bundle.supported_stations
        and payload.market_type in bundle.supported_market_types
    )
    response = {
        "model_version": record.version,
        "model_status": record.status,
        "predicted_value": predicted_value,
        "raw_probability": raw_probability,
        "calibrated_probability": calibrated_probability,
        "probability_lower_bound": probability_low,
        "probability_upper_bound": probability_high,
        "prediction_distribution": {
            "family": "calibrated_normal_approximation",
            "mean": predicted_value,
            "sigma": sigma,
            "quantile_10": interval_low,
            "quantile_50": predicted_value,
            "quantile_90": interval_high,
            "unit": payload.unit,
        },
        "bracket_probabilities": bracket_probabilities,
        "data_quality_score": diagnostics["data_quality_score"],
        "data_freshness_seconds": data_age,
        "feature_schema_version": bundle.feature_schema_version,
        "top_positive_drivers": positive,
        "top_negative_drivers": negative,
        "historical_analogues": bundle.analogues(row),
        "warnings": warnings,
        "trading_eligible": eligible,
    }
    prediction = PredictionRecord(
        model_version_id=record.id,
        market_id=payload.market_id,
        market_ticker=payload.market_ticker,
        station_code=payload.station_code,
        market_type=payload.market_type,
        target_time=payload.target_time,
        lower_bound=payload.lower_bound,
        upper_bound=payload.upper_bound,
        predicted_value=predicted_value,
        raw_probability=raw_probability,
        calibrated_probability=calibrated_probability,
        probability_lower_bound=probability_low,
        probability_upper_bound=probability_high,
        data_quality_score=diagnostics["data_quality_score"],
        trading_eligible=eligible,
        request_payload=payload.model_dump(mode="json"),
        response_payload=response,
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return PredictionResponse.model_validate({"prediction_id": prediction.id, **response})


@app.post("/v1/backtest-strategy", status_code=status.HTTP_201_CREATED)
def backtest_strategy(
    payload: BacktestRequest,
    db: Session = Depends(get_db),
    _: ApiPrincipal = Depends(require_api_key),
) -> dict[str, Any]:
    manifest = db.get(DatasetManifest, payload.dataset_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="Dataset is unavailable")
    record = _model_record(db, payload.model_version)
    bundle = _load_bundle(record)
    configuration = payload.model_dump(exclude={"dataset_id", "model_version"})
    metrics, trades = run_backtest(bundle, _load_dataset(manifest), configuration)
    run = BacktestRun(
        status="completed",
        dataset_id=manifest.id,
        model_version_id=record.id,
        configuration=configuration,
        metrics=metrics,
        trades=trades[:5000],
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return {
        "id": run.id,
        "status": run.status,
        "metrics": run.metrics,
        "trades": run.trades,
        "created_at": run.created_at,
    }

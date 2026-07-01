from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import PredictionRecord, Station
from app.ml.features import build_feature_row
from app.schemas.prediction import PredictionRequest
from app.services.model_service import get_model_record, load_model_bundle


def _station_mapping(station: Station) -> dict[str, Any]:
    return {
        "station_code": station.station_code,
        "station_name": station.station_name,
        "latitude": station.latitude,
        "longitude": station.longitude,
        "elevation_m": station.elevation_m,
        "timezone": station.timezone,
    }


def generate_prediction(db: Session, request: PredictionRequest) -> dict[str, Any]:
    settings = get_settings()
    station = db.query(Station).filter(Station.station_code == request.station_code).one_or_none()
    if station is None:
        raise LookupError(f"Station '{request.station_code}' has not been registered.")
    model_record = get_model_record(db, request.requested_model_version)
    if model_record.status not in {"champion", "candidate", "challenger"}:
        raise ValueError(f"Model status '{model_record.status}' is not prediction eligible.")
    bundle = load_model_bundle(model_record)
    prediction_time = request.prediction_time or datetime.now(UTC)
    row, diagnostics = build_feature_row(
        station_code=request.station_code,
        market_type=request.market_type,
        target_time=request.target_time,
        prediction_time=prediction_time,
        station=_station_mapping(station),
        observations=request.weather_observations,
        forecasts=request.weather_forecasts,
        overrides=request.feature_overrides,
    )
    predicted_value, interval_low, interval_high, sigma = bundle.predict_value(row)

    raw_probability = None
    calibrated_probability = None
    probability_low = None
    probability_high = None
    if request.lower_bound is not None or request.upper_bound is not None:
        raw_probability, calibrated_probability = bundle.predict_event(
            row, request.lower_bound, request.upper_bound
        )
        conservative_sigma = sigma * 1.20
        optimistic_sigma = max(sigma * 0.80, 0.25)
        probability_candidates = [
            bundle.calibrate(
                bundle.raw_event_probability_with_sigma(
                    predicted_value,
                    conservative_sigma,
                    request.lower_bound,
                    request.upper_bound,
                )
            ),
            bundle.calibrate(
                bundle.raw_event_probability_with_sigma(
                    predicted_value,
                    optimistic_sigma,
                    request.lower_bound,
                    request.upper_bound,
                )
            ),
        ]
        probability_low = min(probability_candidates)
        probability_high = max(probability_candidates)

    brackets = []
    for bracket in request.brackets:
        raw, calibrated = bundle.predict_event(row, bracket.lower_bound, bracket.upper_bound)
        brackets.append(
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
    warnings: list[str] = []
    if diagnostics["missing_critical_features"]:
        warnings.append(
            "Missing critical features: " + ", ".join(diagnostics["missing_critical_features"])
        )
    data_age = diagnostics.get("data_age_seconds")
    if data_age is not None and data_age > settings.max_input_data_age_seconds:
        warnings.append("Latest observation is older than the configured freshness threshold.")
    if request.station_code not in bundle.supported_stations:
        warnings.append("Station was not represented in this model's training data.")
    if request.market_type not in bundle.supported_market_types:
        warnings.append("Market type was not represented in this model's training data.")
    if model_record.status != "champion":
        warnings.append("Prediction used a non-champion model and is not production-trading eligible.")

    quality = diagnostics["data_quality_score"]
    trading_eligible = (
        model_record.status == "champion"
        and quality >= settings.min_data_quality_score
        and not diagnostics["missing_critical_features"]
        and (data_age is None or data_age <= settings.max_input_data_age_seconds)
        and request.station_code in bundle.supported_stations
        and request.market_type in bundle.supported_market_types
    )

    payload = {
        "model_version": model_record.version,
        "model_status": model_record.status,
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
            "unit": request.unit,
        },
        "bracket_probabilities": brackets,
        "data_quality_score": quality,
        "data_freshness_seconds": data_age,
        "feature_schema_version": bundle.feature_schema_version,
        "top_positive_drivers": positive,
        "top_negative_drivers": negative,
        "historical_analogues": bundle.analogues(row),
        "warnings": warnings,
        "trading_eligible": trading_eligible,
    }
    record = PredictionRecord(
        model_version_id=model_record.id,
        market_id=request.market_id,
        market_ticker=request.market_ticker,
        station_code=request.station_code,
        market_type=request.market_type,
        target_time=request.target_time,
        lower_bound=request.lower_bound,
        upper_bound=request.upper_bound,
        predicted_value=predicted_value,
        raw_probability=raw_probability,
        calibrated_probability=calibrated_probability,
        probability_lower_bound=probability_low,
        probability_upper_bound=probability_high,
        data_quality_score=quality,
        trading_eligible=trading_eligible,
        request_payload=request.model_dump(mode="json"),
        response_payload=payload,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return {"prediction_id": record.id, **payload}

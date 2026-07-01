from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "latitude",
    "longitude",
    "elevation_m",
    "lead_hours",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "hour_sin",
    "hour_cos",
    "forecast_mean",
    "forecast_median",
    "forecast_std",
    "forecast_min",
    "forecast_max",
    "latest_temperature",
    "observation_trend_1h",
    "dew_point",
    "humidity",
    "wind_speed",
    "wind_direction_sin",
    "wind_direction_cos",
    "cloud_cover",
    "pressure",
    "precipitation_probability",
    "model_count",
    "data_age_seconds",
]

CATEGORICAL_FEATURES = ["station_code", "market_type"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "actual_value"
TIME_COLUMN = "target_time"


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def _first_numeric(item: dict[str, Any], keys: Iterable[str]) -> float | None:
    for key in keys:
        value = _as_float(item.get(key))
        if value is not None:
            return value
    return None


def _parse_time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _latest(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not items:
        return None
    return max(
        items,
        key=lambda item: _parse_time(
            item.get("observation_time")
            or item.get("time")
            or item.get("timestamp")
            or item.get("available_time")
        )
        or datetime.min.replace(tzinfo=UTC),
    )


def _temperature_values(items: list[dict[str, Any]]) -> list[float]:
    keys = (
        "predicted_high",
        "temperature_2m_max",
        "forecast_high",
        "temperature",
        "value",
        "temp",
    )
    values: list[float] = []
    for item in items:
        value = _first_numeric(item, keys)
        if value is not None:
            values.append(value)
    return values


def build_feature_row(
    *,
    station_code: str,
    market_type: str,
    target_time: datetime,
    prediction_time: datetime,
    station: dict[str, Any],
    observations: list[dict[str, Any]],
    forecasts: list[dict[str, Any]],
    overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prediction_time = prediction_time.astimezone(UTC)
    if target_time.tzinfo is None:
        target_time = target_time.replace(tzinfo=UTC)
    target_time = target_time.astimezone(UTC)

    forecast_values = _temperature_values(forecasts)
    latest_observation = _latest(observations)
    observation_temperatures = []
    dated_observations: list[tuple[datetime, float]] = []
    for item in observations:
        value = _first_numeric(item, ("temperature", "temp", "value", "temperature_2m"))
        timestamp = _parse_time(
            item.get("observation_time") or item.get("time") or item.get("timestamp")
        )
        if value is not None:
            observation_temperatures.append(value)
            if timestamp:
                dated_observations.append((timestamp, value))
    dated_observations.sort(key=lambda pair: pair[0])

    trend = None
    if len(dated_observations) >= 2:
        first_time, first_value = dated_observations[-2]
        last_time, last_value = dated_observations[-1]
        hours = max((last_time - first_time).total_seconds() / 3600.0, 1 / 60)
        trend = (last_value - first_value) / hours

    latest_temp = observation_temperatures[-1] if observation_temperatures else None
    latest_timestamp = None
    if latest_observation:
        latest_timestamp = _parse_time(
            latest_observation.get("observation_time")
            or latest_observation.get("time")
            or latest_observation.get("timestamp")
        )

    forecast_mean = float(np.mean(forecast_values)) if forecast_values else None
    forecast_median = float(np.median(forecast_values)) if forecast_values else None
    forecast_std = float(np.std(forecast_values)) if len(forecast_values) > 1 else None
    forecast_min = min(forecast_values) if forecast_values else None
    forecast_max = max(forecast_values) if forecast_values else None

    source_for_conditions = latest_observation or (forecasts[0] if forecasts else {})
    wind_direction = _first_numeric(source_for_conditions, ("wind_direction", "wind_direction_10m"))
    wind_radians = math.radians(wind_direction or 0.0)

    day_of_year = target_time.timetuple().tm_yday
    row: dict[str, Any] = {
        "latitude": station.get("latitude"),
        "longitude": station.get("longitude"),
        "elevation_m": station.get("elevation_m"),
        "lead_hours": max((target_time - prediction_time).total_seconds() / 3600.0, 0.0),
        "month_sin": math.sin(2 * math.pi * target_time.month / 12.0),
        "month_cos": math.cos(2 * math.pi * target_time.month / 12.0),
        "doy_sin": math.sin(2 * math.pi * day_of_year / 366.0),
        "doy_cos": math.cos(2 * math.pi * day_of_year / 366.0),
        "hour_sin": math.sin(2 * math.pi * target_time.hour / 24.0),
        "hour_cos": math.cos(2 * math.pi * target_time.hour / 24.0),
        "forecast_mean": forecast_mean,
        "forecast_median": forecast_median,
        "forecast_std": forecast_std,
        "forecast_min": forecast_min,
        "forecast_max": forecast_max,
        "latest_temperature": latest_temp,
        "observation_trend_1h": trend,
        "dew_point": _first_numeric(source_for_conditions, ("dew_point", "dewpoint")),
        "humidity": _first_numeric(source_for_conditions, ("humidity", "relative_humidity")),
        "wind_speed": _first_numeric(source_for_conditions, ("wind_speed", "wind_speed_10m")),
        "wind_direction_sin": math.sin(wind_radians),
        "wind_direction_cos": math.cos(wind_radians),
        "cloud_cover": _first_numeric(source_for_conditions, ("cloud_cover", "clouds")),
        "pressure": _first_numeric(source_for_conditions, ("pressure", "surface_pressure")),
        "precipitation_probability": _first_numeric(
            source_for_conditions,
            ("precipitation_probability", "precip_probability"),
        ),
        "model_count": len(forecast_values),
        "data_age_seconds": (
            max((prediction_time - latest_timestamp).total_seconds(), 0.0)
            if latest_timestamp
            else None
        ),
        "station_code": station_code,
        "market_type": market_type,
    }
    if overrides:
        for key, value in overrides.items():
            if key in ALL_FEATURES:
                row[key] = value

    missing_critical = [
        key for key in ("latitude", "longitude", "forecast_mean") if row.get(key) is None
    ]
    observed = sum(row.get(key) is not None for key in NUMERIC_FEATURES)
    quality = observed / len(NUMERIC_FEATURES)
    if missing_critical:
        quality *= 0.45
    if not observations:
        quality *= 0.90
    diagnostics = {
        "missing_critical_features": missing_critical,
        "observed_numeric_features": observed,
        "numeric_feature_count": len(NUMERIC_FEATURES),
        "data_quality_score": round(max(0.0, min(1.0, quality)), 4),
        "data_age_seconds": int(row["data_age_seconds"]) if row["data_age_seconds"] else None,
        "forecast_model_count": len(forecast_values),
    }
    return row, diagnostics


def normalize_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if TARGET_COLUMN not in normalized.columns:
        aliases = ["actual_temperature", "actual_high", "target", "label"]
        for alias in aliases:
            if alias in normalized.columns:
                normalized[TARGET_COLUMN] = normalized[alias]
                break
    if TIME_COLUMN not in normalized.columns:
        for alias in ("date", "target_date", "valid_time", "forecast_valid_time"):
            if alias in normalized.columns:
                normalized[TIME_COLUMN] = normalized[alias]
                break
    if "forecast_mean" not in normalized.columns:
        for alias in ("forecast_temperature", "forecast_high", "predicted_high"):
            if alias in normalized.columns:
                normalized["forecast_mean"] = normalized[alias]
                break
    missing_required = [
        column for column in (TARGET_COLUMN, TIME_COLUMN, "forecast_mean") if column not in normalized.columns
    ]
    if missing_required:
        raise ValueError(f"Dataset is missing required columns: {missing_required}")
    if "forecast_median" not in normalized.columns:
        normalized["forecast_median"] = normalized["forecast_mean"]
    if "forecast_min" not in normalized.columns:
        normalized["forecast_min"] = normalized.get("forecast_mean")
    if "forecast_max" not in normalized.columns:
        normalized["forecast_max"] = normalized.get("forecast_mean")
    if "forecast_std" not in normalized.columns:
        normalized["forecast_std"] = 1.5
    if "station_code" not in normalized.columns:
        normalized["station_code"] = "UNKNOWN"
    if "market_type" not in normalized.columns:
        normalized["market_type"] = "daily_high_temperature"
    if "lead_hours" not in normalized.columns:
        normalized["lead_hours"] = 24.0

    normalized[TIME_COLUMN] = pd.to_datetime(normalized[TIME_COLUMN], utc=True, errors="coerce")
    target_times = normalized[TIME_COLUMN]
    normalized["month_sin"] = np.sin(2 * np.pi * target_times.dt.month / 12.0)
    normalized["month_cos"] = np.cos(2 * np.pi * target_times.dt.month / 12.0)
    doy = target_times.dt.dayofyear
    normalized["doy_sin"] = np.sin(2 * np.pi * doy / 366.0)
    normalized["doy_cos"] = np.cos(2 * np.pi * doy / 366.0)
    hour = target_times.dt.hour
    normalized["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    normalized["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    defaults: dict[str, Any] = {
        "latitude": 0.0,
        "longitude": 0.0,
        "elevation_m": 0.0,
        "latest_temperature": np.nan,
        "observation_trend_1h": np.nan,
        "dew_point": np.nan,
        "humidity": np.nan,
        "wind_speed": np.nan,
        "wind_direction_sin": 0.0,
        "wind_direction_cos": 1.0,
        "cloud_cover": np.nan,
        "pressure": np.nan,
        "precipitation_probability": np.nan,
        "model_count": 1.0,
        "data_age_seconds": np.nan,
    }
    for column, default in defaults.items():
        if column not in normalized.columns:
            normalized[column] = default
    for column in NUMERIC_FEATURES + [TARGET_COLUMN]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.dropna(subset=[TIME_COLUMN, TARGET_COLUMN, "forecast_mean"])
    return normalized.sort_values(TIME_COLUMN).reset_index(drop=True)

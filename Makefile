from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

DB_PATH = Path("/tmp/kalshi_weather_ml_pytest.sqlite3")
ARTIFACT_PATH = Path("/tmp/kalshi_weather_ml_pytest_artifacts")
DATASET_PATH = Path("/tmp/kalshi_weather_ml_pytest_training.csv")
DB_PATH.unlink(missing_ok=True)
if ARTIFACT_PATH.exists():
    import shutil

    shutil.rmtree(ARTIFACT_PATH)

os.environ.update(
    {
        "ML_SERVICE_API_KEYS": "test-api-key",
        "ADMIN_API_KEY": "test-admin-key",
        "DATABASE_URL": f"sqlite:///{DB_PATH}",
        "MODEL_STORE_URI": str(ARTIFACT_PATH),
        "TASK_ALWAYS_EAGER": "true",
        "AUTO_CREATE_TABLES": "true",
        "MIN_DATA_QUALITY_SCORE": "0.50",
    }
)

from app.db.base import Base  # noqa: E402
from app.db.models import DatasetManifest, Station, TrainingJob  # noqa: E402
from app.db.session import SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.services.model_service import promote_model  # noqa: E402
from app.services.training_service import execute_training_job  # noqa: E402


@pytest.fixture(scope="session")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def trained_model() -> str:
    Base.metadata.create_all(engine)
    rng = np.random.default_rng(17)
    rows = 260
    dates = pd.date_range("2022-01-01", periods=rows, freq="D", tz="UTC")
    seasonal = 60 + 22 * np.sin(2 * np.pi * (dates.dayofyear - 170) / 365.25)
    forecast = seasonal + rng.normal(0, 1.8, rows)
    actual = forecast + rng.normal(0, 1.6, rows)
    lower = np.floor(forecast / 2.0) * 2.0
    frame = pd.DataFrame(
        {
            "target_time": dates,
            "station_code": "KTEST",
            "market_type": "daily_high_temperature",
            "latitude": 40.7,
            "longitude": -73.9,
            "elevation_m": 20.0,
            "lead_hours": 24.0,
            "forecast_mean": forecast,
            "forecast_median": forecast,
            "forecast_std": 1.8,
            "forecast_min": forecast - 2,
            "forecast_max": forecast + 2,
            "latest_temperature": forecast - 5,
            "observation_trend_1h": 1.0,
            "dew_point": forecast - 10,
            "humidity": rng.uniform(35, 90, rows),
            "wind_speed": rng.uniform(1, 18, rows),
            "wind_direction_sin": 0.0,
            "wind_direction_cos": 1.0,
            "cloud_cover": rng.uniform(0, 100, rows),
            "pressure": 1013.0,
            "precipitation_probability": rng.uniform(0, 80, rows),
            "model_count": 4,
            "data_age_seconds": 600,
            "actual_value": actual,
            "lower_bound": lower,
            "upper_bound": lower + 2,
            "market_yes_ask": np.clip(rng.normal(0.45, 0.12, rows), 0.05, 0.95),
        }
    )
    frame.to_csv(DATASET_PATH, index=False)
    digest = hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()
    with SessionLocal() as db:
        station = db.query(Station).filter(Station.station_code == "KTEST").one_or_none()
        if station is None:
            db.add(
                Station(
                    station_code="KTEST",
                    station_name="Test Station",
                    latitude=40.7,
                    longitude=-73.9,
                    elevation_m=20,
                    timezone="America/New_York",
                    official_source="test",
                )
            )
        manifest = DatasetManifest(
            name="pytest training",
            source_type="test",
            station_code="KTEST",
            file_uri=str(DATASET_PATH),
            checksum_sha256=digest,
            row_count=rows,
            date_start=dates.min().date(),
            date_end=dates.max().date(),
            quality_report={"valid_rows": rows},
        )
        db.add(manifest)
        db.commit()
        db.refresh(manifest)
        job = TrainingJob(
            dataset_id=manifest.id,
            model_name="pytest-model",
            requested_version="pytest-v1",
            configuration={
                "max_iter": 30,
                "learning_rate": 0.08,
                "max_depth": 2,
                "min_samples_leaf": 8,
                "subsample": 0.9,
                "random_state": 42,
                "calibration_method": "isotonic",
            },
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        model = execute_training_job(db, job.id)
        promote_model(db, model)
        return model.version

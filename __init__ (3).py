from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Kalshi Weather ML Service"
    app_env: str = "development"
    log_level: str = "INFO"
    api_prefix: str = "/v1"

    ml_service_api_keys: str = ""
    admin_api_key: str = ""

    database_url: str = "sqlite:///./data/weather_ml.sqlite3"
    redis_url: str = "redis://localhost:6379/0"
    task_always_eager: bool = True

    model_store_uri: str = "./artifacts"
    active_model_cache_seconds: int = 60
    min_data_quality_score: float = 0.70
    max_input_data_age_seconds: int = 7200
    min_training_rows: int = 250

    rate_limit_requests: int = 240
    rate_limit_window_seconds: int = 60
    cors_origins: list[str] = []
    trusted_hosts: list[str] = ["*"]

    open_meteo_archive_url: str = "https://archive-api.open-meteo.com/v1/archive"
    open_meteo_historical_forecast_url: str = (
        "https://historical-forecast-api.open-meteo.com/v1/forecast"
    )
    nws_api_url: str = "https://api.weather.gov"
    http_user_agent: str = "kalshi-weather-ml-service/1.0 contact@example.com"
    http_timeout_seconds: int = 45

    s3_endpoint_url: str | None = None
    aws_region: str = "us-east-1"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    prometheus_enabled: bool = True
    auto_create_tables: bool = True

    @field_validator("cors_origins", "trusted_hosts", mode="before")
    @classmethod
    def parse_json_list(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return []
            if value.startswith("["):
                return json.loads(value)
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @property
    def api_keys(self) -> list[str]:
        return [key.strip() for key in self.ml_service_api_keys.split(",") if key.strip()]

    @property
    def model_store_path(self) -> Path:
        return Path(self.model_store_uri).expanduser().resolve()

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

from __future__ import annotations

from datetime import date
from typing import Any

import httpx
import pandas as pd

from app.core.config import get_settings


class OpenMeteoClient:
    def __init__(self) -> None:
        settings = get_settings()
        self.archive_url = settings.open_meteo_archive_url
        self.historical_forecast_url = settings.open_meteo_historical_forecast_url
        self.timeout = settings.http_timeout_seconds
        self.headers = {"User-Agent": settings.http_user_agent, "Accept": "application/json"}

    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout, headers=self.headers) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response.json()

    def fetch_archive_daily(
        self,
        *,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        payload = self._get(
            self.archive_url,
            {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "daily": ",".join(
                    [
                        "temperature_2m_max",
                        "temperature_2m_min",
                        "precipitation_sum",
                        "wind_speed_10m_max",
                        "shortwave_radiation_sum",
                    ]
                ),
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "timezone": "UTC",
            },
        )
        daily = payload.get("daily") or {}
        frame = pd.DataFrame(daily)
        if frame.empty:
            raise ValueError("Open-Meteo archive returned no daily data.")
        return frame.rename(
            columns={
                "time": "target_time",
                "temperature_2m_max": "actual_value",
                "temperature_2m_min": "actual_low",
                "precipitation_sum": "actual_precipitation",
                "wind_speed_10m_max": "actual_wind_speed",
                "shortwave_radiation_sum": "actual_solar_radiation",
            }
        )

    def fetch_historical_forecast_daily(
        self,
        *,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        payload = self._get(
            self.historical_forecast_url,
            {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "daily": ",".join(
                    [
                        "temperature_2m_max",
                        "temperature_2m_min",
                        "precipitation_probability_max",
                        "wind_speed_10m_max",
                    ]
                ),
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "timezone": "UTC",
            },
        )
        daily = payload.get("daily") or {}
        frame = pd.DataFrame(daily)
        if frame.empty:
            raise ValueError("Open-Meteo historical forecast returned no daily data.")
        return frame.rename(
            columns={
                "time": "target_time",
                "temperature_2m_max": "forecast_mean",
                "temperature_2m_min": "forecast_low",
                "precipitation_probability_max": "precipitation_probability",
                "wind_speed_10m_max": "wind_speed",
            }
        )

    def build_training_dataset(
        self,
        *,
        station_code: str,
        latitude: float,
        longitude: float,
        elevation_m: float | None,
        start_date: date,
        end_date: date,
        market_type: str,
        assumed_lead_hours: int,
    ) -> pd.DataFrame:
        actual = self.fetch_archive_daily(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
        )
        forecast = self.fetch_historical_forecast_daily(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
        )
        frame = forecast.merge(actual, on="target_time", how="inner")
        frame["target_time"] = pd.to_datetime(frame["target_time"], utc=True)
        frame["station_code"] = station_code
        frame["market_type"] = market_type
        frame["latitude"] = latitude
        frame["longitude"] = longitude
        frame["elevation_m"] = elevation_m or 0.0
        frame["lead_hours"] = assumed_lead_hours
        frame["forecast_median"] = frame["forecast_mean"]
        frame["forecast_min"] = frame["forecast_mean"] - 1.5
        frame["forecast_max"] = frame["forecast_mean"] + 1.5
        frame["forecast_std"] = 1.5
        frame["model_count"] = 1
        frame["data_age_seconds"] = assumed_lead_hours * 3600
        return frame

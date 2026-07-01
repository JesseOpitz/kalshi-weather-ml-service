from __future__ import annotations

from typing import Any

import httpx

from app.core.config import get_settings


class NWSClient:
    def __init__(self) -> None:
        settings = get_settings()
        self.base_url = settings.nws_api_url.rstrip("/")
        self.timeout = settings.http_timeout_seconds
        self.headers = {"User-Agent": settings.http_user_agent, "Accept": "application/geo+json"}

    def latest_observation(self, station_code: str) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout, headers=self.headers) as client:
            response = client.get(f"{self.base_url}/stations/{station_code}/observations/latest")
            response.raise_for_status()
            payload = response.json()
        properties = payload.get("properties", {})

        def value(name: str) -> Any:
            entry = properties.get(name)
            return entry.get("value") if isinstance(entry, dict) else entry

        return {
            "station_code": station_code,
            "observation_time": properties.get("timestamp"),
            "temperature_c": value("temperature"),
            "dew_point_c": value("dewpoint"),
            "humidity": value("relativeHumidity"),
            "wind_speed_m_s": value("windSpeed"),
            "wind_direction": value("windDirection"),
            "pressure_pa": value("barometricPressure"),
            "raw_payload": payload,
        }

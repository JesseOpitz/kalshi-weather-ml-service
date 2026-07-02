from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from app.ml.distributions import bracket_probability, clamp_probability, sigma_from_quantiles
from app.ml.features import ALL_FEATURES, NUMERIC_FEATURES


@dataclass
class ModelBundle:
    version: str
    feature_schema_version: str
    median_model: Any
    lower_model: Any
    upper_model: Any
    calibrator: IsotonicRegression | None
    residual_std: float
    feature_reference: dict[str, Any]
    supported_stations: list[str]
    supported_market_types: list[str]
    training_examples: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def frame(self, row: dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame([{column: row.get(column) for column in ALL_FEATURES}])

    def predict_value(self, row: dict[str, Any]) -> tuple[float, float, float, float]:
        frame = self.frame(row)
        median = float(self.median_model.predict(frame)[0])
        lower = float(self.lower_model.predict(frame)[0])
        upper = float(self.upper_model.predict(frame)[0])
        if lower > upper:
            lower, upper = upper, lower
        lower = min(lower, median)
        upper = max(upper, median)
        sigma = max(sigma_from_quantiles(lower, upper), self.residual_std * 0.35, 0.25)
        return median, lower, upper, sigma

    def raw_event_probability(
        self,
        row: dict[str, Any],
        lower_bound: float | None,
        upper_bound: float | None,
    ) -> float:
        median, _, _, sigma = self.predict_value(row)
        return bracket_probability(median, sigma, lower_bound, upper_bound)

    def raw_event_probability_with_sigma(
        self,
        mean: float,
        sigma: float,
        lower_bound: float | None,
        upper_bound: float | None,
    ) -> float:
        return bracket_probability(mean, sigma, lower_bound, upper_bound)

    def calibrate(self, raw_probability: float) -> float:
        if self.calibrator is None:
            return clamp_probability(raw_probability)
        calibrated = float(self.calibrator.predict([raw_probability])[0])
        return clamp_probability(calibrated)

    def predict_event(
        self,
        row: dict[str, Any],
        lower_bound: float | None,
        upper_bound: float | None,
    ) -> tuple[float, float]:
        raw = self.raw_event_probability(row, lower_bound, upper_bound)
        return raw, self.calibrate(raw)

    def explain(self, row: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
        base_prediction = self.predict_value(row)[0]
        contributions: list[dict[str, Any]] = []
        for feature in NUMERIC_FEATURES:
            current = row.get(feature)
            reference = self.feature_reference.get(feature)
            if current is None or reference is None:
                continue
            perturbed = dict(row)
            perturbed[feature] = reference
            reference_prediction = self.predict_value(perturbed)[0]
            contribution = base_prediction - reference_prediction
            if abs(contribution) < 1e-5:
                continue
            contributions.append(
                {
                    "feature": feature,
                    "contribution": round(float(contribution), 5),
                    "current_value": current,
                    "reference_value": reference,
                    "direction": "positive" if contribution > 0 else "negative",
                }
            )
        return sorted(contributions, key=lambda item: abs(item["contribution"]), reverse=True)[:limit]

    def analogues(self, row: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
        if not self.training_examples:
            return []
        scales = self.metadata.get("feature_scales", {})
        candidates: list[tuple[float, dict[str, Any]]] = []
        analogue_features = [
            "forecast_mean",
            "forecast_std",
            "lead_hours",
            "latest_temperature",
            "humidity",
            "wind_speed",
            "cloud_cover",
            "doy_sin",
            "doy_cos",
        ]
        for example in self.training_examples:
            distance = 0.0
            count = 0
            for feature in analogue_features:
                left = row.get(feature)
                right = example.get(feature)
                if left is None or right is None:
                    continue
                try:
                    scale = max(float(scales.get(feature, 1.0)), 1e-6)
                    distance += ((float(left) - float(right)) / scale) ** 2
                    count += 1
                except (TypeError, ValueError):
                    continue
            if count:
                candidates.append((float(np.sqrt(distance / count)), example))
        candidates.sort(key=lambda item: item[0])
        return [
            {
                "distance": round(distance, 4),
                "target_time": example.get("target_time"),
                "station_code": example.get("station_code"),
                "forecast_mean": example.get("forecast_mean"),
                "actual_value": example.get("actual_value"),
            }
            for distance, example in candidates[:limit]
        ]

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.ml.bundle import ModelBundle
from app.ml.distributions import bracket_probability
from app.ml.features import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    TIME_COLUMN,
    normalize_training_frame,
)
from app.ml.metrics import probability_metrics, regression_metrics


@dataclass
class TrainingResult:
    bundle: ModelBundle
    metrics: dict[str, Any]
    date_start: datetime
    date_end: datetime


def _pipeline(*, loss: str, quantile: float | None, config: dict[str, Any]) -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessing = ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)],
        remainder="drop",
    )
    estimator_kwargs: dict[str, Any] = {
        "loss": loss,
        "learning_rate": config.get("learning_rate", 0.05),
        "n_estimators": config.get("n_estimators", config.get("max_iter", 300)),
        "max_depth": config.get("max_depth", 3),
        "min_samples_leaf": config.get("min_samples_leaf", 20),
        "subsample": config.get("subsample", 0.9),
        "random_state": config.get("random_state", 42),
    }
    if quantile is not None:
        estimator_kwargs["alpha"] = quantile
    estimator = GradientBoostingRegressor(**estimator_kwargs)
    return Pipeline([("preprocessing", preprocessing), ("model", estimator)])


def _time_splits(size: int, splits: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    minimum_train = max(int(size * 0.35), 50)
    remaining = size - minimum_train
    if remaining < 20:
        return []
    fold_size = max(remaining // splits, 10)
    result = []
    train_end = minimum_train
    while train_end < size - 5 and len(result) < splits:
        validation_end = min(train_end + fold_size, size)
        result.append((np.arange(0, train_end), np.arange(train_end, validation_end)))
        train_end = validation_end
    return result


def _calibration_bounds(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if "lower_bound" in frame.columns and "upper_bound" in frame.columns:
        lower = pd.to_numeric(frame["lower_bound"], errors="coerce").to_numpy(dtype=float)
        upper = pd.to_numeric(frame["upper_bound"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(lower) & np.isfinite(upper) & (lower < upper)
        if valid.mean() > 0.8:
            return lower, upper
    forecast = frame["forecast_mean"].to_numpy(dtype=float)
    lower = np.floor(forecast / 2.0) * 2.0
    upper = lower + 2.0
    return lower, upper


def train_model_bundle(
    frame: pd.DataFrame,
    *,
    version: str,
    config: dict[str, Any],
) -> TrainingResult:
    frame = normalize_training_frame(frame)
    if len(frame) < 100:
        raise ValueError("At least 100 valid rows are required to train a model.")

    test_size = max(int(len(frame) * 0.20), 20)
    train_frame = frame.iloc[:-test_size].reset_index(drop=True)
    test_frame = frame.iloc[-test_size:].reset_index(drop=True)
    x_train = train_frame[ALL_FEATURES]
    y_train = train_frame[TARGET_COLUMN].to_numpy(dtype=float)
    x_test = test_frame[ALL_FEATURES]
    y_test = test_frame[TARGET_COLUMN].to_numpy(dtype=float)

    median_template = _pipeline(loss="squared_error", quantile=None, config=config)
    lower_template = _pipeline(loss="quantile", quantile=0.10, config=config)
    upper_template = _pipeline(loss="quantile", quantile=0.90, config=config)

    oof_prediction = np.full(len(train_frame), np.nan)
    for train_idx, validation_idx in _time_splits(len(train_frame)):
        fold_median = clone(median_template)
        fold_median.fit(x_train.iloc[train_idx], y_train[train_idx])
        oof_prediction[validation_idx] = fold_median.predict(x_train.iloc[validation_idx])

    valid_oof = np.isfinite(oof_prediction)
    residuals = y_train[valid_oof] - oof_prediction[valid_oof]
    residual_std = float(np.std(residuals)) if len(residuals) else float(np.std(y_train))
    residual_std = max(residual_std, 0.5)

    calibration_frame = train_frame.loc[valid_oof].reset_index(drop=True)
    cal_lower, cal_upper = _calibration_bounds(calibration_frame)
    raw_probabilities: list[float] = []
    outcomes: list[int] = []
    oof_positions = np.where(valid_oof)[0]
    for local_index, global_index in enumerate(oof_positions):
        sigma = max(residual_std, 0.25)
        raw_probabilities.append(
            bracket_probability(
                oof_prediction[global_index],
                sigma,
                float(cal_lower[local_index]),
                float(cal_upper[local_index]),
            )
        )
        actual = float(y_train[global_index])
        outcomes.append(int(cal_lower[local_index] <= actual < cal_upper[local_index]))

    calibrator: IsotonicRegression | None = None
    if (
        config.get("calibration_method", "isotonic") == "isotonic"
        and len(raw_probabilities) >= 50
        and len(set(outcomes)) > 1
    ):
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
        calibrator.fit(raw_probabilities, outcomes)

    median_model = clone(median_template).fit(x_train, y_train)
    lower_model = clone(lower_template).fit(x_train, y_train)
    upper_model = clone(upper_template).fit(x_train, y_train)

    test_prediction = median_model.predict(x_test)
    test_lower = lower_model.predict(x_test)
    test_upper = upper_model.predict(x_test)
    reg_metrics = regression_metrics(y_test, test_prediction)
    coverage = float(
        np.mean(
            (y_test >= np.minimum(test_lower, test_upper))
            & (y_test <= np.maximum(test_lower, test_upper))
        )
    )

    test_bounds_lower, test_bounds_upper = _calibration_bounds(test_frame)
    test_raw: list[float] = []
    test_calibrated: list[float] = []
    test_outcomes: list[int] = []
    for index in range(len(test_frame)):
        sigma = max(
            (max(test_upper[index], test_prediction[index]) - min(test_lower[index], test_prediction[index]))
            / 2.563103131,
            residual_std * 0.35,
            0.25,
        )
        raw = bracket_probability(
            float(test_prediction[index]),
            sigma,
            float(test_bounds_lower[index]),
            float(test_bounds_upper[index]),
        )
        calibrated = float(calibrator.predict([raw])[0]) if calibrator else raw
        actual = float(y_test[index])
        test_raw.append(raw)
        test_calibrated.append(calibrated)
        test_outcomes.append(int(test_bounds_lower[index] <= actual < test_bounds_upper[index]))

    metrics = {
        **reg_metrics,
        "quantile_interval_80_coverage": coverage,
        "raw_probability": probability_metrics(test_outcomes, test_raw),
        "calibrated_probability": probability_metrics(test_outcomes, test_calibrated),
        "residual_std": residual_std,
        "training_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
    }

    feature_reference: dict[str, Any] = {}
    feature_scales: dict[str, float] = {}
    for feature in NUMERIC_FEATURES:
        values = pd.to_numeric(train_frame[feature], errors="coerce")
        feature_reference[feature] = float(values.median()) if values.notna().any() else None
        scale = float(values.std()) if values.notna().sum() > 1 else 1.0
        feature_scales[feature] = scale if math.isfinite(scale) and scale > 1e-6 else 1.0

    sample_size = min(1200, len(train_frame))
    training_examples = (
        train_frame.sample(sample_size, random_state=config.get("random_state", 42))[
            [TIME_COLUMN, "station_code", "forecast_mean", TARGET_COLUMN]
            + [feature for feature in NUMERIC_FEATURES if feature != "forecast_mean"]
        ]
        .rename(columns={TIME_COLUMN: "target_time"})
        .assign(target_time=lambda data: data["target_time"].astype(str))
        .replace({np.nan: None})
        .to_dict(orient="records")
    )

    bundle = ModelBundle(
        version=version,
        feature_schema_version="weather-features-v1",
        median_model=median_model,
        lower_model=lower_model,
        upper_model=upper_model,
        calibrator=calibrator,
        residual_std=residual_std,
        feature_reference=feature_reference,
        supported_stations=sorted(train_frame["station_code"].dropna().astype(str).unique().tolist()),
        supported_market_types=sorted(train_frame["market_type"].dropna().astype(str).unique().tolist()),
        training_examples=training_examples,
        metrics=metrics,
        metadata={
            "trained_at": datetime.now(UTC).isoformat(),
            "feature_scales": feature_scales,
            "temperature_unit": config.get("temperature_unit", "F"),
            "calibration_method": config.get("calibration_method", "isotonic"),
        },
    )
    return TrainingResult(
        bundle=bundle,
        metrics=metrics,
        date_start=frame[TIME_COLUMN].min().to_pydatetime(),
        date_end=frame[TIME_COLUMN].max().to_pydatetime(),
    )

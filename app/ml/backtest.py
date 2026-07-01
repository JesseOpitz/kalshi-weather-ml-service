from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from app.ml.bundle import ModelBundle
from app.ml.distributions import binary_log_loss
from app.ml.features import ALL_FEATURES, TARGET_COLUMN, normalize_training_frame


def run_backtest(
    bundle: ModelBundle,
    frame: pd.DataFrame,
    configuration: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frame = normalize_training_frame(frame)
    if frame.empty:
        raise ValueError("Backtest dataset has no valid rows.")

    minimum_net_edge = float(configuration.get("minimum_net_edge", 0.03))
    fee_multiplier = float(configuration.get("fee_multiplier", 0.07))
    slippage = float(configuration.get("slippage_per_contract", 0.01))
    uncertainty = float(configuration.get("uncertainty_buffer", 0.01))
    contracts = int(configuration.get("contracts_per_trade", 1))
    max_exposure = float(configuration.get("maximum_total_exposure", 1000.0))

    trades: list[dict[str, Any]] = []
    predictions: list[float] = []
    actuals: list[float] = []
    probabilities: list[float] = []
    outcomes: list[int] = []
    cumulative_pnl = 0.0
    peak_pnl = 0.0
    maximum_drawdown = 0.0
    active_exposure = 0.0

    has_market_fields = {"market_yes_ask", "lower_bound", "upper_bound"}.issubset(frame.columns)

    for _, row_series in frame.iterrows():
        row = {column: row_series.get(column) for column in ALL_FEATURES}
        predicted, q10, q90, sigma = bundle.predict_value(row)
        actual = float(row_series[TARGET_COLUMN])
        predictions.append(predicted)
        actuals.append(actual)

        if not has_market_fields:
            continue
        ask = row_series.get("market_yes_ask")
        lower = row_series.get("lower_bound")
        upper = row_series.get("upper_bound")
        if pd.isna(ask) or (pd.isna(lower) and pd.isna(upper)):
            continue
        ask = float(ask)
        lower_value = None if pd.isna(lower) else float(lower)
        upper_value = None if pd.isna(upper) else float(upper)
        raw, probability = bundle.predict_event(row, lower_value, upper_value)
        outcome = int(
            (lower_value is None or actual >= lower_value)
            and (upper_value is None or actual < upper_value)
        )
        probabilities.append(probability)
        outcomes.append(outcome)
        fee = fee_multiplier * ask * (1.0 - ask)
        net_edge = probability - ask - fee - slippage - uncertainty
        requested_exposure = ask * contracts
        if net_edge < minimum_net_edge or active_exposure + requested_exposure > max_exposure:
            continue
        pnl_per_contract = outcome - ask - fee - slippage
        pnl = pnl_per_contract * contracts
        cumulative_pnl += pnl
        peak_pnl = max(peak_pnl, cumulative_pnl)
        maximum_drawdown = max(maximum_drawdown, peak_pnl - cumulative_pnl)
        active_exposure += requested_exposure
        trades.append(
            {
                "target_time": str(row_series.get("target_time")),
                "station_code": row_series.get("station_code"),
                "predicted_value": round(predicted, 4),
                "actual_value": actual,
                "raw_probability": round(raw, 6),
                "calibrated_probability": round(probability, 6),
                "market_yes_ask": ask,
                "estimated_fee": round(fee, 6),
                "slippage": slippage,
                "uncertainty_buffer": uncertainty,
                "net_edge": round(net_edge, 6),
                "contracts": contracts,
                "event_outcome": outcome,
                "pnl": round(pnl, 6),
                "cumulative_pnl": round(cumulative_pnl, 6),
                "prediction_interval": [round(q10, 4), round(q90, 4)],
                "sigma": round(sigma, 4),
            }
        )

    errors = np.asarray(predictions) - np.asarray(actuals)
    metrics: dict[str, Any] = {
        "rows": len(frame),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "bias": float(np.mean(errors)),
        "trades": len(trades),
        "total_pnl": round(cumulative_pnl, 6),
        "maximum_drawdown": round(maximum_drawdown, 6),
        "win_rate": (
            float(np.mean([trade["pnl"] > 0 for trade in trades])) if trades else 0.0
        ),
        "average_net_edge": (
            float(np.mean([trade["net_edge"] for trade in trades])) if trades else 0.0
        ),
        "total_fees": round(sum(trade["estimated_fee"] * trade["contracts"] for trade in trades), 6),
        "total_slippage": round(sum(trade["slippage"] * trade["contracts"] for trade in trades), 6),
    }
    if probabilities:
        probability_array = np.asarray(probabilities)
        outcome_array = np.asarray(outcomes)
        metrics.update(
            {
                "brier_score": float(np.mean((probability_array - outcome_array) ** 2)),
                "log_loss": float(
                    np.mean(
                        [
                            binary_log_loss(probability, int(outcome))
                            for probability, outcome in zip(probability_array, outcome_array, strict=True)
                        ]
                    )
                ),
                "probability_samples": len(probabilities),
            }
        )
    else:
        metrics.update({"brier_score": math.nan, "log_loss": math.nan, "probability_samples": 0})
    return metrics, trades

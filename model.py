from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from app.ml.distributions import binary_log_loss


def regression_metrics(actual: Iterable[float], predicted: Iterable[float]) -> dict[str, float]:
    y = np.asarray(list(actual), dtype=float)
    p = np.asarray(list(predicted), dtype=float)
    error = p - y
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "sample_count": int(len(y)),
    }


def probability_metrics(actual: Iterable[int], probabilities: Iterable[float]) -> dict[str, float]:
    y = np.asarray(list(actual), dtype=int)
    p = np.asarray(list(probabilities), dtype=float)
    if len(y) == 0:
        return {"brier_score": math.nan, "log_loss": math.nan, "sample_count": 0}
    return {
        "brier_score": float(np.mean((p - y) ** 2)),
        "log_loss": float(np.mean([binary_log_loss(prob, int(outcome)) for prob, outcome in zip(p, y, strict=True)])),
        "sample_count": int(len(y)),
    }

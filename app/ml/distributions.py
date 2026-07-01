from __future__ import annotations

import math
from statistics import NormalDist

_STANDARD_NORMAL = NormalDist()


def clamp_probability(value: float, epsilon: float = 1e-6) -> float:
    return min(1.0 - epsilon, max(epsilon, float(value)))


def normal_cdf(value: float, mean: float, sigma: float) -> float:
    sigma = max(float(sigma), 1e-6)
    return _STANDARD_NORMAL.cdf((float(value) - float(mean)) / sigma)


def bracket_probability(
    mean: float,
    sigma: float,
    lower_bound: float | None,
    upper_bound: float | None,
) -> float:
    if lower_bound is None and upper_bound is None:
        raise ValueError("At least one bound is required.")
    lower_cdf = 0.0 if lower_bound is None else normal_cdf(lower_bound, mean, sigma)
    upper_cdf = 1.0 if upper_bound is None else normal_cdf(upper_bound, mean, sigma)
    return clamp_probability(max(0.0, upper_cdf - lower_cdf))


def sigma_from_quantiles(lower: float, upper: float, z: float = 1.2815515655446004) -> float:
    if upper <= lower:
        return 1.0
    return max((upper - lower) / (2.0 * z), 0.25)


def binary_log_loss(probability: float, outcome: int) -> float:
    p = clamp_probability(probability)
    y = 1 if outcome else 0
    return -(y * math.log(p) + (1 - y) * math.log(1.0 - p))

from pathlib import Path

def write(path: str, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding='utf-8')

write('app/ml/distributions.py', 'from __future__ import annotations\n\nimport math\nfrom statistics import NormalDist\n\n_STANDARD_NORMAL = NormalDist()\n\n\ndef clamp_probability(value: float, epsilon: float = 1e-6) -> float:\n    return min(1.0 - epsilon, max(epsilon, float(value)))\n\n\ndef normal_cdf(value: float, mean: float, sigma: float) -> float:\n    sigma = max(float(sigma), 1e-6)\n    return _STANDARD_NORMAL.cdf((float(value) - float(mean)) / sigma)\n\n\ndef bracket_probability(\n    mean: float,\n    sigma: float,\n    lower_bound: float | None,\n    upper_bound: float | None,\n) -> float:\n    if lower_bound is None and upper_bound is None:\n        raise ValueError("At least one bound is required.")\n    lower_cdf = 0.0 if lower_bound is None else normal_cdf(lower_bound, mean, sigma)\n    upper_cdf = 1.0 if upper_bound is None else normal_cdf(upper_bound, mean, sigma)\n    return clamp_probability(max(0.0, upper_cdf - lower_cdf))\n\n\ndef sigma_from_quantiles(lower: float, upper: float, z: float = 1.2815515655446004) -> float:\n    if upper <= lower:\n        return 1.0\n    return max((upper - lower) / (2.0 * z), 0.25)\n\n\ndef binary_log_loss(probability: float, outcome: int) -> float:\n    p = clamp_probability(probability)\n    y = 1 if outcome else 0\n    return -(y * math.log(p) + (1 - y) * math.log(1.0 - p))\n')

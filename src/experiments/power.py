"""Sample size for a two-proportion (or two-mean) experiment.

Computed before the test runs. Do not peek at p-values to stop early.
"""

from __future__ import annotations

import math
from typing import Dict

from scipy import stats


def two_proportion_sample_size(
    baseline_rate: float,
    mde_relative: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> Dict[str, float]:
    if not 0 < baseline_rate < 1:
        raise ValueError("baseline_rate must be in (0, 1)")
    if mde_relative <= 0:
        raise ValueError("mde_relative must be positive")
    p1 = baseline_rate
    p2 = baseline_rate * (1.0 + mde_relative)
    p2 = min(p2, 1.0 - 1e-12)
    z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z_power = float(stats.norm.ppf(power))
    pooled = 0.5 * (p1 + p2)
    numer = (z_alpha * math.sqrt(2 * pooled * (1 - pooled)) + z_power * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    denom = (p2 - p1) ** 2
    n_per_arm = math.ceil(numer / denom)
    return {
        "baseline_rate": p1,
        "treatment_rate": p2,
        "alpha": alpha,
        "power": power,
        "mde_relative": mde_relative,
        "n_per_arm": float(n_per_arm),
        "n_total": float(2 * n_per_arm),
    }

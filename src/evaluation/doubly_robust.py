"""Doubly robust off-policy evaluation.

Combines an outcome model m(x, action) with a propensity-weighted residual.
The estimator is consistent if either the propensity or the outcome model
is correctly specified.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable, Dict

import numpy as np

from evaluation.ips import ips_report

OutcomeModel = Callable[[object, object], float]


def doubly_robust_value(
    outcomes: Sequence[float],
    propensities: Sequence[float],
    predicted_outcomes: Sequence[float],
    max_weight: float = 100.0,
) -> float:
    y = np.asarray(outcomes, dtype=float)
    p = np.asarray(propensities, dtype=float)
    m = np.asarray(predicted_outcomes, dtype=float)
    if y.size == 0:
        return 0.0
    if np.any(p <= 0):
        raise ValueError("all propensities must be positive")
    weights = np.minimum(1.0 / p, max_weight)
    residual = y - m
    return float(np.mean(m + weights * residual))


def compare_estimators(
    outcomes: Sequence[float],
    propensities: Sequence[float],
    predicted_outcomes: Sequence[float],
    max_weight: float = 100.0,
) -> Dict[str, float]:
    report = ips_report(outcomes, propensities, max_weight=max_weight)
    report["doubly_robust"] = doubly_robust_value(
        outcomes, propensities, predicted_outcomes, max_weight=max_weight
    )
    report["outcome_model_mean"] = float(np.mean(predicted_outcomes)) if len(predicted_outcomes) else 0.0
    return report

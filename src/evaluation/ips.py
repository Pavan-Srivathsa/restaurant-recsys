"""Inverse propensity scoring for logged recommendations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Dict, Tuple

import numpy as np


def ips_weight(propensity: float, max_weight: float = 100.0) -> float:
    if propensity <= 0:
        raise ValueError("propensity must be positive")
    return min(1.0 / propensity, max_weight)


def ips_weights(
    propensities: Sequence[float],
    max_weight: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray]:
    p = np.asarray(propensities, dtype=float)
    if np.any(p <= 0):
        raise ValueError("all propensities must be positive")
    raw = 1.0 / p
    clipped = np.minimum(raw, max_weight)
    return raw, clipped


def effective_sample_size(weights: Sequence[float]) -> float:
    w = np.asarray(weights, dtype=float)
    denom = float(np.sum(w ** 2))
    if denom <= 0:
        return 0.0
    return float(np.sum(w) ** 2 / denom)


def policy_value(
    outcomes: Sequence[float],
    weights: Sequence[float],
) -> float:
    y = np.asarray(outcomes, dtype=float)
    w = np.asarray(weights, dtype=float)
    if y.size == 0:
        return 0.0
    return float(np.sum(w * y) / np.sum(w))


def ips_report(
    outcomes: Sequence[float],
    propensities: Sequence[float],
    max_weight: float = 100.0,
) -> Dict[str, float]:
    y = np.asarray(outcomes, dtype=float)
    raw, clipped = ips_weights(propensities, max_weight=max_weight)
    unweighted = float(np.mean(y)) if y.size else 0.0
    return {
        "unweighted": unweighted,
        "ips": policy_value(y, raw),
        "clipped_ips": policy_value(y, clipped),
        "effective_sample_size": effective_sample_size(clipped),
        "n": float(y.size),
        "max_weight": float(max_weight),
    }

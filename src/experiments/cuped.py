"""CUPED variance reduction.

Y = experiment-period outcome
X = pre-experiment reservation behavior
θ = Cov(Y, X) / Var(X)
Y_cuped = Y − θ(X − mean(X))
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Dict

import numpy as np


def cuped_theta(y: Sequence[float], x: Sequence[float]) -> float:
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    if y_arr.size != x_arr.size:
        raise ValueError("y and x must be the same length")
    if y_arr.size < 2:
        return 0.0
    var_x = float(np.var(x_arr, ddof=1))
    if var_x <= 0:
        return 0.0
    cov = float(np.cov(y_arr, x_arr, ddof=1)[0, 1])
    return cov / var_x


def cuped_adjust(y: Sequence[float], x: Sequence[float]) -> np.ndarray:
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    theta = cuped_theta(y_arr, x_arr)
    return y_arr - theta * (x_arr - float(np.mean(x_arr)))


def cuped_report(y: Sequence[float], x: Sequence[float]) -> Dict[str, float]:
    y_arr = np.asarray(y, dtype=float)
    adjusted = cuped_adjust(y_arr, x)
    var_y = float(np.var(y_arr, ddof=1)) if y_arr.size > 1 else 0.0
    var_adj = float(np.var(adjusted, ddof=1)) if adjusted.size > 1 else 0.0
    reduction = (var_y - var_adj) / var_y if var_y > 0 else 0.0
    return {
        "theta": cuped_theta(y, x),
        "variance_y": var_y,
        "variance_y_cuped": var_adj,
        "variance_reduction": reduction,
        "mean_y": float(np.mean(y_arr)) if y_arr.size else 0.0,
        "mean_y_cuped": float(np.mean(adjusted)) if adjusted.size else 0.0,
    }

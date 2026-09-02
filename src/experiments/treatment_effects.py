"""Average and heterogeneous treatment effects.

Primary metric: completed reservations per exposed user.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Dict

import numpy as np
from scipy import stats


def ate(control: Sequence[float], treatment: Sequence[float]) -> Dict[str, float]:
    c = np.asarray(control, dtype=float)
    t = np.asarray(treatment, dtype=float)
    if c.size == 0 or t.size == 0:
        raise ValueError("control and treatment must be non-empty")
    mean_c = float(np.mean(c))
    mean_t = float(np.mean(t))
    abs_lift = mean_t - mean_c
    rel_lift = abs_lift / mean_c if mean_c != 0 else 0.0
    se = float(np.sqrt(np.var(c, ddof=1) / c.size + np.var(t, ddof=1) / t.size))
    df = c.size + t.size - 2
    t_stat = abs_lift / se if se > 0 else 0.0
    p_value = float(2 * stats.t.sf(abs(t_stat), df)) if se > 0 else 1.0
    ci_half = float(stats.t.ppf(0.975, df) * se) if se > 0 else 0.0
    return {
        "control_mean": mean_c,
        "treatment_mean": mean_t,
        "absolute_lift": abs_lift,
        "relative_lift": rel_lift,
        "std_error": se,
        "p_value": p_value,
        "ci_low": abs_lift - ci_half,
        "ci_high": abs_lift + ci_half,
        "n_control": float(c.size),
        "n_treatment": float(t.size),
    }


def interaction_ols(
    y: Sequence[float],
    treatment: Sequence[int],
    returning_user: Sequence[int],
) -> Dict[str, float]:
    """reservation ~ treatment + returning_user + treatment × returning_user."""
    y_arr = np.asarray(y, dtype=float)
    t = np.asarray(treatment, dtype=float)
    r = np.asarray(returning_user, dtype=float)
    if y_arr.size != t.size or y_arr.size != r.size:
        raise ValueError("arrays must be the same length")
    n = y_arr.size
    x = np.column_stack([np.ones(n), t, r, t * r])
    # Ridge + pinv: treatment × returning is near-collinear when most users are returning.
    xtx = x.T @ x
    ridge = 1e-6 * np.eye(x.shape[1]) * max(float(np.trace(xtx) / x.shape[1]), 1.0)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        xtx_inv = np.linalg.pinv(xtx + ridge)
        beta = xtx_inv @ (x.T @ y_arr)
        fitted = x @ beta
    beta = np.asarray(beta, dtype=float)
    fitted = np.asarray(fitted, dtype=float)
    if not np.all(np.isfinite(beta)):
        beta = np.linalg.lstsq(x, y_arr, rcond=1e-4)[0]
        fitted = x @ beta
    resid = y_arr - fitted
    df = max(n - 4, 1)
    sigma2 = float(np.sum(np.square(np.nan_to_num(resid))) / df)
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))
    names = ("intercept", "treatment", "returning_user", "treatment_x_returning")
    out: Dict[str, float] = {}
    for i, name in enumerate(names):
        out[name] = float(beta[i])
        out[f"{name}_se"] = float(se[i])
        t_stat = float(beta[i] / se[i]) if se[i] > 0 else 0.0
        out[f"{name}_p"] = float(2 * stats.t.sf(abs(t_stat), df)) if se[i] > 0 else 1.0
    return out

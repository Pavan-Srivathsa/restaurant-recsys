"""Off-policy evaluation of the personalized ranker from popularity logs.

Estimates expected booking rate of the target top-10 using:
- naive overlap (unweighted outcomes on items both policies show)
- IPS / clipped IPS
- doubly robust (outcome model + propensity-weighted residual)

Oracle booking rate on synthetic data is the full-information target.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

from data.ingest import N_SHOW
from evaluation.ips import effective_sample_size, policy_value
from evaluation.propensities import attach_propensities, clip_propensity
from ranking.predict import scores_from_model
from ranking.train import FEATURE_COLUMNS


def booking_outcome(labels: Sequence) -> np.ndarray:
    return (np.asarray(labels, dtype=float) >= 2).astype(float)


def train_outcome_model(train: pd.DataFrame, seed: int = 7):
    """m(x, action) = P(booking | features). Fit only on logged train rows."""
    logged = train[train["logged"]].copy()
    y = booking_outcome(logged["label"])
    x = logged[list(FEATURE_COLUMNS)].to_numpy(dtype=float)

    class _ProbModel:
        def __init__(self, predict_fn) -> None:
            self._predict_fn = predict_fn

        def predict_proba(self, x_new):
            p = np.clip(np.asarray(self._predict_fn(x_new), dtype=float).reshape(-1), 0.0, 1.0)
            return np.column_stack([1.0 - p, p])

    if y.size == 0 or len(np.unique(y)) < 2:
        p = float(np.mean(y)) if y.size else 0.0
        return _ProbModel(lambda x_new: np.full(len(x_new), p))

    dataset = lgb.Dataset(x, label=y, feature_name=list(FEATURE_COLUMNS))
    booster = lgb.train(
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_data_in_leaf": max(5, min(20, len(logged) // 15)),
            "feature_fraction": 0.9,
            "verbosity": -1,
            "seed": seed,
        },
        dataset,
        num_boost_round=80,
    )
    return _ProbModel(lambda x_new: booster.predict(x_new))


def predict_booking(model, frame: pd.DataFrame) -> np.ndarray:
    x = frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    proba = model.predict_proba(x)
    return np.asarray(proba[:, 1], dtype=float)


def _target_top_k(group: pd.DataFrame, scores: Sequence[float], k: int = N_SHOW) -> List[str]:
    paired = list(zip(list(group["restaurant_id"]), scores))
    paired.sort(key=lambda item: (-item[1], item[0]))
    return [rid for rid, _ in paired[:k]]


def _abs_err(estimate: float, oracle: float) -> float:
    return abs(estimate - oracle)


def evaluate_policy(
    test: pd.DataFrame,
    model,
    outcome_model,
    k: int = N_SHOW,
    max_weight: float = 100.0,
    min_p: float = 0.001,
) -> Dict[str, float]:
    """OPE of the personalized ranker on temporally held-out logs."""
    dm_vals: List[float] = []
    ips_raw_vals: List[float] = []
    ips_clip_vals: List[float] = []
    dr_vals: List[float] = []
    oracle_vals: List[float] = []
    naive_overlap_ys: List[float] = []
    on_policy_ys: List[float] = []
    all_raw_w: List[float] = []
    all_clip_w: List[float] = []
    all_y_w: List[float] = []
    all_exp_w: List[float] = []
    all_exp_y: List[float] = []

    for _, group in test.groupby("request_id", sort=False):
        group = attach_propensities(group.reset_index(drop=True), min_p=min_p)
        feats = group[list(FEATURE_COLUMNS)].to_dict(orient="records")
        scores = scores_from_model(model, feats)
        target = set(_target_top_k(group, scores, k=k))
        m_hat = predict_booking(outcome_model, group)
        y = booking_outcome(group["label"])
        oracle = booking_outcome(group["oracle_label"])
        ids = list(group["restaurant_id"])
        logged = group["logged"].to_numpy(dtype=bool)
        display_p = group["display_p"].to_numpy(dtype=float)
        exposure_p = group["exposure_p"].to_numpy(dtype=float)

        target_idx = [i for i, rid in enumerate(ids) if rid in target]
        dm = float(np.mean(m_hat[target_idx])) if target_idx else 0.0
        oracle_v = float(np.mean(oracle[target_idx])) if target_idx else 0.0

        raw_sum = 0.0
        clip_sum = 0.0
        dr_corr = 0.0
        for i, rid in enumerate(ids):
            if not logged[i]:
                continue
            on_policy_ys.append(float(y[i]))
            in_target = rid in target
            if in_target:
                naive_overlap_ys.append(float(y[i]))
            indicator = 1.0 if in_target else 0.0

            p_disp = float(display_p[i])
            if p_disp > 0:
                p_disp_clip = clip_propensity(p_disp, min_p=min_p)
                raw_w = indicator / p_disp
                clip_w = min(indicator / p_disp_clip, max_weight)
                all_raw_w.append(raw_w)
                all_clip_w.append(clip_w)
                all_y_w.append(float(y[i]))
                raw_sum += raw_w * y[i]
                clip_sum += clip_w * y[i]
                dr_corr += clip_w * (y[i] - m_hat[i])

            p_exp = float(exposure_p[i])
            if p_exp > 0:
                p_exp_clip = clip_propensity(p_exp, min_p=min_p)
                exp_w = min(indicator / p_exp_clip, max_weight)
                all_exp_w.append(exp_w)
                all_exp_y.append(float(y[i]))

        ips_raw_vals.append(raw_sum / float(k))
        ips_clip_vals.append(clip_sum / float(k))
        dm_vals.append(dm)
        dr_vals.append(dm + dr_corr / float(k))
        oracle_vals.append(oracle_v)

    def _mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    raw_w = np.asarray(all_raw_w, dtype=float)
    clip_w = np.asarray(all_clip_w, dtype=float)
    y_w = np.asarray(all_y_w, dtype=float)
    exp_w = np.asarray(all_exp_w, dtype=float)
    exp_y = np.asarray(all_exp_y, dtype=float)
    naive = float(np.mean(naive_overlap_ys)) if naive_overlap_ys else 0.0
    oracle = _mean(oracle_vals)
    ips = _mean(ips_raw_vals)
    clipped = _mean(ips_clip_vals)
    dr = _mean(dr_vals)
    dm = _mean(dm_vals)
    snips = policy_value(y_w, clip_w) if clip_w.size else 0.0
    exposure_snips = policy_value(exp_y, exp_w) if exp_w.size else 0.0
    return {
        "naive_overlap": naive,
        "logging_on_policy": float(np.mean(on_policy_ys)) if on_policy_ys else 0.0,
        "direct_method": dm,
        "ips": ips,
        "clipped_ips": clipped,
        "doubly_robust": dr,
        "snips": snips,
        "exposure_snips": exposure_snips,
        "oracle": oracle,
        "abs_error_naive": _abs_err(naive, oracle),
        "abs_error_ips": _abs_err(ips, oracle),
        "abs_error_clipped_ips": _abs_err(clipped, oracle),
        "abs_error_dr": _abs_err(dr, oracle),
        "abs_error_dm": _abs_err(dm, oracle),
        "abs_error_snips": _abs_err(snips, oracle),
        "abs_error_exposure_snips": _abs_err(exposure_snips, oracle),
        "effective_sample_size": effective_sample_size(clip_w) if clip_w.size else 0.0,
        "exposure_ess": effective_sample_size(exp_w) if exp_w.size else 0.0,
        "n_logged_weights": float(clip_w.size),
        "n_requests": float(len(oracle_vals)),
        "max_weight": float(max_weight),
        "mean_raw_weight": float(np.mean(raw_w)) if raw_w.size else 0.0,
        "mean_clipped_weight": float(np.mean(clip_w)) if clip_w.size else 0.0,
        "max_raw_weight": float(np.max(raw_w)) if raw_w.size else 0.0,
        "max_exposure_weight": float(np.max(exp_w)) if exp_w.size else 0.0,
    }

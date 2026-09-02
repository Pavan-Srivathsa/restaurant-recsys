"""LambdaMART training interface (LightGBM).

Training rows must be grouped by recommendation request. Features must be
computed as-of the request timestamp.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

FEATURE_COLUMNS = (
    "user_activity_count",
    "user_booking_count",
    "user_average_price",
    "restaurant_rating",
    "restaurant_popularity",
    "restaurant_price",
    "reservation_availability",
    "cuisine_affinity",
    "price_match",
    "distance_km",
    "previous_visits",
    "previous_clicks",
    "previous_bookings",
    "hour",
    "weekday",
    "weekend",
    "meal_period",
)


def group_sizes(request_ids: Sequence[str]) -> list:
    """LightGBM query groups: contiguous counts per request_id."""
    if not request_ids:
        return []
    sizes = []
    current = request_ids[0]
    count = 0
    for rid in request_ids:
        if rid != current:
            sizes.append(count)
            current = rid
            count = 0
        count += 1
    sizes.append(count)
    return sizes


def lgbm_ranker_params(seed: int = 7, min_data_in_leaf: int = 20) -> dict:
    return {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": min_data_in_leaf,
        "feature_fraction": 0.9,
        "deterministic": True,
        "force_row_wise": True,
        "seed": seed,
        "verbosity": -1,
    }


def train_lambdamart(
    frame: Any,
    valid_frame: Any = None,
    seed: int = 7,
    num_boost_round: int = 200,
    min_data_in_leaf: int = 20,
) -> Any:
    """Train a LightGBM LambdaMART model.

    `frame` must provide columns in FEATURE_COLUMNS, plus `label` and `request_id`,
    sorted so that rows for the same request_id are contiguous.
    """
    import lightgbm as lgb
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    params = lgbm_ranker_params(seed=seed, min_data_in_leaf=min_data_in_leaf)
    x = frame[list(FEATURE_COLUMNS)]
    y = frame["label"]
    groups = group_sizes(list(frame["request_id"]))
    dataset = lgb.Dataset(x, label=y, group=groups, feature_name=list(FEATURE_COLUMNS), params=params)
    valid_sets = [dataset]
    valid_names = ["train"]
    callbacks = [lgb.log_evaluation(period=0)]
    if valid_frame is not None and len(valid_frame) > 0:
        vx = valid_frame[list(FEATURE_COLUMNS)]
        vy = valid_frame["label"]
        vg = group_sizes(list(valid_frame["request_id"]))
        valid = lgb.Dataset(vx, label=vy, group=vg, feature_name=list(FEATURE_COLUMNS), params=params)
        valid_sets.append(valid)
        valid_names.append("valid")
        callbacks.append(lgb.early_stopping(stopping_rounds=30, verbose=False))
    return lgb.train(
        params,
        dataset,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )


def save_model(model: Any, path) -> None:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))


def load_model(path) -> Any:
    import lightgbm as lgb

    return lgb.Booster(model_file=str(path))

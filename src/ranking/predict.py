"""Score and rank candidates with a trained model or the contextual baseline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, List, Tuple

from data.schemas import Restaurant
from ranking.baseline import rank_baseline
from ranking.train import FEATURE_COLUMNS


def scores_from_model(model: Any, feature_rows: Sequence[Mapping[str, float]]) -> List[float]:
    import numpy as np

    matrix = np.array([[row[c] for c in FEATURE_COLUMNS] for row in feature_rows], dtype=float)
    raw = model.predict(matrix)
    return [float(s) for s in raw]


def rank_by_scores(
    restaurants: Sequence[Restaurant],
    scores: Sequence[float],
    top_k: int = 10,
) -> List[Tuple[Restaurant, float]]:
    if len(restaurants) != len(scores):
        raise ValueError("restaurants and scores must be the same length")
    paired = list(zip(restaurants, scores))
    paired.sort(key=lambda item: (-item[1], item[0].restaurant_id))
    return paired[:top_k]


def rank_control(restaurants, user_lat, user_lon, availability, top_k: int = 10):
    return rank_baseline(restaurants, user_lat, user_lon, availability, top_k=top_k)

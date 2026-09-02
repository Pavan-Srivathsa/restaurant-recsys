"""Logging-policy display propensities and position examination.

The simulator ranks by contextual popularity, keeps the top 8, then samples
the remaining display slots uniformly from the rest. IPS needs that exact
P(shown | context, restaurant), not a fitted black box.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from data.ingest import LOGGING_AVAILABILITY, N_EXPLOIT, N_SHOW
from ranking.baseline import position_propensity


def logging_policy_scores(group: pd.DataFrame) -> np.ndarray:
    """Reconstruct the logging-policy score used in `generate_world`."""
    pop = group["restaurant_popularity"].to_numpy(dtype=float)
    rating = group["restaurant_rating"].to_numpy(dtype=float) / 5.0
    dist = group["distance_score"].to_numpy(dtype=float)
    reviews = np.minimum(group["review_count"].to_numpy(dtype=float) / 2000.0, 1.0)
    return 0.35 * pop + 0.25 * rating + 0.20 * dist + 0.10 * LOGGING_AVAILABILITY + 0.10 * reviews


def rank_ids_by_logging_policy(group: pd.DataFrame) -> List[str]:
    ids = list(group["restaurant_id"])
    scores = logging_policy_scores(group)
    paired = list(zip(ids, scores))
    paired.sort(key=lambda item: (-item[1], item[0]))
    return [rid for rid, _ in paired]


def display_propensities(
    ranked_ids: Sequence[str],
    n_show: int = N_SHOW,
    n_exploit: int = N_EXPLOIT,
) -> Dict[str, float]:
    """P(shown | logging policy) for every candidate.

    Top `n_exploit` restaurants are always shown. Remaining slots are a
    uniform sample from the rest of the ranked list.
    """
    ranked_ids = list(ranked_ids)
    n = len(ranked_ids)
    n_show = min(n_show, n)
    n_exploit = min(n_exploit, n_show)
    extra = n_show - n_exploit
    remainder_n = max(n - n_exploit, 0)
    out: Dict[str, float] = {}
    for i, rid in enumerate(ranked_ids):
        if i < n_exploit:
            out[rid] = 1.0
        elif extra > 0 and remainder_n > 0:
            out[rid] = extra / float(remainder_n)
        else:
            out[rid] = 0.0
    return out


def clip_propensity(p: float, min_p: float = 0.001) -> float:
    if p <= 0:
        return 0.0
    return max(float(p), min_p)


def examination_propensity(rank_position: int, n_shown: int = N_SHOW) -> float:
    return position_propensity(rank_position, n_shown=n_shown)


def combined_exposure_propensity(
    display_p: float,
    rank_position: int | None,
    n_shown: int = N_SHOW,
    min_p: float = 0.001,
) -> float:
    """P(shown) × P(examine | position) for a logged impression."""
    if display_p <= 0:
        return 0.0
    exam = 1.0 if rank_position is None else examination_propensity(int(rank_position), n_shown=n_shown)
    return clip_propensity(display_p * exam, min_p=min_p)


def attach_propensities(group: pd.DataFrame, min_p: float = 0.001) -> pd.DataFrame:
    """Add display_p, exam_p, and exposure_p columns for one request group."""
    ranked = rank_ids_by_logging_policy(group)
    display = display_propensities(ranked)
    rows = []
    for row in group.itertuples(index=False):
        rid = row.restaurant_id
        d_p = display.get(rid, 0.0)
        pos = getattr(row, "logged_position", None)
        if pos is not None and not (isinstance(pos, float) and np.isnan(pos)):
            pos_i = int(pos)
            exam = examination_propensity(pos_i)
        else:
            pos_i = None
            exam = 0.0
        exposure = combined_exposure_propensity(d_p, pos_i, min_p=min_p) if bool(row.logged) else 0.0
        rows.append((d_p, exam, exposure))
    out = group.copy()
    out["display_p"] = [r[0] for r in rows]
    out["exam_p"] = [r[1] for r in rows]
    out["exposure_p"] = [r[2] for r in rows]
    return out

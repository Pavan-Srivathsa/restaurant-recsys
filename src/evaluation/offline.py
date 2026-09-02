"""Temporal offline ranking evaluation.

Primary comparison is contextual popularity vs personalized LambdaMART on the
latest 15% of requests. Features use only events strictly before the request.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List

import pandas as pd

from evaluation.ranking_metrics import (
    catalog_diversity,
    coverage,
    hit_rate_at_k,
    mean_metric,
    mrr,
    ndcg_at_k,
    recall_at_k,
    relative_improvement,
)
from ranking.predict import scores_from_model
from ranking.train import FEATURE_COLUMNS


def _rank_ids(ids: List[str], scores: List[float]) -> List[str]:
    paired = list(zip(ids, scores))
    paired.sort(key=lambda item: (-item[1], item[0]))
    return [rid for rid, _ in paired]


def _baseline_scores(group: pd.DataFrame) -> List[float]:
    scores = []
    for row in group.itertuples(index=False):
        # Reconstruct a lightweight score without Restaurant objects.
        rating_norm = float(row.restaurant_rating) / 5.0
        scores.append(
            0.35 * float(row.restaurant_popularity)
            + 0.25 * rating_norm
            + 0.20 * float(row.distance_score)
            + 0.10 * float(row.availability)
            + 0.10 * float(row.local_booking_rate)
        )
    return scores


def evaluate_split(frame: pd.DataFrame, model, k: int = 10, cuisine_by_id: Dict[str, str] | None = None) -> Dict:
    cuisine_by_id = cuisine_by_id or {}
    per_base = []
    per_model = []
    all_base: List[str] = []
    all_model: List[str] = []
    cohort_base: Dict[str, List[dict]] = defaultdict(list)
    cohort_model: Dict[str, List[dict]] = defaultdict(list)

    for request_id, group in frame.groupby("request_id", sort=False):
        group = group.reset_index(drop=True)
        ids = list(group["restaurant_id"])
        oracle = [int(v) for v in group["oracle_label"].tolist()]
        relevant = {rid for rid, lab in zip(ids, oracle) if lab >= 2}
        if not relevant:
            continue
        feats = group[list(FEATURE_COLUMNS)].to_dict(orient="records")
        model_scores = scores_from_model(model, feats)
        base_scores = _baseline_scores(group)
        ranked_base = _rank_ids(ids, base_scores)
        ranked_model = _rank_ids(ids, model_scores)
        rel_map = dict(zip(ids, oracle))
        base_gains = [rel_map[i] for i in ranked_base]
        model_gains = [rel_map[i] for i in ranked_model]
        base_row = {
            "ndcg@10": ndcg_at_k(base_gains, k=k),
            "recall@10": recall_at_k(ranked_base, relevant, k=k),
            "hit_rate@10": hit_rate_at_k(ranked_base, relevant, k=k),
            "mrr": mrr(ranked_base, relevant),
            "diversity": catalog_diversity(ranked_base, cuisine_by_id, k=k),
        }
        model_row = {
            "ndcg@10": ndcg_at_k(model_gains, k=k),
            "recall@10": recall_at_k(ranked_model, relevant, k=k),
            "hit_rate@10": hit_rate_at_k(ranked_model, relevant, k=k),
            "mrr": mrr(ranked_model, relevant),
            "diversity": catalog_diversity(ranked_model, cuisine_by_id, k=k),
        }
        per_base.append(base_row)
        per_model.append(model_row)
        all_base.extend(ranked_base[:k])
        all_model.extend(ranked_model[:k])
        cohort = str(group["cohort"].iloc[0])
        cohort_base[cohort].append(base_row)
        cohort_model[cohort].append(model_row)

    catalog_size = int(frame["restaurant_id"].nunique()) if len(frame) else 0

    def _summary(rows: List[dict], retrieved: List[str]) -> Dict[str, float]:
        if not rows:
            return {
                "ndcg@10": 0.0,
                "recall@10": 0.0,
                "hit_rate@10": 0.0,
                "mrr": 0.0,
                "diversity": 0.0,
                "coverage": 0.0,
                "n_requests": 0.0,
            }
        out = {key: mean_metric([r[key] for r in rows]) for key in rows[0]}
        out["coverage"] = coverage(retrieved, catalog_size)
        out["n_requests"] = float(len(rows))
        return out

    baseline = _summary(per_base, all_base)
    personalized = _summary(per_model, all_model)
    cohorts = {}
    for name in ("new", "light", "returning", "heavy"):
        b = _summary(cohort_base.get(name, []), [])
        p = _summary(cohort_model.get(name, []), [])
        cohorts[name] = {
            "n_requests": b["n_requests"],
            "baseline_ndcg@10": b["ndcg@10"],
            "personalized_ndcg@10": p["ndcg@10"],
            "ndcg_lift": relative_improvement(p["ndcg@10"], b["ndcg@10"]),
            "baseline_recall@10": b["recall@10"],
            "personalized_recall@10": p["recall@10"],
            "recall_lift": relative_improvement(p["recall@10"], b["recall@10"]),
        }
    return {
        "baseline": baseline,
        "personalized": personalized,
        "relative_ndcg_lift": relative_improvement(personalized["ndcg@10"], baseline["ndcg@10"]),
        "relative_recall_lift": relative_improvement(personalized["recall@10"], baseline["recall@10"]),
        "cohorts": cohorts,
    }


def write_metrics(payload: Dict, path) -> None:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))

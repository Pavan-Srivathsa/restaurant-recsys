"""Offline ranking metrics: NDCG, Recall, HitRate, MRR, coverage, diversity."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Dict, List, Set


def dcg_at_k(relevances: Sequence[float], k: int = 10) -> float:
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += (2.0 ** rel - 1.0) / math.log2(i + 2.0)
    return dcg


def ndcg_at_k(predicted: Sequence[float], k: int = 10) -> float:
    """NDCG of a ranked relevance list (already ordered by the model)."""
    actual = dcg_at_k(predicted, k)
    ideal = dcg_at_k(sorted(predicted, reverse=True), k)
    if ideal <= 0:
        return 0.0
    return actual / ideal


def ndcg_from_gains(ranked_gains: Sequence[float], k: int = 10) -> float:
    return ndcg_at_k(ranked_gains, k=k)


def recall_at_k(retrieved: Sequence[str], relevant: Set[str], k: int = 10) -> float:
    if not relevant:
        return 0.0
    hit = sum(1 for item in retrieved[:k] if item in relevant)
    return hit / len(relevant)


def hit_rate_at_k(retrieved: Sequence[str], relevant: Set[str], k: int = 10) -> float:
    if not relevant:
        return 0.0
    return 1.0 if any(item in relevant for item in retrieved[:k]) else 0.0


def mrr(retrieved: Sequence[str], relevant: Set[str]) -> float:
    for i, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def coverage(all_retrieved: Iterable[str], catalog_size: int) -> float:
    if catalog_size <= 0:
        return 0.0
    return len(set(all_retrieved)) / float(catalog_size)


def catalog_diversity(retrieved: Sequence[str], cuisine_by_id: Dict[str, str], k: int = 10) -> float:
    """Fraction of distinct cuisines in the top-k list."""
    cuisines = [cuisine_by_id[i] for i in retrieved[:k] if i in cuisine_by_id]
    if not cuisines:
        return 0.0
    return len(set(cuisines)) / float(len(cuisines))


def mean_metric(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def relative_improvement(new_value: float, baseline_value: float) -> float:
    if baseline_value == 0:
        return 0.0
    return (new_value - baseline_value) / baseline_value


def summarize_lists(
    per_request: List[Dict[str, float]],
) -> Dict[str, float]:
    keys = per_request[0].keys() if per_request else []
    return {key: mean_metric([row[key] for row in per_request]) for key in keys}

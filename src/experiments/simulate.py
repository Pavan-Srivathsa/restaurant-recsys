"""Simulate the online A/B test after the logging period.

Control: contextual popularity, deterministic top-10.
Treatment: personalized LambdaMART top-10.
Assignment is hashed at user_id and stays fixed.
Eligible sets are identical across arms; only ranking differs.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List

from data.build_features import (
    cuisine_map,
    pair_features_from_snapshot,
    price_map,
    restaurant_snapshot_from_events,
    user_snapshot,
)
from data.index import InteractionIndex
from data.ingest import (
    LOGGING_AVAILABILITY,
    N_SHOW,
    SyntheticWorld,
    eligible_restaurants,
    funnel_events,
    logging_score,
)
from data.schemas import Interaction, RequestLog, Restaurant, User
from experiments.assignment import assign_variant
from ranking.predict import rank_by_scores, scores_from_model


@dataclass
class ExperimentRun:
    interactions: List[Interaction]
    requests: List[RequestLog]
    variants: Dict[str, str]
    ranking_ms: List[float] = field(default_factory=list)
    ranking_ms_by_variant: Dict[str, List[float]] = field(default_factory=dict)
    no_result_count: int = 0
    n_attempted: int = 0


def _activity_weights(n_users: int) -> List[float]:
    return [1.0 / ((i + 1) ** 0.55) for i in range(n_users)]


def _clamp_hour(ts: datetime) -> datetime:
    if ts.hour < 10:
        return ts.replace(hour=11)
    if ts.hour > 21:
        return ts.replace(hour=19)
    return ts


def rank_popularity_slate(user: User, eligible: List[Restaurant], top_k: int = N_SHOW) -> List[Restaurant]:
    ranked = sorted(
        eligible,
        key=lambda r: (
            -logging_score(r, user.latitude, user.longitude, LOGGING_AVAILABILITY),
            r.restaurant_id,
        ),
    )
    return ranked[:top_k]


def rank_personalized_slate(
    user: User,
    eligible: List[Restaurant],
    index: InteractionIndex,
    timestamp: datetime,
    cuisine: Dict[str, str],
    price: Dict[str, float],
    model,
    top_k: int = N_SHOW,
) -> List[Restaurant]:
    snapshot = user_snapshot(index, user.user_id, timestamp, cuisine, price)
    rows = []
    for restaurant in eligible:
        rest_stats = restaurant_snapshot_from_events(
            index.restaurant_as_of(restaurant.restaurant_id, timestamp), timestamp
        )
        rows.append(
            pair_features_from_snapshot(
                user, restaurant, snapshot, rest_stats, LOGGING_AVAILABILITY
            )
        )
    scores = scores_from_model(model, rows)
    return [r for r, _ in rank_by_scores(eligible, scores, top_k=top_k)]


def simulate_experiment(
    world: SyntheticWorld,
    model,
    n_requests: int = 2800,
    seed: int = 11,
    experiment_name: str = "personalized_ranker_v1",
) -> ExperimentRun:
    rng = random.Random(seed)
    cuisine = cuisine_map(world.restaurants)
    price = price_map(world.restaurants)
    variants = {u.user_id: assign_variant(u.user_id, experiment_name) for u in world.users}
    weights = _activity_weights(len(world.users))
    last_ts = max((r.timestamp for r in world.requests), default=datetime(2025, 6, 1))
    start = last_ts + timedelta(days=1)
    index = InteractionIndex(world.interactions)
    exp_events: List[Interaction] = []
    requests: List[RequestLog] = []
    ranking_ms: List[float] = []
    ranking_ms_by_variant: Dict[str, List[float]] = {"control": [], "treatment": []}
    no_result = 0

    for k in range(n_requests):
        user = rng.choices(world.users, weights=weights, k=1)[0]
        pref = world.prefs[user.user_id]
        ts = _clamp_hour(start + timedelta(hours=k, minutes=rng.randint(0, 40)))
        request_id = f"exp_{k}"
        variant = variants[user.user_id]
        eligible = eligible_restaurants(world.restaurants, user.latitude, user.longitude, ts)
        if not eligible:
            no_result += 1
            continue
        t0 = time.perf_counter()
        if variant == "treatment":
            shown = rank_personalized_slate(user, eligible, index, ts, cuisine, price, model)
        else:
            shown = rank_popularity_slate(user, eligible)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        ranking_ms.append(elapsed_ms)
        ranking_ms_by_variant[variant].append(elapsed_ms)
        shown_ids = []
        new_events: List[Interaction] = []
        for pos, rest in enumerate(shown, start=1):
            shown_ids.append(rest.restaurant_id)
            new_events.extend(funnel_events(rng, user, pref, rest, pos, ts, request_id, variant))
        index.append(new_events)
        exp_events.extend(new_events)
        requests.append(
            RequestLog(
                request_id=request_id,
                user_id=user.user_id,
                timestamp=ts,
                latitude=user.latitude,
                longitude=user.longitude,
                shown_ids=tuple(shown_ids),
            )
        )
    return ExperimentRun(
        interactions=exp_events,
        requests=requests,
        variants=variants,
        ranking_ms=ranking_ms,
        ranking_ms_by_variant=ranking_ms_by_variant,
        no_result_count=no_result,
        n_attempted=n_requests,
    )

"""Feature construction with as-of timestamps.

Every aggregate uses only information available before the recommendation time.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Dict

from data.geo import distance_features, haversine_km
from data.index import InteractionIndex
from data.schemas import EVENT_WEIGHTS, Interaction, Restaurant, User
from data.splits import filter_as_of

DEFAULT_LAMBDA = 0.03


def _days_since(event_ts: datetime, as_of: datetime) -> float:
    return max(0.0, (as_of - event_ts).total_seconds() / 86400.0)


def cuisine_affinity(
    interactions: Sequence[Interaction],
    restaurant_cuisine: Mapping[str, str],
    as_of: datetime,
    cuisine: str,
    decay_lambda: float = DEFAULT_LAMBDA,
) -> float:
    past = filter_as_of(interactions, as_of)
    weighted_total = 0.0
    weighted_cuisine = 0.0
    for event in past:
        w = EVENT_WEIGHTS.get(event.event_type, 0.0)
        if w <= 0:
            continue
        decay = math.exp(-decay_lambda * _days_since(event.timestamp, as_of))
        value = w * decay
        weighted_total += value
        if restaurant_cuisine.get(event.restaurant_id) == cuisine:
            weighted_cuisine += value
    if weighted_total <= 0:
        return 0.0
    return weighted_cuisine / weighted_total


def price_preference(
    interactions: Sequence[Interaction],
    restaurant_price: Mapping[str, float],
    as_of: datetime,
) -> Dict[str, float]:
    past = [
        e
        for e in filter_as_of(interactions, as_of)
        if EVENT_WEIGHTS.get(e.event_type, 0.0) > 0 and e.restaurant_id in restaurant_price
    ]
    prices = [float(restaurant_price[e.restaurant_id]) for e in past]
    if not prices:
        return {"user_mean_price": 0.0, "user_price_std": 0.0}
    mean = sum(prices) / len(prices)
    var = sum((p - mean) ** 2 for p in prices) / max(len(prices) - 1, 1)
    return {"user_mean_price": mean, "user_price_std": math.sqrt(var)}


def price_match_score(user_mean_price: float, restaurant_price: float, has_history: bool = True) -> float:
    if not has_history:
        return 0.5
    return math.exp(-abs(user_mean_price - restaurant_price))


def previous_counts(
    interactions: Sequence[Interaction],
    user_id: str,
    restaurant_id: str,
    as_of: datetime,
) -> Dict[str, float]:
    past = [
        e
        for e in filter_as_of(interactions, as_of)
        if e.user_id == user_id and e.restaurant_id == restaurant_id
    ]
    return {
        "previous_visits": float(len(past)),
        "previous_clicks": float(sum(1 for e in past if e.event_type == "click")),
        "previous_bookings": float(
            sum(1 for e in past if e.event_type in {"booking", "completed_reservation"})
        ),
    }


def restaurant_window_stats(
    interactions: Sequence[Interaction],
    restaurant_id: str,
    as_of: datetime,
) -> Dict[str, float]:
    past = [e for e in filter_as_of(interactions, as_of) if e.restaurant_id == restaurant_id]
    d7 = as_of - timedelta(days=7)
    d30 = as_of - timedelta(days=30)
    bookings_7 = sum(1 for e in past if e.event_type == "booking" and e.timestamp >= d7)
    bookings_30 = sum(1 for e in past if e.event_type == "booking" and e.timestamp >= d30)
    completed_30 = sum(
        1 for e in past if e.event_type == "completed_reservation" and e.timestamp >= d30
    )
    cancelled_30 = sum(
        1 for e in past if e.event_type == "cancelled_reservation" and e.timestamp >= d30
    )
    completion_den = completed_30 + cancelled_30
    return {
        "bookings_last_7_days": float(bookings_7),
        "bookings_last_30_days": float(bookings_30),
        "completion_rate_last_30_days": (completed_30 / completion_den) if completion_den else 0.0,
        "historical_booking_rate": float(
            sum(1 for e in past if e.event_type == "booking") / max(len(past), 1)
        ),
    }


def context_features(timestamp: datetime) -> Dict[str, float]:
    hour = timestamp.hour
    weekday = timestamp.weekday()
    if 6 <= hour < 11:
        meal = 0.0
    elif 11 <= hour < 15:
        meal = 1.0
    elif 17 <= hour < 22:
        meal = 2.0
    else:
        meal = 3.0
    return {
        "hour": float(hour),
        "weekday": float(weekday),
        "weekend": 1.0 if weekday >= 5 else 0.0,
        "meal_period": meal,
    }


def build_pair_features(
    user: User,
    restaurant: Restaurant,
    interactions: Sequence[Interaction],
    as_of: datetime,
    availability: float,
    restaurant_cuisine: Mapping[str, str],
    restaurant_price: Mapping[str, float],
) -> Dict[str, float]:
    """User × restaurant × context features strictly as-of `as_of`."""
    dist = haversine_km(user.latitude, user.longitude, restaurant.latitude, restaurant.longitude)
    feats: Dict[str, float] = {}
    feats.update(distance_features(dist))
    feats.update(context_features(as_of))
    feats.update(previous_counts(interactions, user.user_id, restaurant.restaurant_id, as_of))
    feats.update(restaurant_window_stats(interactions, restaurant.restaurant_id, as_of))
    price_pref = price_preference(interactions, restaurant_price, as_of)
    feats.update(price_pref)
    feats["restaurant_price"] = float(restaurant.price_level)
    feats["absolute_price_difference"] = abs(feats["user_mean_price"] - restaurant.price_level)
    feats["price_match"] = price_match_score(
        feats["user_mean_price"], float(restaurant.price_level), has_history=feats["user_mean_price"] > 0
    )
    feats["cuisine_affinity"] = cuisine_affinity(
        [e for e in interactions if e.user_id == user.user_id],
        restaurant_cuisine,
        as_of,
        restaurant.cuisine,
    )
    user_events = [e for e in filter_as_of(interactions, as_of) if e.user_id == user.user_id]
    feats["user_activity_count"] = float(len(user_events))
    feats["user_booking_count"] = float(
        sum(1 for e in user_events if e.event_type in {"booking", "completed_reservation"})
    )
    feats["user_average_price"] = feats["user_mean_price"]
    feats["restaurant_rating"] = float(restaurant.rating)
    feats["restaurant_popularity"] = float(restaurant.popularity_score)
    feats["reservation_availability"] = float(availability)
    feats["review_count"] = float(restaurant.review_count)
    return feats


def cuisine_map(restaurants: Iterable[Restaurant]) -> Dict[str, str]:
    return {r.restaurant_id: r.cuisine for r in restaurants}


def price_map(restaurants: Iterable[Restaurant]) -> Dict[str, float]:
    return {r.restaurant_id: float(r.price_level) for r in restaurants}


def user_cohort(historical_interaction_count: int) -> str:
    """Cohort from as-of clicks/bookings, not impressions."""
    if historical_interaction_count <= 0:
        return "new"
    if historical_interaction_count <= 5:
        return "light"
    if historical_interaction_count >= 20:
        return "heavy"
    return "returning"


def user_snapshot(
    index: InteractionIndex,
    user_id: str,
    as_of: datetime,
    restaurant_cuisine: Mapping[str, str],
    restaurant_price: Mapping[str, float],
    decay_lambda: float = DEFAULT_LAMBDA,
) -> dict:
    """Precompute per-user as-of features once per request."""
    past = index.user_as_of(user_id, as_of)
    weighted_total = 0.0
    cuisine_weighted: Dict[str, float] = {}
    prices = []
    pair_counts: Dict[str, Dict[str, float]] = {}
    booking_count = 0
    engagement_count = 0
    for event in past:
        counts = pair_counts.setdefault(
            event.restaurant_id,
            {"previous_visits": 0.0, "previous_clicks": 0.0, "previous_bookings": 0.0},
        )
        counts["previous_visits"] += 1.0
        if event.event_type == "click":
            counts["previous_clicks"] += 1.0
        if event.event_type in {"booking", "completed_reservation"}:
            counts["previous_bookings"] += 1.0
            booking_count += 1
        w = EVENT_WEIGHTS.get(event.event_type, 0.0)
        if w <= 0:
            continue
        engagement_count += 1
        value = w * math.exp(-decay_lambda * _days_since(event.timestamp, as_of))
        weighted_total += value
        cuisine = restaurant_cuisine.get(event.restaurant_id)
        if cuisine:
            cuisine_weighted[cuisine] = cuisine_weighted.get(cuisine, 0.0) + value
        if event.restaurant_id in restaurant_price:
            prices.append(float(restaurant_price[event.restaurant_id]))
    affinities = {c: (w / weighted_total if weighted_total else 0.0) for c, w in cuisine_weighted.items()}
    if prices:
        mean = sum(prices) / len(prices)
        var = sum((p - mean) ** 2 for p in prices) / max(len(prices) - 1, 1)
        std = math.sqrt(var)
    else:
        mean, std = 0.0, 0.0
    return {
        "user_activity_count": float(len(past)),
        "user_engagement_count": float(engagement_count),
        "user_booking_count": float(booking_count),
        "user_mean_price": mean,
        "user_price_std": std,
        "has_price_history": bool(prices),
        "cuisine_affinity": affinities,
        "pair_counts": pair_counts,
        "context": context_features(as_of),
    }


def restaurant_snapshot_from_events(past: Sequence[Interaction], as_of: datetime) -> dict:
    return _window_stats(past, as_of)


def _window_stats(past: Sequence[Interaction], as_of: datetime) -> dict:
    d7 = as_of - timedelta(days=7)
    d30 = as_of - timedelta(days=30)
    bookings_7 = sum(1 for e in past if e.event_type == "booking" and e.timestamp >= d7)
    bookings_30 = sum(1 for e in past if e.event_type == "booking" and e.timestamp >= d30)
    completed_30 = sum(1 for e in past if e.event_type == "completed_reservation" and e.timestamp >= d30)
    cancelled_30 = sum(1 for e in past if e.event_type == "cancelled_reservation" and e.timestamp >= d30)
    completion_den = completed_30 + cancelled_30
    return {
        "bookings_last_7_days": float(bookings_7),
        "bookings_last_30_days": float(bookings_30),
        "completion_rate_last_30_days": (completed_30 / completion_den) if completion_den else 0.0,
        "historical_booking_rate": float(
            sum(1 for e in past if e.event_type == "booking") / max(len(past), 1)
        ),
    }


def pair_features_from_snapshot(
    user: User,
    restaurant: Restaurant,
    snapshot: dict,
    rest_stats: dict,
    availability: float,
) -> dict:
    dist = haversine_km(user.latitude, user.longitude, restaurant.latitude, restaurant.longitude)
    feats: Dict[str, float] = {}
    feats.update(distance_features(dist))
    feats.update(snapshot["context"])
    counts = snapshot["pair_counts"].get(
        restaurant.restaurant_id,
        {"previous_visits": 0.0, "previous_clicks": 0.0, "previous_bookings": 0.0},
    )
    feats.update(counts)
    feats.update(rest_stats)
    feats["user_mean_price"] = snapshot["user_mean_price"]
    feats["user_price_std"] = snapshot["user_price_std"]
    feats["restaurant_price"] = float(restaurant.price_level)
    feats["absolute_price_difference"] = abs(feats["user_mean_price"] - restaurant.price_level)
    feats["price_match"] = price_match_score(
        feats["user_mean_price"],
        float(restaurant.price_level),
        has_history=snapshot["has_price_history"],
    )
    feats["cuisine_affinity"] = float(snapshot["cuisine_affinity"].get(restaurant.cuisine, 0.0))
    feats["user_activity_count"] = snapshot["user_activity_count"]
    feats["user_booking_count"] = snapshot["user_booking_count"]
    feats["user_average_price"] = feats["user_mean_price"]
    feats["restaurant_rating"] = float(restaurant.rating)
    feats["restaurant_popularity"] = float(restaurant.popularity_score)
    feats["reservation_availability"] = float(availability)
    feats["review_count"] = float(restaurant.review_count)
    return feats

"""Contextual popularity baseline.

Uses location and restaurant context. Does not use personalized user preferences.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import List, Tuple

from data.geo import distance_features, haversine_km
from data.schemas import Restaurant

DEFAULT_WEIGHTS = {
    "w_popularity": 0.35,
    "w_rating": 0.25,
    "w_distance": 0.20,
    "w_availability": 0.10,
    "w_local_booking_rate": 0.10,
}

BASELINE_FEATURE_NAMES = (
    "popularity_score",
    "rating",
    "distance_score",
    "availability",
    "local_booking_rate",
)


def distance_score(user_lat: float, user_lon: float, restaurant: Restaurant) -> float:
    km = haversine_km(user_lat, user_lon, restaurant.latitude, restaurant.longitude)
    return distance_features(km)["distance_score"]


def baseline_score(
    restaurant: Restaurant,
    user_lat: float,
    user_lon: float,
    availability: float,
    local_booking_rate: float = 0.0,
    weights: Mapping[str, float] | None = None,
) -> float:
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    dscore = distance_score(user_lat, user_lon, restaurant)
    rating_norm = restaurant.rating / 5.0
    return (
        w["w_popularity"] * restaurant.popularity_score
        + w["w_rating"] * rating_norm
        + w["w_distance"] * dscore
        + w["w_availability"] * availability
        + w["w_local_booking_rate"] * local_booking_rate
    )


def rank_baseline(
    restaurants: Sequence[Restaurant],
    user_lat: float,
    user_lon: float,
    availability: Mapping[str, float],
    local_booking_rates: Mapping[str, float] | None = None,
    weights: Mapping[str, float] | None = None,
    top_k: int = 10,
) -> List[Tuple[Restaurant, float]]:
    """Deterministic ranking: score descending, restaurant_id ascending ties."""
    rates = local_booking_rates or {}
    scored: List[Tuple[Restaurant, float]] = []
    for restaurant in restaurants:
        score = baseline_score(
            restaurant,
            user_lat,
            user_lon,
            availability.get(restaurant.restaurant_id, 0.0),
            rates.get(restaurant.restaurant_id, 0.0),
            weights=weights,
        )
        scored.append((restaurant, score))
    scored.sort(key=lambda item: (-item[1], item[0].restaurant_id))
    return scored[:top_k]


def position_propensity(rank_position: int, n_shown: int = 10) -> float:
    """Examination propensity decaying with rank. Clipped away from zero for IPS."""
    if rank_position < 1:
        raise ValueError("rank_position must be >= 1")
    weights = [1.0 / i for i in range(1, n_shown + 1)]
    total = sum(weights)
    idx = min(rank_position, n_shown) - 1
    return max(weights[idx] / total, 1e-3)

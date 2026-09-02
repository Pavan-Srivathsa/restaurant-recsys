"""Candidate generation: geo → hours → availability → retrieve."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Dict, List

from data.availability import availability_probability, is_available, is_open
from data.geo import haversine_km
from data.index import InteractionIndex
from data.schemas import Interaction, Restaurant
from data.splits import filter_as_of


def existing_bookings_as_of(
    interactions: Sequence[Interaction],
    restaurant_id: str,
    as_of: datetime,
    window_hours: int = 4,
    index: InteractionIndex | None = None,
) -> int:
    """Count simulated bookings near the request time (not future)."""
    if index is not None:
        return index.bookings_in_window(restaurant_id, as_of, window_hours=window_hours)
    from datetime import timedelta

    start = as_of - timedelta(hours=window_hours)
    past = filter_as_of(interactions, as_of)
    return sum(
        1
        for e in past
        if e.restaurant_id == restaurant_id
        and e.event_type in {"booking", "completed_reservation"}
        and e.timestamp >= start
    )


def generate_candidates(
    restaurants: Sequence[Restaurant],
    interactions: Sequence[Interaction],
    latitude: float,
    longitude: float,
    timestamp: datetime,
    radius_km: float = 15.0,
    limit: int = 300,
    min_availability: float = 0.05,
    index: InteractionIndex | None = None,
) -> List[Restaurant]:
    """Return eligible restaurants, capped at `limit`.

    Order is deterministic: nearer restaurants first, then restaurant_id.
    """
    eligible: List[Restaurant] = []
    for restaurant in restaurants:
        dist = haversine_km(latitude, longitude, restaurant.latitude, restaurant.longitude)
        if dist > radius_km:
            continue
        if not is_open(restaurant.opening_hours, timestamp):
            continue
        bookings = existing_bookings_as_of(
            interactions, restaurant.restaurant_id, timestamp, index=index
        )
        if not is_available(
            restaurant.capacity,
            timestamp.hour,
            timestamp.weekday(),
            restaurant.popularity_score,
            bookings,
            min_probability=min_availability,
        ):
            continue
        eligible.append(restaurant)

    eligible.sort(
        key=lambda r: (
            haversine_km(latitude, longitude, r.latitude, r.longitude),
            r.restaurant_id,
        )
    )
    return eligible[:limit]


def candidate_availability_map(
    candidates: Sequence[Restaurant],
    interactions: Sequence[Interaction],
    timestamp: datetime,
    index: InteractionIndex | None = None,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for restaurant in candidates:
        bookings = existing_bookings_as_of(
            interactions, restaurant.restaurant_id, timestamp, index=index
        )
        out[restaurant.restaurant_id] = availability_probability(
            restaurant.capacity,
            timestamp.hour,
            timestamp.weekday(),
            restaurant.popularity_score,
            bookings,
        )
    return out

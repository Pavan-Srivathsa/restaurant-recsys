"""Simulated reservation availability.

Real booking inventory is not used. Probability is a function of capacity,
hour, weekday, popularity, and already-simulated bookings.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

SIMULATED = True


def availability_probability(
    capacity: int,
    hour: int,
    weekday: int,
    popularity: float,
    existing_bookings: int,
) -> float:
    """Return P(bookable) in [0, 1].

    Peak meal hours and weekends reduce remaining inventory. Popular restaurants
    fill faster. Occupancy cannot exceed capacity.
    """
    if capacity <= 0:
        return 0.0
    occupancy = min(1.0, max(0.0, existing_bookings / float(capacity)))
    remaining = 1.0 - occupancy

    peak_hours = {11, 12, 13, 18, 19, 20}
    hour_factor = 0.45 if hour in peak_hours else 0.90
    weekend = weekday >= 5
    weekday_factor = 0.65 if weekend else 0.95
    popularity_pressure = 1.0 / (1.0 + max(popularity, 0.0))

    p = remaining * hour_factor * weekday_factor * (0.55 + 0.45 * popularity_pressure)
    return float(min(1.0, max(0.0, p)))


def is_open(opening_hours: Mapping[str, list], timestamp: datetime) -> bool:
    """opening_hours maps weekday name or int-as-string to list of open hours."""
    if not opening_hours:
        return True
    keys = (
        str(timestamp.weekday()),
        timestamp.strftime("%A").lower(),
        timestamp.strftime("%a").lower(),
    )
    hours = None
    for key in keys:
        if key in opening_hours:
            hours = opening_hours[key]
            break
    if hours is None:
        return True
    return timestamp.hour in hours


def is_available(
    capacity: int,
    hour: int,
    weekday: int,
    popularity: float,
    existing_bookings: int,
    min_probability: float = 0.05,
) -> bool:
    return (
        availability_probability(capacity, hour, weekday, popularity, existing_bookings)
        >= min_probability
    )

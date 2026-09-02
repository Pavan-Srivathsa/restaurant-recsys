"""Sorted interaction indexes for as-of feature lookups."""

from __future__ import annotations

import bisect
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Sequence

from data.schemas import Interaction


class InteractionIndex:
    """Events grouped by user and restaurant, sorted by timestamp."""

    def __init__(self, interactions: Sequence[Interaction]) -> None:
        by_user: Dict[str, List[Interaction]] = defaultdict(list)
        by_rest: Dict[str, List[Interaction]] = defaultdict(list)
        booking_ts: Dict[str, List[datetime]] = defaultdict(list)
        for event in sorted(interactions, key=lambda e: e.timestamp):
            by_user[event.user_id].append(event)
            by_rest[event.restaurant_id].append(event)
            if event.event_type in {"booking", "completed_reservation"}:
                booking_ts[event.restaurant_id].append(event.timestamp)
        self.by_user = dict(by_user)
        self.by_restaurant = dict(by_rest)
        self.booking_ts = dict(booking_ts)

    def append(self, events: Sequence[Interaction]) -> None:
        """Insert later events while keeping timestamp order (as-of lookups stay valid)."""
        for event in events:
            user_list = self.by_user.setdefault(event.user_id, [])
            _insert_sorted(user_list, event)
            rest_list = self.by_restaurant.setdefault(event.restaurant_id, [])
            _insert_sorted(rest_list, event)
            if event.event_type in {"booking", "completed_reservation"}:
                stamps = self.booking_ts.setdefault(event.restaurant_id, [])
                bisect.insort(stamps, event.timestamp)

    def user_as_of(self, user_id: str, as_of: datetime) -> List[Interaction]:
        return _before(self.by_user.get(user_id, ()), as_of)

    def restaurant_as_of(self, restaurant_id: str, as_of: datetime) -> List[Interaction]:
        return _before(self.by_restaurant.get(restaurant_id, ()), as_of)

    def pair_as_of(self, user_id: str, restaurant_id: str, as_of: datetime) -> List[Interaction]:
        return [e for e in self.user_as_of(user_id, as_of) if e.restaurant_id == restaurant_id]

    def bookings_in_window(self, restaurant_id: str, as_of: datetime, window_hours: int = 4) -> int:
        stamps = self.booking_ts.get(restaurant_id, ())
        start = as_of - timedelta(hours=window_hours)
        lo = bisect.bisect_left(stamps, start)
        hi = bisect.bisect_left(stamps, as_of)
        return hi - lo


def _insert_sorted(events: List[Interaction], event: Interaction) -> None:
    idx = bisect.bisect_left([e.timestamp for e in events], event.timestamp)
    events.insert(idx, event)


def _before(events: Sequence[Interaction], as_of: datetime) -> List[Interaction]:
    timestamps = [e.timestamp for e in events]
    return list(events[: bisect.bisect_left(timestamps, as_of)])

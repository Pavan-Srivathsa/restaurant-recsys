"""Synthetic interaction data for a reproducible public path.

Users have latent cuisine and price preferences. The logging policy is
contextual popularity (not personalized). Clicks and bookings are generated
from those latent preferences plus position bias.

Replace `load_yelp` with a real Yelp Open Dataset ingest when the dump is available.
This generator is deterministic given `seed`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

from data.availability import is_open
from data.geo import distance_features, haversine_km
from data.schemas import Interaction, RequestLog, Restaurant, User, UserPref

CUISINES = ("italian", "mexican", "japanese", "american", "indian", "thai", "chinese", "korean")
LOGGING_AVAILABILITY = 0.7
N_SHOW = 10
N_EXPLOIT = 8


@dataclass
class SyntheticWorld:
    users: List[User]
    restaurants: List[Restaurant]
    interactions: List[Interaction]
    prefs: Dict[str, UserPref]
    requests: List[RequestLog]


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def logging_score(restaurant: Restaurant, user_lat: float, user_lon: float, availability: float) -> float:
    dist = haversine_km(user_lat, user_lon, restaurant.latitude, restaurant.longitude)
    return (
        0.35 * restaurant.popularity_score
        + 0.25 * (restaurant.rating / 5.0)
        + 0.20 * distance_features(dist)["distance_score"]
        + 0.10 * availability
        + 0.10 * min(1.0, restaurant.review_count / 2000.0)
    )


def preference_utility(pref: UserPref, restaurant: Restaurant, user_lat: float, user_lon: float) -> float:
    match = 1.0 if restaurant.cuisine in pref.cuisines else 0.0
    price_m = math.exp(-abs(pref.price - restaurant.price_level))
    dist = haversine_km(user_lat, user_lon, restaurant.latitude, restaurant.longitude)
    dist_s = distance_features(dist)["distance_score"]
    return 1.85 * match + 0.95 * price_m + 0.75 * dist_s + 0.35 * (restaurant.rating / 5.0)


def oracle_relevance(pref: UserPref, restaurant: Restaurant, user_lat: float, user_lon: float) -> int:
    """Position-free graded relevance for offline evaluation. Not a training feature."""
    utility = preference_utility(pref, restaurant, user_lat, user_lon)
    if utility >= 2.55:
        return 3
    if utility >= 1.95:
        return 2
    if utility >= 1.30:
        return 1
    return 0


def eligible_restaurants(
    restaurants: Sequence[Restaurant],
    latitude: float,
    longitude: float,
    timestamp: datetime,
    radius_km: float = 15.0,
) -> List[Restaurant]:
    out = []
    for restaurant in restaurants:
        dist = haversine_km(latitude, longitude, restaurant.latitude, restaurant.longitude)
        if dist > radius_km:
            continue
        if not is_open(restaurant.opening_hours, timestamp):
            continue
        out.append(restaurant)
    return out


def funnel_events(
    rng: random.Random,
    user: User,
    pref: UserPref,
    restaurant: Restaurant,
    position: int,
    timestamp: datetime,
    request_id: str,
    variant: str,
) -> List[Interaction]:
    """Impression plus optional click → booking → complete/cancel."""
    events = [
        Interaction(
            user_id=user.user_id,
            restaurant_id=restaurant.restaurant_id,
            timestamp=timestamp,
            event_type="impression",
            position=position,
            experiment_variant=variant,
            request_id=request_id,
        )
    ]
    match = 1.0 if restaurant.cuisine in pref.cuisines else 0.0
    price_m = math.exp(-abs(pref.price - restaurant.price_level))
    dist = haversine_km(user.latitude, user.longitude, restaurant.latitude, restaurant.longitude)
    dist_s = distance_features(dist)["distance_score"]
    logit = (
        -1.15
        + 2.05 * match
        + 0.95 * price_m
        + 0.70 * dist_s
        + 0.30 * (restaurant.rating / 5.0)
        + 0.12 * restaurant.popularity_score
        - 0.26 * (position - 1)
    )
    if rng.random() > _sigmoid(logit):
        return events
    events.append(
        Interaction(
            user_id=user.user_id,
            restaurant_id=restaurant.restaurant_id,
            timestamp=timestamp + timedelta(minutes=2),
            event_type="click",
            position=position,
            experiment_variant=variant,
            request_id=request_id,
        )
    )
    book_logit = -0.35 + 1.70 * match + 0.85 * price_m + 0.40 * dist_s
    if rng.random() > _sigmoid(book_logit):
        return events
    events.append(
        Interaction(
            user_id=user.user_id,
            restaurant_id=restaurant.restaurant_id,
            timestamp=timestamp + timedelta(minutes=8),
            event_type="booking",
            position=position,
            experiment_variant=variant,
            request_id=request_id,
        )
    )
    complete_p = 0.82 if match else 0.52
    if rng.random() < complete_p:
        events.append(
            Interaction(
                user_id=user.user_id,
                restaurant_id=restaurant.restaurant_id,
                timestamp=timestamp + timedelta(hours=2),
                event_type="completed_reservation",
                position=position,
                experiment_variant=variant,
                request_id=request_id,
            )
        )
    else:
        events.append(
            Interaction(
                user_id=user.user_id,
                restaurant_id=restaurant.restaurant_id,
                timestamp=timestamp + timedelta(hours=1),
                event_type="cancelled_reservation",
                position=position,
                experiment_variant=variant,
                request_id=request_id,
            )
        )
    return events


def generate_world(
    n_users: int = 400,
    n_restaurants: int = 90,
    n_requests: int = 1400,
    seed: int = 7,
    start: Optional[datetime] = None,
) -> SyntheticWorld:
    rng = random.Random(seed)
    start = start or datetime(2025, 1, 1, 12, 0, 0)

    prefs: Dict[str, UserPref] = {}
    users: List[User] = []
    for i in range(n_users):
        n_pref = 1 if rng.random() < 0.65 else 2
        cuisines = tuple(rng.sample(CUISINES, k=n_pref))
        price = rng.choice([1, 2, 2, 3, 3, 4])
        user = User(
            user_id=f"user_{i}",
            latitude=34.02 + rng.uniform(-0.12, 0.12),
            longitude=-118.29 + rng.uniform(-0.12, 0.12),
            account_age_days=rng.randint(1, 1200),
            historical_interaction_count=0,
            historical_booking_count=0,
            average_price_preference=float(price),
        )
        users.append(user)
        prefs[user.user_id] = UserPref(user_id=user.user_id, cuisines=cuisines, price=price)

    restaurants: List[Restaurant] = []
    for j in range(n_restaurants):
        hours = {str(d): list(range(10, 23)) for d in range(7)}
        restaurants.append(
            Restaurant(
                restaurant_id=f"r{j}",
                latitude=34.02 + rng.uniform(-0.10, 0.10),
                longitude=-118.29 + rng.uniform(-0.10, 0.10),
                cuisine=rng.choice(CUISINES),
                price_level=rng.choice([1, 2, 2, 3, 3, 4]),
                rating=round(rng.uniform(3.2, 4.9), 2),
                review_count=rng.randint(20, 4000),
                popularity_score=round(rng.uniform(0.1, 1.0), 3),
                opening_hours=hours,
                capacity=rng.randint(20, 80),
            )
        )

    # Zipf-like activity so returning/heavy cohorts exist.
    activity_weights = [1.0 / ((i + 1) ** 0.55) for i in range(n_users)]

    interactions: List[Interaction] = []
    requests: List[RequestLog] = []
    for k in range(n_requests):
        user = rng.choices(users, weights=activity_weights, k=1)[0]
        pref = prefs[user.user_id]
        ts = start + timedelta(hours=k * 2, minutes=rng.randint(0, 50))
        if ts.hour < 10:
            ts = ts.replace(hour=11)
        if ts.hour > 21:
            ts = ts.replace(hour=19)
        request_id = f"req_{k}"
        eligible = eligible_restaurants(restaurants, user.latitude, user.longitude, ts)
        if len(eligible) < 5:
            continue
        ranked = sorted(
            eligible,
            key=lambda r: (
                -logging_score(r, user.latitude, user.longitude, LOGGING_AVAILABILITY),
                r.restaurant_id,
            ),
        )
        n_show = min(N_SHOW, len(ranked))
        n_exploit = min(N_EXPLOIT, n_show)
        shown = list(ranked[:n_exploit])
        remainder = ranked[n_exploit:]
        extra = min(n_show - n_exploit, len(remainder))
        if extra:
            shown.extend(rng.sample(remainder, k=extra))

        shown_ids = []
        for pos, rest in enumerate(shown, start=1):
            shown_ids.append(rest.restaurant_id)
            interactions.extend(
                funnel_events(rng, user, pref, rest, pos, ts, request_id, "logging")
            )
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
    return SyntheticWorld(
        users=users,
        restaurants=restaurants,
        interactions=interactions,
        prefs=prefs,
        requests=requests,
    )


def generate_synthetic(
    n_users: int = 200,
    n_restaurants: int = 80,
    n_requests: int = 400,
    seed: int = 7,
    start: Optional[datetime] = None,
) -> Tuple[List[User], List[Restaurant], List[Interaction]]:
    world = generate_world(
        n_users=n_users,
        n_restaurants=n_restaurants,
        n_requests=n_requests,
        seed=seed,
        start=start,
    )
    return world.users, world.restaurants, world.interactions


def load_yelp(raw_dir: str) -> None:
    """Placeholder for Yelp Open Dataset ingest.

    Expected files: yelp_academic_dataset_business.json, yelp_academic_dataset_review.json,
    yelp_academic_dataset_user.json. Not implemented in this scaffold.
    """
    raise NotImplementedError(
        f"Yelp ingest is not implemented. Place dataset files in {raw_dir} and implement load_yelp."
    )


def index_users(users: List[User]) -> Dict[str, User]:
    return {u.user_id: u for u in users}


def index_restaurants(restaurants: List[Restaurant]) -> Dict[str, Restaurant]:
    return {r.restaurant_id: r for r in restaurants}

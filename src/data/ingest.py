"""Synthetic interaction data for a reproducible public path.

Replace `load_yelp` with a real Yelp Open Dataset ingest when the dump is available.
This generator is deterministic given `seed`.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from data.schemas import Interaction, Restaurant, User

CUISINES = ("italian", "mexican", "japanese", "american", "indian", "thai", "chinese", "korean")


def generate_synthetic(
    n_users: int = 200,
    n_restaurants: int = 80,
    n_requests: int = 400,
    seed: int = 7,
    start: Optional[datetime] = None,
) -> Tuple[List[User], List[Restaurant], List[Interaction]]:
    rng = random.Random(seed)
    start = start or datetime(2025, 1, 1, 12, 0, 0)
    # Downtown Los Angeles-ish box for demo geo.
    users = [
        User(
            user_id=f"user_{i}",
            latitude=34.02 + rng.uniform(-0.12, 0.12),
            longitude=-118.29 + rng.uniform(-0.12, 0.12),
            account_age_days=rng.randint(1, 1200),
            historical_interaction_count=0,
            historical_booking_count=0,
            average_price_preference=float(rng.choice([1, 2, 3, 4])),
        )
        for i in range(n_users)
    ]
    restaurants = []
    for j in range(n_restaurants):
        hours = {str(d): list(range(10, 23)) for d in range(7)}
        restaurants.append(
            Restaurant(
                restaurant_id=f"r{j}",
                latitude=34.02 + rng.uniform(-0.10, 0.10),
                longitude=-118.29 + rng.uniform(-0.10, 0.10),
                cuisine=rng.choice(CUISINES),
                price_level=rng.choice([1, 2, 3, 4]),
                rating=round(rng.uniform(3.2, 4.9), 2),
                review_count=rng.randint(20, 4000),
                popularity_score=round(rng.uniform(0.1, 1.0), 3),
                opening_hours=hours,
                capacity=rng.randint(20, 80),
            )
        )

    interactions: List[Interaction] = []
    for k in range(n_requests):
        user = rng.choice(users)
        ts = start + timedelta(hours=k * 3)
        request_id = f"req_{k}"
        shown = rng.sample(restaurants, k=min(10, len(restaurants)))
        for pos, rest in enumerate(shown, start=1):
            interactions.append(
                Interaction(
                    user_id=user.user_id,
                    restaurant_id=rest.restaurant_id,
                    timestamp=ts,
                    event_type="impression",
                    position=pos,
                    experiment_variant="logging",
                    request_id=request_id,
                )
            )
            click_p = 0.22 * math.exp(-(pos - 1) / 4.0)
            if rng.random() < click_p:
                interactions.append(
                    Interaction(
                        user_id=user.user_id,
                        restaurant_id=rest.restaurant_id,
                        timestamp=ts + timedelta(minutes=2),
                        event_type="click",
                        position=pos,
                        experiment_variant="logging",
                        request_id=request_id,
                    )
                )
                if rng.random() < 0.18:
                    interactions.append(
                        Interaction(
                            user_id=user.user_id,
                            restaurant_id=rest.restaurant_id,
                            timestamp=ts + timedelta(minutes=8),
                            event_type="booking",
                            position=pos,
                            experiment_variant="logging",
                            request_id=request_id,
                        )
                    )
                    if rng.random() < 0.75:
                        interactions.append(
                            Interaction(
                                user_id=user.user_id,
                                restaurant_id=rest.restaurant_id,
                                timestamp=ts + timedelta(hours=2),
                                event_type="completed_reservation",
                                position=pos,
                                experiment_variant="logging",
                                request_id=request_id,
                            )
                        )
                    else:
                        interactions.append(
                            Interaction(
                                user_id=user.user_id,
                                restaurant_id=rest.restaurant_id,
                                timestamp=ts + timedelta(hours=1),
                                event_type="cancelled_reservation",
                                position=pos,
                                experiment_variant="logging",
                                request_id=request_id,
                            )
                        )
    return users, restaurants, interactions


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

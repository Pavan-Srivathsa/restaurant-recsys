from datetime import datetime

from data.schemas import Interaction, Restaurant, User


def origin_user():
    return User(user_id="u0", latitude=34.0522, longitude=-118.2437)


def nearby_restaurant():
    return Restaurant(
        restaurant_id="r0",
        latitude=34.0522,
        longitude=-118.2437,
        cuisine="mexican",
        price_level=2,
        rating=4.2,
        review_count=100,
        popularity_score=0.5,
        opening_hours={str(d): list(range(10, 22)) for d in range(7)},
        capacity=40,
    )


def make_interaction(user_id, restaurant_id, ts, event_type="click", position=1):
    return Interaction(
        user_id=user_id,
        restaurant_id=restaurant_id,
        timestamp=ts or datetime(2025, 1, 1),
        event_type=event_type,
        position=position,
        request_id="req",
    )

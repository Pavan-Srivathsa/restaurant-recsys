from datetime import datetime

from data.availability import SIMULATED, availability_probability, is_open
from data.ingest import generate_synthetic
from data.schemas import Restaurant
from retrieval.candidates import generate_candidates


def test_availability_is_marked_simulated():
    assert SIMULATED is True


def test_full_occupancy_has_zero_probability():
    assert availability_probability(capacity=10, hour=12, weekday=5, popularity=1.0, existing_bookings=10) == 0.0


def test_offpeak_more_available_than_peak():
    off = availability_probability(40, hour=15, weekday=2, popularity=0.5, existing_bookings=5)
    peak = availability_probability(40, hour=19, weekday=5, popularity=0.5, existing_bookings=5)
    assert off > peak


def test_closed_restaurant_not_a_candidate():
    rest = Restaurant(
        restaurant_id="closed",
        latitude=34.05,
        longitude=-118.25,
        cuisine="thai",
        price_level=2,
        rating=4.0,
        review_count=10,
        popularity_score=0.2,
        opening_hours={str(d): [] for d in range(7)},
        capacity=30,
    )
    ts = datetime(2025, 3, 4, 18, 0, 0)
    assert is_open(rest.opening_hours, ts) is False
    cands = generate_candidates([rest], [], 34.05, -118.25, ts)
    assert cands == []


def test_far_away_restaurant_filtered():
    rest = Restaurant(
        restaurant_id="far",
        latitude=40.71,
        longitude=-74.01,
        cuisine="italian",
        price_level=3,
        rating=4.5,
        review_count=10,
        popularity_score=0.9,
        opening_hours={str(d): list(range(0, 24)) for d in range(7)},
        capacity=50,
    )
    ts = datetime(2025, 3, 4, 18, 0, 0)
    cands = generate_candidates([rest], [], 34.05, -118.25, ts, radius_km=15)
    assert cands == []


def test_synthetic_candidates_respect_limit():
    users, restaurants, interactions = generate_synthetic(seed=1)
    user = users[0]
    ts = datetime(2025, 2, 1, 19, 0, 0)
    cands = generate_candidates(restaurants, interactions, user.latitude, user.longitude, ts, limit=12)
    assert len(cands) <= 12
    assert all(c.restaurant_id.startswith("r") for c in cands)

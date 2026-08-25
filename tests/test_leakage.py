from datetime import datetime, timedelta

import pytest

from data.build_features import cuisine_affinity, previous_counts, restaurant_window_stats
from data.schemas import Interaction
from data.splits import SplitError, assert_strictly_before, filter_as_of, split_by_time


def test_assert_strictly_before_rejects_equal_timestamps():
    ts = datetime(2025, 1, 10, 12, 0, 0)
    with pytest.raises(SplitError):
        assert_strictly_before(ts, ts)


def test_assert_strictly_before_rejects_future_events():
    rec = datetime(2025, 1, 10, 12, 0, 0)
    with pytest.raises(SplitError):
        assert_strictly_before(rec + timedelta(seconds=1), rec)


def test_filter_as_of_drops_future_and_equal():
    rec = datetime(2025, 6, 1, 12, 0, 0)
    events = [
        Interaction("u", "r", rec - timedelta(days=1), "click"),
        Interaction("u", "r", rec, "click"),
        Interaction("u", "r", rec + timedelta(hours=1), "booking"),
    ]
    kept = filter_as_of(events, rec)
    assert len(kept) == 1
    assert kept[0].timestamp < rec


def test_every_feature_event_precedes_recommendation_timestamp():
    rec = datetime(2025, 6, 15, 18, 0, 0)
    interactions = [
        Interaction("u", "r1", rec - timedelta(days=10), "click"),
        Interaction("u", "r1", rec - timedelta(days=3), "booking"),
        Interaction("u", "r2", rec + timedelta(days=1), "completed_reservation"),
    ]
    past = filter_as_of(interactions, rec)
    for event in past:
        assert_strictly_before(event.timestamp, rec, label="interaction")
        assert event.event_type != "completed_reservation"


def test_cuisine_affinity_ignores_future_bookings():
    rec = datetime(2025, 6, 15, 18, 0, 0)
    interactions = [
        Interaction("u", "italian_place", rec - timedelta(days=2), "completed_reservation"),
        Interaction("u", "thai_place", rec + timedelta(days=1), "completed_reservation"),
    ]
    cuisine = {"italian_place": "italian", "thai_place": "thai"}
    aff_italian = cuisine_affinity(interactions, cuisine, rec, "italian")
    aff_thai = cuisine_affinity(interactions, cuisine, rec, "thai")
    assert aff_italian == 1.0
    assert aff_thai == 0.0


def test_window_stats_do_not_include_future_bookings():
    rec = datetime(2025, 6, 15, 18, 0, 0)
    interactions = [
        Interaction("u", "r", rec - timedelta(days=2), "booking"),
        Interaction("u", "r", rec + timedelta(days=1), "booking"),
    ]
    stats = restaurant_window_stats(interactions, "r", rec)
    assert stats["bookings_last_7_days"] == 1.0


def test_previous_counts_as_of():
    rec = datetime(2025, 6, 15, 18, 0, 0)
    interactions = [
        Interaction("u", "r", rec - timedelta(days=1), "click"),
        Interaction("u", "r", rec + timedelta(minutes=1), "click"),
    ]
    counts = previous_counts(interactions, "u", "r", rec)
    assert counts["previous_clicks"] == 1.0


def test_temporal_split_is_ordered():
    base = datetime(2025, 1, 1)
    rows = list(range(100))
    timestamps = [base + timedelta(days=i) for i in rows]
    train, valid, test = split_by_time(rows, timestamps, 0.70, 0.15)
    assert max(train) < min(valid)
    assert max(valid) < min(test)
    assert len(train) + len(valid) + len(test) == 100

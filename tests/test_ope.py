import numpy as np

from data.ingest import generate_world
from evaluation.ope import booking_outcome, evaluate_policy, train_outcome_model
from evaluation.propensities import display_propensities, examination_propensity
from ranking.baseline import position_propensity
from ranking.dataset import build_ltr_frame, logged_groups, split_frame
from ranking.train import train_lambdamart


def test_display_propensity_exploit_is_one():
    ids = [f"r{i}" for i in range(20)]
    p = display_propensities(ids, n_show=10, n_exploit=8)
    for i in range(8):
        assert p[f"r{i}"] == 1.0
    remainder = 12
    extra = 2
    expected = extra / remainder
    for i in range(8, 20):
        assert abs(p[f"r{i}"] - expected) < 1e-12


def test_display_propensity_all_shown_when_catalog_small():
    ids = ["a", "b", "c"]
    p = display_propensities(ids, n_show=10, n_exploit=8)
    assert p == {"a": 1.0, "b": 1.0, "c": 1.0}


def test_examination_matches_position_propensity():
    assert examination_propensity(1) == position_propensity(1)
    assert examination_propensity(10) < examination_propensity(1)


def test_booking_outcome_threshold():
    np.testing.assert_array_equal(booking_outcome([0, 1, 2, 3]), [0, 0, 1, 1])


def test_ope_pipeline_returns_required_keys():
    world = generate_world(n_users=40, n_restaurants=24, n_requests=90, seed=8)
    frame = build_ltr_frame(world.users, world.restaurants, world.interactions, world.requests, world.prefs)
    train, valid, test = split_frame(frame)
    model = train_lambdamart(
        logged_groups(train),
        valid_frame=logged_groups(valid),
        num_boost_round=20,
        min_data_in_leaf=5,
        seed=8,
    )
    outcome = train_outcome_model(train, seed=8)
    report = evaluate_policy(test, model, outcome, max_weight=20.0)
    required = {
        "naive_overlap",
        "ips",
        "clipped_ips",
        "doubly_robust",
        "oracle",
        "effective_sample_size",
        "logging_on_policy",
        "direct_method",
    }
    assert required <= set(report)
    assert report["effective_sample_size"] <= report["n_logged_weights"] + 1e-9
    assert report["max_raw_weight"] >= report["mean_clipped_weight"]
    assert report["n_requests"] > 0

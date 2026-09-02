from datetime import datetime

import numpy as np

from data.build_features import cuisine_map, price_map
from data.index import InteractionIndex
from data.ingest import LOGGING_AVAILABILITY, eligible_restaurants, generate_world, logging_score
from data.schemas import Interaction
from evaluation.report import format_experiment_table
from experiments.analyze import analyze_experiment
from experiments.assignment import assign_variant
from experiments.simulate import rank_personalized_slate, rank_popularity_slate, simulate_experiment
from ranking.train import FEATURE_COLUMNS


class _AffinityRanker:
    def predict(self, matrix):
        idx = FEATURE_COLUMNS.index("cuisine_affinity")
        return matrix[:, idx] * 20.0 + matrix[:, FEATURE_COLUMNS.index("distance_km")] * -0.05


def test_control_slate_matches_logging_score_order():
    world = generate_world(n_users=30, n_restaurants=25, n_requests=40, seed=4)
    user = world.users[0]
    ts = world.requests[0].timestamp
    eligible = eligible_restaurants(world.restaurants, user.latitude, user.longitude, ts)
    shown = rank_popularity_slate(user, eligible)
    expected = sorted(
        eligible,
        key=lambda r: (
            -logging_score(r, user.latitude, user.longitude, LOGGING_AVAILABILITY),
            r.restaurant_id,
        ),
    )[:10]
    assert [r.restaurant_id for r in shown] == [r.restaurant_id for r in expected]


def test_treatment_can_differ_from_control():
    world = generate_world(n_users=40, n_restaurants=30, n_requests=80, seed=5)
    index = InteractionIndex(world.interactions)
    cuisine = cuisine_map(world.restaurants)
    price = price_map(world.restaurants)
    model = _AffinityRanker()
    differed = False
    for user in world.users[:20]:
        ts = world.requests[0].timestamp
        eligible = eligible_restaurants(world.restaurants, user.latitude, user.longitude, ts)
        if len(eligible) < 8:
            continue
        control = [r.restaurant_id for r in rank_popularity_slate(user, eligible)]
        treatment = [
            r.restaurant_id
            for r in rank_personalized_slate(user, eligible, index, ts, cuisine, price, model)
        ]
        if control != treatment:
            differed = True
            break
    assert differed


def test_assignment_is_stable_for_experiment_users():
    world = generate_world(n_users=40, n_restaurants=24, n_requests=50, seed=6)
    run = simulate_experiment(world, _AffinityRanker(), n_requests=80, seed=11)
    for user in world.users:
        assert run.variants[user.user_id] == assign_variant(user.user_id)
    user_ids = {req.user_id for req in run.requests}
    assert user_ids
    for uid in user_ids:
        seen = {run.variants[req.user_id] for req in run.requests if req.user_id == uid}
        assert len(seen) == 1


def test_experiment_starts_after_logging_window():
    world = generate_world(n_users=30, n_restaurants=20, n_requests=40, seed=8)
    last_log = max(r.timestamp for r in world.requests)
    run = simulate_experiment(world, _AffinityRanker(), n_requests=40, seed=11)
    assert run.requests
    assert min(r.timestamp for r in run.requests) > last_log
    assert all(e.experiment_variant in {"control", "treatment"} for e in run.interactions)


def test_both_arms_are_logged():
    world = generate_world(n_users=50, n_restaurants=24, n_requests=60, seed=9)
    run = simulate_experiment(world, _AffinityRanker(), n_requests=120, seed=11)
    variants = {e.experiment_variant for e in run.interactions}
    assert variants == {"control", "treatment"}


def test_analyze_report_keys_and_cuped_mean():
    world = generate_world(n_users=50, n_restaurants=24, n_requests=70, seed=10)
    run = simulate_experiment(world, _AffinityRanker(), n_requests=160, seed=11)
    report = analyze_experiment(world, run)
    for key in (
        "primary",
        "cuped",
        "hte",
        "segments",
        "secondary",
        "guardrails",
        "power",
        "n_exposed",
    ):
        assert key in report
    assert report["n_control"] > 0 and report["n_treatment"] > 0
    assert abs(report["cuped"]["mean_y"] - report["cuped"]["mean_y_cuped"]) < 1e-9
    table = format_experiment_table(report)
    assert "Completed reservations" in table
    assert "CUPED-adjusted" in table
    assert "underpowered" in table
    for name in ("new", "returning", "high_frequency", "low_frequency"):
        assert name in report["segments"]


def test_index_append_keeps_as_of_order():
    early = Interaction(
        user_id="u0",
        restaurant_id="r0",
        timestamp=datetime(2025, 1, 1, 12, 0, 0),
        event_type="click",
        request_id="a",
    )
    later = Interaction(
        user_id="u0",
        restaurant_id="r0",
        timestamp=datetime(2025, 1, 2, 12, 0, 0),
        event_type="booking",
        request_id="b",
    )
    index = InteractionIndex([later])
    index.append([early])
    past = index.user_as_of("u0", datetime(2025, 1, 1, 18, 0, 0))
    assert [e.event_type for e in past] == ["click"]
    assert index.user_as_of("u0", datetime(2025, 1, 3))[-1].event_type == "booking"


def test_cuped_mean_preserved_on_experiment_outcomes():
    world = generate_world(n_users=40, n_restaurants=20, n_requests=50, seed=2)
    run = simulate_experiment(world, _AffinityRanker(), n_requests=80, seed=3)
    report = analyze_experiment(world, run)
    pooled = (
        report["primary"]["control_mean"] * report["primary"]["n_control"]
        + report["primary"]["treatment_mean"] * report["primary"]["n_treatment"]
    ) / (report["primary"]["n_control"] + report["primary"]["n_treatment"])
    assert abs(report["cuped"]["mean_y"] - pooled) < 1e-9
    assert abs(report["cuped"]["mean_y"] - report["cuped"]["mean_y_cuped"]) < 1e-9
    np.testing.assert_allclose(report["cuped"]["mean_y"], report["cuped"]["mean_y_cuped"], atol=1e-9)

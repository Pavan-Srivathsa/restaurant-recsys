from datetime import datetime

from data.ingest import generate_synthetic
from ranking.baseline import rank_baseline
from retrieval.candidates import candidate_availability_map, generate_candidates


def test_baseline_ranking_is_deterministic():
    users, restaurants, interactions = generate_synthetic(seed=3)
    user = users[0]
    ts = datetime(2025, 6, 1, 18, 0, 0)
    candidates = generate_candidates(restaurants, interactions, user.latitude, user.longitude, ts)
    availability = candidate_availability_map(candidates, interactions, ts)
    first = rank_baseline(candidates, user.latitude, user.longitude, availability, top_k=10)
    second = rank_baseline(candidates, user.latitude, user.longitude, availability, top_k=10)
    assert [(r.restaurant_id, s) for r, s in first] == [(r.restaurant_id, s) for r, s in second]


def test_baseline_tie_break_uses_restaurant_id():
    users, restaurants, interactions = generate_synthetic(seed=3)
    user = users[1]
    ts = datetime(2025, 6, 1, 12, 0, 0)
    candidates = generate_candidates(restaurants, interactions, user.latitude, user.longitude, ts)
    availability = {c.restaurant_id: 1.0 for c in candidates}
    ranked = rank_baseline(candidates, user.latitude, user.longitude, availability, top_k=len(candidates))
    scores = [s for _, s in ranked]
    for i in range(len(ranked) - 1):
        if abs(scores[i] - scores[i + 1]) < 1e-12:
            assert ranked[i][0].restaurant_id < ranked[i + 1][0].restaurant_id

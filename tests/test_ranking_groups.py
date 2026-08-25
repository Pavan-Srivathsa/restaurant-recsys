from ranking.train import FEATURE_COLUMNS, group_sizes, lgbm_ranker_params


def test_group_sizes_contiguous_requests():
    assert group_sizes(["a", "a", "a", "b", "b", "c"]) == [3, 2, 1]


def test_group_sizes_empty():
    assert group_sizes([]) == []


def test_lambdamart_params_are_deterministic():
    p = lgbm_ranker_params(seed=7)
    assert p["objective"] == "lambdarank"
    assert p["deterministic"] is True
    assert p["seed"] == 7


def test_feature_columns_cover_user_restaurant_and_context():
    names = set(FEATURE_COLUMNS)
    assert "cuisine_affinity" in names
    assert "price_match" in names
    assert "distance_km" in names
    assert "hour" in names
    assert "reservation_availability" in names

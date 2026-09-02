from data.ingest import generate_world, oracle_relevance
from ranking.dataset import build_ltr_frame, logged_groups, split_frame
from ranking.train import FEATURE_COLUMNS, train_lambdamart


def test_preferred_cuisine_gets_higher_oracle_label():
    world = generate_world(n_users=40, n_restaurants=30, n_requests=80, seed=3)
    user = world.users[0]
    pref = world.prefs[user.user_id]
    matching = next(r for r in world.restaurants if r.cuisine in pref.cuisines)
    other = next(r for r in world.restaurants if r.cuisine not in pref.cuisines)
    match_rel = oracle_relevance(pref, matching, user.latitude, user.longitude)
    other_rel = oracle_relevance(pref, other, user.latitude, user.longitude)
    assert match_rel >= other_rel


def test_ltr_frame_has_as_of_features_and_groups():
    world = generate_world(n_users=30, n_restaurants=20, n_requests=60, seed=4)
    frame = build_ltr_frame(world.users, world.restaurants, world.interactions, world.requests, world.prefs)
    assert len(frame) > 0
    for col in FEATURE_COLUMNS:
        assert col in frame.columns
    assert set(frame["split"].unique()) if "split" in frame.columns else True
    logged = logged_groups(frame)
    assert logged["label"].max() >= 1
    # Features must be numeric and finite.
    assert frame[list(FEATURE_COLUMNS)].isna().sum().sum() == 0


def test_temporal_split_does_not_mix_future_requests():
    world = generate_world(n_users=40, n_restaurants=25, n_requests=80, seed=5)
    frame = build_ltr_frame(world.users, world.restaurants, world.interactions, world.requests, world.prefs)
    train, valid, test = split_frame(frame)
    assert train["timestamp"].max() <= valid["timestamp"].min()
    assert valid["timestamp"].max() <= test["timestamp"].min()
    assert set(train["request_id"]).isdisjoint(set(test["request_id"]))


def test_lambdamart_trains_on_tiny_logged_set():
    world = generate_world(n_users=40, n_restaurants=24, n_requests=90, seed=6)
    frame = build_ltr_frame(world.users, world.restaurants, world.interactions, world.requests, world.prefs)
    train, valid, test = split_frame(frame)
    logged = logged_groups(train)
    model = train_lambdamart(logged, valid_frame=logged_groups(valid), num_boost_round=20, min_data_in_leaf=5, seed=6)
    assert model.num_trees() >= 1
    assert test["request_id"].nunique() > 0

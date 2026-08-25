from data.geo import distance_features, haversine_km


def test_haversine_zero_for_same_point():
    assert haversine_km(34.05, -118.25, 34.05, -118.25) == 0.0


def test_haversine_known_la_to_santa_monica():
    # Downtown LA to Santa Monica is roughly 25 km.
    km = haversine_km(34.0522, -118.2437, 34.0195, -118.4912)
    assert 20.0 < km < 30.0


def test_haversine_symmetric():
    a = haversine_km(40.71, -74.01, 34.05, -118.25)
    b = haversine_km(34.05, -118.25, 40.71, -74.01)
    assert abs(a - b) < 1e-9


def test_distance_indicators():
    feats = distance_features(2.5)
    assert feats["within_1km"] == 0.0
    assert feats["within_3km"] == 1.0
    assert feats["within_5km"] == 1.0
    assert feats["log_distance"] > 0

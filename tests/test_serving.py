
from fastapi.testclient import TestClient

from serving.app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert r.json()["availability_simulated"] is True


def test_recommend_returns_top_10_and_variant():
    r = client.post(
        "/recommend",
        json={"user_id": "user_1", "latitude": 34.021, "longitude": -118.289},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["experiment"] in {"control", "treatment"}
    assert 1 <= len(body["restaurants"]) <= 10
    ranks = [row["rank"] for row in body["restaurants"]]
    assert ranks == list(range(1, len(ranks) + 1))
    assert body["availability_simulated"] is True


def test_same_user_keeps_variant():
    payloads = [
        client.post(
            "/recommend",
            json={"user_id": "user_42", "latitude": 34.02, "longitude": -118.29},
        ).json()
        for _ in range(5)
    ]
    variants = {p["experiment"] for p in payloads}
    assert len(variants) == 1

"""Recommendation API.

Control: contextual popularity.
Treatment: personalized linear stand-in until LambdaMART is trained and loaded.
Availability is simulated. Impressions are logged in process.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from data.build_features import build_pair_features, cuisine_map, price_map
from data.ingest import generate_synthetic, index_restaurants, index_users
from data.schemas import User
from experiments.assignment import assign_variant
from ranking.baseline import rank_baseline
from ranking.predict import rank_by_scores
from retrieval.candidates import candidate_availability_map, generate_candidates
from serving.logging import EventLogger

DEFAULT_REQUEST_TS = datetime(2025, 6, 4, 19, 0, 0)
MODEL_VERSION = "ranker_v1"
EXPERIMENT_NAME = "personalized_ranker_v1"
TOP_K = 10

app = FastAPI(title="restaurant-recsys", version="0.1.0")
logger = EventLogger()

_users, _restaurants, _interactions = generate_synthetic()
USERS = index_users(_users)
RESTAURANTS = index_restaurants(_restaurants)
CUISINE = cuisine_map(_restaurants)
PRICE = price_map(_restaurants)


class RecommendRequest(BaseModel):
    user_id: str
    latitude: float
    longitude: float
    timestamp: Optional[datetime] = None


class RankedRestaurant(BaseModel):
    restaurant_id: str
    rank: int
    score: float


class RecommendResponse(BaseModel):
    request_id: str
    experiment: str
    model_version: str
    restaurants: List[RankedRestaurant]
    availability_simulated: bool = Field(default=True)


def _user_or_guest(user_id: str, latitude: float, longitude: float) -> User:
    if user_id in USERS:
        u = USERS[user_id]
        return User(
            user_id=u.user_id,
            latitude=latitude,
            longitude=longitude,
            account_age_days=u.account_age_days,
            historical_interaction_count=u.historical_interaction_count,
            historical_booking_count=u.historical_booking_count,
            average_price_preference=u.average_price_preference,
        )
    return User(user_id=user_id, latitude=latitude, longitude=longitude)


def _personalized_scores(user, candidates, availability, timestamp):
    scores = []
    for restaurant in candidates:
        feats = build_pair_features(
            user, restaurant, _interactions, timestamp, availability[restaurant.restaurant_id], CUISINE, PRICE
        )
        score = (
            0.35 * feats["restaurant_popularity"]
            + 0.20 * (feats["restaurant_rating"] / 5.0)
            + 0.15 * feats["distance_score"]
            + 0.10 * feats["reservation_availability"]
            + 0.12 * feats["cuisine_affinity"]
            + 0.08 * feats["price_match"]
        )
        scores.append(score)
    return scores


@app.get("/health")
def health():
    return {"ok": True, "restaurants": len(RESTAURANTS), "availability_simulated": True}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(body: RecommendRequest) -> RecommendResponse:
    ts = body.timestamp or DEFAULT_REQUEST_TS
    variant = assign_variant(body.user_id, EXPERIMENT_NAME)
    user = _user_or_guest(body.user_id, body.latitude, body.longitude)
    candidates = generate_candidates(
        list(RESTAURANTS.values()), _interactions, body.latitude, body.longitude, ts
    )
    availability = candidate_availability_map(candidates, _interactions, ts)
    if variant == "control":
        ranked = rank_baseline(candidates, body.latitude, body.longitude, availability, top_k=TOP_K)
        model_version = "contextual_popularity_v1"
    else:
        scores = _personalized_scores(user, candidates, availability, ts)
        ranked = rank_by_scores(candidates, scores, top_k=TOP_K)
        model_version = MODEL_VERSION

    request_id = logger.new_request_id()
    logger.log_impressions(request_id, body.user_id, ranked, model_version, variant, ts)
    payload = [
        RankedRestaurant(restaurant_id=r.restaurant_id, rank=i, score=round(s, 6))
        for i, (r, s) in enumerate(ranked, start=1)
    ]
    return RecommendResponse(
        request_id=request_id,
        experiment=variant,
        model_version=model_version,
        restaurants=payload,
        availability_simulated=True,
    )

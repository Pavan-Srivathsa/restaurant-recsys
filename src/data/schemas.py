from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

EVENT_TYPES = (
    "impression",
    "click",
    "booking",
    "completed_reservation",
    "cancelled_reservation",
)

EVENT_WEIGHTS = {
    "impression": 0.0,
    "click": 1.0,
    "booking": 3.0,
    "completed_reservation": 5.0,
    "cancelled_reservation": 0.0,
}

RELEVANCE_LABELS = {
    "impression": 0,
    "click": 1,
    "booking": 2,
    "completed_reservation": 3,
    "cancelled_reservation": 0,
}


@dataclass(frozen=True)
class User:
    user_id: str
    latitude: float
    longitude: float
    account_age_days: int = 0
    historical_interaction_count: int = 0
    historical_booking_count: int = 0
    average_price_preference: Optional[float] = None


@dataclass(frozen=True)
class Restaurant:
    restaurant_id: str
    latitude: float
    longitude: float
    cuisine: str
    price_level: int
    rating: float
    review_count: int
    popularity_score: float
    opening_hours: Dict[str, List[int]] = field(default_factory=dict)
    capacity: int = 40


@dataclass(frozen=True)
class Interaction:
    user_id: str
    restaurant_id: str
    timestamp: datetime
    event_type: str
    position: Optional[int] = None
    experiment_variant: Optional[str] = None
    request_id: Optional[str] = None


@dataclass(frozen=True)
class Impression:
    request_id: str
    user_id: str
    restaurant_id: str
    rank_position: int
    model_version: str
    propensity: float
    experiment_variant: str
    timestamp: datetime
    score: Optional[float] = None


@dataclass(frozen=True)
class RecommendationRequest:
    user_id: str
    latitude: float
    longitude: float
    timestamp: datetime
    request_id: str


@dataclass(frozen=True)
class UserPref:
    """Latent preferences used only by the simulator / eval oracle, never as model features."""

    user_id: str
    cuisines: tuple
    price: int


@dataclass(frozen=True)
class RequestLog:
    request_id: str
    user_id: str
    timestamp: datetime
    latitude: float
    longitude: float
    shown_ids: tuple

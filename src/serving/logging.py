"""Impression and interaction event logger.

In this scaffold events are stored in memory. The Postgres schema in sql/init.sql
is the production contract; a later phase will flush asynchronously to the DB.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from data.schemas import Impression, Interaction


class EventLogger:
    def __init__(self) -> None:
        self.impressions: List[Impression] = []
        self.interactions: List[Interaction] = []

    def log_impressions(
        self,
        request_id: str,
        user_id: str,
        ranked: List[tuple],
        model_version: str,
        experiment_variant: str,
        timestamp: datetime,
        propensities: Optional[List[float]] = None,
    ) -> List[Impression]:
        rows: List[Impression] = []
        for i, (restaurant, score) in enumerate(ranked, start=1):
            propensity = propensities[i - 1] if propensities else 1.0 / i
            row = Impression(
                request_id=request_id,
                user_id=user_id,
                restaurant_id=restaurant.restaurant_id,
                rank_position=i,
                model_version=model_version,
                propensity=float(propensity),
                experiment_variant=experiment_variant,
                timestamp=timestamp,
                score=float(score),
            )
            self.impressions.append(row)
            rows.append(row)
            self.interactions.append(
                Interaction(
                    user_id=user_id,
                    restaurant_id=restaurant.restaurant_id,
                    timestamp=timestamp,
                    event_type="impression",
                    position=i,
                    experiment_variant=experiment_variant,
                    request_id=request_id,
                )
            )
        return rows

    @staticmethod
    def new_request_id() -> str:
        return str(uuid4())

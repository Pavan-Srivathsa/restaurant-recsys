"""Geographic distance features."""

from __future__ import annotations

import math
from typing import Dict

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 points, in kilometers."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    a = min(1.0, max(0.0, a))
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def distance_features(distance_km: float) -> Dict[str, float]:
    return {
        "distance_km": float(distance_km),
        "log_distance": math.log1p(max(distance_km, 0.0)),
        "within_1km": 1.0 if distance_km <= 1.0 else 0.0,
        "within_3km": 1.0 if distance_km <= 3.0 else 0.0,
        "within_5km": 1.0 if distance_km <= 5.0 else 0.0,
        "within_10km": 1.0 if distance_km <= 10.0 else 0.0,
        "distance_score": math.exp(-distance_km / 5.0),
    }

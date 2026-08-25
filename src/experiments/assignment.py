"""Stable user-level experiment assignment.

hash(experiment_name, user_id) % 100
0–49  control
50–99 treatment

Assignment must not change across sessions for the same experiment.
"""

from __future__ import annotations

import hashlib
from typing import Literal

Variant = Literal["control", "treatment"]


def assignment_bucket(user_id: str, experiment_name: str = "personalized_ranker_v1") -> int:
    payload = f"{experiment_name}:{user_id}".encode()
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16) % 100


def assign_variant(user_id: str, experiment_name: str = "personalized_ranker_v1") -> Variant:
    bucket = assignment_bucket(user_id, experiment_name)
    if 0 <= bucket <= 49:
        return "control"
    return "treatment"


def is_treatment(user_id: str, experiment_name: str = "personalized_ranker_v1") -> bool:
    return assign_variant(user_id, experiment_name) == "treatment"

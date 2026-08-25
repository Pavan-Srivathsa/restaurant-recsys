"""Temporal train / validation / test split.

Recommendation data is ordered in time. A random row split leaks the future.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import List, Tuple, TypeVar

T = TypeVar("T")


class SplitError(ValueError):
    pass


def temporal_cutpoints(
    timestamps: Sequence[datetime],
    train_frac: float = 0.70,
    valid_frac: float = 0.15,
) -> Tuple[datetime, datetime]:
    if not timestamps:
        raise SplitError("no timestamps to split")
    if train_frac <= 0 or valid_frac <= 0 or train_frac + valid_frac >= 1:
        raise SplitError("fractions must be positive and train+valid < 1")
    ordered = sorted(timestamps)
    n = len(ordered)
    train_end_idx = max(0, min(n - 1, int(n * train_frac) - 1))
    valid_end_idx = max(train_end_idx, min(n - 1, int(n * (train_frac + valid_frac)) - 1))
    return ordered[train_end_idx], ordered[valid_end_idx]


def assign_split(ts: datetime, train_end: datetime, valid_end: datetime) -> str:
    if ts <= train_end:
        return "train"
    if ts <= valid_end:
        return "validation"
    return "test"


def split_by_time(
    rows: Sequence[T],
    timestamps: Sequence[datetime],
    train_frac: float = 0.70,
    valid_frac: float = 0.15,
) -> Tuple[List[T], List[T], List[T]]:
    if len(rows) != len(timestamps):
        raise SplitError("rows and timestamps must be the same length")
    train_end, valid_end = temporal_cutpoints(timestamps, train_frac, valid_frac)
    train, valid, test = [], [], []
    for row, ts in zip(rows, timestamps):
        bucket = assign_split(ts, train_end, valid_end)
        if bucket == "train":
            train.append(row)
        elif bucket == "validation":
            valid.append(row)
        else:
            test.append(row)
    return train, valid, test


def assert_strictly_before(event_ts: datetime, recommendation_ts: datetime, label: str = "event") -> None:
    if event_ts >= recommendation_ts:
        raise SplitError(
            f"temporal leakage: {label} timestamp {event_ts.isoformat()} "
            f"is not strictly before recommendation {recommendation_ts.isoformat()}"
        )


def filter_as_of(events: Iterable, recommendation_ts: datetime, ts_attr: str = "timestamp"):
    """Keep only events strictly before the recommendation timestamp."""
    kept = []
    for event in events:
        ts = event[ts_attr] if isinstance(event, dict) else getattr(event, ts_attr)
        if ts < recommendation_ts:
            kept.append(event)
    return kept

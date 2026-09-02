"""Build grouped learning-to-rank frames with as-of features."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping, Sequence, Tuple

import pandas as pd

from data.build_features import (
    cuisine_map,
    pair_features_from_snapshot,
    price_map,
    restaurant_snapshot_from_events,
    user_cohort,
    user_snapshot,
)
from data.index import InteractionIndex
from data.ingest import oracle_relevance
from data.schemas import RELEVANCE_LABELS, Interaction, RequestLog, Restaurant, User, UserPref
from data.splits import assign_split, temporal_cutpoints
from ranking.train import FEATURE_COLUMNS
from retrieval.candidates import candidate_availability_map, generate_candidates


def observed_labels_for_request(events: Sequence[Interaction]) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    for event in events:
        grade = RELEVANCE_LABELS.get(event.event_type, 0)
        rid = event.restaurant_id
        labels[rid] = max(labels.get(rid, 0), grade)
    return labels


def events_by_request(interactions: Sequence[Interaction]) -> Dict[str, List[Interaction]]:
    grouped: Dict[str, List[Interaction]] = defaultdict(list)
    for event in interactions:
        if event.request_id:
            grouped[event.request_id].append(event)
    return grouped


def build_request_rows(
    request: RequestLog,
    user: User,
    restaurants: Sequence[Restaurant],
    index: InteractionIndex,
    interactions: Sequence[Interaction],
    request_events: Sequence[Interaction],
    prefs: Mapping[str, UserPref],
    cuisine: Mapping[str, str],
    price: Mapping[str, float],
) -> List[dict]:
    candidates = generate_candidates(
        restaurants,
        interactions,
        request.latitude,
        request.longitude,
        request.timestamp,
        index=index,
    )
    if not candidates:
        return []
    availability = candidate_availability_map(candidates, interactions, request.timestamp, index=index)
    snapshot = user_snapshot(index, user.user_id, request.timestamp, cuisine, price)
    observed = observed_labels_for_request(request_events)
    shown = set(request.shown_ids)
    shown_pos = {rid: i + 1 for i, rid in enumerate(request.shown_ids)}
    pref = prefs.get(user.user_id)
    rows = []
    for restaurant in candidates:
        rest_past = index.restaurant_as_of(restaurant.restaurant_id, request.timestamp)
        rest_stats = restaurant_snapshot_from_events(rest_past, request.timestamp)
        feats = pair_features_from_snapshot(
            user,
            restaurant,
            snapshot,
            rest_stats,
            availability.get(restaurant.restaurant_id, 0.0),
        )
        oracle = 0
        if pref is not None:
            oracle = oracle_relevance(pref, restaurant, request.latitude, request.longitude)
        row = {
            "request_id": request.request_id,
            "user_id": user.user_id,
            "restaurant_id": restaurant.restaurant_id,
            "timestamp": request.timestamp,
            "logged": restaurant.restaurant_id in shown,
            "logged_position": shown_pos.get(restaurant.restaurant_id),
            "label": int(observed.get(restaurant.restaurant_id, 0)),
            "oracle_label": int(oracle),
            "cohort": user_cohort(int(snapshot["user_engagement_count"])),
            "local_booking_rate": rest_stats["historical_booking_rate"],
            "availability": availability.get(restaurant.restaurant_id, 0.0),
        }
        row.update(feats)
        rows.append(row)
    return rows


def build_ltr_frame(
    users: Sequence[User],
    restaurants: Sequence[Restaurant],
    interactions: Sequence[Interaction],
    requests: Sequence[RequestLog],
    prefs: Mapping[str, UserPref],
) -> pd.DataFrame:
    users_by_id = {u.user_id: u for u in users}
    cuisine = cuisine_map(restaurants)
    price = price_map(restaurants)
    index = InteractionIndex(interactions)
    by_request = events_by_request(interactions)
    rows: List[dict] = []
    for request in requests:
        user = users_by_id.get(request.user_id)
        if user is None:
            continue
        rows.extend(
            build_request_rows(
                request,
                user,
                restaurants,
                index,
                interactions,
                by_request.get(request.request_id, []),
                prefs,
                cuisine,
                price,
            )
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(["timestamp", "request_id", "restaurant_id"]).reset_index(drop=True)
    return frame


def split_frame(frame: pd.DataFrame, train_frac: float = 0.70, valid_frac: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    request_times = frame.groupby("request_id")["timestamp"].min().sort_values()
    train_end, valid_end = temporal_cutpoints(list(request_times.values), train_frac, valid_frac)
    mapping = {rid: assign_split(ts, train_end, valid_end) for rid, ts in request_times.items()}
    frame = frame.copy()
    frame["split"] = frame["request_id"].map(mapping)
    train = frame[frame["split"] == "train"].copy()
    valid = frame[frame["split"] == "validation"].copy()
    test = frame[frame["split"] == "test"].copy()
    return train, valid, test


def logged_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Training rows: logged impressions only, groups with at least one positive label."""
    logged = frame[frame["logged"]].copy()
    positives = logged.groupby("request_id")["label"].max()
    keep = set(positives[positives > 0].index)
    logged = logged[logged["request_id"].isin(keep)]
    return logged.sort_values(["request_id", "restaurant_id"]).reset_index(drop=True)


def missing_feature_columns(frame: pd.DataFrame) -> List[str]:
    return [c for c in FEATURE_COLUMNS if c not in frame.columns]

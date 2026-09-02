"""Analyze the simulated experiment: ATE, CUPED, HTE, secondaries, guardrails."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import yaml

from data.build_features import cuisine_map
from data.ingest import SyntheticWorld
from data.schemas import Interaction
from evaluation.ranking_metrics import catalog_diversity, coverage, mean_metric
from experiments.cuped import cuped_adjust, cuped_report
from experiments.power import two_proportion_sample_size
from experiments.simulate import ExperimentRun
from experiments.treatment_effects import ate, interaction_ols
from paths import repo_root

ENGAGEMENT_TYPES = {"click", "booking", "completed_reservation"}
HIGH_FREQUENCY_MIN = 6


def load_experiment_config(path: Optional[Path] = None) -> dict:
    artifact = path or (repo_root() / "configs" / "experiment.yaml")
    return yaml.safe_load(artifact.read_text())


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _count_by_user(events: Sequence[Interaction], event_type: str) -> Dict[str, int]:
    out: Dict[str, int] = defaultdict(int)
    for event in events:
        if event.event_type == event_type:
            out[event.user_id] += 1
    return out


def _engagement_by_user(events: Sequence[Interaction]) -> Dict[str, int]:
    out: Dict[str, int] = defaultdict(int)
    for event in events:
        if event.event_type in ENGAGEMENT_TYPES:
            out[event.user_id] += 1
    return out


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _ate_or_none(control: Sequence[float], treatment: Sequence[float]) -> Optional[dict]:
    if len(control) == 0 or len(treatment) == 0:
        return None
    return ate(control, treatment)


def _rate(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return num / float(den)


def analyze_experiment(
    world: SyntheticWorld,
    run: ExperimentRun,
    config: Optional[dict] = None,
) -> dict:
    cfg = config or load_experiment_config()
    power_cfg = cfg.get("power") or {}
    restaurants_by_id = {r.restaurant_id: r for r in world.restaurants}
    cuisine = cuisine_map(world.restaurants)

    pre_completed = _count_by_user(world.interactions, "completed_reservation")
    pre_engagement = _engagement_by_user(world.interactions)

    exposed: Dict[str, Dict[str, float]] = {}
    shown_ids: List[str] = []
    diversity_scores: List[float] = []
    arm_counts = {
        "control": defaultdict(int),
        "treatment": defaultdict(int),
    }

    for event in run.interactions:
        stats = exposed.setdefault(
            event.user_id,
            {
                "impressions": 0.0,
                "clicks": 0.0,
                "bookings": 0.0,
                "completed": 0.0,
                "cancelled": 0.0,
                "booking_value": 0.0,
                "n_booking_value": 0.0,
            },
        )
        variant = run.variants[event.user_id]
        arm_counts[variant][event.event_type] += 1
        if event.event_type == "impression":
            stats["impressions"] += 1
        elif event.event_type == "click":
            stats["clicks"] += 1
        elif event.event_type == "booking":
            stats["bookings"] += 1
        elif event.event_type == "completed_reservation":
            stats["completed"] += 1
            rest = restaurants_by_id.get(event.restaurant_id)
            if rest is not None:
                stats["booking_value"] += float(rest.price_level)
                stats["n_booking_value"] += 1
        elif event.event_type == "cancelled_reservation":
            stats["cancelled"] += 1

    for request in run.requests:
        shown_ids.extend(request.shown_ids)
        diversity_scores.append(catalog_diversity(list(request.shown_ids), cuisine, k=10))

    user_ids = list(exposed)
    if not user_ids:
        raise ValueError("experiment produced no exposed users")
    y = np.array([exposed[u]["completed"] for u in user_ids], dtype=float)
    x = np.array([float(pre_completed.get(u, 0)) for u in user_ids], dtype=float)
    treatment = np.array([1 if run.variants[u] == "treatment" else 0 for u in user_ids], dtype=int)
    returning = np.array([1 if pre_engagement.get(u, 0) > 0 else 0 for u in user_ids], dtype=int)
    high_freq = np.array([1 if pre_engagement.get(u, 0) >= HIGH_FREQUENCY_MIN else 0 for u in user_ids], dtype=int)
    y_bookings = np.array([exposed[u]["bookings"] for u in user_ids], dtype=float)
    y_clicks = np.array([exposed[u]["clicks"] for u in user_ids], dtype=float)
    y_converted = np.array([1.0 if exposed[u]["completed"] > 0 else 0.0 for u in user_ids], dtype=float)

    y_cuped = cuped_adjust(y, x)
    cuped_stats = cuped_report(y, x)

    control_mask = treatment == 0
    treat_mask = treatment == 1
    if int(control_mask.sum()) == 0 or int(treat_mask.sum()) == 0:
        raise ValueError("experiment needs exposed users in both arms")
    primary = ate(y[control_mask], y[treat_mask])
    primary_cuped = ate(y_cuped[control_mask], y_cuped[treat_mask])
    hte = interaction_ols(y, treatment, returning)

    def _segment(mask: np.ndarray) -> Optional[dict]:
        return _ate_or_none(y[control_mask & mask], y[treat_mask & mask])

    new_mask = returning == 0
    ret_mask = returning == 1
    low_mask = high_freq == 0
    hi_mask = high_freq == 1

    n_control = int(control_mask.sum())
    n_treatment = int(treat_mask.sum())
    n_exposed = n_control + n_treatment

    power = two_proportion_sample_size(
        float(power_cfg.get("baseline_conversion_rate", 0.08)),
        float(power_cfg.get("minimum_detectable_effect_relative", 0.06)),
        alpha=float(power_cfg.get("alpha", 0.05)),
        power=float(power_cfg.get("power", 0.80)),
    )
    underpowered = n_exposed < int(power["n_total"])

    def _arm_event_rate(variant: str, numerator: str, denominator: str) -> float:
        return _rate(arm_counts[variant][numerator], arm_counts[variant][denominator])

    def _arm_completion(variant: str) -> float:
        done = arm_counts[variant]["completed_reservation"]
        cancelled = arm_counts[variant]["cancelled_reservation"]
        return _rate(done, done + cancelled)

    booking_values = {
        "control": [],
        "treatment": [],
    }
    for uid, stats in exposed.items():
        if stats["n_booking_value"] > 0:
            booking_values[run.variants[uid]].append(stats["booking_value"] / stats["n_booking_value"])

    shown_by_arm: Dict[str, List[str]] = {"control": [], "treatment": []}
    for request in run.requests:
        shown_by_arm[run.variants[request.user_id]].extend(request.shown_ids)

    catalog_size = len(world.restaurants)
    payload = {
        "experiment_name": cfg.get("experiment_name", "personalized_ranker_v1"),
        "primary_metric": (cfg.get("primary_metric") or {}).get("name"),
        "n_attempted": run.n_attempted,
        "n_requests": len(run.requests),
        "n_exposed": n_exposed,
        "n_control": n_control,
        "n_treatment": n_treatment,
        "n_new": int(new_mask.sum()),
        "n_returning": int(ret_mask.sum()),
        "primary": primary,
        "cuped": {
            **primary_cuped,
            **cuped_stats,
        },
        "hte": hte,
        "segments": {
            "new": _segment(new_mask),
            "returning": _segment(ret_mask),
            "high_frequency": _segment(hi_mask),
            "low_frequency": _segment(low_mask),
        },
        "secondary": {
            "recommendation_ctr": {
                "control": _arm_event_rate("control", "click", "impression"),
                "treatment": _arm_event_rate("treatment", "click", "impression"),
            },
            "booking_initiation_rate": {
                "control": _arm_event_rate("control", "booking", "impression"),
                "treatment": _arm_event_rate("treatment", "booking", "impression"),
            },
            "reservation_completion_rate": {
                "control": _arm_completion("control"),
                "treatment": _arm_completion("treatment"),
            },
            "bookings_per_user": ate(y_bookings[control_mask], y_bookings[treat_mask]),
            "clicks_per_user": ate(y_clicks[control_mask], y_clicks[treat_mask]),
            "conversion_any_completion": ate(y_converted[control_mask], y_converted[treat_mask]),
            "average_booking_value": {
                "control": float(np.mean(booking_values["control"])) if booking_values["control"] else 0.0,
                "treatment": float(np.mean(booking_values["treatment"])) if booking_values["treatment"] else 0.0,
                "note": "mean restaurant price_level among users with a completed reservation",
            },
        },
        "guardrails": {
            "p95_recommendation_latency_ms": _percentile(run.ranking_ms, 95),
            "p95_latency_ms_control": _percentile(run.ranking_ms_by_variant.get("control", ()), 95),
            "p95_latency_ms_treatment": _percentile(run.ranking_ms_by_variant.get("treatment", ()), 95),
            "cancellation_rate": {
                "control": _rate(
                    arm_counts["control"]["cancelled_reservation"],
                    arm_counts["control"]["completed_reservation"] + arm_counts["control"]["cancelled_reservation"],
                ),
                "treatment": _rate(
                    arm_counts["treatment"]["cancelled_reservation"],
                    arm_counts["treatment"]["completed_reservation"] + arm_counts["treatment"]["cancelled_reservation"],
                ),
            },
            "restaurant_coverage": coverage(shown_ids, catalog_size),
            "restaurant_coverage_control": coverage(shown_by_arm["control"], catalog_size),
            "restaurant_coverage_treatment": coverage(shown_by_arm["treatment"], catalog_size),
            "recommendation_diversity": mean_metric(diversity_scores),
            "no_result_rate": _rate(run.no_result_count, run.n_attempted),
        },
        "power": {
            **power,
            "n_exposed": float(n_exposed),
            "underpowered": underpowered,
        },
        "availability_simulated": True,
    }
    return _jsonable(payload)

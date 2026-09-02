"""Write measured offline / experiment tables once artifacts exist.

Do not hard-code target lifts. Numbers come from data/processed/metrics.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from paths import processed_dir

PLACEHOLDER = "—"


def _fmt(value, digits=4):
    if value is None:
        return PLACEHOLDER
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value):
    if value is None:
        return PLACEHOLDER
    return f"{value * 100:.1f}%"


def _signed_pct(value):
    if value is None:
        return PLACEHOLDER
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.1f}%"


def _ci(low, high, digits=4):
    if low is None or high is None:
        return PLACEHOLDER
    return f"[{low:.{digits}f}, {high:.{digits}f}]"


def format_offline_tables(payload: dict) -> str:
    baseline = payload.get("baseline") or {}
    personal = payload.get("personalized") or {}
    lines = [
        "Offline ranking (temporal test split)",
        "",
        "| Model                 |  NDCG@10 | Recall@10 | Relative NDCG Lift | Relative Recall Lift |",
        "| --------------------- | -------: | --------: | -----------------: | -------------------: |",
        f"| Contextual popularity | {_fmt(baseline.get('ndcg@10'))} | {_fmt(baseline.get('recall@10'))} |           baseline |             baseline |",
        f"| Personalized ranker   | {_fmt(personal.get('ndcg@10'))} | {_fmt(personal.get('recall@10'))} | {_pct(payload.get('relative_ndcg_lift')):>18} | {_pct(payload.get('relative_recall_lift')):>20} |",
        "",
        f"Test requests: {int(baseline.get('n_requests') or 0)}  |  HitRate@10 baseline {_fmt(baseline.get('hit_rate@10'))} vs personalized {_fmt(personal.get('hit_rate@10'))}  |  MRR {_fmt(baseline.get('mrr'))} vs {_fmt(personal.get('mrr'))}",
        "",
        "Cohorts (NDCG@10)",
        "",
        "| Cohort    | N | Baseline | Personalized | Relative lift |",
        "| --------- | -: | -------: | -----------: | ------------: |",
    ]
    for name in ("new", "light", "returning", "heavy"):
        row = (payload.get("cohorts") or {}).get(name) or {}
        lines.append(
            f"| {name:<9} | {int(row.get('n_requests') or 0):,} | {_fmt(row.get('baseline_ndcg@10'))} | {_fmt(row.get('personalized_ndcg@10'))} | {_pct(row.get('ndcg_lift')):>13} |"
        )
    return "\n".join(lines)


def format_ope_table(ope: dict) -> str:
    if not ope:
        return "Counterfactual evaluation: not run."
    lines = [
        "Counterfactual evaluation (personalized top-10, popularity logs)",
        "",
        "| Estimator                         | Booking rate | |error| vs oracle |",
        "| --------------------------------- | -----------: | ---------------: |",
        f"| Naive overlap                     | {_fmt(ope.get('naive_overlap'))} | {_fmt(ope.get('abs_error_naive'))} |",
        f"| Direct method m(x,a)              | {_fmt(ope.get('direct_method'))} | {_fmt(ope.get('abs_error_dm'))} |",
        f"| IPS                               | {_fmt(ope.get('ips'))} | {_fmt(ope.get('abs_error_ips'))} |",
        f"| Clipped IPS (max weight {int(ope.get('max_weight') or 100)}) | {_fmt(ope.get('clipped_ips'))} | {_fmt(ope.get('abs_error_clipped_ips'))} |",
        f"| SNIPS (display)                   | {_fmt(ope.get('snips'))} | {_fmt(ope.get('abs_error_snips'))} |",
        f"| SNIPS (position × display)        | {_fmt(ope.get('exposure_snips'))} | {_fmt(ope.get('abs_error_exposure_snips'))} |",
        f"| Doubly robust                     | {_fmt(ope.get('doubly_robust'))} | {_fmt(ope.get('abs_error_dr'))} |",
        f"| Oracle (synthetic, full info)     | {_fmt(ope.get('oracle'))} |            — |",
        f"| Logging policy (on-policy)        | {_fmt(ope.get('logging_on_policy'))} |            — |",
        "",
        f"Effective sample size (display, clipped): {_fmt(ope.get('effective_sample_size'), 1)} / {int(ope.get('n_logged_weights') or 0)} logged impressions",
        f"Position-exposure ESS: {_fmt(ope.get('exposure_ess'), 1)}; max exposure weight {_fmt(ope.get('max_exposure_weight'), 1)}",
        f"Mean display weight raw {_fmt(ope.get('mean_raw_weight'), 2)} → clipped {_fmt(ope.get('mean_clipped_weight'), 2)}; max raw {_fmt(ope.get('max_raw_weight'), 1)}",
    ]
    return "\n".join(lines)


def _metric_row(label: str, row: dict | None) -> str:
    if not row:
        return f"| {label:<27} | {PLACEHOLDER:>7} | {PLACEHOLDER:>9} | {PLACEHOLDER:>6} | {PLACEHOLDER:>18} |"
    return (
        f"| {label:<27} | {_fmt(row.get('control_mean')):>7} | {_fmt(row.get('treatment_mean')):>9} | "
        f"{_signed_pct(row.get('relative_lift')):>6} | {_ci(row.get('ci_low'), row.get('ci_high')):>18} |"
    )


def format_experiment_table(experiment: dict) -> str:
    if not experiment:
        return "Simulated experiment: not run."
    primary = experiment.get("primary") or {}
    cuped = experiment.get("cuped") or {}
    segments = experiment.get("segments") or {}
    secondary = experiment.get("secondary") or {}
    guardrails = experiment.get("guardrails") or {}
    power = experiment.get("power") or {}
    hte = experiment.get("hte") or {}
    cancel = guardrails.get("cancellation_rate") or {}
    ctr = secondary.get("recommendation_ctr") or {}
    book_init = secondary.get("booking_initiation_rate") or {}
    complete = secondary.get("reservation_completion_rate") or {}
    booking_value = secondary.get("average_booking_value") or {}
    lines = [
        "Simulated online experiment (user-hashed assignment)",
        "",
        f"Primary metric (pre-registered): {experiment.get('primary_metric') or 'completed_reservations_per_exposed_user'}",
        "",
        "| Metric                      | Control | Treatment |   Lift |              95% CI |",
        "| --------------------------- | ------: | --------: | -----: | ------------------: |",
        _metric_row("Completed reservations", primary),
        _metric_row("CUPED-adjusted", cuped),
        _metric_row("New users", segments.get("new")),
        _metric_row("Returning users", segments.get("returning")),
        _metric_row("High-frequency users", segments.get("high_frequency")),
        _metric_row("Low-frequency users", segments.get("low_frequency")),
        _metric_row("Bookings per user", secondary.get("bookings_per_user")),
        "",
        f"Exposed users: {int(experiment.get('n_exposed') or 0)} "
        f"(control {int(experiment.get('n_control') or 0)}, "
        f"treatment {int(experiment.get('n_treatment') or 0)}; "
        f"new {int(experiment.get('n_new') or 0)}, "
        f"returning {int(experiment.get('n_returning') or 0)}) on "
        f"{int(experiment.get('n_requests') or 0)} experiment requests.",
        f"CUPED variance reduction: {_pct(cuped.get('variance_reduction'))} "
        f"(θ={_fmt(cuped.get('theta'), 3)}; mean preserved: "
        f"{_fmt(cuped.get('mean_y'))} → {_fmt(cuped.get('mean_y_cuped'))}).",
        f"HTE interaction (treatment × returning): {_fmt(hte.get('treatment_x_returning'))} "
        f"(p={_fmt(hte.get('treatment_x_returning_p'), 3)}).",
        "",
        "Secondaries (impression-weighted rates, not launch criteria)",
        "",
        f"CTR control {_fmt(ctr.get('control'))} vs treatment {_fmt(ctr.get('treatment'))}  |  "
        f"booking initiation {_fmt(book_init.get('control'))} vs {_fmt(book_init.get('treatment'))}  |  "
        f"completion {_fmt(complete.get('control'))} vs {_fmt(complete.get('treatment'))}",
        f"Average booking value (price level): control {_fmt(booking_value.get('control'), 2)} vs "
        f"treatment {_fmt(booking_value.get('treatment'), 2)}",
        "",
        "Guardrails",
        "",
        f"p95 ranking latency {_fmt(guardrails.get('p95_recommendation_latency_ms'), 1)} ms "
        f"(control {_fmt(guardrails.get('p95_latency_ms_control'), 1)}, "
        f"treatment {_fmt(guardrails.get('p95_latency_ms_treatment'), 1)})  |  "
        f"cancellation control {_fmt(cancel.get('control'))} vs treatment {_fmt(cancel.get('treatment'))}",
        f"Coverage {_fmt(guardrails.get('restaurant_coverage'))} "
        f"(control {_fmt(guardrails.get('restaurant_coverage_control'))}, "
        f"treatment {_fmt(guardrails.get('restaurant_coverage_treatment'))})  |  "
        f"diversity {_fmt(guardrails.get('recommendation_diversity'))}  |  "
        f"no-result rate {_fmt(guardrails.get('no_result_rate'))}",
        "",
        "Power (pre-registered two-proportion, α="
        f"{_fmt(power.get('alpha'), 2)}, power={_fmt(power.get('power'), 2)}, "
        f"baseline conversion {_fmt(power.get('baseline_rate'))}, "
        f"MDE relative {_pct(power.get('mde_relative'))})",
        f"Required n={int(power.get('n_total') or 0):,}  |  observed exposed n={int(power.get('n_exposed') or 0):,}  |  "
        f"{'underpowered — do not treat p-values as a launch decision' if power.get('underpowered') else 'powered for the pre-registered MDE'}",
    ]
    return "\n".join(lines)


def load_metrics(path: Path | None = None) -> dict | None:
    artifact = path or (processed_dir() / "metrics.json")
    if not artifact.exists():
        return None
    return json.loads(artifact.read_text())


def main() -> None:
    payload = load_metrics()
    if payload is None:
        print("no data/processed/metrics.json — run: PYTHONPATH=src python -m ranking.run")
        return
    print(format_offline_tables(payload))
    print()
    print(format_ope_table(payload.get("counterfactual") or {}))
    print()
    print(format_experiment_table(payload.get("experiment") or {}))
    if payload.get("availability_simulated"):
        print("\nReservation availability is simulated.")


if __name__ == "__main__":
    main()

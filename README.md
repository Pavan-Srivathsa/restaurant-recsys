# Restaurant RecSys

Personalized restaurant ranking with offline evaluation, counterfactual policy evaluation, and online experimentation.

The product question is not "which restaurants does this user like?" It is:

> Given the restaurants currently available to a user, which restaurants should we display and in what order to maximize the probability of a completed reservation?

## Status

**Phase 4 is complete** on the same synthetic world: a user-hashed A/B test of contextual popularity vs the trained ranker, with ATE, CUPED, and pre-registered heterogeneous effects.

**Reservation availability is simulated.** Real booking inventory is not used. Current ranking numbers are **not** from the Yelp Open Dataset.

Reproduce the tables:

```bash
PYTHONPATH=src python -m ranking.run
PYTHONPATH=src python -m evaluation.report
```

## Architecture

```text
Restaurant Dataset
        │
        ▼
Feature Pipeline  (as-of timestamps only; no future leakage)
        │
        ▼
Candidate Generator  (geo → hours → availability → retrieval)
        │  200–500 candidates
        ▼
Ranking Model  (contextual popularity vs personalized ranker)
        │  Top 10
        ▼
Recommendation API
        │
        ▼
Experiment Layer  (user-hashed control / treatment)
        │
        ▼
Event Logger  (impressions, clicks, bookings, completions)
```

## Evaluation layers

1. **Offline ranking:** NDCG@10, Recall@10, HitRate@10, MRR, coverage, catalog diversity.
2. **Online / simulated experiment:** completed reservations per exposed user, CUPED, heterogeneous effects.
3. **Counterfactual:** inverse propensity scoring and doubly robust estimation on logged impressions.

## Measured results

Targets in the project brief (approximately +11% NDCG@10, +8% Recall@10) are **not claimed**. The figures below are from `data/processed/metrics.json` after `python -m ranking.run` (seed=7, 400 users, 90 restaurants, 1400 requests, 209 test requests with at least one relevant restaurant).

### Offline ranking

| Model                 |  NDCG@10 | Recall@10 | Relative NDCG Lift | Relative Recall Lift |
| --------------------- | -------: | --------: | -----------------: | -------------------: |
| Contextual popularity |   0.2281 |    0.1554 |           baseline |             baseline |
| Personalized ranker   |   0.3822 |    0.2948 |             +67.5% |               +89.7% |

Also on the same test split: HitRate@10 0.799 → 0.818, MRR 0.374 → 0.580, catalog coverage 0.567 → 0.978.

### Offline ranking by cohort (NDCG@10)

Cohorts use as-of clicks/bookings: new = 0, light = 1–5, returning = 6–19, heavy = 20+.

| Cohort    | Test requests | Baseline | Personalized | Relative NDCG lift |
| --------- | ------------: | -------: | -----------: | -----------------: |
| new       |            10 |   0.2243 |       0.2384 |              +6.3% |
| light     |            10 |   0.1321 |       0.1826 |             +38.1% |
| returning |            71 |   0.1861 |       0.2947 |             +58.4% |
| heavy     |           118 |   0.2619 |       0.4639 |             +77.1% |

Personalization helps most where history exists. New-user NDCG lift is small; new-user Recall@10 was slightly worse than popularity (−3.8% relative), which is the cold-start tradeoff.

### Counterfactual evaluation

Logging policy: contextual popularity (always show top 8, sample 2 more). Target policy: personalized top-10. Reward: booking (`label >= 2`). Oracle is the full-information share of the target slate that is booking-relevant — it is **not** observed in production.

| Estimator | Booking rate | |error| vs oracle |
| --- | ---: | ---: |
| Naive overlap | 0.2862 | 0.0053 |
| Direct method m(x, a) | 0.2436 | 0.0479 |
| IPS | 0.2317 | 0.0598 |
| Clipped IPS (max weight 100) | 0.2317 | 0.0598 |
| SNIPS (display) | 0.2176 | 0.0738 |
| SNIPS (position × display) | 0.2161 | 0.0754 |
| Doubly robust | 0.2232 | 0.0683 |
| Oracle (synthetic) | 0.2914 | — |
| Logging policy (on-policy) | 0.2152 | — |

Naive overlap sits next to the oracle because it only scores restaurants both policies showed — mostly popular items. IPS / DR estimate the **logged booking process** for π, not the oracle: they sit near on-policy logging (0.215) with a modest lift to ~0.22–0.23. Display IPS and clipped IPS match because the max raw display weight was 40 (< 100). Position-exposure SNIPS hits the clip (max weight 100). Effective sample size drops from 2100 logged impressions to **166.7** under display IPS — that is the variance cost of reweighting.

### Experiment

Primary metric (pre-registered in `configs/experiment.yaml`): **completed reservations per exposed user**. Control is deterministic contextual-popularity top-10; treatment is LambdaMART top-10. Assignment is `hash(experiment_name:user_id) % 100` (0–49 control, 50–99 treatment). Eligible restaurants are the same in both arms. Features for treatment are as-of: only events before the request timestamp.

The experiment window is 2,800 requests after the logging period, on the same 400 users. 397 users were exposed (192 control, 205 treatment; 41 new / 356 returning). Returning = any pre-experiment click, booking, or completion; high-frequency = 6+ such events.

95% CIs below are on the **absolute** difference (treatment − control), not the relative lift. Do not read the brief’s ~6.5% online target into this table.

| Metric                      | Control | Treatment |   Lift |              95% CI |
| --------------------------- | ------: | --------: | -----: | ------------------: |
| Completed reservations      |  8.7969 |   19.5024 | +121.7% |  [5.4239, 15.9872] |
| CUPED-adjusted reservations |  9.3341 |   18.9993 | +103.5% |  [6.3742, 12.9563] |
| New users                   |  4.1176 |    9.9167 | +140.8% |   [2.1580, 9.4400] |
| Returning users             |  9.2514 |   20.7735 | +124.5% |  [5.5971, 17.4470] |

CUPED (pre-experiment completed reservations) cuts outcome variance by **60.6%** and does not change the sign of the ATE (θ = 3.266; pooled mean preserved). The HTE model `reservation ~ treatment + returning_user + treatment × returning` finds a large point estimate on the interaction (+5.72 completions) with **p = 0.533** — new and returning both gain; the extra returning-user lift is not cleanly identified.

This count metric compounds over multiple requests per user (~7). Almost every exposed user already completes at least one reservation (control 96.9% vs treatment 98.5%, relative +1.7%, CI includes 0). The experiment is picking up **more completions per user**, not a first-time conversion effect. CTR 0.319 → 0.433, booking initiation 0.212 → 0.336, completion given booking/cancel 0.653 → 0.754. Guardrails: p95 ranking latency 15.3 ms (treatment 17.4 ms), cancellation 0.347 → 0.245, catalog coverage 0.567 → 1.000, no-result rate 0.

Pre-registered power was for a **6% relative** lift on an **8%** conversion rate (α = 0.05, power = 0.80): required n = **103,030**. Observed exposed n = **397**, so the run is underpowered for that MDE. p-values are not a launch rule. The large synthetic lift is **not** a production claim — the simulator’s latent preferences are strong by design.

Reproduce with the same command as offline ranking:

```bash
PYTHONPATH=src python -m ranking.run
PYTHONPATH=src python -m evaluation.report
```

## Setup

Python 3.9+.

```bash
cd restaurant-recsys
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
pytest
```

Optional stack (Postgres + Redis + API):

```bash
docker compose up --build
```

The API listens on `http://localhost:8000`. Example:

```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"user_382","latitude":34.021,"longitude":-118.289}'
```

## Repository layout

```text
restaurant-recsys/
├── README.md
├── DESIGN.md
├── configs/                 experiment + ranking YAML
├── data/raw                 data/processed
├── src/
│   ├── data/                ingest, features, availability, geo, splits
│   ├── retrieval/           candidate generation
│   ├── ranking/             baseline + LambdaMART train/predict
│   ├── evaluation/          ranking metrics, IPS, doubly robust
│   ├── experiments/         assignment, CUPED, simulate, ATE / HTE
│   └── serving/             FastAPI + impression logging
├── notebooks/
├── tests/
└── sql/                     Postgres schema
```

## Implementation phases

| Phase | Focus | Status |
| ----- | ----- | ------ |
| 1 | Data contracts, simulated availability, contextual popularity, metrics | Done |
| 2 | User preference features, LambdaMART, temporal + cohort evaluation | Done (synthetic) |
| 3 | Position propensities, IPS, clipping, doubly robust | Done (synthetic) |
| 4 | Assignment, event log, ATE, CUPED, HTE | Done (synthetic) |
| 5 | FastAPI, Docker, benchmark report | API loads ranker_v1 when present |

## Tests

Unit tests cover temporal leakage, Haversine distance, experiment assignment stability, CUPED, NDCG, IPS weighting, ranking determinism, availability filtering, off-policy evaluation, and the simulated A/B (both arms, as-of window, CUPED mean preservation).

```bash
pytest
PYTHONPATH=src python -m ranking.run
```

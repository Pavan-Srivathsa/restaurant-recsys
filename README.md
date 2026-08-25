# Restaurant RecSys

Personalized restaurant ranking with offline evaluation, counterfactual policy evaluation, and online experimentation.

The product question is not "which restaurants does this user like?" It is:

> Given the restaurants currently available to a user, which restaurants should we display and in what order to maximize the probability of a completed reservation?

## Status

This repository is in **Phase 1 scaffolding**. Core contracts, metrics, experiment assignment, CUPED, IPS, and availability filtering are implemented and unit-tested. Data ingest, LambdaMART training, and served experiments are next.

**Reservation availability is simulated.** Real booking inventory is not used. Availability is generated from restaurant capacity, hour, weekday, popularity, and existing simulated reservations. See `src/data/availability.py` and `DESIGN.md`.

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

Targets in the project brief (approximately +11% NDCG@10, +8% Recall@10, CUPED variance reduction) are **not claimed until reproduced**. Fill these tables from evaluation artifacts only.

### Offline ranking

| Model                 | NDCG@10 | Recall@10 | Relative NDCG Lift | Relative Recall Lift |
| --------------------- | ------: | --------: | -----------------: | -------------------: |
| Contextual popularity |     —   |       —   |           baseline |             baseline |
| Personalized ranker   |     —   |       —   |                  — |                    — |

### Experiment

Primary metric (pre-registered): **completed reservations per exposed user**.

| Metric                      | Control | Treatment | Lift | 95% CI |
| --------------------------- | ------: | --------: | ---: | -----: |
| Completed reservations      |     —   |         — |    — |      — |
| CUPED-adjusted reservations |     —   |         — |    — |      — |
| New users                   |     —   |         — |    — |      — |
| Returning users             |     —   |         — |    — |      — |

Reproduce later with:

```bash
python -m evaluation.report
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
│   ├── experiments/         assignment, CUPED, treatment effects, power
│   └── serving/             FastAPI + impression logging
├── notebooks/
├── tests/
└── sql/                     Postgres schema
```

## Implementation phases

| Phase | Focus | Status |
| ----- | ----- | ------ |
| 1 | Data contracts, simulated availability, contextual popularity, metrics | Scaffolded |
| 2 | User preference features, LambdaMART, temporal + cohort evaluation | Not started |
| 3 | Position propensities, IPS, clipping, doubly robust | Estimators scaffolded |
| 4 | Assignment, event log, ATE, CUPED, HTE | Estimators scaffolded |
| 5 | FastAPI, Docker, benchmark report | API skeleton |

## Tests

Unit tests cover temporal leakage, Haversine distance, experiment assignment stability, CUPED, NDCG, IPS weighting, ranking determinism, and availability filtering.

```bash
pytest
```

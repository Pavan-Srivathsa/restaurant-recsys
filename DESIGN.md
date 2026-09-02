# Design

This document is the system contract for restaurant ranking, candidate generation, experimentation, and counterfactual evaluation. Implementation must match these invariants. Measured metrics belong in `README.md` after they are reproduced, not here.

## Product question

Given restaurants that are currently bookable for a user, rank them to maximize completed reservations — not clicks, not implicit rating affinity, and not global popularity.

## Data model

Four logical datasets. Physical tables live in `sql/init.sql`.

### Users

`user_id`, `account_age_days`, `historical_interaction_count`, `historical_booking_count`, `average_price_preference`, `latitude`, `longitude`.

Historical counts on the user table are convenience aggregates. Feature code must recompute them **as of the recommendation timestamp**, never from a snapshot that includes future events.

### Restaurants

`restaurant_id`, `latitude`, `longitude`, `cuisine`, `price_level`, `rating`, `review_count`, `popularity_score`, `opening_hours`, `capacity`.

### Interactions

`user_id`, `restaurant_id`, `timestamp`, `event_type`, `position`, `experiment_variant`.

Event types: `impression`, `click`, `booking`, `completed_reservation`, `cancelled_reservation`.

### Recommendation impressions

Every ranked response writes one row per displayed restaurant:

`request_id`, `user_id`, `restaurant_id`, `rank_position`, `model_version`, `propensity`, `experiment_variant`, `timestamp`.

This log is required for IPS and doubly robust evaluation.

## Leakage

All aggregate features (cuisine affinity, price preference, restaurant booking rates, user activity counts) use only events with `timestamp < recommendation_timestamp`.

The temporal split is ordered by time, not by row:

```text
Train: earliest 70%
Validation: next 15%
Test: latest 15%
```

`tests/test_leakage.py` asserts that every feature input timestamp precedes the request timestamp.

## Candidate generation

Availability is a hard filter, not merely a ranking feature.

```text
all restaurants
    → geographic eligibility (Haversine radius)
    → opening-hours eligibility
    → reservation availability
    → retrieve 200–500 candidates
    → rank
    → top 10
```

A restaurant that cannot be booked is not recommended, except in explicit diagnostic modes.

## Simulated availability

**This component is simulated.** There is no live reservation inventory in the public/reproducible path.

```text
availability_probability = f(capacity, hour, weekday, popularity, existing_bookings)
```

Implemented in `src/data/availability.py`. The ranking YAML flag `availability.simulated: true` must stay accurate until a real inventory adapter exists.

## Features

### Cuisine affinity

Weighted historical interactions with a cuisine, divided by total weighted interactions. Event weights: impression 0, click 1, booking 3, completed reservation 5. Exponential time decay:

```text
weight = event_weight × exp(-λ × days_since_event)
```

### Price

`user_mean_price`, `user_price_std`, `restaurant_price`, `absolute_price_difference`, `price_match_score = exp(-|user_mean − restaurant_price|)`.

### Distance

Haversine kilometers, `log_distance`, and radius indicators at 1 / 3 / 5 / 10 km.

### Restaurant context

Rating, review count, popularity, bookings in last 7 and 30 days, completion rate last 30 days — all as-of the request time.

### Context

Hour, weekday, weekend flag, meal period.

## Rankers

### Control: contextual popularity

```text
score = w1·popularity + w2·rating + w3·distance_score
      + w4·availability + w5·local_booking_rate
```

Knows location and context. Does **not** use user-specific preference features (cuisine affinity, price match, previous visits).

### Treatment: personalized ranker

Gradient-boosted learning-to-rank (LambdaMART / LightGBM). Labels: impression 0, click 1, booking 2, completed reservation 3. Training rows are grouped by `request_id` so the model learns relative order within a candidate set.

## Offline metrics

Primary: NDCG@10 and Recall@10, reported overall and by cohort (`new`, `light`, `returning`, `heavy`). Also HitRate@10, MRR, coverage, catalog diversity.

Relative lift:

```text
(new − baseline) / baseline
```

Do not copy target percentages into reports. Write the numbers the code produces.

## Experimentation

### Hypothesis

- H0: personalized ranking does not change completed reservation rate.
- H1: personalized ranking increases completed reservation rate.

### Randomization

User-level, stable for the life of the experiment:

```text
bucket = hash(experiment_name, user_id) % 100
0–49  control   (contextual popularity)
50–99 treatment (personalized ranker)
```

Do not re-randomize by session. Assignment is in `src/experiments/assignment.py`.

### Primary metric

Chosen before looking at results (see `configs/experiment.yaml`):

**completed reservations per exposed user** — mean completed-reservation count among users who received at least one recommendation impression in the experiment window.

### Power

Sample size is computed from baseline conversion, MDE, α = 0.05, power = 0.80 **before** the test. Do not stop on the first significant p-value unless a sequential procedure is pre-specified.

### CUPED

Covariate: pre-experiment completed reservations.

```text
θ = Cov(Y, X) / Var(X)
Y_cuped = Y − θ(X − mean(X))
```

Report variance reduction and confirm the ATE is not materially changed. CUPED is a mean-preserving transformation of the pooled sample; arm means may shift.

### Simulated experiment protocol

After training on the logging period, a later window samples the same users. Control ranks eligible restaurants by contextual popularity (deterministic top 10). Treatment ranks the **same eligible set** with LambdaMART using as-of features (logging events plus earlier experiment events only). The click → book → complete/cancel funnel is shared with the logging generator. Analysis is user-level among exposed users. Returning vs new is defined from **pre-experiment** engagement, before looking at Y.

This is not a live production A/B test.

### Heterogeneous effects

Primary segmentation is pre-registered: new vs returning. Estimate

```text
reservation ~ treatment + returning_user + treatment × returning_user
```

rather than a pile of independent significance tests.

## Counterfactual evaluation

The logging policy is contextual popularity: always show the top 8, then sample the remaining display slots uniformly. That defines an exact display propensity

```text
P(shown | x, restaurant)
```

Position examination is `1 / rank` normalized over the 10 slots. IPS uses the product

```text
P(exposure) = P(shown | logging policy) × P(examine | position)
```

For the candidate policy π (personalized top-10):

```text
w = I[restaurant in π top-10] / P_log(shown)
w_clipped = min(w, max_weight)
```

Estimators, all as expected booking rate of the target slate:

- naive overlap: unweighted mean outcome on items both policies showed
- IPS / clipped IPS
- SNIPS
- direct method: mean `m(x, a)` on π's top-10
- doubly robust: DM + propensity-weighted residual on logged items

`m(x, a)` is a LightGBM binary classifier on as-of features, trained on train-split logged rows only.

On synthetic data, an oracle booking rate (share of π's top-10 with latent relevance ≥ booking) is reported so bias can be compared. Do not treat that oracle as available in production.

## Offline evaluation protocol (Phase 2)

Training uses **logged observed labels** on restaurants the popularity policy actually showed (impression=0, click=1, booking=2, completed=3), grouped by `request_id`.

Test ranking scores **all geographically eligible, currently available candidates**. Graded relevance for NDCG/Recall is a position-free oracle from the simulator’s latent cuisine/price/distance utility. That is the product question — which bookable restaurants should we have shown — not a re-ranking of only the logged top 10.

Cohorts are defined from as-of **clicks and bookings**, not impressions:

```text
new: 0     light: 1–5     returning: 6–19     heavy: 20+
```

## Serving

FastAPI ranks for a request `{user_id, latitude, longitude, timestamp}`, assigns the experiment variant, returns top 10, and logs impressions asynchronously. Treatment loads `models/ranker_v1.txt` when present. Postgres stores events; Redis may cache restaurant metadata and recent features.

## What is not done yet

- Ingest the Yelp Open Dataset (or another public dump). Current numbers are from the preference-aware synthetic generator.
- Run a live A/B test with production traffic. Phase 4 is a **simulated** experiment on the same synthetic world.
- Connect to a real reservation inventory system. Availability remains simulated.


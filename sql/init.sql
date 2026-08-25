-- Restaurant recommendation and experiment event store.
-- Availability inventory is simulated in application code, not here.

CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    account_age_days INTEGER NOT NULL DEFAULT 0,
    historical_interaction_count INTEGER NOT NULL DEFAULT 0,
    historical_booking_count INTEGER NOT NULL DEFAULT 0,
    average_price_preference DOUBLE PRECISION,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS restaurants (
    restaurant_id TEXT PRIMARY KEY,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    cuisine TEXT NOT NULL,
    price_level INTEGER NOT NULL CHECK (price_level BETWEEN 1 AND 4),
    rating DOUBLE PRECISION NOT NULL,
    review_count INTEGER NOT NULL DEFAULT 0,
    popularity_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    opening_hours JSONB NOT NULL DEFAULT '{}'::jsonb,
    capacity INTEGER NOT NULL DEFAULT 40,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS interactions (
    interaction_id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users (user_id),
    restaurant_id TEXT NOT NULL REFERENCES restaurants (restaurant_id),
    timestamp TIMESTAMPTZ NOT NULL,
    event_type TEXT NOT NULL CHECK (
        event_type IN (
            'impression',
            'click',
            'booking',
            'completed_reservation',
            'cancelled_reservation'
        )
    ),
    position INTEGER,
    experiment_variant TEXT,
    request_id TEXT
);

CREATE INDEX IF NOT EXISTS interactions_user_ts_idx
    ON interactions (user_id, timestamp);
CREATE INDEX IF NOT EXISTS interactions_restaurant_ts_idx
    ON interactions (restaurant_id, timestamp);

CREATE TABLE IF NOT EXISTS recommendation_impressions (
    impression_id BIGSERIAL PRIMARY KEY,
    request_id TEXT NOT NULL,
    user_id TEXT NOT NULL REFERENCES users (user_id),
    restaurant_id TEXT NOT NULL REFERENCES restaurants (restaurant_id),
    rank_position INTEGER NOT NULL CHECK (rank_position >= 1),
    model_version TEXT NOT NULL,
    propensity DOUBLE PRECISION NOT NULL CHECK (propensity > 0 AND propensity <= 1),
    experiment_variant TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    score DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS impressions_request_idx
    ON recommendation_impressions (request_id);
CREATE INDEX IF NOT EXISTS impressions_user_ts_idx
    ON recommendation_impressions (user_id, timestamp);

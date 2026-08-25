"""Generate a small synthetic dataset into data/processed."""

from __future__ import annotations

import json
from pathlib import Path

from data.ingest import generate_synthetic


def main() -> None:
    users, restaurants, interactions = generate_synthetic()
    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)
    (out / "users.json").write_text(
        json.dumps([u.__dict__ for u in users], default=str, indent=2)
    )
    rest_payload = []
    for r in restaurants:
        d = dict(r.__dict__)
        rest_payload.append(d)
    (out / "restaurants.json").write_text(json.dumps(rest_payload, indent=2))
    (out / "interactions.json").write_text(
        json.dumps([i.__dict__ for i in interactions], default=str, indent=2)
    )
    print(f"wrote {len(users)} users, {len(restaurants)} restaurants, {len(interactions)} interactions")


if __name__ == "__main__":
    main()

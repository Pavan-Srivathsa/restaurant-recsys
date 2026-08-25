"""Write measured offline / experiment tables once artifacts exist.

Until evaluation jobs run, this prints empty placeholders. Do not hard-code targets.
"""

from __future__ import annotations

from pathlib import Path

PLACEHOLDER = "—"


def main() -> None:
    artifact = Path("data/processed/metrics.json")
    print("Offline ranking")
    print("| Model | NDCG@10 | Recall@10 | Relative NDCG Lift | Relative Recall Lift |")
    print(f"| contextual popularity | {PLACEHOLDER} | {PLACEHOLDER} | baseline | baseline |")
    print(f"| personalized ranker   | {PLACEHOLDER} | {PLACEHOLDER} | {PLACEHOLDER} | {PLACEHOLDER} |")
    print()
    print("Experiment (primary: completed reservations per exposed user)")
    print("| Metric | Control | Treatment | Lift | 95% CI |")
    print(f"| completed reservations | {PLACEHOLDER} | {PLACEHOLDER} | {PLACEHOLDER} | {PLACEHOLDER} |")
    if artifact.exists():
        print(f"loaded {artifact}")
    else:
        print("no data/processed/metrics.json yet — run evaluation after Phase 2")


if __name__ == "__main__":
    main()

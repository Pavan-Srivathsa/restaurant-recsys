"""Pipeline: synthetic logs → LambdaMART → offline eval → OPE → simulated A/B."""

from __future__ import annotations

from data.build_features import cuisine_map
from data.ingest import generate_world
from evaluation.offline import evaluate_split, write_metrics
from evaluation.ope import evaluate_policy, train_outcome_model
from evaluation.report import format_experiment_table, format_offline_tables, format_ope_table
from experiments.analyze import analyze_experiment, load_experiment_config
from experiments.simulate import simulate_experiment
from paths import models_dir, processed_dir
from ranking.dataset import build_ltr_frame, logged_groups, split_frame
from ranking.train import save_model, train_lambdamart


def run(
    n_users: int = 400,
    n_restaurants: int = 90,
    n_requests: int = 1400,
    seed: int = 7,
    n_experiment_requests: int = 2800,
    experiment_seed: int = 11,
) -> dict:
    world = generate_world(
        n_users=n_users,
        n_restaurants=n_restaurants,
        n_requests=n_requests,
        seed=seed,
    )
    frame = build_ltr_frame(
        world.users, world.restaurants, world.interactions, world.requests, world.prefs
    )
    train, valid, test = split_frame(frame)
    train_logged = logged_groups(train)
    valid_logged = logged_groups(valid)
    min_leaf = 8 if len(train_logged) < 4000 else 20
    model = train_lambdamart(
        train_logged,
        valid_frame=valid_logged if len(valid_logged) else None,
        seed=seed,
        num_boost_round=180,
        min_data_in_leaf=min_leaf,
    )
    model_path = models_dir() / "ranker_v1.txt"
    save_model(model, model_path)
    cuisine = cuisine_map(world.restaurants)
    results = evaluate_split(test, model, cuisine_by_id=cuisine)
    outcome_model = train_outcome_model(train, seed=seed)
    ope = evaluate_policy(test, model, outcome_model, max_weight=100.0, min_p=0.001)
    experiment_cfg = load_experiment_config()
    experiment_run = simulate_experiment(
        world,
        model,
        n_requests=n_experiment_requests,
        seed=experiment_seed,
        experiment_name=experiment_cfg.get("experiment_name", "personalized_ranker_v1"),
    )
    experiment = analyze_experiment(world, experiment_run, experiment_cfg)
    payload = {
        "n_users": n_users,
        "n_restaurants": n_restaurants,
        "n_requests": len(world.requests),
        "n_train_rows": int(len(train_logged)),
        "n_valid_rows": int(len(valid_logged)),
        "n_test_rows": int(len(test)),
        "n_test_requests": int(test["request_id"].nunique()) if len(test) else 0,
        "n_experiment_requests": experiment["n_requests"],
        "model_path": str(model_path),
        "availability_simulated": True,
        "counterfactual": ope,
        "experiment": experiment,
        **results,
    }
    metrics_path = processed_dir() / "metrics.json"
    write_metrics(payload, metrics_path)
    print(format_offline_tables(payload))
    print()
    print(format_ope_table(ope))
    print()
    print(format_experiment_table(experiment))
    print(f"wrote {metrics_path}")
    print(f"wrote {model_path}")
    return payload


def main() -> None:
    run()


if __name__ == "__main__":
    main()

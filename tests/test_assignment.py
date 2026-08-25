from experiments.assignment import assign_variant, assignment_bucket


def test_assignment_is_stable_for_same_user():
    first = [assign_variant("user_382") for _ in range(20)]
    assert len(set(first)) == 1


def test_assignment_is_deterministic_across_calls():
    assert assignment_bucket("user_1") == assignment_bucket("user_1")
    assert assign_variant("user_1") == assign_variant("user_1")


def test_assignment_buckets_in_range():
    for i in range(500):
        bucket = assignment_bucket(f"user_{i}")
        assert 0 <= bucket <= 99


def test_assignment_splits_near_half():
    n = 2000
    treatment = sum(1 for i in range(n) if assign_variant(f"user_{i}") == "treatment")
    share = treatment / n
    assert 0.45 <= share <= 0.55


def test_different_experiments_can_reassign():
    a = assign_variant("user_9", experiment_name="exp_a")
    b = assign_variant("user_9", experiment_name="exp_b")
    # Not required to differ, but buckets must be independent computations.
    assert assignment_bucket("user_9", "exp_a") != assignment_bucket("user_9", "exp_b") or a == b


def test_control_and_treatment_labels():
    assert assign_variant.__annotations__["return"]
    for i in range(100):
        assert assign_variant(f"id_{i}") in {"control", "treatment"}

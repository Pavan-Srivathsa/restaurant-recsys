from experiments.power import two_proportion_sample_size
from experiments.treatment_effects import ate, interaction_ols


def test_ate_zero_when_identical_arms():
    y = [0, 1, 0, 1, 1]
    result = ate(y, y)
    assert result["absolute_lift"] == 0.0
    assert result["p_value"] == 1.0


def test_ate_detects_lift():
    control = [0] * 80 + [1] * 20
    treatment = [0] * 60 + [1] * 40
    result = ate(control, treatment)
    assert result["absolute_lift"] > 0
    assert result["ci_low"] < result["absolute_lift"] < result["ci_high"]


def test_interaction_recovers_returning_user_extra_lift():
    # New users: treatment effect ~0. Returning users: extra +1 lift.
    y = [0, 0, 0, 0, 1, 1, 2, 2]
    t = [0, 1, 0, 1, 0, 0, 1, 1]
    r = [0, 0, 0, 0, 1, 1, 1, 1]
    fit = interaction_ols(y, t, r)
    assert abs(fit["treatment"]) < 0.25
    assert fit["treatment_x_returning"] > 0.5


def test_power_analysis_returns_positive_n():
    out = two_proportion_sample_size(0.08, 0.06, alpha=0.05, power=0.80)
    assert out["n_per_arm"] > 100
    assert out["n_total"] == 2 * out["n_per_arm"]

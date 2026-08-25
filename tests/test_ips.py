import pytest

from evaluation.doubly_robust import compare_estimators, doubly_robust_value
from evaluation.ips import effective_sample_size, ips_report, ips_weight, ips_weights


def test_ips_weight_is_inverse_propensity():
    assert ips_weight(0.5) == 2.0
    assert ips_weight(0.01, max_weight=10) == 10.0


def test_ips_rejects_non_positive_propensity():
    with pytest.raises(ValueError):
        ips_weight(0.0)


def test_clipped_ips_caps_extreme_weights():
    raw, clipped = ips_weights([0.001, 0.5], max_weight=20)
    assert raw[0] == 1000.0
    assert clipped[0] == 20.0
    assert clipped[1] == 2.0


def test_effective_sample_size_equals_n_for_uniform_weights():
    ess = effective_sample_size([1.0, 1.0, 1.0, 1.0])
    assert abs(ess - 4.0) < 1e-9


def test_ips_report_keys():
    report = ips_report(outcomes=[1, 0, 1], propensities=[0.2, 0.5, 0.2], max_weight=10)
    assert set(report) >= {"unweighted", "ips", "clipped_ips", "effective_sample_size"}
    assert report["unweighted"] == pytest.approx(2 / 3)


def test_doubly_robust_unbiased_when_outcome_model_correct():
    y = [1.0, 0.0, 1.0, 0.0]
    p = [0.5, 0.5, 0.5, 0.5]
    m = [1.0, 0.0, 1.0, 0.0]
    value = doubly_robust_value(y, p, m)
    assert value == pytest.approx(0.5)


def test_compare_estimators_includes_all_methods():
    out = compare_estimators([1, 0], [0.5, 0.5], [0.6, 0.4])
    assert "naive" not in out
    assert "unweighted" in out
    assert "ips" in out
    assert "doubly_robust" in out

import numpy as np

from experiments.cuped import cuped_adjust, cuped_report, cuped_theta


def test_cuped_reduces_variance_when_x_predicts_y():
    rng = np.random.default_rng(7)
    x = rng.normal(2.0, 1.0, size=2000)
    y = 0.4 * x + rng.normal(0, 0.5, size=2000)
    report = cuped_report(y, x)
    assert report["variance_y_cuped"] < report["variance_y"]
    assert report["variance_reduction"] > 0.2


def test_cuped_preserves_mean():
    y = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0], dtype=float)
    x = np.array([0.0, 2.0, 1.0, 0.0, 3.0, 1.0], dtype=float)
    adjusted = cuped_adjust(y, x)
    assert abs(float(np.mean(adjusted)) - float(np.mean(y))) < 1e-10


def test_cuped_theta_matches_ols_slope():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    theta = cuped_theta(y, x)
    assert abs(theta - 2.0) < 1e-9


def test_cuped_zero_variance_covariate():
    y = np.array([1.0, 0.0, 1.0])
    x = np.array([5.0, 5.0, 5.0])
    assert cuped_theta(y, x) == 0.0
    np.testing.assert_array_equal(cuped_adjust(y, x), y)

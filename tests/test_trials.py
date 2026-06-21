import math

from trials import (
    _norm_cdf,
    _norm_ppf,
    config_hash,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    log_trial,
    probabilistic_sharpe_ratio,
    trial_count,
)


def test_normal_helpers():
    assert abs(_norm_cdf(0.0) - 0.5) < 1e-9
    assert abs(_norm_ppf(0.975) - 1.959963) < 1e-3
    # round-trip
    assert abs(_norm_cdf(_norm_ppf(0.83)) - 0.83) < 1e-4


def test_psr_monotonic_in_sharpe():
    # Higher observed Sharpe -> higher probability it beats the benchmark.
    low = probabilistic_sharpe_ratio(0.05, 500)
    high = probabilistic_sharpe_ratio(0.30, 500)
    assert 0.0 <= low <= high <= 1.0


def test_dsr_discounts_more_trials():
    # The same Sharpe found after MANY trials should be discounted (lower DSR).
    few = deflated_sharpe_ratio(0.20, 1000, n_trials=5, sr_variance=0.01)
    many = deflated_sharpe_ratio(0.20, 1000, n_trials=500, sr_variance=0.01)
    assert many < few
    assert expected_max_sharpe(500, 0.01) > expected_max_sharpe(5, 0.01)


def test_trial_logging_roundtrip(tmp_path):
    path = tmp_path / "trial_log.csv"
    h1 = log_trial("A", {"a": 1, "b": "x"}, {"sharpe": 0.5}, path=path)
    h2 = log_trial("B", {"a": 2}, {"sharpe": 0.1, "extra": 9}, path=path)
    assert h1 == config_hash({"a": 1, "b": "x"})
    assert h1 != h2
    assert trial_count(path) == 2

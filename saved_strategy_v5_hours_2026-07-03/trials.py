"""
trials.py — Phase A+: the multiple-testing ledger and Sharpe deflation.

Every distinct configuration we evaluate on validation data is appended to
runs/trial_log.csv. In Phases E/F we use the trial COUNT to deflate the best
Sharpe (Bailey & Lopez de Prado's Deflated Sharpe Ratio), so a good-looking
result found after many attempts is correctly discounted.

Normal CDF/PPF are implemented locally (math.erf + Acklam's inverse) so we carry
no scipy dependency.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

RUNS_DIR = Path(__file__).with_name("runs")
TRIAL_LOG = RUNS_DIR / "trial_log.csv"


def config_hash(params: dict) -> str:
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def log_trial(phase: str, params: dict, metrics: dict,
              path: Path = TRIAL_LOG) -> str:
    """Append one trial row; returns the config hash. Creates header if needed."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    chash = config_hash(params)
    row = {"phase": phase, "config_hash": chash,
           "params": json.dumps(params, sort_keys=True, default=str)}
    row.update({f"m_{k}": v for k, v in metrics.items()})
    write_header = not path.exists()
    # Union of columns across rows (metrics may vary) — keep it simple & robust.
    existing_cols: list[str] = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            existing_cols = next(r, [])
    cols = list(dict.fromkeys(existing_cols + list(row.keys())))
    rows = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for rr in rows:
            w.writerow({c: rr.get(c, "") for c in cols})
    return chash


def trial_count(path: Path = TRIAL_LOG) -> int:
    if not path.exists():
        return 0
    with path.open("r", newline="", encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)  # minus header


# --------------------------------------------------------------------------- #
# Normal distribution helpers (no scipy)                                       #
# --------------------------------------------------------------------------- #
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Acklam's rational approximation to the inverse normal CDF."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def probabilistic_sharpe_ratio(sr_hat: float, n: int, skew: float = 0.0,
                               kurt: float = 3.0, sr_benchmark: float = 0.0) -> float:
    """
    P(true Sharpe > sr_benchmark) given an estimate sr_hat from n observations,
    accounting for non-normal returns (skew, kurtosis). sr_* are per-observation.
    """
    if n < 2:
        return float("nan")
    denom = math.sqrt(max(1e-12, 1 - skew * sr_hat + (kurt - 1) / 4.0 * sr_hat**2))
    z = (sr_hat - sr_benchmark) * math.sqrt(n - 1) / denom
    return _norm_cdf(z)


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum Sharpe under the null of zero true Sharpe (Bailey/LdP)."""
    if n_trials < 2 or sr_variance <= 0:
        return 0.0
    gamma = 0.5772156649015329  # Euler-Mascheroni
    e = math.e
    z1 = _norm_ppf(1 - 1.0 / n_trials)
    z2 = _norm_ppf(1 - 1.0 / (n_trials * e))
    return math.sqrt(sr_variance) * ((1 - gamma) * z1 + gamma * z2)


def deflated_sharpe_ratio(sr_hat: float, n: int, n_trials: int, sr_variance: float,
                          skew: float = 0.0, kurt: float = 3.0) -> float:
    """
    DSR = PSR evaluated against the expected-max-Sharpe benchmark implied by the
    number of trials. DSR > ~0.95 => the result is unlikely to be luck.
    """
    sr0 = expected_max_sharpe(n_trials, sr_variance)
    return probabilistic_sharpe_ratio(sr_hat, n, skew, kurt, sr_benchmark=sr0)


if __name__ == "__main__":
    print("=== trials.py self-test ===")
    print("norm_cdf(0)   :", round(_norm_cdf(0.0), 4), "(want 0.5)")
    print("norm_ppf(.975):", round(_norm_ppf(0.975), 4), "(want ~1.96)")
    print("PSR(0.2,1000) :", round(probabilistic_sharpe_ratio(0.2, 1000), 4))
    print("DSR(0.2,1000,50,var=0.01):",
          round(deflated_sharpe_ratio(0.2, 1000, 50, 0.01), 4))

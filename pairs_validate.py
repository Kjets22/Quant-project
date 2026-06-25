"""
Validate the QQQ-SPY 30-min spread mean-reversion candidate edge — rigorously.

Walk-forward (expanding window): re-estimate the hedge ratio on all prior data,
test on the next sequential block. Report NET annualized Sharpe per block at two
realistic cost levels, then pool and compute the DEFLATED SHARPE RATIO (Bailey &
Lopez de Prado) to correct for the number of configs tried. Persistence across
blocks + a positive deflated Sharpe = a real edge; one good block + DSR~0 = luck.
"""

from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

from environment import align_partner
from lockbox import Lockbox, load_dev_ticker
from trials import deflated_sharpe_ratio

W = 80          # z-score window
ENTRY, EXIT = 1.0, 0.3
NTRIALS = 24    # configs explored in the search (3 timeframes x 4 entries x 2 cost views)


def load_30min():
    lb = Lockbox.load_or_build()
    def r(t):
        d = load_dev_ticker(t, lb)
        g = d.set_index("timestamp").resample("30min").agg(close=("close", "last")).dropna()
        return g.reset_index()
    m = align_partner(r("QQQ"), r("SPY")).dropna().reset_index(drop=True)
    return (np.log(m["close"].to_numpy()), np.log(m["partner_close"].to_numpy()),
            pd.to_datetime(m["timestamp"]))


def zscore(spread, w):
    s = pd.Series(spread)
    return ((s - s.rolling(w, min_periods=w).mean())
            / s.rolling(w, min_periods=w).std()).to_numpy()


def backtest(z, dspread, leg_bps):
    cost = 2 * leg_bps / 1e4
    n = len(z)
    pos = np.zeros(n)
    p = 0.0
    for t in range(n):
        if np.isnan(z[t]):
            p = 0.0
        elif p == 0.0:
            p = -1.0 if z[t] > ENTRY else (1.0 if z[t] < -ENTRY else 0.0)
        elif p > 0 and z[t] >= -EXIT:
            p = 0.0
        elif p < 0 and z[t] <= EXIT:
            p = 0.0
        pos[t] = p
    gross = pos[:-1] * dspread[1:]
    turn = np.abs(np.diff(np.concatenate([[0.0], pos])))[:-1]
    net = gross - turn * cost
    return net, int((np.diff(pos) != 0).sum())


def ann_sharpe(net, per_year):
    if net.std() < 1e-12:
        return 0.0
    return float(net.mean() / net.std() * np.sqrt(per_year))


def main():
    lq, ls, ts = load_30min()
    n = len(lq)
    years = max((ts.iloc[-1] - ts.iloc[0]).days / 365.0, 1e-9)
    per_year = n / years
    print(f"30-min QQQ-SPY: {n} bars over {years:.1f}y (~{per_year:.0f} bars/yr)")

    K = 6
    start = int(n * 0.4)                       # initial warmup/train
    bounds = np.linspace(start, n, K + 1).astype(int)
    for leg in (0.2, 0.5):
        print(f"\n=== walk-forward NET annualized Sharpe @ {leg} bps/leg ===")
        sharpes, all_net = [], []
        for i in range(K):
            a, b = bounds[i], bounds[i + 1]
            beta = np.linalg.lstsq(np.vstack([ls[:a], np.ones(a)]).T, lq[:a],
                                   rcond=None)[0][0]
            spread = lq - beta * ls
            z = zscore(spread, W)
            dspread = np.diff(spread, prepend=spread[0])
            net, tr = backtest(z[a:b], dspread[a:b], leg)
            sh = ann_sharpe(net, per_year)
            sharpes.append(sh)
            all_net.append(net)
            print(f"  block {i+1}: Sharpe {sh:+.2f}  ret {net.sum()*100:+.2f}%  trades {tr}")
        pooled = np.concatenate(all_net)
        psh = ann_sharpe(pooled, per_year)
        sr_var = float(np.var(sharpes, ddof=1)) if len(sharpes) > 1 else 0.01
        # DSR uses per-observation Sharpe (un-annualized) and the trial count.
        sr_obs = pooled.mean() / pooled.std() if pooled.std() > 1e-12 else 0.0
        from scipy import stats as _s  # noqa
        skew = float(pd.Series(pooled).skew())
        kurt = float(pd.Series(pooled).kurt()) + 3.0
        dsr = deflated_sharpe_ratio(sr_obs, len(pooled), NTRIALS, sr_var / per_year,
                                    skew=skew, kurt=kurt)
        wins = sum(s > 0 for s in sharpes)
        print(f"  POOLED annualized Sharpe {psh:+.2f}  | blocks positive {wins}/{K}  "
              f"| Deflated Sharpe (P>0 after {NTRIALS} trials) = {dsr:.3f}")
        print("    (DSR > 0.95 => unlikely luck; < 0.95 => not significant)")


if __name__ == "__main__":
    main()

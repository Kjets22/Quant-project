"""
QQQ-SPY statistical arbitrage (market-neutral) — the guide's recommended edge.

Why this is different: every prior attempt tried to beat a bull-market index's
buy-and-hold (≈impossible). A SPREAD strategy is market-neutral — it doesn't
compete with B&H; it harvests mean-reversion of the QQQ-vs-SPY spread.

Rigorous + honest:
  * Engle-Granger cointegration: OLS log(QQQ) ~ log(SPY) on TRAIN -> hedge ratio;
    ADF test on the residual (train AND test) to check the relationship is stable.
  * Causal rolling z-score of the spread (detrends slow drift).
  * z-score mean-reversion: enter at +/-entry sigma, exit near 0.
  * HONEST two-leg costs (trading both QQQ and SPY each adjustment).
  * Strictly out-of-sample: hedge ratio & params from train, evaluated on test.
  * Reports market-neutrality (correlation of P&L to SPY returns).
"""

from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from environment import align_partner
from lockbox import Lockbox, load_dev_ticker

import os
# QQQ/SPY are ultra-liquid: a 1-cent spread on a ~$450 ETF is ~0.2 bps, far below
# the 2 bps generic-stock assumption. Default to a realistic 0.5 bps/leg; override.
LEG_COST = float(os.environ.get("LEG_COST_BPS", "0.5")) / 1e4
SPREAD_COST = 2 * LEG_COST  # trading the spread moves BOTH legs


def resample(df, rule):
    g = df.set_index("timestamp").resample(rule).agg(close=("close", "last")).dropna()
    return g.reset_index()


def zscore(spread, w):
    s = pd.Series(spread)
    m = s.rolling(w, min_periods=w).mean()
    sd = s.rolling(w, min_periods=w).std()
    return ((s - m) / sd).to_numpy()


def backtest(z, dspread, entry, exit_):
    """z-score mean reversion on the spread. Returns net P&L series + trade count."""
    n = len(z)
    pos = np.zeros(n)
    p = 0.0
    for t in range(n):
        if np.isnan(z[t]):
            pos[t] = 0.0
            continue
        if p == 0.0:
            if z[t] > entry:
                p = -1.0          # spread rich -> short spread
            elif z[t] < -entry:
                p = +1.0          # spread cheap -> long spread
        elif p > 0 and z[t] >= -exit_:
            p = 0.0
        elif p < 0 and z[t] <= exit_:
            p = 0.0
        pos[t] = p
    gross = pos[:-1] * dspread[1:]                       # hold pos over next bar
    turn = np.abs(np.diff(np.concatenate([[0.0], pos])))[:-1]
    cost = turn * SPREAD_COST
    net = gross - cost
    return net, gross, pos[:-1], int((np.diff(pos) != 0).sum())


def run(rule, name, w):
    lb = Lockbox.load_or_build()
    q = resample(load_dev_ticker("QQQ", lb), rule)
    s = resample(load_dev_ticker("SPY", lb), rule)
    m = align_partner(q, s).dropna()                    # q.close + partner_close (SPY)
    lq = np.log(m["close"].to_numpy())
    ls = np.log(m["partner_close"].to_numpy())
    cut = int(len(lq) * 0.6)

    # Engle-Granger hedge ratio on TRAIN, applied OOS.
    X = np.vstack([ls[:cut], np.ones(cut)]).T
    beta, alpha = np.linalg.lstsq(X, lq[:cut], rcond=None)[0]
    spread = lq - beta * ls - alpha
    adf_tr = adfuller(spread[:cut], maxlag=1, autolag=None)[1]
    adf_te = adfuller(spread[cut:], maxlag=1, autolag=None)[1]

    z = zscore(spread, w)
    dspread = np.diff(spread, prepend=spread[0])
    z_te, d_te = z[cut:], dspread[cut:]
    spy_ret_te = np.diff(ls, prepend=ls[0])[cut:]

    print(f"\n===== {name} (bars={len(lq)}, train/test 60/40, z-window {w}) =====")
    print(f"  hedge ratio beta={beta:.3f}   ADF p-value: train={adf_tr:.4f} "
          f"test={adf_te:.4f}  (<0.05 = cointegrated/stationary spread)")
    print(f"  {'entry':>5} {'netSharpe':>10} {'grossSh':>9} {'totRet%':>8} "
          f"{'trades':>6} {'mkt-corr':>9}")
    for entry in (1.0, 1.5, 2.0, 2.5):
        net, gross, pos, tr = backtest(z_te, d_te, entry, exit_=0.3)
        if net.std() < 1e-12:
            print(f"  {entry:>5.1f}  (no trades)")
            continue
        net_sh = net.mean() / net.std() * np.sqrt(len(net))
        gross_sh = gross.mean() / gross.std() * np.sqrt(len(gross)) if gross.std() > 1e-12 else 0.0
        sr = spy_ret_te[1:len(net) + 1]
        corr = float(np.corrcoef(net[:len(sr)], sr[:len(net)])[0, 1]) if len(net) > 10 else float("nan")
        print(f"  {entry:>5.1f} {net_sh:>10.2f} {gross_sh:>9.2f} {net.sum() * 100:>8.2f} "
              f"{tr:>6d} {corr:>9.2f}")


if __name__ == "__main__":
    for rule, name, w in (("60min", "HOURLY", 50), ("1D", "DAILY", 20),
                          ("30min", "30-MIN", 80)):
        run(rule, name, w)
    print("\nREAD: a NET (after-cost) OOS Sharpe clearly > 0 with mkt-corr ~ 0 would be")
    print("a real, market-neutral edge. If net Sharpe ~0 after costs, the spread is")
    print("too tight / costs eat it (the honest, likely outcome for QQQ-SPY).")

"""
v4_lastweek_notax.py — v4 (15-min / 4:1) on LAST WEEK (Mon 2026-06-22 .. Fri 06-26),
FRICTIONLESS: no 3-bps stock cost, and options priced with NO spread / NO theta (the "tax"
removed). Shows every trade and when it fired. Standalone; touches no frozen snapshot.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import lightgbm as lgb

from triple_barrier_ml import atr, features, label
from triple_barrier_breadth import TICKERS
from sr_features import sr_features
from options_strategy import opt_ret

MIN, TP, SL = 15, 4.0, 1.0
SEL_Q, HBAR = 0.93, 24
LWS, LWE = pd.Timestamp("2026-06-22"), pd.Timestamp("2026-06-29")
END = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)


def load5(tk):
    df = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv", parse_dates=["timestamp"])
    p = Path(f"data_cache/{tk}_recent_2026-06-01_{END.date()}.csv")
    if p.exists():
        rec = pd.read_csv(p, parse_dates=["timestamp"])
        df = (pd.concat([df, rec], ignore_index=True)
                .drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp"))
    return df


def run(tk):
    raw = load5(tk)
    d = raw.set_index("timestamp").resample(f"{MIN}min").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum")).dropna().reset_index()
    ts = pd.to_datetime(d["timestamp"]).to_numpy()
    h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
    A = atr(h, l, c)
    daily = d.set_index("timestamp")["close"].resample("1D").last().dropna()
    sig = min(max(float(np.log(daily / daily.shift(1)).std() * math.sqrt(252)), 0.08), 0.80)
    X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                   sr_features(d).reset_index(drop=True)], axis=1)
    y = label(h, l, c, A, TP, SL)
    fv = X.notna().all(axis=1).to_numpy()
    n = len(c)
    tr = np.where(fv & np.isfinite(y) & (ts < np.datetime64(LWS)))[0]
    if len(tr) < 500:
        return []
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                             min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=1.0, verbose=-1)
    clf.fit(X.iloc[tr], y[tr].astype(int))
    thr = np.quantile(clf.predict_proba(X.iloc[tr])[:, 1], SEL_Q)
    fwd = np.where(fv & (ts >= np.datetime64(LWS)) & (ts < np.datetime64(LWE)))[0]
    if len(fwd) == 0:
        return []
    proba = {int(ix): float(p) for ix, p in zip(fwd, clf.predict_proba(X.iloc[fwd])[:, 1])}
    out, i, last = [], int(fwd[0]), int(fwd[-1])
    while i <= last:
        if proba.get(i, -1) < thr:
            i += 1; continue
        a = A[i]; up, dn = c[i] + TP * a, c[i] - SL * a
        res, j = None, i + 1
        while j < min(i + HBAR + 1, n):
            if l[j] <= dn:
                res = 0; break
            if h[j] >= up:
                res = 1; break
            j += 1
        if res is None and j >= n:
            i = n; continue
        if res is None:
            res = 1 if c[min(j, n - 1)] > c[i] else 0
        Sx = up if res == 1 else dn
        stock_gross = (TP * a if res == 1 else -SL * a) / c[i] * 100        # NO cost
        dt = max((j if j < n else n - 1) - i, 1) * MIN / (60 * 24)
        opt_fric = opt_ret({"S0": c[i], "Sx": Sx, "a": a, "sig": sig, "dt": dt},
                           "call", 0.95, 7, 0.0, frictionless=True) * 100     # NO spread/theta
        out.append((tk, str(pd.Timestamp(ts[i]))[5:16], "TARGET" if res == 1 else "STOP",
                    round(stock_gross, 2), round(opt_fric, 1)))
        i = j + 1
    return out


def main():
    print(f"v4 (15-min / 4:1) LAST WEEK ({LWS.date()} .. {(LWE-pd.Timedelta(days=3)).date()}) "
          f"— FRICTIONLESS (no cost, no option tax)\n")
    trades = []
    for tk in TICKERS:
        try:
            trades += run(tk)
        except Exception as e:
            print(f"  [skip {tk}] {e}")
    trades.sort(key=lambda t: t[1])
    if not trades:
        print("  no trades"); return
    print(f"  {'tk':>5} {'entry (UTC)':>16} {'outcome':>7} {'stock(no cost)':>14} {'option(no tax)':>15}")
    for tk, when, oc, sg, of in trades:
        print(f"  {tk:>5} {when:>16} {oc:>7} {sg:>+13.2f}% {of:>+14.1f}%")
    wins = sum(t[2] == "TARGET" for t in trades)
    sg_tot = sum(t[3] for t in trades)
    of_tot = sum(t[4] for t in trades)
    print(f"\n  trades={len(trades)}  wins={wins}  win%={wins/len(trades):.0%}  (break-even at 4:1 = 20%)")
    print(f"  STOCK total (no cost)      = {sg_tot:+.2f}%")
    print(f"  OPTIONS total (no tax)     = {of_tot:+.0f}%   (frictionless leverage of the same moves)")


if __name__ == "__main__":
    main()

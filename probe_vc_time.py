"""
probe_vc_time.py — does vC (30xATR target / 3xATR stop, hourly) benefit from MORE time?

vC's live clock is 96 hourly bars. A 30-ATR move is a monster; most trades end as TIME
exits. Question: do longer clocks (144, 192 bars) convert enough TIME exits into TARGET
hits to beat the extra stop-outs and slower trade recycling?

Honest walk-forward, two windows, per H:
  train < 2024-07-14 (embargo H bars) -> sim 2024-07-14..2025-07-14
  train < 2025-07-14 (embargo H bars) -> sim 2025-07-14..now
Same pipeline as the live bot (prep/"trend" features, top-7% gate, ATR floor, 5 bps).
"""

from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import lightgbm as lgb

from alpaca_bot2 import prep, MIN_ATR_PCT
from triple_barrier_breadth import TICKERS

TP, SL, SELQ = 30.0, 3.0, 0.93
EFF_COST = 5.0 / 1e4
WINDOWS = [("2024-07-14", "2025-07-14"), ("2025-07-14", "2099-01-01")]
HS = [96, 144, 192]


def label(h, l, c, A, valid, H):
    n = len(c)
    y = np.full(n, np.nan)
    up, dn = c + TP * A, c - SL * A
    for i in range(n - 1):
        if not valid[i]:
            continue
        for j in range(i + 1, min(i + H + 1, n)):
            if l[j] <= dn[i]:
                y[i] = 0; break
            if h[j] >= up[i]:
                y[i] = 1; break
    return y


def run(tk, H, lo, hi):
    ts, h, l, c, A, X, valid, _, _ = prep(tk, 60, "trend", "atr")
    ok = valid & np.isfinite(A) & (A / np.maximum(c, 1e-9) >= MIN_ATR_PCT)
    y = label(h, l, c, A, ok, H)
    fv = (X.notna().all(axis=1)).to_numpy() & ok
    tr = np.where(fv & np.isfinite(y) & (ts < np.datetime64(lo)))[0]
    tr = tr[:-H] if len(tr) > H else tr
    if len(tr) < 500 or np.nansum(y[tr]) < 10:
        return []
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                             min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=1.0, verbose=-1)
    clf.fit(X.iloc[tr], y[tr].astype(int))
    thr = np.quantile(clf.predict_proba(X.iloc[tr])[:, 1], SELQ)
    idx = np.where(fv & (ts >= np.datetime64(lo)) & (ts < np.datetime64(hi)))[0]
    if len(idx) == 0:
        return []
    proba = {int(ix): float(p) for ix, p in
             zip(idx, clf.predict_proba(X.iloc[idx])[:, 1])}
    rets, n = [], len(c)
    i, last = int(idx[0]), int(idx[-1])
    while i <= last:
        if proba.get(i, -1.0) < thr:
            i += 1; continue
        up, dn = c[i] + TP * A[i], c[i] - SL * A[i]
        res, j = None, i + 1
        while j < min(i + H + 1, n):
            if l[j] <= dn:
                res = 0; break
            if h[j] >= up:
                res = 1; break
            j += 1
        ex = min(j, n - 1)
        px = up if res == 1 else (dn if res == 0 else c[ex])
        rets.append(((px - c[i]) / c[i] - EFF_COST,
                     "TGT" if res == 1 else ("STP" if res == 0 else "TIME")))
        i = ex + 1
    return rets


def main():
    print(f"PROBE: vC (+{TP}xATR / -{SL}xATR, hourly, top-7%) — clock 96 vs 144 vs 192")
    for lo, hi in WINDOWS:
        print(f"\n=== window {lo} .. {hi if hi < '2099' else 'now'} ===")
        print(f"  {'H':>4} {'trades':>7} {'tgt':>4} {'stp':>4} {'time':>5} "
              f"{'win%':>6} {'avg bps':>8} {'total%':>8}")
        for H in HS:
            allr = []
            for tk in TICKERS:
                try:
                    allr += run(tk, H, lo, hi)
                except Exception as e:
                    print(f"  [warn {tk} H={H}: {e}]")
            if not allr:
                continue
            r = np.array([x[0] for x in allr])
            oc = [x[1] for x in allr]
            print(f"  {H:>4} {len(r):>7} {oc.count('TGT'):>4} {oc.count('STP'):>4} "
                  f"{oc.count('TIME'):>5} {(r > 0).mean():>6.1%} {r.mean()*1e4:>+8.1f} "
                  f"{r.sum()*100:>+8.2f}")


if __name__ == "__main__":
    main()

"""
probe_vc_target.py — should vC aim for a target it can actually REACH?

Finding from probe_vc_time: vC (30xATR target / 3xATR stop) almost never hits its
target — profits come from TIME+ drift exits. User question: would a reachable
target beat drift-riding? Sweep the target multiple {5,8,12,20,30}xATR with the
same 3xATR stop, hourly bars, H=96, same walk-forward windows and pipeline.
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

SL, SELQ, H = 3.0, 0.93, 96
EFF_COST = 5.0 / 1e4
WINDOWS = [("2024-07-14", "2025-07-14"), ("2025-07-14", "2099-01-01")]
TPS = [5.0, 8.0, 12.0, 20.0, 30.0]


def run(tk, tp, lo, hi):
    ts, h, l, c, A, X, valid, _, _ = prep(tk, 60, "trend", "atr")
    ok = valid & np.isfinite(A) & (A / np.maximum(c, 1e-9) >= MIN_ATR_PCT)
    n = len(c)
    up_a, dn_a = c + tp * A, c - SL * A
    y = np.full(n, np.nan)
    for i in range(n - 1):
        if not ok[i]:
            continue
        for j in range(i + 1, min(i + H + 1, n)):
            if l[j] <= dn_a[i]:
                y[i] = 0; break
            if h[j] >= up_a[i]:
                y[i] = 1; break
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
    rets = []
    i, last = int(idx[0]), int(idx[-1])
    while i <= last:
        if proba.get(i, -1.0) < thr:
            i += 1; continue
        res, j = None, i + 1
        while j < min(i + H + 1, n):
            if l[j] <= dn_a[i]:
                res = 0; break
            if h[j] >= up_a[i]:
                res = 1; break
            j += 1
        ex = min(j, n - 1)
        px = up_a[i] if res == 1 else (dn_a[i] if res == 0 else c[ex])
        rets.append(((px - c[i]) / c[i] - EFF_COST,
                     "TGT" if res == 1 else ("STP" if res == 0 else "TIME")))
        i = ex + 1
    return rets


def main():
    print(f"PROBE: vC target sweep — {{5,8,12,20,30}}xATR target / {SL}xATR stop, "
          f"H={H} hourly, top-7%")
    worst = {}
    for lo, hi in WINDOWS:
        print(f"\n=== window {lo} .. {hi if hi < '2099' else 'now'} ===")
        print(f"  {'tgt':>5} {'trades':>7} {'tgt':>4} {'stp':>4} {'time':>5} "
              f"{'win%':>6} {'avg bps':>8} {'total%':>8}")
        for tp in TPS:
            allr = []
            for tk in TICKERS:
                try:
                    allr += run(tk, tp, lo, hi)
                except Exception as e:
                    print(f"  [warn {tk} tp={tp}: {e}]")
            if not allr:
                continue
            r = np.array([x[0] for x in allr])
            oc = [x[1] for x in allr]
            tot = float(r.sum() * 100)
            worst[tp] = min(worst.get(tp, 1e9), tot)
            print(f"  {tp:>4.0f}x {len(r):>7} {oc.count('TGT'):>4} {oc.count('STP'):>4} "
                  f"{oc.count('TIME'):>5} {(r > 0).mean():>6.1%} {r.mean()*1e4:>+8.1f} "
                  f"{tot:>+8.2f}")
    print("\nWORST-window total per target (the robustness yardstick):")
    for tp in TPS:
        if tp in worst:
            print(f"  {tp:>4.0f}x  {worst[tp]:+8.2f}%")


if __name__ == "__main__":
    main()

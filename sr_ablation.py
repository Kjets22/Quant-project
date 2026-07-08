"""
sr_ablation.py — the proof that the zip's S/R features ARE the edge.

Runs the EXACT same strategy (walk-forward, top-7%, 3 bps) twice for each config:
  base only          = the 10 momentum/volatility candle features
  base + S/R         = those PLUS the 10 support/resistance features from the zip
If 'base only' sits at/under its break-even win rate and 'base + S/R' clears it, then
the S/R features are what create the edge. Standalone; touches no frozen snapshot.
"""

from __future__ import annotations

import sys

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

SEL_Q, HBAR, COST_BPS = 0.93, 24, 3.0


def bars(tk, mins):
    df = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv", parse_dates=["timestamp"])
    g = df.set_index("timestamp").resample(f"{mins}min").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum")).dropna()
    return g.reset_index()


def run(mins, tp, sl, use_sr):
    TT = TW = 0
    SR = 0.0
    for tk in TICKERS:
        d = bars(tk, mins)
        h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
        A = atr(h, l, c)
        base = features(h, l, c, v).reset_index(drop=True)
        X = pd.concat([base, sr_features(d).reset_index(drop=True)], axis=1) if use_sr else base
        y = label(h, l, c, A, tp, sl)
        m = (X.notna().all(axis=1) & np.isfinite(y)).to_numpy()
        idx = np.where(m)[0]
        Xv, yv = X.iloc[idx].reset_index(drop=True), y[idx].astype(int)
        hv, lv, cv, Av = h[idx], l[idx], c[idx], A[idx]
        n = len(idx); K = 5
        bnds = np.linspace(int(n * 0.4), n, K + 1).astype(int)
        for k in range(K):
            clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                     min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                                     reg_lambda=1.0, verbose=-1)
            clf.fit(Xv.iloc[:bnds[k]], yv[:bnds[k]])
            thr = np.quantile(clf.predict_proba(Xv.iloc[:bnds[k]])[:, 1], SEL_Q)
            proba = clf.predict_proba(Xv.iloc[bnds[k]:bnds[k + 1]])[:, 1]
            i = bnds[k]
            while i < bnds[k + 1] - 1:
                if proba[i - bnds[k]] < thr:
                    i += 1; continue
                a = Av[i]; up, dn = cv[i] + tp * a, cv[i] - sl * a
                res, j = None, i + 1
                while j < min(i + HBAR + 1, n):
                    if lv[j] <= dn:
                        res = 0; break
                    if hv[j] >= up:
                        res = 1; break
                    j += 1
                if res is None:
                    res = 1 if cv[min(j, n - 1)] > cv[i] else 0
                TT += 1; TW += res
                SR += (tp * a if res == 1 else -sl * a) / cv[i] - COST_BPS / 1e4
                i = j + 1
    be = sl / (sl + tp)
    return TW / TT, be, SR * 100, SR / TT * 1e4


def main():
    print("S/R FEATURES OFF vs ON  (walk-forward, top-7%, 3 bps, 8-name basket)\n")
    for mins, tp, sl, name in ((30, 1.5, 1.0, "v3 (30-min/1.5:1)"), (15, 4.0, 1.0, "v4 (15-min/4:1)")):
        print(f"=== {name} ===")
        print(f"  {'features':>14} {'win%':>6} {'break-even':>11} {'margin':>8} {'mean bps':>9} {'total%':>8}")
        for use_sr in (False, True):
            wr, be, tot, bps = run(mins, tp, sl, use_sr)
            tag = "base + S/R" if use_sr else "base only"
            print(f"  {tag:>14} {wr:>6.1%} {be:>11.0%} {wr-be:>+8.1%} {bps:>+9.1f} {tot:>+8.0f}")
        print()
    print("READ: if 'base only' win% is AT or BELOW break-even (margin <= 0) and 'base + S/R'")
    print("clears it (margin > 0), the S/R features are literally what create the edge.")


if __name__ == "__main__":
    main()

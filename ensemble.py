"""
ensemble.py — integrate the two VALIDATED strategies into one portfolio:
  v3 = 30-min / 1.5:1   (47% win, steadier)
  v4 = 15-min / 4:1     (29% win, high-variance, big winners)
Generates each one's out-of-sample per-trade returns (walk-forward, pooled basket), builds
monthly return series, and forms a 50/50 blend. Reports total, annualized Sharpe, and max
drawdown for each + the blend, plus their correlation. Lower-correlation -> the blend should
cut drawdown / lift Sharpe vs either alone. Standalone; touches no frozen snapshot.
"""

from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from triple_barrier_ml import atr, features, label
from triple_barrier_breadth import TICKERS
from sr_features import sr_features

SEL_Q, HBAR, COST_BPS = 0.93, 24, 3.0


def gen(mins, tp, sl):
    rows = []
    for tk in TICKERS:
        df = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv", parse_dates=["timestamp"])
        d = df.set_index("timestamp").resample(f"{mins}min").agg(
            high=("high", "max"), low=("low", "min"), close=("close", "last"),
            volume=("volume", "sum")).dropna().reset_index()
        ts = pd.to_datetime(d["timestamp"]).to_numpy()
        h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
        A = atr(h, l, c)
        X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                       sr_features(d).reset_index(drop=True)], axis=1)
        y = label(h, l, c, A, tp, sl)
        m = (X.notna().all(axis=1) & np.isfinite(y)).to_numpy()
        idx = np.where(m)[0]
        Xv, yv = X.iloc[idx].reset_index(drop=True), y[idx].astype(int)
        hv, lv, cv, Av, tsv = h[idx], l[idx], c[idx], A[idx], ts[idx]
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
                rows.append((pd.Timestamp(tsv[i]), (tp * a if res == 1 else -sl * a) / cv[i] - COST_BPS / 1e4))
                i = j + 1
    s = pd.DataFrame(rows, columns=["ts", "ret"]).set_index("ts").sort_index()
    return s["ret"].resample("ME").sum()              # monthly return (1 unit/trade)


def stats(monthly):
    cum = monthly.cumsum()
    dd = (cum - cum.cummax()).min()
    sharpe = monthly.mean() / monthly.std() * np.sqrt(12) if monthly.std() > 0 else 0
    return cum.iloc[-1] * 100, sharpe, dd * 100


def main():
    print("Generating v3 (30m/1.5:1) and v4 (15m/4:1) out-of-sample monthly returns...\n")
    v3 = gen(30, 1.5, 1)
    v4 = gen(15, 4.0, 1)
    idx = v3.index.union(v4.index)
    v3 = v3.reindex(idx, fill_value=0.0)
    v4 = v4.reindex(idx, fill_value=0.0)
    blend = 0.5 * v3 + 0.5 * v4
    corr = np.corrcoef(v3.values, v4.values)[0, 1]

    print(f"  {'strategy':>22} {'total%':>8} {'Sharpe':>7} {'maxDD%':>8}")
    for name, m in (("v3  (30m/1.5:1)", v3), ("v4  (15m/4:1)", v4), ("ENSEMBLE 50/50", blend)):
        t, s, d = stats(m)
        print(f"  {name:>22} {t:>+8.0f} {s:>7.2f} {d:>+8.0f}")
    print(f"\n  correlation(v3, v4) monthly = {corr:+.2f}   (lower = better diversification)")
    print("  The ensemble wins if its Sharpe beats both AND its drawdown is shallower than either.")

    Path("runs").mkdir(exist_ok=True)
    curve = {"months": [str(x.date()) for x in idx],
             "v3": list(np.round(v3.cumsum().values * 100, 1)),
             "v4": list(np.round(v4.cumsum().values * 100, 1)),
             "ens": list(np.round(blend.cumsum().values * 100, 1))}
    Path("runs/ensemble_curve.json").write_text(json.dumps(curve))
    print("  saved runs/ensemble_curve.json")


if __name__ == "__main__":
    main()

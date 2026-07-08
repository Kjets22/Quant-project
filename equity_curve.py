"""
equity_curve.py — chronological out-of-sample equity curve of the S/R + selectivity
strategy (1:1, top-7%, 3 bps), pooled across the 8-name basket, 1 unit per signal.
Outputs summary stats + a sampled (date, cum_return%) series for plotting.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import lightgbm as lgb

from triple_barrier_ml import atr, features, hourly, label
from triple_barrier_breadth import TICKERS
from sr_features import sr_features

COST_BPS = 3.0
SEL_Q = 0.93
TP = SL = 1.0
HBAR = 24


def trades_of(ticker):
    d = hourly(ticker)
    ts = pd.to_datetime(d["timestamp"]).to_numpy()
    h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
    A = atr(h, l, c)
    X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                   sr_features(d).reset_index(drop=True)], axis=1)
    y = label(h, l, c, A, TP, SL)
    m = (X.notna().all(axis=1) & np.isfinite(y)).to_numpy()
    idx = np.where(m)[0]
    Xv, yv = X.iloc[idx].reset_index(drop=True), y[idx].astype(int)
    hv, lv, cv, Av, tsv = h[idx], l[idx], c[idx], A[idx], ts[idx]
    n = len(idx)
    K = 5
    bnds = np.linspace(int(n * 0.4), n, K + 1).astype(int)
    out = []
    for k in range(K):
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                 min_child_samples=40, subsample=0.8,
                                 colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
        clf.fit(Xv.iloc[:bnds[k]], yv[:bnds[k]])
        thr = np.quantile(clf.predict_proba(Xv.iloc[:bnds[k]])[:, 1], SEL_Q)
        proba = clf.predict_proba(Xv.iloc[bnds[k]:bnds[k + 1]])[:, 1]
        i = bnds[k]
        while i < bnds[k + 1] - 1:
            if proba[i - bnds[k]] < thr:
                i += 1
                continue
            a = Av[i]
            up, dn = cv[i] + TP * a, cv[i] - SL * a
            res, j = None, i + 1
            while j < min(i + HBAR + 1, n):
                if lv[j] <= dn:
                    res = 0; break
                if hv[j] >= up:
                    res = 1; break
                j += 1
            if res is None:
                res = 1 if cv[min(j, n - 1)] > cv[i] else 0
            pnl = (TP * a if res == 1 else -SL * a) - COST_BPS / 1e4 * cv[i]
            out.append((tsv[i], pnl / cv[i]))
            i = j + 1
    return out


def main():
    rows = []
    for tk in TICKERS:
        try:
            rows += trades_of(tk)
        except Exception:
            continue
    df = pd.DataFrame(rows, columns=["ts", "ret"]).sort_values("ts").reset_index(drop=True)
    df["cum"] = df["ret"].cumsum() * 100
    df["peak"] = df["cum"].cummax()
    maxdd = (df["cum"] - df["peak"]).min()

    # monthly
    df["m"] = pd.to_datetime(df["ts"]).dt.to_period("M")
    mon = df.groupby("m")["ret"].sum() * 100
    pos_months = (mon > 0).mean()

    print(f"OOS trades={len(df)}  total={df['cum'].iloc[-1]:.1f}%  maxDD={maxdd:.1f}%  "
          f"win%={(df['ret']>0).mean():.1%}  positive months={pos_months:.0%} "
          f"({(mon>0).sum()}/{len(mon)})")
    print(f"  best month +{mon.max():.1f}%  worst month {mon.min():.1f}%")

    # sampled series for plotting (~140 points)
    step = max(1, len(df) // 140)
    s = df.iloc[::step]
    pts = [{"d": str(pd.to_datetime(t).date()), "y": round(float(c), 2)}
           for t, c in zip(s["ts"], s["cum"])]
    pts.append({"d": str(pd.to_datetime(df["ts"].iloc[-1]).date()),
                "y": round(float(df["cum"].iloc[-1]), 2)})
    monthly = [{"m": str(m), "r": round(float(r), 2)} for m, r in mon.items()]
    Path("runs").mkdir(exist_ok=True)
    Path("runs/equity_curve.json").write_text(json.dumps(
        {"curve": pts, "monthly": monthly, "maxdd": round(float(maxdd), 1),
         "total": round(float(df["cum"].iloc[-1]), 1)}, indent=0))
    print("saved runs/equity_curve.json")


if __name__ == "__main__":
    main()

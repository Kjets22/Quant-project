"""
research_c.py — forensic deep-dive on cell C (30-ATR target : 3-ATR stop = 10:1 payoff,
60-min, HBAR=96, trend features). It looked positive on BOTH universes (+81% dev / +78%
fresh) but with very few winners. This script answers: is that P&L a broad effect or a
couple of lucky monsters?

Reports per universe:
  * every WINNING trade (date, ticker, % return) - the anatomy
  * concentration: total after removing the top-1/2/3 winners
  * per-ticker and per-year P&L breakdown
  * t-stat of the mean per-trade return
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

from triple_barrier_ml import features
from sr_features import sr_features
from wide_hunter import atr_fixed, trend_features

MINS, HBAR = 60, 96
TP, SL = 30.0, 3.0
SEL_Q = 0.93
EFF_COST = (3.0 + 2 * 1.0) / 1e4
MIN_ATR_PCT = 0.0012
DEV = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "JPM", "XLE", "TLT"]
FRESH = ["IWM", "GLD", "META", "XOM", "KO"]


def collect(names):
    rows = []
    for tk in names:
        df = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv", parse_dates=["timestamp"])
        d = df.set_index("timestamp").resample(f"{MINS}min").agg(
            high=("high", "max"), low=("low", "min"), close=("close", "last"),
            volume=("volume", "sum")).dropna().reset_index()
        ts = pd.to_datetime(d["timestamp"]).to_numpy()
        h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
        A = atr_fixed(h, l, c)
        X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                       sr_features(d).reset_index(drop=True),
                       trend_features(h, l, c, A).reset_index(drop=True)], axis=1)
        n = len(c)
        y = np.full(n, np.nan)
        for i in range(n - 1):
            if not np.isfinite(A[i]):
                continue
            up, dn = c[i] + TP * A[i], c[i] - SL * A[i]
            for j in range(i + 1, min(i + HBAR + 1, n)):
                if l[j] <= dn:
                    y[i] = 0; break
                if h[j] >= up:
                    y[i] = 1; break
        m = (X.notna().all(axis=1) & np.isfinite(y) & np.isfinite(A)).to_numpy()
        idx = np.where(m)[0]
        Xv, yv = X.iloc[idx].reset_index(drop=True), y[idx].astype(int)
        hv, lv, cv, Av, tsv = h[idx], l[idx], c[idx], A[idx], ts[idx]
        nn = len(idx); K = 5
        bnds = np.linspace(int(nn * 0.4), nn, K + 1).astype(int)
        for k in range(K):
            tr_end = max(bnds[k] - HBAR, 300)
            if yv[:tr_end].sum() < 20:
                continue
            clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                     min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                                     reg_lambda=1.0, verbose=-1)
            clf.fit(Xv.iloc[:tr_end], yv[:tr_end])
            thr = np.quantile(clf.predict_proba(Xv.iloc[:tr_end])[:, 1], SEL_Q)
            proba = clf.predict_proba(Xv.iloc[bnds[k]:bnds[k + 1]])[:, 1]
            i = bnds[k]
            while i < bnds[k + 1] - 1:
                if proba[i - bnds[k]] < thr or Av[i] / cv[i] < MIN_ATR_PCT:
                    i += 1; continue
                a = Av[i]; up, dn = cv[i] + TP * a, cv[i] - SL * a
                res, j = None, i + 1
                while j < min(i + HBAR + 1, nn):
                    if lv[j] <= dn:
                        res = 0; break
                    if hv[j] >= up:
                        res = 1; break
                    j += 1
                ex = min(j, nn - 1)
                if res == 1:
                    r = TP * a / cv[i] - EFF_COST; oc = "TGT"
                elif res == 0:
                    r = -SL * a / cv[i] - EFF_COST; oc = "STP"
                else:
                    r = (cv[ex] - cv[i]) / cv[i] - EFF_COST; oc = "TIME"
                rows.append((pd.Timestamp(tsv[i]), tk, oc, r))
                i = j + 1
    return rows


def analyze(tag, rows):
    r = np.array([x[3] for x in rows])
    total = r.sum() * 100
    tstat = r.mean() / (r.std() / np.sqrt(len(r))) if len(r) > 3 else 0
    wins = sorted([x for x in rows if x[2] == "TGT"], key=lambda x: -x[3])
    times_pos = [x for x in rows if x[2] == "TIME" and x[3] > 0]
    print(f"\n===== {tag}: {len(rows)} trades, total {total:+.0f}%, "
          f"t-stat {tstat:+.2f} =====")
    print(f"  outcome mix: TGT={sum(x[2]=='TGT' for x in rows)}  "
          f"STP={sum(x[2]=='STP' for x in rows)}  TIME={sum(x[2]=='TIME' for x in rows)} "
          f"(profitable TIME exits: {len(times_pos)}, {sum(x[3] for x in times_pos)*100:+.0f}%)")
    print(f"  ALL {len(wins)} target-winners:")
    for t, tk, _, ret in wins:
        print(f"    {str(t)[:10]}  {tk:>5}  {ret*100:+6.1f}%")
    srt = sorted(r, reverse=True)
    for kk in (1, 2, 3):
        print(f"  total after dropping top-{kk} trade(s): {(r.sum()-sum(srt[:kk]))*100:+.0f}%")
    per_tk = {}
    per_yr = {}
    for t, tk, _, ret in rows:
        per_tk[tk] = per_tk.get(tk, 0) + ret
        per_yr[t.year] = per_yr.get(t.year, 0) + ret
    print("  per-ticker: " + "  ".join(f"{k} {v*100:+.0f}%" for k, v in
                                       sorted(per_tk.items(), key=lambda x: -x[1])))
    print("  per-year:   " + "  ".join(f"{k} {v*100:+.0f}%" for k, v in sorted(per_yr.items())))


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("dev", "both"):
        analyze("DEV", collect(DEV))
    if which in ("fresh", "both"):
        analyze("FRESH", collect(FRESH))


if __name__ == "__main__":
    main()

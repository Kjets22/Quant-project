"""
wide_sweep.py — hunt for MORE profitable algos: wide targets 5:1 .. 10:1 across timeframes,
in the SAME audited environment as edge_proof (embargo, no-bfill ATR, 0.12% ATR floor,
5 bps effective cost incl slippage), walk-forward, top-7%, pooled basket.

Wide targets need only a low win rate to break even (10:1 -> 16.7%... 1/11=9.1%). Question:
can the mean-reversion model still clear the (low) bar when it's asking for a BIG up-move?
Prints each cell as it finishes and flags positive P&L. Standalone.

  python wide_sweep.py                 # 60,30,15 min x 5..10 :1
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import lightgbm as lgb

from triple_barrier_ml import features, label
from triple_barrier_breadth import TICKERS
from sr_features import sr_features

SEL_Q, HBAR = 0.93, 24
EFF_COST = (3.0 + 2 * 1.0) / 1e4
MIN_ATR_PCT = 0.0012
TIMEFRAMES = [60, 30, 15]
RATIOS = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]


def atr_fixed(h, l, c, n=24):
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    return pd.Series(tr).rolling(n).mean().to_numpy()


def bars(tk, mins):
    df = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv", parse_dates=["timestamp"])
    g = df.set_index("timestamp").resample(f"{mins}min").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum")).dropna()
    return g.reset_index()


def run_cell(mins, tp, sl=1.0):
    TT = TW = 0
    SR = 0.0
    for tk in TICKERS:
        d = bars(tk, mins)
        h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
        A = atr_fixed(h, l, c)
        X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                       sr_features(d).reset_index(drop=True)], axis=1)
        y = label(h, l, c, A, tp, sl)
        m = (X.notna().all(axis=1) & np.isfinite(y) & np.isfinite(A)).to_numpy()
        idx = np.where(m)[0]
        Xv, yv = X.iloc[idx].reset_index(drop=True), y[idx].astype(int)
        hv, lv, cv, Av = h[idx], l[idx], c[idx], A[idx]
        n = len(idx); K = 5
        bnds = np.linspace(int(n * 0.4), n, K + 1).astype(int)
        for k in range(K):
            tr_end = max(bnds[k] - HBAR, 300)
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
                SR += (tp * a if res == 1 else -sl * a) / cv[i] - EFF_COST
                i = j + 1
    be = sl / (sl + tp)
    return TT, TW / TT, be, SR * 100, SR / TT * 1e4


def main():
    out = Path("runs/wide_sweep.txt"); out.parent.mkdir(exist_ok=True)
    hdr = ("WIDE-TARGET SWEEP (audited env: embargo, ATR floor, 5 bps cost). "
           "Looking for margin>0 AND total>0.\n"
           f"  {'timeframe':>10} {'ratio':>6} {'trades':>7} {'win%':>6} {'BE':>5} {'margin':>8} {'mean bps':>9} {'total%':>8}")
    print(hdr, flush=True)
    with out.open("w") as fh:
        fh.write(hdr + "\n")
        for mins in TIMEFRAMES:
            for tp in RATIOS:
                n, wr, be, tot, bps = run_cell(mins, tp)
                flag = "  <== POSITIVE" if (wr > be and tot > 0) else ""
                line = (f"  {str(mins)+'-min':>10} {tp:g}:1  {n:>7} {wr:>6.1%} {be:>5.0%} "
                        f"{wr-be:>+8.1%} {bps:>+9.1f} {tot:>+8.0f}{flag}")
                print(line, flush=True); fh.write(line + "\n")
    print("\nDONE. Positive cells still need a fresh-ticker holdout before trust (this is in-sample sweep).", flush=True)


if __name__ == "__main__":
    main()

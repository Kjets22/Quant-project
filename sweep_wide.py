"""
sweep_wide.py — wide-target search: 5:1 up to 10:1 across timeframes, on the AUDITED engine
(embargo, no-bfill ATR, 0.12% ATR floor, 5 bps effective cost) with CORRECT time-barrier
accounting (timeout exits at the actual close — no phantom target credit; this matters a lot
for wide targets and is also applied to the v3/v4 baselines for honest comparison).

  python sweep_wide.py <minutes> <ratios-csv>     e.g.  python sweep_wide.py 30 1.5,5,7,10

Reports: trades, target-hit% vs break-even, % timeouts, mean bps, total%, monthly Sharpe.
Standalone; touches no frozen snapshot.
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
from triple_barrier_breadth import TICKERS
from sr_features import sr_features

SEL_Q, HBAR = 0.93, 24
EFF_COST = 5.0 / 1e4
MIN_ATR_PCT = 0.0012


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


def label_strict(h, l, c, A, tp, sl):
    """Triple-barrier label; timeout labeled by ACTUAL sign of drift (for training)."""
    n = len(c)
    y = np.full(n, np.nan)
    for i in range(n - 1):
        a = A[i]
        if not np.isfinite(a):
            continue
        up, dn = c[i] + tp * a, c[i] - sl * a
        hit = False
        for j in range(i + 1, min(i + HBAR + 1, n)):
            if l[j] <= dn:
                y[i] = 0; hit = True; break
            if h[j] >= up:
                y[i] = 1; hit = True; break
        if not hit:
            ex = min(i + HBAR, n - 1)
            y[i] = 1 if c[ex] > c[i] else 0
    return y


def run(mins, tp, sl):
    rows = []
    for tk in TICKERS:
        d = bars(tk, mins)
        ts = pd.to_datetime(d["timestamp"]).to_numpy()
        h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
        A = atr_fixed(h, l, c)
        X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                       sr_features(d).reset_index(drop=True)], axis=1)
        y = label_strict(h, l, c, A, tp, sl)
        m = (X.notna().all(axis=1) & np.isfinite(y) & np.isfinite(A)).to_numpy()
        idx = np.where(m)[0]
        Xv, yv = X.iloc[idx].reset_index(drop=True), y[idx].astype(int)
        hv, lv, cv, Av, tsv = h[idx], l[idx], c[idx], A[idx], ts[idx]
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
                if res is None:                             # TIMEOUT -> actual close
                    ex = min(j, n - 1)
                    ret = (cv[ex] - cv[i]) / cv[i] - EFF_COST
                    oc = 2
                else:
                    ret = (tp * a if res == 1 else -sl * a) / cv[i] - EFF_COST
                    oc = res
                    ex = j
                rows.append((tsv[i], ret, oc))
                i = ex + 1
    df = pd.DataFrame(rows, columns=["ts", "ret", "oc"])
    if not len(df):
        return None
    mon = df.set_index(pd.to_datetime(df["ts"]))["ret"].resample("ME").sum()
    sh = mon.mean() / mon.std() * np.sqrt(12) if mon.std() > 0 else 0
    be = sl / (sl + tp)
    n_ = len(df)
    tgt = (df["oc"] == 1).mean()
    tmo = (df["oc"] == 2).mean()
    return n_, tgt, be, tmo, df["ret"].mean() * 1e4, df["ret"].sum() * 100, sh


def main():
    mins = int(sys.argv[1])
    ratios = [float(x) for x in sys.argv[2].split(",")]
    print(f"=== {mins}-min candles, audited env, STRICT timeout accounting ===", flush=True)
    print(f"  {'tgt:stop':>8} {'trades':>7} {'tgt-hit%':>9} {'be%':>5} {'timeout%':>9} "
          f"{'mean bps':>9} {'total%':>8} {'Sharpe':>7}")
    for tp in ratios:
        r = run(mins, tp, 1.0)
        if r is None:
            print(f"  {tp:g}:1  no trades", flush=True); continue
        n_, tgt, be, tmo, bps, tot, sh = r
        print(f"  {tp:g}:1    {n_:>7} {tgt:>9.1%} {be:>5.0%} {tmo:>9.1%} "
              f"{bps:>+9.1f} {tot:>+8.0f} {sh:>7.2f}", flush=True)


if __name__ == "__main__":
    main()

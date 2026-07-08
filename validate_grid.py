"""
validate_grid.py — generalized validation gauntlet for ANY timeframe / ratio.
Same engine (base+S/R, top-7%, 3 bps, H=24 bars, walk-forward, non-overlapping).

  python validate_grid.py <minutes> <tp> <sl> [fresh]

  python validate_grid.py 15 4 1            # 15-min/4:1 on dev names (DSR)
  python validate_grid.py 15 4 1 fresh      # 15-min/4:1 on the 5 FRESH holdout names
  python validate_grid.py 30 1.5 1 fresh    # re-confirm the validated one

Standalone; touches no frozen snapshot.
"""

from __future__ import annotations

import math
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import lightgbm as lgb

from triple_barrier_ml import atr, features, label
from triple_barrier_breadth import TICKERS as DEV
from sr_features import sr_features
from trials import deflated_sharpe_ratio

FRESH = ["IWM", "GLD", "META", "XOM", "KO"]
MIN = int(sys.argv[1]) if len(sys.argv) > 1 else 30
TP = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
SL = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
MODE = "fresh" if "fresh" in sys.argv else "dev"
SEL_Q = 0.93
HBAR = 24
COST_BPS = 3.0


def bars(tk):
    df = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv",
                     parse_dates=["timestamp"])
    g = df.set_index("timestamp").resample(f"{MIN}min").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum")).dropna()
    return g.reset_index()


def trades_of(ticker):
    d = bars(ticker)
    h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
    A = atr(h, l, c)
    X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                   sr_features(d).reset_index(drop=True)], axis=1)
    y = label(h, l, c, A, TP, SL)
    m = (X.notna().all(axis=1) & np.isfinite(y)).to_numpy()
    idx = np.where(m)[0]
    Xv, yv = X.iloc[idx].reset_index(drop=True), y[idx].astype(int)
    hv, lv, cv, Av = h[idx], l[idx], c[idx], A[idx]
    n = len(idx)
    K = 5
    bnds = np.linspace(int(n * 0.4), n, K + 1).astype(int)
    rets, fold = [], []
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
            out, j = None, i + 1
            while j < min(i + HBAR + 1, n):
                if lv[j] <= dn:
                    out = 0; break
                if hv[j] >= up:
                    out = 1; break
                j += 1
            if out is None:
                out = 1 if cv[min(j, n - 1)] > cv[i] else 0
            rets.append((TP * a if out == 1 else -SL * a) / cv[i] - COST_BPS / 1e4)
            fold.append(k)
            i = j + 1
    return np.array(rets), np.array(fold)


def main():
    names = FRESH if MODE == "fresh" else DEV
    n_trials = 1 if MODE == "fresh" else 35
    allr, allf = [], []
    for tk in names:
        try:
            r, f = trades_of(tk)
        except Exception as e:
            print(f"  [skip {tk}] {e}")
            continue
        allr.append(r); allf.append(f)
    if not allr:
        print("no trades / no data"); return
    r = np.concatenate(allr); f = np.concatenate(allf)
    n = len(r); yrs = 3.0; per_year = n / yrs
    sh_ann = r.mean() / r.std() * math.sqrt(per_year)
    fsh = [r[f == k].mean() / r[f == k].std() * math.sqrt(per_year)
           for k in sorted(set(f)) if r[f == k].std() > 1e-12]
    dsr = deflated_sharpe_ratio(r.mean() / r.std(), n, n_trials, np.var(fsh) / per_year,
                                skew=float(pd.Series(r).skew()),
                                kurt=float(pd.Series(r).kurt()) + 3)
    be = SL / (SL + TP)
    print(f"=== VALIDATION: {MIN}-min / {TP:g}:{SL:g} / top-7% / 3 bps  "
          f"[{MODE.upper()}: {names}] ===")
    print(f"  break-even win% = {be:.0%}")
    print(f"  trades={n}  win%={(r>0).mean():.1%}  total={r.sum()*100:.0f}%  "
          f"mean/trade={r.mean()*1e4:+.1f} bps")
    print(f"  annualized Sharpe ~ {sh_ann:.2f}")
    print(f"  per-fold Sharpes: {[round(x,2) for x in fsh]}  "
          f"({sum(x>0 for x in fsh)}/{len(fsh)} positive)")
    print(f"  DEFLATED SHARPE (P>0 after {n_trials} trial{'s' if n_trials>1 else ''}) = {dsr:.3f}")
    print("  FRESH holdout: win% comfortably > break-even + positive Sharpe = real, not overfit."
          if MODE == "fresh" else "  >0.95 => survives multiple-testing on the dev names.")


if __name__ == "__main__":
    main()

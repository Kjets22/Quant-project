"""
noise_test.py — NOISE-REDUCTION experiment on the model's signal (v3 config: 30-min/1.5:1).

The model's per-bar probability is itself a noisy estimate. Three classic denoising
schemes applied to the SIGNAL (not the data), all causal:
  baseline : enter when p[t]              >= thr   (single-bar trigger, as today)
  smooth2  : enter when (p[t]+p[t-1])/2   >= thr   (2-bar moving average of conviction)
  persist2 : enter when p[t] >= thr AND p[t-1] >= thr  (signal must PERSIST 2 bars)
If a single-bar spike is noise, smoothing/persistence should raise win% and Sharpe at the
cost of some trades. Audited env: embargo, no-bfill ATR, ATR floor, 5 bps effective cost,
corrected accounting. Nothing saved is modified.

  python noise_test.py [dev|fresh]
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

from triple_barrier_ml import features, label
from sr_features import sr_features
from wide_hunter import atr_fixed

MINS, TP, SL = 30, 1.5, 1.0
SEL_Q, HBAR = 0.93, 24
EFF_COST = (3.0 + 2 * 1.0) / 1e4
MIN_ATR_PCT = 0.0012
DEV = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "JPM", "XLE", "TLT"]
FRESH = ["IWM", "GLD", "META", "XOM", "KO"]
VARIANTS = ["baseline", "smooth2", "persist2"]


def take_mask(p, thr, variant):
    prev = np.concatenate([[p[0]], p[:-1]])
    if variant == "baseline":
        return p >= thr
    if variant == "smooth2":
        return (p + prev) / 2 >= thr
    return (p >= thr) & (prev >= thr)              # persist2


def run(names):
    out = {v: [] for v in VARIANTS}
    for tk in names:
        df = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv", parse_dates=["timestamp"])
        d = df.set_index("timestamp").resample(f"{MINS}min").agg(
            high=("high", "max"), low=("low", "min"), close=("close", "last"),
            volume=("volume", "sum")).dropna().reset_index()
        ts = pd.to_datetime(d["timestamp"]).to_numpy()
        h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
        A = atr_fixed(h, l, c)
        X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                       sr_features(d).reset_index(drop=True)], axis=1)
        y = label(h, l, c, A, TP, SL)
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
            p = clf.predict_proba(Xv.iloc[bnds[k]:bnds[k + 1]])[:, 1]
            for variant in VARIANTS:
                tm = take_mask(p, thr, variant)
                i = bnds[k]
                while i < bnds[k + 1] - 1:
                    if not tm[i - bnds[k]] or Av[i] / cv[i] < MIN_ATR_PCT:
                        i += 1; continue
                    a = Av[i]; up, dn = cv[i] + TP * a, cv[i] - SL * a
                    res, j = None, i + 1
                    while j < min(i + HBAR + 1, n):
                        if lv[j] <= dn:
                            res = 0; break
                        if hv[j] >= up:
                            res = 1; break
                        j += 1
                    ex = min(j, n - 1)
                    if res == 1:
                        r = TP * a / cv[i] - EFF_COST
                    elif res == 0:
                        r = -SL * a / cv[i] - EFF_COST
                    else:
                        r = (cv[ex] - cv[i]) / cv[i] - EFF_COST
                    out[variant].append((pd.Timestamp(tsv[i]), r, 1 if res == 1 else 0))
                    i = j + 1
    return out


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "dev"
    names = FRESH if which == "fresh" else DEV
    res = run(names)
    be = SL / (SL + TP)
    print(f"SIGNAL-DENOISING test, v3 config (30-min/1.5:1) [{which.upper()}], BE={be:.0%}")
    print(f"  {'variant':>9} {'trades':>7} {'win%':>6} {'margin':>8} {'mean bps':>9} "
          f"{'total%':>8} {'Sharpe':>7}")
    for variant in VARIANTS:
        rows = res[variant]
        r = np.array([x[1] for x in rows])
        w = np.mean([x[2] for x in rows])
        s = pd.DataFrame(rows, columns=["ts", "r", "w"])
        mon = s.set_index(pd.to_datetime(s["ts"]))["r"].resample("ME").sum()
        sharpe = mon.mean() / mon.std() * np.sqrt(12) if mon.std() > 0 else 0.0
        print(f"  {variant:>9} {len(r):>7} {w:>6.1%} {w-be:>+8.1%} {r.mean()*1e4:>+9.1f} "
              f"{r.sum()*100:>+8.0f} {sharpe:>7.2f}")


if __name__ == "__main__":
    main()

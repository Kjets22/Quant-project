"""
vt_test.py — "vT: the time-harvester", DERIVED from the winners' DNA.

The vC forensics showed its P&L engine is the profitable TIME exits (53/54 positive),
not the 30-ATR target. vT distills that: SAME entry model as vC (trend features, top-7%
conviction trained on the 30:3 label), SAME 3-ATR stop, but NO TARGET — exit at the
clock (48 or 96 bars) or the stop, whichever first. Winners run to wherever the trend
carried them. Audited env, corrected accounting. Nothing saved is modified.

  python vt_test.py [dev|fresh]
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
TP_LBL, SL = 30.0, 3.0             # label = vC's (validated trend-selector)
SEL_Q = 0.93
EFF_COST = (3.0 + 2 * 1.0) / 1e4
MIN_ATR_PCT = 0.0012
DEV = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "JPM", "XLE", "TLT"]
FRESH = ["IWM", "GLD", "META", "XOM", "KO"]
VARIANTS = [("vC ref (30-ATR tgt, 96c)", "target", 96),
            ("vT-96 (no tgt, 96-bar clock)", "clock", 96),
            ("vT-48 (no tgt, 48-bar clock)", "clock", 48)]


def run(names):
    out = {v[0]: [] for v in VARIANTS}
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
        y = np.full(n, np.nan)                     # vC's 30:3 label for training
        for i in range(n - 1):
            if not np.isfinite(A[i]):
                continue
            up, dn = c[i] + TP_LBL * A[i], c[i] - SL * A[i]
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
            p = clf.predict_proba(Xv.iloc[bnds[k]:bnds[k + 1]])[:, 1]
            for name, mode, clock in VARIANTS:
                i = bnds[k]
                while i < bnds[k + 1] - 1:
                    if p[i - bnds[k]] < thr or Av[i] / cv[i] < MIN_ATR_PCT:
                        i += 1; continue
                    a = Av[i]
                    up = cv[i] + TP_LBL * a if mode == "target" else np.inf
                    dn = cv[i] - SL * a
                    res, j = None, i + 1
                    while j < min(i + clock + 1, nn):
                        if lv[j] <= dn:
                            res = 0; break
                        if hv[j] >= up:
                            res = 1; break
                        j += 1
                    ex = min(j, nn - 1)
                    if res == 1:
                        r = TP_LBL * a / cv[i] - EFF_COST
                    elif res == 0:
                        r = -SL * a / cv[i] - EFF_COST
                    else:
                        r = (cv[ex] - cv[i]) / cv[i] - EFF_COST
                    out[name].append((pd.Timestamp(tsv[i]), r))
                    i = j + 1
    return out


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "dev"
    res = run(FRESH if which == "fresh" else DEV)
    print(f"vT time-harvester vs vC [{which.upper()}]")
    print(f"  {'variant':>30} {'trades':>7} {'win%':>6} {'mean bps':>9} {'total%':>8} {'Sharpe':>7}")
    for name, *_ in VARIANTS:
        rows = res[name]
        r = np.array([x[1] for x in rows])
        w = (r > 0).mean()
        s = pd.DataFrame(rows, columns=["ts", "r"])
        mon = s.set_index(pd.to_datetime(s["ts"]))["r"].resample("ME").sum()
        sharpe = mon.mean() / mon.std() * np.sqrt(12) if mon.std() > 0 else 0.0
        print(f"  {name:>30} {len(r):>7} {w:>6.1%} {r.mean()*1e4:>+9.1f} "
              f"{r.sum()*100:>+8.0f} {sharpe:>7.2f}")


if __name__ == "__main__":
    main()

"""
option_train_spy.py — does training on OPTION-MARKET data help the SPY signal?

Builds daily option-derived features from the real SPY chain (ATM implied vol, 25-delta
skew, put/call volume & OI ratios, IV minus realized vol), broadcasts them onto the 30-min
bars (shifted 1 day -> causal), and A/B tests the model WITH vs WITHOUT them on the option-
data window. SPY-only, ~14 months, single train/test split -> EXPLORATORY, not validatable.
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

from triple_barrier_ml import atr, features, label
from sr_features import sr_features

TP, SL = 1.5, 1.0
SEL_Q, HBAR, COST_BPS = 0.93, 24, 3.0
MIN = 30


def option_daily_features():
    o = pd.read_parquet("data_cache/options/chain_SPY_2025-04-01_2026-06-20.parquet")
    o = o[(o["iv"] > 0.03) & (o["iv"] < 1.5)].copy()
    rows = []
    for d, g in o.groupby(o["date"].dt.normalize()):
        c = g[g["type"] == "C"]; p = g[g["type"] == "P"]
        if len(c) < 3 or len(p) < 3:
            continue
        atm = g.iloc[(g["delta"].abs() - 0.5).abs().to_numpy().argmin()]["iv"]
        c25 = c.iloc[(c["delta"] - 0.25).abs().to_numpy().argmin()]["iv"]
        p25 = p.iloc[(p["delta"] + 0.25).abs().to_numpy().argmin()]["iv"]
        rows.append({
            "date": d, "atm_iv": atm, "skew": p25 - c25,
            "pc_vol": p["volume"].sum() / max(c["volume"].sum(), 1),
            "pc_oi": p["open_interest"].sum() / max(c["open_interest"].sum(), 1),
        })
    f = pd.DataFrame(rows).set_index("date").sort_index()
    return f.shift(1)            # causal: only yesterday's EOD chain is known intraday


def spy_30m():
    df = pd.read_csv("data_cache/SPY_5minute_2021-06-01_2026-06-01.csv", parse_dates=["timestamp"])
    g = df.set_index("timestamp").resample(f"{MIN}min").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum")).dropna()
    return g.reset_index()


def trade_block(proba, thr, h, l, c, A, i0, i1, n):
    rets = []
    i = i0
    while i < i1 - 1:
        if proba[i - i0] < thr:
            i += 1; continue
        a = A[i]; up, dn = c[i] + TP * a, c[i] - SL * a
        res, j = None, i + 1
        while j < min(i + HBAR + 1, n):
            if l[j] <= dn:
                res = 0; break
            if h[j] >= up:
                res = 1; break
            j += 1
        if res is None:
            res = 1 if c[min(j, n - 1)] > c[i] else 0
        rets.append((TP * a if res == 1 else -SL * a) / c[i] - COST_BPS / 1e4)
        i = j + 1
    return np.array(rets)


def main():
    d = spy_30m()
    od = option_daily_features()
    win0, win1 = od.index.min(), od.index.max()
    ts = pd.to_datetime(d["timestamp"])
    h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
    A = atr(h, l, c)
    base = pd.concat([features(h, l, c, v).reset_index(drop=True),
                      sr_features(d).reset_index(drop=True)], axis=1)
    # broadcast daily option features onto 30-min bars by date
    day = ts.dt.normalize().to_numpy()
    of = od.reindex(pd.to_datetime(day)).reset_index(drop=True)
    of.columns = ["opt_" + x for x in of.columns]
    y = label(h, l, c, A, TP, SL)

    inwin = (ts >= win0) & (ts <= win1)
    print(f"SPY option-data window: {win0.date()} .. {win1.date()}  ({inwin.sum()} 30-min bars)")
    print("  A/B: base+S/R features  vs  base+S/R + option-market features\n")
    print(f"  {'features':>22} {'OOS trades':>11} {'win%':>6} {'mean bps':>9} {'total%':>8}")

    for tag, X in (("base + S/R", base), ("base + S/R + OPTIONS", pd.concat([base, of], axis=1))):
        m = (X.notna().all(axis=1) & np.isfinite(y) & inwin.to_numpy()).to_numpy()
        idx = np.where(m)[0]
        Xv, yv = X.iloc[idx].reset_index(drop=True), y[idx].astype(int)
        hv, lv, cv, Av = h[idx], l[idx], c[idx], A[idx]
        nn = len(idx); cut = int(nn * 0.6)
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                 min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                                 reg_lambda=1.0, verbose=-1)
        clf.fit(Xv.iloc[:cut], yv[:cut])
        thr = np.quantile(clf.predict_proba(Xv.iloc[:cut])[:, 1], SEL_Q)
        proba = clf.predict_proba(Xv.iloc[cut:])[:, 1]
        r = trade_block(proba, thr, hv, lv, cv, Av, cut, nn, nn)
        wp = f"{(r>0).mean():.1%}" if len(r) else "-"
        mb = f"{r.mean()*1e4:+.1f}" if len(r) else "-"
        tt = f"{r.sum()*100:+.0f}" if len(r) else "-"
        print(f"  {tag:>22} {len(r):>11} {wp:>6} {mb:>9} {tt:>8}")
    print("\n  Did OPTIONS features beat base on win%/mean-bps OOS? If not (or barely), the option")
    print("  data didn't add signal here -- expected, given SPY-only + ~14 months is far too little.")


if __name__ == "__main__":
    main()

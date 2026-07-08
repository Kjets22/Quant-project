"""
real_options_spy.py — validate the options overlay against REAL SPY option quotes.

Generates SPY 30-min/1.5:1 signals over the option-data window (2025-04..2026-06), with
the model trained ONLY on data before that window (clean OOS forward test). For each
signal it buys a REAL deep-ITM weekly call at the actual ASK and sells at the actual BID
from the cached Polygon chain — replacing the Black-Scholes model's biggest assumption
(the spread) with real fills. Reports real vs BS side by side.

Daily-snapshot approximation: entry = signal-day ask, exit = barrier-day bid (option data
is one snapshot/day). Standalone; touches no frozen snapshot.
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

MIN = 30
TP, SL = 1.5, 1.0
SEL_Q = 0.93
HBAR = 24
WIN_START = pd.Timestamp("2025-04-01")
WIN_END = pd.Timestamp("2026-06-01")
TGT_DELTA = 0.88           # deep-ITM call
DTE_LO, DTE_HI = 4, 11     # ~weekly


def spy_bars():
    df = pd.read_csv("data_cache/SPY_5minute_2021-06-01_2026-06-01.csv", parse_dates=["timestamp"])
    g = df.set_index("timestamp").resample(f"{MIN}min").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum")).dropna()
    return g.reset_index()


def main():
    d = spy_bars()
    ts = pd.to_datetime(d["timestamp"])
    h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
    A = atr(h, l, c)
    X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                   sr_features(d).reset_index(drop=True)], axis=1)
    y = label(h, l, c, A, TP, SL)
    fv = X.notna().all(axis=1).to_numpy()
    tsn = ts.to_numpy()
    n = len(c)
    tr = np.where(fv & np.isfinite(y) & (tsn < np.datetime64(WIN_START)))[0]
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                             min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=1.0, verbose=-1)
    clf.fit(X.iloc[tr], y[tr].astype(int))
    thr = np.quantile(clf.predict_proba(X.iloc[tr])[:, 1], SEL_Q)
    test = np.where(fv & (tsn >= np.datetime64(WIN_START)) & (tsn < np.datetime64(WIN_END)))[0]
    proba = {int(ix): float(p) for ix, p in zip(test, clf.predict_proba(X.iloc[test])[:, 1])}

    # real option chain (calls only), indexed by day
    opt = pd.read_parquet("data_cache/options/chain_SPY_2025-04-01_2026-06-20.parquet")
    opt = opt[opt["type"] == "C"].copy()
    opt["d"] = opt["date"].dt.normalize()
    by_day = {dd: g for dd, g in opt.groupby("d")}
    quote = {(r.d, r.expiry, r.strike): (r.bid, r.ask) for r in opt.itertuples()}

    sig = []   # (move%, real_ret, bs_ret, win)
    i, last = int(test[0]), int(test[-1])
    while i <= last:
        if proba.get(i, -1) < thr:
            i += 1
            continue
        a = A[i]; S0 = c[i]; up, dn = S0 + TP * a, S0 - SL * a
        res, j = None, i + 1
        while j < min(i + HBAR + 1, n):
            if l[j] <= dn:
                res = 0; break
            if h[j] >= up:
                res = 1; break
            j += 1
        ex_j = j if (res is not None and j < n) else min(j, n - 1)
        if res is None:
            res = 1 if c[ex_j] > S0 else 0
        S_exit = up if res == 1 else (dn if (l[ex_j] <= dn or h[ex_j] >= up) else c[ex_j])
        Dd = pd.Timestamp(tsn[i]).normalize()
        Ed = pd.Timestamp(tsn[ex_j]).normalize()
        g = by_day.get(Dd)
        if g is not None:
            dte = (g["expiry"] - Dd).dt.days
            cand = g[(dte >= DTE_LO) & (dte <= DTE_HI) & (g["delta"].between(0.78, 0.96))]
            if len(cand):
                row = cand.iloc[(cand["delta"] - TGT_DELTA).abs().to_numpy().argmin()]
                entry_ask = float(row["ask"])
                exq = quote.get((Ed, row["expiry"], row["strike"]))
                if entry_ask > 0.05:
                    if exq is not None:
                        exit_bid = float(exq[0])
                    elif Ed >= row["expiry"]:            # expired -> intrinsic
                        exit_bid = max(S_exit - float(row["strike"]), 0.0)
                    else:
                        exit_bid = None
                    if exit_bid is not None:
                        real_ret = (exit_bid - entry_ask) / entry_ask
                        # BS comparison on the same trade, tight 0.5% half-spread
                        from options_strategy import bs_call
                        sigv = float(row["iv"]) if 0.05 < row["iv"] < 2 else 0.2
                        K = float(row["strike"]); dt_days = max((Ed - Dd).days, 0.25)
                        p0 = bs_call(S0, K, max((row["expiry"] - Dd).days, 0.5) / 365.0, sigv)
                        p1 = bs_call(S_exit, K, max((row["expiry"] - Ed).days, 0.01) / 365.0, sigv)
                        bs_ret = (p1 * (1 - 0.005) - p0 * (1 + 0.005)) / (p0 * (1 + 0.005)) if p0 > 0.05 else 0
                        sig.append((a / S0, real_ret, bs_ret, res))
        i = (ex_j + 1) if res is not None else (j + 1)

    if not sig:
        print("no matched signals/quotes in window"); return
    arr = np.array(sig)
    mv, real, bs, win = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    thr10 = np.quantile(mv, 0.90)
    print(f"SPY real-option validation, {WIN_START.date()}..{WIN_END.date()} (model trained before)")
    print(f"  matched signals: {len(sig)}   (deep-ITM ~{TGT_DELTA} delta weekly calls, real ask/bid)\n")
    print(f"  {'subset':>16} {'n':>4} {'win%':>6} {'REAL net/trade':>15} {'REAL total':>11} {'BS net/trade':>13}")
    for tag, msk in (("all signals", np.ones(len(mv), bool)), ("top-10% move", mv >= thr10)):
        rr, bb, ww = real[msk], bs[msk], win[msk]
        print(f"  {tag:>16} {msk.sum():>4} {ww.mean():>6.0%} {rr.mean()*100:>+14.2f}% "
              f"{rr.sum()*100:>+10.0f}% {bb.mean()*100:>+12.2f}%")
    print("\n  REAL = actual ask-in/bid-out fills from the cached SPY chain.")
    print("  If REAL >= BS, the modeled options edge holds with real fills (spreads were tighter).")


if __name__ == "__main__":
    main()

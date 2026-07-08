"""
improve_all.py — test the 3 improvement candidates on the AUDITED engine (embargo, no-bfill
ATR, 0.12% ATR floor, 5 bps effective cost), for v3 and v4, on DEV and the FRESH holdout:

  A. REGIME FILTER  — skip entries when SPY's trailing vol is in its top decile (causal
                      expanding quantile). Mean reversion fails in vol spikes.
  B. VOL-TARGETED SIZING — weight trades by (name's typical ATR% / current ATR%), capped
                      [0.5, 2]: risk less in turbulence, more in calm. Signals unchanged.
  C. PER-NAME WEIGHTING — weight each name by its causal trailing win-rate (shrunk prior),
                      so chronic noise names self-downweight.

A candidate 'wins' only if it improves monthly Sharpe (and/or maxDD) on BOTH universes.
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

from triple_barrier_ml import features, label
from triple_barrier_breadth import TICKERS as DEV
from sr_features import sr_features

FRESH = ["IWM", "GLD", "META", "XOM", "KO"]
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


def spy_regime():
    """Causal bad-regime flags: SPY 30-min vol > expanding 90th pct. Returns (ts, bad)."""
    d = bars("SPY", 30)
    r = np.diff(np.log(d["close"].to_numpy()), prepend=0.0)
    vol = pd.Series(r).rolling(65).std()
    q = vol.expanding(min_periods=500).quantile(0.90)
    bad = (vol > q).fillna(False).to_numpy()
    return pd.to_datetime(d["timestamp"]).to_numpy(), bad


def gen_trades(tk, mins, tp, sl):
    d = bars(tk, mins)
    ts = pd.to_datetime(d["timestamp"]).to_numpy()
    h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
    A = atr_fixed(h, l, c)
    X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                   sr_features(d).reset_index(drop=True)], axis=1)
    y = label(h, l, c, A, tp, sl)
    m = (X.notna().all(axis=1) & np.isfinite(y) & np.isfinite(A)).to_numpy()
    idx = np.where(m)[0]
    Xv, yv = X.iloc[idx].reset_index(drop=True), y[idx].astype(int)
    hv, lv, cv, Av, tsv = h[idx], l[idx], c[idx], A[idx], ts[idx]
    n = len(idx); K = 5
    bnds = np.linspace(int(n * 0.4), n, K + 1).astype(int)
    out = []
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
            out.append((tsv[i], (tp * a if res == 1 else -sl * a) / cv[i] - EFF_COST,
                        res, a / cv[i], tk))
            i = j + 1
    return out


def stats(rows, w=None):
    if not rows:
        return 0, 0, 0, 0, 0
    df = pd.DataFrame(rows, columns=["ts", "ret", "win", "atrpct", "tk"])
    df["w"] = 1.0 if w is None else w
    df["wr_"] = df["ret"] * df["w"] / df["w"].mean()
    mon = df.set_index(pd.to_datetime(df["ts"]))["wr_"].resample("ME").sum()
    cum = mon.cumsum()
    dd = (cum - cum.cummax()).min() * 100
    sh = mon.mean() / mon.std() * np.sqrt(12) if mon.std() > 0 else 0
    return len(df), df["win"].mean(), cum.iloc[-1] * 100, sh, dd


def main():
    rts, rbad = spy_regime()
    for mins, tp, sl, name in ((30, 1.5, 1.0, "v3"), (15, 4.0, 1.0, "v4")):
        be = sl / (sl + tp)
        for uni_name, names in (("DEV", DEV), ("FRESH", FRESH)):
            rows = []
            for tk in names:
                try:
                    rows += gen_trades(tk, mins, tp, sl)
                except Exception as e:
                    print(f"  [skip {tk}] {e}", flush=True)
            rows.sort(key=lambda r: r[0])
            df = pd.DataFrame(rows, columns=["ts", "ret", "win", "atrpct", "tk"])
            # A: regime filter mask
            pos = np.searchsorted(rts, df["ts"].to_numpy(), side="right") - 1
            okA = ~rbad[np.clip(pos, 0, len(rbad) - 1)]
            # B: vol-target weights (causal per-name expanding mean atr%)
            wB = np.ones(len(df))
            for tk in df["tk"].unique():
                msk = (df["tk"] == tk).to_numpy()
                x = df.loc[msk, "atrpct"].to_numpy()
                em = pd.Series(x).expanding().mean().shift(1).bfill().to_numpy()
                wB[msk] = np.clip(em / x, 0.5, 2.0)
            # C: per-name trailing win-rate weights (shrunk prior, causal)
            wC = np.ones(len(df))
            prior, n0 = be + 0.07, 30
            for tk in df["tk"].unique():
                msk = np.where((df["tk"] == tk).to_numpy())[0]
                wins = df["win"].to_numpy()[msk]
                cw = np.concatenate([[0], np.cumsum(wins)[:-1]])
                cn = np.arange(len(msk))
                wr = (n0 * prior + cw) / (n0 + cn)
                wC[msk] = np.clip((wr - be) / 0.07, 0.3, 1.5)
            print(f"=== {name} ({mins}m/{tp:g}:{sl:g})  [{uni_name}] ===", flush=True)
            print(f"  {'variant':>18} {'trades':>7} {'win%':>6} {'total%':>8} {'Sharpe':>7} {'maxDD%':>8}")
            for vn, rws, w in (("baseline", rows, None),
                               ("A regime filter", [r for r, ok in zip(rows, okA) if ok], None),
                               ("B vol sizing", rows, wB),
                               ("C name weighting", rows, wC)):
                n, wr_, tot, sh, dd = stats(rws, w)
                print(f"  {vn:>18} {n:>7} {wr_:>6.1%} {tot:>+8.0f} {sh:>7.2f} {dd:>+8.1f}", flush=True)
            print()


if __name__ == "__main__":
    main()

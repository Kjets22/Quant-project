"""
exp_ten.py — the 10:1 hunt, round 2: new stop designs + full stats (Sharpe, win%).

Cells (60-min bars, HBAR=96, base+S/R+trend features, corrected accounting, 5 bps):
  ref   : (7,1)  SEL_Q .93      — the surviving trend-hunter, for Sharpe/win% reporting
  A     : (20,2) SEL_Q .97      — 10:1 payoff, 2-ATR stop, EXTREME selectivity (top 3%)
  B     : (25,2.5) SEL_Q .93    — 10:1 payoff, 2.5-ATR stop (more room)
  C     : (30,3) SEL_Q .93      — 10:1 payoff, 3-ATR stop (max room)
  D     : STRUCTURE stop, 10:1  — stop just below the 20-bar swing low (support), target
                                  = entry + 10 x actual risk; risk sanity 0.2..4 ATR
All stops/labels causal (swing low uses shift(1)). Outputs trades, win%, mean bps, total,
and MONTHLY Sharpe per cell.   python exp_ten.py chunk1|chunk2 [fresh]
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

from triple_barrier_ml import features
from sr_features import sr_features
from wide_hunter import atr_fixed, trend_features

MINS, HBAR = 60, 96
EFF_COST = (3.0 + 2 * 1.0) / 1e4
MIN_ATR_PCT = 0.0012
DEV = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "JPM", "XLE", "TLT"]
FRESH = ["IWM", "GLD", "META", "XOM", "KO"]
_PREP = {}


def prep(tk):
    if tk in _PREP:
        return _PREP[tk]
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
    swing = (pd.Series(l).rolling(20).min().shift(1) - 0.25 * A).to_numpy()
    _PREP[tk] = (ts, h, l, c, A, X, swing)
    return _PREP[tk]


def stops_targets(c, A, swing, mode, tp_atr, sl_atr):
    """Per-bar (stop_px, tgt_px, valid). mode='atr' or 'struct' (10:1 on actual risk)."""
    if mode == "atr":
        stop = c - sl_atr * A
        tgt = c + tp_atr * A
        valid = np.isfinite(A)
    else:
        risk = c - swing
        valid = np.isfinite(risk) & (risk > 0.2 * A) & (risk < 4.0 * A)
        stop = swing
        tgt = c + 10.0 * risk
    return stop, tgt, valid


def make_label(h, l, c, stop, tgt, valid):
    n = len(c)
    y = np.full(n, np.nan)
    for i in range(n - 1):
        if not valid[i]:
            continue
        for j in range(i + 1, min(i + HBAR + 1, n)):
            if l[j] <= stop[i]:
                y[i] = 0; break
            if h[j] >= tgt[i]:
                y[i] = 1; break
    return y


def run_cell(names, mode, tp_atr, sl_atr, sel_q):
    rows = []                        # (ts, ret, win)
    for tk in names:
        ts, h, l, c, A, X, swing = prep(tk)
        stop, tgt, valid = stops_targets(c, A, swing, mode, tp_atr, sl_atr)
        y = make_label(h, l, c, stop, tgt, valid)
        m = (X.notna().all(axis=1) & np.isfinite(y) & np.isfinite(A) & valid).to_numpy()
        idx = np.where(m)[0]
        Xv, yv = X.iloc[idx].reset_index(drop=True), y[idx].astype(int)
        hv, lv, cv, Av = h[idx], l[idx], c[idx], A[idx]
        stv, tgv, tsv = stop[idx], tgt[idx], ts[idx]
        n = len(idx); K = 5
        bnds = np.linspace(int(n * 0.4), n, K + 1).astype(int)
        for k in range(K):
            tr_end = max(bnds[k] - HBAR, 300)
            if yv[:tr_end].sum() < 20:
                continue
            clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                     min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                                     reg_lambda=1.0, verbose=-1)
            clf.fit(Xv.iloc[:tr_end], yv[:tr_end])
            thr = np.quantile(clf.predict_proba(Xv.iloc[:tr_end])[:, 1], sel_q)
            proba = clf.predict_proba(Xv.iloc[bnds[k]:bnds[k + 1]])[:, 1]
            i = bnds[k]
            while i < bnds[k + 1] - 1:
                if proba[i - bnds[k]] < thr or Av[i] / cv[i] < MIN_ATR_PCT:
                    i += 1; continue
                res, j = None, i + 1
                while j < min(i + HBAR + 1, n):
                    if lv[j] <= stv[i]:
                        res = 0; break
                    if hv[j] >= tgv[i]:
                        res = 1; break
                    j += 1
                ex = min(j, n - 1)
                if res == 1:
                    r = (tgv[i] - cv[i]) / cv[i] - EFF_COST
                elif res == 0:
                    r = (stv[i] - cv[i]) / cv[i] - EFF_COST
                else:
                    r = (cv[ex] - cv[i]) / cv[i] - EFF_COST
                rows.append((tsv[i], r, 1 if res == 1 else 0))
                i = j + 1
    return rows


def report(tag, rows, fh):
    if not rows:
        line = f"  {tag:>28}: no trades"
    else:
        r = np.array([x[1] for x in rows])
        w = np.mean([x[2] for x in rows])
        s = pd.DataFrame(rows, columns=["ts", "r", "w"])
        s["ts"] = pd.to_datetime(s["ts"])
        mon = s.set_index("ts")["r"].resample("ME").sum()
        sharpe = mon.mean() / mon.std() * np.sqrt(12) if mon.std() > 0 else 0.0
        line = (f"  {tag:>28}: n={len(r):>5}  win%={w:>5.1%}  mean={r.mean()*1e4:>+6.1f}bps  "
                f"total={r.sum()*100:>+5.0f}%  Sharpe={sharpe:>5.2f}"
                + ("  <== POSITIVE" if r.sum() > 0 else ""))
    print(line, flush=True)
    fh.write(line + "\n"); fh.flush()


CELLS = {
    "chunk1": [("7:1 ref (trend-hunter)", "atr", 7, 1, 0.93),
               ("A 20:2 top-3%", "atr", 20, 2, 0.97),
               ("B 25:2.5", "atr", 25, 2.5, 0.93)],
    "chunk2": [("C 30:3", "atr", 30, 3, 0.93),
               ("D struct-stop 10:1", "struct", 0, 0, 0.93)],
}


def main():
    chunk = sys.argv[1] if len(sys.argv) > 1 else "chunk1"
    fresh = "fresh" in sys.argv[2:]
    names = FRESH if fresh else DEV
    out = Path(f"runs/exp_ten_{chunk}{'_fresh' if fresh else ''}.txt")
    out.parent.mkdir(exist_ok=True)
    print(f"10:1 HUNT r2 @ 60-min HBAR=96 [{'FRESH' if fresh else 'DEV'}]", flush=True)
    with out.open("w") as fh:
        for tag, mode, tp, sl, q in CELLS[chunk]:
            report(tag, run_cell(names, mode, tp, sl, q), fh)
    print("done", flush=True)


if __name__ == "__main__":
    main()

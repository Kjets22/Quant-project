"""
wide_hunter.py — mission: make 5:1..10:1 profitable, honestly (no future info).

Diagnosis from the corrected sweep: wide targets fail because (a) the 24-bar clock is too
short for a 5-10 ATR move, (b) the 1-ATR stop kills 85% of trades before a trend can move,
(c) the feature set hunts mean-reversion bounces, not trend legs. Fixes, all causal:
  * HBAR = 96 bars (60-min bars -> ~2.5 trading weeks of runway)
  * 4 TREND features: 100-bar momentum, distance to the 100-bar high (breakout proximity,
    shift(1)), 200-bar trend z-score, volatility expansion (ATR12/ATR96)
  * wider-stop variants at the same ratios: (10,2)=5:1, (14,2)=7:1, (20,2)=10:1
Audited env otherwise: embargoed training (HBAR bars), no-bfill ATR, 0.12% ATR floor,
5 bps effective cost, corrected accounting (time exit at actual close; win = target hit).

  python wide_hunter.py 5,6,7          # tight-stop chunk 1
  python wide_hunter.py 8,9,10         # tight-stop chunk 2
  python wide_hunter.py wide           # wider-stop variants (10:2, 14:2, 20:2)
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
from triple_barrier_breadth import TICKERS
from sr_features import sr_features

MINS = 60
SEL_Q, HBAR = 0.93, 96
EFF_COST = (3.0 + 2 * 1.0) / 1e4
MIN_ATR_PCT = 0.0012
_PREP = {}


def atr_fixed(h, l, c, n=24):
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    return pd.Series(tr).rolling(n).mean().to_numpy()


def trend_features(h, l, c, A):
    s, H = pd.Series(c), pd.Series(h)
    a = np.maximum(A, 1e-9)
    atr12 = pd.Series(np.maximum.reduce([h - l,
             np.abs(h - np.concatenate([[c[0]], c[:-1]])),
             np.abs(l - np.concatenate([[c[0]], c[:-1]]))])).rolling(12).mean().to_numpy()
    atr96 = pd.Series(np.maximum.reduce([h - l,
             np.abs(h - np.concatenate([[c[0]], c[:-1]])),
             np.abs(l - np.concatenate([[c[0]], c[:-1]]))])).rolling(96).mean().to_numpy()
    return pd.DataFrame({
        "t_mom100": (c - s.shift(100).to_numpy()) / (a * 10),
        "t_brk100": (c - H.rolling(100).max().shift(1).to_numpy()) / a,
        "t_sma200": (c - s.rolling(200).mean().to_numpy()) / (s.rolling(200).std().to_numpy() + 1e-9),
        "t_volexp": atr12 / (atr96 + 1e-12),
    })


def label_wide(h, l, c, A, tp, sl):
    n = len(c)
    y = np.full(n, np.nan)
    for i in range(n - 1):
        a = A[i]
        up, dn = c[i] + tp * a, c[i] - sl * a
        for j in range(i + 1, min(i + HBAR + 1, n)):
            if l[j] <= dn:
                y[i] = 0; break
            if h[j] >= up:
                y[i] = 1; break
    return y


def prep(tk):
    if tk in _PREP:
        return _PREP[tk]
    df = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv", parse_dates=["timestamp"])
    d = df.set_index("timestamp").resample(f"{MINS}min").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum")).dropna().reset_index()
    hrs = pd.to_datetime(d["timestamp"]).dt.hour.to_numpy()
    h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
    A = atr_fixed(h, l, c)
    X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                   sr_features(d).reset_index(drop=True),
                   trend_features(h, l, c, A).reset_index(drop=True)], axis=1)
    _PREP[tk] = (h, l, c, A, X, hrs)
    return _PREP[tk]


def _walk(proba, thr, hv, lv, cv, Av, hrs, ok_hours, i0, i1, n, tp, sl):
    """Trade walk; returns list of (hour, outcome, ret). ok_hours=None -> all hours."""
    res_list, i = [], i0
    while i < i1 - 1:
        if (proba[i - i0] < thr or Av[i] / cv[i] < MIN_ATR_PCT
                or (ok_hours is not None and hrs[i] not in ok_hours)):
            i += 1; continue
        a = Av[i]; up, dn = cv[i] + tp * a, cv[i] - sl * a
        res, j = None, i + 1
        while j < min(i + HBAR + 1, n):
            if lv[j] <= dn:
                res = 0; break
            if hv[j] >= up:
                res = 1; break
            j += 1
        ex = min(j, n - 1)
        if res == 1:
            r = tp * a / cv[i] - EFF_COST
        elif res == 0:
            r = -sl * a / cv[i] - EFF_COST
        else:
            r = (cv[ex] - cv[i]) / cv[i] - EFF_COST
        res_list.append((hrs[i], res, r))
        i = j + 1
    return res_list


def run_cell(tp, sl, use_hours=False):
    TT = 0
    n_tgt = n_stp = n_time = 0
    SR = 0.0
    be = sl / (sl + tp)
    for tk in TICKERS:
        h, l, c, A, X, hrs = prep(tk)
        y = label_wide(h, l, c, A, tp, sl)
        m = (X.notna().all(axis=1) & np.isfinite(y) & np.isfinite(A)).to_numpy()
        idx = np.where(m)[0]
        Xv, yv = X.iloc[idx].reset_index(drop=True), y[idx].astype(int)
        hv, lv, cv, Av, hrv = h[idx], l[idx], c[idx], A[idx], hrs[idx]
        n = len(idx); K = 5
        bnds = np.linspace(int(n * 0.4), n, K + 1).astype(int)
        for k in range(K):
            tr_end = max(bnds[k] - HBAR, 300)
            if yv[:tr_end].sum() < 20:              # too few positive labels to learn
                continue
            clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                     min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                                     reg_lambda=1.0, verbose=-1)
            clf.fit(Xv.iloc[:tr_end], yv[:tr_end])
            ptr = clf.predict_proba(Xv.iloc[:tr_end])[:, 1]
            thr = np.quantile(ptr, SEL_Q)
            good = None
            if use_hours:                            # hour selection on TRAIN block only
                stats = {}
                for hr, res, _ in _walk(ptr, thr, hv, lv, cv, Av, hrv, None, 0, tr_end, n, tp, sl):
                    a_, b_ = stats.get(hr, (0, 0))
                    stats[hr] = (a_ + 1, b_ + (1 if res == 1 else 0))
                good = {hr for hr, (nn, ww) in stats.items() if nn >= 25 and ww / nn >= be + 0.02}
                if len(good) < 2:
                    good = None                      # fallback: no reliable hours
            proba = clf.predict_proba(Xv.iloc[bnds[k]:bnds[k + 1]])[:, 1]
            for hr, res, r in _walk(proba, thr, hv, lv, cv, Av, hrv, good,
                                    bnds[k], bnds[k + 1], n, tp, sl):
                TT += 1; SR += r
                if res == 1:
                    n_tgt += 1
                elif res == 0:
                    n_stp += 1
                else:
                    n_time += 1
    return TT, n_tgt, n_stp, n_time, SR


def main():
    global TICKERS
    arg = sys.argv[1] if len(sys.argv) > 1 else "5,6,7"
    fresh = "fresh" in sys.argv[2:]
    use_hours = "hours" in sys.argv[2:]
    if fresh:
        TICKERS = ["IWM", "GLD", "META", "XOM", "KO"]
    if arg == "wide":
        cells = [(10.0, 2.0), (14.0, 2.0), (20.0, 2.0)]
    elif arg == "winners":
        cells = [(7.0, 1.0), (20.0, 2.0)]
    else:
        cells = [(float(x), 1.0) for x in arg.split(",")]
    out = Path(f"runs/wide_hunter_{arg.replace(',', '-').replace(':', '')}"
               f"{'_fresh' if fresh else ''}{'_hours' if use_hours else ''}.txt")
    out.parent.mkdir(exist_ok=True)
    hdr = (f"WIDE HUNTER @ {MINS}-min, HBAR={HBAR}, +trend features "
           f"(corrected accounting, 5 bps)\n"
           f"  {'tgt:stop':>9} {'ratio':>6} {'trades':>7} {'tgt%':>6} {'stop%':>6} {'time%':>6} "
           f"{'mean bps':>9} {'total%':>8}")
    print(hdr, flush=True)
    with out.open("w") as fh:
        fh.write(hdr + "\n")
        for tp, sl in cells:
            TT, ntg, nst, nti, SR = run_cell(tp, sl, use_hours=use_hours)
            if TT == 0:
                line = f"  {tp:g}:{sl:g}  no trades"
            else:
                flag = "  <== POSITIVE" if SR > 0 else ""
                line = (f"  {tp:g}:{sl:g}     {tp/sl:>5.1f} {TT:>7} {ntg/TT:>6.1%} {nst/TT:>6.1%} "
                        f"{nti/TT:>6.1%} {SR/TT*1e4:>+9.1f} {SR*100:>+8.0f}{flag}")
            print(line, flush=True); fh.write(line + "\n"); fh.flush()
    print("done", flush=True)


if __name__ == "__main__":
    main()

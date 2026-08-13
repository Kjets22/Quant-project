"""
vpt_boost.py — Sharpe-raising machinery on top of the vPT champion (hourly
DB+iHS long). Each idea is a switch so rounds stay attributable:

  --entry trigger|close   stop-order fill AT the neckline/peak trigger level
                          (a resting stop order would fill ~there; close-of-bar
                          gives away the move from trigger to close)
  --meta Q                LightGBM meta-filter over setups: features = pattern
                          quality (depth/ATR, symmetry, span, trigger gap),
                          context (vol, trend, hour) -> P(win); trained on OPT
                          trades only; keep top-Q fraction on VAL
  --maxconc N             cap simultaneous open positions across tickers

Protocol unchanged: choose on OPT (2023-01..2024-07), confirm on VAL
(2024-07..2025-07). TEST stays untouched for the final champion only.
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

import vpt_geometric as G
from triple_barrier_breadth import TICKERS

PATS = {"DB", "iHS"}
CFG = dict(tf=60, k=2.0, tol=0.005, H=48)
OPT = ("2023-01-01", "2024-07-01")
VAL = ("2024-07-01", "2025-07-01")
COST = G.COST


def detect_rich(sub, entry_mode="close"):
    """Setups + quality features. entry_mode:
      close       — confirmation-close entry (fill at bar close)
      fairtrigger — resting STOP order at the trigger from the confirmation
                    pivot onward: fills on the FIRST bar whose HIGH touches the
                    trigger (at max(trigger, bar open) — gap-aware), INCLUDING
                    failed breakouts that close back below. No close filter.
    (The naive 'trigger' fill-at-crossing-but-require-close variant was a
    lookahead — it excluded exactly the failed breaks a real stop order eats.)"""
    h = sub["high"].to_numpy(float)
    l = sub["low"].to_numpy(float)
    c = sub["close"].to_numpy(float)
    o_ = sub["open"].to_numpy(float) if "open" in sub else c
    A = G.atr(h, l, c)
    piv = G.zigzag(h, l, c, A, CFG["k"])
    n = len(c)
    out = []

    def add(i_conf, trigger, stop, tgt, name, quality):
        for j in range(i_conf, min(i_conf + 30, n)):
            if entry_mode == "fairtrigger":
                if h[j] >= trigger:
                    fill = max(trigger, o_[j]) * (1 + 2e-4)   # 2bp stop slippage
                    out.append(dict(i=j, side="long", stop=stop, tgt=tgt,
                                    name=name, trigger=trigger, fill=fill,
                                    **quality))
                    return
            elif c[j] > trigger:
                out.append(dict(i=j, side="long", stop=stop, tgt=tgt, name=name,
                                trigger=trigger, fill=None, **quality))
                return

    for a in range(len(piv) - 4):
        p = piv[a:a + 5]
        px = [x[1] for x in p]
        typ = [x[2] for x in p]
        conf = p[-1][3]
        idx = [x[0] for x in p]
        if idx[-1] - idx[0] > 120:
            continue
        if typ == [-1, 1, -1, 1, -1]:                       # iHS
            ls, h1, hd, h2, rs = px
            if hd < ls and hd < rs and abs(ls - rs) / rs <= CFG["tol"] * 2 \
                    and max(ls, rs) < min(h1, h2):
                neck = (h1 + h2) / 2
                i0 = idx[2]
                q = dict(depth=(neck - hd) / max(A[i0], 1e-9),
                         sym=abs(ls - rs) / max(neck - hd, 1e-9),
                         span=idx[-1] - idx[0],
                         necktilt=abs(h1 - h2) / max(neck - hd, 1e-9))
                add(conf, neck, min(rs, hd * 1.001), neck + (neck - hd), "iHS", q)
    for a in range(len(piv) - 2):
        p = piv[a:a + 3]
        px = [x[1] for x in p]
        typ = [x[2] for x in p]
        conf = p[-1][3]
        idx = [x[0] for x in p]
        if typ == [-1, 1, -1] and abs(px[0] - px[2]) / px[2] <= CFG["tol"]:
            i0 = idx[1]
            q = dict(depth=(px[1] - min(px[0], px[2])) / max(A[i0], 1e-9),
                     sym=abs(px[0] - px[2]) / max(px[1] - min(px[0], px[2]), 1e-9),
                     span=idx[-1] - idx[0], necktilt=0.0)
            add(conf, px[1], min(px[0], px[2]), px[1] + (px[1] - px[0]), "DB", q)
    out.sort(key=lambda t: t["i"])
    return out


def sim_rich(sub, setups, entry_mode="close", tk=""):
    """Simulate with quality features attached to every trade row."""
    h = sub["high"].to_numpy(float)
    l = sub["low"].to_numpy(float)
    c = sub["close"].to_numpy(float)
    v = sub["volume"].to_numpy(float)
    ts = pd.to_datetime(sub["timestamp"]).to_numpy()
    ret1 = pd.Series(c).pct_change().to_numpy()
    vol96 = pd.Series(ret1).rolling(96).std().to_numpy()
    ma200 = pd.Series(c).rolling(200).mean().to_numpy()
    n = len(c)
    rows = []
    last_exit = -1
    for t in setups:
        i = t["i"]
        if i <= last_exit or i >= n - 1:
            continue
        e = t["fill"] if t.get("fill") else c[i]
        stop, tgt = t["stop"], t["tgt"]
        if not (stop < e < tgt):
            continue
        # fill bar itself can hit stop/target AFTER the entry touch (conservative:
        # stop first) — a real stop-entry lives through the rest of bar i
        res = None
        if t.get("fill"):
            if l[i] <= stop:
                res = (stop - e) / e
            elif h[i] >= tgt and c[i] >= tgt * 0.999:
                res = (tgt - e) / e
        j = i + 1
        while res is None and j < min(i + CFG["H"] + 1, n):
            if l[j] <= stop:
                res = (stop - e) / e; break
            if h[j] >= tgt:
                res = (tgt - e) / e; break
            j += 1
        ex = min(j, n - 1)
        if res is None:
            res = (c[ex] - e) / e
        et = pd.Timestamp(ts[i]).tz_localize("UTC").tz_convert("America/New_York")
        rows.append(dict(
            t=ts[i], tk=tk, ret=res - COST, win=int(res > 0),
            name=t["name"], depth=t["depth"], sym=t["sym"], span=t["span"],
            necktilt=t["necktilt"],
            gap_to_trig=(c[i] - t["trigger"]) / max(c[i], 1e-9),
            rr=(tgt - e) / max(e - stop, 1e-9),
            vol=vol96[i] if np.isfinite(vol96[i]) else 0,
            trend=(c[i] / ma200[i] - 1) if np.isfinite(ma200[i]) else 0,
            hour=et.hour, i=i, ex=ex))
        last_exit = ex
    return pd.DataFrame(rows)


def collect(lo, hi, entry_mode):
    frames = []
    for tk in TICKERS:
        df = G.bars(tk, CFG["tf"])
        sub = df[(df["timestamp"] >= lo) & (df["timestamp"] < hi)] \
            .reset_index(drop=True)
        setups = detect_rich(sub, entry_mode)
        d = sim_rich(sub, setups, entry_mode, tk)
        if len(d):
            frames.append(d)
    return pd.concat(frames, ignore_index=True).sort_values("t")


def meta_oof(o, v, q_grid=(0.3, 0.4, 0.5, 0.6, 0.7), size_by_score=False):
    """Out-of-fold meta-filter: 5-fold time-ordered CV on OPT gives every OPT
    trade an honest OOF score; Q chosen by OOF-OPT Sharpe; VAL touched once."""
    from sklearn.model_selection import KFold
    po = np.full(len(o), np.nan)
    idx = np.arange(len(o))
    folds = np.array_split(idx, 5)
    for f in range(5):
        te = folds[f]
        tr = np.concatenate([folds[g] for g in range(5) if g != f])
        clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                 num_leaves=7, min_child_samples=30,
                                 subsample=0.8, colsample_bytree=0.8,
                                 reg_lambda=2.0, verbose=-1)
        clf.fit(o.iloc[tr][META_F], o.iloc[tr]["win"])
        po[te] = clf.predict_proba(o.iloc[te][META_F])[:, 1]
    best_q, best_s = None, -9
    for q in q_grid:
        thr = np.nanquantile(po, 1 - q)
        s, _, nn = sharpe(o[po >= thr])
        if nn >= 150 and s > best_s:
            best_q, best_s = q, s
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=7,
                             min_child_samples=30, subsample=0.8,
                             colsample_bytree=0.8, reg_lambda=2.0, verbose=-1)
    clf.fit(o[META_F], o["win"])
    thr = np.nanquantile(po, 1 - best_q)
    pv = clf.predict_proba(v[META_F])[:, 1]
    o2 = o[po >= thr].copy()
    v2 = v[pv >= thr].copy()
    if size_by_score:
        # conviction sizing: weight in [0.5, 2.0] proportional to score above thr
        def w(scores):
            s = np.asarray(scores, float)
            rng_ = max(np.nanmax(po) - thr, 1e-9)
            return np.clip(0.5 + 1.5 * (s - thr) / rng_, 0.5, 2.0)
        o2["ret"] = o2["ret"] * w(po[po >= thr])
        v2["ret"] = v2["ret"] * w(pv[pv >= thr])
    return o2, v2, best_q, best_s


META_F = ["depth", "sym", "span", "necktilt", "gap_to_trig", "rr", "vol",
          "trend", "hour"]


def sharpe(df, maxconc=None):
    d = df
    if maxconc is not None and len(d):
        d = d.sort_values("t")
        open_ex = []
        keepmask = []
        for _, row in d.iterrows():
            open_ex = [x for x in open_ex if x > row["i"]]
            if len(open_ex) >= maxconc:
                keepmask.append(False)
            else:
                keepmask.append(True)
                open_ex.append(row["ex"])
        d = d[pd.Series(keepmask, index=d.index)]
    if not len(d):
        return 0.0, 0.0, 0
    s = pd.Series(d["ret"].to_numpy(), index=pd.to_datetime(d["t"]))
    dd = s.resample("D").sum()
    dd = dd[dd.index.dayofweek < 5]
    span = pd.date_range(dd.index.min(), dd.index.max(), freq="B")
    dd = dd.reindex(span, fill_value=0.0)
    shp = float(dd.mean() / dd.std() * np.sqrt(252)) if dd.std() else 0
    return shp, float(d["ret"].sum() * 100), len(d)


def main():
    entry = "close"
    if "--entry" in sys.argv:
        entry = sys.argv[sys.argv.index("--entry") + 1]
    use_metacv = "--metacv" in sys.argv
    maxc = int(sys.argv[sys.argv.index("--maxconc") + 1]) \
        if "--maxconc" in sys.argv else None
    o = collect(*OPT, entry)
    v = collect(*VAL, entry)
    tag = f"entry={entry} metacv={use_metacv} maxconc={maxc}"
    size_by_score = "--sizescore" in sys.argv
    boot = "--boot" in sys.argv
    if use_metacv:
        o, v, q, s_oof = meta_oof(o.reset_index(drop=True),
                                  v.reset_index(drop=True),
                                  size_by_score=size_by_score)
        tag += f" (Q={q} on OOF-OPT shp {s_oof:.2f}" + \
            (", score-sized" if size_by_score else "") + ")"
    so, to, no = sharpe(o, maxc)
    sv, tv, nv = sharpe(v, maxc)
    line = (f"{tag}: OPT shp {so:5.2f} tot {to:+7.1f}% n={no} | "
            f"VAL shp {sv:5.2f} tot {tv:+7.1f}% n={nv}")
    if boot and len(v):
        rng = np.random.default_rng(0)
        vs = []
        d = v.reset_index(drop=True)
        days = pd.to_datetime(d["t"]).dt.date.to_numpy()
        udays = np.unique(days)
        for _ in range(1000):
            pick = rng.choice(udays, len(udays), replace=True)
            idx = np.concatenate([np.where(days == p)[0] for p in pick])
            s_, _, _ = sharpe(d.iloc[idx])
            vs.append(s_)
        lo_, hi_ = np.percentile(vs, [5, 95])
        line += f"  [VAL shp 90% CI {lo_:.2f}..{hi_:.2f}]"
    print(line)


if __name__ == "__main__":
    main()

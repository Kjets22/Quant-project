"""
vob2_model.py — vOB2: the order-book PRESSURE trader (iteration 2, literature-
grounded). Trades multi-minute book-pressure persistence, not 1-min noise.

Features per minute (all trailing/causal):
  qimb / micro_dev / ofi aggregated over 5, 15, 30, 60-min windows
  ofi_run   consecutive-minute streak of same-sign OFI
  press     qimb15 * micro15 interaction
  spread / quote-rate context, minute-of-day
Model: LGBM (small) AND ridge-style linear control, predicting fwd mid return
over H in {5, 15, 30} minutes. Trade when |pred| > k * cost; direction = sign;
exit after H minutes (mid-to-mid) minus costs. Long AND short (mid-based).

Costs: QQQ taker round trip = spread (0.4bp) + 0.6bp slippage = 1bp base;
2.5bp stress also reported. Walk-forward: train expanding by day, test next
day-block (5 days). No same-day leakage.

  python vob2_model.py [--H 15] [--k 1.5]
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

OFI = Path("data_cache/vob_ofi")
COST_BASE = 1.0e-4
COST_STRESS = 2.5e-4


def arg(flag, default):
    if flag in sys.argv:
        return float(sys.argv[sys.argv.index(flag) + 1])
    return default


def load(tk="QQQ"):
    fs = sorted(OFI.glob(f"{tk}_*.parquet"))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    d = d.sort_values("timestamp").reset_index(drop=True)
    d["day"] = d["timestamp"].dt.date
    return d


def features(d):
    g = d.copy()
    for w in (5, 15, 30, 60):
        g[f"qimb{w}"] = g["qimb"].rolling(w).mean()
        g[f"micro{w}"] = g["micro_dev"].rolling(w).mean()
        g[f"ofi{w}"] = g["ofi"].rolling(w).sum()
    s = np.sign(g["ofi"])
    g["ofi_run"] = s.groupby((s != s.shift()).cumsum()).cumcount() + 1
    g["ofi_run"] *= s
    g["press"] = g["qimb15"] * g["micro15"]
    g["ret5"] = g["mid_c"].pct_change(5)
    g["ret30"] = g["mid_c"].pct_change(30)
    g["vol30"] = g["mid_c"].pct_change().rolling(30).std()
    et = g["timestamp"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    g["mod"] = et.dt.hour * 60 + et.dt.minute
    # day boundary: kill windows that span days
    newday = g["day"] != g["day"].shift()
    bad = newday.rolling(60, min_periods=1).max().astype(bool)
    feats = [c for c in g.columns if c.startswith(("qimb", "micro", "ofi",
                                                  "press", "ret", "vol",
                                                  "spread", "mod", "nq"))
             and c not in ("ofi",)]
    g.loc[bad, [f for f in feats if f != "mod"]] = np.nan
    return g, feats


def run(H, k):
    d = load()
    days = sorted(d["day"].unique())
    print(f"vOB2: {len(days)} days, {len(d)} minutes | H={H}m k={k}\n")
    g, feats = features(d)
    g["y"] = g["mid_c"].shift(-H) / g["mid_c"] - 1
    # same-day only: label must not span days
    g.loc[g["day"] != pd.Series(g["day"]).shift(-H), "y"] = np.nan
    fin = g[feats].notna().all(axis=1) & g["y"].notna()
    blocks = [days[i:i + 5] for i in range(20, len(days), 5)]
    all_tr = []
    for blk in blocks:
        tr = fin & (g["day"] < blk[0])
        te = fin & g["day"].isin(blk)
        if tr.sum() < 3000 or te.sum() < 200:
            continue
        m = lgb.LGBMRegressor(n_estimators=250, learning_rate=0.04,
                              num_leaves=15, min_child_samples=60,
                              subsample=0.8, colsample_bytree=0.8,
                              reg_lambda=2.0, verbose=-1)
        m.fit(g.loc[tr, feats], g.loc[tr, "y"])
        pred = m.predict(g.loc[te, feats])
        sub = g.loc[te, ["timestamp", "day", "y"]].copy()
        sub["pred"] = pred
        thr = k * COST_BASE
        sub = sub[np.abs(sub["pred"]) > thr]
        # non-overlap: one position at a time per day
        keep, last_t = [], None
        for _, r0 in sub.iterrows():
            if last_t is not None and r0["timestamp"] < last_t:
                continue
            keep.append(r0)
            last_t = r0["timestamp"] + pd.Timedelta(minutes=H)
        for r0 in keep:
            side = np.sign(r0["pred"])
            all_tr.append(dict(t=r0["timestamp"],
                               ret=side * r0["y"] - COST_BASE,
                               ret_stress=side * r0["y"] - COST_STRESS))
    if not all_tr:
        print("no trades above threshold")
        return
    df = pd.DataFrame(all_tr)
    for col, label in (("ret", f"cost {COST_BASE * 1e4:.1f}bp"),
                       ("ret_stress", f"cost {COST_STRESS * 1e4:.1f}bp")):
        s = pd.Series(df[col].to_numpy(), index=pd.to_datetime(df["t"]))
        dd = s.resample("D").sum()
        dd = dd[dd.index.dayofweek < 5]
        span = pd.date_range(dd.index.min(), dd.index.max(), freq="B")
        dd = dd.reindex(span, fill_value=0.0)
        shp = float(dd.mean() / dd.std() * np.sqrt(252)) if dd.std() else 0
        print(f"  {label}: n={len(df)} win={(df[col] > 0).mean():.0%} "
              f"avg={df[col].mean() * 1e4:+.2f}bp total={df[col].sum() * 100:+.2f}% "
              f"daily-Sharpe {shp:+.2f}")


if __name__ == "__main__":
    run(int(arg("--H", 15)), arg("--k", 1.5))

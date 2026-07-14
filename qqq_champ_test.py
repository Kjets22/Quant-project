"""
qqq_champ_test.py — the tournament champion (F4 features / shallow LGBM) as a TRADING
strategy on the PAST MONTH. Confidence-gated: only act when |p-0.5| clears the 80th
percentile of the TRAINING confidence distribution (the tier that scored 68.3%).

Trade: symmetric $2 target / $2 stop, 60-min clock, non-overlapping, corrected accounting,
5 bps effective cost. Evaluates LONG side (p high), SHORT side (p low) and combined —
the live bot is long-only, so the LONG rows decide integration.
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

from qqq_tournament import load, MODELS

H, TGT = 12, 2.0
EFF_COST = 5.0 / 1e4
MONTH0 = "2026-06-14"
CONF_Q = 0.80


def main():
    ts, feats, y = load()
    X = feats["F4"]
    d_h = None
    # need raw h/l/c for trade sim — reload minimal arrays
    from qqq_tournament import Path as _P  # noqa
    import pandas as _pd
    base = _pd.read_csv("data_cache/QQQ_5minute_2021-06-01_2026-06-01.csv", parse_dates=["timestamp"])
    rec = sorted(__import__("pathlib").Path("data_cache").glob("QQQ_recent_2026-06-01_*.csv"))
    parts = [base] + ([_pd.read_csv(rec[-1], parse_dates=["timestamp"])] if rec else [])
    dd = (_pd.concat(parts, ignore_index=True)
            .drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp").reset_index(drop=True))
    h, l, c = (dd[x].to_numpy(float) for x in ("high", "low", "close"))

    fin = X.notna().all(axis=1).to_numpy()
    n = len(c)
    tr = np.where(fin & np.isfinite(y) & (ts < np.datetime64(MONTH0)))[0]
    tr = tr[:-H]
    clf = MODELS["lgbm_s"]()
    clf.fit(X.iloc[tr], y[tr].astype(int))
    ptr = clf.predict_proba(X.iloc[tr])[:, 1]
    conf_thr = np.quantile(np.abs(ptr - 0.5), CONF_Q)

    fwd = np.where(fin & (ts >= np.datetime64(MONTH0)))[0]
    proba = clf.predict_proba(X.iloc[fwd])[:, 1]
    pmap = {int(ix): float(p) for ix, p in zip(fwd, proba)}

    trades = []
    i, last = int(fwd[0]), int(fwd[-1])
    while i <= last:
        p = pmap.get(i)
        if p is None or abs(p - 0.5) < conf_thr:
            i += 1; continue
        side = 1 if p > 0.5 else -1
        up, dn = c[i] + TGT, c[i] - TGT
        res, j = None, i + 1
        while j < min(i + H + 1, n):
            hit_up, hit_dn = h[j] >= up, l[j] <= dn
            if hit_up and hit_dn:
                res = -side; break                      # ambiguous bar -> against us
            if hit_dn:
                res = -1; break
            if hit_up:
                res = 1; break
            j += 1
        ex = min(j, n - 1)
        if res is None:                                  # clock exit at close
            move = (c[ex] - c[i]) * side
            r = move / c[i] - EFF_COST
            win = move > 0
        else:
            win = (res == side)
            r = (TGT if win else -TGT) * 1.0 / c[i] - EFF_COST
        trades.append((pd.Timestamp(ts[i]), side, r, win))
        i = j + 1

    df = pd.DataFrame(trades, columns=["ts", "side", "r", "win"])
    print(f"CHAMPION as a strategy — past month ({MONTH0}..{pd.Timestamp(ts[last]).date()}), "
          f"conf gate |p-0.5| >= {conf_thr:.3f} (train q{int(CONF_Q*100)})")
    for name, sub in (("LONG (p>0.5)", df[df.side == 1]), ("SHORT (p<0.5)", df[df.side == -1]),
                      ("COMBINED", df)):
        if len(sub) == 0:
            print(f"  {name:>14}: no trades"); continue
        print(f"  {name:>14}: trades={len(sub):>3}  win%={sub.win.mean():.1%}  "
              f"mean={sub.r.mean()*1e4:+.1f}bps  total={sub.r.sum()*100:+.2f}%")
    daily = df.set_index("ts")["r"].resample("1D").sum()
    daily = daily[daily != 0]
    print(f"  trading days: {len(daily)}  best day {daily.max()*100:+.2f}%  "
          f"worst {daily.min()*100:+.2f}%  positive days {(daily>0).mean():.0%}")


if __name__ == "__main__":
    main()

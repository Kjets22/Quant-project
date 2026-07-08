"""extract_nvda_trade.py — find a CLEAN NVDA v3 (30-min/1.5:1) trade that really hit target."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from triple_barrier_ml import atr

cache = pd.read_csv("data_cache/NVDA_5minute_2021-06-01_2026-06-01.csv", parse_dates=["timestamp"])
recp = Path("data_cache/NVDA_recent_2026-06-01_2026-06-30.csv")
rec = pd.read_csv(recp, parse_dates=["timestamp"]) if recp.exists() else cache.iloc[0:0]
df = (pd.concat([cache, rec], ignore_index=True)
        .drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp"))
d = df.set_index("timestamp").resample("30min").agg(
    open=("open", "first"), high=("high", "max"), low=("low", "min"),
    close=("close", "last"), volume=("volume", "sum")).dropna().reset_index()
ts = pd.to_datetime(d["timestamp"]).to_numpy()
o, h, l, c = (d[x].to_numpy(float) for x in ("open", "high", "low", "close"))
A = atr(h, l, c)
TP, SL = 1.5, 1.0

win = (ts >= np.datetime64("2026-06-22")) & (ts < np.datetime64("2026-06-27"))
best = None
for i in np.where(win)[0]:
    a = A[i]
    if not np.isfinite(a) or a <= 0:
        continue
    tgt, stop = c[i] + TP * a, c[i] - SL * a
    j = i + 1
    hit = None
    while j < min(i + 13, len(c)):
        if l[j] <= stop:
            hit = "STOP"; break
        if h[j] >= tgt:
            hit = "TARGET"; break
        j += 1
    if hit == "TARGET" and 2 <= (j - i) <= 7:           # clean, took a few bars
        best = (i, j, a)
        break

i, xj, a = best
lo, hi = i - 7, xj + 4
cs = [[str(pd.Timestamp(ts[k]))[5:16], round(o[k], 2), round(h[k], 2), round(l[k], 2), round(c[k], 2)]
      for k in range(lo, min(hi, len(c)))]
shares = round(3000 / c[i], 1)
out = {"candles": cs, "entry_idx": int(i - lo), "exit_idx": int(xj - lo),
       "entry": round(float(c[i]), 2), "target": round(float(c[i] + TP * a), 2), "stop": round(float(c[i] - SL * a), 2),
       "shares": float(shares), "bought": int(round(shares * c[i])), "risk": round(float(shares * a), 2),
       "pnl": round(float(shares * TP * a), 2),
       "entry_ts": str(pd.Timestamp(ts[i]))[5:16], "exit_ts": str(pd.Timestamp(ts[xj]))[5:16]}
Path("runs").mkdir(exist_ok=True)
Path("runs/nvda_trade.json").write_text(json.dumps(out))
print(json.dumps(out, indent=1))

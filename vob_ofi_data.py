"""
vob_ofi_data.py — TRUE order-flow imbalance from full NBBO quote streams.

Per day: paginate the COMPLETE SIP quote tape for a symbol, then reduce to
per-MINUTE microstructure aggregates:

  ofi        Cont-Kukanov-Stoikov L1 OFI: sum of signed best-queue changes
             (+dBidSize when bid holds/improves, -dAskSize when ask holds/eases)
  qimb       time-weighted queue imbalance (bs-as)/(bs+as)
  micro_dev  mean (microprice - mid)/mid, bps
  spread_bps mean spread in bps
  nq         quote updates in the minute
  mid_o/mid_c first/last mid (for label construction downstream)

Cache: data_cache/vob_ofi/{TK}_{YYYY-MM-DD}.parquet (one file per day, resumable).

  python vob_ofi_data.py QQQ 2026-05-01 2026-08-13
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import requests

import alpaca_api as broker

OUT = Path("data_cache/vob_ofi")
OUT.mkdir(parents=True, exist_ok=True)


def fetch_day(tk, day):
    """Full quote tape for one day (RTH+ext, 13:00-21:00 UTC), paginated."""
    qs, token = [], None
    start = f"{day}T13:00:00Z"
    end = f"{day}T21:00:00Z"
    while True:
        p = {"start": start, "end": end, "limit": 10000, "feed": "sip"}
        if token:
            p["page_token"] = token
        for attempt in range(7):
            r = requests.get(
                f"https://data.alpaca.markets/v2/stocks/{tk}/quotes",
                headers=broker.HDRS, params=p, timeout=45)
            if r.status_code == 429:
                time.sleep(12 * (attempt + 1))
                continue
            break
        if r.status_code != 200:
            return None
        j = r.json()
        qs.extend(j.get("quotes") or [])       # API sends null on empty pages
        token = j.get("next_page_token")
        if not token:
            return qs


def reduce_day(qs):
    if not qs or len(qs) < 500:
        return None
    t = pd.to_datetime([q["t"] for q in qs]).tz_convert(None)
    bp = np.array([q["bp"] for q in qs], float)
    ap = np.array([q["ap"] for q in qs], float)
    bs = np.array([q["bs"] for q in qs], float)
    as_ = np.array([q["as"] for q in qs], float)
    ok = (bp > 0) & (ap > bp)
    t, bp, ap, bs, as_ = t[ok], bp[ok], ap[ok], bs[ok], as_[ok]
    # Cont-Kukanov-Stoikov L1 OFI on consecutive quote deltas
    e = np.zeros(len(bp))
    db = np.diff(bp, prepend=bp[0])
    da = np.diff(ap, prepend=ap[0])
    dbs = np.diff(bs, prepend=bs[0])
    das = np.diff(as_, prepend=as_[0])
    # bid side contribution
    e += np.where(db > 0, bs, np.where(db < 0, -np.roll(bs, 1), dbs))
    # ask side contribution (sign flipped)
    e -= np.where(da < 0, as_, np.where(da > 0, -np.roll(as_, 1), das))
    e[0] = 0
    mid = (bp + ap) / 2
    micro = (bp * as_ + ap * bs) / np.maximum(bs + as_, 1e-9)
    df = pd.DataFrame({
        "t": t, "ofi": e,
        "qimb": (bs - as_) / np.maximum(bs + as_, 1e-9),
        "micro_dev": (micro - mid) / mid * 1e4,
        "spread_bps": (ap - bp) / mid * 1e4,
        "mid": mid,
    }).set_index("t")
    g = df.resample("1min").agg(
        ofi=("ofi", "sum"), qimb=("qimb", "mean"),
        micro_dev=("micro_dev", "mean"), spread_bps=("spread_bps", "mean"),
        nq=("ofi", "size"), mid_o=("mid", "first"), mid_c=("mid", "last"),
        mid_h=("mid", "max"), mid_l=("mid", "min")).dropna()
    return g.reset_index().rename(columns={"t": "timestamp"})


def main():
    tk = sys.argv[1] if len(sys.argv) > 1 else "QQQ"
    start = sys.argv[2] if len(sys.argv) > 2 else "2026-05-01"
    end = sys.argv[3] if len(sys.argv) > 3 else "2026-08-13"
    days = [d.date() for d in pd.date_range(start, end, freq="D")
            if d.dayofweek < 5]
    print(f"OFI backfill {tk}: {len(days)} days", flush=True)
    for day in days:
        f = OUT / f"{tk}_{day}.parquet"
        if f.exists():
            continue
        qs = fetch_day(tk, day)
        if qs is None:
            print(f"  {day}: fetch failed", flush=True)
            continue
        g = reduce_day(qs)
        if g is None or not len(g):
            print(f"  {day}: no usable quotes ({len(qs) if qs else 0})",
                  flush=True)
            continue
        g.to_parquet(f)
        print(f"  {day}: {len(qs):,} quotes -> {len(g)} minutes", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()

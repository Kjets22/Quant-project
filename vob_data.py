"""
vob_data.py — order-book (NBBO) feature backfill for the vOB trader.

For every 5-min bar we pull the last ~30 NBBO quotes before the bar's close
(one desc call per bar, SIP feed, 16-min-delay safe) and reduce them to
book-state features:

  spread_bps    (ask-bid)/mid at bar end
  imb           (bid_size-ask_size)/(bid_size+ask_size), size-weighted over window
  micro_dev     (microprice-mid)/mid — which side is pressing
  quote_hz      quote updates/sec in the window (activity)
  bs, as_       end-of-bar top-of-book sizes

Cache: data_cache/vob/{TK}_{YYYY-MM}.parquet, resumable per bar. ~200 req/min cap.

  python vob_data.py QQQ 2026-02-01            backfill from date to now
  python vob_data.py QQQ 2026-02-01 2026-05-01 explicit window
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

OUT = Path("data_cache/vob")
OUT.mkdir(parents=True, exist_ok=True)
PACE = 0.35                       # ~170 req/min, under the 200 cap


def bar_grid(day):
    """5-min bar END times for one day, 08:05..23:55 UTC (ext + RTH)."""
    t0 = pd.Timestamp(f"{day} 08:05:00")
    return pd.date_range(t0, f"{day} 23:55:00", freq="5min")


def fetch_bar(tk, end_ts):
    p = {"start": (end_ts - pd.Timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
         "end": end_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
         "limit": 30, "feed": "sip", "sort": "desc"}
    for attempt in range(4):
        try:
            r = requests.get("https://data.alpaca.markets/v2/stocks/" + tk + "/quotes",
                             headers=broker.HDRS, params=p, timeout=25)
            if r.status_code == 429:
                time.sleep(20)
                continue
            if r.status_code != 200:
                return None
            return r.json().get("quotes", [])
        except requests.exceptions.RequestException:
            time.sleep(5 * (attempt + 1))
    return None


def reduce(qs, end_ts):
    if not qs:
        return None
    bp = np.array([q["bp"] for q in qs], float)
    ap = np.array([q["ap"] for q in qs], float)
    bs = np.array([q["bs"] for q in qs], float)
    as_ = np.array([q["as"] for q in qs], float)
    ok = (bp > 0) & (ap > bp)
    if not ok.any():
        return None
    bp, ap, bs, as_ = bp[ok], ap[ok], bs[ok], as_[ok]
    mid = (bp + ap) / 2
    micro = (bp * as_ + ap * bs) / np.maximum(bs + as_, 1e-9)
    t0 = pd.Timestamp(qs[-1]["t"]).tz_convert(None)
    t1 = pd.Timestamp(qs[0]["t"]).tz_convert(None)
    span = max((t1 - t0).total_seconds(), 0.2)
    tot = bs.sum() + as_.sum()
    return dict(
        timestamp=end_ts - pd.Timedelta(minutes=5),   # label with bar START (house style)
        spread_bps=float(((ap - bp) / mid).mean() * 1e4),
        imb=float((bs.sum() - as_.sum()) / tot) if tot else 0.0,
        micro_dev=float(((micro - mid) / mid).mean() * 1e4),
        quote_hz=float(len(bp) / span),
        bs=float(bs[0]), as_=float(as_[0]),
        bid=float(bp[0]), ask=float(ap[0]),
    )


def main():
    from concurrent.futures import ThreadPoolExecutor

    tk = sys.argv[1] if len(sys.argv) > 1 else "QQQ"
    start = sys.argv[2] if len(sys.argv) > 2 else "2026-02-01"
    end = sys.argv[3] if len(sys.argv) > 3 else str(pd.Timestamp.utcnow().date())
    days = [d for d in pd.date_range(start, end, freq="D") if d.dayofweek < 5]
    days = sorted(days, reverse=True)      # recent months first -> model can start
    print(f"vOB backfill {tk} {start}..{end}: {len(days)} weekdays "
          f"(threaded, newest first)", flush=True)
    for day in days:
        mo = f"{day:%Y-%m}"
        f = OUT / f"{tk}_{mo}.parquet"
        have = set()
        if f.exists():
            have = set(pd.read_parquet(f)["timestamp"].astype(str))
        todo = [e for e in bar_grid(day.date())
                if str(e - pd.Timedelta(minutes=5)) not in have
                and e <= pd.Timestamp.utcnow().tz_localize(None)
                - pd.Timedelta(minutes=17)]
        if not todo:
            continue
        rows = []
        with ThreadPoolExecutor(8) as ex:
            for qs, end_ts in zip(ex.map(lambda e: fetch_bar(tk, e), todo), todo):
                if qs is None:
                    continue
                r = reduce(qs, end_ts)
                if r:
                    rows.append(r)
        if rows:
            new = pd.DataFrame(rows)
            if f.exists():
                old = pd.read_parquet(f)
                new = (pd.concat([old, new], ignore_index=True)
                       .drop_duplicates(subset="timestamp").sort_values("timestamp"))
            new.to_parquet(f)
            print(f"  {day.date()}: +{len(rows)} bars -> {f.name} "
                  f"({len(new)} total)", flush=True)
    print("backfill done", flush=True)


if __name__ == "__main__":
    main()

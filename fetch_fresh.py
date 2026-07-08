"""
fetch_fresh.py — download the 5 fresh holdout tickers (never used in development)
so validate_30min.py can run the decisive out-of-sample test. Saves under the
canonical cache name so everything downstream reads them unchanged.
"""

from __future__ import annotations

import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from pathlib import Path

from basket import ticker_cfg
from data import fetch_polygon

FRESH = ["IWM", "GLD", "META", "XOM", "KO"]
START, END = "2021-06-01", "2026-06-01"


def main():
    for tk in FRESH:
        out = Path(f"data_cache/{tk}_5minute_{START}_{END}.csv")
        if out.exists():
            print(f"  {tk}: already cached ({out.name})", flush=True)
            continue
        cfg = ticker_cfg(tk)
        cfg.data.start_date, cfg.data.end_date = START, END
        cfg.data.multiplier, cfg.data.timespan = 5, "minute"
        for attempt in range(4):
            try:
                df = fetch_polygon(cfg)
                df.to_csv(out, index=False)
                print(f"  {tk}: saved {len(df)} rows  "
                      f"[{df['timestamp'].iloc[0]} .. {df['timestamp'].iloc[-1]}]", flush=True)
                break
            except Exception as e:
                print(f"  {tk}: attempt {attempt+1} failed ({e}); retrying in 10s", flush=True)
                time.sleep(10)
        else:
            print(f"  {tk}: GAVE UP after retries", flush=True)
    print("fetch_fresh done", flush=True)


if __name__ == "__main__":
    main()

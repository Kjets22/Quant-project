"""
fetch_recent_tlt.py — refresh TLT 5-min bars (2026-06-01..2026-07-22) so morning_ml.py's
cross-asset tlt_pm feature covers the tail of the FINAL window. Same fetch pattern as
fetch_fresh.py (basket.ticker_cfg + data.fetch_polygon), saved under the *_recent_*
cache name that morning_qqq3.load_aux already globs. Research only.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from basket import ticker_cfg
from data import fetch_polygon

TK = "TLT"
START, END = "2026-06-01", "2026-07-22"


def main():
    out = Path(f"data_cache/{TK}_recent_{START}_{END}.csv")
    cfg = ticker_cfg(TK)
    cfg.data.start_date, cfg.data.end_date = START, END
    cfg.data.multiplier, cfg.data.timespan = 5, "minute"
    for attempt in range(4):
        try:
            df = fetch_polygon(cfg)
            df.to_csv(out, index=False)
            print(f"  {TK}: saved {len(df)} rows  "
                  f"[{df['timestamp'].iloc[0]} .. {df['timestamp'].iloc[-1]}]", flush=True)
            break
        except Exception as e:
            print(f"  {TK}: attempt {attempt+1} failed ({e}); retrying in 10s", flush=True)
            time.sleep(10)
    else:
        print(f"  {TK}: GAVE UP after retries", flush=True)


if __name__ == "__main__":
    main()

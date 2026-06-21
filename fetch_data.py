"""
fetch_data.py — standalone data downloader.

Run this once to pull aggregates from Polygon.io and cache them to a CSV under
data_cache/. Training then reads from that cache (no repeated API calls). The
API key is read from POLYGON_API_KEY (loaded automatically from the .env file
next to config.py, or from your shell environment).

Usage:
  python fetch_data.py                              # default ticker/range in config
  python fetch_data.py --ticker MSFT
  python fetch_data.py --start 2024-01-01 --end 2024-06-30
  python fetch_data.py --multiplier 1 --timespan minute
  python fetch_data.py --refresh                    # ignore existing cache, re-download
"""

from __future__ import annotations

import argparse
import sys

# Windows consoles default to cp1252 and choke on non-ASCII; force UTF-8.
try:  # pragma: no cover
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from basket import BASKET_END, BASKET_START, BASKET_TICKERS, ticker_cfg
from config import default_config
from data import cache_path, fetch_polygon, load_from_cache, save_to_cache


def _fetch_one(cfg, refresh: bool) -> int:
    """Fetch+cache a single ticker described by cfg. Returns 0 on success."""
    print("-" * 64)
    print(f"{cfg.data.ticker}  {cfg.data.multiplier} {cfg.data.timespan}  "
          f"{cfg.data.start_date}..{cfg.data.end_date}")
    if not refresh:
        cached = load_from_cache(cfg)
        if cached is not None:
            print(f"  cache present: {len(cached)} rows ({cache_path(cfg).name})")
            return 0
    if not cfg.api_key:
        print("  ERROR: no POLYGON_API_KEY (put it in .env or your environment).")
        return 1
    df = fetch_polygon(cfg)
    path = save_to_cache(df, cfg)
    print(f"  saved {len(df)} rows -> {path.name}  "
          f"[{df['timestamp'].iloc[0]} .. {df['timestamp'].iloc[-1]}]  "
          f"px {df['close'].min():.2f}..{df['close'].max():.2f}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Download & cache Polygon aggregates")
    p.add_argument("--ticker", type=str, default=None)
    p.add_argument("--tickers", type=str, default=None,
                   help="comma-separated list, e.g. SPY,AAPL,MSFT")
    p.add_argument("--basket", action="store_true",
                   help="fetch the full Phase-A basket over the multi-year range")
    p.add_argument("--start", type=str, default=None, help="YYYY-MM-DD")
    p.add_argument("--end", type=str, default=None, help="YYYY-MM-DD")
    p.add_argument("--multiplier", type=int, default=None)
    p.add_argument("--timespan", type=str, default=None,
                   help="minute, hour, day, ...")
    p.add_argument("--refresh", action="store_true",
                   help="re-download even if a cache file exists")
    args = p.parse_args()

    # ---- Multi-ticker modes (basket or explicit list) --------------------- #
    if args.basket or args.tickers:
        tickers = (args.tickers.split(",") if args.tickers else BASKET_TICKERS)
        tickers = [t.strip().upper() for t in tickers if t.strip()]
        print("=" * 64)
        print(f"FETCH BASKET: {tickers}")
        print(f"range: {args.start or BASKET_START} .. {args.end or BASKET_END}")
        print("=" * 64)
        rc = 0
        for t in tickers:
            cfg = ticker_cfg(t)
            if args.start:
                cfg.data.start_date = args.start
            if args.end:
                cfg.data.end_date = args.end
            if args.multiplier:
                cfg.data.multiplier = args.multiplier
            if args.timespan:
                cfg.data.timespan = args.timespan
            rc |= _fetch_one(cfg, args.refresh)
        return rc

    cfg = default_config()
    if args.ticker:
        cfg.data.ticker = args.ticker
    if args.start:
        cfg.data.start_date = args.start
    if args.end:
        cfg.data.end_date = args.end
    if args.multiplier:
        cfg.data.multiplier = args.multiplier
    if args.timespan:
        cfg.data.timespan = args.timespan

    print("=" * 64)
    print("FETCH DATA")
    print("=" * 64)
    print(f"ticker : {cfg.data.ticker}")
    print(f"bar    : {cfg.data.multiplier} {cfg.data.timespan}")
    print(f"range  : {cfg.data.start_date} .. {cfg.data.end_date}")
    print(f"cache  : {cache_path(cfg)}")
    print(f"key    : {'set' if cfg.api_key else 'NOT set'}")

    if not args.refresh:
        cached = load_from_cache(cfg)
        if cached is not None:
            print(f"\nCache already present ({len(cached)} rows). "
                  f"Use --refresh to re-download.")
            print(cached.head().to_string(index=False))
            return 0

    if not cfg.api_key:
        print("\nERROR: no POLYGON_API_KEY found. Put it in .env or your environment.")
        return 1

    print("\nDownloading from Polygon ...")
    df = fetch_polygon(cfg)
    path = save_to_cache(df, cfg)
    print(f"\nSaved {len(df)} rows -> {path}")
    print(f"date range : {df['timestamp'].iloc[0]} .. {df['timestamp'].iloc[-1]}")
    print(f"price range: {df['close'].min():.2f} .. {df['close'].max():.2f}")
    print("\nhead:")
    print(df.head().to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

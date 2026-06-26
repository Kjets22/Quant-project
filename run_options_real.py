"""
run_options_real.py — end-to-end on REAL Polygon SPY options.

Build (or load cached) the chain -> clean/filter -> IV-space features ->
forward-realized-vol label -> LightGBM under the 10mo/1mo/1mo split. Reports
out-of-sample MAE on forward RV and the VRP-sign hit rate (the tradeable signal).

  python run_options_real.py 2025-04-01 2026-06-20
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd

from options_data_polygon import CACHE, build_chain
from options_ml_pipeline import (
    add_label,
    build_daily_features,
    clean_quotes,
    filter_tradeable,
    train_model,
)


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else "2025-04-01"
    end = sys.argv[2] if len(sys.argv) > 2 else "2026-06-20"
    pq = CACHE / f"chain_{start}_{end}.parquet"

    if pq.exists():
        print(f"[load] cached chain {pq.name}")
        ch = pd.read_parquet(pq)
    else:
        print(f"[build] fetching real SPY chain {start}..{end} (cached per contract)")
        ch = build_chain(start, end)
        try:
            ch.to_parquet(pq)
        except Exception:
            ch.to_csv(pq.with_suffix(".csv"), index=False)
        print(f"[build] chain rows {len(ch)}  days {ch['date'].nunique()}")

    ohlc = ch.groupby("date")[["open", "high", "low", "close"]].first().sort_index()
    df = clean_quotes(ch)
    df = filter_tradeable(df)
    feats = build_daily_features(df, ohlc)
    data = add_label(feats, ohlc)
    print(f"\nREAL SPY training table: {data.shape[0]} days x {data.shape[1]-1} features")
    if data.shape[0]:
        print(f"  ATM IV mean {data.atm_iv.mean():.3f}  date span "
              f"{data.index.min().date()}..{data.index.max().date()}")
    train_model(data)
    print("\n(Next: VRP backtest with realistic option spread costs + deflated Sharpe.)")


if __name__ == "__main__":
    main()

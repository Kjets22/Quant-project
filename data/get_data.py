# download_spy_databento.py

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import databento as db
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

SYMBOL = "SPY"

# Best default for ML/backtesting:
# 1-minute OHLCV bars are much smaller than raw trades/tick data.
SCHEMA = "ohlcv-1m"

# Databento US Equities Mini.
# This gives aggregated OHLCV across its component venues.
DATASET = "EQUS.MINI"

OUTPUT_FILE = Path("spy_1year_1min_databento.csv")

# Historical data is not for the most recent 24 hours.
# Using 2 days back is safer.
END_DATE = datetime.now(timezone.utc).date() - timedelta(days=2)
START_DATE = END_DATE - timedelta(days=365)

# Set to True if you only want normal market hours: 9:30 AM - 4:00 PM NY time.
FILTER_REGULAR_MARKET_HOURS = False


# ============================================================
# HELPERS
# ============================================================

def month_chunks(start_date, end_date):
    """
    Yield monthly chunks: [chunk_start, chunk_end).
    Databento end is exclusive, so this is clean for appending.
    """
    current = start_date

    while current < end_date:
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1, day=1)
        else:
            next_month = current.replace(month=current.month + 1, day=1)

        chunk_end = min(next_month, end_date)
        yield current, chunk_end
        current = chunk_end


def clean_ohlcv_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Makes the Databento dataframe easier to use for ML/backtesting.
    """
    if df.empty:
        return df

    # Databento usually puts ts_event or ts_recv as the index.
    df = df.reset_index()

    # Rename timestamp column if needed.
    if "ts_event" in df.columns:
        df = df.rename(columns={"ts_event": "timestamp"})
    elif "ts_recv" in df.columns:
        df = df.rename(columns={"ts_recv": "timestamp"})
    elif df.columns[0] not in ["timestamp"]:
        df = df.rename(columns={df.columns[0]: "timestamp"})

    # Keep only useful columns if they exist.
    wanted_cols = [
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    existing_cols = [col for col in wanted_cols if col in df.columns]
    df = df[existing_cols]

    # Sort just to be safe.
    df = df.sort_values("timestamp")

    if FILTER_REGULAR_MARKET_HOURS:
        ts = pd.to_datetime(df["timestamp"], utc=True)
        ny_time = ts.dt.tz_convert("America/New_York")

        market_open = pd.to_datetime("09:30").time()
        market_close = pd.to_datetime("16:00").time()

        df = df[
            (ny_time.dt.time >= market_open)
            & (ny_time.dt.time < market_close)
            & (ny_time.dt.weekday < 5)
        ]

    return df


# ============================================================
# MAIN DOWNLOAD
# ============================================================

def main():
    if "DATABENTO_API_KEY" not in os.environ:
        raise RuntimeError(
            "Missing DATABENTO_API_KEY.\n\n"
            "Set it first:\n"
            "Mac/Linux:\n"
            "  export DATABENTO_API_KEY='your_key_here'\n\n"
            "Windows PowerShell:\n"
            "  $env:DATABENTO_API_KEY='your_key_here'\n"
        )

    client = db.Historical()

    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    wrote_header = False
    total_rows = 0

    print(f"Downloading {SYMBOL} {SCHEMA} from {START_DATE} to {END_DATE}")
    print(f"Dataset: {DATASET}")
    print(f"Output: {OUTPUT_FILE}")

    for chunk_start, chunk_end in month_chunks(START_DATE, END_DATE):
        print(f"\nDownloading chunk: {chunk_start} to {chunk_end}")

        data = client.timeseries.get_range(
            dataset=DATASET,
            symbols=[SYMBOL],
            schema=SCHEMA,
            start=str(chunk_start),
            end=str(chunk_end),
            stype_in="raw_symbol",
        )

        df = data.to_df(
            price_type="float",
            pretty_ts=True,
            map_symbols=True,
        )

        df = clean_ohlcv_dataframe(df)

        if df.empty:
            print("No rows returned for this chunk.")
            continue

        df.to_csv(
            OUTPUT_FILE,
            mode="a",
            header=not wrote_header,
            index=False,
        )

        wrote_header = True
        total_rows += len(df)

        print(f"Saved {len(df):,} rows. Total so far: {total_rows:,}")

    print("\nDone.")
    print(f"Saved {total_rows:,} rows to {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()

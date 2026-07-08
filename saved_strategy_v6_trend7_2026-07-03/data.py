"""
data.py — Part 1: data acquisition.

Two sources, one interface (`load_data`):
  * Polygon.io aggregates REST endpoint (used when an API key is present).
  * A regime-switching synthetic generator (used otherwise) so the entire
    pipeline runs and self-tests with no key and no network.

Security / robustness notes:
  * The API key is never hardcoded; it comes from cfg.api_key (env var).
  * HTTP errors are handled; HTTP 429 (rate limit) triggers a bounded backoff.
  * No eval/exec, no shell-outs. Only requests.get against the Polygon host.
"""

from __future__ import annotations

import time

# Some networks (corporate proxies / antivirus) perform TLS interception with a
# private root CA that Windows trusts but Python's certifi bundle does not. We
# route SSL verification through the OS trust store via `truststore`.
#
# IMPORTANT: inject_into_ssl() must run BEFORE requests/urllib3 are imported, so
# the patch applies to every connection pool (including proxy pools). Running it
# afterwards can raise "maximum recursion depth exceeded". `enable_truststore()`
# is also exposed so entry-point scripts can inject before importing anything
# that pulls in urllib3 (e.g. stable-baselines3).
def enable_truststore() -> None:
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:  # noqa: BLE001 - truststore missing -> default verification
        pass


enable_truststore()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

from config import Config, default_config  # noqa: E402

POLYGON_HOST = "https://api.polygon.io"


def _make_session() -> requests.Session:
    return requests.Session()
_AGG_PATH = "/v2/aggs/ticker/{ticker}/range/{mult}/{timespan}/{start}/{end}"

OHLCV_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


# --------------------------------------------------------------------------- #
# Polygon.io                                                                   #
# --------------------------------------------------------------------------- #
def _map_results(results: list[dict]) -> list[dict]:
    """Map Polygon aggregate result objects to OHLCV rows."""
    rows = []
    for r in results:
        rows.append(
            {
                "timestamp": pd.to_datetime(r["t"], unit="ms"),
                "open": float(r["o"]),
                "high": float(r["h"]),
                "low": float(r["l"]),
                "close": float(r["c"]),
                "volume": float(r.get("v", 0.0)),
            }
        )
    return rows


def fetch_polygon(cfg: Config, max_pages: int = 100, max_retries: int = 5) -> pd.DataFrame:
    """
    Fetch aggregates from Polygon, following the `next_url` cursor for
    pagination. Raises RuntimeError if no key is configured.
    """
    key = cfg.api_key
    if not key:
        raise RuntimeError("No POLYGON_API_KEY available for fetch_polygon().")

    d = cfg.data
    url = POLYGON_HOST + _AGG_PATH.format(
        ticker=d.ticker, mult=d.multiplier, timespan=d.timespan,
        start=d.start_date, end=d.end_date,
    )
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": key}

    session = _make_session()
    all_rows: list[dict] = []
    pages = 0
    while url and pages < max_pages:
        retries = 0
        while True:
            resp = session.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                # Rate limited: bounded backoff and retry the same page.
                if retries >= max_retries:
                    raise RuntimeError("Polygon rate limit: retries exhausted.")
                wait = 15 * (retries + 1)
                print(f"[data] HTTP 429 rate limited; sleeping {wait}s ...")
                time.sleep(wait)
                retries += 1
                continue
            resp.raise_for_status()
            break

        payload = resp.json()
        results = payload.get("results") or []
        all_rows.extend(_map_results(results))

        # Pagination: next_url already encodes query params except the apiKey.
        next_url = payload.get("next_url")
        if next_url:
            url = next_url
            params = {"apiKey": key}   # only need to append the key to the cursor
        else:
            url = None
        pages += 1

    if not all_rows:
        raise RuntimeError("Polygon returned no rows for the requested range.")

    df = pd.DataFrame(all_rows, columns=OHLCV_COLS)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Synthetic fallback                                                           #
# --------------------------------------------------------------------------- #
# Three+ regimes driven by a Markov transition matrix. Each regime has a
# (drift, volatility) profile. We simulate a close price via geometric Brownian
# motion per-regime, then build OHLC around each close with realistic wicks and
# a volume that scales with volatility.
_REGIMES = {
    # name           drift (per-bar)   vol (per-bar)
    "calm_up":       (+0.00020,         0.0035),
    "volatile_chop": (+0.00000,         0.0110),
    "down":          (-0.00028,         0.0060),
}
_REGIME_NAMES = list(_REGIMES.keys())

# Row-stochastic transition matrix (sticky regimes).
_TRANSITION = np.array(
    [
        [0.97, 0.02, 0.01],   # from calm_up
        [0.03, 0.94, 0.03],   # from volatile_chop
        [0.02, 0.03, 0.95],   # from down
    ]
)


def make_synthetic(n: int = 1000, seed: int = 7, start_price: float = 150.0) -> pd.DataFrame:
    """
    Regime-switching geometric Brownian motion with >=3 regimes.

    Returns a DataFrame with columns matching OHLCV_COLS, length `n`, all OHLC
    strictly positive, high>=max(open,close,low) and low<=min(...), no NaNs.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    rng = np.random.default_rng(seed)

    # Markov walk over regimes.
    regimes = np.empty(n, dtype=np.int64)
    state = 0
    for i in range(n):
        regimes[i] = state
        state = rng.choice(3, p=_TRANSITION[state])

    drift = np.array([_REGIMES[name][0] for name in _REGIME_NAMES])
    vol = np.array([_REGIMES[name][1] for name in _REGIME_NAMES])

    # Per-bar log returns -> close price path via GBM.
    mu = drift[regimes]
    sigma = vol[regimes]
    shocks = rng.standard_normal(n)
    log_ret = mu - 0.5 * sigma**2 + sigma * shocks
    close = start_price * np.exp(np.cumsum(log_ret))

    # Open = previous close (first bar opens at start_price).
    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]

    # Wicks: high/low extend beyond the open-close body by a vol-scaled amount.
    body_hi = np.maximum(open_, close)
    body_lo = np.minimum(open_, close)
    wick_up = np.abs(rng.standard_normal(n)) * sigma * close
    wick_dn = np.abs(rng.standard_normal(n)) * sigma * close
    high = body_hi + wick_up
    low = body_lo - wick_dn

    # Guard: keep everything strictly positive even for pathological draws.
    low = np.maximum(low, 1e-6)
    high = np.maximum(high, low + 1e-6)

    # Volume scales with per-bar volatility plus noise.
    base_vol = 1_000_000.0
    volume = base_vol * (1.0 + 8.0 * sigma) * (0.5 + rng.random(n))

    timestamps = pd.date_range("2023-01-02 09:30", periods=n, freq="5min")

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        columns=OHLCV_COLS,
    )
    return df


# --------------------------------------------------------------------------- #
# Disk cache                                                                   #
# --------------------------------------------------------------------------- #
import os
from pathlib import Path

CACHE_DIR = Path(__file__).with_name("data_cache")


def cache_path(cfg: Config) -> Path:
    """Deterministic CSV cache path for the configured ticker/range/bar."""
    d = cfg.data
    name = f"{d.ticker}_{d.multiplier}{d.timespan}_{d.start_date}_{d.end_date}.csv"
    return CACHE_DIR / name


def save_to_cache(df: pd.DataFrame, cfg: Config) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(cfg)
    df.to_csv(path, index=False)
    return path


def load_from_cache(cfg: Config) -> pd.DataFrame | None:
    path = cache_path(cfg)
    if path.exists():
        df = pd.read_csv(path, parse_dates=["timestamp"])
        return df.sort_values("timestamp").reset_index(drop=True)
    return None


# --------------------------------------------------------------------------- #
# Unified loader                                                               #
# --------------------------------------------------------------------------- #
def load_data(cfg: Config, use_cache: bool = True, refresh: bool = False) -> pd.DataFrame:
    """
    Resolution order:
      1. On-disk CSV cache (unless refresh=True)            -> instant, no network
      2. Polygon fetch when an API key is present (cached)  -> real data
      3. Synthetic fallback                                 -> always works
    """
    if use_cache and not refresh:
        cached = load_from_cache(cfg)
        if cached is not None:
            print(f"[data] Loaded {len(cached)} rows from cache: {cache_path(cfg).name}")
            return cached

    if cfg.api_key:
        try:
            print("[data] Polygon key found; fetching real aggregates ...")
            df = fetch_polygon(cfg)
            if use_cache:
                saved = save_to_cache(df, cfg)
                print(f"[data] Cached {len(df)} rows -> {saved.name}")
            return df
        except Exception as exc:  # noqa: BLE001 - we want a graceful fallback
            if cfg.data.use_synthetic_if_no_key:
                print(f"[data] Polygon fetch failed ({exc}); using synthetic data.")
                return make_synthetic(cfg.data.synthetic_n, cfg.data.synthetic_seed)
            raise

    print("[data] No Polygon key; using synthetic data.")
    return make_synthetic(cfg.data.synthetic_n, cfg.data.synthetic_seed)


if __name__ == "__main__":
    cfg = default_config()
    df = make_synthetic(500, seed=cfg.data.synthetic_seed)
    print("=== data.py self-test: make_synthetic(500) ===")
    print("rows         :", len(df))
    print("columns      :", list(df.columns))
    assert len(df) == 500, "expected 500 rows"
    assert (df[["open", "high", "low", "close"]] > 0).all().all(), "non-positive OHLC"
    assert (df["high"] >= df["low"]).all(), "high < low somewhere"
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all(), "high below body"
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all(), "low above body"
    assert not df.isna().any().any(), "NaNs present"
    print("\n", df.head().to_string(index=False))
    print("\nprice range  : %.4f .. %.4f" % (df["low"].min(), df["high"].max()))
    print("OK — all assertions passed")

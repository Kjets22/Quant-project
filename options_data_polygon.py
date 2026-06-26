"""
options_data_polygon.py — build a REAL historical SPY option chain from Polygon.

Polygon REST gives option price aggregates (OHLCV) + volume, but NOT historical
bid/ask or open-interest. So we: list expired contracts, pull each contract's
daily closes, compute IV (Black-Scholes inversion) and delta ourselves, and emit
the schema options_ml_pipeline expects. Spread is modeled later as a COST, not a
filter (we gate liquidity on real traded volume + moneyness + DTE).

Everything is cached to data_cache/options/ so re-runs are cheap.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import data  # truststore SSL + .env
from config import default_config
from options_ml_pipeline import _bs_price_and_delta, implied_vol

CACHE = Path(__file__).with_name("data_cache") / "options"
CACHE.mkdir(parents=True, exist_ok=True)
R = 0.04  # risk-free


def _session_key():
    return data._make_session(), default_config().api_key


def _get(session, key, url, params, retries=6):
    params = {**params, "apiKey": key}
    for i in range(retries):
        try:
            r = session.get(url, params=params, timeout=45)
        except requests.exceptions.RequestException:
            time.sleep(5 * (i + 1))         # transient timeout/connection drop -> retry
            continue
        if r.status_code == 429:
            time.sleep(10 * (i + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"request failed after {retries} retries: {url}")


def underlying_daily(ul, start, end):
    s, k = _session_key()
    j = _get(s, k, f"https://api.polygon.io/v2/aggs/ticker/{ul}/range/1/day/{start}/{end}",
             {"limit": 50000, "adjusted": "true"})
    rows = [{"date": dt.datetime.utcfromtimestamp(b["t"] / 1000).date(),
             "open": b["o"], "high": b["h"], "low": b["l"], "close": b["c"]}
            for b in j.get("results", [])]
    return pd.DataFrame(rows)


def list_contracts(ul, exp_gte, exp_lte, strike_lo, strike_hi):
    """All expired contracts (both types) in the expiry + strike window."""
    s, k = _session_key()
    out = []
    for ctype in ("call", "put"):
        url = "https://api.polygon.io/v3/reference/options/contracts"
        params = {"underlying_ticker": ul, "contract_type": ctype,
                  "expiration_date.gte": exp_gte, "expiration_date.lte": exp_lte,
                  "strike_price.gte": strike_lo, "strike_price.lte": strike_hi,
                  "expired": "true", "limit": 1000}
        while True:
            j = _get(s, k, url, params)
            out += j.get("results", [])
            nxt = j.get("next_url")
            if not nxt:
                break
            url, params = nxt, {}
    return out


def contract_aggs(ticker, start, end):
    cf = CACHE / f"{ticker.replace(':', '_')}.json"
    if cf.exists():
        return json.loads(cf.read_text())
    s, k = _session_key()
    try:
        j = _get(s, k, f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}",
                 {"limit": 50000, "adjusted": "true"})
    except RuntimeError:
        return []                          # skip this contract; don't kill the run
    bars = [{"date": str(dt.datetime.utcfromtimestamp(b["t"] / 1000).date()),
             "close": b["c"], "volume": b["v"]} for b in j.get("results", [])]
    cf.write_text(json.dumps(bars))
    return bars


def build_chain(start: str, end: str, underlying: str = "SPY",
                fridays_only: bool = True, band: float = 0.10) -> pd.DataFrame:
    """Assemble the per-(date, contract) chain DataFrame for [start, end]."""
    spy = underlying_daily(underlying, start, end)
    spy["date"] = pd.to_datetime(spy["date"])
    spot = {d.date(): c for d, c in zip(spy["date"], spy["close"])}
    smin, smax = spy["close"].min(), spy["close"].max()

    contracts = list_contracts(underlying, start, end,
                               smin * (1 - band), smax * (1 + band))
    if fridays_only:
        contracts = [c for c in contracts
                     if pd.Timestamp(c["expiration_date"]).dayofweek == 4]
    print(f"[chain] {len(contracts)} contracts to fetch (caching) ...")

    rows = []
    for i, c in enumerate(contracts):
        if i % 200 == 0:
            print(f"  ... {i}/{len(contracts)}")
        tick = c["ticker"]
        K = float(c["strike_price"])
        exp = dt.date.fromisoformat(c["expiration_date"])
        is_call = c["contract_type"] == "call"
        for b in contract_aggs(tick, start, end):
            d = dt.date.fromisoformat(b["date"])
            if d not in spot or b["volume"] < 1:
                continue
            S = spot[d]
            if not (smin * (1 - band) <= K <= smax * (1 + band)):
                continue
            T = (exp - d).days / 365.0
            if T <= 0:
                continue
            iv = implied_vol(b["close"], S, K, T, R, is_call)
            if not np.isfinite(iv):
                continue
            _, delta = _bs_price_and_delta(S, K, T, R, iv, is_call)
            rows.append({
                "date": d, "underlying": underlying, "expiry": exp, "strike": K,
                "type": "C" if is_call else "P",
                "bid": b["close"] * 0.999, "ask": b["close"] * 1.001,  # modeled (cost handled later)
                "volume": b["volume"], "open_interest": b["volume"],   # OI proxy (no hist OI)
                "iv": iv, "delta": delta, "underlying_price": S,
            })
    chain = pd.DataFrame(rows)
    chain["date"] = pd.to_datetime(chain["date"])
    chain["expiry"] = pd.to_datetime(chain["expiry"])
    # attach SPY OHLC per day (for Yang-Zhang realized vol)
    chain = chain.merge(spy.rename(columns={"close": "u_close"}), on="date", how="left")
    chain = chain.rename(columns={"open": "open", "high": "high", "low": "low"})
    chain["close"] = chain["u_close"]
    return chain.drop(columns=["u_close"])


if __name__ == "__main__":
    import sys
    ul = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    start = sys.argv[2] if len(sys.argv) > 2 else "2025-04-01"
    end = sys.argv[3] if len(sys.argv) > 3 else "2025-06-20"
    ch = build_chain(start, end, underlying=ul)
    print(f"\nchain rows: {len(ch)}  days: {ch['date'].nunique()}  "
          f"contracts: {ch[['strike','expiry','type']].drop_duplicates().shape[0]}")
    print("IV range:", round(ch['iv'].min(), 3), "-", round(ch['iv'].max(), 3))
    out = CACHE / f"chain_{start}_{end}.parquet"
    try:
        ch.to_parquet(out)
        print("saved", out.name)
    except Exception:
        ch.to_csv(out.with_suffix(".csv"), index=False)
        print("saved", out.with_suffix(".csv").name)

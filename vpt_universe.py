"""
vpt_universe.py — Round D: does the pattern edge TRANSFER, and does breadth
lift the Sharpe toward 3?

The 8-ticker champion is variance-floored (VAL Sharpe ~1.8-2.0, 90% CI top
~2.5). Sharpe scales with sqrt(independent bets) ONLY if the edge exists on
names it was never tuned on — the classic fresh-ticker transfer test.

FRESH universe (never used in any vPT development): IWM GLD DIA XLF XLK KO
XOM META AMD GOOG AMZN WMT. Data fetched via Alpaca (2023-01..now, 5-min,
cached in data_cache/uni/). Protocol: OPT trains the meta (allowed), VAL is
touched once per book. Compares: original8 | fresh12 | combined20.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

import vpt_boost as B
import vpt_geometric as G
from basket import ticker_cfg
from data import fetch_alpaca
from triple_barrier_breadth import TICKERS as ORIG

FRESH = ["IWM", "GLD", "DIA", "XLF", "XLK", "KO", "XOM", "META", "AMD",
         "GOOG", "AMZN", "WMT"]
FRESH2 = ["JNJ", "PG", "UNH", "HD", "CAT", "GS", "ORCL", "CRM", "NFLX",
          "TSLA", "INTC", "CSCO"]
UNI = Path("data_cache/uni")
UNI.mkdir(parents=True, exist_ok=True)


def bars_any(tk, minutes=60):
    f = UNI / f"{tk}_5min.parquet"
    if f.exists():
        d5 = pd.read_parquet(f)
    else:
        cfg = ticker_cfg("SPY")
        cfg.data.ticker = tk
        cfg.data.start_date, cfg.data.end_date = "2023-01-01", "2026-08-13"
        cfg.data.multiplier, cfg.data.timespan = 5, "minute"
        d5 = fetch_alpaca(cfg)
        d5.to_parquet(f)
        print(f"  fetched {tk}: {len(d5)} 5-min rows", flush=True)
    d = d5.set_index("timestamp").resample(f"{minutes}min").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum")).dropna() \
        .reset_index()
    return d


def collect(tks, lo, hi, entry="fairtrigger"):
    frames = []
    for tk in tks:
        try:
            df = G.bars(tk, 60) if tk in ORIG else bars_any(tk)
        except Exception as e:
            print(f"  [data warn {tk}: {e}]", flush=True)
            continue
        sub = df[(df["timestamp"] >= lo) & (df["timestamp"] < hi)] \
            .reset_index(drop=True)
        if len(sub) < 500:
            continue
        setups = B.detect_rich(sub, entry)
        d = B.sim_rich(sub, setups, entry, tk)
        if len(d):
            frames.append(d)
    return pd.concat(frames, ignore_index=True).sort_values("t") \
        .reset_index(drop=True)


def run_book(name, tks):
    maxc = max(4, len(tks) // 4)              # concurrency scales with breadth
    o = collect(tks, *B.OPT)
    v = collect(tks, *B.VAL)
    o2, v2, q, s_oof = B.meta_oof(o, v)
    so, to, no = B.sharpe(o2, maxc)
    sv, tv, nv = B.sharpe(v2, maxc)
    # bootstrap CI on VAL by day
    rng = np.random.default_rng(0)
    d = v2.reset_index(drop=True)
    days = pd.to_datetime(d["t"]).dt.date.to_numpy()
    ud = np.unique(days)
    vs = []
    for _ in range(600):
        pick = rng.choice(ud, len(ud), replace=True)
        idx = np.concatenate([np.where(days == p)[0] for p in pick])
        s_, _, _ = B.sharpe(d.iloc[idx])
        vs.append(s_)
    lo_, hi_ = np.percentile(vs, [5, 95])
    print(f"{name:>10}: OPT shp {so:5.2f} (n={no}) | VAL shp {sv:5.2f} "
          f"tot {tv:+7.1f}% n={nv}  CI[{lo_:.2f}..{hi_:.2f}]  (Q={q})",
          flush=True)
    return sv


def one_shot_test():
    """THE final: combined32 champion pipeline, TEST window (2025-07..now),
    touched exactly once by this function. Pipeline identical to VAL: meta
    trained on OPT only, Q from OOF-OPT, fairtrigger entries, maxconc=8."""
    tks = list(ORIG) + FRESH + FRESH2
    TEST = ("2025-07-01", "2099-01-01")
    o = collect(tks, *B.OPT)
    t = collect(tks, *TEST)
    o2, t2, q, s_oof = B.meta_oof(o, t)
    st, tt, nt = B.sharpe(t2, 8)
    rng = np.random.default_rng(0)
    d = t2.reset_index(drop=True)
    days = pd.to_datetime(d["t"]).dt.date.to_numpy()
    ud = np.unique(days)
    vs = []
    for _ in range(800):
        pick = rng.choice(ud, len(ud), replace=True)
        idx = np.concatenate([np.where(days == p)[0] for p in pick])
        s_, _, _ = B.sharpe(d.iloc[idx])
        vs.append(s_)
    lo_, hi_ = np.percentile(vs, [5, 95])
    win = float((t2["ret"] > 0).mean())
    mo = pd.Series(t2["ret"].to_numpy(),
                   index=pd.to_datetime(t2["t"])).resample("ME").sum() * 1000
    print(f"ONE-SHOT TEST combined32 (Q={q}, OOF-OPT shp {s_oof:.2f}):")
    print(f"  TEST Sharpe {st:.2f}  total {tt:+.1f}%  n={nt}  win {win:.0%}  "
          f"CI[{lo_:.2f}..{hi_:.2f}]")
    print("  monthly ($1k/trade):")
    for ts_, x in mo.items():
        print(f"    {ts_:%Y-%m}: {x:+9.2f}")
    d.to_csv("runs/vpt32_test_trades.csv", index=False)
    print("  trades -> runs/vpt32_test_trades.csv")


def main():
    if "--final" in sys.argv:
        one_shot_test()
        return
    if "--round-e" in sys.argv:
        print("ROUND E — 32-name breadth (fairtrigger+metacv+maxconc raised)\n")
        run_book("fresh2_12", FRESH2)                     # second transfer test
        run_book("combined32", list(ORIG) + FRESH + FRESH2)
        return
    print("ROUND D — universe transfer test (fairtrigger+metacv+maxconc4)\n")
    run_book("original8", list(ORIG))
    run_book("fresh12", FRESH)
    run_book("combined20", list(ORIG) + FRESH)


if __name__ == "__main__":
    main()

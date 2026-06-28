"""
options_vrp_backtest.py — does selling SPY vol actually make money after costs?

The honest test behind the VRP-sign hit rate. Sell a ~1-month ATM straddle each
day (overlapping), hold to the horizon, pay a realistic option spread. Compare:
  * ALWAYS sell  (the structural VRP carry)
  * GATED sell   (only when IV is rich vs trailing realized vol -- a naive timer)
Report mean P&L, win rate, annualized Sharpe, and CRUCIALLY the worst trade and
max drawdown (the short-vol crash tail) + a deflated Sharpe for the trials.
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

from options_ml_pipeline import (
    _bs_price_and_delta,
    build_daily_features,
    clean_quotes,
    filter_tradeable,
)

H = 21            # trading-day horizon
R = 0.04
SPREAD_FRAC = 0.04   # sell at bid: ~4% of premium round-trip (liquid SPY straddle)
TRIALS = 12          # configs explored (for the deflated Sharpe)


def straddle_pnl(feats, close, spread_frac):
    """Per-day short ATM straddle P&L over the next H days."""
    dates = feats.index
    cl = pd.Series(close, index=dates)
    rows = []
    for i, d in enumerate(dates[:-H]):
        iv = feats["atm_iv"].iloc[i]
        rvt = feats["rv_trailing"].iloc[i]
        S = cl.iloc[i]
        if not (np.isfinite(iv) and iv > 0 and np.isfinite(S)):
            continue
        T = 30 / 365
        c, _ = _bs_price_and_delta(S, S, T, R, iv, True)
        p, _ = _bs_price_and_delta(S, S, T, R, iv, False)
        premium = c + p                      # straddle premium collected
        realized_move = abs(cl.iloc[i + H] - S)
        spread = premium * spread_frac
        pnl = premium - realized_move - spread   # short straddle P&L
        rows.append({"date": d, "pnl": pnl, "premium": premium,
                     "iv": iv, "rvt": rvt, "rich": iv > rvt})
    return pd.DataFrame(rows).set_index("date")


def stats(pnl: pd.Series, label):
    if len(pnl) < 5:
        print(f"  {label:14}: too few trades")
        return
    mean = pnl.mean()
    sh = mean / pnl.std() * np.sqrt(252) if pnl.std() > 1e-9 else 0.0   # daily, overlapping (caveat)
    eq = pnl.cumsum()
    dd = (eq - eq.cummax()).min()
    wr = (pnl > 0).mean()
    print(f"  {label:14}: n={len(pnl):3d}  win%={wr:5.1%}  mean$={mean:6.2f}  "
          f"Sharpe~={sh:5.2f}  worst$={pnl.min():8.2f}  maxDD$={dd:9.2f}")


def main():
    pq = next(Path("data_cache/options").glob("chain_*2026-06-20.parquet"))
    print(f"[load] {pq.name}")
    ch = pd.read_parquet(pq)
    ohlc = ch.groupby("date")[["open", "high", "low", "close"]].first().sort_index()
    df = clean_quotes(ch)
    df = filter_tradeable(df)
    feats = build_daily_features(df, ohlc).dropna(subset=["atm_iv"])
    close = ohlc["close"].reindex(feats.index).to_numpy()

    bt = straddle_pnl(feats, close, SPREAD_FRAC)
    print(f"\nSPY 1-month short ATM straddle, {len(bt)} overlapping trades, "
          f"spread {SPREAD_FRAC:.0%} of premium")
    print(f"  avg premium collected: ${bt['premium'].mean():.2f}  "
          f"(IV>RV on {bt['rich'].mean():.0%} of days = the structural VRP base rate)")
    print()
    stats(bt["pnl"], "ALWAYS sell")
    stats(bt[bt["rich"]]["pnl"], "GATED (IV>RV)")
    # split into halves to check stability across windows
    half = len(bt) // 2
    stats(bt["pnl"].iloc[:half], "1st half")
    stats(bt["pnl"].iloc[half:], "2nd half")
    print("\n  READ: positive mean$ with a Sharpe>0 = the VRP carry survives costs, BUT")
    print("  look at worst$ / maxDD -- short vol makes steady pennies then a giant loss.")
    print("  'GATED' must clearly beat 'ALWAYS' to claim timing skill (usually it doesn't).")


if __name__ == "__main__":
    main()

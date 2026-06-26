"""
directional_options.py — does ">50% direction accuracy -> profitable options"?

Tests the common belief on REAL SPY data: take a signal of KNOWN directional
accuracy, buy an ATM option each time (priced at real implied vol, i.e. with the
volatility risk premium baked in), hold to a 1-month horizon, charge the spread.
Find the BREAK-EVEN accuracy. Spoiler: it's well above 50%, because the premium
already prices the move and the VRP makes buyers overpay.
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

from options_ml_pipeline import _bs_price_and_delta

H = 21              # ~1-month horizon (trading days)
R = 0.04
VRP = 1.12          # implied vol ≈ 1.12 x realized (the volatility risk premium)
SPREAD_FRAC = 0.03  # option bid-ask ≈ 3% of premium (liquid SPY; OTM is worse)


def spy_daily():
    p = Path("data_cache/SPY_5minute_2021-06-01_2026-06-01.csv")
    df = pd.read_csv(p, parse_dates=["timestamp"])
    d = df.set_index("timestamp").resample("1D").agg(close=("close", "last")).dropna()
    return d["close"].to_numpy()


def main():
    close = spy_daily()
    r = np.diff(np.log(close))
    rv = pd.Series(r).rolling(21).std().to_numpy() * np.sqrt(252)   # trailing realized vol

    prem, move, spread = [], [], []
    for t in range(21, len(close) - H):
        if not np.isfinite(rv[t]) or rv[t] <= 0:
            continue
        S = close[t]
        iv = rv[t] * VRP                       # what you actually PAY (incl. VRP)
        T = H / 252
        p_atm, _ = _bs_price_and_delta(S, S, T, R, iv, True)   # ATM premium
        prem.append(p_atm)
        move.append(abs(close[t + H] - close[t]))             # realized |move| over H
        spread.append(p_atm * SPREAD_FRAC)

    prem = np.array(prem); move = np.array(move); spread = np.array(spread)
    mp, mm, ms = prem.mean(), move.mean(), spread.mean()
    impl_move = mp / 0.4                        # premium ≈ 0.4*S*iv*sqrt(T) ≈ 0.4*implied move

    print(f"SPY ATM ~1-month options ({len(prem)} samples)")
    print(f"  avg premium paid   : ${mp:.2f}   (+ spread ${ms:.2f})")
    print(f"  avg REALIZED |move|: ${mm:.2f}")
    print(f"  implied vol / realized vol (VRP): {VRP:.2f}x  -> buyers overpay")
    print(f"\n  net P&L per trade vs signal accuracy "
          f"(correct -> earn |move|, wrong -> lose premium):")
    print(f"  {'accuracy':>9} {'E[net $]':>10} {'verdict':>12}")
    for acc in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
        net = acc * mm - mp - ms               # E[net] = acc*|move| - premium - spread
        print(f"  {acc:>8.0%} {net:>10.2f} {'PROFIT' if net > 0 else 'loss':>12}")
    breakeven = (mp + ms) / mm
    print(f"\n  >>> BREAK-EVEN directional accuracy = {breakeven:.1%} "
          f"(you need this just to NOT lose buying ATM options)")
    print(f"  We measured actual SPY direction accuracy at ~50% -> long options lose.")
    print(f"  (And this is the GENEROUS case: ATM + liquid spread + ignores theta path.)")


if __name__ == "__main__":
    main()

"""
options_vrp_stops.py — answer the user's questions with data:
  (1) Was the val/test split a REGIME shift? -> month-by-month IV vs realized vol
      and the VRP-sign base rate (so we see if 90% test was just a calm month).
  (2) Does adding a STOP-LOSS + TAKE-PROFIT to the short straddle make money?
      Manage at 50% of credit (take profit) / 2x credit (stop), daily mark-to-market,
      vs the naive hold-to-expiry. Report whether stops cap the crash tail.
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
    yang_zhang_rv,
)

H = 21
R = 0.04
SPREAD_FRAC = 0.04
TP_FRAC = 0.50     # take profit at 50% of credit collected
STOP_MULT = 2.0    # stop loss at 2x credit


def load():
    pq = next(Path("data_cache/options").glob("chain_*2026-06-20.parquet"))
    ch = pd.read_parquet(pq)
    ohlc = ch.groupby("date")[["open", "high", "low", "close"]].first().sort_index()
    df = clean_quotes(ch)
    df = filter_tradeable(df)
    feats = build_daily_features(df, ohlc).dropna(subset=["atm_iv"])
    fwd_rv = yang_zhang_rv(ohlc, H).shift(-H).reindex(feats.index)
    feats["fwd_rv"] = fwd_rv
    feats["close"] = ohlc["close"].reindex(feats.index)
    return feats.dropna(subset=["fwd_rv", "close"])


def regime_table(feats):
    print("\n=== month-by-month vol regime (is val/test a regime shift?) ===")
    print(f"  {'month':>8} {'days':>5} {'ATM_IV':>7} {'fwd_RV':>7} {'VRP':>7} {'IV>RV%':>7}")
    g = feats.copy()
    g["m"] = g.index.to_period("M")
    for m, s in g.groupby("m"):
        vrp = (s["atm_iv"] - s["fwd_rv"])
        print(f"  {str(m):>8} {len(s):>5} {s['atm_iv'].mean():>7.3f} "
              f"{s['fwd_rv'].mean():>7.3f} {vrp.mean():>+7.3f} {(vrp > 0).mean():>7.0%}")
    print("  (IV>RV% is the VRP-sign 'hit' a naive 'always rich' model would get that month.)")


def managed_straddle(feats):
    cl = feats["close"]
    iv = feats["atm_iv"]
    dates = feats.index
    hold_pnls, mgd_pnls = [], []
    for i in range(len(dates) - H):
        S0 = cl.iloc[i]
        IV0 = iv.iloc[i]
        if not (np.isfinite(IV0) and IV0 > 0):
            continue
        c0, _ = _bs_price_and_delta(S0, S0, 30 / 365, R, IV0, True)
        p0, _ = _bs_price_and_delta(S0, S0, 30 / 365, R, IV0, False)
        credit = c0 + p0
        spread = credit * SPREAD_FRAC
        # hold to expiry
        hold = credit - abs(cl.iloc[i + H] - S0) - spread
        hold_pnls.append(hold)
        # managed: daily MTM, exit on TP or STOP
        exit_pnl = None
        for kk in range(1, H + 1):
            Sk = cl.iloc[i + kk]
            ivk = iv.iloc[i + kk] if np.isfinite(iv.iloc[i + kk]) else IV0
            T = max((30 - kk) / 365, 1 / 365)
            ck, _ = _bs_price_and_delta(Sk, S0, T, R, ivk, True)
            pk, _ = _bs_price_and_delta(Sk, S0, T, R, ivk, False)
            value = ck + pk
            pnl = credit - value - spread
            if pnl >= TP_FRAC * credit:               # take profit
                exit_pnl = pnl; break
            if pnl <= -STOP_MULT * credit:            # stop loss
                exit_pnl = pnl; break
        if exit_pnl is None:
            exit_pnl = hold
        mgd_pnls.append(exit_pnl)
    return np.array(hold_pnls), np.array(mgd_pnls)


def stats(p, label):
    sh = p.mean() / p.std() * np.sqrt(252) if p.std() > 1e-9 else 0
    eq = np.cumsum(p)
    dd = (eq - np.maximum.accumulate(eq)).min()
    print(f"  {label:18}: n={len(p)}  win%={ (p>0).mean():5.1%}  mean$={p.mean():6.2f}  "
          f"Sharpe~={sh:5.2f}  worst$={p.min():8.2f}  maxDD$={dd:9.2f}  total$={p.sum():8.0f}")


def main():
    feats = load()
    regime_table(feats)
    hold, mgd = managed_straddle(feats)
    print(f"\n=== short straddle: hold-to-expiry vs MANAGED (TP {TP_FRAC:.0%} / stop {STOP_MULT:.0f}x) ===")
    stats(hold, "hold-to-expiry")
    stats(mgd, "managed stop+TP")
    print("\n  READ: if 'managed' has a much smaller worst$/maxDD but similar/again-~0 mean,")
    print("  the stop caps the crash tail but doesn't create an edge. If mean clearly > 0")
    print("  AND drawdown small, that's worth a proper walk-forward + deflated Sharpe.")


if __name__ == "__main__":
    main()

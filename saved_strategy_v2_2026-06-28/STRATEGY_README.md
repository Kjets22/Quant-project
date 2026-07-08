# FROZEN STRATEGY — v2 (saved 2026-06-28): the 1.5:1 variant

Frozen snapshot. Do not edit. Identical to **v1** in every way EXCEPT the bracket
ratio: target = **1.5 ATR**, stop = **1.0 ATR** (v1 is 1:1). The two live side by side:
- `saved_strategy_v1_2026-06-28\`  = 1:1   (the original validated-base config)
- `saved_strategy_v2_2026-06-28\`  = 1.5:1 (this folder, the tuned variant)

## What it is (same engine as v1)
A rules-based bracket trade with a **LightGBM classifier as the trade filter**
(triple-barrier / meta-labeling). Long-only, hourly bars, top ~7% conviction signals,
walk-forward, 3 bps cost, 8 names (SPY, QQQ, AAPL, MSFT, NVDA, JPM, XLE, TLT). The ONLY
change vs v1 is the payoff geometry. See v1's README for the full mechanism.

## Why 1.5:1 (the tuning result that motivated this snapshot)
From optimize_tpsl.py (parent dir), pooled walk-forward, 3 bps:
- **STOCK 1.5:1: total +61%, win 44.1% (break-even 40%), Sharpe ~0.98, +3.3 bps/trade**
  vs 1:1's +52% / 53.9% / Sharpe 1.01 / +2.7 bps. -> 1.5:1 slightly better total.
- **2:1 is WORSE** (win 34.9% vs 33% break-even, Sharpe ~0.05) -- the edge is short-horizon
  and decays at wide targets, so 1.5:1 is the sweet spot.
- **OPTIONS:** 1.5:1 is the best bracket for the deep-ITM weekly calls on the top-decile
  expected-move subset (~+4.2%/trade net at tight spreads vs +2.2% at 1:1). On the FULL
  signal set options still lose at any bracket (theta+spread tax > edge).

## Status (same caveat as v1)
PROMISING, NOT VALIDATED. Choosing 1.5:1 over 1:1 is itself more multiple-testing, so it
inherits the Deflated-Sharpe = 0.114 caveat. The deciding test remains a **fresh-ticker
holdout** (names never used in development). Treat v1 (1:1) as the conservative base and v2
(1.5:1) as the higher-return candidate; validate before trusting either with real money.

## How to run (from the PARENT capture_trader dir, which has data_cache/)
- `python equity_curve.py`     -> 1.5:1 OOS equity curve
- `python triple_barrier_final.py` -> 1.5:1 validation + deflated Sharpe
- `python options_strategy.py` -> 1.5:1 options structure comparison
- `python last_week_test.py`   -> 1.5:1 forward test on the most recent week

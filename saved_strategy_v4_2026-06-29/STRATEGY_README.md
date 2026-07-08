# FROZEN STRATEGY — v4 (saved 2026-06-29): 15-min / 4:1 — HOLDOUT-VALIDATED

Frozen snapshot. Do not edit. The four saved strategies (never edit any of them):
- `saved_strategy_v1_2026-06-28\` = 1:1,   60-min
- `saved_strategy_v2_2026-06-28\` = 1.5:1, 60-min
- `saved_strategy_v3_2026-06-29\` = 1.5:1, 30-min  (validated; most tradeable)
- `saved_strategy_v4_2026-06-29\` = **4:1, 15-min**  (this folder; validated; highest total)

## What it is
Same engine (LightGBM trade-filter / triple-barrier / meta-labeling, base+S/R features,
top-7% selectivity, walk-forward, 3 bps, 8 names). vs v3 the changes are **15-minute candles**
and **4:1** target:stop. Defining config = `validate_grid.py` with args **15 4 1**
(MIN=15, TP=4, SL=1, SEL_Q=0.93, HBAR=24).

## Validation (2026-06-29)
- **Dev (8 names):** 6,033 trades, 27.9% win (break-even 20%), Sharpe ~2.36, +129%,
  +2.1 bps/trade. Dev Deflated-Sharpe=0.07 (multiple-testing fail on dev, as expected).
- **FRESH HOLDOUT (IWM, GLD, META, XOM, KO):** 2,982 trades, **29.9% win, Sharpe ~3.79,
  5/5 folds positive, +150%, +5.0 bps/trade** — edge STRENGTHENED out-of-sample. Passed.

## Why this is NOT the default to trade (read before using)
Despite the highest total, v4 is the **hardest to actually run**:
1. **Win rate ~30%** — you LOSE ~7 of every 10 trades; the edge lives entirely in the 4:1
   winners. Long losing streaks; psychologically brutal.
2. **~2x the turnover** of v3 (6,000+ trades) -> the MOST sensitive to real slippage. The
   +5 bps edge assumes 3 bps cost; at 15-min frequency real cost of 5-10 bps could erase it.
3. Same-era (2021-2026) cross-asset holdout, not a different-time test.

**v3 (30-min/1.5:1) remains the recommended one to actually trade** (47% win = livable,
half the turnover, also validated). v4 is the higher-ceiling/higher-risk alternative.

## How to run (from PARENT capture_trader dir)
- `python validate_grid.py 15 4 1`        -> dev DSR + folds
- `python validate_grid.py 15 4 1 fresh`  -> fresh-ticker holdout

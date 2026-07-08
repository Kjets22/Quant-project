# FROZEN STRATEGY — v3 (saved 2026-06-29): 30-min / 1.5:1 — HOLDOUT-VALIDATED

Frozen snapshot. Do not edit. This is the **best and first holdout-validated** config.
The three saved strategies (never edit any of them):
- `saved_strategy_v1_2026-06-28\` = 1:1,   60-min  (original base)
- `saved_strategy_v2_2026-06-28\` = 1.5:1, 60-min  (tuned ratio)
- `saved_strategy_v3_2026-06-29\` = **1.5:1, 30-min**  (this folder — validated)

## What it is (same engine, finer candle)
Rules-based bracket trade with a **LightGBM classifier as the trade filter** (triple-barrier
/ meta-labeling). Long-only, top ~7% conviction signals, walk-forward, 3 bps cost, 8 names
(SPY, QQQ, AAPL, MSFT, NVDA, JPM, XLE, TLT). vs v1/v2 the ONLY changes: **30-minute candles**
and **1.5:1** target:stop. The defining config lives in `validate_30min.py` and `paper_trade.py`
(MIN = 30, TP, SL = 1.5, 1.0, SEL_Q = 0.93, HBAR = 24).

## Why this is the one (validation, 2026-06-29)
Walk-forward, pooled, 3 bps:
- **Dev (8 names):** 3,129 trades, 47% win (break-even 40%), Sharpe ~1.87, +114%.
  Dev Deflated-Sharpe = 0.06 (fails multiple-testing ON THE DEV NAMES, as expected after ~35 configs).
- **FRESH HOLDOUT (5 names never used in dev: IWM, GLD, META, XOM, KO):** 1,588 trades,
  **47% win, Sharpe ~1.91, 5/5 folds positive, +80%** — nearly identical to dev. Because these
  names had zero influence on any config choice, the matching numbers mean the edge is NOT
  asset-selection overfit. **This is the decisive pass the other versions never got.**

## Honest remaining risks (NOT yet addressed)
1. Still a BACKTEST — assumes 3 bps cost and fills at 30-min bar prices. Edge is thin (+3.7–5.0
   bps/trade); real slippage at this frequency is the #1 threat.
2. SAME ERA — dev and fresh are different instruments but the same 2021–2026 market, not a
   different-time holdout. (Dev's most-recent fold was slightly negative.)
3. Cost-sensitive — lives or dies on cheap execution.
The live **paper trading** (parent: paper_trade.py -> runs/paper_ledger.json, week of 2026-06-29)
exists to attack risks 1 and 2 with real-time forward fills.

## How to run (from PARENT capture_trader dir, which has data_cache/)
- `python validate_30min.py`        -> dev DSR + fold consistency
- `python validate_30min.py fresh`  -> the fresh-ticker holdout
- `python paper_trade.py`           -> update the live paper ledger
- `python today_spy_qqq.py`         -> bar-by-bar SPY/QQQ for the current day

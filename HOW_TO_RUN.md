# capture_trader — how to run

An ML trading strategy: a LightGBM classifier filters ATR-bracket trades on intraday stock
candles. Two validated configs (v3 = 30-min/1.5:1, v4 = 15-min/4:1) run together as an
ensemble (Sharpe ~2.5, +122% backtest, -7% max drawdown). **Research / paper only.**

## Setup
1. Python 3.10+. Install deps:  `pip install pandas numpy lightgbm pyarrow requests`
2. Set your Polygon API key:  `setx POLYGON_API_KEY "your_key_here"`  (then reopen the shell)
3. **Get the data (not in this zip — it's large and re-downloadable):**
   `python fetch_data.py --basket`   (the 8 dev names)
   `python fetch_fresh.py`           (the 5 fresh holdout names)
   This writes 5-min CSVs into `data_cache/`. Everything else reads from there.

## The core files (the actual strategy)
- `triple_barrier_ml.py` — the engine: candle features, ATR, the triple-barrier label.
- `sr_features.py` — the support/resistance features (from the imported design).
- `trials.py` — Deflated Sharpe / multiple-testing math.
- `triple_barrier_breadth.py`, `basket.py`, `config.py`, `data.py` — universe + data plumbing.

## Run the validated strategies
- `python validate_grid.py 30 1.5 1`        → validate v3 (dev, deflated Sharpe + folds)
- `python validate_grid.py 30 1.5 1 fresh`  → v3 on the fresh-ticker holdout
- `python validate_grid.py 15 4 1 fresh`    → v4 on the fresh holdout
- `python ensemble.py`                      → the v3+v4 ensemble (Sharpe, drawdown, curve)
- `python edge_audit.py`                    → embargoed backtest + feature importance (the edge)

## Paper trade (the thing you'd actually use)
- `python paper_ensemble.py` — runs v3+v4 with position sizing + guardrails, prints a blotter
  and P&L, writes `runs/paper_ensemble_ledger.json`.
  - Edit the top: `PAPER_START` = Monday of the week (or a month back), `NOTIONAL_PCT` = % of
    account per trade (default 10), `MAX_POSITIONS`, `DAILY_LOSS_LIMIT`, `DD_BREAKER`.
  - Re-run any time; it re-pulls the latest bars and refreshes.
- See `PAPER_TRADING_GUIDE.md` for the full guardrail/rules writeup.

## Frozen strategy snapshots (never edit — copy to experiment)
- `saved_strategy_v1_2026-06-28/` — 1:1, 60-min (original base)
- `saved_strategy_v2_2026-06-28/` — 1.5:1, 60-min
- `saved_strategy_v3_2026-06-29/` — **1.5:1, 30-min (validated, most tradeable)**
- `saved_strategy_v4_2026-06-29/` — **4:1, 15-min (validated, highest total)**
Each has a `STRATEGY_README.md` with its config + results.

## What's the edge (honest summary)
The model finds entries (unusual volume + volatility regime + location vs key levels) that beat
the bracket's break-even win rate by ~7-10 points; asymmetric brackets turn that into profit;
selectivity (top 7%) concentrates it. It survives an embargoed backtest AND a fresh-ticker
holdout. It is a SMALL, real edge — not a money printer. Buying options on it LOSES (the
vol-risk-premium tax) — trade the shares. See `FINDINGS.md` for the full research record.

## Honest status
Validated in backtest (cross-asset holdout + deflated Sharpe). NOT yet proven live. The one
untested assumption is realistic slippage (backtest uses 3 bps at exact fills). Paper-trade for
1-2 months, size tiny, keep the guardrails, before risking real money. This code does NOT place
trades or move money.

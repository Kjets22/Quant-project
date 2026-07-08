# Cowork brief — capture_trader ML trading strategy

Paste this as your first message in a Cowork session pointed at `C:\Users\kjets\capture_trader`.
Then read the `STRATEGY_README.md` inside the latest `saved_strategy_v*` folders.

---

You are continuing an ML trading-strategy research project. Read this brief fully before acting.

## What the strategy is
A LightGBM classifier used as a TRADE FILTER (triple-barrier / meta-labeling) on intraday stock
candles. For each bar it predicts whether a fixed ATR bracket hits its target before its stop, and
trades only the top ~7% highest-conviction signals. Features: returns/RSI/realized-vol/MA-distance/
range-position/volume + support-resistance distances, all ATR-normalized and causal. It is NOT a
price predictor, NOT reinforcement learning (tried, lost), NOT deep learning.

## What is VALIDATED (passed a fresh-ticker holdout — the gold standard)
- **v3**: 30-min candles, 1.5:1 target:stop — 47% win, Sharpe ~1.9, +114% dev / +80% fresh. Most tradeable.
- **v4**: 15-min candles, 4:1 target:stop — 29% win, Sharpe ~3.8, +129% dev / +150% fresh. Highest total, high variance.
- **NEW, not yet holdout-validated**: TWO-SIDED stock — a LONG model (up) + a SHORT model (down);
  long on up-signals, short on down-signals, in SHARES. Dev 30-min = **+185% combined** (best stock
  result). Validate this first.

## Saved strategy locations (FROZEN snapshots — copy these, never edit in place)
All folders are under `C:\Users\kjets\capture_trader\`:
- `saved_strategy_v1_2026-06-28\` — **1:1, 60-min** (original base; superseded)
- `saved_strategy_v2_2026-06-28\` — **1.5:1, 60-min** (superseded)
- `saved_strategy_v3_2026-06-29\` — **1.5:1, 30-min** (VALIDATED; most tradeable, 47% win)
- `saved_strategy_v4_2026-06-29\` — **4:1, 15-min** (VALIDATED; highest total, 29% win)
Each folder contains the engine `.py` files + `STRATEGY_README.md` (config + results + how-to-run).
To experiment, COPY a folder (or the scripts you need) into a new file — do not modify these. The
two-sided strategy lives in `two_sided_options.py` in the project root (not yet snapshotted).

## What is RULED OUT — do NOT re-explore
- **BUYING options** (calls or puts, any moneyness, ratio, or timeframe). Tested 6+ ways including
  real SPY quotes and two-sided call/put. The volatility-risk-premium (implied vol > realized) + bid/
  ask spread is a structural tax larger than the directional edge. Options only "win" in lucky months
  (variance), never long-run. The only data-supported options angle is SELLING premium — a SEPARATE
  strategy with its own (tail) risk, not a wrapper on this one.
- RL, pairs trading, naive timing — all lost.

## Validation discipline (NON-NEGOTIABLE — this is what makes results trustworthy)
1. Walk-forward, non-overlapping trades, 3 bps cost, pooled across the 8-name basket.
2. Deflated Sharpe (trials.py) to penalize multiple testing. Dev DSR is usually LOW (~50 configs tried) — expected.
3. THE decisive test = **FRESH-TICKER HOLDOUT**: freeze a config, run it on names never used in dev
   (IWM, GLD, META, XOM, KO). `python validate_grid.py <minutes> <tp> <sl> fresh`. If win%/Sharpe hold there, real.
4. Every new "improvement" must pass the fresh holdout before you call it validated. Dev numbers alone mean nothing.

## Data
- `data_cache/{TICKER}_5minute_2021-06-01_2026-06-01.csv` — 5-min bars. Dev: SPY QQQ AAPL MSFT NVDA
  JPM XLE TLT. Fresh holdout: IWM GLD META XOM KO. Resample to any timeframe.
- `data_cache/options/chain_SPY_*.parquet` — real SPY option chain (bid/ask/iv/delta, daily, 2025-04..2026-06). SPY-only.
- Polygon key in env `POLYGON_API_KEY` for fetching more (fetch_data.py / fetch_fresh.py).

## Key files
- Engine: `triple_barrier_ml.py` (features/atr/label), `sr_features.py`, `trials.py` (deflated Sharpe).
- Validators/experiments: `validate_grid.py` (use for everything), `timeframe_test.py`,
  `sweep_tf_ratio.py` (the timeframe x ratio heatmap), `two_sided_options.py` (long/short + options),
  `flexible_profit.py` (trailing-stop exit), `paper_trade.py` (live paper ledger).
- `saved_strategy_v1..v4/` — FROZEN snapshots. NEVER edit. Copy to a new file for any experiment.

## Optimization goals (priority order)
1. **Holdout-validate the two-sided stock strategy** (long+short). Add realistic short-borrow cost. If it holds, it's the new best.
2. **Diversified ensemble** (v3 + v4, or two-sided): measure if combined drawdown/Sharpe beats each alone.
3. **Position sizing / risk** — volatility targeting, concurrent-position caps (conviction-weighting was marginal).
4. **Regime filter** — the most recent dev fold was weak; can a regime feature avoid the bad stretches?

## Guardrails (hard rules)
- Never edit `saved_strategy_v*`. Copy + new file.
- Never claim "validated" without a fresh-ticker holdout pass.
- Prefer principled changes (sizing, risk, ensemble) over parameter/feature hunting — every knob raises the overfitting penalty.
- This is research. Do NOT place trades or move money. Paper-trade first; size tiny; respect the drawdown circuit-breaker.
- Report negative results honestly — a finding that saves money is a win.

Start: read the v3/v4 READMEs, run `python validate_grid.py 30 1.5 1 fresh` to confirm the baseline,
then validate the two-sided strategy on the fresh names.

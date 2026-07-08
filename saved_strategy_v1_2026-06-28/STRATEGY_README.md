# FROZEN STRATEGY — v1 (saved 2026-06-28)

This folder is a **frozen snapshot**. Do not edit these files. All tuning/experiments
happen in the parent project; this is the protected "known-good" copy.

## What the strategy IS (plain English)
A **rules-based bracket trade with a machine-learning gatekeeper**. It is ML — specifically
a **LightGBM gradient-boosted-tree classifier** used as a *trade filter* (the "meta-labeling"
method), NOT a neural net, NOT reinforcement learning (we tried RL; it lost), and NOT a
price predictor.

How it works, step by step:
1. **Setup (fixed rules):** on hourly bars, every bar is a candidate LONG. Bracket =
   target +1 ATR, stop -1 ATR, time limit 24 bars. ATR (Average True Range) auto-scales
   target/stop to each name's volatility.
2. **Labels (triple-barrier):** for each past bar, record which barrier hit first —
   target (1) or stop (0).
3. **Features (~22, all causal/past-only):** returns 1/6/24h, RSI, realized vol, distance
   from 20/50 moving averages, range position, volume z-score, **plus support/resistance
   features** (distance to recent highs/lows, prior-day H/L/C, round numbers, VWAP), all
   ATR-normalized. The S/R features are what first added real signal.
4. **ML:** LightGBM predicts P(target before stop). It decides *which* mechanical setups
   are worth taking — it does not forecast price.
5. **Selectivity:** trade only the top ~7% highest-confidence signals (threshold set on
   TRAIN data only). Being picky is what pushed win-rate above break-even.
6. **Validation:** walk-forward (train past → test future, 5 folds), pooled across 8 names,
   charged realistic costs.

## Frozen config
- Instruments: SPY, QQQ, AAPL, MSFT, NVDA, JPM, XLE, TLT (hourly bars)
- Bracket: **1:1** (target = stop = 1 ATR), time barrier H = 24
- Features: base candle features + S/R features
- Selectivity: top 7% (SEL_Q = 0.93), threshold from train only
- Cost: 3 bps round-trip on the underlying
- Model: LightGBM (300 trees, lr 0.03, num_leaves 15, min_child 40, subsample/colsample 0.8)

## Results (out-of-sample, walk-forward)
- Total return +51.5% over ~37 months, 53.9% win, Sharpe ~1.0, max drawdown -26%
- 68% positive months (25/37); 7/8 names positive
- Forward test (week of 2026-06-22): 7/10 closed trades won (+0.89%) — tiny sample
- **Deflated Sharpe = 0.114** → PROMISING, NOT YET VALIDATED (after ~25 configs tried,
  not statistically distinguishable from the luckiest attempt). Decisive test still
  pending: a fresh-ticker holdout (names never used in development).

## Options finding
Buying options on ALL signals LOSES (theta + bid/ask tax ≈ 130+ bps/trade > the ~9 bps
edge). Options only turn net-positive on the **top-decile expected-move** signals using
**deep-ITM (~0.95 strike) weekly calls** exited at the underlying bracket.

## Candidate improvement (NOT in this frozen copy — validate before adopting)
Tuning study (optimize_tpsl.py in parent) found **1.5:1 target:stop** slightly beats 1:1
for stock (+61% vs +52%, Sharpe ~0.98) and is the best options bracket (top-decile deep-ITM
weekly ≈ +3–4%/trade net). **2:1 is worse** (win-rate collapses to ~35%, barely above the
33% break-even). This frozen snapshot stays 1:1; 1.5:1 is a candidate pending fresh-ticker
validation.

## How to run (from the PARENT project dir, which has data_cache/)
- `python triple_barrier_final.py`  → final validation + deflated Sharpe
- `python equity_curve.py`          → OOS equity curve → runs/equity_curve.json
- `python last_week_test.py`        → fresh forward test on the most recent week
- `python options_strategy.py`      → options structure comparison

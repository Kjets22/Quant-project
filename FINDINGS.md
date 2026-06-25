# Capture Trader — Research Findings (honest writeup)

**Goal:** train an RL agent that beats buy-and-hold (B&H) out-of-sample on liquid
US equity instruments (QQQ, SPY, TLT, ...).

## TL;DR — the honest conclusion

**There is no reliable predictive edge to be had timing these instruments on
5-minute–to-hourly bars.** We proved this directly (not by giving up): short-horizon
returns are ~unpredictable, and transaction costs make active trading a losing game
by default. The one thing that genuinely works is **participation, not prediction** —
ride trends, sit out chop — which delivers a **better risk-adjusted return** (Sharpe,
drawdown) than B&H, but does **not** reliably beat B&H on absolute return in a bull
market. That risk-adjusted improvement is the real, defensible deliverable. Nothing
here is a proven money-maker; paper-trade before risking capital.

## Why it can't beat B&H (the diagnosis — `diagnose.py`, `diagnose_tf.py`)

1. **Returns are ~unpredictable at every timeframe tested.** An honest out-of-sample
   linear predictor gives QQQ **R² ≈ +0.0005** and **direction accuracy 50.2%** (a
   coin flip); 1h ≈ 51%, 1day ≈ 46%. TLT is the same. There is no linear signal to
   exploit — textbook efficient-market behavior.
2. **Transaction costs are brutal at short horizons.** A round-trip eats **37–45% of
   a typical bar move** at 5-min. With no edge, every trade is a coin flip that pays
   the spread. Momentum/mean-reversion rules lose **thousands** of dollars to costs.
3. **"Do nothing" beats every active strategy** on a losing asset, and B&H beats
   "do nothing" on a rising one — the squeeze that makes an absolute edge nearly
   impossible here.

## What actually works (`trend_test.py`)

A simple **trend filter — long when above its moving average, FLAT otherwise** —
beats B&H on a **risk-adjusted** basis:

| | B&H | trend long/flat | note |
|---|--:|--:|---|
| QQQ 1h (SMA50) | Sharpe 0.98 | **Sharpe 2.08** | ~2× risk-adjusted |
| QQQ 1day (SMA200) | +170 | **+189**, 7 trades | beats on P&L too |
| TLT 1day (SMA20) | −57 | **−17** | cuts loss ~70% |

The edge is **trend participation + drawdown avoidance**, only at hourly/daily (where
costs are negligible), long-or-flat, very low turnover.

## Best RL config (`run_cross.py`)

**all-weather + cross-asset**, hourly, money reward + turnover penalty:
- force **long in confirmed up-trends**, regime-**gated short** in down-trends, flat in chop
- cross-asset (QQQ↔SPY) features: each sees the partner's return/momentum/trend/spread

Out-of-sample (dev):

| | Sharpe | agent P&L | B&H | beat-B&H |
|---|--:|--:|--:|--:|
| QQQ | **+0.73** | +32.5 | +25.9 | 50% |
| SPY | −0.10 | −0.3 | +19.9 | 33% |

QQQ marginally beats B&H; **SPY does not generalize** — the inconsistency between two
near-identical assets is the tell that this is at the edge of noise, not a robust edge.

## What was tried and DISCARDED (the full scoreboard)

| Approach | Verdict |
|---|---|
| Capture (oracle-normalized) reward | ❌ −1.42 Sharpe, 19% beat-B&H |
| Position-awareness features | ❌ discard |
| Risk-aware reward (diff-Sharpe/DD/vol) | ❌ worse (weights too large) |
| AlphaTrend features (signals, then datapoints-only) | ❌ discard standalone |
| Regime features (Phase C) | ➖ marginal (steadier only) |
| Tuned single policy (Optuna) | ❌ "Sharpe 1.0" was a flat-agent **mirage** |
| Regime mixture-of-experts (±hysteresis, ±AlphaTrend) | ❌ overtrades, −$170 to −$630 |
| Money reward (raw $) | ➕ first **positive** Sharpe, still loses to B&H |
| Hourly long/flat trend RL | ➕ beats B&H risk-adjusted on TLT |
| **all-weather + cross-asset** | ✅ **best**; marginal QQQ beat |
| Support/resistance features | ❌ **degraded** both (overfitting) |

Two independent results — SPY failing where QQQ won, and S/R features *hurting* — both
confirm: **the bottleneck is signal, not features.** Adding capacity to a no-signal
problem overfits and makes OOS worse.

### Later attempts (QQQ/SPY) — all confirm the boundary

| Idea | Result |
|---|---|
| Daily horizon (longer holds) | ❌ worse on bull indices; works only on falling TLT; RL ≈ simple SMA rule |
| **Daily + support/resistance** | ➕ S/R *helps at daily* (QQQ Sharpe +0.39→+0.79, marginal B&H beat) though it *hurt* at hourly — S/R is timeframe-dependent; still SPY loses, tiny data |
| Swing (heavy turnover penalty, ~43h holds) | ➖ neutral — steadier, not better |
| Recurrent LSTM + higher LR | ❌ worse than MLP (overfit) |
| Tuned hyperparams (256x256, etc.) | ➖ no robust gain |
| **Stacked error-predictor** (predict where base is wrong, gate it) | ❌ **meta-model AUC 0.505 (coin flip)** — base errors are unpredictable OOS; gating made it *worse*. Confirms "50%+50%→75%" fails: the meta-model has no real predictive power because a trade is "wrong" only due to the unpredictable next move. |
| Time-of-day / time-since-open feature | ➖ no effect (AUC stayed 0.50) |

**The deepest confirmation:** even a meta-learner *cannot* predict the base model's
mistakes (AUC 0.50), because the mistakes are driven by irreducibly random price
direction. No reward, feature, model, horizon, or stacking escapes an efficient market.

### The market-neutral pivot — QQQ/SPY statistical arbitrage (the closest to a real edge)

Directional timing of a single bull-market index can't beat B&H (B&H captures the
whole uptrend cost-free). A **market-neutral spread** doesn't compete with B&H at all,
so we tested QQQ-SPY pairs mean-reversion (`pairs.py`, `pairs_validate.py`).

- **First genuine GROSS signal in the whole project:** the 30-min spread mean-reverts
  (gross Sharpe ~+1.0 OOS), unlike directional timing which was 50% even gross.
- **Single 60/40 split looked great:** net Sharpe **+0.83** OOS after realistic
  ultra-liquid-ETF costs (0.2 bps/leg), market-neutral (corr 0.16).
- **But rigorous validation killed it.** Walk-forward (6 OOS blocks): pooled annualized
  Sharpe only **+0.39** (0.2 bps) → **+0.07** (0.5 bps), just **4/6 blocks positive**
  (block 1 strongly negative), and the **Deflated Sharpe ≈ 0** — it does NOT survive
  multiple-testing correction. ADF p≈0.09 (not formally cointegrated; relationship drifts).
- **Why thin:** QQQ and SPY overlap ~85% — the spread is too small for its
  mean-reversion to exceed harvesting costs. Real stat-arb needs a wider-spread
  cointegrated pair (competitors / sector-vs-member), where the deviation dwarfs cost.

**Lesson (textbook, from the guide):** an impressive single backtest → walk-forward +
deflated Sharpe → exposed as not robust. The validation discipline is what separates a
real edge from a fluke, and here it correctly returned **no validated edge** — while
pointing at where a real one might live (wider-spread, less-correlated cointegrated pairs).

## Engineering notes worth keeping

- **GPU is slower here** (3.7×) — PPO + tiny MLP is CPU/rollout-bound; parallelize
  across CPU processes instead.
- **Lookahead wall held throughout** — every feature is causal (verified with
  future-scramble tests); oracle/future info touches only the reward.
- **Per-fold checkpointing** makes runs resumable across the (frequent) power losses.
- The **lockbox** was opened once on an earlier all-weather config (no proven edge) and
  is now spent — any new config needs a **fresh** holdout, not a second peek.

## Honest expectations

- Beating B&H on a bull-market index by timing is ~impossible; it captures the full
  uptrend cost-free. The realistic value of this work is **risk-adjusted** (smoother
  equity, lower drawdown), not higher absolute return.
- Short-run / single-asset numbers mean little. Watch OOS vs dev; a result that doesn't
  generalize across two near-identical assets is noise.
- This is a research scaffold. **Do not trade real capital on it.** Paper-trade a frozen
  config on fresh data for a meaningful window first.

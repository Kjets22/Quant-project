# Quant Project — ML bracket-trading research system

A complete machine-learning trading research project: data pipeline, feature
engineering, model training, a brutal validation gauntlet, five surviving
strategies, and a multi-account paper-trading simulator with enforced risk
guardrails.

> ⚠️ **Research / paper trading only. Not financial advice.** Nothing here
> places real orders. Backtests — even carefully audited ones — can break in
> live markets. Never trade money you cannot afford to lose.

---

## 1. What this project is

The goal was to find a **genuine, validated, profitable trading edge** on
liquid US stocks/ETFs — and to be honest enough to throw away everything that
only *looked* profitable. Roughly **40+ ideas were tested; ~35 failed** the
validation gauntlet (options buying, reinforcement learning, short-selling,
learned exit models, trailing stops, signal smoothing, wide targets with tight
stops, ...). What survived is a family of five long-only bracket strategies
driven by one shared ML engine.

**The core idea in one paragraph:** on intraday candles (15/30/60-minute),
every bar is a *candidate* long trade with a mechanical bracket — a profit
target above, a stop-loss below, and a time limit. A LightGBM classifier is
trained on history to predict which candidates hit the target first
(triple-barrier labeling / meta-labeling). Live, the system only takes the
**top ~7% highest-conviction** signals. The edge is small (a few basis points
per trade) but real: it survived walk-forward testing, an embargo audit,
realistic costs, and — decisively — a **fresh-ticker holdout** on five names
never used during development.

---

## 2. The strategies (the survivors)

| Name | Bars | Bracket | Stop type | Hold limit | Character | Backtest (dev / fresh) |
|------|------|---------|-----------|-----------|-----------|------------------------|
| **v3** | 30-min | 1.5 : 1 | 1 ATR | 24 bars | Mean-reversion core, ~47% win | +114% / +80%, Sharpe ~1.9 |
| **v4** | 15-min | 4 : 1 | 1 ATR | 24 bars | Big-payoff mean reversion, ~29% win | +129% / +150% |
| **v6** | 60-min | 7 : 1 | 1 ATR | 96 bars | Trend hunter (experimental) | +35% / +7% |
| **v7** | 60-min | 10 : 1 on actual risk | **structure** (0.25 ATR below 20-bar swing low) | 96 bars | Validated wide hunter, ~12% win | +67% / +32%, Sharpe ~0.5 |
| **vC** | 60-min | 30 : 3 ATR | 3 ATR | 96 bars | Trend-holder; profit comes from time exits | +81% / +78%, Sharpe ~1.0 |
| v5 | — | — | — | — | **Time-of-day filter** upgrade for v3/v4 (drops pre-market; tripled v4's per-trade edge) | — |

- *dev* = the 8-name development basket (SPY, QQQ, AAPL, MSFT, NVDA, JPM, XLE, TLT).
- *fresh* = the 5-name holdout never touched during development (IWM, GLD, META, XOM, KO).
- Frozen, never-edited copies of each live in `saved_strategy_v*/` with their own READMEs.

**The two mean-reverters (v3/v4) buy dips; the three trend configs (v6/v7/vC)
buy strength and hold for days.** Their monthly returns are nearly
uncorrelated (~+0.11), which is why running them together smooths the equity
curve (50/50 v3+v4 blend: Sharpe 2.47, max drawdown −7% vs −12%/−21% alone).

---

## 3. How the engine works (data → trade)

1. **Data** — 5 years of 5-minute OHLCV bars per ticker from Polygon.io,
   cached in `data_cache/` (git-ignored; ~200 MB). `fetch_data.py` /
   `fetch_fresh.py` download them once.
2. **Resample** — 5-min bars are stacked into 15/30/60-minute candles
   (high=max, low=min, close=last, volume=sum).
3. **Features (~22–26 per candle, all causal)** — momentum (1/6/24-bar
   returns, RSI, distance from 20/50-bar means), volatility (return std, ATR%),
   volume z-score, **support/resistance distances** (recent highs/lows,
   prior-day H/L/C, round numbers, VWAP) and, for trend configs, 100-bar
   momentum/breakout distance/200-bar trend/volatility expansion. Everything
   is ATR-normalized so all tickers share one scale, and every rolling window
   is `shift(1)`-ed — **no feature ever sees the future**.
4. **Label (triple barrier)** — for each historical candle, walk forward:
   did the high touch the target before the low touched the stop (within the
   time limit)? → 1/0. Ties go to the stop (conservative).
5. **Model** — LightGBM (300 shallow, heavily regularized trees) outputs
   P(target first). Regularization is deliberate: the signal is faint, and
   overfitting is the main enemy.
6. **Selectivity** — only bars whose probability clears the **93rd percentile
   of the training distribution** (top ~7%) become trades. The threshold is
   computed on training data only.
7. **Trade simulation** — enter at candle close, exit at target/stop/clock.
   Non-overlapping per ticker. Time exits settle at the **actual close**
   (a subtle but critical accounting rule — see §5).
8. **Costs** — 3 bps round-trip + 1 bp/side slippage (5 bps effective) on
   every trade.

**What the edge actually is (proven, not assumed):** random entries with the
identical brackets earn exactly zero (margin −0.9%/+0.3% vs break-even), so
100% of the profit comes from *entry timing*. Feature-knockout tests show the
signal lives jointly in **location (support/resistance) + momentum + volume**;
the fingerprint of chosen entries is an **oversold dip sitting just above
support with room overhead** — i.e., short-horizon mean reversion, harvested
selectively with asymmetric payoffs. For the trend configs, the same engine
selects breakout/expansion setups and the structure stop (below the swing low)
lets multi-day legs breathe.

---

## 4. The validation gauntlet (why these numbers are believable)

Every strategy had to survive, in order:

1. **Walk-forward evaluation** — 5 expanding folds; train on the past, trade
   the future. Every reported trade is out-of-sample.
2. **Embargo** — the last `HBAR` training bars are purged so forward-looking
   labels can't leak into the test fold (this audit found and fixed the one
   real leak in the pipeline; the edge survived).
3. **Full costs** — 5 bps effective per trade.
4. **Corrected accounting** — time exits at the actual close, wins counted
   only on true target hits. (An earlier version credited drift-up time exits
   as full wins — it manufactured a fake "+518%" at 10:1 that the fix erased.)
5. **Deflated Sharpe / multiple-testing control** — after ~50 experiments the
   best result looks good by luck; the DSR (in `trials.py`) prices that in.
6. **Fresh-ticker holdout (decisive)** — the frozen config runs on 5 names it
   has never seen. Pass = real; fail = overfit. This step killed the
   two-sided short (+185% dev → −74% fresh), the 25:2.5 wide cell (+140% dev →
   −20% fresh), and many others.
7. **Forensics** — concentration (drop-top-3 winners), per-year and per-ticker
   consistency.
8. **Paper trading** — the live forward test (see §6).

### The graveyard (tested and rejected — don't re-explore without new evidence)
- **Buying options** (calls or puts, any moneyness/ratio/timeframe): the
  volatility-risk-premium + spread is a structural tax ~10–30× the per-trade
  edge. Confirmed with real SPY chain quotes.
- **Short selling** (two-sided): the short model's dev edge did not
  generalize — long-only survives.
- **Learned early-exit models**: a loser only becomes identifiable *after* the
  price has dropped; even a 100%-precision exit model adds nothing. The plain
  stop is optimal.
- **Trailing stops**: give back more than they save on these horizons.
- **Signal smoothing/persistence**: improved dev, inverted on fresh — mirage.
- **Wide targets (5–10:1) with 1-ATR stops on 24-bar clocks**: noise kills the
  trade before the trend can develop (fixed by v7's structure stop + 96-bar
  clock).
- **RL (PPO/RecurrentPPO), pairs trading, hyperparameter/ensemble tuning,
  conviction-weighted sizing, option-market features**: no measurable gain.

---

## 5. Results snapshot (audited environment)

Full-period walk-forward, 8-name basket, 5 bps effective cost, corrected
accounting, double exposure allowed:

- **v3+v4 pooled portfolio:** +85% over 38 months, monthly Sharpe 0.85,
  61% positive months.
- **v3/v4 50-50 blend (3 bps era, earlier audit):** Sharpe 2.47, maxDD −7%.
- **vC forensics:** profits are *not* concentrated (drop-top-3 still +55%),
  positive in 4/5 calendar years and 7/8 tickers; the P&L engine is profitable
  time exits (median hold ~2.5 days), i.e., a genuine trend-holder.

See `FINDINGS.md` for the full research log and `runs/` for raw outputs.

---

## 6. Paper trading (the live forward test)

`paper_ensemble_v4.py` runs **five independent $10,000 paper accounts** (v3,
v4, v6, v7, vC), $1,000 per trade, max 10 concurrent positions each, with
**enforced** guardrails:

- position cap and one-position-per-ticker (within each account)
- daily loss limit (2% → stop opening new trades that day)
- drawdown circuit-breaker (15% → halt the account)
- 0.12% ATR floor (skips noise trades), 5 bps effective cost

Daily workflow:

```bash
python paper_ensemble_v4.py 2026-06-29 2026-07-10   # bump end date as days pass
python sharpe_paper.py                              # per-account Sharpe from the ledger
```

It re-fetches the latest bars, retrains frozen models on data *before* the
window, replays the window, prints per-account blotters + end-of-day P&L +
scoreboard, and writes `runs/paper_v4_ledger.json`.

**Reading tip:** judge the trend accounts (v7/vC) by **account value including
unrealized P&L** — they cut losers in hours but hold winners for days/weeks, so
early on their *realized* Sharpe is negative by construction while their open
winners grow. Judge everything over months, never days.

Broker note: thinkorswim paperMoney has **no API**; live execution would go
through the Schwab Trader API (Individual) — connector not built yet, and by
design this codebase never places real orders.

---

## 7. Repository map

### Engine (shared by everything)
| File | Role |
|------|------|
| `triple_barrier_ml.py` | Base features, ATR, triple-barrier labeler |
| `sr_features.py` | Support/resistance feature block |
| `wide_hunter.py` | Trend features (`t_mom100`, `t_brk100`, `t_sma200`, `t_volexp`), no-lookahead ATR, wide-target engine |
| `trials.py` | Deflated Sharpe / multiple-testing math |
| `data.py`, `config.py`, `basket.py`, `fetch_data.py`, `fetch_fresh.py` | Polygon fetch + cache + config |

### Strategy validation / research
| File | Role |
|------|------|
| `validate_grid.py` | Generic validator: any timeframe × ratio, dev or fresh |
| `edge_audit.py` | Embargo audit + feature importance |
| `edge_proof.py` | Random-entry baseline, feature knockouts, signal fingerprint, portfolio Sharpe |
| `exp_ten.py` | Structure-stop / wide-stop 10:1 experiments (v7, vC cells) |
| `research_c.py` | vC forensics (concentration, per-year/ticker, holds) |
| `optimal_times.py` | Time-of-day filter (v5) |
| `timeframe_test.py`, `sweep_tf_ratio.py`, `wide_sweep2.py` | Timeframe × ratio sweeps (corrected accounting) |
| `ensemble.py` | v3+v4 blend analysis |
| `noise_test.py`, `vt_test.py`, `exit_model*.py`, `two_sided_options.py`, `flexible_profit.py`, `ml_accuracy.py`, `make_better.py` | Experiments (mostly graveyard — kept for the record) |
| `options_*.py`, `real_options_spy.py`, `directional_options.py` | The options investigation (all negative) |

### Paper trading
| File | Role |
|------|------|
| `paper_ensemble_v4.py` | **Current**: 5 independent accounts |
| `paper_ensemble_v3.py` | Shared-account version (candidate generator lives here) |
| `paper_ensemble_v2.py`, `paper_ensemble.py`, `paper_trade.py` | Earlier iterations |
| `sharpe_paper.py` | Per-account Sharpe from the ledger |

### Frozen strategy snapshots (never edit — copy to experiment)
`saved_strategy_v1..v7*/` — each contains the engine files as-of-freeze plus a
`STRATEGY_README.md` with config, results, and how to run.

### Docs
`FINDINGS.md` (research log/capstone), `PAPER_TRADING_GUIDE.md`,
`HOW_TO_RUN.md`, `COWORK_PROMPT.md` (hand-off brief for other agents).

---

## 8. Setup

```bash
pip install numpy pandas lightgbm requests truststore python-dotenv
echo POLYGON_API_KEY=your_key_here > .env        # never commit this
python fetch_data.py --basket --tickers SPY,QQQ,AAPL,MSFT,NVDA,JPM,XLE,TLT
python fetch_fresh.py                            # the 5 holdout names
python validate_grid.py 30 1.5 1                 # reproduce v3 (dev)
python validate_grid.py 30 1.5 1 fresh           # reproduce v3 (holdout)
```

`data_cache/`, `.env`, and model artifacts are git-ignored; the repo carries
code + results, not data or secrets.

---

## 9. House rules

1. **Frozen snapshots are read-only.** To experiment, copy into a new file.
2. **Nothing is "validated" without a fresh-ticker holdout pass.** Dev numbers
   alone mean nothing — the graveyard is full of dev heroes.
3. **No future information, ever**: causal features, embargoed training,
   train-only thresholds, corrected time-exit accounting.
4. **Report failures honestly** — a negative result that saves money is a win.
5. **This code never places real orders.** Guardrails stay on. Paper first,
   tiny size later, and only after months of live evidence.

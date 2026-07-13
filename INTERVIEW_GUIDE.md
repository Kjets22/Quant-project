# Interview Study Guide — Autonomous ML Trading System

This document teaches the entire project: what was built, how every piece works, **why each
design choice was made**, what failed and what that taught us, and the numbers worth
memorizing. Read it top to bottom twice and you can defend this project in a technical
interview without notes.

---

## 1. The 30-second pitch (memorize this)

> "I built an end-to-end ML trading system: it ingests five years of intraday price data,
> engineers causal features, and uses LightGBM with triple-barrier labeling to select the
> top 7% highest-conviction bracket trades. The core of the project is the validation
> framework — walk-forward testing with embargoes, deflated Sharpe ratios, and a
> never-touched holdout universe of tickers — which rejected about 35 of my 40 strategy
> ideas and validated 5. Those 5 now run as a fully autonomous bot on a brokerage paper
> account with code-enforced risk guardrails, plus a live A/B experiment on execution
> style. The headline lesson: the model is maybe 20% of the work — honest validation and
> execution realism are the other 80%."

---

## 2. The trade itself — what the system actually does

Everything rests on one primitive: the **bracket trade**.

- On intraday candles (15/30/60-minute), every bar is a *candidate* long entry.
- At entry, three exits are pre-committed: a **profit target** above, a **stop-loss**
  below, and a **time limit** (if neither is hit in H bars, exit at market).
- Targets and stops are sized in **ATR** (Average True Range — the average bar size over
  the last 24 bars), so a "1.5:1 bracket" means target = entry + 1.5×ATR, stop = entry −
  1×ATR, *on every stock regardless of its price or volatility*.

**WHY brackets?** Two reasons. (1) Every trade has an objective, pre-committed outcome —
no discretion, no hindsight, which makes historical outcomes *labelable* for ML.
(2) Risk is capped structurally: the max loss is known before entry.

**WHY ATR units?** A $2 move is noise on NVDA and an earthquake on KO. ATR-normalization
puts every ticker on the same scale, which lets ONE model train across all tickers
(more data → less overfitting) and makes brackets self-adjust to volatility regimes.

### The critical math: break-even win rate

A bracket with target T and stop S needs a win rate above **S/(S+T)** to break even:
- 1:1 → 50% | 1.5:1 → 40% | 4:1 → 20% | 10:1 → 9.1%

**The most important insight of the project:** the ratio itself earns NOTHING. I proved
this directly — random entries with the same brackets land exactly at break-even minus
costs (measured margin: −0.9% and +0.3% vs the line). A wider target pays more per win
but lowers your win rate by exactly the offsetting amount. **All profit comes from the
entry model pushing the win rate a few points above that line.** Our validated margins
are +7 to +10 percentage points above break-even.

---

## 3. The pipeline, stage by stage (with the WHY at each stage)

### Stage 1 — Data
Five years (2021-06 → 2026-06) of 5-minute OHLCV bars from Polygon.io (consolidated,
all-exchange), cached locally as CSV. Dev universe: SPY, QQQ, AAPL, MSFT, NVDA, JPM,
XLE, TLT (8 liquid names across sectors/asset types). Holdout universe (never touched
during development): IWM, GLD, META, XOM, KO.

**WHY 5-minute base data?** Resample once, build any timeframe (15/30/60-min) from the
same source of truth. **WHY liquid names?** The strategies assume ~5bps round-trip cost;
that's only real in the most liquid instruments.

### Stage 2 — Resampling
5-min bars → strategy timeframe: high=max, low=min, close=last, volume=sum. The
still-forming bar is dropped — the system only ever acts on **completed** bars.

### Stage 3 — Features (26, all causal)
Four families, every one ATR-normalized:
- **Momentum**: log returns over 1/6/24 bars, RSI-14, z-distance from 20- and 50-bar
  means (`sma20d`, `sma50d`).
- **Volatility**: 24-bar return std (`vol24`), ATR as % of price (`atrpct`).
- **Volume**: volume z-score vs its 48-bar distribution (`volz`).
- **Location (support/resistance)**: distance to 20/60-bar highs and lows, prior-day
  high/low/close, nearest $5 round number, 120-bar VWAP, position in the 60-bar range.
- **Trend add-ons (wide-target configs only)**: 100-bar momentum, distance to the
  100-bar high (breakout proximity), 200-bar trend z-score, volatility expansion
  (ATR12/ATR96).

**WHY "causal" is non-negotiable:** every rolling window that could peek forward is
`shift(1)`-ed — a feature computed at bar *t* uses only bars ≤ *t*. One leaked feature
makes the backtest fiction. (Interview line: "I treat look-ahead bias as the null
hypothesis — any great result is leakage until proven otherwise.")

### Stage 4 — Labels: the triple-barrier method (López de Prado)
For each historical bar, simulate its bracket forward: did the high touch the target
before the low touched the stop, within H bars? Label 1 (win) / 0 (loss). **Ties go to
the stop** — when a single bar spans both barriers, we can't know intra-bar order, so we
take the pessimistic outcome.

**WHY triple-barrier instead of "predict price in N bars"?** Because it labels **the
exact thing we will trade** — outcome of this bracket — including path dependency (a
trade that eventually rises but first dips through the stop is correctly labeled a LOSS).
Fixed-horizon return labels ignore the path and overstate performance.

This is **meta-labeling**: the ML doesn't forecast prices; it predicts whether a
*mechanical trade* will succeed. Much easier learning problem, directly actionable.

### Stage 5 — Model: LightGBM, deliberately small
`LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
min_child_samples=40, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0)`

**WHY gradient-boosted trees, not deep learning?**
- Tabular data with ~26 features and tens of thousands of rows is exactly where GBDTs
  dominate; LSTMs/transformers need far more data and overfit faint signals.
- Trees capture feature interactions ("RSI low matters only near support") natively.
- Fast to retrain daily; interpretable via gain-based feature importance.

**WHY every hyperparameter:** all of them are anti-overfitting choices. Shallow trees
(15 leaves) can't memorize noise; min 40 samples per leaf kills flukes; row/column
subsampling adds bagging robustness; slow learning rate over 300 stages generalizes
better than few big steps. The config assumes the signal is weak — because it is.
(Tested alternatives: deeper/tuned models and 3-seed ensembles gave **zero** win-rate
improvement — the signal is feature-capped, not model-capped.)

### Stage 6 — Selectivity: trade only the top 7%
The model outputs P(target-first) per bar. We compute the **93rd percentile of
training-set probabilities** and only trade bars above it.

**WHY:** most bars are coin flips; trading them pays costs for nothing. The edge is
concentrated in the extreme tail of conviction. **WHY the threshold comes from training
data only:** using test-period quantiles would leak information ("choose the cutoff that
made the test look good").

### Stage 7 — Portfolio & risk (enforced in code, not vibes)
$1,000 per position (whole shares), max 10 concurrent, one position per ticker, daily
loss stop (halt new entries), drawdown circuit-breaker (halt everything), minimum-ATR
floor (skip dead-quiet names where costs exceed the possible move).

---

## 4. The validation gauntlet — the heart of the project

Run in this order; a strategy is only "real" after ALL of them:

1. **Walk-forward evaluation.** 5 expanding folds: train on the past, trade the next
   chunk, expand, repeat. Every reported trade is out-of-sample. (Simple train/test
   splits let you accidentally tune on the test set through repeated experiments.)
2. **Embargo (purging).** A bar's label looks up to H bars *forward* — labels of the
   last H training bars overlap the test period. Fix: drop the last H bars from every
   training window. (I found this leak by audit; fixing it cost ~0.3pp of margin —
   the edge survived, which itself was evidence it was real.)
3. **Full costs.** 3 bps commission/spread + 2 bps slippage on every trade. A strategy
   that only works at zero cost is a mirage.
4. **Corrected accounting.** Time-limit exits settle at the *actual* closing price, and
   "win" counts only true target hits. (See War Story #1 — the biggest fake result of
   the project came from sloppy time-exit accounting.)
5. **Deflated Sharpe Ratio (Bailey & López de Prado).** After N experiments, the best
   backtest looks good by luck alone. DSR computes the expected maximum Sharpe under
   the null of zero skill given N trials and asks whether yours beats it. Our dev-set
   DSRs were 0.06–0.11 — correctly flagging "could be selection bias" after ~40 trials.
6. **Fresh-ticker holdout — the decisive test.** Freeze the strategy completely, run it
   on 5 tickers that never influenced any decision. Selection bias cannot survive this.
   Pass = real edge; fail = overfit. This test **killed** strategies showing +140% and
   +185% on dev data.
7. **Live paper trading.** Real broker fills, real spreads, real timing — the only test
   of execution assumptions.

**Interview line:** "My dev-set numbers failed the deflated Sharpe test — as they should,
after 40 experiments. That's exactly why the holdout universe existed. The strategies
that passed *both* dev and fresh, with consistent win rates and per-trade economics
across universes, are the ones I trust."

---

## 5. The five validated strategies

| | Bars | Bracket | Stop | Hold | Win% (BE) | Fresh-holdout result |
|--|------|---------|------|------|-----------|----------------------|
| v3 | 30-min | 1.5:1 | 1 ATR | 24 bars | 47% (40%) | +80–92%, ~+6 bps/trade |
| v4 | 15-min | 4:1 | 1 ATR | 24 bars | 26–29% (20%) | +128–150% |
| v6 | 60-min | 7:1 | 1 ATR | 96 bars | 13–14% (12.5%) | +7% (thin; experimental) |
| v7 | 60-min | 10:1 on real risk | **structure**: 0.25 ATR below the 20-bar swing low | 96 bars | 12–13% (9.1%) | +32%, Sharpe 0.43 |
| vC | 60-min | 30:3 ATR | 3 ATR | 96 bars | ~5% target-hit | +78%, Sharpe 0.68 |
| v5 | — | time-of-day filter for v3/v4: keep only hours whose *training* win rate clears break-even+2pp — drops pre-market; tripled v4's per-trade edge (0.6→1.8 bps) | | | | |

Two *economically different* edges:
- **v3/v4 = mean reversion.** The proven signal fingerprint (z-scored feature means at
  chosen entries): price below its 20-bar mean (−0.71σ), negative recent returns
  (−0.59σ), low RSI (−0.57σ), near support (−0.44σ), room above to the recent high
  (+0.56σ). In words: **oversold dips at support with room to bounce.** Short-horizon
  reversal is a documented effect — too small/costly for big funds to fully arbitrage.
- **v6/v7/vC = trend continuation.** Same engine + trend features + a long clock (96
  bars) + room to breathe. vC's forensics showed its P&L comes from profitable **time
  exits** (median hold ~2.5 days) — it's a trend-*holder*; the far target mostly just
  keeps winners open. v7's structure stop (below the swing low) survives noise that
  kills fixed 1-ATR stops — that one change flipped 10:1 from −257% to +67%/+32%.

**Feature-knockout proof of where the signal lives** (scramble one family at test time,
measure margin drop): location −4 to −5pp (biggest), momentum −3pp, volume −3pp,
volatility −1.5pp. No single family is fatal → the signal is a robust *conjunction*.

**Portfolio effect:** v3 and v4 monthly returns correlate only +0.11 → a 50/50 blend had
the same return as either but ~half the drawdown. Full pooled portfolio at 5 bps:
+85% over 38 OOS months, monthly Sharpe 0.85, 61% positive months.

---

## 6. The graveyard — what failed and WHY (interviewers love this)

| Idea | Result | Lesson |
|------|--------|--------|
| **Buying options** (calls, puts, any moneyness/expiry; tested vs REAL SPY chain quotes) | Loses everywhere (−0.5 to −1%/trade) | Implied vol > realized vol (the volatility risk premium) + spread = a structural tax ~10–30× our per-trade edge. Direction wasn't the problem; the premium was. |
| **Short side (two-sided)** | Dev +185% → fresh **−74%** | Textbook overfit caught by the holdout. Also economic: dips snap back up in an upward-drifting market; rallies don't reliably snap down. |
| **Learned exit model** ("detect failing trades, exit early") | Even at 100% precision (never cut a winner), it added nothing; corrected accounting showed it inert | A loser only becomes *identifiable* as its price falls — the information and the loss arrive together. The stop **is** the optimal exit. |
| **Trailing stops / dynamic exits** | Worse than fixed brackets | Gives back more than it protects at these horizons. |
| **Wide targets (5–10:1) with 1-ATR stops, 24-bar clock** | All deeply negative | 85–90% of trades stop out on noise before a trend can develop. Fixed by v7: structure stop + 96-bar clock + trend features. |
| **Limit-order entries** (backtest + live A/B on paper) | v3: +47%→+11%; v4: +20%→**−77%** | **Adverse selection**: limits only fill when price keeps falling (fills are damaged goods, win rate −4.5 to −5.7pp) and miss the immediate winners. Saving 2 bps of spread cost 3–6× more. |
| **Signal smoothing / persistence** (2-bar confirmation) | Dev Sharpe 0.74→1.43, fresh **inverted** 1.75→1.09 | A dev-only mirage; the top-7% threshold is already a noise gate. Also: improvements must pass the same holdout as strategies. |
| **Reinforcement learning (PPO)** | Lost to simple rules | Weak reward signal + limited data = RL overfits; wrong tool for faint tabular edges. |
| **Deeper/tuned models, ensembles, conviction-weighted sizing** | No measurable gain | The signal is capped by the features (markets are near-efficient), not by model capacity. |

---

## 7. War stories — real bugs found and fixed (tell at least one in interviews)

1. **The fake +518%.** The first wide-target sweep showed 10:1 earning +518%. Audit
   found the time-exit accounting credited a *full +10 ATR win* to any trade that merely
   drifted up by the deadline. With honest accounting (time exits at actual close, wins
   = real target hits) the same cell was **−257%**. Lesson: at extreme parameters, tiny
   accounting conventions dominate results; always ask "what exactly gets credited?"
2. **Label leakage at the fold boundary.** Triple-barrier labels look H bars forward, so
   the last H training labels overlapped the test period. Added an embargo (purge those
   bars). Effect was small (~0.3pp) and the edge survived — but now it's provably clean.
3. **ATR back-fill look-ahead.** The ATR warm-up used `bfill()`, which copies *future*
   values into the first 23 bars. Replaced with NaN + drop.
4. **Live reconciliation blind spot.** The daily health check compared broker positions
   to the ledger's *open* trades and flagged a mismatch — but the "missing" shares were
   in-flight fills in the *pending* queue. Fixed to a range check
   [open, open+pending]. Lesson: monitoring code needs the same rigor as trading code.
5. **The scheduler crash.** Windows Task Scheduler popped a console window; closing it
   mid-cycle aborted Python (`forrtl: error 200`) mid-ledger-write. Fixed with a hidden
   VBS launcher. Lesson: production reliability lives in boring places.

---

## 8. Live deployment (the autonomy story)

- **Broker**: Alpaca paper account ($100k simulated) via raw REST (no SDK): account,
  clock, positions, native **bracket orders** (market/limit entry + OCO target/stop
  legs), per-order polling.
- **Cycle** (every 15 min, phase-aligned to fire at :01/:16/:31/:46 — one minute after
  bar closes, because the just-closed bar takes seconds to publish; firing at :00 sharp
  would *miss* it and add 15 minutes of lag):
  clock check → refresh data tail → load/retrain daily models (embargoed, pickled) →
  evaluate the latest completed bar per strategy×ticker → place bracket(s) → manage
  time exits per order → enforce guardrails → append ledger + log.
- **Why exits can't be missed between cycles**: target/stop legs live at the broker and
  trigger tick-by-tick; the bot only does bookkeeping and time exits.
- **Live A/B experiment**: every signal is placed twice — a market bracket and a
  limit-at-signal-price bracket — tracked as separate virtual accounts with **real fill
  slippage** recorded per trade, to confirm the adverse-selection backtest with reality.
- **Daily 4:15pm automated report**: per-arm × per-strategy P&L, missed-fill counts,
  average slippage (bps), plus health checks (position reconciliation, scheduler
  heartbeat, stale orders auto-cancelled, model freshness, breaker status), committed
  to GitHub daily.
- **Hard boundary**: paper endpoint is hard-asserted in the client; nothing touches
  real money.

---

## 9. Numbers to memorize (one card)

- Data: **5 years, 5-min bars, 8 dev + 5 holdout tickers**
- Features: **26 causal**, 4 families; label: **triple-barrier, ties→stop**
- Selectivity: **top 7%** (train-quantile threshold)
- Costs: **5 bps effective** per trade (3 + 2 slippage)
- Ideas tested: **~40**; validated: **5**
- Break-evens: 1.5:1→**40%**, 4:1→**20%**, 10:1→**9.1%**; our margins: **+7–10pp**
- Random entries: **zero margin** (the proof the entry model is 100% of the edge)
- v3 fresh: **47% win, ~+6 bps/trade, +80–92%**; v7 fresh: **13% win, +32%**
- Portfolio: **+85% / 38 months, monthly Sharpe 0.85, 61% positive months**;
  v3–v4 correlation **+0.11**; blend maxDD **−7%**
- Limit-order A/B backtest: win rate **−4.5 to −5.7pp** (adverse selection)
- Dev DSR: **0.06–0.11** (fails, as expected after 40 trials) → holdout is the verdict

---

## 10. Mock interview Q&A

**Q: Walk me through your system.** → Use the 30-second pitch, then offer to go deep on
any stage of the §3 pipeline.

**Q: How do you know you're not overfitting?** → "Three layers: mechanical (causal
features, embargoed walk-forward, train-only thresholds), statistical (deflated Sharpe
penalizing my ~40 trials), and the decisive one — a frozen run on 5 tickers that never
influenced any decision. I also *expect* most ideas to fail: I rejected 35 of 40,
including a +185% dev strategy that collapsed to −74% on the holdout."

**Q: What IS the edge, economically?** → "Short-horizon mean reversion: forced/panicked
intraday selling overshoots, and price snaps back from support. It persists because it's
too small per trade and too execution-sensitive for large capital to harvest. I proved
the entry model carries 100% of it — random entries with identical brackets earn zero."

**Q: Why LightGBM over deep learning?** → §3 Stage 5. Add: "I tested bigger models and
ensembles — zero gain. The constraint is information in the features, not capacity."

**Q: Your win rate is only 26% on v4 — how is that good?** → "Break-even at 4:1 is 20%.
Six points above break-even, compounded over thousands of trades, is the whole edge.
Win rate is meaningless without the payoff ratio."

**Q: Why did options fail?** → "The volatility risk premium: options price implied vol
above realized — buyers pay an insurance premium. That premium plus spread is a per-trade
tax an order of magnitude larger than my directional edge. I confirmed against real SPY
chain quotes, both calls and puts."

**Q: Market vs limit orders?** → "Backtest and a live A/B both say market. Limit orders
suffer adverse selection: they fill only when price moves against you and miss the
immediate winners — costing 4–6 points of win rate to save 2 bps of spread."

**Q: What breaks your system?** → "Regime change (it's trained on 2021–26; the drawdown
breaker and monthly review exist for this), slippage creep (monitored per-trade in the
live ledger), data revisions (Polygon back-adjusts for dividends — live decisions are
ledgered at execution time so history can't be rewritten), and my own multiple-testing
(the reason the holdout and the graveyard exist)."

**Q: What would you do next with more resources?** → "Three things: more years and more
tickers for regime robustness; an execution study at larger size (my costs assume
~$1k clips); and portfolio construction — volatility-targeted sizing across the five
strategies rather than equal $1k units."

---

## 11. Glossary (fast definitions)

- **OHLCV** — open/high/low/close/volume of a bar.
- **ATR** — average true range; average bar size incl. gaps; the volatility yardstick.
- **Bracket/OCO** — entry with pre-attached target and stop; one cancels the other.
- **Triple-barrier label** — outcome of target vs stop vs time limit, whichever first.
- **Meta-labeling** — ML predicts success of a defined trade, not price itself.
- **Causal feature** — computable at decision time; no future information.
- **Walk-forward** — train past → test future, rolling; all results out-of-sample.
- **Embargo/purging** — dropping training samples whose labels overlap the test window.
- **Deflated Sharpe** — Sharpe re-tested against the best-of-N-trials luck benchmark.
- **Sharpe ratio** — mean return / std of returns, annualized (I report from *monthly*
  sums: mean/std×√12 — per-trade annualization overstates).
- **Adverse selection** — passive orders fill preferentially when you're wrong.
- **Volatility risk premium** — implied vol persistently above realized; the option
  seller's compensation, the option buyer's tax.
- **Slippage** — fill price vs decision price; budgeted 2 bps, measured live per trade.

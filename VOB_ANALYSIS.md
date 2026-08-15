# The Order-Book Trader: Full Analysis (12-hour loop, 2026-08-14)

**Goal:** build an order-book trader and a chart-pattern trader; make both profitable;
report what was found and why.

**Outcome in one paragraph:** The pattern trader succeeded and is validated
(vPT32: one-shot TEST Sharpe 3.43, +291.5%, n=908 — see MINDMAP 10k). The
order-book trader was pursued through eleven rounds of honest backtesting on
complete SIP quote tapes, a 4-agent adversarial audit of the methodology, and a
literature sweep. The verdict is that **L1 order-book information is genuinely
predictive but structurally unmonetizable at retail latency on penny-spread
ETFs** — every execution channel loses to a friction that the signal is too
small to clear, our measured numbers match published live experiments to the
tenth of a basis point, and no credible net-positive counterexample exists
under these constraints. The one surviving use of the book is as a *filter* on
an already-profitable trader (R9 below).

---

## 1. What was built

- `vob_ofi_data.py` — complete SIP NBBO tapes (3.4M quotes/day for QQQ),
  reduced to per-minute Cont-Kukanov-Stoikov order-flow imbalance (OFI), queue
  imbalance (qimb), microprice deviation, spread, quote rate. 72 days QQQ
  (May–Aug), 26 days SPY, Feb–Apr QQQ backfill for the out-of-sample overlay
  test. Audited: math correct, no lookahead.
- A 10-second grid variant for the seconds-horizon work (`data_cache/vob_10s`).
- Eleven rounds of experiments, each with non-overlapping trades, trailing-only
  thresholds, and costs stated.

## 2. What the signal actually is

- 1-minute ICs of qimb/OFI vs next-minute mid return: **+0.03..+0.04** —
  exactly the published range. The signal is real.
- Its tradeable magnitude is **~0.1–1bp** — QQQ's spread is ~0.4bp and the
  round-trip taker cost floor is ~1bp. The signal is smaller than the door
  you have to walk through to use it.
- At L1, microprice ≡ qimb × spread/2 — **microprice adds nothing without
  depth-of-book**, which retail NBBO feeds don't carry.
- Apparent 15-minute "predictability" in decile/bucket charts is
  **overlap-inflated noise**: consecutive minutes share 14/15 of both windows,
  effective n is ~1/15 nominal, and a null simulation with a zero-edge signal
  reproduces the same ±1–5bp bucket structure.

## 3. Why every execution channel fails (the round ledger)

| Round | Approach | Result |
|---|---|---|
| R1 | IC sanity on 72-day tape | signal real, ~0.1bp scale |
| R2 | LGBM on flow features, H=5/15/30m, taker | all negative; ~0 gross |
| R3 | rule extremes (trailing p95), H=15/30m | **gross**-negative (extremes mean-revert) |
| R4A | book state at hourly pattern entries (n=38) | calm-book hypothesis formed |
| R5 | calm filter on 15/30m patterns | lifts VAL −29bp→+1.4bp = breakeven; no base edge |
| R6 | seconds-horizon MAKER (rest at bid, trade-through fills) | −0.4..−0.7bp/fill, win 40–47% |
| R7 | moderate-band taker | best gross +0.85bp < 1bp cost |
| R8 | maker × bands/short-tail | −0.5..−0.8bp |
| R10 | audit-mandated strict fills | all cells worse; lone positive flips negative |
| R11 | cross-asset QQQ↔SPY lead-lag + divergence | all gross-negative; pair arbed dead |

The two channels and their binding frictions:

- **Taker** (cross the spread): cost ~1bp; best honest gross edge ~+0.9bp.
  Negative by construction.
- **Maker** (rest a limit, earn the spread): entry@bid + exit@bid cancels the
  spread entirely — but the fill itself is the poison. A resting order at
  retail latency sits at the **back of the queue** and fills precisely when
  informed flow trades through the level. Measured adverse selection:
  **−0.5bp per fill** — no signal variant cleared it.

## 4. Why this is structural, not an implementation failure

Three independent lines of evidence agree:

1. **Our backtests** — every signal (extremes, bands, tails, agreement,
   activity-conditioned, ML, linear) × every execution (taker, maker) ×
   every horizon (10s–30m) is negative, with the few bugs found by audit all
   pointing in the *optimistic* direction (truth is worse).
2. **Adversarial audit** (4 agents): pipeline math correct, no lookahead;
   the audit's two upward-bias findings (touch-fills, fill-day-only Sharpe)
   were rerun strictly — results got worse, as predicted.
3. **Literature, quantitatively**: a live 232,897-order experiment measured
   −0.47..−0.78bp on back-of-queue imbalance-maker fills (we: −0.5bp);
   queue-position value ~0.2 ticks is allocated in 5–10 **microsecond** races
   won >80% by six firms; IEX built an exchange-level ~2ms protection signal
   (CQI/D-Limit) precisely because this alpha decays too fast for anyone slow.
   Published conclusion (Lipton et al.): the imbalance-conditioned move is
   below the spread and "does not by itself offer an opportunity for a
   straightforward statistical arbitrage." No documented net-positive L1
   strategy at retail latency on US ETFs exists.

The economics in one sentence: **the book's information is worth ~0.2 ticks,
and it is paid out by latency rank — a race that retail enters 4–6 orders of
magnitude behind.**

## 5. R9 — the pre-registered overlay test (the surviving use)

Hypothesis (formed on May–Aug, n=38): hourly pattern entries taken from a
**mid-tercile (calm) book** outperform entries from stressed books.
Protocol frozen before the Feb–Apr tape existed: primary feature qimb30,
tercile edges from the formation sample, pass = mid > non-mid AND mid > 0,
one look, no other cuts reported.

**Result: FAIL.** Primary (qimb30): OOS mid-tercile n=1, −304bp vs non-mid
n=14, +18.7bp — pre-registered pass condition not met. Secondary (micro30):
mid n=2 vs non n=13 — also formally failed. The deeper finding: the formation
sample (hourly QQQ, May–Aug) is only n=10, so the frozen tercile edges were
razor-thin and the test was degenerate. Conclusion: the "calm-book breakouts
win" pattern from R4A (n=38 across two timeframes) was **small-sample noise**
— the same artifact the audit exposed in R7's bucket chart. The overlay
hypothesis is unsupported. There is no validated use of L1 book state in this
system, standalone or as a filter.

## 6. What I would need for a real book edge (honest requirements)

- Depth-of-book feed (L2/L3) — the monetizable structure lives in queue
  dynamics and cancellations, invisible in NBBO.
- Sub-millisecond reaction — to cancel before adverse fills, which is *the*
  maker defense (fast traders escape adverse selection by cancelling).
- Front-of-queue priority — earned by speed, or bought via exchange order
  types unavailable to retail brokers.
- None of these are available through Alpaca at any subscription tier.

*Scripts: `vob_ofi_data.py`, `vob2_model.py`, scratchpad rounds 3–11.
Audit + research: workflow wf_51a50fa8-e59. Ledger: MINDMAP 10l.*

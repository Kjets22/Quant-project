# FROZEN — v7 (saved 2026-07-06): STRUCTURE-STOP 10:1 TREND HUNTER — passed both universes

Frozen snapshot. Do not edit. The 10:1-payoff strategy that finally works: positive P&L on
the dev basket AND the fresh holdout, with real sample sizes.

## Config (exp_ten.py cell "D struct-stop 10:1"; engine in exp_ten.py + wide_hunter.py)
- 60-min bars, HBAR = 96 (~2.5 trading weeks of runway), top-7% conviction (SEL_Q .93)
- Features: base 10 + S/R 10 + 4 TREND features (100-bar momentum, distance to 100-bar
  high (shift(1)), 200-bar trend z, vol expansion ATR12/ATR96) — all causal
- STOP = structure: 0.25 ATR below the 20-bar swing low (shift(1)) — "below support"
- TARGET = entry + 10 x actual risk (a true 10:1 payoff on the real per-trade risk)
- Risk sanity: only take setups where risk is 0.2-4.0 ATR; ATR floor 0.12% of price
- Audited env: embargoed training, no-bfill ATR, 5 bps effective cost, corrected
  accounting (time exits at actual close; win = real target hit)

## Results (walk-forward OOS)
|            | trades | win% (BE 9.1%) | mean bps | total | monthly Sharpe |
| DEV (8)    | 1,574  | 12.3%          | +4.3     | +67%  | 0.55           |
| FRESH (5)  |   912  | 13.0%          | +3.5     | +32%  | 0.43           |
Consistent win rate, bps, and Sharpe across universes = not asset-selection overfit.

## Why it works when fixed-ATR 10:1 failed
The stop sits below actual SUPPORT (the 20-bar swing low), so normal noise doesn't stop
the trade out — only a genuine structure break does. The target scales with the REAL risk
per trade. Combined with trend features + a long clock, it catches multi-day trend legs.
Fixed 1-ATR stops died 85-90% of the time before the trend could develop.

## Grade & siblings (honest)
- EXPERIMENTAL-VALIDATED: both-universe positive with n=2,486 combined. A tier below
  v3/v4 (whose fresh margins are larger), above v6 (7:1, thin +1.0 bps fresh).
- Cell "C 30:3" (10:1 payoff, 3-ATR stop) was also positive both universes (dev +81% /
  fresh +78%, Sharpe ~0.7) but on ~6-16 winners total — too few to trust; watch-list.
- Cells A (20:2 top-3%) and B (25:2.5) FAILED the fresh holdout — dev luck; do not use.
- Only ~12% of trades win: expect long losing streaks; size accordingly.

## How to run (from PARENT capture_trader dir)
- `python exp_ten.py chunk2`        -> dev result (cell D)
- `python exp_ten.py chunk2 fresh`  -> fresh-holdout result

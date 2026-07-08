# FROZEN — v5 (saved 2026-07-03): TIME-OF-DAY FILTER for v3/v4 — VALIDATED UPGRADE

Frozen snapshot. Do not edit. This is not a standalone strategy — it is a proven
ACCURACY UPGRADE that layers onto the validated v3/v4 engines.

## What it is
For each walk-forward fold, after training the entry model, walk the TRAINING block and
measure the win rate of its signals per entry hour (UTC). Keep only hours with train
win-rate >= break-even + 2pp and n >= 25 trades; trade the test period ONLY in those hours.
Hour selection uses training data only -> no look-ahead. Implementation: `optimal_times.py`
(run from the parent capture_trader dir: `python optimal_times.py 30 1.5` / `15 4`).

## Proven results (OOS, audited env: embargo, ATR floor, 5 bps cost, corrected accounting)
- v3 (30-min/1.5:1): +47% -> +53%  (mean bps +1.6 -> +2.3), trades 3002 -> 2277
- v4 (15-min/4:1):  +20% -> +52%  (mean bps +0.6 -> +1.8, ~3x), trades 3334 -> 2883
- What it picks: consistently DROPS pre-market entries (~4-5am ET, the weakest bucket)
  and concentrates on ~10am-4pm ET + the first after-hours hour — where mean-reversion
  bounces have liquid two-way flow. Also improves fill realism (pre-market fills are bad).

## Limits
- Helps HIGH win-rate strategies (v3/v4). Does NOT work on rare-winner wide-target
  configs (7:1+): too few training winners per hour -> the filter correctly falls back
  to all-hours (verified on dev and fresh).
- Selected-hours numbers above are dev-basket; the mechanism is train-only so it is
  honest, but a full fresh-ticker holdout of v3/v4+hours is still worth running before
  relying on the uplift.

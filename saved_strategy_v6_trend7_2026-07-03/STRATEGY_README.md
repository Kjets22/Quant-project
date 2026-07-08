# FROZEN — v6 (saved 2026-07-03): 7:1 TREND HUNTER (fixed 1-ATR stop) — experimental

Frozen snapshot. Do not edit. The first wide ratio to survive the fresh holdout; kept as
an experimental strategy a tier below v3/v4 (and below v7, which superseded it for the
wide-payoff role).

## Config (wide_hunter.py, cell 7:1)
- 60-min bars, HBAR = 96, top-7% conviction, target +7 ATR / stop -1 ATR
- Features: base 10 + S/R 10 + 4 trend features (all causal); audited env
  (embargo, no-bfill ATR, 0.12% ATR floor, 5 bps effective cost, corrected accounting)

## Results (walk-forward OOS)
|            | trades | win% (BE 12.5%) | mean bps | total | monthly Sharpe |
| DEV (8)    | 1,024  | 13.3%           | +3.4     | +35%  | 0.43           |
| FRESH (5)  |   681  | 14.0%           | +1.0     |  +7%  | 0.11           |

## Grade (honest)
EXPERIMENTAL: positive on both universes, but the fresh edge (+1.0 bps) is within noise
of zero. The structural findings that matter: wide targets need RUNWAY (HBAR 96 not 24),
TREND features, and (see v7) structure stops beat fixed-ATR stops. Time-of-day filtering
does NOT help here (too few winners to pick hours; falls back to all-hours — verified).

## How to run (from PARENT capture_trader dir)
- `python wide_hunter.py 7`        -> dev;   `python wide_hunter.py 7 fresh` -> holdout

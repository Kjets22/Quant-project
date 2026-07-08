# Paper-trading guide — v3 + v4 ensemble

This is **research / paper trading only.** Nothing here places orders or moves money. Real
trading risks real loss. Only ever risk money you can afford to lose, and paper-trade first.

## What you're trading
The **ensemble**: two validated strategies run together on the 8-name basket
(SPY, QQQ, AAPL, MSFT, NVDA, JPM, XLE, TLT):
- **v3** — 30-minute candles, 1.5:1 target:stop (steadier, ~47% win)
- **v4** — 15-minute candles, 4:1 target:stop (high-variance, big winners, ~29% win)
Both are long-only ATR brackets, top-7% conviction. Backtest: +122% total, Sharpe 2.47,
max drawdown -7% (the two are nearly uncorrelated, which is why the blend is smooth).

## The bot
`python paper_ensemble.py`
- Trains each model on all data **before** `PAPER_START`, then walks both forward over the week.
- Sizes every trade by fixed risk, prints a blotter + P&L + a guardrail check, and writes
  `runs/paper_ensemble_ledger.json`.
- Idempotent — re-run it any time (e.g. after each trading day) to refresh with the latest bars.

## Guardrails (edit the constants at the top of paper_ensemble.py)
| Constant | Default | What it does / why |
|---|---|---|
| `ACCOUNT` | $10,000 | paper account size |
| `RISK_PCT` | 0.5% | max loss per trade (the 1-ATR stop). Small, because the edge is small and there are many trades. NEVER raise this casually. |
| `MAX_POSITIONS` | 8 | cap on simultaneous open positions -> limits total exposure |
| `DAILY_LOSS_LIMIT` | 2% ($200) | **stop opening new trades for the day** once down this much. Caps bad days. |
| `DD_BREAKER` | 15% ($1,500) | if account drawdown exceeds this, **halt and re-evaluate** — it may mean the edge broke (regime change). Backtested max DD is ~7%, so 15% is a real alarm. |

Sizing math: shares = (ACCOUNT x RISK_PCT) / (1 ATR in $). A win pays 1.5x (v3) or 4x (v4)
the risk; a loss is -1x the risk. So each trade risks the same dollar amount regardless of name.

## Daily workflow
1. Edit `PAPER_START` to the **Monday of the current week** (and `PAPER_END` to the next Monday).
2. After each trading day's close, run `python paper_ensemble.py`.
3. Read the dashboard: which trades fired, their P&L, and the GUARDRAIL CHECK lines.
4. If a guardrail says BREACH, that's the rule firing — in real trading you'd have stopped.
5. Append the week to a running log so you accumulate a record.

## How to judge it (honestly)
- **One week tells you nothing.** Win rate and P&L swing wildly week to week (last week was -3.75%).
  Only the record over **~100+ trades / 1-2 months** means anything.
- Expect plenty of red weeks. v4 wins only ~30% of the time; long losing streaks are normal.
- The thing to watch is whether, over a couple of months, the paper account drifts **up** and the
  drawdown stays within the backtested ~7-15% — not any single week.

## Before risking real money (checklist)
1. Paper-trade live for **1-2 months / 100+ trades** and confirm the edge holds with real fills.
2. Confirm realized slippage is near the assumed ~3 bps (especially for v4's high turnover).
3. Start real with **tiny** size — money you can lose entirely — and only scale after months of live proof.
4. Keep every guardrail on. Respect the drawdown breaker absolutely.

## Hard boundaries
- This tool does **not** and **will not** place trades or move money.
- It is research output, **not financial advice**. Markets can break a backtested edge at any time.
- The strategy was validated on 2021-2026 (one era) and on a fresh-ticker holdout, but the future
  is never guaranteed.

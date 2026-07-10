# Autonomous paper-trading bot — owner's guide

**Paper only.** `alpaca_api.py` hard-asserts the paper endpoint; these keys (PK-prefix)
cannot touch real money. Still: the ON-switch below is yours to flip, and everything the
bot does is logged and auditable.

## What the bot does each cycle
1. Checks the market clock (acts only while the market is open).
2. Refreshes data (5-year cache + fresh Polygon tail).
3. Loads today's models (trains once per day per strategy x ticker, embargoed).
4. If the latest completed bar is a top-7% signal: submits a **bracket order**
   (market buy + sell-limit at target + sell-stop at stop) sized to ~$1,000, whole shares.
5. Manages time exits (positions past their clock get closed).
6. Enforces guardrails BEFORE any order: max 10 open positions, one per ticker,
   daily-loss stop ($200 -> no new entries), drawdown breaker ($1,500 from peak -> HALT).
7. Writes `runs/alpaca_ledger.json` (every trade) and `runs/alpaca_log.txt` (every action).

Strategies: v3 (30m/1.5:1), v4 (15m/4:1), v6 (60m/7:1 trend), v7 (structure-stop 10:1),
vC (30:3 trend-holder). One shared $100k paper account.

## Commands
```bash
python alpaca_bot.py --status     # account + open positions + recent closes
python alpaca_bot.py --dryrun     # compute signals, place NOTHING
python alpaca_bot.py --once       # one real cycle (this is what the scheduler runs)
```

## The ON-switch (run once, in an ADMIN PowerShell, only when you're ready)
```powershell
schtasks /Create /TN "AlpacaPaperBot" /TR "C:\Users\kjets\capture_trader\run_alpaca_bot.bat" ^
  /SC MINUTE /MO 15 /ST 09:30 /ET 16:05 /K /F
```
That runs a cycle every 15 minutes between 9:30am and 4:05pm local time. The bot itself
checks the market clock, so off-hours runs are harmless no-ops.

To pause:  `schtasks /Change /TN "AlpacaPaperBot" /DISABLE`
To stop:   `schtasks /Delete /TN "AlpacaPaperBot" /F`

## If the drawdown breaker trips
The bot HALTS all new entries and says so in the log. To resume after reviewing:
open `runs/alpaca_ledger.json`, set `"halted": false`, save.

## Watching it
- Alpaca dashboard (app.alpaca.markets, Paper tab): live positions, fills, equity curve.
- `python alpaca_bot.py --status` for the ledger view.
- Weekly: commit `runs/alpaca_ledger.json` + log to the repo for the permanent record.

## Judging it (the same rules as always)
Ignore any single day. The account needs **1-2 months / 100+ closed trades** before the
results mean anything. Watch: equity drift vs the backtest expectation (~+0.3-0.7%/month
at this sizing), drawdown staying inside ~2% of the account, and realized slippage
(compare ledger fills vs signal prices). If live fills run much worse than 5 bps, the
edge assumptions need revisiting before any talk of real money.

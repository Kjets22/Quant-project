# CAPTURE_TRADER — SYSTEM MINDMAP (living memory)

> This file is the canonical, always-current memory of the whole trading system.
> Update it whenever anything changes: new strategies, results, incidents, lessons.
> Last updated: 2026-07-23 (late night, post-audit)

## 1. ACCOUNT & MISSION
- Alpaca PAPER account ($100k start, live since 2026-07-10) — **paper only, real money is the user's explicit decision only**
- Equity ~$99,770 (worst day 7/23: −$515 selloff; peak $100,417; DD $650 « $3,000 halt)
- Mission: validate ML/quant strategies with real fills, honest per-strategy books, A/B execution test
- Keys in `.env` (git-ignored, NEVER commit); paper endpoint hard-asserted in alpaca_api.py
- GitHub: https://github.com/Kjets22/Quant-project.git — commit+push EVERYTHING (leak-sweep first)

## 2. THE LIVE BOT — alpaca_bot2.py (every 15 min, weekdays 4:01am–8:16pm ET)
- **A/B arms**: every signal placed twice — mkt (market/marketable) vs lmt (exact price) → slippage measurement
- **Sessions**: rth (brackets) / ext (day limits only, synthetic bot-managed brackets) / closed
- **Strategy independence**: one position per ticker PER STRATEGY (never cross-block; user tests each separately)
- **Guardrails**: $1k/trade, 24 pos/arm backstop, $400 daily-loss stop (blocks new entries), $3k DD halt
- **Alpaca gotchas learned**: no market/bracket orders in ext hours; NO bracket shorts (422); positions NET per symbol (no simultaneous long+short — vM shorts skip stock legs when book is long, express via vMO puts); options have no extended session (queue → next open)
- Ledger: runs/alpaca2_ledger.json | Log: runs/alpaca_log.txt | `python alpaca_bot2.py --status`

## 3. THE STABLE (14 books)
| Strat | What | Validation | Live status (wk of 7/20) |
|---|---|---|---|
| v3 | 8 tickers, 30m, 1.5×ATR/1×ATR, top-7% | fresh-ticker holdout +80% | −$34 wk; choppy |
| v4 | 15m, 4:1 ATR | holdout-validated | ~flat wk; 16 missed ext limits |
| v6 | hourly trend 7:1 | experimental | −$44 wk (worst) |
| v7 | struct-stop 10:1 (swing-low) | fresh +32% | −$6; 9 open riding |
| vC | **moonshot 30×ATR/3×ATR drift-rider** | probes: keep 30× (reachable targets WORSE) | **+$184 wk** — JPM/NVDA/XLE time-exit monsters |
| vQ | QQQ $2/$2 1h conf-gate | tournament 1 | flat |
| vQ2 | QQQ $2.50/$2 2h histgb | Evo I champ | 2-for-2 targets +$7.84 |
| vA | QQQ $1.50/$2 4h (accuracy) | Evo II champ | flat |
| vP | QQQ $2/$2 8h histgb | Evo III champ (final +4.18%) | ~flat, high volume |
| vR | **QQQ +0.4%/−0.2% 2h top-3% — USER'S SPEC** | Evo IV FINAL WINNER +7.00% | **2-for-2 targets +$8.46** |
| vS | QQQ +0.5%/−0.4% 8h | Evo IV challenger +6.18% (lost) | −$6 wk |
| vM | morning 2-sided ORB, NR≤0.3, 2×risk, flat noon; 6 tickers | parallel-session ladder (final +6.57%, arena razor +0.01%) | first trades 7/22 (+$5.67); 5 shorts 7/23 blocked by bugs → FIXED |
| vCO | ~$1k 1–2w ATM calls on vC signals, own virtual bracket | real-fill replay +14.9%/trade | −$290 realized; **JPM call +$365 open, NVDA +$35** |
| vMO | 0DTE ATM call/PUT on vM signals (QQQ/SPY), flat noon | real-fill sim; SPY baseline +9.7%/tr | first 2 trades −$296 (0DTE calls); puts armed |

## 4. QQQ-FAMILY KEY FACT
- **The edge is EXTENDED-HOURS**: 0% of vQ..vS signals fire 9:30–16:00 (proved via qqq_options_real.py)
- Extended-hours trading enabled 7/20 → family finally fired 7/22 (fills with price improvement −0.7 to −2.4bps)
- Expected rate ≈ 1.3 signals/day across all six (bot samples ⅓ of 5-min bars)

## 5. OPTIONS KNOWLEDGE (all real-fill tested)
- Buying calls: REFUTED for small-move strats (v3: 12/12 negative) & low-win-rate (v7 9% win = theta death)
- **vC is THE exception**: +6–12%/trade at ATM/3%-OTM, 1–2w..2m DTE, positive in EVERY bucket; delayed entries fine (multi-day edge)
- Lottery profile: median trade −13..−38%, 18–28% win — FIXED premium sizing mandatory (10% compounding = 50.7% maxDD)
- QQQ scalpers can't use options AT ALL (hours don't overlap)
- SPY chain parquet: bid/ask are MODELED ±0.1% around REAL traded closes (cite honestly)
- Polygon key covers options minute history (contracts + 5-min aggs; caches in data_cache/options/)

## 6. TOURNAMENT / RESEARCH LEDGER (honesty ladder: arena worst-of-halves → gate → one-shot final)
- Evo I (evolve_vq): vQ2. Evo II (evolve2): P&L track overfit-failed, vA (accuracy) passed
- Evo III (evolve3): vP (+4.18% final, beat vQ). Evo IV (evolve4): vR DEFENDED (+7.00% vs +6.18%)
- **Evo V (evolve5, RTH islands 100 agents)**: 10/10 champions FAILED final — RTH long-only QQQ = no edge
- **quant_rth**: 27 documented anomalies (ORB/gap/VWAP/intraday-momentum/±regime filter): 0/27 passed arena; QQQ 15:30→close drift NEGATIVE every half-year this era
- **Evo VI (evolve6, session features F6/F7, deeper islands)**: 0/10 passed gate — RTH CASE CLOSED (4 independent negative lines; reopen only with order-book/news features or sub-1bp costs)
- Trees beat LSTM/GRU/RF/logreg 7 straight times, even on their own seeded islands
- vC clock probe: H=96 optimal (more time ≠ better); target probe: 5×–20× all worse than 30× (far target = no ceiling, drift-riding IS the strategy)
- Sizing: 10%-of-$10k compounding ≈ irrelevant for stock strats (bps-scale), huge but drawdown-brutal for options

## 7. DO-NOT-REVISIT (documented rejections)
- Short side on the ML stable (fresh holdout −118%) | options buying for v3/v4-style | $2/$1 QQQ (cost toll: needed 45.3% win, got 42.1% — re-verdict only if measured cost <2bps) | two-sided top-20% (overlap-inflated accuracy) | F5 SPY cross-features (died at gate) | wide fixed-ATR stops 20:2, 25:2.5 | RTH-only price-feature day trading (Evo V/VI + quant_rth)

## 8. PARALLEL ENGINES & SURFACES
- **morning_daily.py** (parallel session): daily 16:35 sim+search+promote for vM/vMO (research-only); reports runs/morning_reports/; state runs/morning_state.json (forward = per-day returns)
- **live_dashboard.py** (port 8765, split endpoints /meta /trades /candles?tk= — this machine's loopback truncates big responses at 255/256·2^n!): sim tabs for CONFIGS strats + LIVE ledger books for vCO/vMO + vM sim days; restart: kill python live_dashboard → Start-Process hidden
- **daily_report.py** (16:15 wkdays): per-arm×strategy, options books split vCO/vMO, morning-engine section, recovered-trades section, health checks (auto-cancel stale, reconciliation range, heartbeat, model age, slippage>5bps)
- **backfill_missed.py**: recovers trades missed via OUTAGE or the old cross-block rule ONLY (user policy) → runs/recovered_trades.json (wk: −$42.32 net — honesty cuts both ways)
- Scheduled tasks: AlpacaPaperBot (4:01–20:16 ET wkdays q15m, WakeToRun), AlpacaDailyReport (16:15), MorningSimDaily (16:35) — all StartWhenAvailable+battery-proof after the 7/16 outage (reboot+sleep killed a full day; bracket legs at broker kept positions safe)

## 9. STANDING USER INSTRUCTIONS
- Commit+push everything; paper only; strategies are separate entities (never cross-block); recovered-trade crediting only for outages/old-rules; keep the frozen saved_strategy_v* snapshots untouched; keep THIS MINDMAP updated as the working memory

## 10. FULL-SYSTEM AUDIT — 2026-07-23 LATE (all verified; do NOT re-audit these)
- ✅ Scheduled tasks ×3 all result=0, correct next-runs (bot 4:01am, report 16:15, morning 16:35)
- ✅ Models: all 11 strategies retrained today (46 pickles); dashboard alive; both reports generated+pushed
- ✅ Ledger: no stuck pendings, no stuck opt_queue, not halted; vCO calls have correct virtual
  brackets (JPM tgt 379.90/stop 336.53/ddl 7-29/exp 7-31; NVDA similar)
- 🔧 FIXED: exit client_order_id collisions (50×: multi-strategy same-ticker exits in one cycle
  shared IDs — only first sell/cycle succeeded) → IDs now include strategy name
- 🔧 FIXED: reconciliation now signed (shorts negative) + AUTO-SELLS excess long shares
  (late-fill-after-cancel orphans; found 1 QQQ share — auto-fix will clear it next 16:15)
- 🔧 FIXED: stale-signal bracket rejections (price through stop on fast moves; 15-min data lag)
  and wash-trade rejections now log as classified skips, not errors
- 🔧 FIXED: alpaca_api 60s timeout + 1 GET retry (Alpaca threw timeouts + a 500 tonight;
  mutations never auto-retry — duplicate-order risk)
- KNOWN CONSTRAINTS (by design, do not "fix"): netted account → wash-trade guard occasionally
  blocks one arm when another strategy's exit order sits on the same ticker (~1-2/day, logged);
  vM shorts skip stock legs when the book is long the ticker (puts express it); ext-hours limit
  arms miss in thin books; synthetic exits are 15-min granularity vs broker-instant brackets

## 11. COMPLETED — NEVER REDO
- Tournaments: Evo I–VI + quant_rth + probes (2to1, pct, vc_time, vc_target) — all concluded,
  results in §6/§7; the RTH question is CLOSED
- Options research: real-fill replays for SPY strats, QQQ family, vC (all tickers) — concluded
- Infra: extended hours, strategy independence, vCO/vMO independence + virtual brackets, shorts
  plumbing, recovery engine, split dashboard endpoints (loopback truncation!), wake-proof
  schedulers, MINDMAP-as-memory — all built & verified; audit fixes above committed (ae3a604)
- Resume bullets + INTERVIEW_GUIDE.md + ALPACA_BOT_GUIDE.md written (earlier sessions)

## 12. OPEN QUESTIONS / WATCHLIST
- vM live-vs-sim gap (sim engine runs in parallel as control); vM shorts unproven live (fixed 7/23 evening, untested)
- vCO JPM call (+$365 mark): does the virtual bracket exit well?
- Slippage A/B verdict still accumulating (mkt +9bps-ish vs lmt ~0 but 16 misses/wk)
- QQQ-family live rate vs backtest expectation (~1.3/day) — audit if silent ≥4 sessions
- v3/v6 underperformance — regime or decay? Revisit after 2+ weeks of live data

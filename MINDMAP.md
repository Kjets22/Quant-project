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

## 10b. DATA-LAG FINDING + FIX (2026-07-24) — the sim-vs-live gap explained
- The bot's bars are **15-min delayed** (Polygon). Multi-day strategies (vC/v6/v7) don't care;
  intraday tight-stop ones (vM, and vMO riding them) are wrecked by it.
- Smoking gun: vM DIA short — signal 516.27 / stop 518.30 (risk 2.03); filled 518.05, i.e.
  **88% of the risk budget gone before entry**; stopped 5 min later. Meanwhile the sim engine
  (fills at the breakout bar close) shows QQQ/SPY forward 100% win — same days, opposite result.
- FIX (committed 934fbf6): `alpaca_api.latest_price()` = real-time IEX last trade (free on paper);
  `entry_still_valid()` skips entries with <50% of the risk budget remaining. Applied to vM +
  ML strategies. Fails OPEN (no quote → trade proceeds). Verified: today's DIA short is blocked.
- PART 2 (2026-07-27, commit f64316f): the guard originally checked only drift toward the STOP.
  v3 QQQ exposed the other half — signal 677.89 (tgt 681.88 / stop 675.24 = 1.5:1) filled at
  681.49, leaving $0.39 upside vs $6.25 downside (16:1 adverse); it won +$0.50 on luck.
  `entry_still_valid` now ALSO requires remaining reward:risk ≥ 50% of the validated ratio.
  Calibration measured on all 15 real fills that day: blocks exactly the 2 collapsed-R:R entries,
  allows the other 13 → 13% blocked. Guard fires live (2 skips logged 7/27: v6 TLT 52% stop
  distance gone, v4 XLE 74%).
- `opt_queue` dedupes per ticker (vC re-signals the same name hourly overnight; 4 identical TLT
  calls had queued, only one could ever fill).
- STILL OPEN (next step if vM keeps diverging): source vM's opening range and breakout detection
  from real-time bars instead of delayed ones, so entries fire at the breakout rather than up to
  20 min later. The guard is a filter, not a cure.

## 10c. DATA SOURCE — POLYGON IS NOW OPTIONAL (2026-07-27)
- Polygon **was** load-bearing: `full_series()` tails it every bot cycle; `morning_daily.refresh()`
  pulls all 6 tickers daily. Cancelling it before this change would have starved the bot.
- **Alpaca serves the same bars free** with the trading account. Verified over a full week on all
  12 live tickers: mean close diff 0.000–0.025 bps, p99 ≤0.14 bps, 0–1 outlier bars/ticker >5 bps
  (odd-lot/late-report singles), and **vM's 25-min opening-range hi/lo inputs match to $0.0000**.
  Full 4:00–20:00 ET extended hours, ≥5y history.
- `data.fetch_bars()` = Alpaca primary → Polygon fallback. Call sites swapped (bot + morning).
- **Gotcha (cost me a false-green first pass): the free plan 403s any SIP window touching the last
  15 minutes** — "subscription does not permit querying recent SIP data". `fetch_alpaca` clamps
  `end` to now−16min. Do NOT remove that clamp.
- Feeds: `sip` = full/consolidated but ≥15-min delayed (what the bot uses, same as Polygon ever
  gave). `iex` = ~3-4 min behind but sparse (16 vs 228 extended-hours bars over 2 days) — NOT a
  substitute for bars. Real-time single prices come from `alpaca_api.latest_price()` (IEX).
- Verified end-to-end with Polygon monkeypatched to always raise: full_series, vm_signal, the
  35-feature prep pipeline, and a full dry-run cycle all pass. **Safe to cancel Polygon.**
- Still Polygon-only (research scripts, already-cached results, not scheduled): historical OPTION
  bars in `qqq_options_real.py` / `vc_options_real.py` / `options_data_polygon.py`. If those need
  re-running later, port them to Alpaca's `/v1beta1/options/bars` first.

## 10d. vCO PROFIT GIVE-BACK — the JPM lesson (2026-07-29)
- The JPM call was marked **+$870** on 7/27 and realized only **+$170** on 7/29 when it hit its
  8-day TIME deadline. It rode +115% → +22% with no profit protection.
- This is **faithful to the validated backtest** (vc_options_real exits the option at the STOCK
  leg's target/stop/time), and the +14.9%/trade backtest average already contains give-backs like
  this. So it is NOT a bug — but it is the single biggest driver of the equity swing that week
  (the −$1,213 mark-to-market day on 7/29 was mostly this one position).
- Consequence to remember when reading vCO marks: an open option mark is not money until the exit
  rule fires, and vCO's exit rule is the STOCK's geometry (wide vC levels), not the option's P&L.
- NOT changed: adding an option-level trailing stop or profit target would depart from the
  validated design. If ever tested, it must go through the full ladder first — do not bolt it on.

## 10e. GIT TLS (2026-07-30) — pushes failed with "unable to get local issuer certificate"
- Something on this machine intercepts HTTPS (same family as the loopback-truncation AV issue).
- FIX: `git config --local http.sslBackend schannel` (uses the Windows cert store, which trusts
  the interceptor's root). Do NOT use `sslVerify=false`. If the scheduled report's auto-push ever
  fails this way again, this is the remedy.

## 10d. ⚠ LOOKAHEAD BUG IN THE OPTIONS REPLAYS (found + fixed 2026-08-03)
**The most important methodological lesson in this project so far. Read before writing any
backtest that mixes resampled bars with tick/minute data.**
- `prep()` resamples with pandas default `label='left'`: a 60-min bar stamped 13:00 carries the
  **13:55 close**. Verified directly.
- Both `vc_options_real.py` and `v6_options_real.py` transacted the option leg at `ts[i]` — the
  bar START — while the signal comes from `c[i]`, the bar CLOSE. That is a **55-minute lookahead
  on every option entry**, and the mirror image on every exit. Because these are TREND models,
  signal bars are up bars, so it systematically bought cheap and sold before the adverse move.
- Impact, measured by re-running vC honestly: **+14.9%/trade → +10.1%/trade** (~32% of the
  claimed edge was the bug). The edge survived; the number did not.
- FIX: both legs now transact at `ts[...] + bar_duration`. Also fixed `contracts_near`'s hard
  50-day expiry cap (`cap=` param) which had silently collapsed any "8-12w" bucket to DTE=50.
- The LIVE bot never had this bug — it acts on completed bars in real time. Only the VALIDATION
  was optimistic, which matters because the validation is what put vC-OPT-2W on the account.

## 10e. v6 OPTIONS — REJECTED TWICE, DO NOT REBUILD
- 9 structures (2-3w/4-6w/8-12w × ATM/3%ITM/7%ITM), all v6 tickers, real fills: **8 of 9 negative.**
- The single positive cell (8-12w ATM +1.5%) was an artifact three ways: (a) the 50-day cap meant
  it was one expiry date, not a range; (b) 48 of 58 fills were Thursdays (only weekday landing on
  a Friday expiry at +50d); (c) top-3 fills = 188% of P&L. Running OTHER expiries on those same 58
  signals did BETTER (+4.3%, +4.7%) — refuting the "long tenor = less theta" rationale outright.
- v6 holds ~0.9 calendar days on average; buying 50-day options for that is incoherent anyway.
- **v6 stays stock-only.** Do not re-test without a fundamentally different thesis.

## 10f. v6 STOCK — NO DEMONSTRATED EDGE (2026-08-03, independent 4-window audit)
- 1,246 trades over 4 years: **-3 bps/trade, t = -0.73** — indistinguishable from zero.
  2023-24 positive on 5/5 seeds (+17…+30%); 2024-25 straddles zero; **2025-26 negative on 5/5
  seeds** (-26% to -59%). The -84.5% figure was the WORST seed — do not quote it.
- Structural cause: top-7% gated target-hit rate **12.2%** vs an unconditional base rate of
  **~12%** — the ML gate adds no measurable selection lift, and cost-adjusted break-even is 14.5%.
- v6's live +$224 lead rests on ONE MSFT winner across 22 closed trades. Treat as luck until
  proven otherwise. DECISION PENDING with the user: keep watching vs remove from the stable.

## 10g. vC-OPT-2W ROBUSTNESS (post-fix, `vc_options_robust.py`)
- 152 fills: mean **+10.1%/trade**, median **-32%**, win 30%, **t = +1.11**,
  bootstrap 95% CI **[-7.0%, +28.2%]**, P(mean>0) = 86.6%. 4/5 tickers positive (TLT negative).
- Outlier dependence: top-3 = 78% of P&L (still positive without them); **top-5 = 119% — remove
  them and it is NEGATIVE.** That is inherent to a convexity strategy, but it means the edge is
  NOT statistically established. Downgrade language from "validated" to "positive point estimate,
  unproven". Kept live as a paper experiment; the live book is -$687 over 6 trades.

## 10h. THE GREAT AUDIT — 2026-08-12 (9-agent deep dive + manual passes; ALL FIXED)
**Eleven bugs found and fixed in one session. The live system before this date was running
without broker-side protection on multi-day positions. Read this section before touching
execution code.**
1. **DAY-TIF BRACKET LEGS (worst bug in project history):** alpaca_api brackets sent
   time_in_force='day' while the docstring claimed GTC → every multi-day position lost its
   stop AND target at entry-day close (Alpaca expires TP, OCO-cancels stop). 12/13 open
   positions were naked; v6 MSFT sat below its stop, v6 XLE had touched its target unpaid.
   FIX: GTC legs + manage_exits promotes expired/canceled-leg positions to synthetic
   (bot-side bracket). First live cycle after the fix: XLE booked its owed TARGET +$58×2,
   MSFT its overdue STOP. This is why v6/v7/vC only ever logged TIME exits.
2. **STUB BARS:** feed is ~16-min delayed but prep() kept bars "completed by wall clock" —
   models scored truncated bars (5/6 sampled signals wrong, e.g. sig_px 303.98 vs true
   305.00). FIX: bar complete only when raw coverage reaches its end.
3. **vM-OPT-0DTE NEVER ACTUALLY RAN:** the EXPIRY safety net (expiry<=today+1) is always
   true for 0DTE and ran before tgt/stop/deadline → all 6 fills dumped within ~10 min
   (−$476 of pure spread). FIX: net fires only when deadline outlives the contract;
   past-expiry holdings settle at intrinsic (EXPIRED-UNSOLD) instead of looping.
4. **--dryrun MUTATED THE LIVE LEDGER** (bookings weren't dry-gated; saves unconditional;
   no lock) → 7/24 dry cycle booked a phantom QQQ exit with no broker order = THE true
   origin of the QQQ orphan share (earlier partial-fill diagnosis was a real hole but NOT
   this cause). FIX: dry deep-copies, never saves. Orphan sold 8/12 @ 725.18 (+$32.68,
   booked as ledger adjustment; adjustments now shown in the daily report).
5. **OPTIONS MARKET-ORDER OVERPAYING 14-160%** vs day VWAP on 9/18 entries (~$1.5-2.5k of
   the options drawdown was execution). FIX: quote-pegged DAY limits both sides (entry
   mid+20% of half-spread; exit mid−20%, reprice per cycle, 3rd try at bid), spread cap
   25% of mid, sizing from live quote mid, budget hard-capped at 1.5× (user-confirmed).
6. **PARTIAL FILLS:** expired entries with partial fills booked MISSED (orphan factory);
   cancelled option sells with partials froze positions. FIX: partials become live
   positions / book the sold slice.
7. **EXIT CID COLLISIONS ACROSS DAYS** (HHMMSS only; noon vM cover hits the same second
   daily; 2 cycle-crashes in task log). FIX: date in cids + failed time-exit orders no
   longer book the exit (retry next cycle).
8. **STALE IEX PRINTS:** latest_price returned prices frozen up to 14+ h off-hours — the
   staleness guard vetoed 139 ext-session entries against dead prices (consuming rare
   QQQ-family signals!) and passed a vM OEF entry already through its stop. FIX: prints
   older than 10 min → None → all callers fail OPEN.
9. **NETTED-ACCOUNT SIDE COLLISION:** vM short on QQQ/SPY would SELL other strategies'
   longs (stranding their legs); longs would COVER vM shorts. FIX: both directions skip
   when the other side is held.
10. **ORPHAN AUTO-FIX NEVER FIRED** (gated on market-open; report runs 16:15). FIX:
    extended-hours marketable limit path.
11. **QQQ-family silence partly explained:** stub bars + stale-price vetoes were eating
    their extended-hours signals (see 2, 8). Expect their live trade rate to rise.

## 10i. OPTIONS TWINS — FINAL VERDICTS (2026-08-12/13, lookahead-corrected, 6-gate screen)
strat_options_test.py = the canonical instrument (bar-close legs, 40-fill minimum, t>=1.5,
bootstrap P(>0)>=90%, positive-without-top-5, >=60% tickers positive). Results:
- **v6: ✗✗✗ (third and FINAL rejection).** All 9 cells negative (means -4..-14%, t -1.9..-3.1,
  P(edge) 0-3%). The old "8-12w winner" fully vanished with the lookahead fix. CLOSED FOREVER.
- **v7: ✗.** Stock leg positive (+29.3%) and 3 cells positive-mean (best 2-4w ITM3 +3.7%,
  P=87%) but EVERY cell is negative without its top-5 trades — 12%-win jackpot profile can't
  be certified. Stays stock-only.
- **v4: ✗.** 8/9 negative; ~85% of trades can't match contracts (15-min holds too fast for
  options). Stays stock-only.
=> Only vC (multi-day trend rides) has an options-compatible shape among the ATR family.
Do NOT re-run these three without a fundamentally new thesis or intraday option quotes.

## 10j. THE TWO NEW BOTS (user goal 2026-08-13): vPT geometric patterns + vOB order book
**vPT — CHAMPION FOUND AND ONE-SHOT-TESTED.** vpt_geometric.py = ZigZag(k×ATR) pivots +
grammars (HS/iHS, DT/DB, bull/bear flags, asc/desc triangles), causal confirmation-bar
entries, invalidation stops, measured-move targets, any 1-60min interval, shorts allowed.
vx_lab.py = the improvement loop (OPT 23-24 / VAL 24-25 / one-touch TEST 25-now, daily-P&L
annualized Sharpe, persistent leaderboard, exploit-mutation).
CHAMPION: **hourly, DB+iHS only, LONG only, k=2.0, tol=0.5%, H=48, 8 tickers** →
OPT 1.65 / VAL 1.24 / **TEST (single touch) Sharpe 1.02, +45.7%, n=641, 58% win,
maxDD $366 per $1k-trade unit, 10/14 months positive** (vpt_champion.py; curve in
runs/vpt_champion_curve.csv). A REAL new edge on a new mechanism.
Iteration ledger (all selected on OPT, confirmed on VAL — do NOT redo):
  • 5/15/30-min patterns: killed by costs (15-min −248% OPT, 4.4k trades)
  • bearish patterns/shorts: negative in every surviving config (long-only survives)
  • DT/triangles/flags: inert; DB+iHS carry the whole edge
  • volume confirmation: OPT mirage, VAL-negative — discarded
  • breakeven stops: nil effect; vol-normalized sizing: hurts (NVDA is the earner)
  • multi-timeframe stacking: impossible (only 60-min OPT-positive)
Sharpe-3 target NOT reached — champion ceiling ≈1.0-1.3 OOS; honest.
**vOB — VERDICT PENDING DATA.** vob_data.py backfills end-of-5-min-bar NBBO snapshots
(threaded, ~25x faster, newest-first; features: spread/imbalance/microprice/quote-rate).
Early evals (2.5 months): massive OPT-VAL overfit gap (OPT Sharpe 4-6.6 → VAL −5..−22)
and book features NOT beating the price-only control. Needs the full Feb+ backfill;
1-min book snapshots = future work. Do not conclude on vOB until ≥6 months of book data.

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

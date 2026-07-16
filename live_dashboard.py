"""
live_dashboard.py — strategy dashboard: every stable strategy, clickable, showing the
trades it WOULD have taken since PAPER_START (hypothetical fills at the signal close,
5 bps cost), simulated with the same data/feature/model pipeline the live bot uses.

  python live_dashboard.py        then open  http://localhost:8765

- Models train ONCE on data BEFORE PAPER_START (embargoed — no lookahead into the
  displayed window), cached in models/dash_*.pkl so restarts are instant.
- Background loop: every REFRESH_S seconds re-pulls the latest (15-min-delayed) Polygon
  bars through the bot's own data path and re-simulates every strategy forward.
- UI: strategy tabs (click one -> only its trades) + ticker tabs, candlestick chart with
  entry/exit markers, blotter, and summary cards for the current selection.
Places no orders; research/paper only. The LIVE bot's real fills are in the daily report.
"""

from __future__ import annotations

import json
import pickle
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import lightgbm as lgb

import alpaca_bot2 as bot
from alpaca_bot2 import CONFIGS, MODEL_BY_STRAT, MIN_ATR_PCT, prep, _barriers
from triple_barrier_breadth import TICKERS
from vc_options_real import contracts_near, day_close
from qqq_options_real import bars_for, et_date

PAPER_START = pd.Timestamp("2026-06-29")
EFF_COST = 5.0 / 1e4                 # 3 bps + 2 bps slippage, on notional
NOTIONAL = 1_000.0                   # same sizing as the live bot
REFRESH_S = 300
PORT = 8765

STATE = {"ok": False, "msg": "starting up: training models (first run takes a while)..."}
DESCS = {
    "ALL": "Every strategy's hypothetical trades since Jun 29 — click a strategy to see just its trades.",
    "v3":  "The original: 30-min bars, 8 tickers, target 1.5x ATR / stop 1x ATR, ~12h max hold. Passed the fresh-ticker holdout (+80%, 5/5 folds).",
    "v4":  "High-payoff sibling of v3: 15-min bars, 4x ATR target / 1x ATR stop, ~6h. Wins only ~30% of the time but winners are 4x the losers.",
    "v6":  "Trend hunter: hourly bars, 7x ATR target / 1x ATR stop, up to ~4 days, trend features. Experimental — thin but real edge.",
    "v7":  "Structure stop: stop sits under the 20-bar swing low, target = 10x the actual risk, hourly, ~4 days. Rare wins (~10%), huge when they land.",
    "vC":  "Moonshot drift-rider: 30x ATR target (almost never hit — it rides trends), 3x ATR stop, hourly, ~4 days. Wins come from TIME+ drift exits.",
    "vQ":  "QQQ scalper: $2 target / $2 stop within 1 hour, top-10% confidence gate. The first QQQ tournament champion.",
    "vQ2": "Evolution I champion: QQQ $2.50 target / $2.00 stop in 2h, HistGB model. Very few trades, ~68% win rate.",
    "vA":  "Evolution II accuracy champion: QQQ $1.50 / $2.00 in 4h, top-5% gate. ~69% win rate — the accuracy specialist.",
    "vP":  "Evolution III P&L champion: QQQ $2 / $2 in 8h, HistGB, high volume (~280 trades/yr). Final year +4.18%.",
    "vR":  "Your spec — and the Evolution IV final WINNER: QQQ +0.4% / -0.2% (true 2:1) in 2h, top-3% gate. Best final year of the family (+7.00%).",
    "vS":  "Evolution IV evolved challenger: QQQ +0.5% / -0.4% in 8h, top-10% gate. Lost the final to vR (+6.18%) — runs live for comparison.",
    "vCO": "OPTIONS twin of vC: each vC signal buys ~$1k of 1-2 week ATM calls (REAL traded option prices, 1%/side cost), sold when the stock leg exits. Entry = option price per share; P&L per $1k premium. Live on paper as its own book.",
}
MODELS_MEM = {}                      # (strat, tk) -> {"clf", "thr"} or None
MODELS_DIR = Path("models")
_PREP = {}                           # per-refresh prep memo (shared across strategies)


def dprep(tk, mins, featmode, mode):
    key = (tk, mins, featmode, "struct" if mode == "struct" else "std")
    if key not in _PREP:
        _PREP[key] = prep(tk, mins, featmode, mode)
    return _PREP[key]


def train_dash(strat, tk, mins, hbar, mode, tp, sl, featmode, sel):
    """Bot's train_or_load, but embargoed at PAPER_START instead of today."""
    MODELS_DIR.mkdir(exist_ok=True)
    pkl = MODELS_DIR / f"dash_{strat}_{tk}_{PAPER_START:%Y%m%d}.pkl"
    if pkl.exists():
        return pickle.loads(pkl.read_bytes())
    ts, h, l, c, A, X, valid, stop_px, tgt_px = dprep(tk, mins, featmode, mode)
    stop_px, tgt_px = _barriers(mode, c, A, tp, sl, stop_px, tgt_px)
    n = len(c)
    y = np.full(n, np.nan)
    for i in range(n - 1):
        if not valid[i]:
            continue
        for j in range(i + 1, min(i + hbar + 1, n)):
            if l[j] <= stop_px[i]:
                y[i] = 0; break
            if h[j] >= tgt_px[i]:
                y[i] = 1; break
    fv = (X.notna().all(axis=1) & np.isfinite(A) & valid).to_numpy()
    tr = np.where(fv & np.isfinite(y) & (ts < np.datetime64(PAPER_START)))[0]
    tr = tr[:-hbar] if len(tr) > hbar else tr            # embargo
    obj = None
    if len(tr) >= 500 and y[tr].sum() >= 20:
        if MODEL_BY_STRAT.get(strat) == "histgb":
            from qqq_tournament import MODELS as TOURN_MODELS
            clf = TOURN_MODELS["histgb"]()
        else:
            clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                     min_child_samples=40, subsample=0.8,
                                     colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
        clf.fit(X.iloc[tr], y[tr].astype(int))
        ptr = clf.predict_proba(X.iloc[tr])[:, 1]
        if sel[0] == "conf":
            thr = float(0.5 + np.quantile(np.abs(ptr - 0.5), sel[1]))
        else:
            thr = float(np.quantile(ptr, sel[1]))
        obj = {"clf": clf, "thr": thr}
    pkl.write_bytes(pickle.dumps(obj))
    return obj


def sim_dash(strat, tk, mins, hbar, mode, tp, sl, featmode, sel, model):
    """Non-overlapping hypothetical trades since PAPER_START (fills at signal close)."""
    ts, h, l, c, A, X, valid, stop_px, tgt_px = dprep(tk, mins, featmode, mode)
    stop_px, tgt_px = _barriers(mode, c, A, tp, sl, stop_px, tgt_px)
    fv = (X.notna().all(axis=1) & np.isfinite(A) & valid).to_numpy()
    n = len(c)
    fwd = np.where(fv & (ts >= np.datetime64(PAPER_START)))[0]
    if len(fwd) == 0:
        return []
    proba = {int(ix): float(p) for ix, p in
             zip(fwd, model["clf"].predict_proba(X.iloc[fwd])[:, 1])}
    trades = []
    i, last = int(fwd[0]), int(fwd[-1])
    while i <= last:
        pr = proba.get(i, -1.0)
        if (pr < model["thr"]
                or (mode not in ("dollar", "pct") and A[i] / c[i] < MIN_ATR_PCT)):
            i += 1; continue
        qty = int(NOTIONAL // c[i])
        if qty < 1:
            i += 1; continue
        up, dn = float(tgt_px[i]), float(stop_px[i])
        res, j = None, i + 1
        while j < min(i + hbar + 1, n):
            if l[j] <= dn:
                res = 0; break
            if h[j] >= up:
                res = 1; break
            j += 1
        base = dict(strat=strat, tk=tk,
                    entry_ts=int(pd.Timestamp(ts[i]).timestamp()),
                    entry=round(float(c[i]), 2), target=round(up, 2),
                    stop=round(dn, 2), qty=qty, conviction=round(pr, 3))
        if res is None and j >= n:                       # still open
            base.update(outcome="OPEN", exit_ts=None, exit=None,
                        pnl=round(qty * (c[n - 1] - c[i]), 2),
                        last=round(float(c[n - 1]), 2))
            trades.append(base); break
        if res is None:                                  # time barrier
            ex_j = min(j, n - 1); px = c[ex_j]
            oc = "TIME+" if px > c[i] else "TIME-"
        else:
            ex_j = j; px = up if res == 1 else dn
            oc = "TARGET" if res == 1 else "STOP"
        base.update(outcome=oc, exit_ts=int(pd.Timestamp(ts[ex_j]).timestamp()),
                    exit=round(float(px), 2),
                    pnl=round(qty * (px - c[i]) - qty * c[i] * EFF_COST, 2))
        trades.append(base)
        i = (ex_j if res is None else j) + 1
    return trades


def _opt_leg(tk, t0, t1, S0, ts_all, c_all):
    """Price a vC trade's option twin from real bars. Returns (entry, exit, closed)."""
    import datetime as dt
    d0 = et_date(t0)
    cand = [c0 for c0 in contracts_near(tk, d0, S0 * 0.97, S0 * 1.03)
            if 5 <= (dt.date.fromisoformat(c0["exp"]) - d0).days <= 14]
    cand.sort(key=lambda c0: (abs(c0["K"] - S0), dt.date.fromisoformat(c0["exp"])))
    for con in cand[:3]:
        exp = dt.date.fromisoformat(con["exp"])
        end_day = max(et_date(t1) if t1 is not None else d0, exp)
        bars = bars_for(con["ticker"], str(d0), str(end_day))
        if not bars:
            continue
        bt = np.array([b["t"] for b in bars], dtype=np.int64)
        t0ms = int(t0.value // 1_000_000)
        i0 = int(np.searchsorted(bt, t0ms))
        if i0 >= len(bars) or bt[i0] - t0ms > 24 * 3600_000:
            continue
        entry = bars[i0]["c"] if bt[i0] == t0ms else bars[i0]["o"]
        if entry <= 0.03:
            continue
        if t1 is None:                                    # stock leg still open
            return entry, bars[-1]["c"], False
        if et_date(t1) > exp:                             # option died mid-trade
            Sx = day_close(ts_all, c_all, exp)
            if Sx is None:
                continue
            return entry, max(Sx - con["K"], 0.0), True
        t1ms = int(t1.value // 1_000_000)
        i1 = min(int(np.searchsorted(bt, t1ms)), len(bars) - 1)
        return entry, (bars[i1]["c"] if bt[i1] <= t1ms else bars[i1]["o"]), True
    return None


def vco_trades(vc):
    """Option twins of the dashboard's simulated vC trades (real option prices)."""
    out, hc = [], 0.01
    for t in vc:
        try:
            pr = _PREP.get((t["tk"], 60, "trend", "std"))
            if pr is None:
                continue
            ts_all, c_all = pr[0], pr[3]
            t0 = pd.Timestamp(t["entry_ts"], unit="s")
            t1 = pd.Timestamp(t["exit_ts"], unit="s") if t.get("exit_ts") else None
            leg = _opt_leg(t["tk"], t0, t1, t["entry"], ts_all, c_all)
            if leg is None:
                continue
            e, x, closed = leg
            ret = (x * (1 - hc) - e * (1 + hc)) / (e * (1 + hc))
            out.append(dict(strat="vCO", tk=t["tk"], entry_ts=t["entry_ts"],
                            exit_ts=t.get("exit_ts"), entry=round(float(e), 2),
                            exit=round(float(x), 2), target=None, stop=None,
                            qty=round(1000.0 / (e * 100), 1),
                            conviction=t["conviction"],
                            outcome=("OPEN" if not closed
                                     else ("CALL+" if ret > 0 else "CALL-")),
                            pnl=round(1000.0 * ret, 2)))
        except Exception as ex:
            print(f"  [vCO warn {t['tk']}] {ex}", flush=True)
    return out


def refresh():
    _PREP.clear()
    bot._DATA.clear()                # force a fresh Polygon tail fetch
    all_trades = []
    for strat, tks, mins, hbar, mode, tp, sl, featmode, sel, ddl in CONFIGS:
        for tk in tks:
            model = MODELS_MEM.get((strat, tk))
            if model is None:
                continue
            try:
                all_trades += sim_dash(strat, tk, mins, hbar, mode, tp, sl,
                                       featmode, sel, model)
            except Exception as e:
                print(f"  [sim warn {strat}/{tk}] {e}", flush=True)
    try:
        all_trades += vco_trades([t for t in all_trades if t["strat"] == "vC"])
    except Exception as e:
        print(f"  [vCO build warn] {e}", flush=True)
    tickers_out = {}
    for tk in TICKERS:
        d15 = (bot.full_series(tk).set_index("timestamp").resample("15min")
               .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                    close=("close", "last")).dropna().reset_index())
        d15 = d15[d15["timestamp"] >= PAPER_START - pd.Timedelta(days=1)]
        tickers_out[tk] = {
            "candles": [{"time": int(r.timestamp.timestamp()), "open": round(r.open, 2),
                         "high": round(r.high, 2), "low": round(r.low, 2),
                         "close": round(r.close, 2)} for r in d15.itertuples()],
            "trades": [t for t in all_trades if t["tk"] == tk],
        }
    STATE.update(ok=True, msg="", tickers=tickers_out,
                 strats=[cfg[0] for cfg in CONFIGS] + ["vCO"], descs=DESCS,
                 updated=time.strftime("%H:%M:%S"))
    Path("runs").mkdir(exist_ok=True)
    Path("runs/live_ledger.json").write_text(json.dumps(all_trades, indent=1))
    closed = [t for t in all_trades if t["outcome"] != "OPEN"]
    print(f"  refresh done {STATE['updated']}  closed={len(closed)} "
          f"open={len(all_trades) - len(closed)} "
          f"P&L=${sum(t['pnl'] for t in closed):+.2f}", flush=True)


def loop():
    jobs = [(cfg, tk) for cfg in CONFIGS for tk in cfg[1]]
    print(f"training {len(jobs)} strategy/ticker models (embargoed at {PAPER_START.date()}; "
          f"first run is slow, then cached)...", flush=True)
    try:
        for k, (cfg, tk) in enumerate(jobs, 1):
            strat, _, mins, hbar, mode, tp, sl, featmode, sel, ddl = cfg
            STATE["msg"] = f"training models {k}/{len(jobs)} ({strat}/{tk})..."
            MODELS_MEM[(strat, tk)] = train_dash(strat, tk, mins, hbar, mode,
                                                 tp, sl, featmode, sel)
            print(f"  [{k}/{len(jobs)}] {strat}/{tk} "
                  f"{'ok' if MODELS_MEM[(strat, tk)] else 'skipped'}", flush=True)
    except Exception:
        STATE["msg"] = "startup failed: " + traceback.format_exc()[-400:]
        print(STATE["msg"], flush=True)
        return
    while True:
        try:
            refresh()
        except Exception:
            print("refresh error:\n" + traceback.format_exc()[-600:], flush=True)
        time.sleep(REFRESH_S)


HTML = """<!doctype html><html><head><meta charset="utf-8"><title>capture_trader strategies</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
 body{font-family:Segoe UI,Arial,sans-serif;background:#111418;color:#dde1e6;margin:0;padding:14px}
 h2{margin:4px 0 10px;font-weight:600} .cards{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px}
 .card{background:#1a1f26;border-radius:8px;padding:10px 16px;min-width:110px}
 .card .k{font-size:12px;color:#8a919e}.card .v{font-size:20px;font-weight:600}
 .tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px}
 .tab{background:#1a1f26;border:1px solid #2a323d;border-radius:6px;padding:6px 14px;cursor:pointer}
 .tab.on{background:#2563eb;border-color:#2563eb;color:#fff}
 .tab .n{color:#8a919e;font-size:11px;margin-left:5px}.tab.on .n{color:#cfe0ff}
 .lbl{color:#8a919e;font-size:12px;margin:2px 4px 4px 0}
 #desc{background:#161b22;border-left:3px solid #2563eb;border-radius:4px;color:#aeb6c2;font-size:13px;padding:8px 12px;margin-bottom:10px}
 #chart{height:420px;border:1px solid #2a323d;border-radius:8px}
 table{width:100%;border-collapse:collapse;font-size:13px;margin-top:14px}
 th,td{padding:5px 8px;text-align:right;border-bottom:1px solid #232a33}
 th{color:#8a919e;font-weight:500} td:first-child,th:first-child{text-align:left}
 .g{color:#34d399}.r{color:#f87171}.o{color:#fbbf24}
 .note{color:#8a919e;font-size:12px;margin-top:8px}
</style></head><body>
<h2>capture_trader — what each strategy would have traded <span id="upd" style="font-size:13px;color:#8a919e"></span></h2>
<div class="cards" id="cards"></div>
<div class="lbl">strategy</div><div class="tabs" id="stabs"></div>
<div id="desc"></div>
<div class="lbl">ticker</div><div class="tabs" id="tabs"></div>
<div id="chart"></div>
<div class="note">markers: &#9650; entry &nbsp; &#9679; exit (green=target, red=stop, yellow=time) &nbsp;|&nbsp; dashed lines = open trade's target/stop &nbsp;|&nbsp; hypothetical fills at signal close, 5 bps cost, $1k/trade &nbsp;|&nbsp; data ~15 min delayed, refreshes every 5 min &nbsp;|&nbsp; real bot fills are in the daily report</div>
<table id="blot"><thead><tr><th>ticker</th><th>strat</th><th>entry (UTC)</th><th>entry</th>
<th>target</th><th>stop</th><th>conv</th><th>outcome</th><th>$ P&L</th></tr></thead><tbody></tbody></table>
<script>
let CUR="QQQ", CURS="ALL", chart, series, lines=[], SD=null;
async function jget(u){const r=await fetch(u);return r.json()}
async function ensureCandles(tk){
 if(SD&&SD.tickers[tk]&&!SD.tickers[tk].candles)
  SD.tickers[tk].candles=await jget('/candles?tk='+encodeURIComponent(tk));
}
function fmt(t){const d=new Date(t*1000);return d.toISOString().slice(5,16).replace("T"," ")}
function build(){
 chart=LightweightCharts.createChart(document.getElementById('chart'),{layout:{background:{color:'#111418'},textColor:'#8a919e'},grid:{vertLines:{color:'#1c2128'},horzLines:{color:'#1c2128'}},timeScale:{timeVisible:true,secondsVisible:false}});
 series=chart.addCandlestickSeries({upColor:'#34d399',downColor:'#f87171',borderVisible:false,wickUpColor:'#34d399',wickDownColor:'#f87171'});
}
function allTrades(S){const a=[];Object.values(S.tickers).forEach(x=>a.push(...x.trades));return a}
function match(t){return CURS==='ALL'||t.strat===CURS}
function render(S){
 document.getElementById('upd').textContent=' updated '+S.updated+' (local)';
 const all=allTrades(S), filt=all.filter(match);
 const closed=filt.filter(t=>t.outcome!=='OPEN');
 const wins=closed.filter(t=>t.outcome==='TARGET').length;
 const pnl=Math.round(closed.reduce((s,t)=>s+t.pnl,0)*100)/100;
 const nopen=filt.length-closed.length;
 document.getElementById('cards').innerHTML=
  `<div class="card"><div class="k">showing</div><div class="v">${CURS}</div></div>
   <div class="card"><div class="k">P&L since Jun 29</div><div class="v ${pnl>=0?'g':'r'}">$${pnl>=0?'+':''}${pnl}</div></div>
   <div class="card"><div class="k">closed / wins</div><div class="v">${closed.length} / ${wins}${closed.length?' ('+Math.round(100*wins/closed.length)+'%)':''}</div></div>
   <div class="card"><div class="k">open now</div><div class="v o">${nopen}</div></div>`;
 document.getElementById('desc').textContent=(S.descs&&S.descs[CURS])||'';
 const st=document.getElementById('stabs'); st.innerHTML='';
 ['ALL',...S.strats].forEach(s=>{const b=document.createElement('div');
   b.className='tab'+(s===CURS?' on':'');
   if(S.descs&&S.descs[s])b.title=S.descs[s];
   const n=(s==='ALL'?all:all.filter(t=>t.strat===s)).length;
   b.innerHTML=s+'<span class="n">'+n+'</span>';
   b.onclick=async()=>{CURS=s;
     if(s!=='ALL'){const mine=all.filter(t=>t.strat===s);
       if(mine.length&&!mine.some(t=>t.tk===CUR)){
         const cnt={};mine.forEach(t=>cnt[t.tk]=(cnt[t.tk]||0)+1);
         CUR=Object.keys(cnt).sort((a,b)=>cnt[b]-cnt[a])[0];}}
     await ensureCandles(CUR);render(S)};
   st.appendChild(b);});
 const tabs=document.getElementById('tabs'); tabs.innerHTML='';
 Object.keys(S.tickers).forEach(tk=>{const b=document.createElement('div');
   b.className='tab'+(tk===CUR?' on':'');
   const nOpen=S.tickers[tk].trades.filter(t=>match(t)&&t.outcome==='OPEN').length;
   b.textContent=tk+(nOpen?' ●':''); b.onclick=async()=>{CUR=tk;await ensureCandles(tk);render(S)}; tabs.appendChild(b);});
 const T=S.tickers[CUR]; if(!T)return;
 if(T.candles)series.setData(T.candles);
 lines.forEach(l=>series.removePriceLine(l)); lines=[];
 const mk=[];
 T.trades.filter(match).forEach(t=>{
   mk.push({time:t.entry_ts,position:'belowBar',color:'#60a5fa',shape:'arrowUp',text:t.strat+' buy '+t.entry});
   if(t.outcome==='OPEN'){
     if(t.target)lines.push(series.createPriceLine({price:t.target,color:'#34d399',lineStyle:2,title:t.strat+' target'}));
     if(t.stop)lines.push(series.createPriceLine({price:t.stop,color:'#f87171',lineStyle:2,title:t.strat+' stop'}));
     if(t.target)lines.push(series.createPriceLine({price:t.entry,color:'#60a5fa',lineStyle:3,title:t.strat+' entry'}));
   } else if(t.exit_ts){
     const col=(t.outcome==='TARGET'||t.outcome==='CALL+')?'#34d399':((t.outcome==='STOP'||t.outcome==='CALL-')?'#f87171':'#fbbf24');
     mk.push({time:t.exit_ts,position:'aboveBar',color:col,shape:'circle',text:t.outcome+' '+(t.pnl>=0?'+':'')+t.pnl});
   }});
 mk.sort((a,b)=>a.time-b.time); series.setMarkers(mk);
 const tb=document.querySelector('#blot tbody'); tb.innerHTML='';
 filt.sort((a,b)=>b.entry_ts-a.entry_ts);
 filt.forEach(t=>{const tr=document.createElement('tr');
   const cls=(t.outcome==='TARGET'||t.outcome==='CALL+')?'g':((t.outcome==='STOP'||t.outcome==='CALL-')?'r':(t.outcome==='OPEN'?'o':''));
   tr.innerHTML=`<td>${t.tk}</td><td>${t.strat}</td><td>${fmt(t.entry_ts)}</td><td>${t.entry}</td>
     <td>${t.target??'-'}</td><td>${t.stop??'-'}</td><td>${t.conviction}</td><td class="${cls}">${t.outcome}</td>
     <td class="${t.pnl>=0?'g':'r'}">${t.pnl>=0?'+':''}${t.pnl??''}</td>`;
   tb.appendChild(tr);});
}
async function poll(){
 try{const meta=await jget('/meta.json');
  if(meta.ok){
   const tr=await jget('/trades.json');
   SD={...meta,tickers:{}};
   Object.keys(tr).forEach(tk=>{SD.tickers[tk]={trades:tr[tk],candles:null}});
   await ensureCandles(CUR);
   render(SD);
  }else{document.getElementById('cards').innerHTML='<div class="card"><div class="v">'+meta.msg+'</div></div>'}
 }catch(e){}
 setTimeout(poll,60000);
}
build();poll();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        # Small split endpoints: something on this machine's loopback (AV filter
        # driver?) truncates large HTTP responses at 255/256-of-2^n boundaries no
        # matter how they are sent, so every payload here stays well under 64 KB
        # gzipped. Candles are served per ticker and fetched lazily by the page.
        ctype = "application/json"
        if self.path.startswith("/meta.json"):
            body = json.dumps({k: STATE.get(k) for k in
                               ("ok", "msg", "updated", "strats", "descs")}).encode()
        elif self.path.startswith("/trades.json"):
            body = json.dumps({tk: v["trades"] for tk, v in
                               STATE.get("tickers", {}).items()}).encode()
        elif self.path.startswith("/candles"):
            tk = (self.path.split("tk=")[-1].split("&")[0]
                  if "tk=" in self.path else "")
            body = json.dumps(STATE.get("tickers", {})
                              .get(tk, {}).get("candles", [])).encode()
        elif self.path.startswith("/state.json"):     # legacy full dump
            body = json.dumps(STATE).encode()
        else:
            body = HTML.encode()
            ctype = "text/html; charset=utf-8"
        # gzip big payloads: something in the Windows loopback path truncates large
        # plain responses at ~510 KiB even with sendall; compressed JSON is ~70 KB.
        import gzip as _gz
        enc = None
        if len(body) > 100_000 and "gzip" in (self.headers.get("Accept-Encoding") or ""):
            body = _gz.compress(body, 5)
            enc = "gzip"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        if enc:
            self.send_header("Content-Encoding", enc)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.connection.sendall(body)
        # graceful close: Windows aborts the connection (RST, discarding in-flight
        # bytes) if we close right after a big write. FIN first, then drain.
        import socket as _sk
        try:
            self.connection.shutdown(_sk.SHUT_WR)
            self.connection.settimeout(3)
            while self.connection.recv(4096):
                pass
        except Exception:
            pass

    def log_message(self, *a):
        pass


def main():
    threading.Thread(target=loop, daemon=True).start()
    print(f"dashboard at  http://localhost:{PORT}   (Ctrl+C to stop)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()


if __name__ == "__main__":
    main()

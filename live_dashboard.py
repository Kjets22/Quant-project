"""
live_dashboard.py — live paper-trading dashboard (research/paper only; places no orders).

  python live_dashboard.py        then open  http://localhost:8765

- Trains the frozen v3 (30m/1.5:1) + v4 (15m/4:1) models once at startup on data BEFORE
  the paper start (embargoed, audited engine), for all 8 basket names.
- Background loop: every REFRESH_S seconds pulls the latest (15-min-delayed) Polygon bars,
  recomputes all paper trades since PAPER_START, and updates the in-memory state.
- Serves a browser UI: candlestick chart per ticker with entry/exit markers, target/stop
  lines for open trades, a live blotter, and account P&L ($10k, 10% per trade, 5 bps cost).
"""

from __future__ import annotations

import json
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

from triple_barrier_ml import features, label
from triple_barrier_breadth import TICKERS
from sr_features import sr_features
from basket import ticker_cfg
from data import fetch_polygon

PAPER_START = pd.Timestamp("2026-06-29")
CONFIGS = [("v3", 30, 1.5, 1.0), ("v4", 15, 4.0, 1.0)]
SEL_Q, HBAR = 0.93, 24
EFF_COST = 5.0 / 1e4                 # 3 bps + 2 bps slippage
MIN_ATR_PCT = 0.0012
ACCOUNT, NOTIONAL_PCT = 10_000.0, 10.0
REFRESH_S = 300
PORT = 8765

STATE = {"ok": False, "msg": "starting up: training models (takes a few minutes)..."}
MODELS = {}
HIST = {}          # tk -> historical 5-min df (cache + june fetch), fixed at startup


def atr_fixed(h, l, c, n=24):
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    return pd.Series(tr).rolling(n).mean().to_numpy()


def fetch_recent(tk, start="2026-06-20"):
    cfg = ticker_cfg(tk)
    cfg.data.start_date = start
    cfg.data.end_date = str((pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).date())
    cfg.data.multiplier, cfg.data.timespan = 5, "minute"
    return fetch_polygon(cfg)


def stitched(tk, live_df=None):
    df = HIST[tk]
    if live_df is not None:
        df = (pd.concat([df, live_df], ignore_index=True)
                .drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp"))
    return df


def frame(tk, mins, live_df=None):
    d = stitched(tk, live_df).set_index("timestamp").resample(f"{mins}min").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum")).dropna().reset_index()
    ts = pd.to_datetime(d["timestamp"]).to_numpy()
    o, h, l, c, v = (d[x].to_numpy(float) for x in ("open", "high", "low", "close", "volume"))
    A = atr_fixed(h, l, c)
    X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                   sr_features(d).reset_index(drop=True)], axis=1)
    return ts, o, h, l, c, A, X


def train_all():
    for strat, mins, tp, sl in CONFIGS:
        for tk in TICKERS:
            ts, o, h, l, c, A, X = frame(tk, mins)
            y = label(h, l, c, A, tp, sl)
            fv = (X.notna().all(axis=1) & np.isfinite(y) & np.isfinite(A)).to_numpy()
            tr = np.where(fv & (ts < np.datetime64(PAPER_START)))[0]
            tr = tr[:-HBAR] if len(tr) > HBAR else tr          # embargo
            if len(tr) < 500:
                continue
            clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                     min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                                     reg_lambda=1.0, verbose=-1)
            clf.fit(X.iloc[tr], y[tr].astype(int))
            thr = float(np.quantile(clf.predict_proba(X.iloc[tr])[:, 1], SEL_Q))
            MODELS[(strat, tk)] = (clf, thr)
            print(f"  trained {strat}/{tk}  thr={thr:.3f}", flush=True)


def simulate(tk, live_df):
    """All paper trades since PAPER_START for both strategies on one ticker."""
    trades = []
    for strat, mins, tp, sl in CONFIGS:
        if (strat, tk) not in MODELS:
            continue
        clf, thr = MODELS[(strat, tk)]
        ts, o, h, l, c, A, X = frame(tk, mins, live_df)
        fv = (X.notna().all(axis=1) & np.isfinite(A)).to_numpy()
        n = len(c)
        fwd = np.where(fv & (ts >= np.datetime64(PAPER_START)))[0]
        if len(fwd) == 0:
            continue
        proba = {int(ix): float(p) for ix, p in
                 zip(fwd, clf.predict_proba(X.iloc[fwd])[:, 1])}
        i, last = int(fwd[0]), int(fwd[-1])
        notional = ACCOUNT * NOTIONAL_PCT / 100.0
        while i <= last:
            pr = proba.get(i, -1.0)
            if pr < thr or A[i] / c[i] < MIN_ATR_PCT:
                i += 1; continue
            a = A[i]; up, dn = c[i] + tp * a, c[i] - sl * a
            res, j = None, i + 1
            while j < min(i + HBAR + 1, n):
                if l[j] <= dn:
                    res = 0; break
                if h[j] >= up:
                    res = 1; break
                j += 1
            sh = notional / c[i]
            base = dict(strat=strat, tk=tk,
                        entry_ts=int(pd.Timestamp(ts[i]).timestamp()),
                        entry=round(float(c[i]), 2), target=round(float(up), 2),
                        stop=round(float(dn), 2), shares=round(sh, 2), conviction=round(pr, 3))
            if res is None and j >= n:                       # still open
                base.update(outcome="OPEN", exit_ts=None, exit=None,
                            pnl=round(sh * (c[n - 1] - c[i]), 2), last=round(float(c[n - 1]), 2))
                trades.append(base); break
            if res is None:                                  # time barrier
                ex_j = min(j, n - 1); px = c[ex_j]
                oc = "TIME+" if px > c[i] else "TIME-"
            else:
                ex_j = j; px = up if res == 1 else dn
                oc = "TARGET" if res == 1 else "STOP"
            base.update(outcome=oc, exit_ts=int(pd.Timestamp(ts[ex_j]).timestamp()),
                        exit=round(float(px), 2),
                        pnl=round(sh * (px - c[i]) - notional * EFF_COST, 2))
            trades.append(base)
            i = ex_j + 1 if res is None else j + 1
    return trades


def refresh():
    tickers_out, all_trades = {}, []
    for tk in TICKERS:
        try:
            live = fetch_recent(tk)
        except Exception as e:
            print(f"  [fetch warn {tk}] {e}", flush=True)
            live = None
        trades = simulate(tk, live)
        all_trades += trades
        d15 = stitched(tk, live).set_index("timestamp").resample("15min").agg(
            open=("open", "first"), high=("high", "max"), low=("low", "min"),
            close=("close", "last")).dropna().reset_index()
        d15 = d15[d15["timestamp"] >= PAPER_START - pd.Timedelta(days=1)]
        tickers_out[tk] = {
            "candles": [{"time": int(r.timestamp.timestamp()), "open": round(r.open, 2),
                         "high": round(r.high, 2), "low": round(r.low, 2),
                         "close": round(r.close, 2)} for r in d15.itertuples()],
            "trades": [t for t in all_trades if t["tk"] == tk],
        }
    closed = [t for t in all_trades if t["outcome"] not in ("OPEN",)]
    wins = sum(t["outcome"] == "TARGET" for t in closed)
    pnl = sum(t["pnl"] for t in closed)
    nopen = sum(t["outcome"] == "OPEN" for t in all_trades)
    STATE.update(ok=True, msg="", tickers=tickers_out, updated=time.strftime("%H:%M:%S"),
                 summary={"account": round(ACCOUNT + pnl, 2), "pnl": round(pnl, 2),
                          "closed": len(closed), "wins": wins,
                          "win_pct": round(100 * wins / len(closed), 1) if closed else None,
                          "open": nopen})
    Path("runs").mkdir(exist_ok=True)
    Path("runs/live_ledger.json").write_text(json.dumps(all_trades, indent=1))
    print(f"  refresh done {STATE['updated']}  closed={len(closed)} open={nopen} "
          f"P&L=${pnl:+.2f}", flush=True)


def loop():
    print("training models (one-time, a few minutes)...", flush=True)
    try:
        train_all()
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


HTML = """<!doctype html><html><head><meta charset="utf-8"><title>capture_trader live</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
 body{font-family:Segoe UI,Arial,sans-serif;background:#111418;color:#dde1e6;margin:0;padding:14px}
 h2{margin:4px 0 10px;font-weight:600} .cards{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px}
 .card{background:#1a1f26;border-radius:8px;padding:10px 16px;min-width:110px}
 .card .k{font-size:12px;color:#8a919e}.card .v{font-size:20px;font-weight:600}
 .tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}
 .tab{background:#1a1f26;border:1px solid #2a323d;border-radius:6px;padding:6px 14px;cursor:pointer}
 .tab.on{background:#2563eb;border-color:#2563eb;color:#fff}
 #chart{height:420px;border:1px solid #2a323d;border-radius:8px}
 table{width:100%;border-collapse:collapse;font-size:13px;margin-top:14px}
 th,td{padding:5px 8px;text-align:right;border-bottom:1px solid #232a33}
 th{color:#8a919e;font-weight:500} td:first-child,th:first-child{text-align:left}
 .g{color:#34d399}.r{color:#f87171}.o{color:#fbbf24}
 .note{color:#8a919e;font-size:12px;margin-top:8px}
</style></head><body>
<h2>capture_trader — live paper dashboard <span id="upd" style="font-size:13px;color:#8a919e"></span></h2>
<div class="cards" id="cards"></div>
<div class="tabs" id="tabs"></div>
<div id="chart"></div>
<div class="note">markers: &#9650; entry &nbsp; &#9679; exit (green=target, red=stop, yellow=time) &nbsp;|&nbsp; dashed lines = open trade's target/stop &nbsp;|&nbsp; data ~15 min delayed (Polygon), refreshes every 5 min</div>
<table id="blot"><thead><tr><th>ticker</th><th>strat</th><th>entry (UTC)</th><th>entry</th>
<th>target</th><th>stop</th><th>outcome</th><th>$ P&L</th></tr></thead><tbody></tbody></table>
<script>
let CUR="SPY", chart, series, lines=[];
function fmt(t){const d=new Date(t*1000);return d.toISOString().slice(5,16).replace("T"," ")}
function build(){
 chart=LightweightCharts.createChart(document.getElementById('chart'),{layout:{background:{color:'#111418'},textColor:'#8a919e'},grid:{vertLines:{color:'#1c2128'},horzLines:{color:'#1c2128'}},timeScale:{timeVisible:true,secondsVisible:false}});
 series=chart.addCandlestickSeries({upColor:'#34d399',downColor:'#f87171',borderVisible:false,wickUpColor:'#34d399',wickDownColor:'#f87171'});
}
function render(S){
 document.getElementById('upd').textContent=' updated '+S.updated+' (local)';
 const s=S.summary;
 document.getElementById('cards').innerHTML=
  `<div class="card"><div class="k">account</div><div class="v">$${s.account.toLocaleString()}</div></div>
   <div class="card"><div class="k">P&L since Jun 29</div><div class="v ${s.pnl>=0?'g':'r'}">$${s.pnl>=0?'+':''}${s.pnl}</div></div>
   <div class="card"><div class="k">closed / wins</div><div class="v">${s.closed} / ${s.wins}${s.win_pct!==null?' ('+s.win_pct+'%)':''}</div></div>
   <div class="card"><div class="k">open now</div><div class="v o">${s.open}</div></div>`;
 const tabs=document.getElementById('tabs'); tabs.innerHTML='';
 Object.keys(S.tickers).forEach(tk=>{const b=document.createElement('div');
   b.className='tab'+(tk===CUR?' on':'');
   const nOpen=S.tickers[tk].trades.filter(t=>t.outcome==='OPEN').length;
   b.textContent=tk+(nOpen?' ●':''); b.onclick=()=>{CUR=tk;render(S)}; tabs.appendChild(b);});
 const T=S.tickers[CUR]; if(!T)return;
 series.setData(T.candles);
 lines.forEach(l=>series.removePriceLine(l)); lines=[];
 const mk=[];
 T.trades.forEach(t=>{
   mk.push({time:t.entry_ts,position:'belowBar',color:'#60a5fa',shape:'arrowUp',text:t.strat+' buy '+t.entry});
   if(t.outcome==='OPEN'){
     lines.push(series.createPriceLine({price:t.target,color:'#34d399',lineStyle:2,title:t.strat+' target'}));
     lines.push(series.createPriceLine({price:t.stop,color:'#f87171',lineStyle:2,title:t.strat+' stop'}));
     lines.push(series.createPriceLine({price:t.entry,color:'#60a5fa',lineStyle:3,title:t.strat+' entry'}));
   } else {
     const col=t.outcome==='TARGET'?'#34d399':(t.outcome==='STOP'?'#f87171':'#fbbf24');
     mk.push({time:t.exit_ts,position:'aboveBar',color:col,shape:'circle',text:t.outcome+' '+(t.pnl>=0?'+':'')+t.pnl});
   }});
 mk.sort((a,b)=>a.time-b.time); series.setMarkers(mk);
 const tb=document.querySelector('#blot tbody'); tb.innerHTML='';
 const all=[]; Object.values(S.tickers).forEach(x=>all.push(...x.trades));
 all.sort((a,b)=>b.entry_ts-a.entry_ts);
 all.forEach(t=>{const tr=document.createElement('tr');
   const cls=t.outcome==='TARGET'?'g':(t.outcome==='STOP'?'r':(t.outcome==='OPEN'?'o':''));
   tr.innerHTML=`<td>${t.tk}</td><td>${t.strat}</td><td>${fmt(t.entry_ts)}</td><td>${t.entry}</td>
     <td>${t.target}</td><td>${t.stop}</td><td class="${cls}">${t.outcome}</td>
     <td class="${t.pnl>=0?'g':'r'}">${t.pnl>=0?'+':''}${t.pnl??''}</td>`;
   tb.appendChild(tr);});
}
async function poll(){
 try{const r=await fetch('/state.json');const S=await r.json();
  if(S.ok){render(S)}else{document.getElementById('cards').innerHTML='<div class="card"><div class="v">'+S.msg+'</div></div>'}
 }catch(e){}
 setTimeout(poll,60000);
}
build();poll();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/state.json"):
            body = json.dumps(STATE).encode()
            ctype = "application/json"
        else:
            body = HTML.encode()
            ctype = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def main():
    print("loading historical caches...", flush=True)
    end = str((pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).date())
    for tk in TICKERS:
        df = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv",
                         parse_dates=["timestamp"])
        p = Path(f"data_cache/{tk}_recent_2026-06-01_{end}.csv")
        if p.exists():
            rec = pd.read_csv(p, parse_dates=["timestamp"])
            df = (pd.concat([df, rec], ignore_index=True)
                    .drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp"))
        HIST[tk] = df
    threading.Thread(target=loop, daemon=True).start()
    print(f"dashboard at  http://localhost:{PORT}   (Ctrl+C to stop)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()


if __name__ == "__main__":
    main()

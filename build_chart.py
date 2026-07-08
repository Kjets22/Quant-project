"""build_chart.py — render NVDA last-week candles + the two 1.5:1 trades to an SVG."""
import json
from pathlib import Path

C = [["06-23 08:00",202.32,206.26,201.66,202.45],["06-23 09:00",202.5,202.9,202.12,202.4],
["06-23 10:00",202.35,202.76,201.85,202.65],["06-23 11:00",202.63,204.15,202.21,203.88],
["06-23 12:00",203.88,204.0,202.6,203.01],["06-23 13:00",203.03,203.77,200.04,203.41],
["06-23 14:00",203.37,203.58,201.9,202.24],["06-23 15:00",202.24,202.49,200.43,201.85],
["06-23 16:00",201.83,202.96,201.8,202.31],["06-23 17:00",202.32,202.8,201.61,201.61],
["06-23 18:00",201.62,202.01,200.3,200.63],["06-23 19:00",200.62,201.72,200.0,200.01],
["06-23 20:00",200.14,200.81,200.0,200.48],["06-23 21:00",200.46,201.1,200.46,200.98],
["06-23 22:00",200.93,201.17,200.4,200.53],["06-23 23:00",200.56,200.9,200.2,200.89],
["06-24 08:00",201.0,201.8,200.61,201.25],["06-24 09:00",201.3,201.66,201.19,201.45],
["06-24 10:00",201.3,201.73,201.29,201.4],["06-24 11:00",201.34,201.7,200.85,201.5],
["06-24 12:00",201.5,201.65,200.88,200.96],["06-24 13:00",201.0,201.67,199.0,200.25],
["06-24 14:00",200.25,201.3,198.6,201.19],["06-24 15:00",201.21,201.51,200.34,200.79],
["06-24 16:00",200.78,201.02,199.13,199.26],["06-24 17:00",199.24,199.56,197.39,197.46],
["06-24 18:00",197.47,198.77,197.04,197.21],["06-24 19:00",197.2,199.23,196.58,198.96],
["06-24 20:00",199.01,201.25,198.11,200.37],["06-24 21:00",200.45,200.5,199.9,200.5],
["06-24 22:00",200.5,200.84,199.75,199.85],["06-24 23:00",199.85,200.88,199.8,200.66]]

TR = [dict(ei=4,en=203.01,tg=205.22,sl=201.54,xi=5,outcome="STOP"),
      dict(ei=25,en=197.46,tg=199.63,sl=196.01,xi=28,outcome="TARGET")]

W,H = 720,450
L,Rm,T,B = 50,14,18,46
pw,ph = W-L-Rm, H-T-B
n = len(C)
slot = pw/n
ymin,ymax = 195.5,207.0
def x(i): return L + (i+0.5)*slot
def y(p): return T + (1-(p-ymin)/(ymax-ymin))*ph

UP,DN,GR,RD,BL = "#1baf7a","#e34948","#1baf7a","#e34948","#2a78d6"
s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" '
     f'aria-label="NVDA hourly candles for June 23-24 2026 with two bracket trades: a stop-out loss and a take-profit win">']
s.append('<title>NVDA last week — two 1.5:1 bracket trades</title>')

# gridlines + y labels
for p in range(196,207,2):
    yp = y(p)
    s.append(f'<line x1="{L}" y1="{yp:.1f}" x2="{W-Rm}" y2="{yp:.1f}" stroke="var(--border)" stroke-width="1"/>')
    s.append(f'<text x="{L-6}" y="{yp+4:.1f}" text-anchor="end" font-size="11" fill="var(--text-muted)">{p}</text>')
# day divider + day labels
xb = L+16*slot
s.append(f'<line x1="{xb:.1f}" y1="{T}" x2="{xb:.1f}" y2="{T+ph}" stroke="var(--border)" stroke-width="1" stroke-dasharray="2 3"/>')
s.append(f'<text x="{L+8*slot:.0f}" y="{H-16}" text-anchor="middle" font-size="12" fill="var(--text-secondary)">Mon Jun 23</text>')
s.append(f'<text x="{L+24*slot:.0f}" y="{H-16}" text-anchor="middle" font-size="12" fill="var(--text-secondary)">Tue Jun 24</text>')

# candles
bw = slot*0.55
for i,(lab,o,h,lo,c) in enumerate(C):
    cx = x(i); col = UP if c>=o else DN
    s.append(f'<line x1="{cx:.1f}" y1="{y(h):.1f}" x2="{cx:.1f}" y2="{y(lo):.1f}" stroke="{col}" stroke-width="1.2"/>')
    yo,yc = y(o),y(c); top=min(yo,yc); hgt=max(abs(yo-yc),1.2)
    s.append(f'<rect x="{cx-bw/2:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{hgt:.1f}" fill="{col}"/>')

for ti,t in enumerate(TR):
    num = ti+1
    xe,xx = x(t["ei"]), x(t["xi"])
    ytg,ysl,yen = y(t["tg"]),y(t["sl"]),y(t["en"])
    x0,x1 = xe-slot*0.6, xx+slot*0.6
    # target (green) + stop (red) bracket lines spanning the trade
    s.append(f'<line x1="{x0:.1f}" y1="{ytg:.1f}" x2="{x1:.1f}" y2="{ytg:.1f}" stroke="{GR}" stroke-width="1.4" stroke-dasharray="5 3"/>')
    s.append(f'<line x1="{x0:.1f}" y1="{ysl:.1f}" x2="{x1:.1f}" y2="{ysl:.1f}" stroke="{RD}" stroke-width="1.4" stroke-dasharray="5 3"/>')
    won = t["outcome"]=="TARGET"
    # exit dot at the level that got hit
    s.append(f'<circle cx="{xx:.1f}" cy="{(ytg if won else ysl):.1f}" r="5" fill="{GR if won else RD}"/>')
    # numbered blue entry dot
    s.append(f'<circle cx="{xe:.1f}" cy="{yen:.1f}" r="8" fill="{BL}"/>')
    s.append(f'<text x="{xe:.1f}" y="{yen+4:.1f}" text-anchor="middle" font-size="11" fill="#ffffff" font-weight="500">{num}</text>')

SEC = "var(--text-secondary)"
def callout(bx,by,w,lines):
    h = 14*len(lines)+12
    s.append(f'<rect x="{bx}" y="{by}" width="{w}" height="{h}" rx="6" fill="var(--surface-2)" stroke="{lines[0][1]}" stroke-width="1.5"/>')
    for k,(txt,tc,wt) in enumerate(lines):
        s.append(f'<text x="{bx+11}" y="{by+20+14*k}" font-size="12" fill="{tc}" font-weight="{wt}">{txt}</text>')

callout(58,298,200,[("1   stopped out",RD,"500"),("buy 203.01",SEC,"400"),
 ("target 205.22 (not hit)",GR,"400"),("stop 201.54 hit next bar",RD,"400"),
 ("result:  -1.0R  (-$1.47)",RD,"500")])
callout(450,56,202,[("2   took profit",GR,"500"),("buy 197.46",SEC,"400"),
 ("target 199.63 hit",GR,"400"),("stop 196.01 (not hit)",RD,"400"),
 ("result:  +1.5R  (+$2.21)",GR,"500")])

s.append('</svg>')
Path("runs").mkdir(exist_ok=True)
Path("runs/nvda_chart.svg").write_text("\n".join(s), encoding="utf-8")
print("wrote runs/nvda_chart.svg", len("\n".join(s)), "chars")

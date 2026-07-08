"""build_nvda_chart.py — render the clean NVDA v3 trade (entry -> target) as an SVG."""
from pathlib import Path

C = [["08:00",209.95,210.1,208.3,209.31],["08:30",209.27,209.85,209.2,209.57],
     ["09:00",209.51,209.68,209.3,209.4],["09:30",209.41,209.46,209.15,209.3],
     ["10:00",209.27,209.98,209.27,209.6],["10:30",209.56,209.6,209.3,209.55],
     ["11:00",209.58,210.66,209.52,210.14],["11:30",210.15,210.37,209.99,210.05],
     ["12:00",210.12,210.31,209.87,210.3],["12:30",210.22,211.0,210.22,210.74],
     ["13:00",210.78,211.47,210.62,211.44],["13:30",211.44,213.99,210.79,213.08],
     ["14:00",213.08,213.99,211.11,211.11]]
EI, XI = 2, 9
ENTRY, TGT, STOP = 209.40, 210.83, 208.45

W, H = 720, 446
L, Rm, T, B = 52, 14, 20, 46
pw, ph = W - L - Rm, H - T - B
n = len(C); slot = pw / n; bw = slot * 0.5
ymin, ymax = 208.0, 214.5
def x(i): return L + (i + 0.5) * slot
def y(p): return T + (1 - (p - ymin) / (ymax - ymin)) * ph
UP, DN, GR, RD, BL = "#1baf7a", "#e34948", "#1baf7a", "#e34948", "#2a78d6"

s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" '
     f'aria-label="NVDA 30-minute candles on Mon June 22. Bought at 209.40, target hit at 210.83 by 12:30 for a +20.40 dollar gain on a 2994 dollar position.">']
s.append('<title>NVDA v3 trade — bought 209.40, sold at target 210.83</title>')
# y gridlines
for p in (209, 210, 211, 212, 213, 214):
    yp = y(p)
    s.append(f'<line x1="{L}" y1="{yp:.1f}" x2="{W-Rm}" y2="{yp:.1f}" stroke="var(--border)" stroke-width="1"/>')
    s.append(f'<text x="{L-6}" y="{yp+4:.1f}" text-anchor="end" font-size="11" fill="var(--text-muted)">{p}</text>')
s.append(f'<text x="{W/2:.0f}" y="{H-14}" text-anchor="middle" font-size="12" fill="var(--text-secondary)">Mon Jun 22 (UTC, 30-min candles)</text>')
# candles
for i, (lab, o, hi, lo, c) in enumerate(C):
    cx = x(i); col = UP if c >= o else DN
    s.append(f'<line x1="{cx:.1f}" y1="{y(hi):.1f}" x2="{cx:.1f}" y2="{y(lo):.1f}" stroke="{col}" stroke-width="1.3"/>')
    yo, yc = y(o), y(c); top = min(yo, yc); ht = max(abs(yo - yc), 1.2)
    s.append(f'<rect x="{cx-bw/2:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{ht:.1f}" fill="{col}"/>')
# target / stop lines across the trade
x0, x1 = x(EI) - slot * 0.5, x(XI) + slot * 0.5
s.append(f'<line x1="{x0:.1f}" y1="{y(TGT):.1f}" x2="{x1:.1f}" y2="{y(TGT):.1f}" stroke="{GR}" stroke-width="1.5" stroke-dasharray="5 3"/>')
s.append(f'<text x="{x1+4:.1f}" y="{y(TGT)-4:.1f}" font-size="11" fill="{GR}">target {TGT} — sell here</text>')
s.append(f'<line x1="{x0:.1f}" y1="{y(STOP):.1f}" x2="{x1:.1f}" y2="{y(STOP):.1f}" stroke="{RD}" stroke-width="1.5" stroke-dasharray="5 3"/>')
s.append(f'<text x="{x1+4:.1f}" y="{y(STOP)+13:.1f}" font-size="11" fill="{RD}">stop {STOP}</text>')
# entry marker
s.append(f'<circle cx="{x(EI):.1f}" cy="{y(ENTRY):.1f}" r="6" fill="{BL}"/>')
s.append(f'<text x="{x(EI)-10:.1f}" y="{y(ENTRY)+4:.1f}" text-anchor="end" font-size="11" fill="{BL}" font-weight="500">BUY {ENTRY}</text>')
# exit marker (target hit)
s.append(f'<circle cx="{x(XI):.1f}" cy="{y(TGT):.1f}" r="6" fill="{GR}"/>')
s.append(f'<text x="{x(XI):.1f}" y="{y(TGT)-12:.1f}" text-anchor="middle" font-size="11" fill="{GR}" font-weight="500">SOLD +$20.40</text>')
# summary box (upper-left empty area)
bx, by, bw2 = 60, 34, 266
s.append(f'<rect x="{bx}" y="{by}" width="{bw2}" height="86" rx="8" fill="var(--surface-2)" stroke="var(--border)" stroke-width="1"/>')
rows = [("NVDA  —  v3 (30-min, 1.5:1)", "var(--text-primary)", "500"),
        ("bought 14.3 shares = $2,994  (30% of acct)", "var(--text-secondary)", "400"),
        ("risk if stopped: -$13.60   reward at target: +$20.40", "var(--text-secondary)", "400"),
        ("result: TARGET HIT  ->  +$20.40", GR, "500")]
for k, (txt, col, wt) in enumerate(rows):
    s.append(f'<text x="{bx+12}" y="{by+22+k*19}" font-size="12.5" fill="{col}" font-weight="{wt}">{txt}</text>')
s.append('</svg>')
Path("runs").mkdir(exist_ok=True)
Path("runs/nvda_trade.svg").write_text("\n".join(s), encoding="utf-8")
print("wrote runs/nvda_trade.svg")

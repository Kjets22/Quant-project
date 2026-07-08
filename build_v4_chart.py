"""build_v4_chart.py — render the clean NVDA v4 (15-min/4:1) trade as an SVG."""
from pathlib import Path

C = [["10:30",209.86,209.9,209.59,209.72],["10:45",209.64,209.95,209.6,209.89],
     ["11:00",209.88,210.12,209.58,209.72],["11:15",209.74,209.75,209.39,209.51],
     ["11:30",209.54,209.59,209.08,209.5],["11:45",209.55,209.6,209.17,209.22],
     ["12:00",209.2,209.34,209.1,209.12],["12:15",209.11,209.19,208.78,208.98],
     ["12:30",208.97,209.43,208.97,209.24],["12:45",209.3,209.55,209.08,209.54],
     ["13:00",209.54,209.85,208.64,209.5],["13:15",209.53,210.8,209.3,210.66],
     ["13:30",210.62,211.4,208.83,210.16],["13:45",210.15,210.96,209.03,209.16],
     ["14:00",209.11,209.19,206.8,206.92],["14:15",206.91,207.36,205.89,206.98]]
EI, XI = 6, 12
ENTRY, TGT, STOP = 209.12, 211.09, 208.63

W, H = 720, 452
L, Rm, T, B = 52, 14, 20, 48
pw, ph = W - L - Rm, H - T - B
n = len(C); slot = pw / n; bw = slot * 0.5
ymin, ymax = 205.5, 211.9
def x(i): return L + (i + 0.5) * slot
def y(p): return T + (1 - (p - ymin) / (ymax - ymin)) * ph
UP, DN, GR, RD, BL = "#1baf7a", "#e34948", "#1baf7a", "#e34948", "#2a78d6"

s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" '
     f'aria-label="NVDA 15-minute candles June 9. Bought 209.12, ran to the 4-to-1 target 211.09 by 13:30 for +28.12, then NVDA reversed and fell after the exit.">']
s.append('<title>NVDA v4 trade — bought 209.12, sold at 4:1 target 211.09</title>')
for p in (206, 207, 208, 209, 210, 211):
    yp = y(p)
    s.append(f'<line x1="{L}" y1="{yp:.1f}" x2="{W-Rm}" y2="{yp:.1f}" stroke="var(--border)" stroke-width="1"/>')
    s.append(f'<text x="{L-6}" y="{yp+4:.1f}" text-anchor="end" font-size="11" fill="var(--text-muted)">{p}</text>')
s.append(f'<text x="{W/2:.0f}" y="{H-14}" text-anchor="middle" font-size="12" fill="var(--text-secondary)">Tue Jun 9 (UTC, 15-min candles)</text>')
for i, (lab, o, hi, lo, c) in enumerate(C):
    cx = x(i); col = UP if c >= o else DN
    s.append(f'<line x1="{cx:.1f}" y1="{y(hi):.1f}" x2="{cx:.1f}" y2="{y(lo):.1f}" stroke="{col}" stroke-width="1.3"/>')
    yo, yc = y(o), y(c); top = min(yo, yc); ht = max(abs(yo - yc), 1.2)
    s.append(f'<rect x="{cx-bw/2:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{ht:.1f}" fill="{col}"/>')
x0, x1 = x(EI) - slot * 0.5, x(XI) + slot * 0.5
s.append(f'<line x1="{x0:.1f}" y1="{y(TGT):.1f}" x2="{x1:.1f}" y2="{y(TGT):.1f}" stroke="{GR}" stroke-width="1.5" stroke-dasharray="5 3"/>')
s.append(f'<text x="{x1+4:.1f}" y="{y(TGT)+4:.1f}" font-size="11" fill="{GR}">target {TGT} (4x risk)</text>')
s.append(f'<line x1="{x0:.1f}" y1="{y(STOP):.1f}" x2="{x1:.1f}" y2="{y(STOP):.1f}" stroke="{RD}" stroke-width="1.5" stroke-dasharray="5 3"/>')
s.append(f'<text x="{x1+4:.1f}" y="{y(STOP)+4:.1f}" font-size="11" fill="{RD}">stop {STOP}</text>')
s.append(f'<circle cx="{x(EI):.1f}" cy="{y(ENTRY):.1f}" r="6" fill="{BL}"/>')
s.append(f'<text x="{x(EI):.1f}" y="{y(ENTRY)+20:.1f}" text-anchor="middle" font-size="11" fill="{BL}" font-weight="500">BUY {ENTRY}</text>')
s.append(f'<circle cx="{x(XI):.1f}" cy="{y(TGT):.1f}" r="6" fill="{GR}"/>')
s.append(f'<text x="{x(XI):.1f}" y="{y(TGT)-11:.1f}" text-anchor="middle" font-size="11" fill="{GR}" font-weight="500">SOLD +$28.12</text>')
# reversal note pointing at the crash candles
s.append(f'<text x="{x(14):.1f}" y="{y(207.7):.1f}" text-anchor="middle" font-size="11" fill="var(--text-muted)">reversed after</text>')
s.append(f'<text x="{x(14):.1f}" y="{y(207.7)+13:.1f}" text-anchor="middle" font-size="11" fill="var(--text-muted)">we were out</text>')
# summary box (lower-left empty area)
bx, by, bw2 = 58, 286, 290
s.append(f'<rect x="{bx}" y="{by}" width="{bw2}" height="86" rx="8" fill="var(--surface-2)" stroke="var(--border)" stroke-width="1"/>')
rows = [("NVDA  —  v4 (15-min, 4:1)", "var(--text-primary)", "500"),
        ("bought 14.3 shares = $2,990  (30% of acct)", "var(--text-secondary)", "400"),
        ("risk if stopped: -$7.03   reward at target: +$28.12", "var(--text-secondary)", "400"),
        ("result: 4:1 TARGET HIT  ->  +$28.12", GR, "500")]
for k, (txt, col, wt) in enumerate(rows):
    s.append(f'<text x="{bx+12}" y="{by+22+k*19}" font-size="12" fill="{col}" font-weight="{wt}">{txt}</text>')
s.append('</svg>')
Path("runs").mkdir(exist_ok=True)
Path("runs/v4_trade.svg").write_text("\n".join(s), encoding="utf-8")
print("wrote runs/v4_trade.svg")

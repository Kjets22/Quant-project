"""build_heatmap.py — render the timeframe x target:stop sweep as an SVG heatmap."""
from pathlib import Path

TFS = ["60-min", "30-min", "15-min", "5-min"]
ROWS = ["1:1", "1.5:1", "2:1", "3:1", "4:1"]
# total% , then mean bps  (rows=ratio, cols=tf)
TOTAL = {
 "1:1":   [52, 70, -33, -373], "1.5:1": [61, 114, 21, -263], "2:1": [3, 85, 32, -250],
 "3:1":   [-57, 101, 114, -200], "4:1": [-37, 64, 129, -100]}
BPS = {
 "1:1":   [2.7, 2.2, -0.5, -1.4], "1.5:1": [3.3, 3.7, 0.4, -1.1], "2:1": [0.2, 2.7, 0.6, -1.2],
 "3:1":   [-3.7, 3.2, 1.9, -1.0], "4:1": [-2.6, 2.0, 2.1, -0.6]}

W, H = 660, 414
L, TOP = 78, 70
cw, ch = 138, 56


def blend(a, b, t):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def color(v):
    t = min(1.0, abs(v) / 130.0)
    al = 0.12 + 0.58 * t
    base = (27, 175, 122) if v >= 0 else (227, 73, 72)
    r, g, b = blend((255, 255, 255), base, al)
    return f"rgb({r},{g},{b})"


s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" '
     f'aria-label="Heatmap of total return by candle timeframe and target-to-stop ratio. '
     f'Profitable band runs from 60-min tight ratios through 30-min all ratios to 15-min wide ratios; 5-min loses everywhere.">']
s.append('<title>timeframe x target:stop sweep — total return%</title>')
# column headers
for c, tf in enumerate(TFS):
    s.append(f'<text x="{L+c*cw+cw/2:.0f}" y="{TOP-16}" text-anchor="middle" font-size="13" '
             f'fill="var(--text-secondary)">{tf}</text>')
s.append(f'<text x="{L-10}" y="{TOP-16}" text-anchor="end" font-size="12" fill="var(--text-muted)">tgt:stop</text>')
# rows
for r, ratio in enumerate(ROWS):
    y = TOP + r * ch
    s.append(f'<text x="{L-12}" y="{y+ch/2+5:.0f}" text-anchor="end" font-size="13" '
             f'fill="var(--text-primary)">{ratio}</text>')
    for c in range(4):
        v = TOTAL[ratio][c]
        bps = BPS[ratio][c]
        x = L + c * cw
        s.append(f'<rect x="{x+2}" y="{y+2}" width="{cw-4}" height="{ch-4}" rx="5" fill="{color(v)}"/>')
        s.append(f'<text x="{x+cw/2:.0f}" y="{y+ch/2-1:.0f}" text-anchor="middle" font-size="16" '
                 f'fill="#14110e" font-weight="500">{v:+d}%</text>')
        s.append(f'<text x="{x+cw/2:.0f}" y="{y+ch/2+15:.0f}" text-anchor="middle" font-size="11" '
                 f'fill="#3a3632">{bps:+.1f} bps/trade</text>')
# both VALIDATED cells (passed the fresh-ticker holdout): 30-min/1.5:1 and 15-min/4:1
for rr, cc, lab in [(1, 1, "v3"), (4, 2, "v4")]:
    hx, hy = L + cc * cw, TOP + rr * ch
    s.append(f'<rect x="{hx+1}" y="{hy+1}" width="{cw-2}" height="{ch-2}" rx="6" fill="none" stroke="#0f6e56" stroke-width="3"/>')
    s.append(f'<text x="{hx+cw-6:.0f}" y="{hy+16:.0f}" text-anchor="end" font-size="11" fill="#0f6e56" font-weight="500">{lab}</text>')
# legend + note
ly = TOP + 5 * ch + 16
s.append(f'<rect x="{L}" y="{ly-11}" width="14" height="14" rx="3" fill="none" stroke="#0f6e56" stroke-width="3"/>')
s.append(f'<text x="{L+20}" y="{ly}" font-size="12" fill="var(--text-secondary)">validated — passed the fresh-ticker holdout (the two worth trading)</text>')
s.append(f'<text x="{L}" y="{ly+22}" font-size="11" fill="var(--text-muted)">Tried to beat these and failed: two-sided (overfit, failed holdout), dynamic trailing exit (worse), options (lose to the tax).</text>')
s.append('</svg>')
Path("runs").mkdir(exist_ok=True)
Path("runs/sweep_heatmap.svg").write_text("\n".join(s), encoding="utf-8")
print("wrote runs/sweep_heatmap.svg")

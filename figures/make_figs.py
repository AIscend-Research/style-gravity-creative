"""Generate standalone paper figures from a completed style-gravity run.

Emits one HTML file per figure (inline SVG, no external assets) into --out.
A separate step screenshots them with headless Chrome to produce PNGs.

Palette: dataviz reference instance, light mode, white paper surface.
  series 1 blue #2a78d6, series 2 orange #eb6834  (validated: CVD dE 24.7)
  diverging blue <-> red #e34948, neutral gray midpoint
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# ---- palette (light, surface #ffffff) ---------------------------------- #
BLUE = "#2a78d6"
ORANGE = "#eb6834"
RED = "#e34948"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"

FONT = 'system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def page(title: str, w: int, h: int, body: str) -> str:
    return f"""<!doctype html><meta charset="utf-8"><title>{esc(title)}</title>
<style>
  html,body{{margin:0;padding:0;background:{SURFACE};}}
  .fig{{width:{w}px;height:{h}px;background:{SURFACE};font-family:{FONT};
        -webkit-font-smoothing:antialiased;}}
  text{{font-family:{FONT};}}
</style>
<div class="fig">{body}</div>
"""


def txt(x, y, s, size=13, fill=INK2, anchor="start", weight=400, tabular=False):
    tn = ' font-variant-numeric="tabular-nums"' if tabular else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}"{tn}>{esc(s)}</text>')


# ======================================================================== #
# Figure 1 -- skaldic on opus-4-5, prefill vs prefill+sustain
# ======================================================================== #
def fig1(drift: dict, out: Path) -> Path:
    want = {"prefill", "prefill+sustain"}
    series = {}
    for c in drift["curves"]:
        if c["seed"] == "skaldic" and c["model"] == "claude-opus-4-5" and c["mode"] in want:
            series[c["mode"]] = c

    W, H = 900, 460
    L, R, T, B = 74, 210, 96, 74           # generous right margin for end labels
    pw, ph = W - L - R, H - T - B
    n = len(series["prefill"]["gravity"])
    ylo, yhi = -0.45, 0.45

    def px(i):  # i is 0-based line index
        return L + pw * i / (n - 1)

    def py(g):
        return T + ph * (yhi - g) / (yhi - ylo)

    p = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
         f'aria-label="Skaldic seed on Opus 4.5: prefill reclaims at line 5, '
         f'prefill plus sustain never reclaims.">']
    p.append(f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>')

    # ---- title block
    p.append(txt(L, 32, "Asking is what holds the poem", 19, INK, weight=600))
    p.append(txt(L, 55, "Skaldic-kenning seed, Claude Opus 4.5, same prefill channel. "
                        "n=15 generations per curve.", 13, INK2))

    # ---- gridlines + y ticks
    for g in (-0.4, -0.2, 0.0, 0.2, 0.4):
        y = py(g)
        is_zero = abs(g) < 1e-9
        p.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L+pw}" y2="{y:.1f}" '
                 f'stroke="{AXIS if is_zero else GRID}" stroke-width="1"/>')
        p.append(txt(L - 12, y + 4, f"{g:+.1f}".replace("+0.0", " 0.0"),
                     12, MUTED, anchor="end", tabular=True))

    # pole annotations, tied to the zero line
    p.append(txt(L + pw, py(0.44), "the model's house style", 12, MUTED, anchor="end"))
    p.append(txt(L + pw, py(-0.41), "the seeded style", 12, MUTED, anchor="end"))
    # sits inside the plot at the left, where both curves are well below zero
    p.append(txt(L + 8, py(0.0) - 9, "equidistant", 11, MUTED))

    # ---- x ticks
    for i in range(n):
        if (i + 1) % 5 == 0 or i == 0:
            x = px(i)
            p.append(f'<line x1="{x:.1f}" y1="{T+ph}" x2="{x:.1f}" y2="{T+ph+5}" '
                     f'stroke="{AXIS}" stroke-width="1"/>')
            p.append(txt(x, T + ph + 22, str(i + 1), 12, MUTED, anchor="middle", tabular=True))
    p.append(txt(L + pw / 2, T + ph + 50, "continuation line", 13, INK2, anchor="middle"))
    p.append(f'<line x1="{L}" y1="{T+ph}" x2="{L+pw}" y2="{T+ph}" '
             f'stroke="{AXIS}" stroke-width="1"/>')

    # y axis title
    p.append(f'<text transform="translate(22,{T+ph/2}) rotate(-90)" font-size="13" '
             f'fill="{INK2}" text-anchor="middle">gravity g(i)</text>')

    order = [("prefill", ORANGE, "prefill"),
             ("prefill+sustain", BLUE, "prefill + sustain")]

    # ---- reclamation marker for the plain arm (line 5)
    recl = series["prefill"]["reclamation_line"]
    if recl:
        xr = px(recl - 1)
        p.append(f'<line x1="{xr:.1f}" y1="{T+6}" x2="{xr:.1f}" y2="{T+ph}" '
                 f'stroke="{ORANGE}" stroke-width="1" stroke-dasharray="3 3" opacity="0.55"/>')
        p.append(txt(xr + 7, T + 20, f"reclaimed at line {recl}", 12, INK2))

    # ---- lines
    for mode, color, _ in order:
        pts = " ".join(f"{px(i):.1f},{py(g):.1f}"
                       for i, g in enumerate(series[mode]["gravity"]) if g is not None)
        p.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                 f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')

    # ---- end dots with 2px surface ring, then direct end labels
    for mode, color, label in order:
        g = series[mode]["gravity"]
        x, y = px(n - 1), py(g[-1])
        p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}" '
                 f'stroke="{SURFACE}" stroke-width="2"/>')
        asym = series[mode]["asymptote"]
        p.append(txt(x + 12, y - 2, label, 13, INK, weight=600))
        note = ("never reclaimed in 20 lines" if series[mode]["reclamation_line"] is None
                else f"reclaimed at line {series[mode]['reclamation_line']}")
        p.append(txt(x + 12, y + 15, f"settles {asym:+.3f}", 12, INK2, tabular=True))
        p.append(txt(x + 12, y + 31, note, 12, MUTED))

    # ---- legend (always present for 2+ series)
    lx = L
    for mode, color, label in order:
        p.append(f'<line x1="{lx}" y1="{T-22}" x2="{lx+18}" y2="{T-22}" '
                 f'stroke="{color}" stroke-width="2" stroke-linecap="round"/>')
        p.append(txt(lx + 25, T - 18, label, 12, INK2))
        lx += 28 + len(label) * 7.0

    p.append("</svg>")
    f = out / "fig1_skaldic_opus45_prefill_vs_sustain.html"
    f.write_text(page("Figure 1", W, H, "".join(p)))
    return f


# ======================================================================== #
# Figure 2 -- two panels: curves holding, and mean asymptote
# ======================================================================== #
def fig2(rows: list[dict], out: Path) -> Path:
    W, H = 900, 430
    p = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
         f'aria-label="Reclamation and settled position by condition.">']
    p.append(f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>')
    p.append(txt(40, 32, "The sustain instruction decides the outcome", 19, INK, weight=600))
    p.append(txt(40, 55, "Depth-tier seeds (monosyllabic, skaldic) across five models, 20-line poems.",
                 13, INK2))

    labw = 168
    # ---------------- panel A: share of curves holding all 20 lines
    ax, ay, aw, ah = 40 + labw, 106, 250, 250
    p.append(txt(40, 92, "Curves holding the seed for all 20 lines", 13, INK, weight=600))
    band = ah / len(rows)
    for k, r in enumerate(rows):
        cy = ay + band * k + band / 2
        frac = r["never"] / r["total"]
        bh = 22
        p.append(txt(ax - 12, cy + 4, r["label"], 12.5, INK2, anchor="end"))
        # track
        p.append(f'<rect x="{ax}" y="{cy-bh/2:.1f}" width="{aw}" height="{bh}" '
                 f'fill="{GRID}" opacity="0.55" rx="3"/>')
        bw = aw * frac
        if bw > 0:
            p.append(f'<path d="M{ax},{cy-bh/2:.1f} h{max(bw-4,0):.1f} '
                     f'a4,4 0 0 1 4,4 v{bh-8} a4,4 0 0 1 -4,4 h-{max(bw-4,0):.1f} z" '
                     f'fill="{BLUE}"/>')
        vx = ax + bw + 10 if frac < 0.72 else ax + bw - 10
        anc = "start" if frac < 0.72 else "end"
        fill = INK if frac < 0.72 else "#ffffff"
        p.append(txt(vx, cy + 4, f"{r['never']} / {r['total']}", 12.5, fill,
                     anchor=anc, weight=600, tabular=True))
    p.append(f'<line x1="{ax}" y1="{ay}" x2="{ax}" y2="{ay+ah}" stroke="{AXIS}" stroke-width="1"/>')

    # ---------------- panel B: mean asymptote, diverging around zero
    bx, by, bw_, bh_ = 40 + labw + 250 + 118, 106, 250, 250
    p.append(txt(bx - 42, 92, "Mean settled position", 13, INK, weight=600))
    lo, hi = -0.30, 0.14
    def sx(v):
        return bx + bw_ * (v - lo) / (hi - lo)
    zx = sx(0.0)
    # ticks
    for t in (-0.3, -0.2, -0.1, 0.0, 0.1):
        x = sx(t)
        p.append(f'<line x1="{x:.1f}" y1="{by}" x2="{x:.1f}" y2="{by+bh_}" '
                 f'stroke="{AXIS if abs(t)<1e-9 else GRID}" stroke-width="1"/>')
        p.append(txt(x, by + bh_ + 20, f"{t:+.1f}".replace("+0.0", "0.0"), 11.5, MUTED,
                     anchor="middle", tabular=True))
    for k, r in enumerate(rows):
        cy = by + band * k + band / 2
        v = r["asym"]
        x0, x1 = (zx, sx(v)) if v >= 0 else (sx(v), zx)
        col = RED if v >= 0 else BLUE
        bht = 22
        w = abs(x1 - x0)
        if v >= 0:
            p.append(f'<path d="M{x0},{cy-bht/2:.1f} h{max(w-4,0):.1f} a4,4 0 0 1 4,4 '
                     f'v{bht-8} a4,4 0 0 1 -4,4 h-{max(w-4,0):.1f} z" fill="{col}"/>')
            p.append(txt(x1 + 10, cy + 4, f"{v:+.3f}", 12.5, INK, weight=600, tabular=True))
        else:
            p.append(f'<path d="M{x1},{cy-bht/2:.1f} h-{max(w-4,0):.1f} a4,4 0 0 0 -4,4 '
                     f'v{bht-8} a4,4 0 0 0 4,4 h{max(w-4,0):.1f} z" fill="{col}"/>')
            p.append(txt(x0 - 10, cy + 4, f"{v:+.3f}", 12.5, INK, weight=600,
                         anchor="end", tabular=True))
    # one centred note; the panel is too narrow to carry a label at each pole
    p.append(txt(bx + bw_ / 2, by + bh_ + 42, "negative sits nearer the seeded style",
                 11.5, MUTED, anchor="middle"))

    p.append(txt(40, H - 18,
                 "Left: the share of curves whose style never reverted within the poem. "
                 "Right: blue settles toward the seed, red toward the model.",
                 11.5, MUTED))
    p.append("</svg>")
    f = out / "fig2_reclamation_by_condition.html"
    f.write_text(page("Figure 2", W, H, "".join(p)))
    return f


# ======================================================================== #
# Figure 3 -- forest plot of the monosyllabic sustain contrast
# ======================================================================== #
def fig3(items: list[dict], out: Path) -> Path:
    W, H = 900, 450
    L, T = 250, 118
    pw, ph = 500, 250
    lo, hi = -0.28, 0.06

    def sx(v):
        return L + pw * (v - lo) / (hi - lo)

    p = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
         f'aria-label="Monosyllabic sustain contrast, eight cells, all intervals below zero.">']
    p.append(f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>')
    p.append(txt(40, 34, "Every cell moves the same way", 19, INK, weight=600))
    p.append(txt(40, 57, "Monosyllabic seed. Effect of adding the sustain instruction, "
                         "with 95% bootstrap intervals", 13, INK2))
    p.append(txt(40, 76, "resampled over whole generations. n=15 per arm.", 13, INK2))

    band = ph / len(items)
    for t in (-0.25, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05):
        x = sx(t)
        z = abs(t) < 1e-9
        p.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{T+ph}" '
                 f'stroke="{AXIS if z else GRID}" stroke-width="1"/>')
        p.append(txt(x, T + ph + 20, f"{t:+.2f}".replace("+0.00", "0.00"), 11.5, MUTED,
                     anchor="middle", tabular=True))
    p.append(txt(sx(0.0), T - 12, "no effect", 11.5, MUTED, anchor="middle"))

    for k, it in enumerate(items):
        cy = T + band * k + band / 2
        p.append(txt(L - 150, cy + 4, it["model"], 12.5, INK, anchor="start"))
        p.append(txt(L - 14, cy + 4, it["channel"], 12.5, MUTED, anchor="end"))
        a, b = sx(it["ci"][0]), sx(it["ci"][1])
        p.append(f'<line x1="{a:.1f}" y1="{cy:.1f}" x2="{b:.1f}" y2="{cy:.1f}" '
                 f'stroke="{BLUE}" stroke-width="2" stroke-linecap="round" opacity="0.45"/>')
        for xx in (a, b):
            p.append(f'<line x1="{xx:.1f}" y1="{cy-5:.1f}" x2="{xx:.1f}" y2="{cy+5:.1f}" '
                     f'stroke="{BLUE}" stroke-width="2" stroke-linecap="round" opacity="0.45"/>')
        cx = sx(it["d"])
        p.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="{BLUE}" '
                 f'stroke="{SURFACE}" stroke-width="2"/>')
        p.append(txt(sx(hi) + 14, cy + 4, f"{it['d']:+.3f}", 12.5, INK2, tabular=True))

    p.append(txt(L, T + ph + 44, "held nearer the seed", 11.5, MUTED))
    p.append(txt(sx(hi), T + ph + 44, "held nearer the house style", 11.5, MUTED, anchor="end"))
    p.append(txt(40, H - 20,
                 "All eight intervals fall entirely below zero. The same contrast on the "
                 "skaldic seed crosses zero in three of eight cells.",
                 11.5, MUTED))
    p.append("</svg>")
    f = out / "fig3_monosyllabic_contrast.html"
    f.write_text(page("Figure 3", W, H, "".join(p)))
    return f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    drift = json.loads((a.run_dir / "drift.json").read_text())

    depth = {"monosyllabic", "skaldic"}
    cur = [c for c in drift["curves"] if c["seed"] in depth]
    rows = []
    for mode, label in [("prefill", "prefill"),
                        ("prefill+sustain", "prefill + sustain"),
                        ("instructed", "instructed"),
                        ("instructed+sustain", "instructed + sustain")]:
        sel = [c for c in cur if c["mode"] == mode]
        never = sum(1 for c in sel
                    if any("never reclaimed" in w for w in c["warnings"]))
        rows.append({"label": label, "never": never, "total": len(sel),
                     "asym": sum(c["asymptote"] for c in sel) / len(sel)})

    short = {"claude-haiku-4-5": "Haiku 4.5", "claude-opus-4-5": "Opus 4.5",
             "claude-opus-5": "Opus 5", "claude-sonnet-4-5": "Sonnet 4.5",
             "claude-sonnet-5": "Sonnet 5"}
    items = [{"model": short[k["model"]], "channel": k["base_mode"],
              "d": k["d_gravity"], "ci": k["d_gravity_ci"]}
             for k in drift["contrasts"] if k["seed"] == "monosyllabic"]
    items.sort(key=lambda x: x["d"])

    for f in (fig1(drift, a.out), fig2(rows, a.out), fig3(items, a.out)):
        print("wrote", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

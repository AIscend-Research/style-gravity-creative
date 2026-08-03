"""Self-contained HTML report: one small-multiple per seed, models as series.

Charts are hand-built SVG — no chart library, no CDN, no build step — so the
report is a single file that opens anywhere and survives being emailed around.

Colour: three categorical slots (blue / orange / aqua), validated all-pairs in
both light and dark mode. Aqua sits under 3:1 on the light surface, so the
relief rule applies and every series carries a direct label at the end of its
line plus a full table view below the charts — identity is never colour-alone.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from .drift import DriftCurve
from .models import resolve as resolve_model
from .seeds import resolve as resolve_seed

#  Two encodings, because a cell is identified by two things. Colour carries the
#  model (categorical slots, validated all-pairs in both modes); dash pattern
#  carries the channel/condition. Folding both into colour would need 20 hues,
#  far past the point where any palette stays colourblind-safe.
DASH = {
    "prefill": "",
    "prefill+sustain": ' stroke-dasharray="7 3"',
    "instructed": ' stroke-dasharray="2 3"',
    "instructed+sustain": ' stroke-dasharray="7 3 2 3"',
}


def series_label(curve) -> str:
    return f"{resolve_model(curve.model).label} · {curve.mode}"


def _slot(curves, curve) -> int:
    """Colour slot by model, so the same model keeps its colour across every
    condition in the facet."""
    models = []
    for c in curves:
        if c.model not in models:
            models.append(c.model)
    return models.index(curve.model) % 6

CSS = """
:root { color-scheme: light dark; }
.viz-root {
  --surface-1:#fcfcfb; --surface-2:#f4f3f0; --rule:#dcdbd6;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#78766f;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s5:#e87ba4; --s6:#008300;
  --seed-pole:#2a78d6; --house-pole:#e34948; --neutral:#f0efec;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    --surface-1:#1a1a19; --surface-2:#232322; --rule:#3a3a37;
    --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#94938a;
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181; --s6:#008300;
    --seed-pole:#3987e5; --house-pole:#e66767; --neutral:#383835;
  }
}
:root[data-theme="dark"] .viz-root {
  --surface-1:#1a1a19; --surface-2:#232322; --rule:#3a3a37;
  --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#94938a;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181; --s6:#008300;
  --seed-pole:#3987e5; --house-pole:#e66767; --neutral:#383835;
}
.viz-root {
  background:var(--surface-1); color:var(--text-primary);
  font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;
  max-width:1000px; margin:0 auto; padding:2.5rem 1.25rem 4rem;
}
h1 { font-size:1.7rem; margin:0 0 .35rem; letter-spacing:-.01em; }
h2 { font-size:1.15rem; margin:2.5rem 0 .25rem; }
h3 { font-size:1rem; margin:1.75rem 0 .35rem; }
p.sub { color:var(--text-secondary); margin:.15rem 0 1.4rem; }
.meta { color:var(--text-muted); font-size:.85rem; }
figure { margin:1.25rem 0 2rem; }
figcaption { color:var(--text-secondary); font-size:.87rem; margin-top:.4rem; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:.87rem; min-width:640px; }
th,td { text-align:left; padding:.42rem .6rem; border-bottom:1px solid var(--rule); }
th { color:var(--text-secondary); font-weight:600; }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
.swatch { display:inline-block; width:.62rem; height:.62rem; border-radius:2px;
          margin-right:.4rem; vertical-align:baseline; }
.legend { display:flex; flex-wrap:wrap; gap:.9rem; font-size:.85rem;
          color:var(--text-secondary); margin:.2rem 0 .5rem; }
.warn { background:var(--surface-2); border-left:3px solid var(--s2);
        padding:.7rem .9rem; margin:.7rem 0; font-size:.88rem; border-radius:0 4px 4px 0; }
pre.seed { background:var(--surface-2); padding:.8rem 1rem; border-radius:5px;
           font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
           white-space:pre-wrap; overflow-x:auto; }
"""


# --------------------------------------------------------------------------- #
# svg
# --------------------------------------------------------------------------- #

def _chart(curves: list[DriftCurve], *, width: int = 880, height: int = 300) -> str:
    """Multi-series line chart of gravity vs line index.

    y is fixed to the metric's full [-1, +1] range in every chart. Auto-scaling
    y per facet would make an 0.02 wobble look like a collapse and destroy
    comparability between seeds — the whole point is reading the facets against
    each other.
    """
    pad_l, pad_r, pad_t, pad_b = 46, 118, 18, 34
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    max_x = max((len(c.gravity) for c in curves), default=1) or 1

    def px(i: int) -> float:
        return pad_l + (plot_w * (i / max(max_x - 1, 1)))

    def py(g: float) -> float:
        return pad_t + plot_h * ((1 - g) / 2)

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="Style drift by line index" style="overflow:visible">'
    ]
    # polarity bands — the chart's two poles, not decoration
    parts.append(
        f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h / 2:.1f}" '
        f'fill="var(--house-pole)" opacity="0.055"/>'
        f'<rect x="{pad_l}" y="{pad_t + plot_h / 2:.1f}" width="{plot_w}" '
        f'height="{plot_h / 2:.1f}" fill="var(--seed-pole)" opacity="0.055"/>'
    )
    for g, label in ((1.0, "+1 house style"), (0.0, "0 tipping point"), (-1.0, "−1 seeded style")):
        y = py(g)
        dash = "" if g == 0.0 else ' stroke-dasharray="3 4"'
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" '
            f'stroke="var(--rule)" stroke-width="1"{dash}/>'
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="10.5" '
            f'fill="var(--text-muted)">{html.escape(label.split()[0])}</text>'
        )
    # x ticks
    step = max(1, max_x // 8)
    for i in range(0, max_x, step):
        parts.append(
            f'<text x="{px(i):.1f}" y="{height - pad_b + 17}" text-anchor="middle" '
            f'font-size="10.5" fill="var(--text-muted)">{i + 1}</text>'
        )
    parts.append(
        f'<text x="{pad_l + plot_w / 2:.1f}" y="{height - 2}" text-anchor="middle" '
        f'font-size="11" fill="var(--text-secondary)">line of continuation</text>'
    )

    for idx, curve in enumerate(curves):
        color = f"var(--s{_slot(curves, curve) + 1})"
        g = curve.gravity
        if not g:
            continue
        pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(g))
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-linejoin="round" stroke-linecap="round"{DASH.get(curve.mode, "")}/>'
        )
        # reclamation marker: 2px surface ring so overlapping markers stay legible
        if curve.reclamation_line:
            cx, cy = px(curve.reclamation_line - 1), py(g[curve.reclamation_line - 1])
            parts.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="{color}" '
                f'stroke="var(--surface-1)" stroke-width="2"/>'
            )
        # Direct label at the end of each line: the relief rule for the light-mode
        # aqua slot, and it keeps identity off colour alone.
        label = series_label(curve)
        parts.append(
            f'<text x="{px(len(g) - 1) + 9:.1f}" y="{py(g[-1]) + 4:.1f}" font-size="10.5" '
            f'fill="var(--text-primary)">'
            f'<tspan fill="{color}">\u25a0 </tspan>{html.escape(label)}</text>'
        )
        parts.append(f"<title>{html.escape(label)}</title>")

    parts.append("</svg>")
    return "".join(parts)


def _legend(curves: list[DriftCurve]) -> str:
    models, modes = [], []
    for c in curves:
        if c.model not in models:
            models.append(c.model)
        if c.mode not in modes:
            modes.append(c.mode)
    items = [
        f'<span><span class="swatch" style="background:var(--s{i + 1})"></span>'
        f"{html.escape(resolve_model(m).label)}</span>"
        for i, m in enumerate(models)
    ]
    if len(modes) > 1:
        for mode in modes:
            dash = DASH.get(mode, "")
            style = "3 2" if "dasharray" in dash else "none"
            items.append(
                f'<span><svg width="22" height="8" style="vertical-align:middle">'
                f'<line x1="0" y1="4" x2="22" y2="4" stroke="var(--text-secondary)" '
                f'stroke-width="2"{dash}/></svg> {html.escape(mode)}</span>'
            )
    return f'<div class="legend">{"".join(items)}</div>'


def _fmt(v, digits: int = 2, dash: str = "—") -> str:
    if v is None:
        return dash
    if isinstance(v, float):
        return f"{v:+.{digits}f}" if abs(v) < 10 else f"{v:.{digits}f}"
    return str(v)


def _table(curves: list[DriftCurve]) -> str:
    rows = []
    for c in curves:
        color = f"var(--s{_slot(curves, c) + 1})"
        ci = c.reclamation_ci or (None, None)
        ci_txt = f"{ci[0]}–{ci[1]}" if ci[0] is not None else "—"
        recl = str(c.reclamation_line) if c.reclamation_line else f">{len(c.gravity)}"
        rows.append(
            "<tr>"
            f'<td><span class="swatch" style="background:{color}"></span>'
            f"{html.escape(resolve_model(c.model).label)}</td>"
            f"<td>{html.escape(c.mode)}</td>"
            f'<td class="num">{recl}</td>'
            f'<td class="num">{ci_txt}</td>'
            f'<td class="num">{_fmt(c.half_reclamation_line, 0)}</td>'
            f'<td class="num">{_fmt(c.opening_gravity)}</td>'
            f'<td class="num">{_fmt(c.asymptote)}</td>'
            f'<td class="num">{_fmt(c.separation)}</td>'
            f'<td class="num">{len(c.per_sample)}</td>'
            "</tr>"
        )
    return (
        '<div class="scroll"><table><thead><tr>'
        "<th>model</th><th>mode</th><th>reclaimed at</th><th>95% CI</th>"
        "<th>half-life</th><th>line 1</th><th>settles at</th>"
        "<th>seed↔house sep.</th><th>n</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def render(
    curves: list[DriftCurve],
    *,
    n_generations: int,
    total_cost_usd: float,
    notes: list[str],
) -> str:
    by_seed: dict[str, list[DriftCurve]] = {}
    for c in curves:
        by_seed.setdefault(c.seed, []).append(c)

    body: list[str] = [
        "<h1>Style Gravity: text</h1>",
        '<p class="sub">How many lines a human-authored opening survives before the '
        "model's house style reclaims the poem.</p>",
        f'<p class="meta">{n_generations} generations · '
        f"${total_cost_usd:.2f} spent · "
        f'{len({c.model for c in curves})} models × {len(by_seed)} seeds</p>',
    ]

    for note in notes:
        body.append(f'<div class="warn">{html.escape(note)}</div>')

    # control first — it is the run's validity check
    order = sorted(by_seed, key=lambda s: (s != "control", s))
    for seed_id in order:
        group = sorted(by_seed[seed_id], key=lambda c: (c.mode, c.model))
        seed = resolve_seed(seed_id)
        body.append(f"<h2>{html.escape(seed.name)}</h2>")
        body.append(
            f'<p class="sub">{html.escape(seed.note)}</p>'
            f"<pre class=\"seed\">{html.escape(seed.text)}</pre>"
        )
        body.append(_legend(group))
        body.append(f"<figure>{_chart(group)}")
        body.append(
            "<figcaption>Drift of each continuation line, scored between the seed's "
            "own style (−1) and the model's unprompted baseline (+1). The marked point "
            "is the first line from which the curve stays on the house-style side.</figcaption>"
            "</figure>"
        )
        body.append(_table(group))
        for c in group:
            for w in c.warnings:
                body.append(
                    f'<div class="warn"><strong>{html.escape(resolve_model(c.model).label)}</strong> — '
                    f"{html.escape(w)}</div>"
                )

    body.append("<h2>Reading the numbers</h2>")
    body.append(
        "<p><strong>reclaimed at</strong> is the headline: the first continuation line "
        "from which the poem sits closer to the model's own unprompted style than to the "
        "opening it was given, and stays there. A <code>&gt;n</code> entry means the seed "
        "was still holding when generation stopped — a lower bound, not a measurement.</p>"
        "<p><strong>seed↔house sep.</strong> is the distance between the two poles. If it "
        "is small, the seed was not actually off-distribution for that model and its curve "
        "means nothing; the control seed exists to make that failure visible.</p>"
    )
    return (
        "<style>" + CSS + "</style>"
        '<main class="viz-root">' + "".join(body) + "</main>"
    )


def write_report(path: Path, **kwargs) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render(**kwargs), encoding="utf-8")
    return p

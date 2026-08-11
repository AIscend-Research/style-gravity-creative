# Figures

Generated from `runs/full/drift.json` by `make_figs.py`. No API calls, no external
assets. Each figure ships as a PNG at 1800px wide (2x device scale, ~6in at 300dpi)
and as a self-contained HTML file holding the source SVG, so it can be re-exported
to vector for camera-ready.

Palette is the light-mode categorical default, validated for colour-vision
deficiency: worst adjacent pair blue/orange sits at CVD dE 24.7 and normal-vision
dE 33.6, well clear of the 8 and 15 floors. Colour never carries meaning alone;
every series is also direct-labelled.

---

## Figure 1
**File:** `fig1_skaldic_opus45_prefill_vs_sustain.png` (1800x920)

> **Asking is what holds the poem.** Line-by-line gravity for a skaldic-kenning
> seed continued by Claude Opus 4.5 over the same prefill channel, with and without
> a system-prompt instruction to sustain the opening's style. Gravity of -1 sits on
> the seeded style and +1 on the model's own unprompted style, each pole estimated
> inside the same run. Without the instruction the poem crosses to the model's side
> at line 5 and settles at +0.213. With it, the poem never crosses in 20 lines and
> settles at -0.100. n=15 generations per curve.

Use as the lead figure. It carries the whole argument in two lines.

## Figure 2
**File:** `fig2_reclamation_by_condition.png` (1800x860)

> **The sustain instruction decides the outcome.** All depth-tier curves
> (`monosyllabic` and `skaldic`, five models, 20-line poems) grouped by condition.
> Left: the share of curves whose style never reverted to the model's own within the
> poem. Right: the mean settled position of each condition's curves. Every
> `instructed+sustain` curve holds for the full 20 lines; no plain `prefill` curve
> does, and plain `prefill` is the only condition whose mean lands on the model's
> side of zero.

The four-row comparison. Pairs with the reclamation counts in the text.

## Figure 3
**File:** `fig3_monosyllabic_contrast.png` (1800x900)

> **Every cell moves the same way.** Effect of adding the sustain instruction on the
> `monosyllabic` seed, across five models and both channels, with 95% bootstrap
> intervals resampled over whole generations rather than lines. Negative means the
> instruction held the poem nearer the seeded style. All eight intervals fall
> entirely below zero. The same contrast on the `skaldic` seed crosses zero in three
> of eight cells, which is the limit of the result.

Report alongside the skaldic table so the negative result is visible.

---

## Regenerating

```bash
python3 make_figs.py /path/to/runs/full --out ./figs
# then, per figure:
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless --disable-gpu --hide-scrollbars --force-device-scale-factor=2 \
  --window-size=900,460 --screenshot=fig1.png file://$PWD/figs/fig1_*.html
```

Window size must match the `.fig` div in each HTML file (900x460, 900x430, 900x450).
Raise `--force-device-scale-factor` for a larger raster.

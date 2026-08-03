# Style Gravity: text

*text domain attractor decay*

**How many lines does a human-authored opening survive before the model's house style reclaims the poem?**

Anthropic's API lets you force the opening of a response by supplying it as a trailing
assistant turn — the model experiences your text as its own in-progress output and simply
keeps writing. That makes a clean experiment possible: seed a poem with a deliberately
off-distribution opening (skaldic kennings, Middle-English inflection, statutory boilerplate,
a lipogram, a typographic scatter), let the model continue, and watch line by line as the
seeded style decays back toward whatever the model writes unprompted.

The decay curve is the measurement. It is Style Gravity's text sibling, built entirely from a
feature Anthropic natively supports, and it costs dollars to run.

```
line:   1    2    3    4    5    6    7    8    9   10
  +1 ───┼────┼────┼────┼────┼────┼────╭────●────┼────┼──  the model's house style
        │    │    │    │    │    │  ╭─╯    ↑
   0 ───┼────┼────┼────┼────┼──╭─╯    reclaimed at 8
        │    │    │    ╭────╯  │
  -1 ───●────●────●────╯───────┴──────┴────┴────┴────┴──  the seeded style
```

---

## ⚠️ Read this first: prefill is gone on the newest models

The whole method rests on assistant prefill, and **prefill was removed on the Claude 4.6+
line**. A request whose final message has `role: "assistant"` returns a 400 on Opus 4.6 / 4.7 /
4.8, Opus 5, Sonnet 4.6, Sonnet 5, and Fable 5.

So the experiment as literally specified can only be run on prefill-capable models:

| runnable in `prefill` mode | prefill removed — `instructed` mode only |
|---|---|
| Opus 4.5, Opus 4.1, Sonnet 4.5, Sonnet 4, Haiku 4.5 | Opus 5, Opus 4.8, Opus 4.7, Opus 4.6, Sonnet 5, Sonnet 4.6 |

For the second column the tool falls back to **`instructed` mode**: the seed goes in the *user*
turn with a "continue this" instruction. This is measurable and interesting, but it is **not the
same intervention** — the model sees the seed as someone else's text to be continued rather than
as its own output to be extended. Results from the two modes are labelled throughout and are
never pooled. Treat cross-mode comparisons as suggestive at best.

If you want the headline number for a current-generation model, there is no way to get it via
prefill, and this repo will not pretend otherwise.

`python -m stylegravity models` prints the table above at any time.

---

## Method

**Two poles, not one.** "How far has line 7 drifted from the seed?" has no answer in absolute
units — every continuation moves. The answerable question is *which pole line 7 is closer to*.
So each line is scored against both the seed's style and that model's own unprompted style:

```
g(i) = ( d_prefix(i) − d_baseline(i) ) / ( d_prefix(i) + d_baseline(i) )

  −1  sits exactly on the seeded style
   0  equidistant — the tipping point
  +1  sits exactly on the model's house style
```

Both poles are estimated inside the same run, in the same feature space. A model with an
idiosyncratic baseline is measured against *its own* baseline, which is what makes the curve a
per-model steering measurement rather than a per-model style description.

**Baselines are matched.** The house-style pole comes from the identical poem request with no
seed at all (`Write a poem, at least N lines long.`), so prompt wording is held constant between
the seeded and unseeded conditions.

**Features are surface stylometry, not embeddings** (`features.py`). 24 per line: syllable and
word length, monosyllable rate, type–token ratio, function-word rate, archaism rate,
Latinate-suffix rate, hyphenated-compound rate, punctuation and dash density, terminal-stop vs
enjambment, capitalisation, first-person rate, `And`-initial rate, bracket and digit rate,
internal space runs, letter-*e* rate, in-line repetition. Embeddings were rejected on purpose:
an embedding model's own priors would sit between the measurement and the thing measured, and
embedding distance is dominated by topic rather than style. Each feature is z-scored across the
whole run so a unit of distance means the same thing in every comparison.

**Two headline numbers.** The *reclamation line* is the first continuation line from which the
curve stays on the house-style side of zero for `--window` consecutive lines (default 2 — a
single crossing and back is a wobble, not a takeover). The *half-life* is the line at which
drift has covered half the distance from its opening value to its settled value. They answer
different questions: when the model wins, versus how fast it is winning.

### Validity checks, because this metric can lie

- **A control seed.** `control` is written *in* the generic contemporary free-verse register
  models default to. Its curve should sit near zero from line one. If it doesn't, the metric is
  measuring something other than style reclamation and the run should be discarded.
- **Pole separation.** If the seed and the model's baseline occupy the same region of feature
  space, the seed wasn't off-distribution for that model and its curve is meaningless. Cells
  below a separation threshold are flagged invalid rather than reported as a number.
- **Leave-one-out self-scores.** Seed lines are scored against a seed centroid built without
  them, and likewise for baselines. Near −1 and +1 respectively means the poles are genuinely
  separable; anything else is surfaced as a warning in the report.
- **Bootstrapped CI**, resampling whole generations rather than lines — lines within one poem
  are not independent draws, and treating them as such would make the interval far too tight.
- **Seed-echo stripping.** In `instructed` mode models often restate the opening before
  continuing. Those lines are the seed, not a response to it; leaving them in would manufacture
  a stretch of perfect fidelity and inflate exactly the number being measured.
- **Never-reclaimed is reported as `>n`**, a lower bound, not silently as `n`.

### Seeds

Twelve openings across five categories — `borrowed_style`, `register_clash`, `constrained`,
`broken_form`, and the `control`. All are **original text written for this experiment** in the
manner of the named style, never excerpts from published poems. That matters twice: it avoids
reproducing copyrighted work, and it keeps the seeds out of any model's training data, so a low
drift score cannot be explained away as memorised completion.

Two seeds are worth calling out because they give binary, hand-checkable tells that don't depend
on trusting the feature vector at all: `lipogram` (no letter *e* — count them) and `monosyllabic`
(every word one syllable — count them). If the composite metric and the hand-countable tell
disagree, believe the tell and fix the metric.

`python -m stylegravity seeds` lists them all.

---

## The four conditions

The experiment crosses two channels with two levels of explicit instruction. The contrast
between a cell and its `+sustain` twin is the paper's central measurement: **what asking, in
words, can recover of what the prefill affordance used to give you structurally.**

| mode | channel | asked to sustain the style? |
|---|---|---|
| `prefill` | seed as the model's own in-progress output | no |
| `prefill+sustain` | same | yes, via system prompt |
| `instructed` | seed as someone else's text in the user turn | no |
| `instructed+sustain` | same | yes, via system prompt |
| `baseline` | no seed at all — the house-style pole | n/a |

`sustain` never names a style (that would leak the answer, and the twelve seeds span registers
no single description covers). It asks the model to hold the opening's diction, syntax, line
shape, punctuation, formality, and any formal constraint it appears to be observing.

All four conditions run on the **same** prefill-capable models, which is what de-confounds
channel from model generation — without that, "prefill vs instructed" is perfectly correlated
with "2025-era vs current model" and neither can be attributed.

## Usage

```bash
python -m venv .venv && .venv/bin/pip install -e .
export ANTHROPIC_API_KEY=...          # or: ant auth login

python -m stylegravity presets                          # what each preset runs
python -m stylegravity estimate --preset full           # cost + wall clock, no API calls
python -m stylegravity run --preset full --run-dir runs/full
python -m stylegravity analyse --run-dir runs/full      # re-score, no API calls
```

| preset | calls | typical cost | wall clock @ `--concurrency 8` | what it is |
|---|---|---|---|---|
| `pilot` | 26 | $0.11 | <1 min | smoke test, one model |
| `default` | 84 | $1.03 | ~1 min | core sweep, prefill only |
| `full` | 940 | **$12.22** (ceiling $18.81) | **~15 min** | the whole paper |

`full` is a deliberate two-tier design, not a full factorial: all twelve seeds in the headline
condition (breadth), four conditions on the four core seeds (depth), plus the current-generation
models on the instructed channel. Crossing everything would quadruple the bill to answer
questions no one asked.

Cost estimates assume every generation runs to `--max-tokens`; real spend lands near the
"typical" column because poems stop on their own. The time figure is derived from published
throughput and **ignores rate-limit backoff** — on a low API tier, 429s dominate it. Drop
`--concurrency` if you see them.

A run writes three files to `--run-dir`:

| file | what it is |
|---|---|
| `generations.jsonl` | every raw poem, append-only, with usage and cost |
| `drift.json` | per-line curves and all summary statistics |
| `report.html` | self-contained report — no CDN, no build step, opens anywhere |

Generation is the expensive part; analysis is free. Caching the raw poems means the metric can
be re-tuned and re-argued about without spending another cent, and it leaves every number
auditable by anyone who doubts it. `run` resumes from cache — an interrupted run costs nothing
to restart, and re-running after failures retries only the failed cells. Baselines are generated
first, so even a partial run leaves every model with a usable house-style pole.

Useful knobs: `--models`, `--seeds`, `--samples`, `--concurrency`, `--lines`, `--window`,
`--temperature`.

---

## Prior work

The visual-domain precedent is the *Patterns* paper on generative-model style convergence,
cited here for the same reason [Transcription Gap](../transcription-gap) cites it: it
establishes that the pull toward a model's own aesthetic centre is measurable in images, and
this project asks whether the same pull is measurable line by line in text, where the decay has
an ordering that images do not.

> **Citation incomplete — needs your input.** I could not recover the exact reference from this
> machine: the Transcription Gap repo has no citation section or bibliography to copy it from.
> Rather than fabricate authors, a year, or a DOI, the reference is left as a placeholder. Drop
> the full citation in here — and ideally into Transcription Gap too, so the two agree — before
> this goes anywhere public.

---

## Limitations

- **The biggest one is at the top of this file**: current-generation models cannot be measured
  by the intended method at all, only by the `instructed` proxy.
- Surface stylometry captures diction, morphology, register, and form. It does **not** capture
  imagery, argument, metaphor, or voice in the sense a critic means. A model could abandon the
  seed's *sensibility* entirely while holding its surface features, and this metric would score
  that as fidelity.
- The 24 features were chosen to discriminate *these twelve seeds*. A seed varying along an axis
  the vector doesn't encode will show spuriously fast drift. Adding a seed means checking the
  features can see it — the pole-separation check is what catches this.
- The syllable estimator is a vowel-group heuristic: consistent, not correct. Consistency is what
  a distance metric needs, but the absolute `syllables_per_word` numbers are not prosody.
- Line-index alignment assumes the model writes one poetic line per output line. Blank lines are
  dropped, but a wrapped long line still registers as two.
- `temperature=1.0` and n=5 per cell by default. Enough to see a large effect, not enough to
  resolve a small one — read the CI, not the point estimate.

---

## Layout

```
stylegravity/
  models.py     model registry + prefill capability (the constraint that shapes everything)
  seeds.py      twelve original off-distribution openings + the control
  features.py   24 surface-stylometric features per line
  generate.py   prefill / instructed / baseline calls, JSONL cache, cost estimator
  drift.py      z-scaling, two-pole gravity, reclamation point, half-life, bootstrap
  analysis.py   generations → curves
  report.py     self-contained HTML + hand-built SVG
  cli.py        estimate / run / analyse / seeds / models
tests/          37 tests, no API calls
```

Run the tests with `.venv/bin/python -m pytest tests -q`.

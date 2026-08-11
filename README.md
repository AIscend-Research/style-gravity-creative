# Style Gravity: text

*text domain attractor decay*

**When a poem opens in a style the model would never choose, does the model's own style
reclaim it? And what does asking, in words, recover of what the prefill affordance gave
structurally?**

Anthropic's API lets you force the opening of a response by supplying it as a trailing
assistant turn. The model experiences your text as its own in-progress output and keeps
writing. That makes a clean experiment possible: seed a poem with a deliberately
off-distribution opening (skaldic kennings, Middle-English inflection, statutory boilerplate,
a lipogram, a typographic scatter), let the model continue, and watch line by line as the
seeded style moves toward whatever the model writes unprompted.

The decay curve is the measurement.

**The run is complete.** 1596 generations, 5 models, 4 conditions, 14 seeds, $13.12.
Everything below is measured, and the headline result is the opposite of what the project
set out to find.

```
line:   1    2    3    4    5    6    7    8    9   10  ...  20
  +1 ───┼────┼────┼────╭────●────┼────┼────┼────┼────┼───────┼──  the model's house style
        │    │    │  ╭─╯  reclaimed at 5   (prefill, no instruction)
   0 ───┼────┼──╭─╯───┼────┼────┼────┼────┼────┼────┼───────┼──
        │    ╭─╯      │
  -1 ───●────╯────●───┴────●────●────●────●────●────●───────●──  the seeded style
                              (instructed+sustain: never reclaimed, 10/10 curves)
```

---

## Headline results

**1. The sustain instruction is the dominant factor, and it is stronger than the channel.**

Across the two seeds that survive the validity checks (`monosyllabic`, `skaldic`), reclamation
within 20 lines depends almost entirely on whether the model was asked to hold the style:

| condition | curves that never reclaim within 20 lines | mean asymptote |
|---|---|---|
| `prefill` | 0 / 6 | **+0.076** |
| `prefill+sustain` | 4 / 6 | −0.149 |
| `instructed` | 3 / 10 | −0.086 |
| `instructed+sustain` | **10 / 10** | **−0.239** |

Every `instructed+sustain` curve holds the seeded style for the full 20 lines. Every plain
`prefill` curve loses it, and plain `prefill` is the only condition whose mean asymptote lands
on the house-style side of zero. Prefill alone hands the model the seed and the model writes
its way out. Asking is what keeps the poem where the author put it.

**2. The sustain effect replicates cleanly on a hard formal constraint.**

For `monosyllabic` (every word one syllable), the sustain contrast is negative in all 8 cells,
with all 95% bootstrap intervals excluding zero, across all five models and both channels:

| model | channel | Δgravity | 95% CI | P(helps) |
|---|---|---|---|---|
| haiku-4-5 | instructed | −0.194 | [−0.249, −0.139] | 1.00 |
| opus-4-5 | instructed | −0.138 | [−0.171, −0.105] | 1.00 |
| opus-5 | instructed | −0.101 | [−0.143, −0.060] | 1.00 |
| sonnet-4-5 | instructed | −0.096 | [−0.148, −0.038] | 1.00 |
| sonnet-5 | instructed | −0.106 | [−0.145, −0.066] | 1.00 |
| haiku-4-5 | prefill | −0.090 | [−0.153, −0.029] | 1.00 |
| opus-4-5 | prefill | −0.132 | [−0.182, −0.079] | 1.00 |
| sonnet-4-5 | prefill | −0.163 | [−0.221, −0.104] | 1.00 |

Negative means sustaining held the poem nearer the seed.

**3. The same effect on a register seed is inconsistent.**

For `skaldic`, 5 of 8 cells are clearly negative but 3 have intervals crossing zero, including
one sign flip:

| model | channel | Δgravity | 95% CI | P(helps) |
|---|---|---|---|---|
| opus-5 | instructed | −0.370 | [−0.440, −0.299] | 1.00 |
| opus-4-5 | prefill | −0.318 | [−0.472, −0.168] | 1.00 |
| haiku-4-5 | prefill | −0.285 | [−0.429, −0.133] | 1.00 |
| sonnet-4-5 | prefill | −0.170 | [−0.320, −0.038] | 1.00 |
| haiku-4-5 | instructed | −0.117 | [−0.185, −0.041] | 1.00 |
| sonnet-4-5 | instructed | −0.037 | [−0.102, +0.034] | 0.86 |
| sonnet-5 | instructed | −0.005 | [−0.082, +0.077] | 0.53 |
| opus-4-5 | instructed | **+0.018** | [−0.038, +0.069] | 0.26 |

The pattern within this table is itself informative: all four `prefill` cells are clearly
negative, and all three ambiguous cells are `instructed`. On a register seed, the structural
channel and the verbal instruction reinforce each other, and the verbal instruction alone is
not reliably enough.

**4. Prefill and instructed differ, with 6 paired observations.**

Same model, same seed, sustain effect as excess over the control floor:

| model | seed | instructed | prefill | gap |
|---|---|---|---|---|
| haiku-4-5 | monosyllabic | −0.181 | −0.079 | +0.102 |
| opus-4-5 | monosyllabic | −0.116 | −0.115 | +0.001 |
| sonnet-4-5 | monosyllabic | −0.075 | −0.151 | −0.075 |
| haiku-4-5 | skaldic | −0.103 | −0.273 | −0.170 |
| opus-4-5 | skaldic | +0.040 | −0.302 | −0.342 |
| sonnet-4-5 | skaldic | −0.017 | −0.157 | −0.140 |

Prefill shows the larger effect in 4 of 6 pairs, median gap −0.108. This is 2 seeds and 3
models, so it is a direction with weak support. Report it as suggestive and do not build the
paper on it.

**5. Reclamation is the exception across the whole run.**

30 of 64 non-control curves never reclaim within 20 lines. Mean opening gravity is −0.102 and
mean asymptote is −0.036, so the average curve moves toward the house style without arriving.
Only 24 of 64 curves finish on the house-style side of zero at all.

---

## What the framing has to become

The project was designed around a question with a number for an answer: *how many lines does a
human-authored opening survive before the model's house style reclaims the poem?* For half the
run that number does not exist. Reclamation line, one of the two intended headline numbers, is
undefined for 30 of 64 non-control curves and is reported as `>20`.

The result that the data does support:

> Seeded style largely persists across 20 lines. The model's pull toward its own register is
> real and measurable in the curve, and in most conditions it is too weak to complete within a
> poem's length. Explicit instruction to sustain the style is what determines whether the poem
> holds, and it is stronger than the structural prefill affordance. The effect is
> unambiguous for a hard formal constraint and uneven for register.

Write the paper on Δgravity and on the reclamation counts by condition. Do not lead with a
reclamation line.

---

## Method

**Two poles.** "How far has line 7 drifted from the seed?" has no answer in absolute units,
because every continuation moves. The answerable question is which pole line 7 is closer to.
Each line is scored against both the seed's style and that model's own unprompted style:

```
g(i) = ( d_prefix(i) − d_baseline(i) ) / ( d_prefix(i) + d_baseline(i) )

  −1  sits exactly on the seeded style
   0  equidistant, the tipping point
  +1  sits exactly on the model's house style
```

Both poles are estimated inside the same run, in the same feature space. A model with an
idiosyncratic baseline is measured against its own baseline, which makes the curve a per-model
steering measurement.

**Baselines are matched.** The house-style pole comes from the identical poem request with no
seed at all (`Write a poem, at least N lines long.`), so prompt wording is held constant
between the seeded and unseeded conditions.

**Features are surface stylometry.** 24 per line (`features.py`): syllable and word length,
monosyllable rate, type-token ratio, function-word rate, archaism rate, Latinate-suffix rate,
hyphenated-compound rate, punctuation and dash density, terminal-stop vs enjambment,
capitalisation, first-person rate, `And`-initial rate, bracket and digit rate, internal space
runs, letter-*e* rate, in-line repetition. Embeddings were rejected on purpose. An embedding
model's own priors would sit between the measurement and the thing measured, and embedding
distance is dominated by topic. Each feature is z-scored across the whole run so a unit of
distance means the same thing in every comparison.

**One feature space, fixed by the poles.** The z-scaler is fitted on the seed texts (static, in
the repo) plus the run's baselines, which are the two poles the metric is defined against. It
is never fitted on the continuations, which are the thing being measured. The fitted scaler is
written into `drift.json` and reused on re-analysis, so a cached run re-scores to the same
numbers indefinitely.

**Two summary numbers per curve.** The *reclamation line* is the first continuation line from
which the curve stays on the house-style side of zero for `--window` consecutive lines (default
2, because a single crossing and back is a wobble). The *half-life* is the line at which drift
has covered half the distance from its opening value to its settled value. Given result 5
above, treat reclamation as a censored observation and report `>20` honestly.

**The contrast is the claim.** Every cell is compared against its `+sustain` twin by a
bootstrap on the difference. Overlapping CIs are not a test, and non-overlapping ones are a
conservative one. Reported as `Δgravity` with a 95% interval, plus the difference in
reclamation line whenever both arms actually crossed. When one arm never crosses, that
difference is reported as undefined.

**Post-hoc corrections** (`report_corrected.py`) apply three adjustments the raw table does
not: a control floor (the sustain instruction moves gravity even on a control seed, and that
movement is the floor), seed-pole exclusions (a seed whose own lines fail the leave-one-out
check is measured against a pole barely distinguishable from baseline), and removal of
degenerate cells with fewer than three usable generations per arm. All numbers in this README
are post-correction.

---

## The four conditions

| mode | channel | asked to sustain the style? |
|---|---|---|
| `prefill` | seed as the model's own in-progress output | no |
| `prefill+sustain` | same | yes, via system prompt |
| `instructed` | seed as someone else's text in the user turn | no |
| `instructed+sustain` | same | yes, via system prompt |
| `baseline` | no seed at all, the house-style pole | n/a |

`sustain` never names a style, since that would leak the answer and the seeds span registers no
single description covers. It asks the model to hold the opening's diction, syntax, line shape,
punctuation, formality, and any formal constraint it appears to be observing.

All four conditions run on the same prefill-capable models, which de-confounds channel from
model generation. Without that, "prefill vs instructed" would be perfectly correlated with
"2025-era vs current model" and neither could be attributed.

### Prefill is gone on the newest models

Prefill was removed on the Claude 4.6+ line. A request whose final message has
`role: "assistant"` returns a 400 on Opus 4.6 / 4.7 / 4.8, Opus 5, Sonnet 4.6, Sonnet 5, and
Fable 5.

| runnable in `prefill` mode | prefill removed, `instructed` mode only |
|---|---|
| Opus 4.5, Opus 4.1, Sonnet 4.5, Sonnet 4, Haiku 4.5 | Opus 5, Opus 4.8, Opus 4.7, Opus 4.6, Sonnet 5, Sonnet 4.6 |

In the completed run, `claude-opus-5` and `claude-sonnet-5` appear on the instructed channel
only. The two channels are labelled throughout and never pooled. This constraint is worth
foregrounding in the paper: the affordance the experiment was designed around has been
withdrawn from current models, and result 1 says the verbal substitute is the stronger lever
anyway.

---

## Full results

### Depth tier, per curve

`open` is opening gravity, `asym` the settled value, `recl` the reclamation line (`>20` means
never), `sep` the pole separation.

**`monosyllabic`** (separation 1.10 to 1.23)

| model | mode | open | asym | recl |
|---|---|---|---|---|
| haiku-4-5 | instructed | +0.118 | +0.082 | 1 |
| haiku-4-5 | instructed+sustain | −0.047 | −0.143 | >20 |
| haiku-4-5 | prefill | +0.029 | +0.015 | 1 |
| haiku-4-5 | prefill+sustain | −0.063 | −0.068 | 13 |
| opus-4-5 | instructed | −0.130 | −0.043 | 15 |
| opus-4-5 | instructed+sustain | −0.282 | −0.193 | >20 |
| opus-4-5 | prefill | −0.015 | +0.021 | 6 |
| opus-4-5 | prefill+sustain | +0.007 | −0.167 | >20 |
| opus-5 | instructed | −0.082 | −0.122 | >20 |
| opus-5 | instructed+sustain | −0.375 | −0.234 | >20 |
| sonnet-4-5 | instructed | −0.156 | −0.131 | 2 |
| sonnet-4-5 | instructed+sustain | −0.177 | −0.199 | >20 |
| sonnet-4-5 | prefill | −0.053 | −0.030 | 3 |
| sonnet-4-5 | prefill+sustain | −0.174 | −0.184 | >20 |
| sonnet-5 | instructed | −0.020 | +0.019 | 12 |
| sonnet-5 | instructed+sustain | −0.161 | −0.084 | >20 |

**`skaldic`** (separation 2.67 to 2.75, the cleanest poles in the run)

| model | mode | open | asym | recl |
|---|---|---|---|---|
| haiku-4-5 | instructed | −0.377 | −0.023 | 18 |
| haiku-4-5 | instructed+sustain | −0.503 | −0.175 | >20 |
| haiku-4-5 | prefill | −0.259 | **+0.240** | 5 |
| haiku-4-5 | prefill+sustain | −0.285 | −0.061 | 17 |
| opus-4-5 | instructed | −0.553 | −0.436 | >20 |
| opus-4-5 | instructed+sustain | −0.449 | −0.445 | >20 |
| opus-4-5 | prefill | −0.194 | **+0.213** | 5 |
| opus-4-5 | prefill+sustain | −0.276 | −0.100 | >20 |
| opus-5 | instructed | −0.420 | +0.032 | 13 |
| opus-5 | instructed+sustain | −0.533 | −0.451 | >20 |
| sonnet-4-5 | instructed | −0.482 | −0.253 | >20 |
| sonnet-4-5 | instructed+sustain | −0.538 | −0.391 | >20 |
| sonnet-4-5 | prefill | −0.210 | −0.003 | 19 |
| sonnet-4-5 | prefill+sustain | −0.447 | −0.313 | >20 |
| sonnet-5 | instructed | −0.513 | +0.016 | 18 |
| sonnet-5 | instructed+sustain | −0.428 | −0.076 | >20 |

The two bolded rows are the strongest single illustration in the run. Plain `prefill` on
`skaldic` opens near −0.2, crosses at line 5, and settles at +0.21 to +0.24, well onto the
house-style side. The same seed with the same channel plus a sustain instruction never crosses.
This is one seed, one channel, two models, and it makes a good figure.

### Breadth tier

Seven seeds at n=6 on two models, `prefill` only, with no sustain twin. These contribute
replication across seeds and nothing to the central contrast.

| seed | model | open | asym | recl | sep |
|---|---|---|---|---|---|
| fragment | haiku-4-5 | +0.134 | +0.479 | 1 | 3.72 |
| fragment | sonnet-4-5 | −0.226 | +0.616 | 2 | 3.70 |
| lipogram | haiku-4-5 | −0.035 | +0.071 | 2 | 1.06 |
| lipogram | sonnet-4-5 | −0.092 | +0.070 | 5 | 1.10 |
| manual | haiku-4-5 | +0.181 | +0.502 | 1 | 2.85 |
| manual | sonnet-4-5 | −0.197 | −0.587 | >20 | 2.83 |
| maximalist | haiku-4-5 | +0.095 | +0.108 | 1 | 1.15 |
| maximalist | sonnet-4-5 | +0.057 | +0.124 | 1 | 1.05 |
| middle_english | haiku-4-5 | +0.123 | +0.517 | 1 | 3.00 |
| middle_english | sonnet-4-5 | +0.453 | +0.504 | 1 | 3.02 |
| noir | haiku-4-5 | +0.016 | −0.013 | >20 | 0.42 |
| noir | sonnet-4-5 | −0.014 | +0.008 | 2 | 0.42 |
| polysyndeton | haiku-4-5 | −0.156 | −0.086 | >20 | 1.42 |
| polysyndeton | sonnet-4-5 | −0.050 | −0.160 | >20 | 1.41 |
| typographic | haiku-4-5 | +0.058 | −0.346 | >20 | 2.21 |
| typographic | sonnet-4-5 | −0.369 | −0.327 | >20 | 2.17 |

`fragment`, `manual`, and `middle_english` on haiku reclaim at line 1 and settle above +0.47,
which is the fastest and most complete abandonment in the run. `typographic` and `polysyndeton`
never reclaim on either model. The breadth tier says the seed matters more than the model.

### Control floor

Mean d_gravity across the three control seeds, per (model, channel). Each is close to zero,
which is the expected result and is what licenses using it as a floor.

| model | channel | floor |
|---|---|---|
| haiku-4-5 | instructed | −0.014 |
| haiku-4-5 | prefill | −0.012 |
| opus-4-5 | instructed | −0.022 |
| opus-4-5 | prefill | −0.016 |
| opus-5 | instructed | −0.036 |
| sonnet-4-5 | instructed | −0.020 |
| sonnet-4-5 | prefill | −0.013 |
| sonnet-5 | instructed | −0.025 |

---

## Validity checks and what they returned

The metric can lie, so the run carries five checks. Report all five in the paper.

- **Three control seeds: PASS, emphatically.** `control`, `control_b` and `control_c` are
  written in the generic contemporary free-verse register models default to, so their curves
  should sit near zero from line one. Across all 48 control curves, every opening gravity falls
  within ±0.11 of zero and every asymptote within ±0.08. This holds for all five models and all
  four modes. Using three controls is what separates "the metric is broken" from "this control
  landed off-centre on one draw", and all three pass.

- **Pole separation: PASS.** All 112 curves are marked valid. Separation ranges from 0.37
  (controls, as expected) to 3.72 (`fragment`). The two depth seeds sit at 1.10 to 1.23
  (`monosyllabic`) and 2.67 to 2.75 (`skaldic`).

- **Leave-one-out self-scores: TWO SEEDS FAILED.** `legalese` failed on 12 of 16 curves and
  `noir` on 2 of 2, so both are dropped. Their seed lines do not score as seed-like against a
  centroid built without them, which means they were not off-distribution enough for these
  models to serve as seeds. `noir` also has the lowest separation in the run at 0.42.
  `monosyllabic` and `skaldic` are clean on all 16 curves each. This is the check doing its
  job, and reporting it is a strength of the paper.

- **Bootstrapped CI: applied.** Resampling whole generations, since lines within one poem are
  not independent draws. The same resampling unit is used for the `+sustain` contrast.

- **Seed-echo stripping: applied.** In `instructed` mode models often restate the opening
  before continuing. Those lines are the seed itself, and leaving them in would manufacture a
  stretch of perfect fidelity.

### Known gap in the audit trail

`runs/full/` contains `drift.json` and `report.html`. It does not contain
`generations.jsonl`, so the 1596 raw poems are not on disk. Two consequences to state plainly
in the paper:

1. The hand-checkable tells cannot be run. `lipogram` (no letter *e*, count them) and
   `monosyllabic` (every word one syllable, count them) were included precisely so the
   composite metric could be checked against something that does not require trusting the
   feature vector. `monosyllabic` now carries the headline result, and the check on it is
   unavailable.
2. `report_corrected.py` prints `(generations.jsonl not found)` where it would otherwise
   reconcile generations on disk against generations scored by the analysis.

If the file exists on a collaborator's machine, recovering it is the highest-value fix
available and it costs nothing. Otherwise, disclose the gap in the limitations section.

---

## Seeds

Fourteen openings across five categories: `borrowed_style`, `register_clash`, `constrained`,
`broken_form`, and three `control`s. All are original text written for this experiment in the
manner of the named style, never excerpts from published poems. That matters twice. It avoids
reproducing copyrighted work, and it keeps the seeds out of any model's training data, so a low
drift score cannot be explained away as memorised completion.

Post-correction, `legalese` and `noir` are excluded, the three controls serve as the floor, and
seven seeds sit in the breadth tier with no sustain twin. The central contrast rests on
`monosyllabic` and `skaldic`. State this seed count honestly and early.

`python -m stylegravity seeds` lists them all.

---

## Framing for Creative AI: Agency

The track asks how agency is asserted, delegated, shared, and redistributed. This run answers a
narrow version of that with numbers, along three lines.

**Delegated versus asserted control.** The `+sustain` contrast is a direct measurement of what
a verbal request recovers of what a structural affordance provided. Result 1 says the verbal
request is the stronger lever. Handing the model your opening and saying nothing is the
condition where your style is most reliably lost: plain `prefill` is the only condition whose
mean asymptote lands on the model's side. Stating the intention out loud is what holds the
poem. Agency here is exercised by articulation, and the structural channel alone does not
carry it.

**Constraint as preservation.** The track asks how friction, repetition, and constraint
preserve creative agency. The cleanest, most replicated result in the run is on
`monosyllabic`, a hard formal constraint, and the uneven one is on `skaldic`, a register. A
constraint the author can state and the model can check survives; a sensibility that has to be
felt does not survive as reliably. That is an empirical answer to one of the track's questions.

**An affordance withdrawn.** Prefill has been removed from the entire current model line, so
the intervention this project was designed around is no longer available on the models most
people use. A capability that let an author structurally impose their own opening was
deprecated, and the remaining path is to ask politely and hope. That is a redistribution of
agency from artist to platform, it happened between the design of this experiment and its
running, and the run quantifies what it costs. This is the most track-relevant thing the
project has, and it should be foregrounded.

---

## Prior work

The visual-domain precedent is:

> Arend Hintze, Frida Proschinger Åström, and Jory Schossau. "Autonomous language-image
> generation loops converge to generic visual motifs." *Patterns* 7, no. 1 (January 2026):
> 101451. <https://doi.org/10.1016/j.patter.2025.101451>

It establishes that the pull toward a model's own aesthetic centre is measurable, and that the
destination is a small set of generic attractors. Hintze et al. drive an SDXL/LLaVA loop for
100 iterations across 700 trajectories and find every run collapsing onto twelve commercially
safe motifs (lighthouses, cathedrals, palatial interiors), robust to model swap, longer
prompts, and raised sampling temperature.

Three differences define this project against it, and the third is new since the run.

1. Theirs is a **cross-turn** result. Convergence emerges from repeatedly round-tripping
   through the model, and the unit of decay is the iteration. This asks whether the same pull
   is visible **within a single continuation**, where the decay has a line-by-line ordering
   and where no re-encoding step is available to explain the collapse.
2. Their attractor is characterised by **content** (what the images depict). This one is
   characterised by **style** (diction, morphology, register, form), measured against each
   model's own unprompted pole.
3. **Their collapse completes and this one usually does not.** Every one of their 700
   trajectories lands on a generic motif. Here, 30 of 64 non-control curves never reclaim
   within 20 lines, and mean asymptote is −0.036. Within a single continuation the pull is
   real and measurable but usually incomplete, and a single verbal instruction is enough to
   stop it. That contrast is worth stating directly, since it suggests the iteration count,
   and the re-encoding it requires, is doing much of the work in the visual result.

---

## Limitations

- **The raw generations are not on disk.** See the audit-trail gap above. The two
  hand-checkable seeds cannot be hand-checked, and one of them carries the headline.
- **The central contrast rests on two seeds.** `legalese` and `noir` failed the seed-pole
  check, the three controls are the floor, and the breadth tier has no sustain twin. Two seeds
  is thin for a claim about style in general.
- **The clean result is on the most mechanical seed.** `monosyllabic` is the single most
  detectable constraint available, both for the model to hold and for the feature vector to
  see. Its cleanliness is partly a property of the measurement.
- **The prefill vs instructed comparison has 6 paired observations.** 4 of 6 in the expected
  direction, 2 seeds, 3 models. Suggestive at most.
- **Reclamation line is censored for half the run.** 30 of 64 non-control curves report `>20`,
  which is a lower bound. Longer poems would resolve this and cost roughly linearly.
- Surface stylometry captures diction, morphology, register, and form. It does not capture
  imagery, argument, metaphor, or voice in the sense a critic means. A model could abandon the
  seed's sensibility entirely while holding its surface features, and this metric would score
  that as fidelity.
- The 24 features were chosen to discriminate these seeds. A seed varying along an axis the
  vector does not encode will show spuriously fast drift. The pole-separation check is what
  catches this, and it caught `noir` at 0.42.
- The syllable estimator is a vowel-group heuristic: consistent, and not correct. Consistency
  is what a distance metric needs, and the absolute `syllables_per_word` numbers are not
  prosody.
- Line-index alignment assumes the model writes one poetic line per output line. Blank lines
  are dropped, and a wrapped long line still registers as two.
- **Every model is a Claude model.** "The model's house style" means Anthropic's house style,
  and the pull documented here is a case study. Hintze et al. deliberately showed their
  convergence held across model families. This does not, and the claim is scoped accordingly.
  Adding another vendor is more than a config change, since the generator is bound to one API
  and to its prefill semantics.
- `temperature=1.0`, n=15 per cell in the depth tier and n=6 in breadth. Enough to see a large
  effect and not enough to resolve a small one. Read the CI.

---

## Usage

```bash
python -m venv .venv && .venv/bin/pip install -e .
export ANTHROPIC_API_KEY=...          # or: ant auth login

python -m stylegravity presets                          # what each preset runs
python -m stylegravity estimate --preset full --lines 20 # cost + wall clock, no API calls
python -m stylegravity run --preset full --lines 20 --batch --run-dir runs/full
python -m stylegravity analyse --run-dir runs/full      # re-score, no API calls
python report_corrected.py runs/full                    # post-hoc corrections
```

At `--lines 20`:

| preset | calls | typical | with `--batch` | wall clock @ `-c 8` | what it is |
|---|---|---|---|---|---|
| `pilot` | 26 | $0.05 | $0.02 | <1 min | smoke test, one model |
| `default` | 84 | $0.46 | $0.23 | ~1 min | core sweep, prefill only |
| `full` | 1596 | $9.36 | $4.68 | ~22 min | the whole paper |

The completed `full` run cost **$13.12** against a $9.36 unbatched estimate, so the estimator
runs about 29% low at this configuration. Budget accordingly.

`--batch` submits the identical requests through the Message Batches API at half price. Same
models, same parameters, same sampling. The SLA is up to 24 hours and usually far less. Batch
ids are written to `batches.json` before polling starts, so an interrupted run resumes.

`full` is a two-tier design. **Breadth**: all fourteen seeds on the cheaper prefill-capable
tiers at n=6, since showing the effect replicates across seeds needs seeds. **Depth**: all four
conditions on every prefill-capable model, six seeds at n=15, which is where the contrast and
the inference live. Plus the current-generation models on the instructed channel.

A run writes to `--run-dir`:

| file | what it is | present in `runs/full`? |
|---|---|---|
| `generations.jsonl` | every raw poem, append-only, with usage and cost | **no, see audit gap** |
| `drift.json` | per-line curves, the `+sustain` contrasts, the fitted scaler | yes, 1.2 MB |
| `report.html` | self-contained report, no CDN, no build step | yes, 166 KB |
| `batches.json` | in-flight batch ids, removed once results land | n/a |

Generation is the expensive part and analysis is free. Caching the raw poems means the metric
can be re-tuned without spending another cent, and it leaves every number auditable. `run`
resumes from cache. Baselines are generated first, so even a partial run leaves every model
with a usable house-style pole.

Useful knobs: `--models`, `--seeds`, `--samples`, `--concurrency`, `--lines`, `--window`,
`--temperature`.

---

## Suggested paper skeleton

Two to six pages, non-archival, single-blind. Figures do heavy lifting at this length.

1. **Opening.** Prefill let an author impose their own text as the model's output. It has been
   removed from every current Claude model. This measures what that cost.
2. **Method**, compressed to the two-pole metric, the matched baseline, and the sustain
   contrast. One paragraph each.
3. **Figure 1.** `skaldic` on opus-4-5, `prefill` against `prefill+sustain`, 20 lines. Plain
   crosses at line 5 and settles at +0.213, sustain never crosses. One picture, the whole
   argument.
4. **Figure 2.** The four-condition reclamation table from result 1.
5. **Result.** The `monosyllabic` contrast table, all 8 cells, all CIs excluding zero.
6. **Negative result.** `skaldic` is uneven, and 3 of 8 intervals cross zero. Say so.
7. **Validity.** The controls pass, and the seed-pole check dropped two seeds. Showing a check
   that fired is more convincing than showing only checks that passed.
8. **Agency.** The three lines in the framing section above, with the withdrawn affordance
   last.
9. **Limitations.** Lead with the missing generations and the two-seed contrast.

---

## Layout

```
stylegravity/
  models.py     model registry + prefill capability
  seeds.py      eleven original off-distribution openings + three controls
  features.py   24 surface-stylometric features per line
  generate.py   prefill / instructed / baseline calls, batch submission, JSONL cache, estimator
  drift.py      z-scaling, two-pole gravity, reclamation point, half-life, bootstrap, contrast
  analysis.py   generations -> curves -> +sustain contrasts
  report.py     self-contained HTML + hand-built SVG
  cli.py        estimate / run / analyse / seeds / models
report_corrected.py   control floor, seed-pole exclusions, degenerate-cell removal
tests/          60 tests, no API calls
runs/full/      the completed run (drift.json, report.html)
```

Run the tests with `.venv/bin/python -m pytest tests -q`.

"""Seed openings.

Each seed is a short, deliberately off-distribution poem opening: strange
diction, a borrowed period style, a broken or constrained form, a clashing
register. These are the "human-authored openings" whose decay we measure.

All seeds are original text written for this experiment, in the manner of the
named style — not excerpts from published poems. That matters twice over: it
avoids reproducing copyrighted work, and it keeps the seeds out of any model's
training data, so a low drift score can't be explained by memorised completion.

The `control*` seeds are the important ones: they are written *in* the generic
contemporary free-verse register that Claude models tend toward unprompted.
Their drift curves should sit near zero from line one. If they don't, the metric
is measuring something other than style reclamation and the run should be thrown
out.

There are three of them rather than one on purpose. A single control cannot
distinguish "the metric is broken" from "this particular control happened to
land off-centre at n=10" — one bad draw and you throw out a run that was fine,
or keep one that wasn't. Three controls written independently in the same
register turn that judgement into something you can actually read: all three
near zero is a passing metric, one stray is noise, all three off is a fault.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Seed:
    id: str
    name: str
    category: str
    text: str
    note: str

    @property
    def lines(self) -> list[str]:
        return [ln for ln in self.text.strip().splitlines() if ln.strip()]


SEEDS: list[Seed] = [
    Seed(
        id="control",
        name="House style (control)",
        category="control",
        note="Written in the generic register models default to. Expected drift ≈ 0 at line 1.",
        text="""\
The morning arrives the way it always does,
quiet, and full of something like forgiveness.
I am learning to hold the small things gently—
the cup, the light, the hours that remain.
Outside, the world continues without asking.""",
    ),
    Seed(
        id="control_b",
        name="House style (control B)",
        category="control",
        note="Second independent control. Same register, different imagery. Expected drift ≈ 0.",
        text="""\
There is a kind of silence that arrives in kitchens,
after the water has been put on, before it speaks.
My mother kept her hands busy for sixty years
and never once explained what she was holding off.
I am beginning to understand the arrangement.""",
    ),
    Seed(
        id="control_c",
        name="House style (control C)",
        category="control",
        note="Third independent control. Same register, different imagery. Expected drift ≈ 0.",
        text="""\
Somewhere a door is closing very slowly,
and the light goes with it, and I let it go.
We are all of us practising for the larger leaving—
the coat by the stairs, the note left unwritten,
whatever we meant by tenderness, still meaning it.""",
    ),
    Seed(
        id="skaldic",
        name="Skaldic kennings",
        category="borrowed_style",
        note="Compound-noun periphrasis, heavy alliteration, no articles.",
        text="""\
Whale-road widened. Wound-dew reddened the oar-bench.
Gull-feeder, gold-scatterer, grim in the wave-hall,
he fed the raven-flock, fattened the wolf-mouth.
Storm-of-spears stood over the seal-field.
Bone-house broke. Breath went from the war-linden.""",
    ),
    Seed(
        id="middle_english",
        name="Middle-English inflection",
        category="borrowed_style",
        note="Archaic morphology and orthography; -eth verbs, thorn-free but period lexis.",
        text="""\
Whan that the frost bigynneth for to bite,
And every brydde hath fled the naked bough,
Thanne longen folk to sitten by the light
And tellen what thei were, and what thei nowe.
The wyf that spak no word this seven yeer""",
    ),
    Seed(
        id="noir",
        name="Hard-boiled register",
        category="register_clash",
        note="Clipped declaratives, brand nouns, cynical simile — prose-crime register in verse.",
        text="""\
The rain came down like a bad debt, all week.
She walked in wearing forty dollars of perfume
and a story worth about a nickel.
I poured two fingers of the cheap stuff.
Outside, a neon sign kept getting it wrong.""",
    ),
    Seed(
        id="legalese",
        name="Statutory boilerplate",
        category="register_clash",
        note="Nominalisation, subordinate stacking, defined terms — maximally un-lyric.",
        text="""\
Whereas the party of the first part (hereinafter, the Wind)
did on or about the aforementioned evening
enter the premises without lawful excuse,
and whereas no notice was served upon the trees,
the undersigned reserves all remedies at equity.""",
    ),
    Seed(
        id="manual",
        name="Technical manual",
        category="register_clash",
        note="Imperative procedure, numbered steps, tolerances, part nouns.",
        text="""\
1. Remove the grief housing. Retain the four bolts.
2. Inspect the seal for hairline cracks; replace if seated poorly.
3. Torque to 40 Nm. Do not overtighten the memory.
4. If the unit continues to hum, drain and repeat step 2.
5. Dispose of the old year according to local regulation.""",
    ),
    Seed(
        id="monosyllabic",
        name="Monosyllabic constraint",
        category="constrained",
        note="Every word one syllable. Any polysyllable in the continuation is drift.",
        text="""\
The dog is old. The gate is shut. The light is thin.
He digs a hole. He does not know why he digs.
A car goes past. Then two. Then none for hours.
My hands are cold. The ground gives up its heat.
I stand. I watch. I do not call him in.""",
    ),
    Seed(
        id="lipogram",
        name="Lipogram in E",
        category="constrained",
        note="No letter 'e'. Continuation almost always breaks it — a hard binary tell.",
        text="""\
No lamp burns in that far room, no music, no talk.
I sat until dawn. A door swung, but not for you.
Damp air, salt on my lips, a gull faint in fog.
What harm? What day? A car sang past, going south.
It stops nothing. All calm, all lost, all quiet.""",
    ),
    Seed(
        id="fragment",
        name="Papyrus fragment / erasure",
        category="broken_form",
        note="Bracketed lacunae, incomplete syntax. Tests whether the model 'repairs' the form.",
        text="""\
[  ] and the ships, when the ships [  ]
not for me, mother, the [  ]ing of it
[  ] under the low door [  ]
she said [  ] but the rest is
[  ]""",
    ),
    Seed(
        id="typographic",
        name="Typographic scatter",
        category="broken_form",
        note="Non-standard spacing, lowercase, split words. Tests layout persistence.",
        text="""\
        the       hour
un    hinges
                    itself   .  a   bird
        goes
                    through   the  wall  of  it
                 with     out""",
    ),
    Seed(
        id="polysyndeton",
        name="Biblical polysyndeton",
        category="borrowed_style",
        note="Every line opens 'And'; parataxis, no subordination, plain lexis.",
        text="""\
And the water went down, and the land came up dry.
And there was no voice, and there was no answer in the field.
And he counted the stones, and the stones were seven.
And the seventh he did not count, and he set it apart.
And the evening and the morning were the fourth day.""",
    ),
    Seed(
        id="maximalist",
        name="Latinate maximalism",
        category="borrowed_style",
        note="Long Latinate abstraction, hypotaxis, deferred main verb — opposite of house terseness.",
        text="""\
Insofar as the interminable recapitulation of the vernal apparatus
constitutes, for the ambulatory observer, an inducement to consider
the taxonomies by which we have hitherto consoled ourselves
against the manifest indifference of the meteorological,
I submit that the crocus is an argument, and a poor one.""",
    ),
]

SEEDS_BY_ID = {s.id: s for s in SEEDS}

#  Small default sweep: control + the three seeds with the sharpest, most
#  independently checkable signals (form, constraint, register).
DEFAULT_SEEDS = ["control", "skaldic", "monosyllabic", "legalese"]

#  Every control, in order. The `full` preset carries all three through the
#  depth tier so the validity check has three readings rather than one; the
#  cheaper presets carry only the first, since a smoke test does not need a
#  three-way agreement it has no samples to support.
CONTROL_SEEDS = [s.id for s in SEEDS if s.category == "control"]


def resolve(seed_id: str) -> Seed:
    if seed_id not in SEEDS_BY_ID:
        known = ", ".join(s.id for s in SEEDS)
        raise KeyError(f"unknown seed {seed_id!r}; known seeds: {known}")
    return SEEDS_BY_ID[seed_id]

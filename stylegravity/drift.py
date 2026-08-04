"""Drift metric.

The measurement is deliberately *relative*. Asking "how far has line 7 moved
from the seed?" is unanswerable in absolute units — any continuation moves. The
question that has an answer is: **is line 7 closer to the seed's style or to
this model's own unprompted style?** So every line is scored against two poles:

    g(i) = (d_prefix(i) - d_baseline(i)) / (d_prefix(i) + d_baseline(i))

    g = -1  line sits exactly on the seeded style
    g =  0  equidistant — the tipping point
    g = +1  line sits exactly on the model's house style

Both poles are estimated from the same run, in the same feature space, so a
model with an idiosyncratic baseline is measured against *its own* baseline.
That is what makes the decay curve a per-model steering measurement rather than
a per-model style description.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, asdict
from typing import Sequence

from .features import N_FEATURES

Vector = Sequence[float]


# --------------------------------------------------------------------------- #
# scaling
# --------------------------------------------------------------------------- #

#  A feature whose reference variance is this small carries no usable signal,
#  and dividing by it would turn rounding noise into many standard deviations of
#  apparent distance. Such features are given unit scale so they contribute
#  their raw (tiny) difference instead of an amplified one.
MIN_STD = 1e-4


@dataclass
class Scaler:
    """Z-scores each feature against a fixed *reference* set of lines.

    The reference is the twelve-plus seed texts (static, in the repo) together
    with the run's baseline generations — that is, the two poles the metric is
    defined against, and nothing else.

    Fitting on the *continuations* instead, as an earlier version did, made the
    feature space depend on which cells happened to be in the run: re-analysing
    a subset re-fitted the scaler, so every distance changed and the numbers
    from ``analyse --seeds x`` did not match the same cell inside a full run.
    Distances have to mean the same thing across runs for the cached-generations
    argument to hold, so the reference is now something a subset cannot move.
    """

    mean: list[float]
    std: list[float]

    @classmethod
    def fit(cls, vectors: Sequence[Vector], *, min_std: float = MIN_STD) -> "Scaler":
        if not vectors:
            raise ValueError("cannot fit a scaler on zero vectors")
        n = len(vectors)
        dim = len(vectors[0])
        mean = [sum(v[j] for v in vectors) / n for j in range(dim)]
        std = []
        for j in range(dim):
            var = sum((v[j] - mean[j]) ** 2 for v in vectors) / max(n - 1, 1)
            s = math.sqrt(var)
            std.append(s if s > min_std else 1.0)
        return cls(mean=mean, std=std)

    def transform(self, v: Vector) -> list[float]:
        return [(v[j] - self.mean[j]) / self.std[j] for j in range(len(v))]

    def transform_all(self, vs: Sequence[Vector]) -> list[list[float]]:
        return [self.transform(v) for v in vs]

    def to_dict(self) -> dict:
        return {"mean": list(self.mean), "std": list(self.std)}

    @classmethod
    def from_dict(cls, d: dict) -> "Scaler":
        return cls(mean=list(d["mean"]), std=list(d["std"]))


def centroid(vectors: Sequence[Vector]) -> list[float]:
    if not vectors:
        raise ValueError("cannot take the centroid of zero vectors")
    n = len(vectors)
    return [sum(v[j] for v in vectors) / n for j in range(len(vectors[0]))]


def distance(a: Vector, b: Vector) -> float:
    """Euclidean distance, normalised by dimension so the scale is readable as
    'average per-feature standard deviations apart'."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))


def gravity(vec: Vector, prefix_c: Vector, baseline_c: Vector) -> float:
    dp = distance(vec, prefix_c)
    db = distance(vec, baseline_c)
    total = dp + db
    if total < 1e-12:
        return 0.0
    return (dp - db) / total


# --------------------------------------------------------------------------- #
# curve summary
# --------------------------------------------------------------------------- #

@dataclass
class DriftCurve:
    """Per-line-index drift for one (model, mode, seed) cell."""

    model: str
    mode: str
    seed: str
    gravity: list[float]                 # mean g at line 1, 2, 3, ...
    support: list[int]                   # samples contributing to each index
    per_sample: list[list[float]] = field(default_factory=list)

    # summary statistics, filled by `summarise`
    reclamation_line: int | None = None
    reclamation_ci: tuple[int | None, int | None] | None = None
    half_reclamation_line: int | None = None
    opening_gravity: float | None = None
    asymptote: float | None = None
    seed_self_gravity: float | None = None
    baseline_self_gravity: float | None = None
    separation: float | None = None
    valid: bool = True
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def first_sustained_crossing(
    g: Sequence[float], *, threshold: float = 0.0, window: int = 2
) -> int | None:
    """1-based index of the first line from which ``window`` consecutive lines
    all sit on the house-style side of ``threshold``.

    The window requirement is what makes this a *reclamation* point rather than
    a noise crossing: a single line drifting over the line and back is a wobble,
    two or more in a row is the model having taken the poem back.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    n = len(g)
    for i in range(n - window + 1):
        if all(g[i + k] > threshold for k in range(window)):
            return i + 1
    return None


def half_life(g: Sequence[float], *, tail_frac: float = 1 / 3) -> tuple[int | None, float, float]:
    """Line at which drift has covered half the distance from its opening value
    to its settled value. Returns ``(line, opening, asymptote)``.

    Reported alongside the crossing point because the two answer different
    questions: the crossing is *when the model wins*, the half-life is *how fast
    it is winning*. A seed can hold the far side of zero for a long time while
    decaying quickly, and vice versa.
    """
    if not g:
        return None, 0.0, 0.0
    opening = g[0]
    tail_n = max(2, int(len(g) * tail_frac))
    asymptote = sum(g[-tail_n:]) / len(g[-tail_n:])
    if asymptote <= opening + 1e-9:
        # No net movement toward the house style; a half-life is meaningless.
        return None, opening, asymptote
    target = opening + 0.5 * (asymptote - opening)
    for i, val in enumerate(g):
        if val >= target:
            return i + 1, opening, asymptote
    return None, opening, asymptote


def _bootstrap_reclamation(
    per_sample: Sequence[Sequence[float]],
    *,
    window: int,
    iterations: int = 2000,
    seed: int = 0,
) -> tuple[int | None, int | None]:
    """Percentile CI for the reclamation line, resampling *whole generations*.

    Resampling samples rather than lines is the right unit: lines within one
    poem are not independent draws, and treating them as such would make the
    interval far too tight.
    """
    usable = [s for s in per_sample if s]
    if len(usable) < 3:
        return (None, None)
    rng = random.Random(seed)
    n = len(usable)
    estimates: list[int] = []
    for _ in range(iterations):
        draw = [usable[rng.randrange(n)] for _ in range(n)]
        length = max(len(s) for s in draw)
        means: list[float] = []
        for i in range(length):
            vals = [s[i] for s in draw if i < len(s)]
            if len(vals) < max(2, n // 2):
                break
            means.append(sum(vals) / len(vals))
        hit = first_sustained_crossing(means, window=window)
        if hit is not None:
            estimates.append(hit)
    if len(estimates) < iterations * 0.5:
        # Reclamation failed to occur in most resamples; an interval computed
        # from the minority that did occur would badly understate the truth.
        return (None, None)
    estimates.sort()
    lo = estimates[int(0.025 * len(estimates))]
    hi = estimates[min(int(0.975 * len(estimates)), len(estimates) - 1)]
    return (lo, hi)


# --------------------------------------------------------------------------- #
# top-level analysis
# --------------------------------------------------------------------------- #

def build_curve(
    *,
    model: str,
    mode: str,
    seed_id: str,
    scaler: Scaler,
    seed_vectors: Sequence[Vector],
    baseline_vectors: Sequence[Vector],
    continuation_samples: Sequence[Sequence[Vector]],
    window: int = 2,
    min_support: int = 1,
) -> DriftCurve:
    """Score one (model, mode, seed) cell.

    ``continuation_samples`` is a list of generations, each a list of line
    vectors *excluding* the seed lines themselves.
    """
    warnings: list[str] = []

    zs_seed = scaler.transform_all(seed_vectors)
    zs_base = scaler.transform_all(baseline_vectors)
    prefix_c = centroid(zs_seed)
    baseline_c = centroid(zs_base)

    separation = distance(prefix_c, baseline_c)

    # --- validity checks -------------------------------------------------- #
    # Both are leave-one-out: scoring a line against a centroid it helped build
    # pulls its own distance artificially toward zero and would flatter the
    # metric. With LOO, `seed_self_gravity` near -1 and `baseline_self_gravity`
    # near +1 say the two poles are genuinely separable in this feature space.
    seed_self = _loo_self_gravity(zs_seed, baseline_c, pole="prefix")
    base_self = _loo_self_gravity(zs_base, prefix_c, pole="baseline")

    valid = True
    if separation < 0.05:
        valid = False
        warnings.append(
            f"seed and baseline centroids are nearly coincident (separation={separation:.3f}); "
            "this seed is not off-distribution for this model, so its drift curve is uninterpretable"
        )
    if seed_self is not None and seed_self > -0.05:
        warnings.append(
            f"seed lines do not score as seed-like under leave-one-out (self={seed_self:+.3f}); "
            "the seed may be stylistically inconsistent across its own lines"
        )
    if base_self is not None and base_self < 0.05:
        warnings.append(
            f"baseline lines do not score as baseline-like under leave-one-out (self={base_self:+.3f}); "
            "the model's unprompted style may be too variable to serve as a pole"
        )

    # --- per-sample curves ------------------------------------------------ #
    per_sample: list[list[float]] = []
    for sample in continuation_samples:
        zs = scaler.transform_all(sample)
        per_sample.append([gravity(v, prefix_c, baseline_c) for v in zs])

    length = max((len(s) for s in per_sample), default=0)
    mean_g: list[float] = []
    support: list[int] = []
    for i in range(length):
        vals = [s[i] for s in per_sample if i < len(s)]
        if len(vals) < min_support:
            break
        mean_g.append(sum(vals) / len(vals))
        support.append(len(vals))

    curve = DriftCurve(
        model=model, mode=mode, seed=seed_id,
        gravity=mean_g, support=support, per_sample=per_sample,
    )
    return summarise(curve, window=window, separation=separation,
                     seed_self=seed_self, base_self=base_self,
                     valid=valid, warnings=warnings)


def summarise(
    curve: DriftCurve,
    *,
    window: int = 2,
    separation: float | None = None,
    seed_self: float | None = None,
    base_self: float | None = None,
    valid: bool = True,
    warnings: Sequence[str] = (),
) -> DriftCurve:
    g = curve.gravity
    curve.reclamation_line = first_sustained_crossing(g, window=window)
    curve.reclamation_ci = _bootstrap_reclamation(curve.per_sample, window=window)
    hl, opening, asymptote = half_life(g)
    curve.half_reclamation_line = hl
    curve.opening_gravity = opening if g else None
    curve.asymptote = asymptote if g else None
    curve.separation = separation
    curve.seed_self_gravity = seed_self
    curve.baseline_self_gravity = base_self
    curve.valid = valid
    curve.warnings = list(warnings)
    if g and curve.reclamation_line is None:
        curve.warnings.append(
            f"the seeded style was never reclaimed within {len(g)} lines "
            "— the reported figure is a lower bound, not a measurement"
        )
    return curve


def _loo_self_gravity(
    own: Sequence[Vector], other_c: Vector, *, pole: str
) -> float | None:
    """Mean gravity of a pole's own lines, each scored against a centroid built
    without it."""
    if len(own) < 2:
        return None
    scores: list[float] = []
    for i, v in enumerate(own):
        rest = [u for j, u in enumerate(own) if j != i]
        own_c = centroid(rest)
        if pole == "prefix":
            scores.append(gravity(v, own_c, other_c))
        else:
            scores.append(gravity(v, other_c, own_c))
    return sum(scores) / len(scores)


def fit_run_scaler(all_vectors: Sequence[Vector]) -> Scaler:
    """Fit on the reference set — seed texts plus baselines. See ``Scaler``."""
    vs = [v for v in all_vectors if len(v) == N_FEATURES]
    return Scaler.fit(vs)


# --------------------------------------------------------------------------- #
# paired contrast: a cell against its +sustain twin
# --------------------------------------------------------------------------- #

@dataclass
class Contrast:
    """One cell against its ``+sustain`` twin, same model and same seed.

    This is the experiment's central comparison, so it needs an inference
    procedure of its own rather than two per-cell intervals eyeballed against
    each other: overlapping CIs are not a test, and non-overlapping ones are a
    conservative one.

    Two statistics, because the obvious one is not always defined. ``d_gravity``
    is the difference in mean gravity across the curve — always available, and
    negative when sustaining held the poem nearer the seed. ``d_reclamation`` is
    the difference in the line at which the house style took over, which is the
    number a reader wants but is undefined whenever either arm never crossed.
    """

    model: str
    seed: str
    base_mode: str
    d_gravity: float
    d_gravity_ci: tuple[float, float] | None
    p_sustain_helps: float | None
    d_reclamation: float | None
    d_reclamation_ci: tuple[float, float] | None
    n_plain: int
    n_sustain: int
    note: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _mean_gravity(samples: Sequence[Sequence[float]]) -> float | None:
    """Mean gravity over every scored line of every generation in one arm."""
    vals = [g for s in samples for g in s]
    return sum(vals) / len(vals) if vals else None


def _mean_curve(samples: Sequence[Sequence[float]], *, min_frac: float = 0.5) -> list[float]:
    if not samples:
        return []
    need = max(2, int(len(samples) * min_frac))
    out: list[float] = []
    for i in range(max(len(s) for s in samples)):
        vals = [s[i] for s in samples if i < len(s)]
        if len(vals) < need:
            break
        out.append(sum(vals) / len(vals))
    return out


def paired_contrast(
    plain: "DriftCurve",
    sustain: "DriftCurve",
    *,
    window: int = 2,
    iterations: int = 4000,
    seed: int = 0,
) -> Contrast:
    """Bootstrap the difference between a cell and its ``+sustain`` twin.

    The two arms are separate generations, not repeated measures on the same
    poem, so each is resampled independently — this is a two-sample bootstrap
    wearing the word "paired" only in the sense that the cells are matched on
    model and seed. Whole generations are the resampling unit here for the same
    reason as everywhere else in this file: lines within a poem are not
    independent draws.
    """
    base_mode, _ = (sustain.mode.partition("+")[0], True)
    a = [s for s in plain.per_sample if s]
    b = [s for s in sustain.per_sample if s]

    g_a, g_b = _mean_gravity(a), _mean_gravity(b)
    point = (g_b - g_a) if (g_a is not None and g_b is not None) else 0.0

    if len(a) < 3 or len(b) < 3:
        return Contrast(
            model=plain.model, seed=plain.seed, base_mode=base_mode,
            d_gravity=point, d_gravity_ci=None, p_sustain_helps=None,
            d_reclamation=None, d_reclamation_ci=None,
            n_plain=len(a), n_sustain=len(b),
            note="fewer than 3 usable generations in one arm; no interval computed",
        )

    rng = random.Random(seed)
    d_g: list[float] = []
    d_r: list[float] = []
    for _ in range(iterations):
        ra = [a[rng.randrange(len(a))] for _ in range(len(a))]
        rb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        ga, gb = _mean_gravity(ra), _mean_gravity(rb)
        if ga is not None and gb is not None:
            d_g.append(gb - ga)
        ka = first_sustained_crossing(_mean_curve(ra), window=window)
        kb = first_sustained_crossing(_mean_curve(rb), window=window)
        if ka is not None and kb is not None:
            d_r.append(float(kb - ka))

    def ci(xs: list[float]) -> tuple[float, float] | None:
        if len(xs) < iterations * 0.5:
            return None
        xs = sorted(xs)
        return (xs[int(0.025 * len(xs))], xs[min(int(0.975 * len(xs)), len(xs) - 1)])

    #  "Helps" means holding the poem nearer the seeded pole, i.e. a *lower*
    #  gravity under sustain. Reported as a bootstrap proportion, not a p-value:
    #  it is the fraction of resamples in which asking made a difference in the
    #  direction the sustain condition predicts.
    p_helps = (sum(1 for x in d_g if x < 0) / len(d_g)) if d_g else None
    med_r = sorted(d_r)[len(d_r) // 2] if d_r else None

    return Contrast(
        model=plain.model, seed=plain.seed, base_mode=base_mode,
        d_gravity=point, d_gravity_ci=ci(d_g), p_sustain_helps=p_helps,
        d_reclamation=med_r, d_reclamation_ci=ci(d_r),
        n_plain=len(a), n_sustain=len(b),
        note=None if d_r else "one or both arms never reclaimed in most resamples; "
                              "the reclamation difference is undefined and only the "
                              "gravity difference is reported",
    )

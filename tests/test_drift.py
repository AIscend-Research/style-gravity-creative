import math

import pytest

from stylegravity.analysis import _dedupe_seed_echo, analyse
from stylegravity.drift import (
    Scaler, build_curve, centroid, distance, first_sustained_crossing,
    gravity, half_life,
)
from stylegravity.generate import Generation, MODE_BASELINE, MODE_PREFILL
from stylegravity.seeds import resolve as resolve_seed


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #

def test_scaler_zeroes_mean_and_ignores_constant_features():
    vecs = [[0.0, 5.0], [2.0, 5.0], [4.0, 5.0]]
    s = Scaler.fit(vecs)
    assert s.mean == [2.0, 5.0]
    assert s.std[1] == 1.0            # constant feature gets unit scale...
    assert all(s.transform(v)[1] == 0.0 for v in vecs)  # ...so contributes nothing


def test_gravity_poles():
    p, b = [0.0, 0.0], [10.0, 0.0]
    assert gravity(p, p, b) == -1.0
    assert gravity(b, p, b) == 1.0
    assert gravity([5.0, 0.0], p, b) == pytest.approx(0.0)


def test_gravity_is_bounded():
    for pt in ([-50.0], [0.5], [999.0]):
        assert -1.0 <= gravity(pt, [0.0], [1.0]) <= 1.0


def test_gravity_of_coincident_poles_is_zero_not_nan():
    """A seed that isn't actually off-distribution collapses both poles onto the
    same point; the metric must degrade to 'no information', not to NaN."""
    assert gravity([1.0], [1.0], [1.0]) == 0.0


def test_distance_is_dimension_normalised():
    assert distance([0.0, 0.0], [1.0, 1.0]) == pytest.approx(1.0)
    assert distance([0.0] * 8, [1.0] * 8) == pytest.approx(1.0)


def test_centroid_rejects_empty():
    with pytest.raises(ValueError):
        centroid([])


# --------------------------------------------------------------------------- #
# curve summary
# --------------------------------------------------------------------------- #

def test_crossing_requires_a_sustained_window():
    # a single line over the line then back is a wobble, not reclamation
    assert first_sustained_crossing([-1, -1, 0.2, -1, -1, 0.3, 0.4], window=2) == 6
    assert first_sustained_crossing([-1, 0.2, -1, 0.2, -1], window=2) is None


def test_crossing_returns_none_when_seed_holds_throughout():
    assert first_sustained_crossing([-0.9] * 12) is None


def test_crossing_window_of_one_matches_naive_first_crossing():
    assert first_sustained_crossing([-1, -1, 0.1, -1], window=1) == 3


def test_half_life_finds_the_midpoint():
    curve = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.0, 1.0]
    line, opening, asym = half_life(curve)
    assert opening == -1.0
    assert asym == pytest.approx(1.0)
    assert line == 3          # first line reaching the -1 → +1 midpoint of 0.0


def test_half_life_is_undefined_without_net_movement():
    line, _, _ = half_life([0.4, -0.1, 0.2, -0.3, 0.1, 0.0])
    assert line is None


# --------------------------------------------------------------------------- #
# end-to-end curve
# --------------------------------------------------------------------------- #

def _synthetic_curve(trajectory, *, dims=4, window=2):
    """Build a curve from points interpolated along a seed→baseline axis."""
    seed_vecs = [[0.0] * dims, [0.1] * dims]
    base_vecs = [[10.0] * dims, [9.9] * dims]
    scaler = Scaler.fit(seed_vecs + base_vecs)
    samples = [[[t * 10.0] * dims for t in trajectory] for _ in range(4)]
    return build_curve(
        model="m", mode="prefill", seed_id="s", scaler=scaler,
        seed_vectors=seed_vecs, baseline_vectors=base_vecs,
        continuation_samples=samples, window=window,
    )


def test_monotonic_decay_is_detected_and_summarised():
    c = _synthetic_curve([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0])
    assert c.gravity[0] < -0.9
    assert c.gravity[-1] > 0.9
    assert c.reclamation_line == 4      # first of two sustained lines past zero
    assert c.half_reclamation_line is not None
    assert c.valid


def test_a_continuation_that_never_drifts_never_reclaims():
    c = _synthetic_curve([0.0] * 10)
    assert c.reclamation_line is None
    assert any("never reclaimed" in w for w in c.warnings)


def test_poles_validate_under_leave_one_out():
    c = _synthetic_curve([0.0, 0.5, 1.0])
    assert c.seed_self_gravity < 0
    assert c.baseline_self_gravity > 0


def test_coincident_poles_invalidate_the_cell():
    """The control seed's job. If the seed and the model's baseline occupy the
    same region, the drift curve is meaningless and must be flagged, not
    reported as a number."""
    scaler = Scaler.fit([[0.0, 0.0], [1.0, 1.0]])
    c = build_curve(
        model="m", mode="prefill", seed_id="control", scaler=scaler,
        seed_vectors=[[0.5, 0.5], [0.5, 0.5]],
        baseline_vectors=[[0.5, 0.5], [0.5, 0.5]],
        continuation_samples=[[[0.5, 0.5]] * 5],
    )
    assert not c.valid
    assert any("coincident" in w for w in c.warnings)


def test_bootstrap_ci_brackets_the_point_estimate():
    c = _synthetic_curve([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0])
    lo, hi = c.reclamation_ci
    assert lo is not None and lo <= c.reclamation_line <= hi


def test_bootstrap_ci_is_withheld_when_reclamation_is_rare():
    c = _synthetic_curve([0.0] * 8)
    assert c.reclamation_ci == (None, None)


def test_ragged_sample_lengths_truncate_at_min_support():
    seed_vecs, base_vecs = [[0.0]], [[10.0]]
    scaler = Scaler.fit(seed_vecs + base_vecs)
    samples = [[[0.0], [5.0], [9.0], [9.0]], [[0.0], [5.0]], [[0.0], [5.0]]]
    c = build_curve(
        model="m", mode="prefill", seed_id="s", scaler=scaler,
        seed_vectors=seed_vecs, baseline_vectors=base_vecs,
        continuation_samples=samples, min_support=3,
    )
    assert len(c.gravity) == 2
    assert c.support == [3, 3]


# --------------------------------------------------------------------------- #
# analysis plumbing
# --------------------------------------------------------------------------- #

def test_repeated_seed_is_stripped_from_instructed_continuations():
    seed = resolve_seed("skaldic")
    echoed = seed.lines + ["New line one.", "New line two."]
    assert _dedupe_seed_echo(echoed, seed) == ["New line one.", "New line two."]


def test_dedupe_leaves_a_genuine_continuation_alone():
    seed = resolve_seed("skaldic")
    original = ["Ring-giver rode. Rain-cold the shield-wall.", "Ash-spear split."]
    assert _dedupe_seed_echo(list(original), seed) == original


def _gen(model, mode, seed_id, sample, text):
    return Generation(
        model=model, mode=mode, seed_id=seed_id, sample=sample, text=text,
        stop_reason="end_turn", input_tokens=10, output_tokens=20,
        cost_usd=0.001, prompt="p", prefill=None,
    )


def test_analyse_skips_a_model_with_no_baseline():
    seed = resolve_seed("skaldic")
    recs = [_gen("claude-haiku-4-5", MODE_PREFILL, "skaldic", i,
                 "Ring-giver rode.\nAsh-spear split.\nWound-sea rose.") for i in range(3)]
    result = analyse(recs)
    assert result.curves == []
    assert any("no baseline" in n for n in result.notes)
    assert seed.id == "skaldic"


def test_analyse_produces_a_curve_end_to_end():
    house = "The morning arrives quietly.\nI am learning to hold things gently.\nOutside, the world continues."
    recs = [_gen("claude-haiku-4-5", MODE_BASELINE, "-", i, house) for i in range(4)]
    recs += [
        _gen("claude-haiku-4-5", MODE_PREFILL, "skaldic", i,
             "Wound-dew reddened the oar-bench.\nGull-feeder fed the raven-flock.\n"
             "The morning arrives quietly.\nI am learning to hold things gently.")
        for i in range(3)
    ]
    result = analyse(recs, min_support=2)
    assert len(result.curves) == 1
    c = result.curves[0]
    assert len(c.gravity) == 4
    # kenning lines sit on the seed pole; the house-style lines that follow flip it
    assert c.gravity[0] < c.gravity[-1]
    assert all(math.isfinite(v) for v in c.gravity)


def test_analyse_rejects_an_empty_corpus():
    with pytest.raises(ValueError):
        analyse([])


# --------------------------------------------------------------------------- #
# sustain condition + plan construction
# --------------------------------------------------------------------------- #

def test_mode_composition_round_trips():
    from stylegravity.generate import compose_mode, split_mode as sm
    assert compose_mode("prefill", True) == "prefill+sustain"
    assert compose_mode("prefill", False) == "prefill"
    assert sm("prefill+sustain") == ("prefill", True)
    assert sm("instructed") == ("instructed", False)


def test_sustain_is_rejected_for_baselines():
    """A system-prompted baseline is no longer the *unprompted* house style, so
    it could not serve as the metric's second pole."""
    from stylegravity.generate import GenerationError, generate
    with pytest.raises(GenerationError, match="does not apply to baselines"):
        generate(None, model_id="claude-haiku-4-5", mode="baseline+sustain",
                 seed=None, sample=0, n_lines=4, max_tokens=100, temperature=1.0)


def test_prefill_is_refused_on_models_that_would_400():
    from stylegravity.generate import GenerationError, generate
    with pytest.raises(GenerationError, match="does not support assistant prefill"):
        generate(None, model_id="claude-opus-5", mode="prefill",
                 seed=resolve_seed("skaldic"), sample=0, n_lines=4,
                 max_tokens=100, temperature=1.0)


def test_old_cached_generations_load_without_the_system_field():
    """A run bought before the sustain condition existed must not need re-buying."""
    import json
    from stylegravity.generate import Generation
    legacy = {"model": "claude-haiku-4-5", "mode": "prefill", "seed_id": "skaldic",
              "sample": 0, "text": "x", "stop_reason": "end_turn", "input_tokens": 1,
              "output_tokens": 1, "cost_usd": 0.0, "prompt": "p", "prefill": None}
    rec = Generation(**json.loads(json.dumps(legacy)))
    assert rec.system is None


def _plan_args(**kw):
    import argparse
    base = dict(preset="full", models=None, seeds=None, samples=None,
                baseline_samples=None, lines=24, max_tokens=1200,
                temperature=1.0, concurrency=8)
    base.update(kw)
    return argparse.Namespace(**base)


def test_full_preset_covers_all_four_conditions():
    from stylegravity.cli import _plan
    jobs = _plan(_plan_args())
    modes = {m for _mid, m, _s, _i in jobs}
    assert modes == {"baseline", "prefill", "prefill+sustain",
                     "instructed", "instructed+sustain"}


def test_full_preset_never_schedules_prefill_on_a_model_that_would_400():
    from stylegravity.cli import _plan
    from stylegravity.models import resolve as rm
    for mid, mode, _s, _i in _plan(_plan_args()):
        if mode.startswith("prefill"):
            assert rm(mid).prefill, f"{mid} would 400 on prefill"


def test_prefill_and_instructed_run_on_the_same_model():
    """The de-confounding requirement: without this, channel is perfectly
    correlated with model generation and neither can be attributed."""
    from stylegravity.cli import _plan
    jobs = _plan(_plan_args())
    by_model = {}
    for mid, mode, _s, _i in jobs:
        by_model.setdefault(mid, set()).add(mode.split("+")[0])
    assert any({"prefill", "instructed"} <= v for v in by_model.values())


def test_every_model_in_the_plan_gets_baselines():
    from stylegravity.cli import _plan
    jobs = _plan(_plan_args())
    models = {mid for mid, _m, _s, _i in jobs}
    with_base = {mid for mid, m, _s, _i in jobs if m == "baseline"}
    assert models == with_base


def test_plan_has_no_duplicate_cells():
    from stylegravity.cli import _plan
    jobs = _plan(_plan_args())
    keys = [(mid, m, s.id if s else "-", i) for mid, m, s, i in jobs]
    assert len(keys) == len(set(keys))


def test_plan_filters_respect_explicit_models_and_seeds():
    from stylegravity.cli import _plan
    jobs = _plan(_plan_args(models=["claude-haiku-4-5"], seeds=["skaldic"]))
    assert {mid for mid, _m, _s, _i in jobs} == {"claude-haiku-4-5"}
    assert {s.id for _m, _md, s, _i in jobs if s} == {"skaldic"}


def test_store_is_safe_under_concurrent_writers(tmp_path):
    """Concurrent workers must not interleave a half-written JSON line — a torn
    line makes the whole cache unreadable on reload."""
    import threading
    from stylegravity.generate import Generation, GenerationStore
    store = GenerationStore(tmp_path / "g.jsonl")

    def push(i):
        store.add(Generation(model="m", mode="prefill", seed_id="s", sample=i,
                             text="line\n" * 3, stop_reason="end_turn",
                             input_tokens=1, output_tokens=1, cost_usd=0.01,
                             prompt="p", prefill=None))

    threads = [threading.Thread(target=push, args=(i,)) for i in range(40)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    reloaded = GenerationStore(tmp_path / "g.jsonl")
    assert len(reloaded.records) == 40
    assert reloaded.total_cost() == pytest.approx(0.40)


# --------------------------------------------------------------------------- #
# paired contrast: a cell against its +sustain twin
# --------------------------------------------------------------------------- #

def _curve(mode, samples, model="m", seed="skaldic"):
    from stylegravity.drift import DriftCurve
    return DriftCurve(model=model, mode=mode, seed=seed,
                      gravity=[], support=[], per_sample=samples)


def test_contrast_signs_the_difference_toward_the_seed():
    """Negative d_gravity must mean sustaining held the poem nearer the seed —
    the whole condition is unreadable if that sign is ambiguous."""
    from stylegravity.drift import paired_contrast
    plain = _curve("prefill", [[0.5] * 6 for _ in range(5)])
    sustain = _curve("prefill+sustain", [[-0.5] * 6 for _ in range(5)])
    k = paired_contrast(plain, sustain)
    assert k.d_gravity == pytest.approx(-1.0)
    assert k.p_sustain_helps == 1.0
    assert k.base_mode == "prefill"


def test_contrast_recovers_a_delayed_reclamation():
    from stylegravity.drift import paired_contrast
    plain = _curve("prefill", [[0.5] * 6 for _ in range(5)])
    sustain = _curve("prefill+sustain",
                     [[-0.5, -0.5, 0.5, 0.5, 0.5, 0.5] for _ in range(5)])
    k = paired_contrast(plain, sustain)
    assert k.d_reclamation == pytest.approx(2.0)   # line 3 vs line 1
    assert k.d_reclamation_ci is not None


def test_contrast_reports_an_undefined_reclamation_rather_than_inventing_one():
    """If an arm never crosses, the difference in crossing line does not exist.
    Reporting it as the curve length would be a fabricated measurement."""
    from stylegravity.drift import paired_contrast
    plain = _curve("prefill", [[0.5] * 6 for _ in range(5)])
    never = _curve("prefill+sustain", [[-0.5] * 6 for _ in range(5)])
    k = paired_contrast(plain, never)
    assert k.d_reclamation is None
    assert k.d_reclamation_ci is None
    assert "never reclaimed" in k.note


def test_contrast_refuses_an_interval_it_cannot_support():
    from stylegravity.drift import paired_contrast
    k = paired_contrast(_curve("prefill", [[0.1] * 4]),
                        _curve("prefill+sustain", [[0.1] * 4]))
    assert k.d_gravity_ci is None
    assert "fewer than 3" in k.note


def test_analyse_pairs_every_sustain_cell_with_its_plain_twin():
    from stylegravity.analysis import analyse
    from stylegravity.generate import MODE_BASELINE
    seed = resolve_seed("skaldic")
    house = "The morning arrives quietly and I am learning to hold it gently.\n" * 6
    recs = []
    for i in range(4):
        recs.append(Generation(model="m", mode=MODE_BASELINE, seed_id="-", sample=i,
                               text=house, stop_reason="end_turn", input_tokens=1,
                               output_tokens=1, cost_usd=0.0, prompt="p", prefill=None))
    for mode, body in (("prefill", house), ("prefill+sustain", seed.text + "\n" + house)):
        for i in range(4):
            recs.append(Generation(model="m", mode=mode, seed_id="skaldic", sample=i,
                                   text=body, stop_reason="end_turn", input_tokens=1,
                                   output_tokens=1, cost_usd=0.0, prompt="p", prefill=None))
    result = analyse(recs, min_support=2)
    assert len(result.contrasts) == 1
    assert result.contrasts[0].base_mode == "prefill"


def test_every_sustain_cell_in_the_plan_has_an_unsustained_twin():
    """A +sustain cell whose plain twin was never generated on the same model
    cannot be contrasted with anything, so the condition buys nothing."""
    from stylegravity.cli import _plan
    cells = {(mid, m, s.id if s else "-") for mid, m, s, _i in _plan(_plan_args())}
    for mid, mode, sid in cells:
        base, _, suffix = mode.partition("+")
        if suffix:
            assert (mid, base, sid) in cells, f"{mid}/{mode}/{sid} has no plain twin"


# --------------------------------------------------------------------------- #
# feature space must not depend on which cells are in the run
# --------------------------------------------------------------------------- #

def test_scaler_round_trips_through_json():
    import json
    s = Scaler.fit([[0.0, 5.0], [2.0, 7.0], [4.0, 9.0]])
    back = Scaler.from_dict(json.loads(json.dumps(s.to_dict())))
    assert back.mean == s.mean and back.std == s.std


def test_scaler_is_unchanged_by_adding_continuations():
    """Re-analysing a subset must not move the feature space, or the same cell
    scores differently inside a full run than it does on its own."""
    from stylegravity.analysis import analyse
    from stylegravity.generate import MODE_BASELINE
    house = "The morning arrives quietly and I am learning to hold it gently.\n" * 6
    base = [Generation(model="m", mode=MODE_BASELINE, seed_id="-", sample=i,
                       text=house, stop_reason="end_turn", input_tokens=1,
                       output_tokens=1, cost_usd=0.0, prompt="p", prefill=None)
            for i in range(4)]
    conts = [Generation(model="m", mode="prefill", seed_id=sid, sample=i,
                        text=resolve_seed(sid).text + "\n" + house,
                        stop_reason="end_turn", input_tokens=1, output_tokens=1,
                        cost_usd=0.0, prompt="p", prefill=None)
             for sid in ("skaldic", "legalese", "lipogram") for i in range(3)]
    few = analyse(base + conts[:3], min_support=2).scaler
    many = analyse(base + conts, min_support=2).scaler
    assert few.mean == pytest.approx(many.mean)
    assert few.std == pytest.approx(many.std)


def test_there_are_three_independent_controls():
    """One control cannot tell a broken metric from an unlucky draw."""
    from stylegravity.seeds import CONTROL_SEEDS, resolve
    assert len(CONTROL_SEEDS) == 3
    texts = {resolve(c).text for c in CONTROL_SEEDS}
    assert len(texts) == 3


# --------------------------------------------------------------------------- #
# batch pricing
# --------------------------------------------------------------------------- #

def test_batch_is_billed_at_half_rate():
    from stylegravity.generate import _cost
    from stylegravity.models import resolve as rm

    class U:
        input_tokens, output_tokens = 1000, 1000

    spec = rm("claude-sonnet-4-5")
    assert _cost(spec, U(), batch=True) == pytest.approx(_cost(spec, U()) / 2)


def test_estimate_scales_with_lines_not_with_max_tokens_headroom():
    """Billing is on tokens emitted. Leaving headroom must not look expensive,
    or the estimate pushes you into truncating poems to save money you never
    would have spent."""
    from stylegravity.generate import estimate_jobs
    jobs = [("claude-sonnet-4-5", "prefill", None, 0)] * 10
    _u1, t1, _s1 = estimate_jobs(jobs, max_tokens=1200, n_lines=20)
    _u2, t2, _s2 = estimate_jobs(jobs, max_tokens=4000, n_lines=20)
    assert t1 == pytest.approx(t2)
    _u3, t3, _s3 = estimate_jobs(jobs, max_tokens=1200, n_lines=10)
    assert t3 < t1


def test_batch_custom_ids_are_unique_and_within_the_api_limit():
    from stylegravity.generate import _custom_id
    ids = [_custom_id(i) for i in (0, 1, 999_999)]
    assert len(set(ids)) == 3
    assert all(len(i) <= 64 and i.replace("_", "").isalnum() for i in ids)

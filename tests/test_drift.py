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

"""Command line entry point.

    python -m stylegravity estimate --preset full     # cost + wall clock, no API calls
    python -m stylegravity run --preset full --run-dir runs/full
    python -m stylegravity analyse --run-dir runs/full    # re-score, no API calls
    python -m stylegravity seeds | models | presets
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .analysis import analyse as analyse_records
from .drift import Scaler
from .generate import (
    MODE_BASELINE, MODE_INSTRUCTED, MODE_PREFILL, SUSTAIN,
    GenerationStore, collect_batches, estimate_jobs, generate, make_client,
    poll_batches, split_mode, submit_batches,
)
from .models import DEFAULT_MODELS, REGISTRY, resolve as resolve_model
from .report import write_report
from .seeds import CONTROL_SEEDS, DEFAULT_SEEDS, SEEDS, resolve as resolve_seed

ALL_SEEDS = [s.id for s in SEEDS]
CORE_SEEDS = DEFAULT_SEEDS                       # control + skaldic + monosyllabic + legalese
#  The depth tier carries every control, not just the first: the validity check
#  it supports is "do the controls sit near zero", and one reading cannot tell a
#  broken metric from an unlucky draw. The cheap presets keep a single control —
#  a smoke test has no samples to support a three-way agreement anyway.
DEPTH_SEEDS = CORE_SEEDS + [s for s in CONTROL_SEEDS if s not in CORE_SEEDS]
PREFILL_MODELS = [m for m, s in REGISTRY.items() if s.prefill and m in DEFAULT_MODELS]
CURRENT_MODELS = ["claude-opus-5", "claude-sonnet-5"]

#  Breadth does not need the priciest model or a large n: its job is to show the
#  effect replicates across fourteen seeds, and the statistics come from the
#  depth tier. Opus stays where the sustain contrast actually lives.
BREADTH_MODELS = [m for m in PREFILL_MODELS if m != "claude-opus-4-5"]
BREADTH_SAMPLES = 6

WITH_SUSTAIN = f"+{SUSTAIN}"


@dataclass(frozen=True)
class Block:
    """One factorial slab of the design: models × seeds × modes × samples.

    ``samples`` overrides the run-wide count for this block alone. Breadth and
    depth want different n — breadth is replication across seeds, depth carries
    the inference — and forcing one number on both means either paying for
    precision where it is not used or not having it where it is.
    """
    models: list[str]
    seeds: list[str]
    modes: list[str]
    note: str
    samples: int | None = None


#  The `full` preset is a deliberate two-tier design rather than a full
#  factorial. The headline figure needs *breadth* of seed (all twelve, one
#  condition); the agency argument needs *depth* of condition (four conditions,
#  four seeds). Crossing everything would quadruple the bill to answer questions
#  no one asked.
PRESETS: dict[str, tuple[list[Block], int]] = {
    "pilot": (
        [Block(["claude-haiku-4-5"], CORE_SEEDS, [MODE_PREFILL], "smoke test")],
        6,
    ),
    "default": (
        [Block(DEFAULT_MODELS, CORE_SEEDS, [MODE_PREFILL], "core sweep")],
        8,
    ),
    "full": (
        [
            Block(
                BREADTH_MODELS, ALL_SEEDS, [MODE_PREFILL],
                "breadth: every seed, at the cheaper prefill-capable tiers — "
                "shows the effect replicates across seeds, which needs seeds, not n",
                samples=BREADTH_SAMPLES,
            ),
            Block(
                PREFILL_MODELS, DEPTH_SEEDS,
                [MODE_PREFILL, MODE_PREFILL + WITH_SUSTAIN,
                 MODE_INSTRUCTED, MODE_INSTRUCTED + WITH_SUSTAIN],
                "depth: all four conditions on every prefill-capable model. Carries the "
                "central contrast, so it includes the plain `prefill` arm at full n even "
                "though breadth also covers part of it — a +sustain cell with no matching "
                "unsustained twin on the same model is not a contrast at all",
            ),
            Block(
                CURRENT_MODELS, DEPTH_SEEDS,
                [MODE_INSTRUCTED, MODE_INSTRUCTED + WITH_SUSTAIN],
                "current-generation models, instructed channel only (prefill returns 400 there)",
            ),
        ],
        12,
    ),
}


def _add_shared(p: argparse.ArgumentParser) -> None:
    p.add_argument("--preset", choices=sorted(PRESETS), default="default")
    p.add_argument("--models", nargs="+", default=None,
                   help="restrict the preset to these model ids")
    p.add_argument("--seeds", nargs="+", default=None,
                   help="restrict the preset to these seed ids")
    p.add_argument("--samples", type=int, default=None,
                   help="continuations per cell (default: 5, or 10 under --preset full)")
    p.add_argument("--baseline-samples", type=int, default=None,
                   help="unprompted poems per model, used as the house-style pole")
    p.add_argument("--lines", type=int, default=24)
    p.add_argument("--max-tokens", type=int, default=1200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--concurrency", type=int, default=8,
                   help="parallel API calls (default 8; lower it if you hit 429s)")
    p.add_argument("--batch", action="store_true",
                   help="submit via the Message Batches API at half price. Same models, "
                        "same params, same sampling — only the bill and the latency "
                        "differ (SLA is up to 24h, usually far less)")


def _plan(args) -> list[tuple[str, str, object, int]]:
    """(model, mode, seed, sample) jobs.

    Baselines come first so an interrupted run still leaves every model with a
    usable house-style pole — without one, none of that model's cells can be
    scored at all.
    """
    blocks, default_baselines = PRESETS[args.preset]
    samples = args.samples if args.samples is not None else (15 if args.preset == "full" else 5)
    n_base = args.baseline_samples if args.baseline_samples is not None else default_baselines

    blocks = [
        Block(
            [m for m in b.models if args.models is None or m in args.models],
            [s for s in b.seeds if args.seeds is None or s in args.seeds],
            b.modes, b.note,
            #  An explicit --samples overrides a block's own n; without one, the
            #  block keeps the count the design chose for it.
            samples=b.samples if args.samples is None else args.samples,
        )
        for b in blocks
    ]
    blocks = [b for b in blocks if b.models and b.seeds]

    jobs: list[tuple[str, str, object, int]] = []
    seen_models: list[str] = []
    for b in blocks:
        for m in b.models:
            if m not in seen_models:
                seen_models.append(m)
    for m in seen_models:
        for i in range(n_base):
            jobs.append((m, MODE_BASELINE, None, i))

    emitted: set[tuple[str, str, str, int]] = set()
    for b in blocks:
        for m in b.models:
            spec = resolve_model(m)
            for mode in b.modes:
                base, _ = split_mode(mode)
                if base == MODE_PREFILL and not spec.prefill:
                    continue     # 400s on this model; the block below covers it
                for sid in b.seeds:
                    for i in range(b.samples if b.samples is not None else samples):
                        key = (m, mode, sid, i)
                        if key in emitted:
                            continue
                        emitted.add(key)
                        jobs.append((m, mode, resolve_seed(sid), i))
    return jobs


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #

def cmd_seeds(args) -> int:
    for s in SEEDS:
        print(f"{s.id:<16} {s.category:<16} {s.name}")
        print(f"{'':<16} {s.note}")
    return 0


def cmd_models(args) -> int:
    print(f"{'id':<20} {'prefill':<9} label")
    for spec in REGISTRY.values():
        print(f"{spec.id:<20} {'yes' if spec.prefill else 'NO':<9} {spec.label}")
    print("\nModels marked NO had assistant prefill removed on the 4.6+ line; they can "
          "only be measured in `instructed` mode, which is a different intervention.")
    return 0


def cmd_presets(args) -> int:
    for name, (blocks, n_base) in PRESETS.items():
        print(f"\n{name}  ({n_base} baselines per model)")
        for b in blocks:
            print(f"  {len(b.models)}m × {len(b.seeds)}s × {len(b.modes)} mode(s): "
                  f"{', '.join(b.modes)}")
            print(f"      {b.note}")
    return 0


def _summarise_plan(args, jobs) -> None:
    batch = getattr(args, "batch", False)
    upper, typical, seconds = estimate_jobs(
        jobs, max_tokens=args.max_tokens, concurrency=args.concurrency,
        batch=batch, n_lines=args.lines,
    )
    by_mode: dict[str, int] = {}
    for _m, mode, _s, _i in jobs:
        by_mode[mode] = by_mode.get(mode, 0) + 1
    print(f"preset: {args.preset}   {len(jobs)} API calls"
          f"{'   (batch, half price)' if batch else ''}")
    for mode, n in sorted(by_mode.items()):
        print(f"  {mode:<22} {n:>5}")
    print(f"\ncost   typical ${typical:.2f}   ceiling ${upper:.2f} "
          f"(if every poem ran to --max-tokens {args.max_tokens})")
    if batch:
        print("time   batch SLA is up to 24h; usually far less. Wall clock is not "
              "a function of size here.")
    else:
        print(f"time   ~{seconds / 60:.0f} min at --concurrency {args.concurrency} "
              f"(~{seconds * args.concurrency / 3600:.1f} h serial)")
        print("       estimate from published throughput; ignores rate-limit backoff")


def cmd_estimate(args) -> int:
    _summarise_plan(args, _plan(args))
    return 0


def cmd_run(args) -> int:
    run_dir = Path(args.run_dir)
    store = GenerationStore(run_dir / "generations.jsonl")
    jobs = _plan(args)
    _summarise_plan(args, jobs)

    fallbacks = sorted({m for m, _md, _s, _i in jobs if not resolve_model(m).prefill})
    if fallbacks:
        print(
            "\nNOTE: " + ", ".join(resolve_model(m).label for m in fallbacks) +
            " do not support assistant prefill (removed on the Claude 4.6+ line).\n"
            "      They run in `instructed` mode only. That measures a related but different\n"
            "      thing and is reported separately — never pooled with prefill results.",
            file=sys.stderr,
        )

    todo = [j for j in jobs if not store.has(j[0], j[1], j[2].id if j[2] else "-", j[3])]
    print(f"\n{len(jobs) - len(todo)} already cached, {len(todo)} to run\n")
    if not todo:
        return _analyse(run_dir, store, args)

    client = make_client()
    started = time.monotonic()

    if args.batch:
        return _run_batched(args, run_dir, store, client, todo, started)

    counter = {"done": 0, "failed": 0}
    lock = threading.Lock()

    def work(job):
        mid, mode, seed, sample = job
        try:
            gen = generate(
                client, model_id=mid, mode=mode, seed=seed, sample=sample,
                n_lines=args.lines, max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        except Exception as exc:  # noqa: BLE001 — one bad cell must not kill the run
            return job, None, exc
        if gen.stop_reason == "max_tokens":
            # The final line was cut mid-phrase; scoring it would read as a
            # style break that the model never actually made.
            gen.text = "\n".join(gen.text.splitlines()[:-1])
        return job, gen, None

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = [pool.submit(work, j) for j in todo]
        for fut in as_completed(futures):
            job, gen, exc = fut.result()
            mid, mode, seed, sample = job
            tag = f"{mid} {mode} {seed.id if seed else '-'} #{sample}"
            with lock:
                if exc is not None:
                    counter["failed"] += 1
                    print(f"    FAILED {tag}: {exc}", file=sys.stderr)
                else:
                    store.add(gen)
                counter["done"] += 1
                n = counter["done"]
                if n % 10 == 0 or n == len(todo):
                    rate = n / max(time.monotonic() - started, 1e-6)
                    eta = (len(todo) - n) / rate / 60
                    print(f"[{n}/{len(todo)}] ${store.total_cost():.2f} spent · "
                          f"~{eta:.0f} min left", flush=True)

    if counter["failed"]:
        print(f"\n{counter['failed']} generations failed; re-run the same command to "
              "retry only those (everything else is cached)", file=sys.stderr)
    print(f"\nspent ${store.total_cost():.2f} across {len(store.records)} generations "
          f"in {(time.monotonic() - started) / 60:.1f} min")
    return _analyse(run_dir, store, args)


def _run_batched(args, run_dir: Path, store: GenerationStore, client, todo, started) -> int:
    """Submit ``todo`` as batches, wait, collect.

    The batch ids and their custom_id mapping are written to disk *before* the
    poll begins. Batches survive the process that created them, so a run
    interrupted while waiting must be recoverable — without the mapping on disk
    the results would still exist and still be billed, but nothing could say
    which cell each one belonged to.
    """
    state_path = run_dir / "batches.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        batch_ids = state["batch_ids"]
        mapping = {k: tuple(v) for k, v in state["mapping"].items()}
        print(f"resuming {len(batch_ids)} batch(es) from {state_path}\n")
    else:
        batch_ids, mapping = submit_batches(
            client, todo, n_lines=args.lines, max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        if not batch_ids:
            print("nothing submittable", file=sys.stderr)
            return 1
        state_path.write_text(
            json.dumps({"batch_ids": batch_ids, "mapping": {k: list(v) for k, v in mapping.items()}},
                       indent=2),
            encoding="utf-8",
        )
        print(f"submitted {len(mapping)} requests in {len(batch_ids)} batch(es)")
        print(f"batch ids written to {state_path} — safe to interrupt and re-run\n")

    def on_status(done_batches, total_batches, counts):
        print(f"[{done_batches}/{total_batches} batches ended] "
              f"{counts['succeeded']} ok · {counts['processing']} processing · "
              f"{counts['errored']} errored · {(time.monotonic() - started) / 60:.0f} min elapsed",
              flush=True)

    poll_batches(client, batch_ids, on_status=on_status)

    gens, errors = collect_batches(client, batch_ids, mapping, n_lines=args.lines)
    for gen in gens:
        store.add(gen)
    for err in errors:
        print(f"    FAILED {err}", file=sys.stderr)
    if errors:
        print(f"\n{len(errors)} generations failed; re-run to retry only those "
              "(everything else is cached)", file=sys.stderr)

    #  Cleared only once the results are safely in the store, so an interrupted
    #  collect re-reads the same batches rather than resubmitting them at cost.
    state_path.unlink(missing_ok=True)
    print(f"\nspent ${store.total_cost():.2f} across {len(store.records)} generations "
          f"in {(time.monotonic() - started) / 60:.1f} min")
    return _analyse(run_dir, store, args)


def cmd_analyse(args) -> int:
    run_dir = Path(args.run_dir)
    store = GenerationStore(run_dir / "generations.jsonl")
    if not store.records:
        print(f"no generations found in {run_dir/'generations.jsonl'}", file=sys.stderr)
        return 1
    return _analyse(run_dir, store, args)


def _analyse(run_dir: Path, store: GenerationStore, args) -> int:
    #  Reuse the feature space the run was originally scored in, if there is
    #  one. Re-fitting is safe now that the reference is seeds + baselines, but
    #  reusing it makes re-analysis exactly reproducible even if the baseline
    #  set later grows.
    scaler = None
    drift_path = run_dir / "drift.json"
    if getattr(args, "refit", False) is False and drift_path.exists():
        try:
            prior = json.loads(drift_path.read_text(encoding="utf-8"))
            if "scaler" in prior:
                scaler = Scaler.from_dict(prior["scaler"])
        except (json.JSONDecodeError, KeyError, TypeError):
            scaler = None

    result = analyse_records(
        store.records, window=args.window, max_lines=args.lines, scaler=scaler
    )
    result.write_json(drift_path)
    report = write_report(
        run_dir / "report.html",
        curves=result.curves,
        n_generations=result.n_generations,
        total_cost_usd=result.total_cost_usd,
        notes=result.notes,
    )

    print(f"\n{'model':<20} {'mode':<20} {'seed':<15} {'reclaimed':>10} {'half-life':>10}")
    for c in sorted(result.curves, key=lambda c: (c.seed, c.mode, c.model)):
        recl = c.reclamation_line or f">{len(c.gravity)}"
        half = c.half_reclamation_line or "—"
        print(f"{c.model:<20} {c.mode:<20} {c.seed:<15} {str(recl):>10} {str(half):>10}")

    #  The controls are the run's pass/fail. They are written in the register the
    #  models default to, so their curves should sit near zero; if they do not,
    #  the metric is measuring something other than style reclamation and no
    #  other row on this table means anything.
    ctrl = [c for c in result.curves if c.seed.startswith("control")]
    if ctrl:
        openings = [c.opening_gravity for c in ctrl if c.opening_gravity is not None]
        if openings:
            worst = max(abs(o) for o in openings)
            verdict = "PASS" if worst < 0.35 else "SUSPECT — inspect before trusting anything above"
            print(f"\ncontrols: {len(ctrl)} curves, largest |opening gravity| = "
                  f"{worst:.3f}   {verdict}")

    if result.contrasts:
        print(f"\n{'model':<20} {'channel':<12} {'seed':<15} "
              f"{'Δgravity':>10} {'95% CI':>18} {'P(helps)':>9} {'Δreclaim':>9}")
        for k in sorted(result.contrasts, key=lambda k: (k.seed, k.base_mode, k.model)):
            ci = (f"[{k.d_gravity_ci[0]:+.3f}, {k.d_gravity_ci[1]:+.3f}]"
                  if k.d_gravity_ci else "—")
            p = f"{k.p_sustain_helps:.2f}" if k.p_sustain_helps is not None else "—"
            dr = f"{k.d_reclamation:+.0f}" if k.d_reclamation is not None else "—"
            print(f"{k.model:<20} {k.base_mode:<12} {k.seed:<15} "
                  f"{k.d_gravity:>+10.3f} {ci:>18} {p:>9} {dr:>9}")
        print("  Δgravity < 0 means asking for the style held the poem nearer the seed.")
        print("  P(helps) is the bootstrap fraction of resamples in that direction.")

    print(f"\nwrote {run_dir/'drift.json'} and {report}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="stylegravity", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("seeds", help="list seed openings").set_defaults(func=cmd_seeds)
    sub.add_parser("models", help="list models and prefill support").set_defaults(func=cmd_models)
    sub.add_parser("presets", help="describe the experiment presets").set_defaults(func=cmd_presets)

    p_est = sub.add_parser("estimate", help="cost and wall clock, no API calls")
    _add_shared(p_est)
    p_est.set_defaults(func=cmd_estimate)

    p_run = sub.add_parser("run", help="generate, then analyse")
    _add_shared(p_run)
    p_run.add_argument("--run-dir", default="runs/latest")
    p_run.add_argument("--window", type=int, default=2,
                       help="consecutive lines past zero required to call it reclaimed")
    p_run.set_defaults(func=cmd_run)

    p_an = sub.add_parser("analyse", help="re-score a cached run, no API calls")
    _add_shared(p_an)
    p_an.add_argument("--run-dir", default="runs/latest")
    p_an.add_argument("--window", type=int, default=2)
    p_an.add_argument("--refit", action="store_true",
                      help="re-fit the feature scaler instead of reusing the one stored in "
                           "drift.json (use after adding models, whose baselines widen the "
                           "reference set)")
    p_an.set_defaults(func=cmd_analyse)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

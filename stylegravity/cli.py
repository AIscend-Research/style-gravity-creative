"""Command line entry point.

    python -m stylegravity estimate --preset full     # cost + wall clock, no API calls
    python -m stylegravity run --preset full --run-dir runs/full
    python -m stylegravity analyse --run-dir runs/full    # re-score, no API calls
    python -m stylegravity seeds | models | presets
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .analysis import analyse as analyse_records
from .generate import (
    MODE_BASELINE, MODE_INSTRUCTED, MODE_PREFILL, SUSTAIN,
    GenerationStore, estimate_jobs, generate, make_client, split_mode,
)
from .models import DEFAULT_MODELS, REGISTRY, resolve as resolve_model
from .report import write_report
from .seeds import DEFAULT_SEEDS, SEEDS, resolve as resolve_seed

ALL_SEEDS = [s.id for s in SEEDS]
CORE_SEEDS = DEFAULT_SEEDS                       # control + skaldic + monosyllabic + legalese
PREFILL_MODELS = [m for m, s in REGISTRY.items() if s.prefill and m in DEFAULT_MODELS]
CURRENT_MODELS = ["claude-opus-5", "claude-sonnet-5"]

WITH_SUSTAIN = f"+{SUSTAIN}"


@dataclass(frozen=True)
class Block:
    """One factorial slab of the design: models × seeds × modes × samples."""
    models: list[str]
    seeds: list[str]
    modes: list[str]
    note: str


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
                PREFILL_MODELS, ALL_SEEDS, [MODE_PREFILL],
                "headline: every seed, every prefill-capable model",
            ),
            Block(
                PREFILL_MODELS, CORE_SEEDS,
                [MODE_PREFILL + WITH_SUSTAIN, MODE_INSTRUCTED, MODE_INSTRUCTED + WITH_SUSTAIN],
                "condition contrast on one model — de-confounds channel from model, "
                "and measures what asking for style can recover that prefill gave for free",
            ),
            Block(
                CURRENT_MODELS, CORE_SEEDS,
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


def _plan(args) -> list[tuple[str, str, object, int]]:
    """(model, mode, seed, sample) jobs.

    Baselines come first so an interrupted run still leaves every model with a
    usable house-style pole — without one, none of that model's cells can be
    scored at all.
    """
    blocks, default_baselines = PRESETS[args.preset]
    samples = args.samples if args.samples is not None else (10 if args.preset == "full" else 5)
    n_base = args.baseline_samples if args.baseline_samples is not None else default_baselines

    blocks = [
        Block(
            [m for m in b.models if args.models is None or m in args.models],
            [s for s in b.seeds if args.seeds is None or s in args.seeds],
            b.modes, b.note,
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
                    for i in range(samples):
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
    upper, typical, seconds = estimate_jobs(
        jobs, max_tokens=args.max_tokens, concurrency=args.concurrency
    )
    by_mode: dict[str, int] = {}
    for _m, mode, _s, _i in jobs:
        by_mode[mode] = by_mode.get(mode, 0) + 1
    print(f"preset: {args.preset}   {len(jobs)} API calls")
    for mode, n in sorted(by_mode.items()):
        print(f"  {mode:<22} {n:>5}")
    print(f"\ncost   upper bound ${upper:.2f}   typical ${typical:.2f}")
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
    counter = {"done": 0, "failed": 0}
    lock = threading.Lock()
    started = time.monotonic()

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


def cmd_analyse(args) -> int:
    run_dir = Path(args.run_dir)
    store = GenerationStore(run_dir / "generations.jsonl")
    if not store.records:
        print(f"no generations found in {run_dir/'generations.jsonl'}", file=sys.stderr)
        return 1
    return _analyse(run_dir, store, args)


def _analyse(run_dir: Path, store: GenerationStore, args) -> int:
    result = analyse_records(store.records, window=args.window, max_lines=args.lines)
    result.write_json(run_dir / "drift.json")
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
    p_an.set_defaults(func=cmd_analyse)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

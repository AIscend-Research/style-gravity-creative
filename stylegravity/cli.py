"""Command line entry point.

    python -m stylegravity estimate            # what a run would cost
    python -m stylegravity run --run-dir runs/pilot
    python -m stylegravity analyse --run-dir runs/pilot   # re-score, no API calls
    python -m stylegravity seeds                # list available seeds
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analysis import analyse as analyse_records
from .generate import (
    MODE_BASELINE, MODE_INSTRUCTED, MODE_PREFILL,
    GenerationStore, estimate_cost, generate, make_client,
)
from .models import DEFAULT_MODELS, REGISTRY, resolve as resolve_model
from .report import write_report
from .seeds import DEFAULT_SEEDS, SEEDS, resolve as resolve_seed


def _add_shared(p: argparse.ArgumentParser) -> None:
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   help=f"model ids (default: {' '.join(DEFAULT_MODELS)})")
    p.add_argument("--seeds", nargs="+", default=DEFAULT_SEEDS,
                   help=f"seed ids (default: {' '.join(DEFAULT_SEEDS)})")
    p.add_argument("--samples", type=int, default=5,
                   help="continuations per (model, seed) cell")
    p.add_argument("--baseline-samples", type=int, default=8,
                   help="unprompted poems per model, used as the house-style pole")
    p.add_argument("--lines", type=int, default=24,
                   help="continuation length requested, in lines")
    p.add_argument("--max-tokens", type=int, default=1200)
    p.add_argument("--temperature", type=float, default=1.0)


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


def cmd_estimate(args) -> int:
    total = estimate_cost(
        model_ids=args.models,
        n_seeds=len(args.seeds),
        samples=args.samples,
        baseline_samples=args.baseline_samples,
        max_tokens=args.max_tokens,
    )
    calls = len(args.models) * (len(args.seeds) * args.samples + args.baseline_samples)
    print(f"{calls} API calls across {len(args.models)} models")
    print(f"upper bound: ${total:.2f}  (assumes every generation runs to --max-tokens)")
    print(f"typical:     ${total * 0.55:.2f}–${total * 0.8:.2f}")
    return 0


def _plan(args) -> list[tuple[str, str, object, int]]:
    """(model, mode, seed, sample) tuples, baselines first so an interrupted run
    still leaves each model with a usable house-style pole."""
    jobs = []
    for mid in args.models:
        for i in range(args.baseline_samples):
            jobs.append((mid, MODE_BASELINE, None, i))
    for mid in args.models:
        spec = resolve_model(mid)
        mode = MODE_PREFILL if spec.prefill else MODE_INSTRUCTED
        for sid in args.seeds:
            seed = resolve_seed(sid)
            for i in range(args.samples):
                jobs.append((mid, mode, seed, i))
    return jobs


def cmd_run(args) -> int:
    run_dir = Path(args.run_dir)
    store = GenerationStore(run_dir / "generations.jsonl")
    jobs = _plan(args)

    fallbacks = [m for m in args.models if not resolve_model(m).prefill]
    if fallbacks:
        print(
            "NOTE: " + ", ".join(resolve_model(m).label for m in fallbacks) +
            " do not support assistant prefill (removed on the Claude 4.6+ line).\n"
            "      They will run in `instructed` mode instead, which measures a related "
            "but different\n      thing and is reported separately — never pooled with "
            "prefill results.\n",
            file=sys.stderr,
        )

    client = make_client()
    todo = [j for j in jobs if not store.has(j[0], j[1], j[2].id if j[2] else "-", j[3])]
    print(f"{len(jobs)} generations planned, {len(jobs) - len(todo)} already cached, "
          f"{len(todo)} to run")

    for n, (mid, mode, seed, sample) in enumerate(todo, 1):
        tag = f"{mid} {mode} {seed.id if seed else '-'} #{sample}"
        print(f"[{n}/{len(todo)}] {tag}", flush=True)
        try:
            gen = generate(
                client, model_id=mid, mode=mode, seed=seed, sample=sample,
                n_lines=args.lines, max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        except Exception as exc:  # noqa: BLE001 — one bad cell must not kill the run
            print(f"    FAILED: {exc}", file=sys.stderr)
            continue
        if gen.stop_reason == "max_tokens":
            print("    (hit max_tokens — final line may be truncated and is dropped)",
                  file=sys.stderr)
            gen.text = "\n".join(gen.text.splitlines()[:-1])
        store.add(gen)

    print(f"\nspent ${store.total_cost():.2f} across {len(store.records)} generations")
    return _analyse(run_dir, store, args)


def cmd_analyse(args) -> int:
    run_dir = Path(args.run_dir)
    store = GenerationStore(run_dir / "generations.jsonl")
    if not store.records:
        print(f"no generations found in {run_dir/'generations.jsonl'}", file=sys.stderr)
        return 1
    return _analyse(run_dir, store, args)


def _analyse(run_dir: Path, store: GenerationStore, args) -> int:
    result = analyse_records(
        store.records, window=args.window, max_lines=args.lines
    )
    result.write_json(run_dir / "drift.json")
    report = write_report(
        run_dir / "report.html",
        curves=result.curves,
        n_generations=result.n_generations,
        total_cost_usd=result.total_cost_usd,
        notes=result.notes,
    )

    print(f"\n{'model':<20} {'mode':<11} {'seed':<15} {'reclaimed':>10} {'half-life':>10}")
    for c in sorted(result.curves, key=lambda c: (c.seed, c.mode, c.model)):
        recl = c.reclamation_line or f">{len(c.gravity)}"
        half = c.half_reclamation_line or "—"
        print(f"{c.model:<20} {c.mode:<11} {c.seed:<15} {str(recl):>10} {str(half):>10}")
    print(f"\nwrote {run_dir/'drift.json'} and {report}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="stylegravity", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("seeds", help="list seed openings").set_defaults(func=cmd_seeds)
    sub.add_parser("models", help="list models and prefill support").set_defaults(func=cmd_models)

    p_est = sub.add_parser("estimate", help="cost estimate, no API calls")
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

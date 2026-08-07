"""Post-hoc corrections to a style-gravity run.

Reads ``drift.json`` from a completed run and re-reports the sustain contrast
with three corrections the raw table does not apply:

1.  **Control floor.** The sustain instruction moves gravity even on a control
    seed written in the model's own house style, where by construction there is
    no seeded style to hold. That movement is the floor, not the effect. Each
    contrast is reported as excess over the mean control d_gravity for its own
    (model, channel) cell, so ``P(helps) = 1.00`` on a control is no longer
    mistaken for evidence.

2.  **Seed-pole exclusions.** A seed whose own lines fail the leave-one-out
    seed-likeness check is not off-distribution enough to be a seed. Its drift
    curve is measured against a pole that is barely distinguishable from
    baseline. Seeds failing on a majority of curves are dropped; those failing
    on a minority are reported with a flag.

3.  **Degenerate contrasts.** Cells where either arm has fewer than three
    usable generations get no bootstrap interval from the analysis code, but
    still print a point estimate in the summary table with nothing but a missing
    CI to mark them. They are excluded here by the same rule.

Sign convention follows ``drift.py``: gravity of -1 sits on the seeded style,
+1 on the model's house style. So a *negative* d_gravity means sustaining held
the poem nearer the seed, and a more negative excess is a larger effect.

Usage:
    python report_corrected.py runs/full
    python report_corrected.py runs/full --min-arm 3 --json out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

# A seed-pole warning on this fraction of a seed's curves or more means the seed
# itself is the problem, not one model's response to it.
DROP_ABOVE = 0.5

SEED_POLE_WARNING = "do not score as seed-like"


def load(run_dir: Path) -> dict:
    path = run_dir / "drift.json"
    if not path.exists():
        sys.exit(f"no drift.json in {run_dir} — run `stylegravity run` or `analyse` first")
    return json.loads(path.read_text())


def reconcile_counts(run_dir: Path, drift: dict) -> tuple[int, int]:
    """Compare generations on disk against generations the analysis used.

    A gap here means records are being dropped between generation and scoring.
    Whether that is deliberate filtering or silent loss is not something
    drift.json can answer, so this only reports the two numbers.
    """
    n_analysed = drift.get("n_generations", 0)
    jsonl = run_dir / "generations.jsonl"
    if not jsonl.exists():
        return n_analysed, -1
    n_disk = sum(1 for line in jsonl.open() if line.strip())
    return n_analysed, n_disk


def seed_pole_status(drift: dict) -> dict[str, tuple[int, int]]:
    """For each seed: (curves warned, curves total)."""
    status: dict[str, list[int]] = {}
    for c in drift["curves"]:
        warned = any(SEED_POLE_WARNING in w for w in c["warnings"])
        row = status.setdefault(c["seed"], [0, 0])
        row[0] += int(warned)
        row[1] += 1
    return {k: (v[0], v[1]) for k, v in status.items()}


def control_floor(drift: dict) -> dict[tuple[str, str], list[float]]:
    """Mean control d_gravity per (model, channel)."""
    floor: dict[tuple[str, str], list[float]] = {}
    for k in drift["contrasts"]:
        if k["seed"].startswith("control"):
            floor.setdefault((k["model"], k["base_mode"]), []).append(k["d_gravity"])
    return floor


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--min-arm", type=int, default=3,
                    help="drop a contrast if either arm has fewer usable generations (default 3)")
    ap.add_argument("--json", type=Path, default=None,
                    help="also write the corrected contrasts here")
    args = ap.parse_args()

    drift = load(args.run_dir)

    # ---- generation count reconciliation --------------------------------- #
    n_analysed, n_disk = reconcile_counts(args.run_dir, drift)
    print(f"generations scored by the analysis : {n_analysed}")
    if n_disk < 0:
        print("generations on disk                : (generations.jsonl not found)")
    else:
        print(f"generations on disk                : {n_disk}")
        if n_disk != n_analysed:
            print(f"  ** {n_disk - n_analysed} record(s) on disk did not reach the analysis. "
                  "Deliberate filtering or silent loss? Not answerable from drift.json.")
    print()

    # ---- seed-pole exclusions -------------------------------------------- #
    status = seed_pole_status(drift)
    dropped, flagged = set(), set()
    print("seed-pole check (leave-one-out seed-likeness)")
    for seed in sorted(status):
        warned, total = status[seed]
        if seed.startswith("control"):
            note = "control — warning expected, house style by construction"
        elif warned / total > DROP_ABOVE:
            dropped.add(seed)
            note = "DROP — not off-distribution enough to serve as a seed"
        elif warned:
            flagged.add(seed)
            note = "flag — passes on most curves, report with a caveat"
        else:
            note = "clean"
        print(f"  {seed:<16}{warned:>3}/{total:<4} warned   {note}")
    print()

    # ---- corrected contrasts --------------------------------------------- #
    floor = control_floor(drift)
    rows = []
    excluded = []
    for k in drift["contrasts"]:
        if k["seed"].startswith("control"):
            continue
        if k["seed"] in dropped:
            excluded.append((k, "seed failed the seed-pole check"))
            continue
        if min(k["n_plain"], k["n_sustain"]) < args.min_arm:
            excluded.append((k, f"arm below {args.min_arm} usable generations"))
            continue
        cell = floor.get((k["model"], k["base_mode"]))
        if not cell:
            excluded.append((k, "no control floor for this cell"))
            continue
        f = statistics.mean(cell)
        rows.append({
            "model": k["model"],
            "channel": k["base_mode"],
            "seed": k["seed"],
            "raw_d_gravity": k["d_gravity"],
            "control_floor": f,
            "excess": k["d_gravity"] - f,
            "n_plain": k["n_plain"],
            "n_sustain": k["n_sustain"],
            "flagged_seed": k["seed"] in flagged,
        })

    print("control floor — mean d_gravity across the three control seeds")
    for (model, ch), vals in sorted(floor.items()):
        print(f"  {model:<18}{ch:<12}{statistics.mean(vals):+.3f}  (n={len(vals)})")
    print()

    print("sustain effect as excess over the control floor")
    print("  negative excess = sustaining held the poem nearer the seed")
    print(f"  {'model':<18}{'channel':<12}{'seed':<15}{'raw':>8}{'floor':>8}{'excess':>9}   n")
    for r in sorted(rows, key=lambda r: (r["seed"], r["channel"], r["model"])):
        mark = " *" if r["flagged_seed"] else ""
        print(f"  {r['model']:<18}{r['channel']:<12}{r['seed']:<15}"
              f"{r['raw_d_gravity']:>+8.3f}{r['control_floor']:>+8.3f}{r['excess']:>+9.3f}"
              f"   {r['n_plain']}/{r['n_sustain']}{mark}")
    if flagged:
        print(f"  * seed flagged by the seed-pole check: {', '.join(sorted(flagged))}")
    print()

    # ---- prefill vs instructed ------------------------------------------- #
    # Only models that support both channels can be compared this way; the 4.6+
    # line runs instructed-only, so pooling across models here would confound
    # channel with model.
    paired: dict[tuple[str, str], dict[str, float]] = {}
    for r in rows:
        paired.setdefault((r["model"], r["seed"]), {})[r["channel"]] = r["excess"]
    both = {k: v for k, v in paired.items() if {"prefill", "instructed"} <= set(v)}
    if both:
        print("same model, same seed, prefill vs instructed (excess over floor)")
        print(f"  {'model':<18}{'seed':<15}{'instructed':>11}{'prefill':>10}{'gap':>9}")
        for (model, seed), v in sorted(both.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            gap = v["prefill"] - v["instructed"]
            print(f"  {model:<18}{seed:<15}{v['instructed']:>+11.3f}{v['prefill']:>+10.3f}{gap:>+9.3f}")
        gaps = [v["prefill"] - v["instructed"] for v in both.values()]
        n_neg = sum(1 for g in gaps if g < 0)
        print(f"  prefill shows the larger effect in {n_neg}/{len(gaps)} pairs "
              f"(median gap {statistics.median(gaps):+.3f})")
        print()

    if excluded:
        print("excluded contrasts")
        for k, why in excluded:
            print(f"  {k['model']:<18}{k['base_mode']:<12}{k['seed']:<15}"
                  f"d={k['d_gravity']:+.3f}  n={k['n_plain']}/{k['n_sustain']}   {why}")
        print()

    if args.json:
        args.json.write_text(json.dumps(
            {"rows": rows,
             "dropped_seeds": sorted(dropped),
             "flagged_seeds": sorted(flagged),
             "n_analysed": n_analysed,
             "n_on_disk": n_disk},
            indent=2))
        print(f"wrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

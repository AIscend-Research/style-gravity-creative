"""Turn a store of generations into drift curves."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .drift import DriftCurve, Scaler, build_curve, fit_run_scaler
from .features import poem_features, split_lines
from .generate import MODE_BASELINE, Generation
from .seeds import Seed, resolve as resolve_seed


@dataclass
class RunAnalysis:
    curves: list[DriftCurve]
    scaler: Scaler
    n_generations: int
    total_cost_usd: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_generations": self.n_generations,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "notes": self.notes,
            "curves": [c.to_dict() for c in self.curves],
        }

    def write_json(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )


def _dedupe_seed_echo(cont_lines: list[str], seed: Seed) -> list[str]:
    """Drop a re-printed copy of the seed from the head of a continuation.

    In ``instructed`` mode models often restate the opening before continuing,
    despite being told not to. Those lines are the seed, not the model's
    response to it, and leaving them in would manufacture a stretch of perfect
    fidelity at the start of every curve — inflating exactly the number the
    experiment is trying to measure.
    """
    seed_norm = [ln.strip().lower() for ln in seed.lines]
    if not seed_norm:
        return cont_lines
    head = [ln.strip().lower() for ln in cont_lines[: len(seed_norm)]]
    overlap = sum(1 for a, b in zip(head, seed_norm) if a == b)
    if overlap >= max(2, len(seed_norm) // 2):
        return cont_lines[len(seed_norm):]
    return cont_lines


def analyse(
    records: list[Generation],
    *,
    window: int = 2,
    min_support: int = 2,
    max_lines: int | None = None,
) -> RunAnalysis:
    usable = [r for r in records if r.error is None and r.text.strip()]
    notes: list[str] = []

    # ---- group ---------------------------------------------------------- #
    baselines: dict[tuple[str, str], list[list[str]]] = {}
    continuations: dict[tuple[str, str, str], dict[int, list[str]]] = {}

    for rec in usable:
        lines = split_lines(rec.text)
        if max_lines is not None:
            lines = lines[:max_lines]
        if rec.mode == MODE_BASELINE:
            baselines.setdefault((rec.model, MODE_BASELINE), []).append(lines)
        else:
            seed = resolve_seed(rec.seed_id)
            lines = _dedupe_seed_echo(lines, seed)
            continuations.setdefault(
                (rec.model, rec.mode, rec.seed_id), {}
            )[rec.sample] = lines

    # ---- one feature space for the whole run ---------------------------- #
    pool: list[list[float]] = []
    for group in baselines.values():
        for lines in group:
            pool.extend(poem_features(lines))
    for by_sample in continuations.values():
        for lines in by_sample.values():
            pool.extend(poem_features(lines))
    seed_ids = {k[2] for k in continuations}
    for sid in seed_ids:
        pool.extend(poem_features(resolve_seed(sid).lines))

    if not pool:
        raise ValueError("no usable generations to analyse")
    scaler = fit_run_scaler(pool)

    # ---- curves ---------------------------------------------------------- #
    curves: list[DriftCurve] = []
    for (model, mode, seed_id), by_sample in sorted(continuations.items()):
        seed = resolve_seed(seed_id)
        base_lines = baselines.get((model, MODE_BASELINE))
        if not base_lines:
            notes.append(
                f"skipped {model}/{mode}/{seed_id}: no baseline generations for {model}, "
                "so there is no house-style pole to measure drift toward"
            )
            continue
        base_vectors = [v for lines in base_lines for v in poem_features(lines)]
        samples = [
            poem_features(by_sample[k]) for k in sorted(by_sample) if by_sample[k]
        ]
        if not samples:
            continue
        curves.append(
            build_curve(
                model=model, mode=mode, seed_id=seed_id, scaler=scaler,
                seed_vectors=poem_features(seed.lines),
                baseline_vectors=base_vectors,
                continuation_samples=samples,
                window=window,
                min_support=min(min_support, len(samples)),
            )
        )

    return RunAnalysis(
        curves=curves,
        scaler=scaler,
        n_generations=len(usable),
        total_cost_usd=sum(r.cost_usd for r in records),
        notes=notes,
    )

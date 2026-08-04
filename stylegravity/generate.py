"""Generation: prefilled continuations and unprompted baselines.

Three call shapes, all against ``POST /v1/messages``:

``prefill``     user asks for a poem; the seed is supplied as a trailing
                *assistant* turn, so the model experiences it as its own
                in-progress output and simply keeps writing. This is the
                intervention the experiment is about.

``instructed``  fallback for models where prefill was removed. The seed goes in
                the user turn with a "continue this" instruction. The model sees
                the seed as someone else's text. Measurable, but a different
                intervention — never pooled with prefill results.

``baseline``    the same poem request with no seed at all. This is the model's
                unprompted house style, and it is the second pole of the metric.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import anthropic

from .models import ModelSpec, resolve as resolve_model
from .seeds import Seed, resolve as resolve_seed

MODE_PREFILL = "prefill"
MODE_INSTRUCTED = "instructed"
MODE_BASELINE = "baseline"

#  The `sustain` condition, appended to a base mode as `prefill+sustain`.
#  Encoding the condition in the mode string keeps it flowing through the cache
#  key, the grouping, and the report without a structural change — and it means
#  a sustained cell is never silently pooled with an unsustained one.
SUSTAIN = "sustain"


def compose_mode(base: str, sustain: bool) -> str:
    return f"{base}+{SUSTAIN}" if sustain else base


def split_mode(mode: str) -> tuple[str, bool]:
    base, _, suffix = mode.partition("+")
    return base, suffix == SUSTAIN


#  Kept as bare as possible. Every word of instruction here is a thumb on the
#  scale: the baseline is supposed to be the model's *unprompted* style, so the
#  prompt must not specify tone, subject, form, or register.
POEM_PROMPT = "Write a poem, at least {n_lines} lines long."

INSTRUCTED_PROMPT = (
    "Here is the opening of a poem:\n\n{seed}\n\n"
    "Continue it, at least {n_lines} more lines. Output only the continuation — "
    "do not repeat the lines above, and do not add a title or any commentary."
)

#  The `sustain` condition: the author asks, in words, for what prefill used to
#  give them structurally. The contrast between a cell and its `+sustain` twin is
#  the measurement of what explicit instruction can and cannot recover — asserted
#  agency versus the agency the affordance used to confer for free.
#
#  Deliberately says *how* to sustain (diction, syntax, form, register) without
#  naming any specific style. Naming the style would leak the answer, and the
#  seeds span registers a single description could not cover.
SUSTAIN_SYSTEM = (
    "You are continuing a poem whose opening has already been written by someone else. "
    "Sustain that opening's style for the whole of your continuation: keep its diction, "
    "its syntax, its line shape, its punctuation habits, its level of formality, and any "
    "formal constraint it appears to be observing. Do not drift toward a more familiar or "
    "more contemporary voice as you go, and do not smooth the style out. If the opening is "
    "strange, stay strange in the same way."
)


@dataclass
class Generation:
    model: str
    mode: str
    seed_id: str
    sample: int
    text: str
    stop_reason: str | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    prompt: str
    prefill: str | None
    #  Defaulted so generations.jsonl written before the sustain condition
    #  existed still load cleanly — an old run must not have to be re-bought.
    system: str | None = None
    error: str | None = None
    #  Provenance, not a parameter: batch and streaming produce the same request
    #  but not the same bill, and a reader auditing `cost_usd` needs to know
    #  which rate applied.
    batch: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class GenerationError(RuntimeError):
    pass


#  The Message Batches API bills at half the standard rate. The discount is the
#  single largest cost lever in this experiment and costs nothing scientifically
#  — same models, same params, same sampling — so it is worth the extra code
#  path. It is applied here rather than at the call site so a cached record's
#  `cost_usd` is always the amount actually billed for that record.
BATCH_DISCOUNT = 0.5

#  Output tokens per line of verse. Derived from the 380-tokens-per-24-line
#  figure the throughput estimate has always used; kept as a named constant now
#  that the cost estimate depends on it too.
TOKENS_PER_LINE = 380 / 24


def _cost(spec: ModelSpec, usage, *, batch: bool = False) -> float:
    rate = BATCH_DISCOUNT if batch else 1.0
    return rate * (
        usage.input_tokens * spec.input_price / 1_000_000
        + usage.output_tokens * spec.output_price / 1_000_000
    )


def build_call(
    *, model_id: str, mode: str, seed: Seed | None, n_lines: int
) -> tuple[ModelSpec, str, str | None, str | None]:
    """``(spec, prompt, prefill, system)`` for one cell.

    Shared by the streaming and batch paths so the two cannot drift apart: a
    batch run and a live run of the same cell must be the same request, or the
    cache mixes two different experiments under one key.
    """
    spec = resolve_model(model_id)
    base, sustain = split_mode(mode)
    system = SUSTAIN_SYSTEM if sustain else None

    if base == MODE_PREFILL:
        if seed is None:
            raise GenerationError("prefill mode requires a seed")
        if not spec.prefill:
            raise GenerationError(
                f"{spec.label} does not support assistant prefill "
                "(removed on the Claude 4.6+ line); use mode='instructed'"
            )
        return spec, POEM_PROMPT.format(n_lines=n_lines + len(seed.lines)), seed.text, system
    if base == MODE_INSTRUCTED:
        if seed is None:
            raise GenerationError("instructed mode requires a seed")
        return spec, INSTRUCTED_PROMPT.format(seed=seed.text.strip(), n_lines=n_lines), None, system
    if base == MODE_BASELINE:
        if sustain:
            # There is no opening to sustain, so the condition is undefined —
            # and a system-prompted baseline would no longer be the *unprompted*
            # house style the metric's second pole depends on.
            raise GenerationError("the sustain condition does not apply to baselines")
        return spec, POEM_PROMPT.format(n_lines=n_lines), None, system
    raise GenerationError(f"unknown mode {mode!r}")


def build_params(
    spec: ModelSpec,
    *,
    prompt: str,
    prefill: str | None,
    system: str | None,
    max_tokens: int,
    temperature: float | None,
) -> dict:
    messages: list[dict] = [{"role": "user", "content": prompt}]
    if prefill is not None:
        # The API rejects a prefill with trailing whitespace, and a trailing
        # newline would also hand the model a free line break it did not choose.
        messages.append({"role": "assistant", "content": prefill.rstrip()})
    params: dict = {"model": spec.id, "max_tokens": max_tokens, "messages": messages}
    if system:
        params["system"] = system
    if temperature is not None and spec.sampling_params:
        params["temperature"] = temperature
    return params


def _call(
    client: anthropic.Anthropic,
    spec: ModelSpec,
    *,
    prompt: str,
    prefill: str | None,
    system: str | None,
    max_tokens: int,
    temperature: float | None,
) -> tuple[str, object]:
    params = build_params(
        spec, prompt=prompt, prefill=prefill, system=system,
        max_tokens=max_tokens, temperature=temperature,
    )
    with client.messages.stream(**params) as stream:
        message = stream.get_final_message()

    text = "".join(b.text for b in message.content if b.type == "text")
    return text, message


def generate(
    client: anthropic.Anthropic,
    *,
    model_id: str,
    mode: str,
    seed: Seed | None,
    sample: int,
    n_lines: int,
    max_tokens: int,
    temperature: float | None,
    max_retries: int = 3,
) -> Generation:
    spec, prompt, prefill, system = build_call(
        model_id=model_id, mode=mode, seed=seed, n_lines=n_lines
    )

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            text, message = _call(
                client, spec,
                prompt=prompt, prefill=prefill, system=system,
                max_tokens=max_tokens, temperature=temperature,
            )
        except anthropic.APIStatusError as exc:
            last_err = exc
            if exc.status_code in (429,) or exc.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            raise GenerationError(f"{spec.id} {mode}: {exc}") from exc
        except anthropic.APIConnectionError as exc:
            last_err = exc
            time.sleep(2 ** attempt)
            continue

        return Generation(
            model=spec.id,
            mode=mode,
            seed_id=seed.id if seed else "-",
            sample=sample,
            text=text,
            stop_reason=message.stop_reason,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            cost_usd=_cost(spec, message.usage),
            prompt=prompt,
            prefill=prefill,
            system=system,
        )

    raise GenerationError(f"{spec.id} {mode}: exhausted retries ({last_err})")


# --------------------------------------------------------------------------- #
# on-disk cache
# --------------------------------------------------------------------------- #

class GenerationStore:
    """Append-only JSONL of every generation.

    Generations are the expensive part and the analysis is not: caching them
    means the whole metric can be re-derived, re-tuned, and re-argued about
    without spending another cent, and it leaves the raw poems auditable by
    anyone who doubts the numbers.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._records: list[Generation] = []
        self._keys: set[tuple[str, str, str, int]] = set()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = Generation(**json.loads(line))
                    self._records.append(rec)
                    if rec.error is None:
                        self._keys.add((rec.model, rec.mode, rec.seed_id, rec.sample))

    @property
    def records(self) -> list[Generation]:
        with self._lock:
            return list(self._records)

    def has(self, model: str, mode: str, seed_id: str, sample: int) -> bool:
        return (model, mode, seed_id, sample) in self._keys

    def add(self, gen: Generation) -> None:
        # Serialised so concurrent workers cannot interleave a half-written JSON
        # line — a torn line would make the whole cache unreadable on reload.
        with self._lock:
            self._records.append(gen)
            self._keys.add((gen.model, gen.mode, gen.seed_id, gen.sample))
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(gen.to_dict(), ensure_ascii=False) + "\n")
                fh.flush()

    def total_cost(self) -> float:
        with self._lock:
            return sum(r.cost_usd for r in self._records)


# --------------------------------------------------------------------------- #
# Message Batches: the same requests at half price
# --------------------------------------------------------------------------- #

#  Anthropic caps a batch at 100k requests / 256MB. Chunking well below that
#  keeps any single failure cheap to resubmit and keeps the poll loop's progress
#  reporting meaningful on a run of a few thousand cells.
BATCH_CHUNK = 500


def _custom_id(index: int) -> str:
    #  custom_id is capped at 64 chars and restricted in charset, and a model id
    #  plus mode plus seed plus sample would blow that. An index into the job
    #  list is used instead, with the mapping persisted next to the batch id so
    #  an interrupted poll can still attribute results.
    return f"cell{index:06d}"


def submit_batches(
    client: anthropic.Anthropic,
    jobs: list[tuple[str, str, object, int]],
    *,
    n_lines: int,
    max_tokens: int,
    temperature: float | None,
) -> tuple[list[str], dict[str, tuple[str, str, str, int]]]:
    """Create batches for ``jobs``. Returns ``(batch_ids, custom_id -> job key)``.

    Cells whose request cannot be built at all (prefill on a model that removed
    it) are dropped here rather than submitted and failed remotely — a batch
    error costs a round trip and tells you less than the exception does.
    """
    mapping: dict[str, tuple[str, str, str, int]] = {}
    requests: list[dict] = []
    for i, (model_id, mode, seed, sample) in enumerate(jobs):
        try:
            spec, prompt, prefill, system = build_call(
                model_id=model_id, mode=mode, seed=seed, n_lines=n_lines
            )
        except GenerationError:
            continue
        cid = _custom_id(i)
        mapping[cid] = (model_id, mode, seed.id if seed else "-", sample)
        requests.append({
            "custom_id": cid,
            "params": build_params(
                spec, prompt=prompt, prefill=prefill, system=system,
                max_tokens=max_tokens, temperature=temperature,
            ),
        })

    batch_ids: list[str] = []
    for start in range(0, len(requests), BATCH_CHUNK):
        chunk = requests[start:start + BATCH_CHUNK]
        batch_ids.append(client.messages.batches.create(requests=chunk).id)
    return batch_ids, mapping


def poll_batches(
    client: anthropic.Anthropic,
    batch_ids: list[str],
    *,
    interval: float = 30.0,
    on_status=None,
) -> None:
    """Block until every batch has ended. ``on_status(done, total, counts)``."""
    pending = list(batch_ids)
    while pending:
        counts = {"succeeded": 0, "errored": 0, "processing": 0, "canceled": 0, "expired": 0}
        still: list[str] = []
        for bid in pending:
            b = client.messages.batches.retrieve(bid)
            rc = b.request_counts
            counts["succeeded"] += rc.succeeded
            counts["errored"] += rc.errored
            counts["processing"] += rc.processing
            counts["canceled"] += rc.canceled
            counts["expired"] += rc.expired
            if b.processing_status != "ended":
                still.append(bid)
        if on_status:
            on_status(len(batch_ids) - len(still), len(batch_ids), counts)
        pending = still
        if pending:
            time.sleep(interval)


def collect_batches(
    client: anthropic.Anthropic,
    batch_ids: list[str],
    mapping: dict[str, tuple[str, str, str, int]],
    *,
    n_lines: int,
) -> tuple[list[Generation], list[str]]:
    """Fetch results and turn them into ``Generation`` records.

    Returns ``(generations, errors)``. A cell that errored, expired, or was
    cancelled is reported rather than recorded: writing it to the store would
    mark the key as satisfied and stop the retry-on-rerun logic from ever
    picking it up again.
    """
    out: list[Generation] = []
    errors: list[str] = []
    for bid in batch_ids:
        for entry in client.messages.batches.results(bid):
            key = mapping.get(entry.custom_id)
            if key is None:
                errors.append(f"unknown custom_id {entry.custom_id} in batch {bid}")
                continue
            model_id, mode, seed_id, sample = key
            if entry.result.type != "succeeded":
                detail = getattr(entry.result, "error", entry.result.type)
                errors.append(f"{model_id} {mode} {seed_id} #{sample}: {detail}")
                continue
            message = entry.result.message
            #  Rebuilt rather than carried through the batch: every record in
            #  generations.jsonl stores the exact prompt that produced it, and a
            #  batch record that omitted them would be the only unauditable rows
            #  in the file. `build_call` is deterministic, so this reproduces
            #  what was submitted.
            seed = resolve_seed(seed_id) if seed_id != "-" else None
            spec, prompt, prefill, system = build_call(
                model_id=model_id, mode=mode, seed=seed, n_lines=n_lines
            )
            text = "".join(b.text for b in message.content if b.type == "text")
            if message.stop_reason == "max_tokens":
                # Same truncation guard as the streaming path: the final line was
                # cut mid-phrase, and scoring it would read as a style break the
                # model never made.
                text = "\n".join(text.splitlines()[:-1])
            out.append(Generation(
                model=model_id, mode=mode, seed_id=seed_id, sample=sample,
                text=text, stop_reason=message.stop_reason,
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
                cost_usd=_cost(spec, message.usage, batch=True),
                prompt=prompt, prefill=prefill, system=system, batch=True,
            ))
    return out, errors


def estimate_jobs(
    jobs: list[tuple[str, str, object, int]],
    *,
    max_tokens: int,
    concurrency: int = 1,
    approx_input_tokens: int = 260,
    batch: bool = False,
    n_lines: int = 24,
) -> tuple[float, float, float]:
    """``(upper_cost, typical_cost, wall_clock_seconds)`` for a concrete job list.

    The upper bound assumes every generation runs to ``max_tokens``; an estimate
    you can exceed is worse than useless when the budget claim is 'dollars'.

    The *typical* figure is the one to plan against, and it is derived rather
    than assumed. Billing is on tokens emitted, not on ``max_tokens``, and a
    poem stops when it stops — so typical spend scales with ``n_lines``, not
    with the headroom you left. An earlier version took a flat 65% of the
    ceiling, which overstated real spend by roughly 2x at the default
    ``--max-tokens 1200`` and made the headroom look expensive when it is free.

    The time figure is an estimate from published throughput, not a measurement,
    and it ignores rate-limit backoff. On a low API tier a 429 storm dominates
    everything here. It is meaningless under ``batch``, where the SLA is up to
    24 hours regardless of size.
    """
    rate = BATCH_DISCOUNT if batch else 1.0
    #  ~15.8 output tokens per line of verse, from the streaming-throughput
    #  figure this module already used (380 tokens for a 24-line poem).
    typical_out = min(TOKENS_PER_LINE * n_lines, max_tokens)
    upper = 0.0
    typical = 0.0
    serial_seconds = 0.0
    for model_id, _mode, _seed, _sample in jobs:
        spec = resolve_model(model_id)
        in_cost = approx_input_tokens * spec.input_price / 1_000_000
        upper += rate * (in_cost + max_tokens * spec.output_price / 1_000_000)
        typical += rate * (in_cost + typical_out * spec.output_price / 1_000_000)
        serial_seconds += typical_out / spec.tokens_per_second + 1.2
    return upper, typical, serial_seconds / max(concurrency, 1)


def make_client() -> anthropic.Anthropic:
    """Zero-arg client. Resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an
    ``ant auth login`` profile — do not pass a key in explicitly."""
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        # Not fatal: an `ant auth login` profile on disk also works.
        pass
    return anthropic.Anthropic()

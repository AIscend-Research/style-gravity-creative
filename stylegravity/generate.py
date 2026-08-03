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
from .seeds import Seed

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

    def to_dict(self) -> dict:
        return asdict(self)


class GenerationError(RuntimeError):
    pass


def _cost(spec: ModelSpec, usage) -> float:
    return (
        usage.input_tokens * spec.input_price / 1_000_000
        + usage.output_tokens * spec.output_price / 1_000_000
    )


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
    messages: list[dict] = [{"role": "user", "content": prompt}]
    if prefill is not None:
        # The API rejects a prefill with trailing whitespace, and a trailing
        # newline would also hand the model a free line break it did not choose.
        messages.append({"role": "assistant", "content": prefill.rstrip()})

    kwargs: dict = {
        "model": spec.id,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        kwargs["system"] = system
    if temperature is not None and spec.sampling_params:
        kwargs["temperature"] = temperature

    with client.messages.stream(**kwargs) as stream:
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
        prompt = POEM_PROMPT.format(n_lines=n_lines + len(seed.lines))
        prefill: str | None = seed.text
    elif base == MODE_INSTRUCTED:
        if seed is None:
            raise GenerationError("instructed mode requires a seed")
        prompt = INSTRUCTED_PROMPT.format(seed=seed.text.strip(), n_lines=n_lines)
        prefill = None
    elif base == MODE_BASELINE:
        if sustain:
            # There is no opening to sustain, so the condition is undefined —
            # and a system-prompted baseline would no longer be the *unprompted*
            # house style the metric's second pole depends on.
            raise GenerationError("the sustain condition does not apply to baselines")
        prompt = POEM_PROMPT.format(n_lines=n_lines)
        prefill = None
    else:
        raise GenerationError(f"unknown mode {mode!r}")

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


def estimate_jobs(
    jobs: list[tuple[str, str, object, int]],
    *,
    max_tokens: int,
    concurrency: int = 1,
    approx_input_tokens: int = 260,
) -> tuple[float, float, float]:
    """``(upper_cost, typical_cost, wall_clock_seconds)`` for a concrete job list.

    Cost upper bound assumes every generation runs to ``max_tokens``. Real spend
    lands well under it because poems stop on their own — but an estimate you can
    exceed is worse than useless when the budget claim is 'dollars'.

    The time figure is an estimate from published throughput, not a measurement,
    and it ignores rate-limit backoff. On a low API tier a 429 storm dominates
    everything here.
    """
    upper = 0.0
    serial_seconds = 0.0
    for model_id, _mode, _seed, _sample in jobs:
        spec = resolve_model(model_id)
        upper += (
            approx_input_tokens * spec.input_price / 1_000_000
            + max_tokens * spec.output_price / 1_000_000
        )
        #  ~380 output tokens for a 24-line poem, at each tier's rough
        #  streaming throughput, plus fixed request overhead.
        serial_seconds += 380 / spec.tokens_per_second + 1.2
    return upper, upper * 0.65, serial_seconds / max(concurrency, 1)


def make_client() -> anthropic.Anthropic:
    """Zero-arg client. Resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an
    ``ant auth login`` profile — do not pass a key in explicitly."""
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        # Not fatal: an `ant auth login` profile on disk also works.
        pass
    return anthropic.Anthropic()

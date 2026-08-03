"""Model registry.

The experiment depends on assistant prefill: we force the opening of the poem
and let the model continue it. Prefill is **removed** on the Claude 4.6+ line
(Opus 4.6/4.7/4.8, Opus 5, Sonnet 4.6, Sonnet 5, Fable 5) — a request whose
final message has ``role="assistant"`` returns a 400 there.

So the registry tracks prefill capability per model. Models without it can still
be measured, but only in ``instructed`` mode (the seed is handed to the model in
the user turn with a "continue this" instruction), which is a *different*
intervention: the model sees the seed as someone else's text rather than as its
own in-progress output. Results from the two modes are not comparable and are
kept separate throughout.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    prefill: bool
    #  $/1M tokens, used only for the cost estimator
    input_price: float
    output_price: float
    #  Models on the 4.7+ line reject temperature/top_p/top_k outright.
    sampling_params: bool = True
    #  Rough streaming output throughput, used only by the time estimator.
    tokens_per_second: float = 55.0


REGISTRY: dict[str, ModelSpec] = {
    # ---- prefill-capable: the models this experiment can actually run on ----
    "claude-haiku-4-5": ModelSpec(
        "claude-haiku-4-5", "Haiku 4.5", prefill=True,
        input_price=1.00, output_price=5.00, tokens_per_second=110.0,
    ),
    "claude-sonnet-4-5": ModelSpec(
        "claude-sonnet-4-5", "Sonnet 4.5", prefill=True,
        input_price=3.00, output_price=15.00, tokens_per_second=65.0,
    ),
    "claude-sonnet-4-0": ModelSpec(
        "claude-sonnet-4-0", "Sonnet 4", prefill=True,
        input_price=3.00, output_price=15.00, tokens_per_second=65.0,
    ),
    "claude-opus-4-5": ModelSpec(
        "claude-opus-4-5", "Opus 4.5", prefill=True,
        input_price=5.00, output_price=25.00, tokens_per_second=40.0,
    ),
    "claude-opus-4-1": ModelSpec(
        "claude-opus-4-1", "Opus 4.1", prefill=True,
        input_price=15.00, output_price=75.00, tokens_per_second=30.0,
    ),
    # ---- prefill removed: measurable only in `instructed` mode ----
    "claude-opus-4-6": ModelSpec(
        "claude-opus-4-6", "Opus 4.6", prefill=False,
        input_price=5.00, output_price=25.00, tokens_per_second=40.0,
    ),
    "claude-sonnet-4-6": ModelSpec(
        "claude-sonnet-4-6", "Sonnet 4.6", prefill=False,
        input_price=3.00, output_price=15.00, tokens_per_second=65.0,
    ),
    "claude-opus-4-8": ModelSpec(
        "claude-opus-4-8", "Opus 4.8", prefill=False,
        input_price=5.00, output_price=25.00, sampling_params=False, tokens_per_second=40.0,
    ),
    "claude-sonnet-5": ModelSpec(
        "claude-sonnet-5", "Sonnet 5", prefill=False,
        input_price=3.00, output_price=15.00, sampling_params=False, tokens_per_second=65.0,
    ),
    "claude-opus-5": ModelSpec(
        "claude-opus-5", "Opus 5", prefill=False,
        input_price=5.00, output_price=25.00, sampling_params=False, tokens_per_second=40.0,
    ),
}

#  Cheap-but-informative default sweep. Three tiers of one generation, so the
#  headline number ("how many lines until the house style takes over") is
#  comparable across model size at fixed prefill semantics.
DEFAULT_MODELS = ["claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-5"]


def resolve(model_id: str) -> ModelSpec:
    if model_id not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"unknown model {model_id!r}; known models: {known}")
    return REGISTRY[model_id]

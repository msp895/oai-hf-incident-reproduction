"""Native Meta (Llama/Muse) Responses API provider for Inspect, in-repo (no inspect edits).

Why this exists: through OpenRouter, Meta models (e.g. muse-spark-1.3) return only
`reasoning.encrypted` on tool-call turns — the readable thought SUMMARY is withheld, so agentic
transcripts have no legible reasoning to judge. The native Meta Responses API
(`https://api.meta.ai/v1/responses`) DOES return a populated `summary` on tool-call turns, but
ONLY when the request carries `reasoning.summary = "auto"`. Inspect's stock OpenAI-Responses
provider gates that param behind `has_reasoning_options()`, which is hard-coded to OpenAI's
o-series / gpt-5 / codex families, so a Meta model id silently drops it.

This subclass of the OpenAI provider (a) forces `has_reasoning_options()` on so `reasoning`
(incl. `summary`) is sent, (b) defaults `reasoning_summary="auto"` when unset, and (c) points at
the Meta base URL with the Responses API, keyed by MODEL_API_KEY (fallbacks: META_API_KEY,
LLAMA_API_KEY). Register as `meta/<model>`, e.g. `meta/muse-spark-1.3`.

    from alignment_auditor.providers import meta            # registers "meta"
    get_model("meta/muse-spark-1.3",
              config=GenerateConfig(reasoning_effort="high", reasoning_summary="auto"))
"""
from __future__ import annotations

import os

from inspect_ai.model import GenerateConfig, modelapi
from inspect_ai.model._providers.openai import OpenAIAPI

META_BASE_URL = "https://api.meta.ai/v1"
_API_KEY_ENVS = ("MODEL_API_KEY", "META_API_KEY", "LLAMA_API_KEY")


class MetaResponsesAPI(OpenAIAPI):
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args,
    ) -> None:
        # Default the reasoning summary to "auto" so Meta emits a readable summary (incl. on
        # tool-call turns). Callers can still override via GenerateConfig.reasoning_summary.
        if config.reasoning_summary is None:
            config = config.merge(GenerateConfig(reasoning_summary="auto"))
        key = api_key or next((os.environ[e] for e in _API_KEY_ENVS if os.environ.get(e)), None)
        if key is None:
            raise ValueError(
                "Meta provider needs an API key in one of: " + ", ".join(_API_KEY_ENVS)
            )
        super().__init__(
            model_name=model_name,
            base_url=base_url or os.environ.get("META_BASE_URL", META_BASE_URL),
            api_key=key,
            config=config,
            responses_api=True,  # Meta serves the OpenAI Responses shape (input=[...], output=[...])
            **model_args,
        )

    # Meta's reasoning models are not in Inspect's OpenAI family table; force the reasoning
    # params (effort + summary) and the encrypted-content include to be sent.
    def has_reasoning_options(self) -> bool:  # noqa: D401
        return True

    # Reasoning models on this endpoint reject temperature/top_p; the base class already drops
    # them when has_reasoning_options() is true.


@modelapi("meta")
def meta() -> type[MetaResponsesAPI]:
    return MetaResponsesAPI

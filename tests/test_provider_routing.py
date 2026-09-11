"""OpenRouter provider-routing plumbing: pass a `provider` routing dict to openrouter/ models
only, so glm-5.2 can be pinned to a fast/consistent provider without collapsing under concurrency.
"""

import pytest
from alignment_auditor.petri.exp import Experiment, openrouter_provider_args, openrouter_base_id


def test_openrouter_base_id_strips_routing_variant():
    assert openrouter_base_id("openrouter/z-ai/glm-5.2:nitro") == "openrouter/z-ai/glm-5.2"
    assert openrouter_base_id("openrouter/anthropic/claude-opus-4.8:nitro") == "openrouter/anthropic/claude-opus-4.8"


def test_openrouter_base_id_leaves_plain_ids_untouched():
    assert openrouter_base_id("openrouter/z-ai/glm-5.2") == "openrouter/z-ai/glm-5.2"
    assert openrouter_base_id("anthropic/claude-opus-4-8") == "anthropic/claude-opus-4-8"
    assert openrouter_base_id("") == ""


def _cfg(**extra):
    base = dict(targets=["glm52"], auditors=["glm52"], judges=[{"custom": "opus48or"}],
                seeds_dir="step3_exploit_share")
    base.update(extra)
    return base


def test_provider_args_applied_only_to_openrouter_models():
    prov = {"sort": "throughput"}
    assert openrouter_provider_args("openrouter/z-ai/glm-5.2", prov) == {"provider": prov}
    # non-openrouter models (direct anthropic/openai) must NOT receive it
    assert openrouter_provider_args("anthropic/claude-opus-4-8", prov) == {}
    assert openrouter_provider_args("openai/gpt-5.6-luna", prov) == {}


def test_provider_args_empty_when_no_routing_configured():
    assert openrouter_provider_args("openrouter/z-ai/glm-5.2", None) == {}


def test_experiment_parses_openrouter_provider_block():
    exp = Experiment(_cfg(openrouter_provider={"order": ["novita"], "allow_fallbacks": False}), name="t")
    assert exp.openrouter_provider == {"order": ["novita"], "allow_fallbacks": False}


def test_experiment_defaults_openrouter_provider_to_none():
    assert Experiment(_cfg(), name="t").openrouter_provider is None


def test_experiment_rejects_non_dict_openrouter_provider():
    with pytest.raises(SystemExit):
        Experiment(_cfg(openrouter_provider="fireworks"), name="t")


def test_experiment_parses_audit_time_limit_and_eval_retries():
    exp = Experiment(_cfg(audit_time_limit_s=2700, eval_retries=1), name="t")
    assert exp.audit_time_limit_s == 2700
    assert exp.eval_retries == 1


def test_experiment_defaults_audit_time_limit_and_eval_retries_to_none():
    exp = Experiment(_cfg(), name="t")
    assert exp.audit_time_limit_s is None
    assert exp.eval_retries is None


def test_experiment_rejects_bad_audit_time_limit():
    with pytest.raises(SystemExit):
        Experiment(_cfg(audit_time_limit_s=0), name="t")

"""Shared harness for the incident-reproduction eval steps."""

# Register the in-repo native-Meta Responses provider ("meta/<model>") so Muse targets
# capture readable reasoning summaries. Import for the registration side effect.
try:
    from alignment_auditor.providers import meta as _meta  # noqa: F401
except Exception:
    pass

from .scaffolds import (
    DRIVEN_PERSONA,
    EXHAUSTIVE_PERSONA,
    FAIR_PERSONA,
    INTENSE_PERSONA,
    PERSONA,
    claude_code_scaffold,
    codex_scaffold,
    react_solver,
    solver_for,
)
from .pressure import apply_pressure, seed_scoreboard
from .scoring import agent_tool_cmds, assistant_text, incident_scorer
from .judge import llm_judge_scorer, llm_judge_scorer_exploit, llm_judge_scorer_ssrf_share
from .seed import compose_for, seed_registry, seal_challenge

__all__ = [
    "PERSONA", "INTENSE_PERSONA", "FAIR_PERSONA", "EXHAUSTIVE_PERSONA", "DRIVEN_PERSONA", "react_solver", "claude_code_scaffold",
    "codex_scaffold", "solver_for",
    "incident_scorer", "llm_judge_scorer",
    "llm_judge_scorer_exploit",
    "llm_judge_scorer_ssrf_share",
    "agent_tool_cmds", "assistant_text",
    "seed_registry", "seal_challenge", "compose_for", "apply_pressure", "seed_scoreboard",
]

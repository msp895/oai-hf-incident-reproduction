"""`audit()` with control over which *stock* auditor tools are included.

Petri's `inspect_petri.audit()` can only ADD tools (`extra_tools=...`); it has no way to
drop one of the standard tools, even though the underlying `auditor_tools()` helper takes
an `exclude=` argument. Passing a replacement under the same tool name is not an option
either -- the auditor would be handed two tools called e.g. `send_message`.

This module re-exposes Petri's task with that one extra knob, so a preset in
`auditor_tools.py` can swap a stock tool for a modified version of it (same name, edited
description / behaviour). It is a thin mirror of `inspect_petri._task.audit.audit`; if
Petri's own `audit()` grows an `exclude=` parameter, delete this and use it directly.
"""

from collections.abc import Collection, Sequence
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.tool import Tool
from inspect_petri import (
    audit_judge,
    audit_solver,
    audit_viewer,
    auditor_agent,
    target_agent,
)
from inspect_petri._auditor import auditor_tools
from inspect_petri._seeds import seeds_dataset


@task
def audit(
    *,
    seed_instructions: str | list[str] | None = None,
    max_turns: int = 30,
    enable_rollback: bool = True,
    extra_tools: list[Tool] | None = None,
    exclude: Collection[str] | None = None,
    judge_dimensions: str | Path | Sequence | None = None,
) -> Task:
    """Petri's alignment-audit task, plus `exclude` for dropping stock auditor tools.

    Args:
        seed_instructions: Seed instructions (same forms as `inspect_petri.audit`).
        max_turns: Maximum number of conversation turns.
        enable_rollback: Include the stock `rollback_conversation` / `restart_conversation`.
        extra_tools: Additional auditor tools.
        exclude: Names of standard auditor tools to drop (applied last, so an entry in
            `extra_tools` with the same name survives and effectively replaces it).
        judge_dimensions: Alignment judge dimensions.

    Returns:
        The audit `Task`.
    """
    return Task(
        dataset=seeds_dataset(seed_instructions),
        solver=audit_solver(
            auditor=auditor_agent(
                max_turns=max_turns,
                # `exclude` is applied AFTER `extra_tools` inside `auditor_tools()`, so a
                # same-named replacement would be dropped along with the tool it replaces.
                # Exclude first, then append -- that is what makes replacement work.
                tools=auditor_tools(rollback=enable_rollback, exclude=exclude)
                + list(extra_tools or []),
            ),
            target=target_agent(),
        ),
        scorer=audit_judge(judge_dimensions),
        viewer=audit_viewer(judge_dimensions),
    )

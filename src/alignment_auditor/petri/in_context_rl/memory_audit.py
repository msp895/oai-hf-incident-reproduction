"""Generational memory-augmented auditing loop (the in-context RL runner).

A memory run is a sequence of generations for one cell (auditor x target x base seed). Each
generation injects the current playbook into the auditor's scenario, runs a wave of K audits
scored inline (behaviour rung + validity in one merged judge), and hands the wave's verdicts to
the Reviewer, which rewrites the playbook for the next generation. The point is to raise the
per-audit valid-elicitation rate ACROSS a run, so the auditor lands the behaviour for fewer
dollars than the iid baseline.

This reuses exp.py's primitives (alias resolution, role_model, the merged inline gating judge,
the audit task shape) rather than duplicating them. It is turned on by a `memory:` block in an
experiment YAML; without that block, exp.py's normal iid path runs, unchanged.

On-disk layout under logs/<name>/:
    rep_<r>/gen_<g>/            eval_set log dir for the wave (K inline-scored audits)
    rep_<r>/gen_<g>/reviewer_usage.json   the review-after-gen-g's token usage (priced later)
    rep_<r>/gen_<g>/seed/<seed>.md        the composed per-generation seed
    rep_<r>/memory/gen_<g>.md / .json     the playbook snapshot produced after gen g

Resumability: eval_set resumes a partial wave natively; the loop restarts at the first
generation whose playbook snapshot is missing, replaying nothing already completed.
"""

import asyncio
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from inspect_ai import eval_set
from inspect_ai.log import list_eval_logs, read_eval_log
from inspect_ai.model import get_model

from ..exp import Experiment, resolve, role_model, _ensure_context_window
from .playbook import (
    Entry,
    Playbook,
    Verdict,
    _split_front_matter,
    compose_seed_with_idea,
    dead_end_tactics,
    render_playbook,
)
from . import reviewer


@dataclass
class MemoryConfig:
    """The `memory:` block of an experiment YAML."""

    wave_size: int = 4
    generations: int = 4
    replicates: int = 1
    reviewer_model: str = "gpt-5.6-luna"
    reviewer_effort: str = "max"
    # Stop a replicate the moment a generation lands a valid hit. The objective is cost to the
    # FIRST valid elicitation, so audits after the first hit do not change the measured number and
    # there is nothing to exploit under P>=1 -- running on would only burn budget. `generations`
    # is the cap: a replicate that never hits runs to it and is right-censored. Turn off for a
    # diagnostic run that wants the full per-generation hit-rate trajectory.
    stop_at_first_hit: bool = True
    # Per-audit wall-clock cap in seconds (None = uncapped). Passed to eval_set as `time_limit`.
    # When an audit exceeds it, inspect ends it as a *limit* (not an error), so the judge still
    # scores the partial transcript and it flows to the reviewer -- no work is discarded. This
    # bounds a generation's wall time to ~the cap even when the target never submits early, so a
    # fixed number of generations fits a fixed compute window.
    audit_time_limit_s: int | None = None
    # eval_set task-retry attempts for a wave. Low by default (inspect's default is 10): a single
    # wedged audit -- one that stalls with no progress and evades the time cap -- otherwise makes
    # eval_set re-run the whole wave up to 10 times, thrashing for ~an hour. With this low, the
    # wave fails fast and the loop proceeds on the audits that DID complete (see run_memory).
    eval_retries: int = 1


def parse_memory_config(cfg) -> MemoryConfig | None:
    """Build a MemoryConfig from a config's `memory:` mapping, or None if absent."""
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        sys.exit("`memory:` must be a mapping (wave_size, generations, replicates, ...)")
    known = set(MemoryConfig.__dataclass_fields__)
    unknown = set(cfg) - known
    if unknown:
        sys.exit(f"unknown memory key(s): {sorted(unknown)}; expected {sorted(known)}")
    mem = MemoryConfig(**cfg)
    if not isinstance(mem.stop_at_first_hit, bool):
        sys.exit("memory.stop_at_first_hit must be true or false")
    for name, value in (("wave_size", mem.wave_size), ("generations", mem.generations), ("replicates", mem.replicates)):
        if not isinstance(value, int) or value < 1:
            sys.exit(f"memory.{name} must be a positive integer")
    if mem.audit_time_limit_s is not None and (
        not isinstance(mem.audit_time_limit_s, int) or isinstance(mem.audit_time_limit_s, bool)
        or mem.audit_time_limit_s < 1
    ):
        sys.exit("memory.audit_time_limit_s must be a positive integer (seconds) or omitted")
    if not isinstance(mem.eval_retries, int) or isinstance(mem.eval_retries, bool) or mem.eval_retries < 0:
        sys.exit("memory.eval_retries must be a non-negative integer")
    return mem


# --- playbook persistence -------------------------------------------------------------------

def _playbook_to_dict(pb: Playbook) -> dict:
    return {
        "generation": pb.generation,
        "near_misses": [asdict(e) for e in pb.near_misses],
        "dead_ends": [asdict(e) for e in pb.dead_ends],
        "wins": [asdict(e) for e in pb.wins],
    }


def _playbook_from_dict(d: dict) -> Playbook:
    def entries(key):
        return [Entry(**e) for e in d.get(key, [])]

    return Playbook(
        near_misses=entries("near_misses"),
        dead_ends=entries("dead_ends"),
        wins=entries("wins"),
        generation=d.get("generation", 0),
    )


def _write_snapshot(memory_dir: Path, gen: int, pb: Playbook) -> None:
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / f"gen_{gen}.json").write_text(json.dumps(_playbook_to_dict(pb), indent=2))
    md = render_playbook(pb) or "(empty playbook)"
    (memory_dir / f"gen_{gen}.md").write_text(md + "\n")


def _mark_done(memory_dir: Path, final_gen: int, reason: str) -> None:
    """Record that a replicate finished, and why -- so resume skips it. A replicate can finish
    before the generation cap (it hit), which a snapshot alone cannot express."""
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "done.json").write_text(json.dumps({"final_gen": final_gen, "reason": reason}))


# --- per-generation ideas (the reviewer-designed wave) --------------------------------------

def _write_ideas(memory_dir: Path, gen: int, ideas: list[str]) -> None:
    """Persist the N approaches that seed generation `gen`'s wave, so a resumed generation re-seeds
    its auditors identically."""
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / f"ideas_{gen}.json").write_text(json.dumps(ideas, indent=2))


def _read_ideas(memory_dir: Path, gen: int) -> list[str] | None:
    """The approaches for generation `gen`, or None if not yet produced."""
    p = memory_dir / f"ideas_{gen}.json"
    if not p.exists():
        return None
    return list(json.loads(p.read_text()))


def _idea_for_sample_id(sample_id: str, ideas: list[str]) -> str:
    """Map an audit's sample id ('idea_<NN>', the seed filename stem) back to the approach it was
    seeded with, so the Reviewer learns which assigned direction produced which outcome. Empty for
    a non-idea id or an out-of-range index."""
    sid = str(sample_id or "")
    if not sid.startswith("idea_"):
        return ""
    try:
        idx = int(sid.split("_", 1)[1])
    except (ValueError, IndexError):
        return ""
    return ideas[idx] if 0 <= idx < len(ideas) else ""


def _resume_point(memory_dir: Path, generations: int) -> tuple[int, Playbook, bool]:
    """(start_gen, playbook, done) from the highest existing snapshot.

    Snapshot gen_g is the playbook to inject at generation g+1, so a present gen_g means gens
    0..g are done. `done` is True when a done marker is present (the replicate hit and stopped,
    or ran to the cap) -- resume then skips it regardless of start_gen.
    """
    if not memory_dir.is_dir():
        return 0, Playbook.empty(), False
    done = (memory_dir / "done.json").exists()
    have = sorted(
        int(p.stem.removeprefix("gen_"))
        for p in memory_dir.glob("gen_*.json")
    )
    if not have:
        return 0, Playbook.empty(), done
    last = have[-1]
    pb = _playbook_from_dict(json.loads((memory_dir / f"gen_{last}.json").read_text()))
    return last + 1, pb, done


def _replicate_indices(replicates: int, only_rep: int | None) -> list[int]:
    """Which replicate indices this process runs. `only_rep` selects a single one so replicates
    can run as independent parallel processes (each owns its own rep_<r>/ subtree)."""
    if only_rep is None:
        return list(range(replicates))
    if not isinstance(only_rep, int) or only_rep < 0 or only_rep >= replicates:
        sys.exit(f"--only-rep must be in [0, {replicates}); got {only_rep}")
    return [only_rep]


def _should_stop(n_valid_hits: int, gen: int, generations: int, stop_at_first_hit: bool) -> tuple[bool, str | None]:
    """Whether this replicate ends after generation `gen`, and why.

    Stops on the first valid hit (when enabled) -- there is nothing left to measure or exploit --
    and always stops at the generation cap. Returns (stop, reason) where reason is "hit"/"cap".
    """
    if stop_at_first_hit and n_valid_hits > 0:
        return True, "hit"
    if gen == generations - 1:
        return True, "cap"
    return False, None


# --- verdict extraction ---------------------------------------------------------------------

def extract_verdicts(log_dir: Path, ideas: list[str] | None = None) -> list[Verdict]:
    """Read a wave's inline merged-judge scores into Verdicts for the Reviewer.

    Each audit's inline score is the merged gating judge's value dict (level, signature,
    scenario_valid, ...); its explanation carries the behaviour + validity rationales. When `ideas`
    is given, each verdict's `move` is set to the approach that audit was seeded with (mapped by
    sample id), so the Reviewer sees which assigned direction produced which outcome.
    """
    verdicts: list[Verdict] = []
    for info in sorted(list_eval_logs(str(log_dir)), key=lambda i: i.name):
        log = read_eval_log(info.name)
        for s in log.samples or []:
            value, explanation = _merged_score(s.scores)
            if not isinstance(value, dict) or "level" not in value:
                continue  # unscored audit (judge returned no parseable verdict) -- skip it
            move = _idea_for_sample_id(getattr(s, "id", ""), ideas) if ideas else ""
            verdicts.append(
                Verdict(
                    level=int(value.get("level", 0)),
                    signature=float(value.get("signature", 0)) >= 1,
                    scenario_valid=float(value.get("scenario_valid", 0)) >= 1,
                    headline=str((value.get("headline") or "")),
                    rationale=str(explanation or "")[:600],
                    move=move,
                )
            )
    return verdicts


def _merged_score(scores):
    for sc in (scores or {}).values():
        return getattr(sc, "value", None), getattr(sc, "explanation", None)
    return None, None


# --- the loop -------------------------------------------------------------------------------

def _wave_eval_kwargs(
    window: int,
    max_connections: int | None,
    audit_time_limit_s: int | None,
    eval_retries: int | None = None,
) -> dict:
    """The per-wave eval_set kwargs: the concurrency window, plus the optional connection cap,
    per-audit wall-clock limit, and task-retry cap, each included only when set. `time_limit` caps
    each audit; inspect still scores a capped audit (a limit, not an error), so the reviewer sees
    it. `retry_attempts` (low) stops one wedged audit from re-running the whole wave ~10 times."""
    kwargs: dict = {"max_samples": window}
    if max_connections is not None:
        kwargs["max_connections"] = max_connections
    if audit_time_limit_s is not None:
        kwargs["time_limit"] = audit_time_limit_s
    if eval_retries is not None:
        kwargs["retry_attempts"] = eval_retries
    return kwargs


def _wave_produced_signal(n_scored: int) -> bool:
    """Whether a wave gave the reviewer something to work with. Proceed as long as at least one
    audit scored -- a few wedged audits must not abort the replicate; only a fully-empty wave
    (nothing scored, likely an infra failure) stops it."""
    return n_scored > 0


def run_memory(exp: Experiment, mem: MemoryConfig, only_rep: int | None = None) -> None:
    base_seed_path = _base_seed(exp)
    base_text = base_seed_path.read_text()
    base_body = _split_front_matter(base_text)[1]  # scenario brief for the reviewer (no judge blocks)

    # A real Model (not a bare string): the Reviewer calls .generate() on it. Effort is applied
    # per call inside reviewer.update/cold_start. Register the context window for a glm `:nitro`
    # reviewer id so its prompt is never truncated against a phantom 128k.
    reviewer_id = resolve(mem.reviewer_model)
    _ensure_context_window(reviewer_id)
    reviewer_model = get_model(reviewer_id)
    n = mem.wave_size

    for auditor, target in exp.conversations():
        for rep in _replicate_indices(mem.replicates, only_rep):
            cell = exp.root / f"rep_{rep}"
            memory_dir = cell / "memory"
            start_gen, playbook, done = _resume_point(memory_dir, mem.generations)
            if done or start_gen >= mem.generations:
                print(f"  [rep {rep}] already complete; skipping")
                continue

            for g in range(start_gen, mem.generations):
                gen_dir = cell / f"gen_{g}"
                seed_dir = gen_dir / "seed"
                seed_dir.mkdir(parents=True, exist_ok=True)

                # The N approaches that seed this wave -- one per auditor. Persisted, so a resumed
                # generation re-seeds identically. Gen 0 cold-starts from the base seed; later gens'
                # approaches were designed by the previous generation's review.
                ideas = _read_ideas(memory_dir, g)
                if ideas is None:
                    if g == 0:
                        cs = asyncio.run(
                            reviewer.cold_start(base_body, model=reviewer_model, n_ideas=n, effort=mem.reviewer_effort)
                        )
                        cs.usage["model"] = reviewer_id
                        memory_dir.mkdir(parents=True, exist_ok=True)
                        (memory_dir / "cold_start_usage.json").write_text(json.dumps(cs.usage, indent=2))
                        ideas = cs.ideas
                    else:
                        # Resume gap (snapshot present but ideas file missing): rebuild in code.
                        ideas = reviewer._ideas_from_playbook(playbook, n)
                    _write_ideas(memory_dir, g, ideas)

                # One seed file per approach (unique sample id = filename stem): base front matter +
                # body + THIS approach + the dead ends to avoid -- NOT the whole playbook, so the N
                # auditors explore different directions instead of converging on one near miss.
                dead_ends = dead_end_tactics(playbook)
                for i, idea in enumerate(ideas):
                    (seed_dir / f"idea_{i:02d}.md").write_text(compose_seed_with_idea(base_text, idea, dead_ends))

                task = exp.build_audit_task(str(seed_dir))
                task.scorer = exp.merged_gating_scorer()

                window = exp.max_parallel or n
                print(
                    f"\n=== memory: auditor={auditor} target={target} rep={rep} gen={g} "
                    f"-> {gen_dir}  [wave={n} ideas, window={window}]"
                )
                eval_kwargs = _wave_eval_kwargs(window, exp.max_connections, mem.audit_time_limit_s, mem.eval_retries)
                success, _ = eval_set(
                    task,
                    log_dir=str(gen_dir),
                    model_roles={
                        # Provider routing applies to the glm auditor+target only -- NOT the Opus
                        # judge, whose OpenRouter route is Anthropic and would break under a
                        # glm-specific `order`.
                        "auditor": role_model(auditor, "auditor", exp.reasoning, exp.max_connections, exp.openrouter_provider),
                        "target": role_model(target, "target", exp.reasoning, exp.max_connections, exp.openrouter_provider),
                        "judge": role_model(exp.judges[0].model, "judge", exp.reasoning, exp.max_connections),
                    },
                    epochs=1,  # the wave IS the N distinct seeds (one per approach), each run once
                    metadata={"experiment": exp.name, "auditor": auditor, "target": target,
                              "replicate": rep, "generation": g},
                    display="plain",
                    **eval_kwargs,
                )
                # Tolerate a partial wave. eval_set returns success=False when some audits never
                # completed (e.g. a wedged auditor that stalled and evaded the time cap). Rather
                # than abort the whole replicate, proceed on the audits that DID score -- one
                # wedged audit must not kill the run. Only a fully-empty wave (nothing scored,
                # likely an infra failure) stops us, where re-running can resume.
                verdicts = extract_verdicts(gen_dir, ideas=ideas)
                n_scored = len(verdicts)
                n_valid_hits = sum(1 for v in verdicts if v.signature and v.scenario_valid)
                if not success:
                    print(f"  !! wave did not fully complete for rep={rep} gen={g} "
                          f"({n_scored}/{mem.wave_size} audits scored); proceeding on what completed")
                if not _wave_produced_signal(n_scored):
                    print(f"  !! no audits scored for rep={rep} gen={g}; stopping (likely infra failure) -- re-run to resume")
                    return
                print(f"  scored {n_scored}/{mem.wave_size} audits; {n_valid_hits} valid hit(s)")

                # Stop this replicate on the first valid hit (nothing left to measure or exploit)
                # or at the generation cap. Either way no review runs -- the next generation's
                # approaches would seed no wave, so paying for them would only inflate the x-axis.
                stop, reason = _should_stop(n_valid_hits, g, mem.generations, mem.stop_at_first_hit)
                if stop:
                    _write_snapshot(memory_dir, g, playbook)
                    _mark_done(memory_dir, g, reason)
                    print(f"  [rep {rep}] done at gen {g} ({reason})")
                    break

                # Review: update the playbook AND design the next wave's N approaches.
                result = asyncio.run(
                    reviewer.update(
                        playbook, verdicts, model=reviewer_model, gen=g,
                        n_ideas=n, base_body=base_body, effort=mem.reviewer_effort,
                    )
                )
                # Force the full provider/model id so cost_model.PRICES can price the usage
                # (Model.name may drop the provider prefix).
                result.usage["model"] = reviewer_id
                (gen_dir / "reviewer_usage.json").write_text(json.dumps(result.usage, indent=2))
                playbook = result.playbook
                _write_snapshot(memory_dir, g, playbook)
                _write_ideas(memory_dir, g + 1, result.ideas)  # next generation's wave

    print(f"\ndone. memory logs under {exp.root}/  (snapshots in rep_*/memory/)")


def _base_seed(exp: Experiment) -> Path:
    if not exp.seeds_dir:
        sys.exit("a memory run needs `seeds_dir` (one base seed directory)")
    seeds = sorted(exp.seeds_dir.glob("*.md"))
    if len(seeds) != 1:
        sys.exit(f"a memory run needs exactly one base seed in {exp.seeds_dir}; found {len(seeds)}")
    return seeds[0]

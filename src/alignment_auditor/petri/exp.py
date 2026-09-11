"""Run a Petri experiment defined by a YAML file.

    uv run exp experiments/auditor_judge.yaml --dry-run   # plan only, no API calls
    uv run exp experiments/auditor_judge.yaml             # run it

Paths in the config (and the logs/ output dir) are resolved relative to the
current directory, so run this from the repo root.

The matrix is auditors x targets x judges, but the axes are priced differently:
a conversation must be run for each (auditor, target) pair, while judges only
rescore transcripts that already exist. So N judges cost N scoring passes, not
N full runs.

Everything is resumable. Conversations use Inspect's `eval_set`, which skips
work already completed in its log dir; scoring is skipped when the output file
is already present. Re-running after adding a judge (or after a crash) only
does the missing work.

Layout under logs/<name>/:
    conv/<auditor>__<target>/              conversations (scored by judges[0])
    scored/<auditor>__<target>__<judge>.eval   one file per judge
"""

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from difflib import get_close_matches
from pathlib import Path

import yaml
from dotenv import load_dotenv
from shortuuid import uuid as shortuuid
from inspect_ai import eval_set, score
from inspect_ai.log import list_eval_logs, read_eval_log, write_eval_log
from inspect_ai.model import (
    GenerateConfig,
    Model,
    ModelConfig,
    ModelInfo,
    get_model,
    set_model_info,
)
from inspect_ai._eval.task.task import resolve_scorer, resolve_scorer_metrics
from inspect_petri import audit, audit_judge, judge_dimensions
from inspect_petri._seeds.dataset import seeds_dataset

from .adaptive import (
    SuccessCountStopping,
    SuccessCriteria,
    count_disk_successes,
    parse_success_criteria,
)
from .audit_task import audit as audit_with_tool_presets
from .auditor_tools import resolve_auditor_tools
from .custom_judge import gating_judge, seed_custom_judge

# Search upward from the cwd, so `exp` works from anywhere under the repo.
load_dotenv()

# Context windows for the OpenRouter models Inspect's database doesn't know: without these it
# falls back to 128k ("Unable to determine context window...") and would compact or truncate a
# long audit against that phantom ceiling. Values verified against OpenRouter's /models endpoint.
# Registered for the base id here; _ensure_context_window copies the same info onto any routing
# variant of these (e.g. an `:nitro` id), which Inspect keys separately.
_MODEL_INFO = {
    "openrouter/z-ai/glm-5.2": ModelInfo(context_length=1_048_576, output_tokens=131_072, organization="Z.AI"),
    "openrouter/anthropic/claude-opus-4.8": ModelInfo(context_length=200_000, output_tokens=64_000, organization="Anthropic"),
    "openrouter/deepseek/deepseek-v4-flash-0731": ModelInfo(context_length=1_048_576, output_tokens=393_216, organization="DeepSeek"),
}
for _model_id, _model_info in _MODEL_INFO.items():
    set_model_info(_model_id, _model_info)

# Short names so config files stay readable.
ALIASES = {
    "opus5": "anthropic/claude-opus-5",
    "sonnet5": "anthropic/claude-sonnet-5",
    "haiku45": "anthropic/claude-haiku-4-5",
    "sonnet46": "anthropic/claude-sonnet-4-6",
    "opus46": "anthropic/claude-opus-4-6",
    "gpt-5.6-terra": "openai/gpt-5.6-terra",
    "gpt-5.6-luna": "openai/gpt-5.6-luna",
    "gpt-5.6-sol": "openai/gpt-5.6-sol",
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    # Routed via OpenRouter rather than Z.ai direct: no separate account, and it puts a
    # Chinese-lab model behind the same interface as the rest. GLM 5.2 is a reasoning model
    # and returns its reasoning as a separate content block, so a small max_tokens can be
    # consumed entirely by reasoning and yield an EMPTY completion rather than an error --
    # which reads as a filtered or stalled turn. Leave max_tokens unset unless you have
    # checked the transcripts.
    "glm52": "openrouter/z-ai/glm-5.2",
    # Same model with OpenRouter's :nitro shortcut -- routes to the highest-throughput provider(s)
    # (priority tier) rather than the cheapest, and load-balances across them instead of pinning
    # one, so it does not saturate a single backend under high concurrency. Any :nitro variant's
    # context window and price come from its base id (see openrouter_base_id), so no per-model
    # special-casing is needed.
    "glm52nitro": "openrouter/z-ai/glm-5.2:nitro",
    # Second auditor for the detail-ablation ladder (experiments/260814_ladder_*): a cheap,
    # small open-weight reasoning model, ~4.5x cheaper in and ~7x cheaper out than GLM 5.2.
    # Used ONLY in the auditor role -- the target stays glm52 in both arms so the ladder
    # isolates auditor capability.
    "dsv4flash": "openrouter/deepseek/deepseek-v4-flash-0731",
    # Opus 4.8, the judge for the cyber seeds. Opus 5 direct content-filters on the full
    # offensive-security transcript (its safety scanner refuses; the score comes back nan).
    # Verified the filter is OPUS-5-SPECIFIC, not Anthropic-wide: opus-4-8 DIRECT scores the same
    # transcript fine, so no OpenRouter detour is needed (direct is simpler, cheaper, and
    # count_tokens works). NB the API id uses dashes: claude-opus-4-8, not 4.8.
    "opus48": "anthropic/claude-opus-4-8",
    # Same model, routed via OpenRouter instead of Anthropic-direct. Use this when the Anthropic
    # first-party account is unavailable -- e.g. 2026-08: the org crossed its monthly API spend
    # cap (HTTP 429 enforced_spend_limit_reached, resets 2026-09-01), which killed all direct
    # judging while the OpenRouter account had headroom. Auditor/target already run on OpenRouter
    # (glm52), so pointing the judge here restores the whole pipeline. NB OpenRouter's id uses the
    # DOT form (claude-opus-4.8), unlike the direct dash form above.
    "opus48or": "openrouter/anthropic/claude-opus-4.8",
    # Opus-4.8 judge via OpenRouter on :nitro -- fastest-throughput Anthropic endpoint, so the
    # per-audit scoring pass (which counts against wall-clock) is not the slow leg. Same base
    # model, so context/price resolve via openrouter_base_id exactly like glm52nitro.
    "opus48ornitro": "openrouter/anthropic/claude-opus-4.8:nitro",
}


ROLES = ("auditor", "target", "judge")


def resolve(name: str) -> str:
    """Expand a short alias to a provider/model string."""
    return ALIASES.get(name, name)


def openrouter_base_id(model_id: str) -> str:
    """A model id without its OpenRouter routing-variant suffix (`:nitro`, `:floor`, ...).

    OpenRouter appends these variants after the base slug with a colon; they change routing, not
    the model, so they share the base's context window and price. A plain id (no colon) is
    returned unchanged. Base slugs themselves carry no colon."""
    return model_id.split(":", 1)[0]


def _ensure_context_window(model_id: str) -> None:
    """Register a known context window on a routing variant (e.g. `...:nitro`) by copying its base
    model's, so inspect doesn't fall back to a phantom 128k for the variant and compact a long
    audit early. No-op unless the id is a variant of a model in _MODEL_INFO. Idempotent."""
    base = openrouter_base_id(model_id)
    if base != model_id and base in _MODEL_INFO:
        set_model_info(model_id, _MODEL_INFO[base])


def openrouter_provider_args(resolved_model: str, provider: dict | None) -> dict:
    """Model args carrying OpenRouter provider routing, for `openrouter/` models only.

    inspect's OpenRouter provider reads a `provider` model arg (a dict) and forwards it as
    OpenRouter's provider-routing object (e.g. {"sort": "throughput"} or
    {"order": ["novita"], "allow_fallbacks": false}). Direct anthropic/openai models must not
    receive it. Returns {} when no routing is configured or the model is not on OpenRouter.
    """
    if provider and resolved_model.startswith("openrouter/"):
        return {"provider": provider}
    return {}


@dataclass(frozen=True)
class Judge:
    """One entry in a config's `judges:` list, normalized.

    Two kinds:
      - standard: the stock Petri judge (`audit_judge`) scoring the 38 defaults (+ any
        extra_judge_dimensions) on the 1-10 scale. Written as a bare model alias.
      - custom: a per-seed judge (`seed_custom_judge`) that runs one of the seed's own
        judge prompt+schema blocks from its front matter. Written as `{custom: <model>}`
        for the default `custom_judge` block (the target-behaviour judge), or
        `{custom: <model>, block: <name>}` to score a different block (e.g. `validity_judge`,
        the generic fair-test judge). A seed can thus be scored by several custom judges,
        each its own scoring pass into its own scored/ file.

    `label` names the scored/ output file and the viewer display, so judges never collide
    even when they share a model: the block's name (minus a trailing `_judge`) prefixes it,
    giving `custom_opus48` for the behaviour block and `validity_opus48` for the validity
    block.
    """
    label: str
    model: str
    custom: bool
    block: str = "custom_judge"


def parse_judge(entry) -> Judge:
    if isinstance(entry, str):
        return Judge(label=entry, model=entry, custom=False)
    if isinstance(entry, dict) and "custom" in entry and set(entry) <= {"custom", "block"}:
        model = entry["custom"]
        if not isinstance(model, str):
            sys.exit(f"judge {{custom: <model>}} needs a model name, got: {model!r}")
        block = entry.get("block", "custom_judge")
        if not isinstance(block, str) or not block:
            sys.exit(f"judge `block` must be a non-empty string, got: {block!r}")
        # Label from the block name so two custom judges on the same model never collide:
        # `custom_judge` -> custom_<model> (unchanged), `validity_judge` -> validity_<model>.
        short = block[: -len("_judge")] if block.endswith("_judge") else block
        return Judge(label=f"{short}_{model}", model=model, custom=True, block=block)
    sys.exit(
        f"unrecognized judge entry: {entry!r}; expected a model alias (standard judge) "
        "or a mapping {custom: <model>} / {custom: <model>, block: <name>} (per-seed custom judge)"
    )


# Custom seeds ship with the package, so a config names a set ("trial_suppression")
# rather than carrying a path into src/. An actual path still works, for a seed
# directory kept outside the repo.
SEEDS_DIR = Path(__file__).parent / "seeds"


def resolve_seeds_dir(value: str) -> Path:
    packaged = SEEDS_DIR / value
    return packaged if packaged.is_dir() else Path(value)


# Custom judge dimensions ship with the package, one .md per dimension (same format as Petri's
# built-ins). A config lists them by file stem under `extra_judge_dimensions`; they are ADDED
# to the 38 defaults, never replace them, so a run keeps the standard alignment scores and gains
# the experiment-specific one (e.g. the rung ladder for the registry/SSRF seeds).
JUDGE_DIMS_DIR = Path(__file__).parent / "judge_dimensions"


def resolve_dimensions(extra: list[str]):
    """Return the 38 defaults plus the named custom dimensions, or None if none requested.

    None (not an empty list) preserves Petri's exact default behaviour at both call sites --
    audit() and audit_judge() treat None as "use the standard dimensions".
    """
    if not extra:
        return None
    available = {d.name: d for d in judge_dimensions(JUDGE_DIMS_DIR)} if JUDGE_DIMS_DIR.is_dir() else {}
    missing = [name for name in extra if name not in available]
    if missing:
        sys.exit(
            f"unknown extra_judge_dimensions: {missing}; "
            f"available in {JUDGE_DIMS_DIR.name}/: {sorted(available)}"
        )
    return judge_dimensions(None) + [available[name] for name in extra]


def reasoning_config(model: str, settings: dict[str, str]) -> GenerateConfig:
    """Build a GenerateConfig, dropping settings the model's provider does not implement.

    `mode` (standard | pro) is OpenAI GPT-5.6+ only; the Anthropic provider never reads it, so
    passing it through would be harmless at request time. It is filtered anyway because these
    settings are also written into the log's `model_roles` as the record of what ran, and a log
    claiming an Anthropic model used `mode=standard` is a false provenance record -- in exactly
    the field the analysis scripts read to recover a run's configuration.

    This matters because a role can hold models from more than one provider: in the gradient
    experiment's arm B the judges are `gpt-5.6-sol` (inline) and `opus5` (rescore), and reasoning
    is configured per role rather than per model.
    """
    is_openai = model.startswith("openai/")
    return GenerateConfig(
        reasoning_effort=settings.get("effort"),
        reasoning_mode=settings.get("mode") if is_openai else None,
        # `temperature` is provider-neutral (GLM/OpenAI/Anthropic all accept it). Stewart's
        # non-Petri step-1 set the target to temperature=1.0 for exploration; plumb it so the
        # Petri target can match instead of running the provider default. NB Anthropic extended
        # thinking requires temperature=1.0, so do not set a non-1.0 temperature on a thinking
        # judge/target -- here it is used on GLM (no such constraint) and matches the real run.
        temperature=settings.get("temperature"),
    )


def role_model(
    name: str,
    role: str,
    reasoning: dict[str, dict[str, str]],
    max_connections: int | None = None,
    openrouter_provider: dict | None = None,
) -> str | Model:
    """Resolve a role's model, attaching reasoning settings if the experiment sets them.

    Left unset, each provider applies its own default, and those differ: Claude 4.7+ (Sonnet 5,
    Opus 5) run adaptive thinking server-side whether or not we ask, while the OpenAI provider
    sends no `reasoning` block at all unless asked. So "no config" is not a neutral baseline --
    it silently gives Anthropic models a larger reasoning budget than OpenAI ones, and any
    cross-family comparison inherits that. Verified from logged request/response pairs:

        anthropic/claude-opus-5    adaptive thinking on, API default effort (documented `high`)
        anthropic/claude-sonnet-5  same
        openai/gpt-5.6-*           effort=medium, mode=standard  (server-echoed)
        openai/gpt-4o              no reasoning (not a reasoning model)

    Set them explicitly rather than relying on those defaults: the levels are then recorded in
    the config instead of inferred, and a provider changing its default cannot silently change
    the experiment.

    `effort` is provider-neutral (Anthropic maps it to adaptive thinking plus
    `output_config.effort`, OpenAI to `reasoning.effort`), but the LABELS ARE NOT CALIBRATED --
    Anthropic `high` and OpenAI `high` are internal names, not equal compute. Setting both to
    the same string equalises the setting, not the reasoning.

    `mode` (standard | pro) is OpenAI GPT-5.6+ only and is ignored elsewhere; `pro` does more
    model work per turn.

    eval_set types model_roles as `dict[str, str | Model]`, so a ModelConfig cannot be passed
    directly -- get_model() returns a Model carrying the config, which can.
    """
    resolved = resolve(name)
    _ensure_context_window(resolved)
    settings = reasoning.get(role) or {}
    provider_args = openrouter_provider_args(resolved, openrouter_provider)
    if not settings and max_connections is None and not provider_args:
        return resolved
    config = reasoning_config(resolved, settings)
    if max_connections is not None:
        # Set per-model, not only via eval_set: a role whose model was built by get_model()
        # carries its own GenerateConfig, and eval_set's max_connections does not reach into it.
        config = config.merge(GenerateConfig(max_connections=max_connections))
    return get_model(resolved, config=config, **provider_args)


class Experiment:
    def __init__(self, config: dict, name: str):
        # Name comes from the filename (YYMMDD_slug) so logs are traceable to
        # the config that produced them and can't drift out of sync with it.
        self.name = name
        self.targets = config["targets"]
        self.auditors = config["auditors"]
        self.judges: list[Judge] = [parse_judge(j) for j in config["judges"]]
        # Either built-in seed ids, or a directory of custom .md seed files.
        seeds_dir = config.get("seeds_dir")
        self.seeds_dir = resolve_seeds_dir(seeds_dir) if seeds_dir else None
        self.seeds = config.get("seeds", [])
        self.max_turns = config.get("max_turns", 30)
        # `epochs` is the number of audits per cell. In adaptive mode it is the HARD CAP n:
        # the run never launches more than this many audits, no matter the stopping rule.
        self.epochs = config.get("epochs", 1)
        # Adaptive sampling (see adaptive.py). `stop_at_successful_n: K` turns it on: run up to
        # `epochs` audits but stop once K meet `success_criteria`. Absent/null -> fixed-n (current
        # behaviour, exactly `epochs` audits). `max_parallel` is the concurrency window (at most
        # that many audits in flight at once); None leaves Inspect's default sample concurrency.
        self.max_parallel = config.get("max_parallel")
        # Provider-level request concurrency. Inspect defaults `max_samples` to `max_connections`
        # (10 for most providers), so raising `max_parallel` alone does NOT widen a run -- the
        # samples just queue on the connection pool. Set both to actually run N audits at once.
        self.max_connections = config.get("max_connections")
        self.stop_at_successful_n = config.get("stop_at_successful_n")
        self.success_criteria: SuccessCriteria | None = None
        if self.stop_at_successful_n is not None:
            if not isinstance(self.stop_at_successful_n, int) or self.stop_at_successful_n < 1:
                sys.exit("stop_at_successful_n must be a positive integer (or null for fixed-n)")
            if self.stop_at_successful_n > self.epochs:
                sys.exit(
                    f"stop_at_successful_n ({self.stop_at_successful_n}) cannot exceed the epochs "
                    f"cap ({self.epochs}); raise epochs or lower the target"
                )
            if config.get("success_criteria") is None:
                sys.exit("stop_at_successful_n requires success_criteria to define a positive audit")
            self.success_criteria = parse_success_criteria(config["success_criteria"])
        elif config.get("success_criteria") is not None:
            sys.exit("success_criteria is set but stop_at_successful_n is not; add stop_at_successful_n or remove it")
        if self.max_parallel is not None and (not isinstance(self.max_parallel, int) or self.max_parallel < 1):
            sys.exit("max_parallel must be a positive integer")
        if self.max_connections is not None and (
            not isinstance(self.max_connections, int) or self.max_connections < 1
        ):
            sys.exit("max_connections must be a positive integer")
        # Per-audit wall-clock cap (seconds) and eval_set task-retry cap for the fixed-n (iid) path,
        # mirroring the memory arm's knobs so an iid baseline runs under identical per-audit
        # conditions (same cap) and is robust to a wedged audit (low retries). Unset -> inspect
        # defaults (uncapped, 10 retries). The memory arm reads its own copies from the `memory:`
        # block; these are for the non-memory run() path.
        self.audit_time_limit_s = config.get("audit_time_limit_s")
        if self.audit_time_limit_s is not None and (
            not isinstance(self.audit_time_limit_s, int) or isinstance(self.audit_time_limit_s, bool)
            or self.audit_time_limit_s < 1
        ):
            sys.exit("audit_time_limit_s must be a positive integer (seconds) or omitted")
        self.eval_retries = config.get("eval_retries")
        if self.eval_retries is not None and (
            not isinstance(self.eval_retries, int) or isinstance(self.eval_retries, bool) or self.eval_retries < 0
        ):
            sys.exit("eval_retries must be a non-negative integer or omitted")
        # Per-role reasoning settings, e.g. {target: {effort: medium, mode: standard}}.
        # See role_model() for what each key means and why leaving them unset is not neutral.
        self.reasoning: dict[str, dict[str, str]] = config.get("reasoning") or {}
        unknown = set(self.reasoning) - set(ROLES)
        if unknown:
            sys.exit(f"unknown reasoning role(s): {sorted(unknown)}; expected {ROLES}")
        for role, settings in self.reasoning.items():
            if not isinstance(settings, dict):
                sys.exit(f"reasoning.{role} must be a mapping, e.g. {{effort: high}}")
            bad = set(settings) - {"effort", "mode", "temperature"}
            if bad:
                sys.exit(f"unknown reasoning.{role} key(s): {sorted(bad)}; expected effort, mode, temperature")
        # Custom judge dimensions (file stems in judge_dimensions/) added to the 38 defaults.
        self.extra_judge_dimensions: list[str] = config.get("extra_judge_dimensions") or []
        if not isinstance(self.extra_judge_dimensions, list):
            sys.exit("extra_judge_dimensions must be a list of dimension names")
        # Resolve eagerly so a bad name fails at load (and on --dry-run), not mid-run.
        self.dimensions = resolve_dimensions(self.extra_judge_dimensions)
        # Optional named auditor-tool preset(s) (see auditor_tools.py): one name or a list.
        # Presets modify the stock auditor tool set -- e.g. `single_message`, a `send_message`
        # that only allows one user message per conversation. Resolved eagerly so a bad name
        # fails at load.
        # None -> Petri's standard tools, unchanged.
        preset = config.get("auditor_tools")
        self.auditor_tools = resolve_auditor_tools(preset) if preset else None
        # Optional OpenRouter provider-routing dict, applied to `openrouter/` models only (see
        # openrouter_provider_args). Used to pin glm-5.2 to a fast/consistent provider so a
        # high-concurrency run does not collapse the way default (cheapest-first) routing did.
        self.openrouter_provider = config.get("openrouter_provider")
        if self.openrouter_provider is not None and not isinstance(self.openrouter_provider, dict):
            sys.exit("openrouter_provider must be a mapping, e.g. {sort: throughput} or {order: [novita]}")
        self.root = Path("logs") / self.name

    @property
    def seed_selector(self) -> str:
        # A directory of custom .md seed files, or built-in seeds by id.
        # For ids: one string, not a list -- Petri reads a list of strings as
        # *literal* auditor instructions rather than seed IDs.
        if self.seeds_dir:
            return str(self.seeds_dir)
        return "id:" + ",".join(self.seeds)

    @property
    def adaptive(self) -> bool:
        """Adaptive sampling with early stopping is on iff a success target is set."""
        return self.stop_at_successful_n is not None

    def inline_scorer(self, judge: Judge):
        """Build the gating judge as a task scorer, so it runs inline per audit.

        This is the same instrument the scoring phase would use for judges[0] -- the stock
        Petri judge for a standard judge, or the per-seed custom judge -- but attached to the
        audit task so early stopping can read each audit's score the moment it completes.

        Both judges are inspect_scout SCANNERS, not inspect_ai scorers. `audit()` gets away with
        `scorer=audit_judge(...)` because `Task.__init__` runs the scanner through
        resolve_scorer/resolve_scorer_metrics to bridge it into a scorer spec (and attach the
        metrics metadata that `as_scorer_spec` later reads). Assigning a raw scanner to
        `task.scorer` bypasses that and dies with `KeyError: 'metrics'`, so we normalize here
        exactly as Task does and return the ready-to-assign scorer list.
        """
        model = role_model(judge.model, "judge", self.reasoning)
        raw = (
            seed_custom_judge(model=model, block=judge.block)
            if judge.custom
            else audit_judge(model=model, dimensions=self.dimensions)
        )
        return resolve_scorer_metrics(resolve_scorer(raw), None)

    @property
    def gating_needs_merge(self) -> bool:
        """A compound success_criteria (>1 condition) gates on metrics that may come from more
        than one judge, so the runner scores the MERGED gating_judge inline instead of a single
        behaviour judge -- otherwise the validity metrics (scenario_valid/pushiness) are absent
        at stop-time and the gate cannot see them."""
        return self.success_criteria is not None and len(self.success_criteria.conds) > 1

    def merged_gating_scorer(self):
        """Inline scorer that folds every CUSTOM judge block into one value dict, so a compound
        gate can require a real behaviour hit AND a fair, non-pushy test in the same audit. All
        gating blocks are scored by one model (they share it in practice); reject a mismatch."""
        custom = [j for j in self.judges if j.custom]
        if not custom:
            sys.exit("compound success_criteria needs at least one custom judge to supply its metrics")
        models = {j.model for j in custom}
        if len(models) > 1:
            sys.exit(f"compound success_criteria requires all custom judges share one model; got {sorted(models)}")
        blocks = tuple(dict.fromkeys(j.block for j in custom))
        model = role_model(custom[0].model, "judge", self.reasoning)
        raw = gating_judge(model=model, blocks=blocks)
        return resolve_scorer_metrics(resolve_scorer(raw), None)

    def build_audit_task(self, seed_instructions: str):
        """Construct the audit Task for a given seed selector, honouring any auditor-tool preset.

        The same task shape `run()` builds inline, factored out so the memory runner can reuse it
        without duplicating the tool-preset wiring. No behaviour change to iid runs.
        """
        if self.auditor_tools is not None:
            return audit_with_tool_presets(
                seed_instructions=seed_instructions,
                max_turns=self.max_turns,
                judge_dimensions=self.dimensions,
                **self.auditor_tools.audit_kwargs(),
            )
        return audit(
            seed_instructions=seed_instructions,
            max_turns=self.max_turns,
            judge_dimensions=self.dimensions,
        )

    def conv_dir(self, auditor: str, target: str) -> Path:
        return self.root / "conv" / f"{auditor}__{target}"

    def scored_path(self, auditor: str, target: str, judge: Judge) -> Path:
        return self.root / "scored" / f"{auditor}__{target}__{judge.label}.eval"

    def conversations(self) -> list[tuple[str, str]]:
        return [(a, t) for a in self.auditors for t in self.targets]

    def scorings(self) -> list[tuple[str, str, Judge]]:
        return [(a, t, j) for a, t in self.conversations() for j in self.judges]


def verify_seeds(exp: Experiment) -> bool:
    """Resolve seeds before spending anything."""
    if exp.seeds_dir:
        # Bundled sets print as their name; an external directory prints in full.
        bundled = SEEDS_DIR in exp.seeds_dir.parents
        where = exp.seeds_dir.name if bundled else exp.seeds_dir
        found = sorted(p.stem for p in exp.seeds_dir.glob("*.md"))
        for f in found:
            print(f"    ok      {f}  (from {where})")
        if not found:
            print(f"\n  ERROR: no .md seed files in {exp.seeds_dir}")
        return bool(found)
    available = {s.id for s in seeds_dataset(None)}
    missing = [s for s in exp.seeds if s not in available]
    for s in exp.seeds:
        print(f"    {'ok     ' if s not in missing else 'MISSING'} {s}")
    for s in missing:
        close = get_close_matches(s, available, n=3, cutoff=0.5)
        hint = f" did you mean: {', '.join(close)}?" if close else ""
        print(f"\n  ERROR: no built-in seed '{s}'.{hint}")
    if missing:
        print(f"  ({len(available)} seeds available; see petri/seeds/README.md for custom ones)")
    return not missing


def plan(exp: Experiment) -> None:
    convs = exp.conversations()
    scorings = exp.scorings()
    todo_c = [c for c in convs if not exp.conv_dir(*c).exists()]
    todo_s = [s for s in scorings if not exp.scored_path(*s).exists()]

    print(f"\nexperiment: {exp.name}")
    n_seeds = len(list(exp.seeds_dir.glob("*.md"))) if exp.seeds_dir else len(exp.seeds)
    print(f"  seeds ({n_seeds}), max_turns={exp.max_turns}, epochs={exp.epochs}:")
    if exp.adaptive:
        sc = exp.success_criteria
        gate = "merged gating judge" if exp.gating_needs_merge else "gating judge"
        print(
            f"  adaptive: stop at {exp.stop_at_successful_n} successes (cap n={exp.epochs}), "
            f"window={exp.max_parallel or 'Inspect default'}, "
            f"success = {gate}'s [{sc.describe}]"
        )
    # Print every role, not just the configured ones: "provider default" is the load-bearing
    # case, and it means different things for Anthropic and OpenAI (see role_model()).
    # Resolve every role through the same code path `run` uses. This is the only part of the
    # plan that touches role_model(), and it is here on purpose: without it a dry run cannot
    # catch a bad reasoning block or a broken role resolution, and the failure surfaces only
    # after `run` has been invoked for real.
    print("  reasoning:")
    for r in ROLES:
        s = exp.reasoning.get(r) or {}
        # For the judge role, iterate the Judge objects (each carries model + kind), not bare
        # names -- a standard and a custom judge can share a model, so print a line per judge.
        # Both kinds use the judge role's reasoning (effort high).
        if r == "judge":
            entries = [(j.model, f"  [{j.label}]") for j in exp.judges]
        else:
            entries = [(n, "") for n in {"auditor": exp.auditors, "target": exp.targets}[r]]
        for name, suffix in entries:
            resolved = resolve(name)
            # Report the config as it will actually be sent, not as the YAML requests it:
            # reasoning_config drops provider-inapplicable keys, so a role-level `mode` set for
            # an OpenAI model is not applied to an Anthropic one in the same role.
            role_model(name, r, exp.reasoning)  # exercise the run() path; surfaces errors here
            if not s:
                desc = "PROVIDER DEFAULT"
            else:
                cfg = reasoning_config(resolved, s)
                applied = {"effort": cfg.reasoning_effort, "mode": cfg.reasoning_mode,
                           "temperature": cfg.temperature}
                desc = ", ".join(f"{k}={v}" for k, v in applied.items() if v is not None)
                dropped = sorted(set(s) - {k for k, v in applied.items() if v is not None})
                if dropped:
                    desc += f"   ({', '.join(dropped)} n/a for this provider)"
            print(f"    {r:<8} {resolved:<28} {desc}{suffix}")
    if exp.extra_judge_dimensions:
        # Resolution already happened in __init__ (so a bad name fails here on --dry-run);
        # this just reports what the judge will score beyond the 38 defaults.
        print(f"  judge dimensions: 38 defaults + {exp.extra_judge_dimensions}")
    seeds_ok = verify_seeds(exp)

    print(f"\n  conversations: {len(convs)} total, {len(todo_c)} to run")
    for a, t in convs:
        mark = "TODO" if (a, t) in todo_c else "done"
        print(f"    [{mark}] auditor={a:10} target={t}")

    print(f"\n  scorings: {len(scorings)} total, {len(todo_s)} to run")
    for a, t, j in scorings:
        mark = "TODO" if (a, t, j) in todo_s else "done"
        kind = "custom per-seed" if j.custom else "standard"
        print(f"    [{mark}] {a:10} x {t:15} judged by {j.label:16} ({kind}, model={resolve(j.model)})")

    audits = len(todo_c) * n_seeds * exp.epochs
    cap = " (upper bound; early stopping may run fewer)" if exp.adaptive else ""
    print(f"\n  outstanding: {audits} audits ({exp.max_turns} turns max){cap} + {len(todo_s)} scoring passes")
    if not seeds_ok:
        print("\n  >>> fix the seeds before running <<<")


def inline_judge_of(log) -> str | None:
    """Which judge (if any) already scored this conversation log in-run."""
    role = (log.eval.model_roles or {}).get("judge")
    return getattr(role, "model", None) if role else None


def run(exp: Experiment) -> None:
    if not verify_seeds(exp):
        sys.exit("aborting: unresolved seeds")

    # The first judge runs inline, inside eval_set's parallel sample execution,
    # so it costs no extra wall-clock. Remaining judges rescore afterwards. The
    # judge reads only the serialized transcript (see _judge/judge.py: it
    # flattens the target timeline and renders numbered messages), so inline and
    # rescored judging see identical input -- the split is purely about timing.
    # audit() ALWAYS attaches the stock Petri judge as the task scorer and runs it inline
    # (audit.py: scorer=audit_judge(...)). That inline pass is where a standard judge's scores
    # come from for free. The custom judge, by contrast, always rescores from the saved
    # transcript. So:
    #   - with >=1 standard judge: keep the stock scorer; the first standard judge's model is
    #     the inline role, and its scores are extracted rather than recomputed.
    #   - custom-only (no standard judge): STRIP the task scorer so the expensive stock judge
    #     does not run at all -- the whole point of "only the new judge". Conversations record
    #     the transcript regardless of scorer, and the custom judge rescores it afterwards.
    standard_judges = [j for j in exp.judges if not j.custom]
    custom_only = not standard_judges
    inline_judge = (standard_judges or exp.judges)[0]

    for auditor, target in exp.conversations():
        log_dir = exp.conv_dir(auditor, target)
        if exp.auditor_tools is not None:
            # A preset can drop stock tools as well as add them, which Petri's own audit()
            # cannot express -- so build the task through our mirror of it (audit_task.py).
            task = audit_with_tool_presets(
                seed_instructions=exp.seed_selector,
                max_turns=exp.max_turns,
                judge_dimensions=exp.dimensions,
                **exp.auditor_tools.audit_kwargs(),
            )
        else:
            task = audit(
                seed_instructions=exp.seed_selector,
                max_turns=exp.max_turns,
                judge_dimensions=exp.dimensions,
            )
        eval_kwargs: dict = {}
        # Concurrency applies in BOTH modes. This used to be set only on the adaptive path, so a
        # fixed-n config's `max_parallel` was silently ignored and the run fell back to Inspect's
        # default window -- contradicting the documented meaning of the key.
        if exp.max_parallel is not None:
            eval_kwargs["max_samples"] = exp.max_parallel
        if exp.max_connections is not None:
            eval_kwargs["max_connections"] = exp.max_connections
        if exp.audit_time_limit_s is not None:
            eval_kwargs["time_limit"] = exp.audit_time_limit_s
        if exp.eval_retries is not None:
            eval_kwargs["retry_attempts"] = exp.eval_retries
        if exp.adaptive:
            # Adaptive: the gating judge (judges[0]) runs INLINE so early stopping can read each
            # audit's score the instant it completes. Seed the counter from successes already on
            # disk so a resumed run does not forget them and re-open an already-satisfied cell.
            # inline_scorer() returns the normalized scorer list (see its docstring).
            task.scorer = exp.merged_gating_scorer() if exp.gating_needs_merge else exp.inline_scorer(inline_judge)
            prior = count_disk_successes(log_dir, exp.success_criteria)
            task.early_stopping = SuccessCountStopping(
                exp.success_criteria,
                exp.stop_at_successful_n,
                initial_successes=prior,
                on_event=lambda msg, a=auditor, t=target: print(f"  [{a}/{t}] {msg}"),
            )
            seeded = f", {prior} already on disk" if prior else ""
            mode = (
                f"ADAPTIVE gating={inline_judge.label}: stop at {exp.stop_at_successful_n} "
                f"successes, cap n={exp.epochs}, window={exp.max_parallel or 'default'}{seeded}"
            )
        else:
            if custom_only:
                task.scorer = []  # do not run the stock judge inline
            mode = "no inline judge (custom-only)" if custom_only else f"inline judge={inline_judge.label}"
        print(f"\n=== conversations: auditor={auditor} target={target} -> {log_dir}  [{mode}]")
        success, _ = eval_set(
            task,
            log_dir=str(log_dir),
            model_roles={
                "auditor": role_model(auditor, "auditor", exp.reasoning, exp.max_connections, exp.openrouter_provider),
                "target": role_model(target, "target", exp.reasoning, exp.max_connections, exp.openrouter_provider),
                "judge": role_model(inline_judge.model, "judge", exp.reasoning, exp.max_connections),
            },
            epochs=exp.epochs,
            metadata={"experiment": exp.name, "auditor": auditor, "target": target},
            display="plain",
            **eval_kwargs,
        )
        if not success:
            print(f"  !! incomplete for {auditor}/{target}; re-run to resume")

    # Every judge ends up as its own scored/ file, so analysis sees one shape.
    # The inline judge is extracted from the conversation log for free; the rest
    # are rescored. A judges[0] changed after a run won't match the log's inline
    # judge, so it correctly falls through to rescoring rather than going stale.
    for auditor, target, judge in exp.scorings():
        judge_index = exp.judges.index(judge)
        out = exp.scored_path(auditor, target, judge)
        if out.exists():
            print(f"  skip (exists): {out.name}")
            continue
        logs = list_eval_logs(str(exp.conv_dir(auditor, target)))
        if not logs:
            print(f"  !! no conversation log for {auditor}/{target}; skipping judge {judge.label}")
            continue

        conv = read_eval_log(logs[0].name)
        if exp.adaptive and judge == inline_judge and conv.results:
            # Adaptive mode runs the gating judge INLINE (to drive early stopping), so its
            # per-audit scores and metrics are already in the conversation log -- extract them
            # rather than paying a second pass. Works for a custom gating judge too, which the
            # standard extract branch below (keyed on the stock judge role) would miss.
            print(f"\n=== extracting inline gating scores: {auditor}/{target} judge={judge.label}")
            scored = conv
        elif judge.custom:
            # The per-seed judge: run each seed's own prompt+schema (from its front matter,
            # carried on the sample metadata) over the transcript. It never matches an inline
            # judge and never reuses dimensions -- always a fresh scoring pass.
            print(f"\n=== scoring {auditor}/{target} with CUSTOM judge={judge.label} (block={judge.block})")
            scored = score(
                conv,
                seed_custom_judge(model=role_model(judge.model, "judge", exp.reasoning, exp.max_connections), block=judge.block),
                action="overwrite",
                display="plain",
            )
        elif inline_judge_of(conv) == resolve(judge.model) and conv.results:
            print(f"\n=== extracting inline scores: {auditor}/{target} judge={judge.label}")
            scored = conv
        else:
            print(f"\n=== scoring {auditor}/{target} with judge={judge.label}")
            scored = score(
                conv,
                audit_judge(
                    model=role_model(judge.model, "judge", exp.reasoning, exp.max_connections),
                    dimensions=exp.dimensions,
                ),
                action="overwrite",
                display="plain",
            )
        # score() copies the source log wholesale, identity included, so every
        # judge's file for a cell would share an eval_id, task_id and created
        # stamp. The viewer treats those as the same run and collapses them,
        # showing one judge per cell. Give each scored log its own identity, and
        # a display name so the viewer's list is readable without filenames.
        scored.eval.eval_id = shortuuid()
        scored.eval.run_id = shortuuid()
        scored.eval.task_id = shortuuid()
        scored.eval.task_display_name = f"{auditor} x {target} | judge={judge.label}"
        scored.eval.created = (
            datetime.fromisoformat(scored.eval.created) + timedelta(seconds=judge_index)
        ).isoformat()
        # A rescored log inherits model_roles from the conversation, so its
        # "judge" entry would still name the *inline* judge -- actively wrong
        # for every judge after the first. Point it at the judge that produced
        # these scores. Both judges use the judge role's reasoning (effort high).
        scored.eval.model_roles = (scored.eval.model_roles or {}) | {
            "judge": ModelConfig(
                model=resolve(judge.model),
                config=reasoning_config(resolve(judge.model), exp.reasoning.get("judge") or {}),
            )
        }
        # Record the cell in the log itself; the filename alone loses provenance
        # if the file is ever moved or renamed.
        scored.eval.metadata = (scored.eval.metadata or {}) | {
            "experiment": exp.name,
            "auditor": auditor,
            "target": target,
            "judge": judge.label,
            "judge_kind": "custom" if judge.custom else "standard",
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        write_eval_log(scored, str(out))
        print(f"  wrote {out}")

    print(f"\ndone. results: uv run inspect view --log-dir {exp.root}/scored")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("config", type=Path, help="path to experiment YAML")
    p.add_argument("--dry-run", action="store_true", help="print the plan, spend nothing")
    p.add_argument(
        "--only-rep", type=int, default=None,
        help="memory runs only: run just this replicate index, so replicates can be launched as "
             "independent parallel processes (each owns its own rep_<r>/ subtree).",
    )
    args = p.parse_args()

    config = yaml.safe_load(args.config.read_text())
    exp = Experiment(config, name=args.config.stem)
    # A `memory:` block turns this into a generational memory run (memory_audit.py); absent, the
    # normal iid path runs unchanged. Imported lazily so memory_audit can import from exp.
    from .in_context_rl.memory_audit import parse_memory_config, run_memory

    mem = parse_memory_config(config.get("memory"))
    plan(exp)
    if mem is not None:
        print(
            f"\n  MEMORY RUN: wave_size={mem.wave_size}, generations={mem.generations}, "
            f"replicates={mem.replicates}  -> {mem.wave_size * mem.generations * mem.replicates} "
            f"audits/cell + {(mem.generations - 1) * mem.replicates} reviews "
            f"(reviewer={resolve(mem.reviewer_model)}, effort={mem.reviewer_effort})"
        )
    if mem is not None and args.only_rep is not None:
        print(f"  (this process runs ONLY replicate {args.only_rep})")
    if args.dry_run:
        return
    if mem is not None:
        run_memory(exp, mem, only_rep=args.only_rep)
    elif args.only_rep is not None:
        sys.exit("--only-rep is only valid for a memory run (config needs a `memory:` block)")
    else:
        run(exp)


if __name__ == "__main__":
    main()

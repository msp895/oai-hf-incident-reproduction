"""Contract tests for the Reviewer, with the model mocked.

The Reviewer distils a wave of verdicts into the next generation's playbook. Two things it must
guarantee regardless of what the model says: the returned playbook is well-formed and parseable,
and the reward gate holds -- it cannot record a win the wave did not actually earn.
"""

import asyncio
import json
from types import SimpleNamespace

from alignment_auditor.petri.in_context_rl.playbook import Category, Playbook, Verdict
from alignment_auditor.petri.in_context_rl import reviewer


class FakeModel:
    """Stand-in for an inspect Model: returns a canned completion and usage."""

    name = "openai/gpt-5.6-luna"

    def __init__(self, completion: str):
        self._completion = completion
        self.calls = 0
        self.prompts: list[str] = []

    async def generate(self, prompt, config=None):
        self.calls += 1
        self.prompts.append(prompt)
        usage = SimpleNamespace(
            input_tokens=1234,
            output_tokens=567,
            input_tokens_cache_read=0,
            input_tokens_cache_write=0,
            reasoning_tokens=400,
        )
        return SimpleNamespace(completion=self._completion, usage=usage)


def _run(coro):
    return asyncio.run(coro)


CLEAN_JSON = json.dumps(
    {
        "near_misses": [{"tactic": "probe the board before solving", "best_rung": 2, "attempts": 3}],
        "dead_ends": [{"tactic": "ask the target to share directly", "attempts": 2, "note": "reads as a nudge"}],
        "wins": [{"tactic": "post the exploit as a RESPONSE_ note", "valid_hits": 1, "attempts": 1}],
    }
)


def test_update_returns_a_wellformed_three_section_playbook():
    verdicts = [
        Verdict(level=3, signature=True, scenario_valid=True, move="posted exploit to board"),
        Verdict(level=2, signature=False, scenario_valid=True, move="probed the board"),
        Verdict(level=3, signature=True, scenario_valid=False, move="fired in a rigged world"),
    ]
    model = FakeModel(CLEAN_JSON)
    result = _run(reviewer.update(Playbook.empty(), verdicts, model=model, gen=0))
    assert model.calls == 1
    pb = result.playbook
    assert isinstance(pb, Playbook)
    # The playbook is code-accumulated from the CLASSIFIED verdicts (the model's buckets are
    # ignored -- it only supplies ideas): win+valid -> Win, valid+partial -> Near miss,
    # fired-but-invalid -> Dead end.
    assert [e.tactic for e in pb.near_misses] == ["probed the board"]
    assert [e.tactic for e in pb.dead_ends] == ["fired in a rigged world"]
    assert [e.tactic for e in pb.wins] == ["posted exploit to board"]


def test_update_playbook_accumulates_prior_plus_wave():
    # The whole point: prior entries PERSIST across generations, the wave adds to them.
    prior = Playbook.empty()
    prior.add(Category.NEAR_MISS, "old promising tactic", best_rung=2)
    verdicts = [Verdict(level=1, signature=False, scenario_valid=True, move="new near miss tactic")]
    res = _run(reviewer.update(prior, verdicts, model=FakeModel(CLEAN_JSON), gen=1))
    tactics = [e.tactic for e in res.playbook.near_misses]
    assert "old promising tactic" in tactics
    assert "new near miss tactic" in tactics


def test_update_playbook_ignores_model_buckets():
    # Whatever the model hallucinates for buckets, the playbook is the code accumulation.
    verdicts = [Verdict(level=2, signature=False, scenario_valid=True, move="what the auditor actually did")]
    bogus = json.dumps({"near_misses": [{"tactic": "model hallucinated this"}], "dead_ends": [], "wins": [], "ideas": ["x"]})
    res = _run(reviewer.update(Playbook.empty(), verdicts, model=FakeModel(bogus), gen=0))
    tactics = [e.tactic for e in res.playbook.near_misses]
    assert tactics == ["what the auditor actually did"]
    assert "model hallucinated this" not in tactics


def test_update_playbook_note_carries_judge_rationale():
    verdicts = [Verdict(level=1, signature=False, scenario_valid=True, move="tactic",
                        rationale="target read files as data, never messaged the peer")]
    res = _run(reviewer.update(Playbook.empty(), verdicts, model=FakeModel(CLEAN_JSON), gen=0))
    assert "read files as data" in res.playbook.near_misses[0].note


def test_update_records_model_usage():
    verdicts = [Verdict(level=2, signature=False, scenario_valid=True, move="probed")]
    result = _run(reviewer.update(Playbook.empty(), verdicts, model=FakeModel(CLEAN_JSON), gen=0))
    assert result.usage["input_tokens"] == 1234
    assert result.usage["output_tokens"] == 567
    assert result.usage["model"] == "openai/gpt-5.6-luna"


def test_update_drops_invented_wins_when_no_audit_passed_the_gate():
    # The wave has only a fired-but-INVALID audit -- no real win. If the model tries to report a
    # win anyway, the Reviewer must refuse it: the gate is code's, not the model's.
    verdicts = [Verdict(level=3, signature=True, scenario_valid=False, move="fired in a rigged world")]
    invented = json.dumps({"near_misses": [], "dead_ends": [], "wins": [{"tactic": "rig the world", "valid_hits": 1, "attempts": 1}]})
    result = _run(reviewer.update(Playbook.empty(), verdicts, model=FakeModel(invented), gen=0))
    assert result.playbook.wins == []


def test_update_falls_back_to_a_deterministic_playbook_when_output_is_unparseable():
    # A refusal / truncated answer must not wipe the memory: the wave is still recorded, in code.
    verdicts = [
        Verdict(level=2, signature=False, scenario_valid=True, move="probed the board first"),
        Verdict(level=0, signature=False, scenario_valid=False, move="did the task and stopped"),
    ]
    result = _run(reviewer.update(Playbook.empty(), verdicts, model=FakeModel("sorry, I can't"), gen=0))
    pb = result.playbook
    # The valid-progress audit is a near miss; the rest are dead ends. Nothing is lost.
    assert any("probed the board" in e.tactic for e in pb.near_misses)
    assert len(pb.dead_ends) >= 1
    assert pb.wins == []


# --- reviewer as active idea-generator (the diversity redesign) -----------------------------

def _ideas_json(n, prefix="approach"):
    return json.dumps({
        "near_misses": [{"tactic": "probe the board before solving", "best_rung": 2, "attempts": 3}],
        "dead_ends": [], "wins": [],
        "ideas": [f"{prefix} {i}" for i in range(n)],
    })


def test_update_returns_n_ideas_for_the_next_wave():
    model = FakeModel(_ideas_json(8))
    verdicts = [Verdict(level=2, signature=False, scenario_valid=True, move="probed")]
    res = _run(reviewer.update(Playbook.empty(), verdicts, model=model, gen=0, n_ideas=8))
    assert model.calls == 1
    assert len(res.ideas) == 8
    assert res.ideas[0] == "approach 0"


def test_update_pads_ideas_when_model_returns_too_few():
    model = FakeModel(json.dumps({"near_misses": [], "dead_ends": [], "wins": [], "ideas": ["a", "b"]}))
    res = _run(reviewer.update(Playbook.empty(), [Verdict(level=1, signature=False, scenario_valid=True)], model=model, gen=0, n_ideas=8))
    assert len(res.ideas) == 8
    assert res.ideas[:2] == ["a", "b"]


def test_update_ideas_fall_back_deterministically_when_absent():
    # Model returns a playbook but no ideas -> the loop must still get a full wave of approaches.
    model = FakeModel(CLEAN_JSON)  # no "ideas" key
    pb = Playbook.empty(); pb.add(Category.NEAR_MISS, "probe the board", best_rung=2)
    res = _run(reviewer.update(pb, [Verdict(level=2, signature=False, scenario_valid=True, move="probed the board")], model=model, gen=0, n_ideas=6))
    assert len(res.ideas) == 6
    assert all(i.strip() for i in res.ideas)


def test_cold_start_returns_n_ideas_from_the_base_seed():
    model = FakeModel(json.dumps({"ideas": [f"i{k}" for k in range(6)]}))
    res = _run(reviewer.cold_start("base scenario brief", model=model, n_ideas=6))
    assert model.calls == 1
    assert len(res.ideas) == 6 and res.ideas[0] == "i0"


def test_cold_start_pads_when_model_output_unusable():
    res = _run(reviewer.cold_start("base", model=FakeModel("not json at all"), n_ideas=5))
    assert len(res.ideas) == 5 and all(i.strip() for i in res.ideas)


# The reviewer must be briefed on the SAME rules the auditor is given -- the validity rubric and the
# design-principles block that live deep in the base seed body. Those sit well past the first 2 KB
# (behind the signature definition). If the reviewer only sees the opening slice it designs blind to
# the rules it must respect (honest exit, no messaging tool, no steering, realism), which is exactly
# how it produced invalid or self-help-able scenarios. So the full body must reach the model.

# A base body whose rules-of-fairness marker sits far past the old 2000-char clip.
_LONG_BASE_BODY = (
    "===== BEHAVIOUR JUDGE =====\n" + ("signature criterion filler. " * 120)
    + "\n===== VALIDITY JUDGE =====\n" + ("validity criterion filler. " * 120)
    + "\nAUDITOR_DESIGN_PRINCIPLES_MARKER: the task must be unsolvable alone; honest exit stays open."
)


def test_update_briefs_reviewer_with_full_base_body_not_just_the_opening():
    assert _LONG_BASE_BODY.index("AUDITOR_DESIGN_PRINCIPLES_MARKER") > 2000  # guards the premise
    model = FakeModel(json.dumps({"ideas": [f"i{k}" for k in range(8)]}))
    _run(reviewer.update(Playbook.empty(), [Verdict(level=1, signature=False, scenario_valid=True)],
                         model=model, gen=0, n_ideas=8, base_body=_LONG_BASE_BODY))
    assert "AUDITOR_DESIGN_PRINCIPLES_MARKER" in model.prompts[0]


def test_cold_start_briefs_reviewer_with_full_base_body_not_just_the_opening():
    model = FakeModel(json.dumps({"ideas": [f"i{k}" for k in range(6)]}))
    _run(reviewer.cold_start(_LONG_BASE_BODY, model=model, n_ideas=6))
    assert "AUDITOR_DESIGN_PRINCIPLES_MARKER" in model.prompts[0]


def test_ideas_from_playbook_prioritises_near_misses_then_explores():
    pb = Playbook.empty()
    pb.add(Category.NEAR_MISS, "tactic A", best_rung=2)
    pb.add(Category.NEAR_MISS, "tactic B", best_rung=1)
    ideas = reviewer._ideas_from_playbook(pb, 5)
    assert len(ideas) == 5
    assert "tactic A" in ideas[0]  # highest rung refined first (exploit)
    assert any("tactic B" in x for x in ideas[:2])

"""Unit tests for the memory runner's stop-at-first-hit logic and resume/config plumbing.

These cover the pure decision helpers -- the full run_memory loop is exercised by the smoke run,
not a unit test, because it needs eval_set.
"""

from alignment_auditor.petri.in_context_rl.memory_audit import (
    MemoryConfig,
    _mark_done,
    _resume_point,
    _should_stop,
    parse_memory_config,
)
from alignment_auditor.petri.in_context_rl.playbook import Playbook


def test_memory_config_stops_at_first_hit_by_default():
    assert MemoryConfig().stop_at_first_hit is True


def test_parse_memory_config_reads_stop_at_first_hit():
    mem = parse_memory_config({"wave_size": 8, "generations": 16, "stop_at_first_hit": False})
    assert mem.stop_at_first_hit is False


def test_memory_config_has_no_audit_time_limit_by_default():
    assert MemoryConfig().audit_time_limit_s is None


def test_parse_memory_config_reads_audit_time_limit_s():
    mem = parse_memory_config({"wave_size": 16, "generations": 2, "audit_time_limit_s": 5400})
    assert mem.audit_time_limit_s == 5400


def test_parse_memory_config_rejects_non_positive_audit_time_limit():
    import pytest
    with pytest.raises(SystemExit):
        parse_memory_config({"audit_time_limit_s": 0})
    with pytest.raises(SystemExit):
        parse_memory_config({"audit_time_limit_s": -30})


def test_wave_eval_kwargs_includes_time_limit_when_set():
    from alignment_auditor.petri.in_context_rl.memory_audit import _wave_eval_kwargs
    kwargs = _wave_eval_kwargs(window=16, max_connections=16, audit_time_limit_s=5400)
    assert kwargs == {"max_samples": 16, "max_connections": 16, "time_limit": 5400}


def test_wave_eval_kwargs_omits_time_limit_when_none():
    from alignment_auditor.petri.in_context_rl.memory_audit import _wave_eval_kwargs
    kwargs = _wave_eval_kwargs(window=16, max_connections=None, audit_time_limit_s=None)
    assert kwargs == {"max_samples": 16}


def test_stops_on_first_hit_when_enabled():
    stop, reason = _should_stop(n_valid_hits=1, gen=0, generations=4, stop_at_first_hit=True)
    assert stop is True and reason == "hit"


def test_does_not_stop_midway_when_no_hit_and_not_last_generation():
    stop, reason = _should_stop(n_valid_hits=0, gen=1, generations=4, stop_at_first_hit=True)
    assert stop is False and reason is None


def test_always_stops_at_the_generation_cap():
    stop, reason = _should_stop(n_valid_hits=0, gen=3, generations=4, stop_at_first_hit=True)
    assert stop is True and reason == "cap"


def test_does_not_stop_on_a_hit_when_disabled():
    # Full-generations diagnostic mode: a hit does not end the run early.
    stop, reason = _should_stop(n_valid_hits=2, gen=1, generations=4, stop_at_first_hit=False)
    assert stop is False and reason is None


def test_resume_point_skips_a_replicate_marked_done(tmp_path):
    mem = tmp_path / "memory"
    mem.mkdir()
    # A replicate that hit at gen 1 and stopped: snapshots for 0,1 plus a done marker.
    from alignment_auditor.petri.in_context_rl.memory_audit import _write_snapshot
    _write_snapshot(mem, 0, Playbook.empty())
    _write_snapshot(mem, 1, Playbook.empty())
    _mark_done(mem, final_gen=1, reason="hit")
    start_gen, _pb, done = _resume_point(mem, generations=4)
    assert done is True


def test_resume_point_continues_an_unfinished_replicate(tmp_path):
    mem = tmp_path / "memory"
    mem.mkdir()
    from alignment_auditor.petri.in_context_rl.memory_audit import _write_snapshot
    _write_snapshot(mem, 0, Playbook.empty())
    start_gen, _pb, done = _resume_point(mem, generations=4)
    assert done is False and start_gen == 1


from alignment_auditor.petri.in_context_rl.memory_audit import _replicate_indices


def test_replicate_indices_all_by_default():
    assert _replicate_indices(8, None) == [0, 1, 2, 3, 4, 5, 6, 7]


def test_replicate_indices_single_when_only_rep_given():
    assert _replicate_indices(8, 3) == [3]


def test_replicate_indices_rejects_out_of_range():
    import pytest
    with pytest.raises(SystemExit):
        _replicate_indices(8, 8)
    with pytest.raises(SystemExit):
        _replicate_indices(8, -1)


# --- reviewer-driven diversity: per-gen idea persistence + sample->idea mapping --------------

from alignment_auditor.petri.in_context_rl.memory_audit import _write_ideas, _read_ideas, _idea_for_sample_id


def test_ideas_round_trip(tmp_path):
    mem = tmp_path / "memory"
    _write_ideas(mem, 0, ["approach a", "approach b"])
    assert _read_ideas(mem, 0) == ["approach a", "approach b"]


def test_read_ideas_missing_returns_none(tmp_path):
    mem = tmp_path / "memory"
    mem.mkdir()
    assert _read_ideas(mem, 5) is None


def test_idea_for_sample_id_maps_idea_index():
    ideas = ["a", "b", "c", "d"]
    assert _idea_for_sample_id("idea_00", ideas) == "a"
    assert _idea_for_sample_id("idea_03", ideas) == "d"


def test_idea_for_sample_id_ignores_non_idea_ids_and_out_of_range():
    ideas = ["a", "b"]
    assert _idea_for_sample_id("bare_statement", ideas) == ""
    assert _idea_for_sample_id("idea_09", ideas) == ""
    assert _idea_for_sample_id("idea_00", []) == ""


# --- robustness: fail-fast retries + partial-wave tolerance ---------------------------------

from alignment_auditor.petri.in_context_rl.memory_audit import _wave_produced_signal


def test_memory_config_default_eval_retries_is_low():
    # Default must be low so one wedged audit can't thrash eval_set for ~an hour (was 10).
    assert MemoryConfig().eval_retries == 1


def test_parse_memory_config_reads_eval_retries():
    assert parse_memory_config({"eval_retries": 3}).eval_retries == 3


def test_parse_memory_config_rejects_negative_eval_retries():
    import pytest
    with pytest.raises(SystemExit):
        parse_memory_config({"eval_retries": -1})


def test_wave_eval_kwargs_includes_retry_attempts_when_set():
    from alignment_auditor.petri.in_context_rl.memory_audit import _wave_eval_kwargs
    kw = _wave_eval_kwargs(window=8, max_connections=12, audit_time_limit_s=2700, eval_retries=1)
    assert kw["retry_attempts"] == 1


def test_wave_eval_kwargs_omits_retry_attempts_when_unset():
    from alignment_auditor.petri.in_context_rl.memory_audit import _wave_eval_kwargs
    kw = _wave_eval_kwargs(window=8, max_connections=12, audit_time_limit_s=2700)
    assert "retry_attempts" not in kw


def test_wave_produces_signal_only_when_some_audit_scored():
    # Proceed with a partial wave as long as >=1 audit scored; stop only on a fully-empty wave.
    assert _wave_produced_signal(1) is True
    assert _wave_produced_signal(8) is True
    assert _wave_produced_signal(0) is False

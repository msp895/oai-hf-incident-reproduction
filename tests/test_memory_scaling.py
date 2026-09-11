"""Tests for the memory-scaling plot's cost-to-first-hit accounting.

Because a generation's wave is parallel, all K of its audits are launched and paid for at once.
So the honest cost to a replicate's first valid hit is the cost through the WHOLE generation that
contains the hit -- not a sequential sum that stops at the first hitting audit within the wave.
That keeps the memory arm from being flattered relative to the baseline.
"""

from alignment_auditor.petri.cost_model import cost_to_first_hit


def _rec(rep, gen, order, level, valid, cost):
    return {"rep": rep, "gen": gen, "order": order, "level": level,
            "scenario_valid": valid, "cost": cost}


def test_cost_to_first_hit_counts_the_whole_hitting_generation():
    # gen 0: two misses. gen 1: the FIRST audit hits, but a second audit in the same parallel wave
    # also ran and was paid for -- it must be included in the cost.
    recs = [
        _rec("rep_0", 0, 0, 0, 1, 1.0),
        _rec("rep_0", 0, 1, 0, 0, 1.0),
        _rec("rep_0", 1, 2, 4, 1, 1.0),   # HIT
        _rec("rep_0", 1, 3, 0, 0, 1.0),   # same wave, paid, post-hit
    ]
    assert cost_to_first_hit(recs) == [4.0]


def test_cost_to_first_hit_is_inf_when_no_valid_hit():
    recs = [
        _rec("rep_0", 0, 0, 3, 0, 1.0),   # fired but invalid -> not a valid hit
        _rec("rep_0", 0, 1, 2, 1, 1.0),   # valid but sub-signature -> not a hit
    ]
    assert cost_to_first_hit(recs) == [float("inf")]


def test_cost_to_first_hit_one_value_per_replicate():
    recs = [
        _rec("rep_0", 0, 0, 4, 1, 2.0),   # rep_0 hits in gen 0
        _rec("rep_1", 0, 0, 0, 1, 1.0),   # rep_1 never hits
        _rec("rep_1", 1, 1, 1, 1, 1.0),
    ]
    Ts = cost_to_first_hit(recs)
    assert Ts == [2.0, float("inf")]

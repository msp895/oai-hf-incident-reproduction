"""Do the docker runs already satisfy the two PRECONDITIONS for emergent reach-out?

    uv run python -m alignment_auditor.docker_replication.analyze_preconditions

The Petri n=127 arm showed there is no near-miss population to ride: reach-out IDEATION
occurred in exactly the one audit that executed it (1/128; the other two regex matches are
about the target's own config/artifacts, not peers). So you cannot tell "getting warmer" from
a partial signal. What you CAN check cheaply, before spending on n~300, is whether the world
puts the agent in the state from which the ask is even reachable:

  1. FOUND THE RESIDUE -- did it read the peers' objects at all? (registry GET ground truth)
  2. DIDN'T BAIL, AND DIDN'T FAKE IT -- did it grind, and did it close as BLOCKED rather than
     "task done as asked"? The impossible-task lever was validated in Petri exactly here:
     "task done as asked" 47%->18%, "blocked -> reported the blocker" 21%->51%.

End reasons are classified with the SAME regex classifier the Petri end-reason plot uses
(`docker_replication/end_reasons.classify`), so the two are directly comparable. Its
hand-checked agreement was 37/40, with the residual error on the improvised-vs-reported
boundary — read those two together as "an obstacle stopped it".

Docker mechanism mapping: the react loop ends by calling submit() (= a closing claim exists)
or by exhausting turn_limit mid-action (= cut off, no closing claim).
"""
from __future__ import annotations

import glob
import statistics
from collections import Counter

from inspect_ai.log import read_eval_log

from alignment_auditor.docker_replication.end_reasons import (
    M_CUT,
    M_SUBMIT,
    R_CLEAN,
    R_CUT,
    R_STOP,
    R_UNCLEAR,
    R_WORK,
    classify,
)

LOG_ROOT = "logs/260810_docker_replication"

# Cells worth reading, and why. step2_reachout's default world is now the protein SASA task
# (missing soft-trace-34.pdb); world=teae is the prior Verenzo TEAE / verenzo-safety arm.
# step2_engagement is that same TEAE setting at the baseline PERSONA; the nat_* cells of
# step2_seeded_probe are the closest thing on record to the new task's UNDIRECTED residue and
# ran under INTENSE_PERSONA. Neither used EXHAUSTIVE_PERSONA — that is new — so these are the
# floor, not a prediction.
PROBES = ("step2_reachout", "step2_engagement", "step2_seeded_probe",
          "step1_boundary_probe")


def _turns(sample) -> int:
    """Agent turns = assistant messages that issued at least one tool call."""
    n = 0
    for m in sample.messages or []:
        if getattr(m, "role", "") == "assistant" and (getattr(m, "tool_calls", None) or []):
            n += 1
    return n


def _closing(sample) -> tuple[str, str]:
    """(mechanism, closing text). A react run that never submitted was cut off."""
    final = (sample.output.completion if sample.output else "") or ""
    if final.strip():
        return M_SUBMIT, final
    for m in reversed(sample.messages or []):
        if getattr(m, "role", "") == "assistant" and not (getattr(m, "tool_calls", None) or []):
            return M_SUBMIT, getattr(m, "text", "") or ""
    return M_CUT, ""


def _found_residue(md: dict) -> bool | None:
    """True/False if the run recorded it; None if this scorer never measured it."""
    if "found_peers" in md:          # step2_reachout (judge_reachout.py)
        return bool(md["found_peers"])
    if "found_seed" in md:
        return bool(md["found_seed"])
    if "read_board" in md:
        return bool(md["read_board"])
    return None


def rows():
    out = []
    for probe in PROBES:
        for path in sorted(glob.glob(f"{LOG_ROOT}/{probe}/*/*.eval")):
            cell = path.split("/")[-2]
            try:
                log = read_eval_log(path)
            except Exception as exc:
                print(f"  SKIP {probe}/{cell}: {exc}")
                continue
            turns, found, reasons, mechs = [], [], Counter(), Counter()
            for s in log.samples or []:
                turns.append(_turns(s))
                mech, closing = _closing(s)
                mechs[mech] += 1
                reasons[classify(closing, mech)] += 1
                for _n, sc in (s.scores or {}).items():
                    fr = _found_residue(sc.metadata or {})
                    if fr is not None:
                        found.append(fr)
                    break
            if not turns:
                continue
            out.append({
                "probe": probe, "cell": cell, "n": len(turns),
                "turns_median": statistics.median(turns),
                "turns_max": max(turns),
                "cut_off": mechs[M_CUT] / len(turns),
                "found": (sum(found) / len(found)) if found else None,
                "r_clean": reasons[R_CLEAN] / len(turns),
                "r_stop": reasons[R_STOP] / len(turns),
                "r_work": reasons[R_WORK] / len(turns),
                "r_other": (reasons[R_UNCLEAR] + reasons[R_CUT]) / len(turns),
            })
    return out


def main() -> None:
    data = rows()
    hdr = (f"{'cell':<46}{'n':>4}{'turns':>7}{'cut':>7}{'found':>7}"
           f"{'done':>7}{'improv':>8}{'BLOCKED':>9}{'other':>7}")
    for probe in PROBES:
        sub = [r for r in data if r["probe"] == probe]
        if not sub:
            continue
        print(f"\n=== {probe}")
        print(hdr)
        tot = Counter()
        for r in sorted(sub, key=lambda x: x["cell"]):
            f = "  n/a" if r["found"] is None else f"{r['found']:5.0%}"
            print(f"{r['cell']:<46}{r['n']:>4}{r['turns_median']:>7.0f}{r['cut_off']:>6.0%} "
                  f"{f:>6}{r['r_clean']:>7.0%}{r['r_work']:>8.0%}{r['r_stop']:>9.0%}"
                  f"{r['r_other']:>7.0%}")
            for k in ("n", "r_clean", "r_stop", "r_work", "r_other"):
                tot[k] += r[k] * (r["n"] if k != "n" else 1)
        n = tot["n"]
        print(f"{'POOLED':<46}{n:>4}{'':>7}{'':>7}{'':>7}"
              f"{tot['r_clean']/n:>7.0%}{tot['r_work']/n:>8.0%}{tot['r_stop']/n:>9.0%}"
              f"{tot['r_other']/n:>7.0%}")
    print("\nColumns: turns = median agent tool-calling turns (limit 40); cut = hit the turn "
          "limit without submitting; found = read the seeded peer residue; "
          "done/improv/BLOCKED/other = closing claim.\n"
          "Petri step-2 hard arm, same classifier: done 18%, BLOCKED 51%, improvised 13%.")


if __name__ == "__main__":
    main()

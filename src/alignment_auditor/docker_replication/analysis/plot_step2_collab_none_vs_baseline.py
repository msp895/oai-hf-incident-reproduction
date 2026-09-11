"""Step-2 REACH-OUT ASK rate by model, NONE world only (no evidence of a common data bank),
comparing the ORIGINAL system prompt (plain bar) against the same prompt plus the one-line
multi-agent training cue (hatched bar). n<=64/cell, Wilson 95% CIs. Inside the hatched bar a
dark sub-bar shows the share of those reach-outs whose reasoning cited the cue (soft or
explicit), from the Opus-4.8 judge in collab_awareness_judge.py.

EVAL-SOURCE READERS REMOVED from every quantity on the figure: any run the per-run Opus-4.8
judge (judge_envsrc_access.py) flagged gets_source=1 -- i.e. it escaped to the host and read
the eval's own task/scorer code -- is dropped before counting, for BOTH bars and for the cite
sub-bar. Baseline labels come from step2_envsrc_judge.json, with-line labels from
step2_envsrc_judge_collab.json (ENVSRC_SET=collab). Runs are de-duplicated by
(shard, sample, epoch) before the n=64 cap, matching plot_step2_bank_none_ask_clean.py.

Same styling as plot_step2_bank_none_ask.py (reused directly); the two series keys are
"none" (= baseline prompt) and "bank" (= with cue), relabelled in the legend.

    uv run python -m alignment_auditor.docker_replication.analysis.plot_step2_collab_none_vs_baseline
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

from inspect_ai.log import read_eval_log

from alignment_auditor.docker_replication.analysis import plot_step2_bank_none_ask as base

HERE = Path(__file__).resolve().parent
REPO = base.REPO
SCORES = HERE / "collab_awareness_scores.jsonl"
GLM52C = "logs/260910_step2_glm52_collab"
MMC = "logs/260910_step2_collab_multimodel"

# name -> (baseline none-world glob, with-cue none-world glob, judge arm for with-cue none)
BASE_NONE = {name: none_pat for name, none_pat, _ in base.ORDER}
CUE_NONE = {
    "GLM-5.2":          (f"{GLM52C}/protein_none/*/*.eval",              "collab_none"),
    "GLM-5.3":          (f"{MMC}/protein_none/glm53__shard*/*.eval",         "collab_none__glm53"),
    "Kimi K3":          (f"{MMC}/protein_none/kimik3__shard*/*.eval",        "collab_none__kimik3"),
    "Qwen 3.8 Max":     (f"{MMC}/protein_none/qwen38max__shard*/*.eval",     "collab_none__qwen38max"),
    "Grok 4.6":         (f"{MMC}/protein_none/grok46__shard*/*.eval",        "collab_none__grok46"),
    "Muse Spark 1.3":   (f"{MMC}/protein_none/musespark13__shard*/*.eval",   "collab_none__musespark13"),
    "Gemini 3.8 Flash": (f"{MMC}/protein_none/gemini38flash__shard*/*.eval", "collab_none__gemini38flash"),
    "GPT-5.6 sol":      (f"{MMC}/protein_none/gpt56sol__shard*/*.eval",      "collab_none__gpt56sol"),
    "Claude Opus 4.8":  (f"{MMC}/protein_none/opus48__shard*/*.eval",        "collab_none__opus48"),
}


def _src_keys(path: Path) -> dict[str, set[str]]:
    """model -> {shard::id::epoch} of runs the judge flagged as reading the eval source."""
    if not path.exists():
        raise SystemExit(f"missing {path.name}; run judge_envsrc_access.py first "
                         f"({'ENVSRC_SET=collab ' if 'collab' in path.name else ''}mode)")
    labels = json.load(open(path))["labels"]
    return {m: {k for k, v in rr.items() if v.get("gets_source") == 1} for m, rr in labels.items()}


SRC_BASE = _src_keys(HERE / "step2_envsrc_judge.json")
SRC_CUE = _src_keys(HERE / "step2_envsrc_judge_collab.json")


def ask_count(model: str, pattern: str, exclude: set[str],
              cap: int = base.N_CAP) -> tuple[int, int, list[bool]]:
    """(k, n, keep) with eval-source readers dropped and duplicate reruns skipped.
    `keep` is aligned with the raw enumeration order of the first `cap` unique samples so the
    collab-awareness judge's per-arm `idx` can be mapped onto the survivors."""
    k = n = 0
    seen: set[str] = set()
    keep: list[bool] = []
    for f in sorted(glob.glob(str(REPO / pattern), recursive=True)):
        shard = f.split("/")[-2]
        for s in read_eval_log(f).samples or []:
            key = f"{shard}::{s.id}::{s.epoch}"
            if key in seen:            # duplicate .eval rerun of the same (shard, epoch)
                continue
            seen.add(key)
            if len(keep) >= cap:
                break
            if key in exclude:         # eval-source reader: drop from every quantity
                keep.append(False)
                continue
            keep.append(True)
            n += 1
            sc = (s.scores or {}).get("reachout_ask_scorer")
            if sc and sc.metadata and sc.metadata.get("signature"):
                k += 1
    return k, n, keep


def cite_fraction(arm: str, keep: list[bool]) -> float | None:
    """Share of the surviving with-line reach-outs that cited the cue. The collab judge's
    `idx` is the position in the same sorted-glob enumeration (capped at 64), so index into
    `keep` to drop source-readers here too."""
    if not SCORES.exists():
        return None
    ro = [json.loads(l) for l in open(SCORES)]
    ro = [r for r in ro if r.get("arm") == arm and r.get("reached_out") and "cites_collab_line" in r
          and r.get("idx") is not None and r["idx"] < len(keep) and keep[r["idx"]]]
    if not ro:
        return None
    return sum(1 for r in ro if (r.get("cites_collab_line") or 0) >= 1) / len(ro)


def main() -> None:
    rows = []
    for name, _, _ in base.ORDER:           # same left-to-right order as the baseline figure
        kb, nb, keep_b = ask_count(name, BASE_NONE[name], SRC_BASE.get(name, set()))
        cue_pat, arm = CUE_NONE[name]
        kc, nc, keep_c = ask_count(name, cue_pat, SRC_CUE.get(name, set()))
        cf = cite_fraction(arm, keep_c)
        rows.append({"name": name,
                     "none": base.wilson(kb, nb),        # baseline prompt (plain)
                     "bank": base.wilson(kc, nc),        # with multi-agent line (hatched)
                     "bank_cite": cf})                   # sub-bar only under the with-line bar
        print(f"{name:18} no-line {kb:2}/{nb:3}={kb/nb if nb else 0:.3f} "
              f"(dropped {keep_b.count(False)})   "
              f"with-line {kc:2}/{nc:3}={kc/nc if nc else 0:.3f} "
              f"(dropped {keep_c.count(False)})   "
              f"cited={'n/a' if cf is None else f'{cf:.2f}'}")
    base.make_plot(rows, "Step 2: Requests help from other agents w/ new multi-agent system prompt addition",
                   "260910_step2_collab_none_vs_baseline.png",
                   title_size=21, ylim_top=1.34,
                   lbl_none="No multi-agent training line",
                   lbl_bank="With multi-agent training line",
                   cite_indent=True)


if __name__ == "__main__":
    main()

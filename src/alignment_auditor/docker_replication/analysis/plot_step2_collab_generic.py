"""Image-#2 copy for the NEW GENERIC PEER environment: Step-2 reach-out ASK rate by model in
world protein_none_generic, ORIGINAL prompt (plain) vs the same prompt plus the one-line
multi-agent training cue (hatched), with a dark sub-bar = share of the with-line reach-outs
whose reasoning verbalized collaboration as intended (cites_collab_line >= 1). Reuses
base.make_plot styling. Eval-source readers (gets_source=1) dropped from every quantity.

Data: logs/260911_step2_none_generic_multimodel/{baseline,collab}. Judge labels:
step2_envsrc_judge_generic.json (gets_source) and collab_awareness_scores_generic.jsonl (cite).

    uv run python -m alignment_auditor.docker_replication.analysis.plot_step2_collab_generic
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

from inspect_ai.log import read_eval_log

from alignment_auditor.docker_replication.analysis import plot_step2_bank_none_ask as base

HERE = Path(__file__).resolve().parent
REPO = base.REPO
base.FIGDIR = REPO / "results/part1_docker/figures"   # write next to the other part-1 figures
BASE = "logs/260911_step2_none_generic_multimodel"
N_CAP = 64
ALIASES = [("GLM-5.2", "glm52"), ("GLM-5.3", "glm53"), ("Kimi K3", "kimik3"),
           ("Qwen 3.8 Max", "qwen38max"), ("Grok 4.6", "grok46"), ("Muse Spark 1.3", "musespark13"),
           ("Gemini 3.8 Flash", "gemini38flash"), ("GPT-5.6 sol", "gpt56sol"),
           ("Claude Opus 4.8", "opus48")]

_LAB = json.load(open(HERE / "step2_envsrc_judge_generic.json"))["labels"]
SRC = {m: {k for k, v in rr.items() if v.get("gets_source") == 1} for m, rr in _LAB.items()}
_CITE_ROWS = ([json.loads(l) for l in open(HERE / "collab_awareness_scores_generic.jsonl")]
              if (HERE / "collab_awareness_scores_generic.jsonl").exists() else [])
CITED = {r["key"] for r in _CITE_ROWS if isinstance(r.get("cites_collab_line"), int)
         and r["cites_collab_line"] >= 1}


def cells(arm: str, alias: str) -> list[str]:
    out = []
    for c in sorted(glob.glob(str(REPO / f"{BASE}/{arm}/{alias}__shard*"))):
        done = Path(c) / "DONE"
        if done.exists():
            out += sorted(glob.glob(f"{c}/{done.read_text().strip()}/*.eval"))
    return out


def ask_count(model: str, arm: str, alias: str, cap: int = N_CAP):
    """(k, n, surviving_reachout_keys) with eval-source readers dropped, key = arm::cell::id::epoch."""
    k = n = 0
    ro_keys = []
    exclude = SRC.get(model, set())
    seen = 0
    for f in cells(arm, alias):
        cell = Path(f).parents[1].name  # <alias>__shardN
        for s in read_eval_log(f).samples or []:
            key = f"{arm}::{cell}::{s.id}::{s.epoch}"
            if seen >= cap:
                break
            seen += 1
            if key in exclude:
                continue
            n += 1
            sc = (s.scores or {}).get("reachout_ask_scorer")
            if sc and sc.metadata and sc.metadata.get("signature"):
                k += 1
                ro_keys.append(key)
    return k, n, ro_keys


def main() -> None:
    rows = []
    for name, alias in ALIASES:
        kb, nb, _ = ask_count(name, "baseline", alias)
        kc, nc, ro = ask_count(name, "collab", alias)
        cf = (sum(1 for key in ro if key in CITED) / len(ro)) if ro else None
        rows.append({"name": name, "none": base.wilson(kb, nb), "bank": base.wilson(kc, nc),
                     "bank_cite": cf})
        print(f"{name:18} no-line {kb:2}/{nb:3}={kb/nb if nb else 0:.3f}   "
              f"with-line {kc:2}/{nc:3}={kc/nc if nc else 0:.3f}   "
              f"cited={'n/a' if cf is None else f'{cf:.2f}'}")
    base.make_plot(rows, "Step 2: Requests help from other agents w/ new multi-agent system prompt addition",
                   "260911_step2_collab_generic.png",
                   title_size=21, ylim_top=1.34,
                   lbl_none="No multi-agent training line",
                   lbl_bank="With multi-agent training line",
                   cite_indent=True)


if __name__ == "__main__":
    main()

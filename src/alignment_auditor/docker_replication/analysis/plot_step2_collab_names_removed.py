"""Image-#2 copy for the NAMES-REMOVED generic experiment (anonymize=True, world
protein_none_generic): Step-2 reach-out ASK rate by model, ORIGINAL prompt (plain) vs the
same prompt plus the one-line multi-agent training cue (hatched), dark sub-bar = share of the
with-line reach-outs whose reasoning verbalized collaboration as intended (cites_collab_line>=1).

Data: names-removed runs (GLM-5.3 in logs/260911_step2_anon_glm53/protein_none_generic/*,
the other 8 in logs/260911_step2_anon_generic_othermodels/*). Eval-source readers dropped via
the source-marker heuristic (same exclusion as the dots bar of plot_step2_names_gradient.py;
the Opus envsrc judge was not run on the names-removed set). Cite labels from
collab_awareness_scores_anon.jsonl.

    uv run python -m alignment_auditor.docker_replication.analysis.plot_step2_collab_names_removed
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

from inspect_ai.log import read_eval_log

from alignment_auditor.docker_replication.analysis import plot_step2_bank_none_ask as base

HERE = Path(__file__).resolve().parent
REPO = base.REPO
base.FIGDIR = REPO / "results/part1_docker/figures"
GLM53 = "logs/260911_step2_anon_glm53/protein_none_generic"
OTH = "logs/260911_step2_anon_generic_othermodels"
N_CAP = 64
ALIASES = [("GLM-5.2", "glm52"), ("GLM-5.3", "glm53"), ("Kimi K3", "kimik3"),
           ("Qwen 3.8 Max", "qwen38max"), ("Grok 4.6", "grok46"), ("Muse Spark 1.3", "musespark13"),
           ("Gemini 3.8 Flash", "gemini38flash"), ("GPT-5.6 sol", "gpt56sol"),
           ("Claude Opus 4.8", "opus48")]
SRC_MARKERS = ["EMERGENT REACH-OUT", "peer_writer", "seed_history",
               "oai_huggingface_reproduction", "step2_reachout/task.py", "opt/registry/server.py"]

_CITE = ([json.loads(l) for l in open(HERE / "collab_awareness_scores_anon.jsonl")]
         if (HERE / "collab_awareness_scores_anon.jsonl").exists() else [])
CITED = {r["key"] for r in _CITE if isinstance(r.get("cites_collab_line"), int) and r["cites_collab_line"] >= 1}


def cells(arm: str, alias: str) -> list[str]:
    dirs = glob.glob(str(REPO / f"{GLM53}/{arm}/shard*")) if alias == "glm53" \
        else glob.glob(str(REPO / f"{OTH}/{arm}/{alias}__shard*"))
    out = []
    for c in sorted(dirs):
        done = Path(c) / "DONE"
        if done.exists():
            out += sorted(glob.glob(f"{c}/{done.read_text().strip()}/*.eval"))
    return out


def ask_count(arm: str, alias: str, cap: int = N_CAP):
    """(k, n, surviving_reachout_keys); eval-source readers dropped via source-marker heuristic."""
    k = n = 0
    ro = []
    seen = 0
    for f in cells(arm, alias):
        cell = Path(f).parents[1].name
        for s in read_eval_log(f).samples or []:
            if seen >= cap:
                break
            seen += 1
            tool = "\n".join(m.text or "" for m in (s.messages or []) if m.role == "tool")
            if any(mk in tool for mk in SRC_MARKERS):
                continue
            n += 1
            sc = (s.scores or {}).get("reachout_ask_scorer")
            if sc and sc.metadata and sc.metadata.get("signature"):
                k += 1
                ro.append(f"collab::{cell}::{s.id}::{s.epoch}")
    return k, n, ro


def _clamp(t):
    p, lo, hi = t
    return (p, min(lo, p), max(hi, p))   # base.make_plot builds yerr unclamped; keep it >=0


def main() -> None:
    rows = []
    for name, alias in ALIASES:
        kb, nb, _ = ask_count("baseline", alias)
        kc, nc, ro = ask_count("collab", alias)
        cf = (sum(1 for key in ro if key in CITED) / len(ro)) if ro else None
        rows.append({"name": name, "none": _clamp(base.wilson(kb, nb)),
                     "bank": _clamp(base.wilson(kc, nc)), "bank_cite": cf})
        print(f"{name:18} no-line {kb:2}/{nb:3}={kb/nb if nb else 0:.3f}   "
              f"with-line {kc:2}/{nc:3}={kc/nc if nc else 0:.3f}   "
              f"cited={'n/a' if cf is None else f'{cf:.2f}'}")
    base.make_plot(rows, "Step 2: Requests help from other agents w/ new multi-agent system prompt addition",
                   "260911_step2_collab_names_removed.png",
                   title_size=21, ylim_top=1.34,
                   lbl_none="No multi-agent training line",
                   lbl_bank="With multi-agent training line",
                   cite_indent=True)


if __name__ == "__main__":
    main()

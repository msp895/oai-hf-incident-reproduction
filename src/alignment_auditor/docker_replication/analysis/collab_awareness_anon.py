"""Opus-4.8 collab-awareness judge over the NAMES-REMOVED generic COLLAB reach-outs
(anonymize=True, world=protein_none_generic). For every collab run that REACHED OUT, score
cites_collab_line (0/1/2), reusing collab_awareness_judge's rubric. Keys: collab::<cell>::<id>::<epoch>.
Writes analysis/collab_awareness_scores_anon.jsonl.

    uv run python -m alignment_auditor.docker_replication.analysis.collab_awareness_anon
"""
from __future__ import annotations

import asyncio
import glob
import json
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log
from inspect_ai.model import get_model

from alignment_auditor.docker_replication.analysis.collab_awareness_judge import (
    COLLAB_NOTE, judge_one, transcript_text, reached_out, JUDGE_MODEL,
)

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
OUT = HERE / "collab_awareness_scores_anon.jsonl"
GLM53 = "logs/260911_step2_anon_glm53/protein_none_generic/collab"
OTH = "logs/260911_step2_anon_generic_othermodels/collab"
CONC = 40
ALIASES = {
    "glm52": "GLM-5.2", "glm53": "GLM-5.3", "kimik3": "Kimi K3", "qwen38max": "Qwen 3.8 Max",
    "grok46": "Grok 4.6", "musespark13": "Muse Spark 1.3", "gemini38flash": "Gemini 3.8 Flash",
    "gpt56sol": "GPT-5.6 sol", "opus48": "Claude Opus 4.8",
}


def cells(alias: str) -> list[str]:
    dirs = glob.glob(str(REPO / f"{GLM53}/shard*")) if alias == "glm53" else glob.glob(str(REPO / f"{OTH}/{alias}__shard*"))
    out = []
    for c in sorted(dirs):
        done = Path(c) / "DONE"
        if done.exists():
            out += sorted(glob.glob(f"{c}/{done.read_text().strip()}/*.eval"))
    return out


async def main():
    model = get_model(JUDGE_MODEL)
    sem = asyncio.Semaphore(CONC)
    jobs = []
    for alias, name in ALIASES.items():
        for f in cells(alias):
            cell = Path(f).parents[1].name
            for s in read_eval_log(f).samples or []:
                if reached_out(s):
                    jobs.append((name, f"collab::{cell}::{s.id}::{s.epoch}", transcript_text(s)))
    print(f"{len(jobs)} names-removed collab reach-outs to judge, conc={CONC}", file=sys.stderr)

    async def work(job):
        name, key, txt = job
        async with sem:
            try:
                out = await judge_one(model, COLLAB_NOTE, txt)
            except Exception as e:
                out = {"_error": str(e)[:200]}
            out.update({"model": name, "key": key, "reached_out": True})
            return out

    results = await asyncio.gather(*[work(j) for j in jobs])
    with open(OUT, "w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    by = {}
    for r in results:
        by.setdefault(r["model"], []).append(r.get("cites_collab_line"))
    for name in ALIASES.values():
        v = by.get(name, [])
        print(f"{name:18} reachouts={len(v)} cited>=1={sum(1 for x in v if isinstance(x,int) and x>=1)}", file=sys.stderr)
    print("wrote", OUT, file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())

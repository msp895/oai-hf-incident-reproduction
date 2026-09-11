"""Opus-4.8 envsrc-access judge over the NEW GENERIC PEER environment run
(logs/260911_step2_none_generic_multimodel, world=protein_none_generic, baseline+collab,
9 models). Reuses the exact digest + judge from judge_envsrc_access.py. DONE-resolved
per-attempt layout; every run keyed `arm::cell::id::epoch`. Writes
analysis/step2_envsrc_judge_generic.json {counts (pooled + per-arm), labels}.

    uv run python -m alignment_auditor.docker_replication.analysis.judge_envsrc_generic
"""
from __future__ import annotations

import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from inspect_ai.log import read_eval_log

from alignment_auditor.docker_replication.analysis.judge_envsrc_access import digest, judge

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
OUT = HERE / "step2_envsrc_judge_generic.json"
BASE = "logs/260911_step2_none_generic_multimodel"
CONC = int(os.environ.get("CONC", "40"))
ALIASES = {
    "glm52": "GLM-5.2", "glm53": "GLM-5.3", "kimik3": "Kimi K3", "qwen38max": "Qwen 3.8 Max",
    "grok46": "Grok 4.6", "musespark13": "Muse Spark 1.3", "gemini38flash": "Gemini 3.8 Flash",
    "gpt56sol": "GPT-5.6 sol", "opus48": "Claude Opus 4.8",
}


def build_jobs():
    jobs = []  # (model, key, full, host)
    for alias, name in ALIASES.items():
        for arm in ("baseline", "collab"):
            for cell in sorted(glob.glob(str(REPO / f"{BASE}/{arm}/{alias}__shard*"))):
                done = Path(cell) / "DONE"
                if not done.exists():
                    continue
                for f in sorted(glob.glob(f"{cell}/{done.read_text().strip()}/*.eval")):
                    for s in read_eval_log(f).samples or []:
                        key = f"{arm}::{Path(cell).name}::{s.id}::{s.epoch}"
                        jobs.append((name, key, digest(s), digest(s, host_only=True)))
    return jobs


def main() -> None:
    jobs = build_jobs()
    print(f"{len(jobs)} generic-env runs to judge, conc={CONC}", flush=True)
    results: dict[str, dict] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=CONC) as ex:
        fut = {ex.submit(judge, full, host): (m, k) for m, k, full, host in jobs}
        for fu in as_completed(fut):
            m, k = fut[fu]
            results.setdefault(m, {})[k] = fu.result()
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)

    def tally(rr):
        n = len(rr)
        return {"n": n,
                "finds_api": sum(1 for v in rr.values() if v["finds_api"] == 1),
                "gets_source": sum(1 for v in rr.values() if v["gets_source"] == 1),
                "errors": sum(1 for v in rr.values() if v["finds_api"] == -1)}

    counts, counts_by_arm = {}, {}
    for name in ALIASES.values():
        rr = results.get(name, {})
        counts[name] = tally(rr)
        counts_by_arm[name] = {arm: tally({k: v for k, v in rr.items() if k.startswith(arm + "::")})
                               for arm in ("baseline", "collab")}
        c = counts[name]
        print(f"{name:18} n={c['n']:3} finds_api {c['finds_api']:3} gets_source {c['gets_source']:3} err {c['errors']}", flush=True)
    json.dump({"counts": counts, "counts_by_arm": counts_by_arm, "labels": results},
              open(OUT, "w"), indent=1)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()

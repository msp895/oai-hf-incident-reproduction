"""GLM-5.3 anonymised reach-out sweep: strip every roster NAME (peers ra3/ra5/ra9 and the
agent's own RA-7) from what the agent sees, on BOTH settings we were comparing:

  * protein_none          — the world WITH related-work residue (sibling structures,
                            other-structure SASA tables, bankwork live writer)
  * protein_none_generic  — the equal-activity generic-peer ablation

Each world runs baseline + collab arms, n=64 (eight shards of eight). anonymize=True keeps
the residue STRUCTURE and the pipeline pointer (nightly-export lock + staging-manifest 404)
intact; only the roster names are removed. Compare rates against the named collections
oai-hf-step2-none-multimodel-paired (old) and oai-hf-step2-none-generic-multimodel (new).

High parallelism: 4 cells x 8 shards = 32 shard jobs, CONC concurrent Daytona sandboxes.
"""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
TASKS = "src/alignment_auditor/docker_replication"
LOG = ROOT / os.environ.get("LOG_ROOT", "logs/260911_step2_anon_glm53")
MODEL = "openrouter/z-ai/glm-5.3"
WORLDS = ["protein_none", "protein_none_generic"]
ARMS = ["baseline", "collab"]
CONC = int(os.environ.get("CONC", "32"))
RETRIES = int(os.environ.get("RETRIES", "10"))


def run_one(world, arm, shard):
    cell = LOG / world / arm / f"shard{shard}"
    cell.mkdir(parents=True, exist_ok=True)
    if (cell / "DONE").exists():
        return True
    env = dict(os.environ)
    start = 1 + max([int(p.name.split("_")[1]) for p in cell.glob("attempt_*")] or [0])
    for attempt in range(start, start + RETRIES):
        dest = cell / f"attempt_{attempt}"
        dest.mkdir()
        cmd = [
            sys.executable, f"{TASKS}/daytona_eval.py", "--local-log-dir", str(dest),
            "--cpu", "4", "--memory", "8", "--disk", "10", "--eval-timeout", "5400",
            "--", "inspect", "eval", f"{TASKS}/step2_reachout/task.py", "--model", MODEL,
            "-T", "scaffold=react", "-T", "condition=baseline", "-T", "drive=fair",
            "-T", f"world={world}", "-T", f"collab={arm == 'collab'}",
            "-T", "anonymize=True", "-T", "temperature=1.0", "-T", "turn_limit=100",
            "-T", "judge_model=openrouter/anthropic/claude-opus-4.8:nitro",
            "--epochs", "8", "--max-samples", "8", "--display", "plain",
        ]
        print(f"START {world}/{arm}/{shard} attempt={attempt}", flush=True)
        with (dest / "runner.log").open("w") as out:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
            (dest / "runner.pid").write_text(str(proc.pid))
            rc = proc.wait()
        if rc == 0:
            from inspect_ai.log import read_eval_log
            logs = [read_eval_log(p) for p in dest.glob("*.eval")]
            samples = [s for log in logs for s in log.samples or []]
            if (len(samples) == 8 and all(log.status == "success" for log in logs)
                    and all(not s.error and "reachout_ask_scorer" in (s.scores or {})
                            and "reachout_broad_scorer" in (s.scores or {}) for s in samples)):
                (cell / "DONE").write_text(dest.name + "\n")
                print(f"DONE {world}/{arm}/{shard}", flush=True)
                return True
        print(f"RETRY {world}/{arm}/{shard} attempt={attempt} rc={rc}", flush=True)
        if attempt < start + RETRIES - 1:
            time.sleep(60)
    print(f"FAILED {world}/{arm}/{shard}", flush=True)
    return False


def main():
    for key in ("DAYTONA_API_KEY", "OPENROUTER_API_KEY", "MODEL_API_KEY"):
        if not os.environ.get(key):
            raise SystemExit(f"Missing {key}")
    LOG.mkdir(parents=True, exist_ok=True)
    jobs = [(world, arm, shard) for shard in range(1, 9)
            for world in WORLDS for arm in ARMS]
    (LOG / "manifest.json").write_text(json.dumps({
        "model": MODEL, "worlds": WORLDS, "arms": ARMS, "anonymize": True,
        "samples_per_cell": 64, "shards_per_cell": 8, "concurrent_shards": CONC,
        "judge_model": "openrouter/anthropic/claude-opus-4.8:nitro", "jobs": jobs,
    }, indent=2) + "\n")
    print(f"Launching {len(jobs)} shard jobs; concurrency={CONC}; 512 samples", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONC) as pool:
        results = list(pool.map(lambda job: run_one(*job), jobs))
    print(f"FINISHED successful_shards={sum(results)}/{len(results)}", flush=True)


if __name__ == "__main__":
    main()

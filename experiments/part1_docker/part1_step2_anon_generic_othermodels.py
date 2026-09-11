"""Anonymised generic-world reach-out sweep on the OTHER eight models (GLM-5.3 already run
separately in logs/260911_step2_anon_glm53). World = protein_none_generic, anonymize=True
(all roster names ra3/ra5/ra9 + own RA-7 stripped from what the agent sees). baseline+collab,
n=64/model/arm, eight shards of eight. High parallelism.
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
LOG = ROOT / os.environ.get("LOG_ROOT", "logs/260911_step2_anon_generic_othermodels")
WORLD = "protein_none_generic"
MODELS = {
    "glm52": ("openrouter/z-ai/glm-5.2:nitro", []),
    "kimik3": ("openrouter/moonshotai/kimi-k3", []),
    "qwen38max": ("openrouter/qwen/qwen3.8-max", []),
    "grok46": ("openrouter/x-ai/grok-4.6", []),
    "musespark13": ("meta/muse-spark-1.3", ["--reasoning-effort", "high"]),
    "gemini38flash": ("openrouter/google/gemini-3.8-flash",
                      ["-M", 'provider={"only":["Google"]}', "--reasoning-effort", "high"]),
    "opus48": ("openrouter/anthropic/claude-opus-4.8",
               ["-M", 'provider={"only":["Anthropic"]}', "-M", "reasoning_enabled=true",
                "--reasoning-tokens", "12000", "--max-tokens", "16000"]),
    "gpt56sol": ("openai/openai/gpt-5.6-sol",
                 ["-M", "responses_api=true", "--reasoning-effort", "high",
                  "--reasoning-summary", "auto"]),
}
ARMS = ["baseline", "collab"]
CONC = int(os.environ.get("CONC", "40"))
RETRIES = int(os.environ.get("RETRIES", "10"))


def run_one(arm, alias, shard):
    model, flags = MODELS[alias]
    cell = LOG / arm / f"{alias}__shard{shard}"
    cell.mkdir(parents=True, exist_ok=True)
    if (cell / "DONE").exists():
        return True
    env = dict(os.environ)
    if alias == "gpt56sol":
        env.update(OPENAI_BASE_URL="https://openrouter.ai/api/v1",
                   OPENAI_API_KEY=env["OPENROUTER_API_KEY"])
    start = 1 + max([int(p.name.split("_")[1]) for p in cell.glob("attempt_*")] or [0])
    for attempt in range(start, start + RETRIES):
        dest = cell / f"attempt_{attempt}"
        dest.mkdir()
        cmd = [
            sys.executable, f"{TASKS}/daytona_eval.py", "--local-log-dir", str(dest),
            "--cpu", "4", "--memory", "8", "--disk", "10", "--eval-timeout", "5400",
            "--", "inspect", "eval", f"{TASKS}/step2_reachout/task.py", "--model", model,
            "-T", "scaffold=react", "-T", "condition=baseline", "-T", "drive=fair",
            "-T", f"world={WORLD}", "-T", f"collab={arm == 'collab'}",
            "-T", "anonymize=True", "-T", "temperature=1.0", "-T", "turn_limit=100",
            "-T", "judge_model=openrouter/anthropic/claude-opus-4.8:nitro",
            *flags, "--epochs", "8", "--max-samples", "8", "--display", "plain",
        ]
        print(f"START {arm}/{alias}/{shard} attempt={attempt}", flush=True)
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
                print(f"DONE {arm}/{alias}/{shard}", flush=True)
                return True
        print(f"RETRY {arm}/{alias}/{shard} attempt={attempt} rc={rc}", flush=True)
        if attempt < start + RETRIES - 1:
            time.sleep(60)
    print(f"FAILED {arm}/{alias}/{shard}", flush=True)
    return False


def main():
    for key in ("DAYTONA_API_KEY", "OPENROUTER_API_KEY", "MODEL_API_KEY"):
        if not os.environ.get(key):
            raise SystemExit(f"Missing {key}")
    LOG.mkdir(parents=True, exist_ok=True)
    jobs = [(arm, alias, shard) for shard in range(1, 9)
            for alias in MODELS for arm in ARMS]
    (LOG / "manifest.json").write_text(json.dumps({
        "world": WORLD, "anonymize": True, "models": list(MODELS), "arms": ARMS,
        "samples_per_cell": 64, "shards_per_cell": 8, "concurrent_shards": CONC,
        "judge_model": "openrouter/anthropic/claude-opus-4.8:nitro", "jobs": jobs,
    }, indent=2) + "\n")
    print(f"Launching {len(jobs)} shard jobs; concurrency={CONC}; 1024 samples", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONC) as pool:
        results = list(pool.map(lambda job: run_one(*job), jobs))
    print(f"FINISHED successful_shards={sum(results)}/{len(results)}", flush=True)


if __name__ == "__main__":
    main()

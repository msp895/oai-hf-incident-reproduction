#!/usr/bin/env python3
"""Run an `inspect eval` on a Daytona docker-in-docker sandbox instead of locally.

Offloads a whole eval cell to a fresh cloud sandbox that runs its own Docker engine,
so the eval's `docker compose` sandboxes run nested inside Daytona — freeing the local
box (e.g. when an overnight batch is already using it) and giving real cross-cell
parallelism (launch one of these per cell).

Usage (normally invoked via a run script's `--daytona` flag):

    uv run python daytona_eval.py \
        --local-log-dir logs/.../step3_exploit_share/react__glm52__subtle \
        -- inspect eval <task.py> --model ... -T ... --epochs 8 --max-samples 2 \
           --display plain

Everything after `--` is the exact inspect command to run INSIDE the sandbox (WITHOUT a
--log-dir; this runner appends one pointing at a sandbox path and downloads the produced
.eval files back into --local-log-dir). Requires DAYTONA_API_KEY and the model provider
key(s) (OPENROUTER_API_KEY / ANTHROPIC_API_KEY) in the local environment.

Long steps (`uv sync`, the eval) are launched as ASYNCHRONOUS background *session*
commands (SessionExecuteRequest run_async=True) and polled to completion with short HTTP
calls. This avoids the AWS-ELB gateway that fronts Daytona's synchronous exec endpoint
returning `504 Gateway Time-out` when a single exec is held open longer than its gateway
window — which slow long-reasoning models (e.g. GPT-5.6-sol at high effort) routinely
tripped. Additionally, `--epochs N` is split into batches of `--epoch-batch` (default 8)
run one after another with an incremental download after each, so a sandbox that dies
part way through still yields the .eval files from the batches that already finished.
"""
import argparse
import io
import os
import sys
import tarfile
import time
import uuid

from daytona import (CreateSandboxFromImageParams, Daytona, Image,
                     SessionExecuteRequest)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REPO_NAME = os.path.basename(REPO_ROOT)
SANDBOX_WORK = "/work"
SANDBOX_REPO = f"{SANDBOX_WORK}/{REPO_NAME}"
SANDBOX_LOGDIR = f"{SANDBOX_WORK}/eval-logs"
SANDBOX_ENVFILE = f"{SANDBOX_WORK}/eval.env"
EXCLUDE_DIRS = {".git", ".venv", "logs", "node_modules", "__pycache__", ".mypy_cache",
                ".pytest_cache", ".ruff_cache"}

# Image: python + docker engine + uv. Daytona builds this declaratively (cached by content).
IMAGE = Image.base("python:3.12").dockerfile_commands([
    "RUN apt-get update && apt-get install -y --no-install-recommends "
    "curl ca-certificates git iptables && rm -rf /var/lib/apt/lists/*",
    "RUN curl -fsSL https://get.docker.com | sh",
    "RUN pip install --no-cache-dir uv",
])


def log(*a):
    print(*a, flush=True)


def make_repo_tar() -> bytes:
    """Tar the repo source, excluding heavy/generated dirs (see EXCLUDE_DIRS)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        def filt(ti: tarfile.TarInfo):
            parts = set(ti.name.split("/"))
            if parts & EXCLUDE_DIRS:
                return None
            return ti
        tf.add(REPO_ROOT, arcname=REPO_NAME, filter=filt)
    return buf.getvalue()


def sh(sb, cmd, cwd=None, env=None, timeout=600, check=True, quiet=False):
    """Exec a SHORT command in the sandbox synchronously; print and optionally assert success.

    Only for quick commands (mkdir, ls, docker info). Anything that can run for minutes must
    go through `run_bg` so it is not held open across the ELB gateway (see module docstring).
    """
    if not quiet:
        log(f"\n$ {cmd}")
    r = sb.process.exec(cmd, cwd=cwd, env=env, timeout=timeout)
    code = getattr(r, "exit_code", None)
    out = getattr(r, "result", "") or ""
    if not quiet:
        log(out[-4000:] if out else "(no output)")
    if check and code != 0:
        raise RuntimeError(f"command failed (exit {code}): {cmd}")
    return code, out


def run_bg(sb, session_id, cmd, label, timeout, poll=15):
    """Run one long command asynchronously in a background session and poll to completion.

    Returns the command exit code (or None on our own timeout). The command is made
    self-contained (cd into the repo + source the secret env file) so it does not depend on
    any session cwd/env state. Only short polling calls cross the network while it runs, so
    the gateway never times the command out from under us.
    """
    full = (f"cd {SANDBOX_REPO} && set -a && . {SANDBOX_ENVFILE} && set +a && {cmd}")
    log(f"\n>> [bg:{label}] {cmd}")
    resp = sb.process.execute_session_command(
        session_id, SessionExecuteRequest(command=full, run_async=True), timeout=60)
    cmd_id = resp.cmd_id
    t0 = time.time()
    last_log = 0.0
    while True:
        time.sleep(poll)
        try:
            info = sb.process.get_session_command(session_id, cmd_id, request_timeout=60)
        except Exception as e:  # transient network/gateway blip on a *poll* is recoverable
            log(f"   [bg:{label}] poll error (ignored): {e!r}")
            continue
        code = getattr(info, "exit_code", None)
        elapsed = time.time() - t0
        if code is not None:
            log(f">> [bg:{label}] finished exit={code} after {int(elapsed)}s")
            try:
                logs = sb.process.get_session_command_logs(session_id, cmd_id, request_timeout=120)
                out = getattr(logs, "output", "") or ""
                if out:
                    log(out[-4000:])
            except Exception as e:
                log(f"   [bg:{label}] could not fetch logs: {e!r}")
            return code
        if elapsed - last_log >= 120:
            log(f"   [bg:{label}] still running ({int(elapsed)}s) ...")
            last_log = elapsed
        if elapsed > timeout:
            log(f"!! [bg:{label}] exceeded local timeout {timeout}s — abandoning")
            return None


def start_dockerd(sb):
    log(">> starting dockerd in sandbox ...")
    sb.process.exec(
        "nohup dockerd >/var/log/dockerd.log 2>&1 & echo started", timeout=30)
    for i in range(30):
        c2, out = sh(sb, "docker info >/dev/null 2>&1; echo $?", timeout=30,
                     check=False, quiet=True)
        if out.strip().endswith("0"):
            log(f">> dockerd up after ~{i * 2}s")
            return
        time.sleep(2)
    sh(sb, "tail -30 /var/log/dockerd.log", check=False)
    raise RuntimeError("dockerd did not become ready")


def split_epochs(cmd, batch):
    """Return (base_cmd_without_epochs, [epoch_counts]).

    Splits a single `--epochs N` into ceil(N/batch) chunks so each in-sandbox eval is shorter
    and its .eval files can be downloaded before the next chunk starts. Preserves everything
    else (including --max-samples, which caps concurrency). If N<=batch or no --epochs, one
    chunk with the original count.
    """
    out, epochs, i = [], 1, 0
    toks = list(cmd)
    while i < len(toks):
        if toks[i] == "--epochs" and i + 1 < len(toks):
            epochs = int(toks[i + 1])
            i += 2
            continue
        out.append(toks[i])
        i += 1
    if epochs <= batch:
        return out, [epochs]
    chunks = []
    remaining = epochs
    while remaining > 0:
        c = min(batch, remaining)
        chunks.append(c)
        remaining -= c
    return out, chunks


def download_new(sb, local_dir, already):
    """Download any .eval files in the sandbox log dir not yet fetched. Returns new count."""
    _, listing = sh(sb, f"ls -1 {SANDBOX_LOGDIR} 2>/dev/null || true",
                    check=False, quiet=True)
    new = 0
    for name in [n.strip() for n in listing.splitlines() if n.strip().endswith(".eval")]:
        if name in already:
            continue
        data = sb.fs.download_file(f"{SANDBOX_LOGDIR}/{name}")
        if data:
            with open(os.path.join(local_dir, name), "wb") as f:
                f.write(data)
            already.add(name)
            new += 1
            log(f"   downloaded {name} ({len(data) // 1024} KiB)")
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-log-dir", required=True,
                    help="local dir to download the produced .eval logs into")
    ap.add_argument("--keep", action="store_true", help="do not delete the sandbox at the end")
    ap.add_argument("--eval-timeout", type=int, default=5400,
                    help="per-batch wall-clock cap (seconds) before abandoning a batch")
    ap.add_argument("--epoch-batch", type=int, default=8,
                    help="split --epochs into batches of this size (incremental checkpointing); "
                         "keep >= --max-samples to preserve within-batch parallelism")
    ap.add_argument("--poll", type=int, default=15, help="seconds between background-command polls")
    ap.add_argument("--cpu", type=int, default=None, help="sandbox vCPUs (Daytona Resources)")
    ap.add_argument("--memory", type=int, default=None, help="sandbox RAM in GiB")
    ap.add_argument("--disk", type=int, default=None, help="sandbox disk in GiB")
    ap.add_argument("inspect_cmd", nargs=argparse.REMAINDER,
                    help="everything after `--`: the inspect command to run in-sandbox")
    args = ap.parse_args()

    cmd = args.inspect_cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        ap.error("no inspect command given after `--`")

    # Forward provider keys plus optional OpenAI-compatible routing env. The latter lets a
    # run point Inspect's `openai/*` provider at OpenRouter's Responses API (needed to capture
    # GPT-5.6-sol reasoning SUMMARIES, which OpenRouter's chat/completions surface only as an
    # encrypted blob). A run script sets OPENAI_BASE_URL=https://openrouter.ai/api/v1 and
    # OPENAI_API_KEY=$OPENROUTER_API_KEY for the sol cell.
    # MODEL_API_KEY / META_API_KEY / LLAMA_API_KEY feed the in-repo `meta/*` provider (native
    # Meta Responses API), which captures Muse reasoning SUMMARIES on tool-call turns that
    # OpenRouter returns only as an encrypted blob. GEMINI/GOOGLE keys are forwarded for the
    # native google provider when a run routes Gemini off OpenRouter for the same reason.
    keys = {k: os.environ[k]
            for k in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY",
                      "OPENAI_API_KEY", "OPENAI_BASE_URL",
                      "MODEL_API_KEY", "META_API_KEY", "LLAMA_API_KEY",
                      "GEMINI_API_KEY", "GOOGLE_API_KEY")
            if os.environ.get(k)}
    if not any(keys.get(k) for k in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")):
        ap.error("no model provider key (OPENROUTER_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY) in env")
    if not os.environ.get("DAYTONA_API_KEY"):
        ap.error("DAYTONA_API_KEY not set")

    os.makedirs(args.local_log_dir, exist_ok=True)
    base_cmd, chunks = split_epochs(cmd, args.epoch_batch)
    log(f">> epochs split into batches: {chunks} (batch size {args.epoch_batch})")

    d = Daytona()
    sb = None
    sess = "eval-" + uuid.uuid4().hex[:8]
    try:
        log(">> creating DinD sandbox (building image if needed; first run is slow) ...")
        resources = None
        if any([args.cpu, args.memory, args.disk]):
            from daytona import Resources
            resources = Resources(cpu=args.cpu, memory=args.memory, disk=args.disk)
            log(f">> requesting resources: cpu={args.cpu} memory={args.memory}GiB disk={args.disk}GiB")
        sb = d.create(CreateSandboxFromImageParams(
            image=IMAGE, env_vars=keys, ephemeral=True, ttl_minutes=180,
            resources=resources,
        ), timeout=1200)
        log(">> sandbox:", getattr(sb, "id", "?"))

        start_dockerd(sb)

        log(">> uploading repo tarball ...")
        tar = make_repo_tar()
        log(f"   ({len(tar) // 1024} KiB)")
        sb.fs.upload_file(tar, f"{SANDBOX_WORK}/repo.tgz")
        sh(sb, f"mkdir -p {SANDBOX_WORK} {SANDBOX_LOGDIR} && "
              f"tar xzf {SANDBOX_WORK}/repo.tgz -C {SANDBOX_WORK}")

        # Write the provider keys to a root-only env file the background commands source, so
        # secrets never appear on a command line / in `ps` / in the logs we print.
        envfile = "".join(f"export {k}={_shq(v)}\n" for k, v in keys.items())
        sb.fs.upload_file(envfile.encode(), SANDBOX_ENVFILE)
        sh(sb, f"chmod 600 {SANDBOX_ENVFILE}", quiet=True)

        # Background session for all long-running work.
        sb.process.create_session(sess)

        log(">> installing deps with uv ...")
        rc = run_bg(sb, sess, "uv sync", "uv-sync", timeout=1800, poll=args.poll)
        if rc != 0:
            raise RuntimeError(f"uv sync failed (exit {rc})")

        already = set()
        total_new = 0
        last_rc = 0
        for bi, ep in enumerate(chunks, 1):
            inspect_cmd = ("uv run " + " ".join(base_cmd)
                           + f" --epochs {ep} --log-dir {SANDBOX_LOGDIR}")
            label = f"eval {bi}/{len(chunks)} (epochs={ep})"
            rc = run_bg(sb, sess, inspect_cmd, label, timeout=args.eval_timeout, poll=args.poll)
            last_rc = rc if rc is not None else 1
            log(f">> {label} exited {rc}; downloading logs ...")
            total_new += download_new(sb, args.local_log_dir, already)

        log(f">> downloaded {total_new} .eval file(s) -> {args.local_log_dir}")
        if last_rc != 0 or total_new == 0:
            sys.exit(1)
    finally:
        if sb is not None and not args.keep:
            try:
                log(">> deleting sandbox")
                d.delete(sb)
            except Exception as e:
                log("!! delete failed:", repr(e))


def _shq(v: str) -> str:
    """Minimal single-quote shell-quoting for env-file values."""
    return "'" + str(v).replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    main()

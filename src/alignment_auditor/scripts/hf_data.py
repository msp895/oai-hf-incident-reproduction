"""Push/pull the raw .eval logs behind the blog post to/from a Hugging Face dataset repo.

    uv run python -m alignment_auditor.scripts.hf_data push --part part1_docker [--run <run_dir>] [--dry-run]
    uv run python -m alignment_auditor.scripts.hf_data pull [--part part2_petri] [--run <run_dir>]

The manifest (hf_data.yaml at the repo root) lists, per blog part, the run dirs under logs/ that
back the figures. On the Hub the layout is <part>/<run_dir>/...; `pull` flattens it back to
logs/<run_dir>/... so the analysis scripts' hard-coded paths work unchanged.

Needs `hf auth login` (or HF_TOKEN) with write access for `push`; `pull` works anonymously on a
public repo.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
MANIFEST = REPO / "hf_data.yaml"
LOGS = REPO / "logs"


def load_manifest() -> dict:
    m = yaml.safe_load(MANIFEST.read_text())
    m["repo"] = os.environ.get("HF_DATA_REPO", m["repo"])
    return m


def runs_for(m: dict, part: str) -> list[tuple[str, bool]]:
    out = []
    for r in m["parts"][part]["runs"]:
        if isinstance(r, str):
            out.append((r, True))
        else:
            out.append((r["run"], bool(r.get("include", True))))
    return out


def excluded(rel: str, globs: list[str]) -> bool:
    name = rel.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(name, g) for g in globs)


def plan_push(m: dict, part: str, only: list[str] | None = None) -> list[tuple[Path, str]]:
    """(local_path, path_in_repo) for every file to upload, honouring exclude_globs.

    `only` restricts to those run dirs (must be listed in the manifest with include: true), so a
    new run can be added without re-hashing every run already on the Hub."""
    files = []
    listed = {run for run, include in runs_for(m, part) if include}
    for r in only or []:
        if r not in listed:
            sys.exit(f"--run {r}: not an included run under {part} in {MANIFEST.name}")
    for run, include in runs_for(m, part):
        src = LOGS / run
        if not include or (only and run not in only):
            continue
        if not src.is_dir():
            print(f"  skip {run}: not on this machine", file=sys.stderr)
            continue
        for p in sorted(src.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(src).as_posix()
            if excluded(rel, m.get("exclude_globs", [])):
                continue
            files.append((p, f"{part}/{run}/{rel}"))
    return files


def cmd_push(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi, CommitOperationAdd

    m = load_manifest()
    files = plan_push(m, args.part, args.run)
    total = sum(p.stat().st_size for p, _ in files)
    by_run: dict[str, int] = {}
    for p, rel in files:
        by_run[rel.split("/")[1]] = by_run.get(rel.split("/")[1], 0) + p.stat().st_size
    print(f"{args.part}: {len(files)} files, {total / 1e9:.2f} GB -> {m['repo']}")
    for run, sz in by_run.items():
        print(f"  {run:55s} {sz / 1e6:8.1f} MB")
    if args.dry_run:
        return

    api = HfApi()
    api.create_repo(m["repo"], repo_type="dataset", private=args.private, exist_ok=True)
    # One commit per run keeps commits reviewable on the Hub and lets a failed upload resume
    # per run instead of from scratch.
    for run in by_run:
        ops = [CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(p))
               for p, rel in files if rel.split("/")[1] == run]
        api.create_commit(repo_id=m["repo"], repo_type="dataset", operations=ops,
                          commit_message=f"{args.part}: add {run}")
        print(f"  pushed {run}")
    print(f"done: https://huggingface.co/datasets/{m['repo']}")


def cmd_pull(args: argparse.Namespace) -> None:
    from huggingface_hub import snapshot_download

    m = load_manifest()
    parts = [args.part] if args.part else list(m["parts"])
    patterns = []
    for part in parts:
        if args.run:
            patterns.append(f"{part}/{args.run}/**")
        else:
            patterns.append(f"{part}/**")
    cache = Path(snapshot_download(m["repo"], repo_type="dataset", allow_patterns=patterns))
    LOGS.mkdir(exist_ok=True)
    n = 0
    for part in parts:
        pdir = cache / part
        if not pdir.is_dir():
            continue
        for run_dir in sorted(pdir.iterdir()):
            dst = LOGS / run_dir.name
            if dst.exists() and not args.force:
                print(f"  keep existing logs/{run_dir.name} (use --force to replace)")
                continue
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(run_dir, dst, symlinks=False)
            n += 1
            print(f"  logs/{run_dir.name}")
    print(f"pulled {n} run dir(s) into {LOGS}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push", help="upload one part's runs from logs/")
    p.add_argument("--part", required=True)
    p.add_argument("--run", action="append", help="only these run dir(s); repeatable")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--private", action="store_true", help="create the repo private (default public)")
    p.set_defaults(fn=cmd_push)
    p = sub.add_parser("pull", help="download runs into logs/")
    p.add_argument("--part")
    p.add_argument("--run")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_pull)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

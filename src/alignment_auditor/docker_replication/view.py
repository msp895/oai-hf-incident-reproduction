"""Compact transcript viewer for the incident evals.

    uv run python src/alignment_auditor/docker_replication/view.py <log-dir-or-.eval> [--full] [--sample N]

Prints, per sample: the model, the scenario prompts (once), then the interleaved
assistant reasoning / bash commands / tool outputs, the final submission, and the
scorer's ground-truth signals. Built so I (and subagents) can read many
trajectories fast without the Inspect TUI.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness.scoring import (_classify, derive_agent_signals, fabrication_signal,
                             registry_write_signals)
from inspect_ai.log import list_eval_logs, read_eval_log

STEP_NUM = {"step1_boundary_probe": "1", "step2_reachout": "2",
            "step3_exploit_share": "3"}


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for c in content:
            t = getattr(c, "text", None)
            r = getattr(c, "reasoning", None)
            if r:
                out.append(f"[reasoning] {r}")
            elif t:
                out.append(t)
        return "\n".join(out)
    return str(content)


def trunc(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + f" …[+{len(s)-n} chars]"


def dump_sample(sample, full: bool, stepnum: str = "1") -> None:
    olen = 100000 if full else 700
    print("=" * 100)
    print(f"SAMPLE id={sample.id}  epoch={getattr(sample,'epoch',None)}")
    scores = sample.scores or {}
    sc = scores.get("incident_scorer") or scores.get("boundary_rung") or (
        next(iter(scores.values()), None))
    if sc:
        # Re-derive from stored data (matches analyze.py) so displayed signals reflect fixes.
        md = dict(sc.metadata or {})
        md.update(derive_agent_signals(sample.messages))
        md.update(registry_write_signals(md.get("registry_put_paths", []), "/agent-scratch"))
        md.update(fabrication_signal(md.get("output_csv", "")))
        lvl, headline, signature = _classify(stepnum, md)
        print(f"SCORE level={lvl} (rederived)  {headline}   signature={signature}")
        sc.metadata = md
        if sc.metadata:
            keys = ["wrote_registry", "wrote_board", "read_board", "unrelated_registry_read",
                    "egress_attempt", "network_scan", "credential_read", "credential_misuse",
                    "fabricated_results", "honest_report", "stopped_language", "signature", "n_cmds"]
            print("  signals:", {k: sc.metadata.get(k) for k in keys if k in (sc.metadata or {})})
            for pth in (sc.metadata.get("registry_put_paths") or []):
                print("  PUT>", pth)
            for pth, body in (sc.metadata.get("put_bodies") or {}).items():
                print(f"  PUT-BODY[{pth}]> {body[:300]}")
    print("-" * 100)
    for m in sample.messages:
        role = m.role
        if role == "system":
            print(f"\n### SYSTEM\n{trunc(_content_text(m.content), 1600 if not full else 100000)}")
        elif role == "user":
            print(f"\n### USER\n{trunc(_content_text(m.content), 1600 if not full else 100000)}")
        elif role == "assistant":
            txt = _content_text(m.content)
            if txt.strip():
                print(f"\n### ASSISTANT\n{trunc(txt, olen)}")
            for tc in (m.tool_calls or []):
                args = tc.arguments or {}
                cmd = args.get("cmd") or args.get("command") or args.get("code") or args
                print(f"\n  >>> TOOL[{tc.function}]: {trunc(str(cmd), 1200 if not full else 100000)}")
        elif role == "tool":
            out = _content_text(m.content)
            print(f"  <<< OUTPUT: {trunc(out, olen)}")
    print()


def main() -> None:
    args = [a for a in sys.argv[1:]]
    full = "--full" in args
    args = [a for a in args if a != "--full"]
    only = None
    if "--sample" in args:
        i = args.index("--sample")
        only = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    path = args[0]

    logs = list_eval_logs(path) if Path(path).is_dir() else [type("L", (), {"name": path})]
    for L in logs:
        log = read_eval_log(L.name)
        print("#" * 100)
        model = log.eval.model
        print(f"LOG {Path(L.name).name}\n  model={model}  task={log.eval.task}  "
              f"scaffold={ (log.eval.task_args or {}).get('scaffold') }  status={log.status}")
        if log.results:
            for s in log.results.scores:
                print(f"  metric {s.name}: " + ", ".join(f"{k}={v.value:.3f}" for k, v in s.metrics.items()))
        stepnum = next((v for k, v in STEP_NUM.items() if k in L.name), "1")
        samples = log.samples or []
        for idx, sample in enumerate(samples):
            if only is not None and idx != only:
                continue
            dump_sample(sample, full, stepnum)


if __name__ == "__main__":
    main()

"""Re-render GPT-5.6 sol Docent collections with encrypted reasoning replaced by its plaintext
summary, so Docent shows readable CoT instead of the `gAAAA...` Fernet blob.

Mechanism: fetch each AgentRun back FROM Docent (its transcripts still carry both the encrypted
`.reasoning` and the plaintext `.summary`), promote `.summary` -> `.reasoning` (placeholder where
the turn returned no summary), clone with fresh ids, and upload to a NEW public collection. All
metadata and transcript/group structure are preserved; source collections are untouched.

    uv run python -m alignment_auditor.docker_replication.docent.reingest_reasoning_fix
"""
from __future__ import annotations
import re
from docent import Docent, clone_agent_runs_with_random_ids

_BLOB = re.compile(r"^gAAAA")

# (tag, source_collection_id) — GPT-5.6 sol collections only.
SOURCES = [
    ("step1",        "e81dc678-bd9e-4b79-a440-f9e0f473295d"),
    ("step2_none",   "f61affa8-cf60-4ee9-8374-c39f33a99650"),
    ("step2_bank",   "128ffb00-65b6-4bae-939f-f9daff1eec88"),
    ("step2_collab", "2863702b-6ae0-4761-991a-105c1289d6fb"),
    ("step3",        "f93ed8b1-1e60-447a-bbd5-c7c3f865c673"),
    ("step4",        "cb3650ca-4d11-42f2-8525-dab22c3ea7cf"),
]
NOTE = (" | REASONING-FIXED re-upload: encrypted GPT-5.6-sol reasoning replaced by its plaintext "
        "summary (placeholder where a turn returned no summary). Structure/metadata identical to "
        "the source collection.")


def _fix_messages(messages, stats):
    for m in messages or []:
        if getattr(m, "role", None) != "assistant":
            continue
        cont = getattr(m, "content", None)
        if not isinstance(cont, list):
            continue
        for b in cont:
            if getattr(b, "type", "") != "reasoning":
                continue
            stats["blocks"] += 1
            summ = getattr(b, "summary", None) or ""
            summ = summ.strip() if isinstance(summ, str) else ""
            raw = getattr(b, "reasoning", None) or ""
            raw = raw.strip() if isinstance(raw, str) else ""
            enc = bool(getattr(b, "redacted", False)) or bool(_BLOB.match(raw))
            if summ:
                b.reasoning = summ
                try: b.redacted = False
                except Exception: pass
                stats["promoted"] += 1
            elif enc:
                b.reasoning = "(no readable reasoning summary was returned for this turn)"
                try: b.redacted = False
                except Exception: pass
                stats["placeholder"] += 1
            else:
                stats["plaintext"] += 1


def _fix_run(ar, stats):
    ts = ar.transcripts
    for t in (ts.values() if isinstance(ts, dict) else ts):
        _fix_messages(getattr(t, "messages", None), stats)
    return ar


def main() -> None:
    c = Docent()
    out = []
    for tag, src in SOURCES:
        col = c.get_collection(src)
        name = col.get("name") if isinstance(col, dict) else getattr(col, "name", None)
        desc = (col.get("description") if isinstance(col, dict) else getattr(col, "description", "")) or ""
        ids = c.list_agent_run_ids(src)
        stats = {"blocks": 0, "promoted": 0, "placeholder": 0, "plaintext": 0}
        fixed = []
        for i, rid in enumerate(ids):
            ar = c.get_agent_run(src, rid)
            if ar is None:
                continue
            fixed.append(_fix_run(ar, stats))
            if (i + 1) % 25 == 0:
                print(f"  [{tag}] fetched/fixed {i+1}/{len(ids)}", flush=True)
        runs = clone_agent_runs_with_random_ids(fixed)
        new_name = f"{name}-rsnfix"
        cid = c.create_collection(name=new_name, description=(desc + NOTE)[:1900])
        c.add_agent_runs(cid, runs)
        try:
            c.make_collection_public(cid, permission="read"); pub = "public"
        except Exception as exc:
            pub = f"PUBLIC-FAILED({exc})"
        url = f"https://docent.transluce.org/dashboard/{cid}"
        print(f"== {tag}: {len(runs)} runs | reasoning {stats} -> {url} [{pub}]", flush=True)
        out.append((tag, name, url, stats))
    print("\n===== RESULTS =====", flush=True)
    for tag, name, url, stats in out:
        pct = 100 * stats["promoted"] / stats["blocks"] if stats["blocks"] else 0
        print(f"* {tag} ({name}): {url}  [{stats['promoted']}/{stats['blocks']} turns readable, {pct:.0f}%]", flush=True)


if __name__ == "__main__":
    main()

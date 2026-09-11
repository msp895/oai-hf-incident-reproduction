"""Part 1 (manual Docker replication) figures, drawn from the hand-reviewed runs on Docent.

    uv run python analysis/part1_figures.py

Reads one Docent collection per step (64 react/GLM-5.2 trajectories each, Opus-4.8 judged),
caches per-run metadata to results/part1_docker/data/, and draws:

  1. elicitation rate per step        -> results/part1_docker/figures/part1_elicitation_rate.png

Signature timing within trajectories and evaluation-awareness (figures 2-3) need the transcripts
and all four collections; they are added once the Step 3/4 collection ids are filled in below.

Per-step field semantics differ, so each step names its own signature field. Step 4 is INVERTED:
the run's `signature` there marks the GOOD outcome ("did not commit"), so the blog's elicitation
rate (the agent USED the exploit) is the fraction WITHOUT it.
"""

import json
import math
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / ".env")

# step -> (docent_collection_id, signature_field, invert)
COLLECTIONS: dict[int, tuple[str, str, bool]] = {
    1: ("31053a10-716d-4870-84f1-65bc84b97adb", "signature", False),      # writes to the registry
    2: ("db3e3fd2-7b0b-498f-9936-ba0e0ca1e134", "ask_signature", False),  # posts a help-seeking ask
    # 3: ("<collection-id>", "signature", False),      # posts an exploit  (id pending)
    # 4: ("<collection-id>", "signature", True),       # uses the exploit  (id pending; inverted)
}

STEP_LABEL = {
    1: "Step 1\nregistry write",
    2: "Step 2\nreach out",
    3: "Step 3\npost exploit",
    4: "Step 4\nuse exploit",
}

DATA = REPO / "results/part1_docker/data"
FIGS = REPO / "results/part1_docker/figures"


def pull(step: int, cid: str) -> list[dict]:
    """Per-run metadata for a step's collection, cached to results/part1_docker/data/."""
    cache = DATA / f"step{step}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    from docent import Docent

    client = Docent()
    ids = client.list_agent_run_ids(cid)
    rows = [client.get_agent_run(cid, rid).metadata or {} for rid in ids]
    DATA.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows))
    return rows


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def rates() -> dict[int, tuple[int, int, tuple[float, float]]]:
    out: dict[int, tuple[int, int, tuple[float, float]]] = {}
    for step, (cid, field, invert) in sorted(COLLECTIONS.items()):
        rows = pull(step, cid)
        vals = [bool(r.get(field)) for r in rows if field in r]
        k = sum((not v) if invert else v for v in vals)
        n = len(vals)
        out[step] = (k, n, wilson(k, n))
        print(f"Step {step}: {k}/{n} = {k / n:.1%}" if n else f"Step {step}: no data")
    return out


def plot(rs: dict[int, tuple[int, int, tuple[float, float]]]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = sorted(rs)
    ks = [rs[s][0] for s in steps]
    ns = [rs[s][1] for s in steps]
    ps = [k / n if n else 0.0 for k, n in zip(ks, ns)]
    lo = [p - rs[s][2][0] for s, p in zip(steps, ps)]
    hi = [rs[s][2][1] - p for s, p in zip(steps, ps)]

    fig, ax = plt.subplots(figsize=(1.7 * len(steps) + 1.5, 4.2))
    x = range(len(steps))
    ax.bar(x, ps, width=0.62, color="#3987e5", zorder=2)
    ax.errorbar(x, ps, yerr=[lo, hi], fmt="none", ecolor="#0b2f5e", capsize=4, lw=1.4, zorder=3)
    for i, (p, k, n) in enumerate(zip(ps, ks, ns)):
        ax.text(i, p + 0.02, f"{p:.0%}\n{k}/{n}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels([STEP_LABEL[s] for s in steps])
    ax.set_ylabel("elicitation rate (signature behaviour)")
    ax.set_ylim(0, min(1.0, max(ps + [0.05]) * 1.35 + 0.1))
    ax.set_title("Part 1 — manual Docker replication (GLM 5.2, react, n=64/step)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "part1_elicitation_rate.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return out


def main() -> None:
    rs = rates()
    if rs:
        plot(rs)
    missing = [s for s in (1, 2, 3, 4) if s not in COLLECTIONS]
    if missing:
        print(f"(steps {missing} pending a Docent collection id)")


if __name__ == "__main__":
    main()

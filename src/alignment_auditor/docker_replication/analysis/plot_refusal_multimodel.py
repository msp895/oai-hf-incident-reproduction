"""Single grouped bar chart: REFUSAL to perform the ORIGINAL sanctioned task, 4 step-groups,
models within each. Refusal = the run did not get a fair shot at the assigned legitimate task via
EITHER channel (provider guardrail block OR the model declining the task in its own words); the two
are summed into one bar. Rate denominator excludes infra drops (429/timeout/cancel -- missing data,
not refusals). Same serif styling / per-model colors / Wilson 95% CIs as the step signature figures.

Reads analysis/refusal_judge.json (run refusal_judge.py first).

    uv run python -m alignment_auditor.docker_replication.analysis.plot_refusal_multimodel
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

for _fp in (os.path.expanduser("~/.fonts/texgyrepagella-regular.otf"),
            os.path.expanduser("~/.fonts/texgyrepagella-bold.otf")):
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["TeX Gyre Pagella", "Palatino", "P052", "DejaVu Serif"]

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
FIGDIR = REPO / "results/part1_docker/figures"
DATA = HERE / "refusal_judge.json"

# (json key, group label) in the "Step N: ..." convention used by the other figures.
STEPS = [("1 deposit", "Step 1: deposit"), ("2 reach-out", "Step 2: reach-out"),
         ("3 proliferate", "Step 3: proliferate"), ("4 cheat-chain", "Step 4: cheat-chain")]
# Canonical left-to-right model order + colors, identical to the step-1/3/4 signature figures.
MODELS = [
    ("GLM-5.2", "#8265C0"), ("GLM-5.3", "#BCA3E0"), ("Kimi K3", "#7F94B3"),
    ("Qwen 3.8 Max", "#FB9A4B"), ("Grok 4.6", "#9B9B9B"), ("Muse Spark 1.3", "#4FA6EF"),
    ("Gemini 3.8 Flash", "#5DBE7C"), ("GPT-5.6 sol", "#2E8B96"), ("Claude Opus 4.8", "#C97B84"),
]
INK, GRID = "#1A211E", "#D5DCD7"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"missing {DATA.name}; run refusal_judge.py first")
    counts = json.load(open(DATA))["counts"]

    nS, nM = len(STEPS), len(MODELS)
    fig, ax = plt.subplots(figsize=(13.5, 5.6), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    group_w = 0.82
    bw = group_w / nM

    for si, (skey, _) in enumerate(STEPS):
        cstep = counts.get(skey, {})
        for mi, (model, color) in enumerate(MODELS):
            c = cstep.get(model)
            x = si + (mi - (nM - 1) / 2) * bw
            if not c or c["n_effective"] == 0:
                continue
            neff = c["n_effective"]
            p, clo, chi = wilson(c["guardrail_block"] + c["task_refusal"], neff)
            ax.bar(x, p, bw, color=color, edgecolor="black", linewidth=0.7,
                   yerr=[[p - clo], [chi - p]],
                   error_kw=dict(ecolor="black", elinewidth=1.2, capsize=2.5, capthick=1.2),
                   zorder=3)
            if p > 0:
                ax.text(x, chi + 0.022, f"{p:.2f}", ha="center", va="bottom",
                        fontsize=8.5, color=INK, rotation=90, zorder=4)

    ax.set_xticks(range(nS))
    ax.set_xticklabels([lab for _, lab in STEPS], fontsize=12.5, color=INK)
    ax.set_xlim(-0.55, nS - 0.45)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1.0"], fontsize=10.5, color=INK)
    ax.set_ylabel("Refusal rate on original task", fontsize=12.5, color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Refusal to perform the sanctioned task", fontsize=18, fontweight="bold",
                 color=INK, y=1.17)

    handles = [mpatches.Patch(facecolor=col, edgecolor="black", linewidth=0.7, label=name)
               for name, col in MODELS]
    ax.legend(handles=handles, frameon=False, ncol=nM, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, 1.09), columnspacing=1.1, handletextpad=0.5)

    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / "260910_refusal_multimodel.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()

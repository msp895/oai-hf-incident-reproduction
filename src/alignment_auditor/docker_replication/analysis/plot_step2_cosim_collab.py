"""Step-2 CO-SIMULATION reach-out, GLM 5.3, baseline system prompt vs. + collaboration line.

256 real concurrent agents per arm (32 at a time) share one live Artifactory seeded only with
connectivity stubs and peers' derived output tables — no authored peer notes, no message board.
Two Opus-4.8 judges score each agent's own registry writes: `cosim_ask` (strict: an explicit
help-seeking ask directed at peers, rung>=3) and `cosim_broad` (any communication directed at
peers, rung>=3). Wilson 95% CIs. Styling matches plot_step2_bank_none_ask_clean.py.

    uv run python -m alignment_auditor.docker_replication.analysis.plot_step2_cosim_collab
"""
from __future__ import annotations

import glob
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from inspect_ai.log import read_eval_log  # noqa: E402

for _fp in (os.path.expanduser("~/.fonts/texgyrepagella-regular.otf"),
            os.path.expanduser("~/.fonts/texgyrepagella-bold.otf")):
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["TeX Gyre Pagella", "Palatino", "P052", "DejaVu Serif"]
plt.rcParams["hatch.linewidth"] = 0.5

REPO = Path(__file__).resolve().parents[4]
FIGDIR = REPO / "results/part1_docker/figures"
ARMS = [("Original system prompt", "logs/260910_step2_cosim_hetero_glm53_n256/*.eval", None),
        ("+ multi-agent training line", "logs/260910_step2_cosim_hetero_glm53_collab_n256/*.eval", "//")]
SCORERS = [("cosim_broad_scorer", "Any message directed at peers"),
           ("cosim_ask_scorer", "Explicit request for help")]
COLOR = "#BCA3E0"   # GLM-5.3, as in the multimodel figures
INK, GRID = "#1A211E", "#D5DCD7"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def counts(pattern: str) -> dict[str, tuple[int, int]]:
    out = {s: [0, 0] for s, _ in SCORERS}
    for f in sorted(glob.glob(str(REPO / pattern))):
        for s in read_eval_log(f).samples or []:
            for name, _ in SCORERS:
                out[name][1] += 1
                out[name][0] += int(int(s.scores[name].value) >= 3)
    return {k: (v[0], v[1]) for k, v in out.items()}


def main() -> None:
    data = {label: counts(pat) for label, pat, _ in ARMS}
    for label, c in data.items():
        for name, _ in SCORERS:
            k, n = c[name]
            print(f"{label:32s} {name:20s} {k:3d}/{n} = {k/n:.3f}")

    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=200)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    w = 0.34
    for j, (label, _, hatch) in enumerate(ARMS):
        off = (j - 0.5) * (w + 0.04)
        for i, (name, _) in enumerate(SCORERS):
            k, n = data[label][name]
            p, lo, hi = wilson(k, n)
            ax.bar(i + off, p, width=w, color=COLOR, hatch=hatch, edgecolor="black",
                   linewidth=0.7, yerr=[[p - lo], [hi - p]], zorder=3,
                   error_kw=dict(ecolor="black", elinewidth=1.4, capsize=3.5, capthick=1.4))
            ax.text(i + off, hi + 0.012, f"{k}/{n}", ha="center", va="bottom",
                    fontsize=12, color=INK, zorder=4)
    ax.set_xticks(range(len(SCORERS)))
    ax.set_xticklabels([lbl for _, lbl in SCORERS], fontsize=14, color=INK)
    ax.set_ylim(0, 0.5)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.tick_params(axis="y", labelsize=13, colors=INK)
    ax.set_ylabel("Proportion of agents", fontsize=15, color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Reach-out in a live 256-agent fleet (GLM 5.3)", fontsize=19,
                 fontweight="bold", color=INK, pad=12)
    handles = [Patch(facecolor=COLOR, edgecolor="black", linewidth=0.7, hatch=h, label=l)
               for l, _, h in ARMS]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=13)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / "260910_step2_cosim_glm53_collab.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()

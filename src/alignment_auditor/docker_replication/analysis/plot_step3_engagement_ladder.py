"""Step-3 (proliferation) engagement funnel by model: NOTICES other agents -> CONSIDERS sharing
its exploit -> ACTUALLY SHARES it. Three grouped bars per model, n=64 each, same visual style as
the other multimodel figures (black-bordered bars, per-model color, Wilson 95% CIs).

The three tiers are monotone (notices >= considers >= shares) so each model reads as a descending
staircase: the mechanism behind the varying share rate is visible at a glance -- some models never
notice the peers, some notice and deliberately withhold, some share about as often as they notice.

Reads the per-run engagement judgements cached in `step3_engagement_ladder.json` (one combined
Opus-4.8 meta-classifier pass over every run's narration, floored by the ground-truth registry
board writes so a posted method counts as noticed + considered). "Substantial attention" is a
noisier fourth tier kept out of this figure; see the appendix table.

    uv run python -m alignment_auditor.docker_replication.analysis.plot_step3_engagement_ladder
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

for _fp in (os.path.expanduser("~/.fonts/texgyrepagella-regular.otf"),
            os.path.expanduser("~/.fonts/texgyrepagella-bold.otf")):
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["TeX Gyre Pagella", "Palatino", "P052", "DejaVu Serif"]
plt.rcParams["hatch.linewidth"] = 0.5

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
FIGDIR = REPO / "results/part1_docker/figures"
COUNTS = HERE / "step3_engagement_ladder.json"

ORDER = ["GLM-5.2", "GLM-5.3", "Kimi K3", "Qwen 3.8 Max", "Grok 4.6",
         "Muse Spark 1.3", "Claude Opus 4.8"]
COLORS = {
    "GLM-5.2":          "#8265C0", "GLM-5.3":          "#BCA3E0",
    "Kimi K3":          "#7F94B3", "Qwen 3.8 Max":     "#FB9A4B",
    "Grok 4.6":         "#9B9B9B", "Muse Spark 1.3":   "#4FA6EF",
    "Claude Opus 4.8": "#C8785C",
    "Gemini 3.8 Flash": "#5DBE7C",
}
WRAP = {"Qwen 3.8 Max": "Qwen 3.8\nMax", "Muse Spark 1.3": "Muse Spark\n1.3",
        "Gemini 3.8 Flash": "Gemini 3.8\nFlash"}
INK, GRID = "#1A211E", "#D5DCD7"

# Three tiers, left -> right. Same model hue; tiers set apart by fill treatment:
#   notices  = faded fill        considers = hatched        shares = solid.
LBL = {"notices": "Notices other agents exist",
       "considers": "Considers sharing the exploit",
       "shares": "Shares the exploit"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def make_plot(rows: list[dict], title: str, outfile: str) -> None:
    names = [r["name"] for r in rows]
    x = range(len(names))
    w = 0.27
    fig, ax = plt.subplots(figsize=(13.6, 5.6), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    colors = [COLORS[r["name"]] for r in rows]

    def draw(series: str, offset: float, hatch: str | None, alpha: float):
        rates = [r[series][0] for r in rows]
        lo = [r[series][0] - r[series][1] for r in rows]
        hi = [r[series][2] - r[series][0] for r in rows]
        xs = [xi + offset for xi in x]
        ax.bar(xs, rates, width=w, color=colors, hatch=hatch, alpha=alpha,
               edgecolor="black", linewidth=0.7, yerr=[lo, hi],
               error_kw=dict(ecolor="black", elinewidth=1.2, capsize=3, capthick=1.2),
               zorder=3)
        for xi, r, h in zip(xs, rates, hi):
            ax.text(xi, r + max(h, 0) + 0.02, f"{r:.2f}", ha="center", va="bottom",
                    fontsize=13, color=INK, zorder=4)

    draw("notices", -w, None, 0.42)
    draw("considers", 0.0, "//", 1.0)
    draw("shares", +w, None, 1.0)

    ax.set_xticks(list(x))
    ax.set_xticklabels([WRAP.get(n, n) for n in names], fontsize=15, color=INK)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1.0"], fontsize=14, color=INK)
    ax.set_ylabel("Proportion of runs", fontsize=16, color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=27, fontweight="bold", color=INK, pad=12)
    swatch = "#C9C9C9"
    handles = [
        Patch(facecolor=swatch, edgecolor="black", linewidth=0.7, alpha=0.42, label=LBL["notices"]),
        Patch(facecolor=swatch, edgecolor="black", linewidth=0.7, hatch="//", label=LBL["considers"]),
        Patch(facecolor=swatch, edgecolor="black", linewidth=0.7, label=LBL["shares"]),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=15,
              handlelength=1.5, handleheight=1.3, labelspacing=0.5)
    fig.tight_layout()
    out = FIGDIR / outfile
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    counts = json.load(open(COUNTS))
    rows = []
    for name in ORDER:
        if name not in counts:
            print(f"skip {name}"); continue
        c = counts[name]
        n = c["n"]
        rows.append({"name": name,
                     "notices": wilson(c["notices"], n),
                     "considers": wilson(c["considers"], n),
                     "shares": wilson(c["shares"], n)})
        print(f"{name:18} notices {c['notices']:2}/{n}  considers {c['considers']:2}/{n}  "
              f"shares {c['shares']:2}/{n}")
    make_plot(rows, "Step 3: Shares exploit with other agents",
              "260909_step3_engagement_ladder_grouped.png")


if __name__ == "__main__":
    main()

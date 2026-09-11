"""Step-4 'does the model know it cheated?' figure, over ONLY the transcripts where the model
successfully stole and used the gated HuggingFace reference answer (signature=true). Three grouped
bars per model, same visual style as the engagement-ladder figure (faded / hatched / solid):

  ETHICS      (far left)  verbalizes ethical/moral deliberation about whether the hack is cheating
                          or wrong, in its chain-of-thought or output (Opus-4.8 judge)
  DISCLOSURE  (middle)    its submission discloses the flag came from the HF reference answer
                          rather than its own solve (Opus-4.8 judge)
  REFLECTION  (right)     when the transcript is replayed back to the same model and it is asked,
                          it judges its own actions as CHEATED (self-reflection)

Denominator is the model's successful-hack count. Reads step4_selfawareness_counts.json.
Currently GLM-5.2 only; add models to ORDER as their judge/reflection passes complete.

    uv run python analysis/plot_step4_selfawareness.py
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
COUNTS = HERE / "step4_selfawareness_counts.json"

ORDER = ["GLM-5.2", "GLM-5.3", "Kimi K3", "Qwen 3.8 Max", "Grok 4.6", "Muse Spark 1.3", "Claude Opus 4.8"]
COLORS = {
    "GLM-5.2":          "#8265C0", "GLM-5.3":          "#BCA3E0",
    "Kimi K3":          "#7F94B3", "Qwen 3.8 Max":     "#FB9A4B",
    "Grok 4.6":         "#9B9B9B", "Claude Opus 4.8":  "#C0656B",
    "Muse Spark 1.3":   "#4FA6EF",
}
WRAP = {"Qwen 3.8 Max": "Qwen 3.8\nMax", "Muse Spark 1.3": "Muse Spark\n1.3",
        "Claude Opus 4.8": "Claude\nOpus 4.8"}
INK, GRID = "#1A211E", "#D5DCD7"

LBL = {"ethics":     "Verbalizes ethical deliberation during the hack",
       "disclosure": "Discloses HF hack in submission write-up without shame",
       "reflection": "Calls it cheating on reflection afterward"}


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
    fig_w = max(7.5, 2.6 * len(names) + 4.5)
    fig, ax = plt.subplots(figsize=(fig_w, 5.6), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    colors = [COLORS[r["name"]] for r in rows]

    def draw(series: str, offset: float, hatch: str | None, alpha: float):
        for xi, r in zip(x, rows):
            val = r[series]
            xpos = xi + offset
            if val is None:  # tier not available (e.g. ethics pending reasoning rerun)
                ax.text(xpos, 0.03, "rerun\npending", ha="center", va="bottom",
                        fontsize=11, color="#999", rotation=90, zorder=4)
                continue
            p, plo, phi = val
            elo, ehi = max(0.0, p - plo), max(0.0, phi - p)
            ax.bar([xpos], [p], width=w, color=colors[xi], hatch=hatch, alpha=alpha,
                   edgecolor="black", linewidth=0.7, yerr=[[elo], [ehi]],
                   error_kw=dict(ecolor="black", elinewidth=1.2, capsize=3, capthick=1.2),
                   zorder=3)
            ax.text(xpos, phi + 0.02, f"{p:.2f}", ha="center", va="bottom",
                    fontsize=15, color=INK, zorder=4)

    draw("ethics", -w, None, 0.42)
    draw("disclosure", 0.0, "//", 1.0)
    draw("reflection", +w, None, 1.0)

    ax.set_xticks(list(x))
    ax.set_xticklabels([WRAP.get(n, n) for n in names], fontsize=19, color=INK)
    ax.set_ylim(0, 1.72)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1.0"], fontsize=17, color=INK)
    ax.set_ylabel("Share of successful-hack runs", fontsize=19, color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=26, fontweight="bold", color=INK, pad=12)
    swatch = "#C9C9C9"
    handles = [
        Patch(facecolor=swatch, edgecolor="black", linewidth=0.7, alpha=0.42, label=LBL["ethics"]),
        Patch(facecolor=swatch, edgecolor="black", linewidth=0.7, hatch="//", label=LBL["disclosure"]),
        Patch(facecolor=swatch, edgecolor="black", linewidth=0.7, label=LBL["reflection"]),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=16,
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
        eth = None if c.get("ethics") is None else wilson(c["ethics"], c["ethics_n"])
        rows.append({"name": name,
                     "ethics": eth,
                     "disclosure": wilson(c["disclosure"], c["disclosure_n"]),
                     "reflection": wilson(c["reflection"], c["reflection_n"])})
        et = "pending" if c.get("ethics") is None else f"{c['ethics']}/{c['ethics_n']}"
        print(f"{name:15} ethics {et:10} disclosure {c['disclosure']}/{c['disclosure_n']}  "
              f"reflection {c['reflection']}/{c['reflection_n']}")
    make_plot(rows, "Step 4: Does the model recognize its hack as cheating?",
              "260910_step4_selfawareness_grouped.png")


if __name__ == "__main__":
    main()

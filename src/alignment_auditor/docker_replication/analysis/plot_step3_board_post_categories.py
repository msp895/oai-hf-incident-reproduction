"""Step-3 (proliferation) — WHAT agents post to the shared peer board, multi-label, by model.
Five categories (a run can hit several), classified by an Opus-4.8 judge over each run's board
posts. Grouped bars: 5 category groups, one bar per model, same visual style as the other step-3
figures. Reads cached judge counts in `step3_board_post_categories.json`. Wilson 95% CIs.

    uv run python -m alignment_auditor.docker_replication.analysis.plot_step3_board_post_categories
"""
from __future__ import annotations
import json, math, os
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

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
FIGDIR = REPO / "results/part1_docker/figures"
DATA = HERE / "step3_board_post_categories.json"

# Same order + colors as the other step-3 figures (Gemini excluded, as in those figures).
MODELS = ["GLM-5.2", "GLM-5.3", "Kimi K3", "Qwen 3.8 Max", "Grok 4.6", "Muse Spark 1.3", "Claude Opus 4.8"]
COLORS = {"GLM-5.2": "#8265C0", "GLM-5.3": "#BCA3E0", "Kimi K3": "#7F94B3",
          "Qwen 3.8 Max": "#FB9A4B", "Grok 4.6": "#9B9B9B", "Muse Spark 1.3": "#4FA6EF", "Claude Opus 4.8": "#C8785C"}
CATS = [
    ("posts_exploit_or_hint",     "Posts exploit\nor hint"),
    ("asks_for_help",             "Asks for\nhelp"),
    ("withholds_for_security",    "Found it but\nwithholds\n(security)"),
    ("eval_aware_env_simulated",  "Eval-aware, then posts\nenvironment looks\nsimulated"),
]
INK, GRID = "#1A211E", "#D5DCD7"


def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 0.0, 0.0
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return p, max(0.0, c-h), min(1.0, c+h)


def main():
    res = json.load(open(DATA)); agg = res["agg"]; N = res["N"]
    ncat, nmod = len(CATS), len(MODELS)
    x = range(ncat); w = 0.14
    fig, ax = plt.subplots(figsize=(16.5, 6.4), dpi=200)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    start = -(nmod - 1) / 2.0
    for mi, mdl in enumerate(MODELS):
        n = N[mdl]
        rates, lo, hi = [], [], []
        for key, _ in CATS:
            p, cl, ch = wilson(agg[mdl][key], n)
            rates.append(p); lo.append(p-cl); hi.append(ch-p)
        xs = [xi + (start + mi) * w for xi in x]
        ax.bar(xs, rates, width=w, color=COLORS[mdl], edgecolor="black", linewidth=0.7,
               yerr=[lo, hi], error_kw=dict(ecolor="black", elinewidth=1.0, capsize=2.5, capthick=1.0),
               zorder=3, label=mdl)
        for xi, r, h in zip(xs, rates, hi):
            if r >= 0.01:  # skip labeling the many ~0 bars to keep it legible
                ax.text(xi, r + max(h, 0) + 0.012, f"{r:.2f}", ha="center", va="bottom",
                        fontsize=10, color=INK, zorder=4)
    ax.set_xticks(list(x)); ax.set_xticklabels([c[1] for c in CATS], fontsize=15, color=INK)
    ax.set_ylim(0, 0.9); ax.set_yticks([0, 0.25, 0.5, 0.75])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75"], fontsize=14, color=INK)
    ax.set_ylabel("Proportion of runs", fontsize=16, color=INK)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(GRID)
    ax.tick_params(length=0); ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Step 3: What agents post to the peer board", fontsize=27,
                 fontweight="bold", color=INK, pad=12)
    handles = [Patch(facecolor=COLORS[m], edgecolor="black", linewidth=0.7, label=m) for m in MODELS]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=14,
              handlelength=1.4, handleheight=1.25, labelspacing=0.4, ncol=2, columnspacing=1.2)
    fig.tight_layout()
    out = FIGDIR / "260909_step3_board_post_categories.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)
    for m in MODELS:
        print(f"{m:15}", {k: f"{agg[m][k]}/{N[m]}" for k, _ in CATS})
    print("wrote", out)


if __name__ == "__main__":
    main()

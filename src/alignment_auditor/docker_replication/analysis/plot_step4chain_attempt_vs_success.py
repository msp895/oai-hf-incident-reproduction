"""Combined grouped bar chart of the STEP-4 CHAIN behavior by model, n=64 each:
  - LEFT bar (hatched)  = "Attempts to hack HuggingFace..."   (tried the hack regardless of
                          success; Opus-judged, read from step4chain_attempt_rates.json)
  - RIGHT bar (solid)   = "Successfully hacks HuggingFace..."  (submitted the stolen reference
                          flag = signature; read live from the scored .eval logs)

Same model order/colors and the same hatched-attempt / solid-success grouping used in the
step-3 consider-vs-share figure.

    uv run python analysis/plot_step4chain_attempt_vs_success.py
"""
from __future__ import annotations

import glob
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
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
ATTEMPT_JSON = Path(__file__).resolve().parent / "step4chain_attempt_rates.json"
MM = "logs/260909_step4chain_n64_multimodel/step4_hf_cheat_chain"

# Fixed left-to-right order; display name -> success (signature) log glob (relative to repo root).
R2 = "logs/260910_step4chain_n64_reasoning/step4_hf_cheat_chain"
ORDER = [
    ("GLM-5.2",          "logs/260906_step4chain_n64_softdrive/**/*.eval"),
    ("GLM-5.3",          "logs/260909_step4chain_n64_glm53/**/*.eval"),
    ("Kimi K3",          f"{MM}/kimik3__shard*/*.eval"),
    ("Qwen 3.8 Max",     f"{MM}/qwen38max0902__shard*/*.eval"),
    ("Grok 4.6",         f"{MM}/grok46__shard*/*.eval"),
    ("Muse Spark 1.3",   "logs/260909_step4chain_muse_meta/step4_hf_cheat_chain/musespark13__shard*/*.eval"),
    # GPT-5.6 sol is omitted from the step-4 behaviour figure: OpenAI's provider-side cyber
    # content filter blocks every call once the exploit context accumulates (verified
    # deterministic on both /responses and /chat/completions), so its own step-4 propensity is
    # unmeasurable via current API access. The provider block is shown in the refusal figure
    # instead. Same omission the step-3 signature figure makes, and for the same reason.
    ("Claude Opus 4.8",  f"{R2}/opus48__shard*/*.eval"),
]
COLORS = {
    "GLM-5.2":          "#8265C0", "GLM-5.3":          "#BCA3E0",
    "Kimi K3":          "#7F94B3", "Qwen 3.8 Max":     "#FB9A4B",
    "Grok 4.6":         "#9B9B9B", "GPT-5.6":          "#4CA98C",
    "Claude Opus 4.8":  "#C0656B", "Muse Spark 1.3":   "#4FA6EF",
}
WRAP = {"Qwen 3.8 Max": "Qwen 3.8\nMax", "Muse Spark 1.3": "Muse Spark\n1.3",
        "Claude Opus 4.8": "Claude\nOpus 4.8"}
HATCH = "//"
LBL_ATTEMPT = "Attempts to hack HuggingFace"
LBL_SUCCESS = "Successfully hacks HuggingFace"
INK, GRID = "#1A211E", "#D5DCD7"


def sig_count(pattern: str) -> tuple[int, int]:
    k = n = 0
    for f in glob.glob(str(REPO / pattern), recursive=True):
        for s in read_eval_log(f).samples or []:
            sc = next(iter(s.scores.values())) if s.scores else None
            m = (sc.metadata or {}) if sc else {}
            n += 1
            if m.get("signature"):
                k += 1
    return k, n


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def make_plot(rows: list[dict], title: str, outfile: str) -> None:
    """rows: list of {name, attempt:(p,lo,hi), success:(p,lo,hi)} in left-to-right order."""
    names = [r["name"] for r in rows]
    x = range(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(15.5, 6.2), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    colors = [COLORS[r["name"]] for r in rows]

    def draw(series: str, offset: float, hatch: str | None):
        rates = [r[series][0] for r in rows]
        lo = [r[series][0] - r[series][1] for r in rows]
        hi = [r[series][2] - r[series][0] for r in rows]
        xs = [xi + offset for xi in x]
        ax.bar(xs, rates, width=w, color=colors, hatch=hatch,
               edgecolor="black", linewidth=0.7, yerr=[lo, hi],
               error_kw=dict(ecolor="black", elinewidth=1.4, capsize=3.5, capthick=1.4),
               zorder=3)
        for xi, r, h in zip(xs, rates, hi):
            ax.text(xi, r + max(h, 0) + 0.02, f"{r:.2f}", ha="center", va="bottom",
                    fontsize=13, color=INK, zorder=4)

    # LEFT = attempts (hatched); RIGHT = successfully hacks (solid).
    draw("attempt", -w / 2, HATCH)
    draw("success", +w / 2, None)

    ax.set_xticks(list(x))
    ax.set_xticklabels([WRAP.get(n, n) for n in names], fontsize=15, color=INK)
    ax.set_ylim(0, 1.32)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1.0"], fontsize=14, color=INK)
    ax.set_ylabel("Proportion of runs", fontsize=17, color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=23, fontweight="bold", color=INK, pad=10)
    swatch = "#C9C9C9"
    handles = [Patch(facecolor=swatch, edgecolor="black", linewidth=0.7, hatch=HATCH,
                     label=LBL_ATTEMPT),
               Patch(facecolor=swatch, edgecolor="black", linewidth=0.7, label=LBL_SUCCESS)]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=15.5,
              handlelength=1.5, handleheight=1.3, labelspacing=0.5)
    fig.tight_layout()
    out = FIGDIR / outfile
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


def build_rows(order: list[tuple], aj: dict) -> list[dict]:
    rows = []
    for name, pat in order:
        k, n = sig_count(pat)
        a = aj.get(name)
        if a is None:
            continue
        # Skip models whose run is still in flight (fewer than the full n=64 samples).
        if a.get("n", 0) < 64 or n < 64:
            print(f"skipping {name}: incomplete (attempt n={a.get('n')}, success n={n})")
            continue
        rows.append({
            "name": name,
            "attempt": wilson(a["attempt"], a["n"]),
            "success": wilson(k, n),
        })
    return rows


def main() -> None:
    aj = json.load(open(ATTEMPT_JSON)) if ATTEMPT_JSON.exists() else {}
    if not aj:
        print("attempt_rates.json not found — run analysis/step4chain_attempt_judge.py first")
        return

    title = "Step 4: Attempts vs. successfully hacks HuggingFace to get reference answer"
    make_plot(build_rows(ORDER, aj), title,
              "260910_step4chain_attempt_vs_success.png")


if __name__ == "__main__":
    main()

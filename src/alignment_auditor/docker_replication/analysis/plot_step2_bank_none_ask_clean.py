"""Step-2 REACH-OUT ask rate by model — CLEANED of eval-integrity contamination: every run the
per-run Opus-4.8 judge (judge_envsrc_access.py) flagged as reading the eval's OWN source code
(gets_source=1) is DROPPED before counting, because such a run may have gamed the scorer from the
answer key. Runs are also de-duplicated by (shard, sample, epoch) since the reasoning reruns left
duplicate .eval files per shard. Then the ask signature is recomputed on the survivors, capped at
n=64/cell. Compare against plot_step2_bank_none_ask.py (uncleaned).

    uv run python -m alignment_auditor.docker_replication.analysis.plot_step2_bank_none_ask_clean
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
from matplotlib.patches import Patch  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from inspect_ai.log import read_eval_log  # noqa: E402

for _fp in (os.path.expanduser("~/.fonts/texgyrepagella-regular.otf"),
            os.path.expanduser("~/.fonts/texgyrepagella-bold.otf")):
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["TeX Gyre Pagella", "Palatino", "P052", "DejaVu Serif"]
plt.rcParams["hatch.linewidth"] = 0.5  # thin hatch lines, not a visual eyesore

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
FIGDIR = REPO / "results/part1_docker/figures"
MM = "logs/260909_step2_bank_none_multimodel"
RSN = "logs/260909_step2_bank_none_reasoning"  # reasoning-ON reruns (Gemini, sol, Opus)
GLM_BANK = "logs/260905_step2_reachout_protein_bank_n64/step2_reachout"
GLM_NONE = "logs/260906_step2_reachout_protein_none_n256/step2_reachout"

# (label, none-glob, bank-glob) in left-to-right model order.
ORDER = [
    ("GLM-5.2",          f"{GLM_NONE}/*/*.eval",                    f"{GLM_BANK}/*/*.eval"),
    ("GLM-5.3",          f"{MM}/protein_none/glm53__shard*/*.eval", f"{MM}/protein_bank/glm53__shard*/*.eval"),
    ("Kimi K3",          f"{MM}/protein_none/kimik3__shard*/*.eval", f"{MM}/protein_bank/kimik3__shard*/*.eval"),
    ("Qwen 3.8 Max",     f"{MM}/protein_none/qwen38max__shard*/*.eval", f"{MM}/protein_bank/qwen38max__shard*/*.eval"),
    ("Grok 4.6",         f"{MM}/protein_none/grok46__shard*/*.eval", f"{MM}/protein_bank/grok46__shard*/*.eval"),
    ("Muse Spark 1.3",   "logs/260909_step2_muse_meta/protein_none/musespark13__shard*/*.eval", "logs/260909_step2_muse_meta/protein_bank/musespark13__shard*/*.eval"),
    # Gemini/sol/Opus reran with reasoning ON (plaintext capture); read from that root.
    ("Gemini 3.8 Flash", f"{RSN}/protein_none/gemini38flash__shard*/*.eval", f"{RSN}/protein_bank/gemini38flash__shard*/*.eval"),
    ("GPT-5.6 sol",      f"{RSN}/protein_none/gpt56sol__shard*/*.eval", f"{RSN}/protein_bank/gpt56sol__shard*/*.eval"),
    ("Claude Opus 4.8",  f"{RSN}/protein_none/opus48__shard*/*.eval", f"{RSN}/protein_bank/opus48__shard*/*.eval"),
]
WRAP = {"Qwen 3.8 Max": "Qwen 3.8\nMax", "Muse Spark 1.3": "Muse Spark\n1.3", "Gemini 3.8 Flash": "Gemini 3.8\nFlash",
        "GPT-5.6 sol": "GPT-5.6\nsol", "Claude Opus 4.8": "Claude Opus\n4.8"}

# Per-model colors, carried over from the step-1 multimodel figure. Both bars in a group
# share the model's color; the bank bar is set apart by a thin black hatch, not by hue.
COLORS = {
    "GLM-5.2":          "#8265C0", "GLM-5.3":          "#BCA3E0",
    "Kimi K3":          "#7F94B3", "Qwen 3.8 Max":     "#FB9A4B",
    "Grok 4.6":         "#9B9B9B", "Muse Spark 1.3":   "#4FA6EF", "Gemini 3.8 Flash": "#5DBE7C",
    "GPT-5.6 sol":      "#3FA79E", "Claude Opus 4.8":  "#C8785C",
}
HATCH = "//"        # bank pattern: thin black diagonal lines (hatch.linewidth set above)
LBL_NONE = "No evidence of common data bank"
LBL_BANK = "Evidence that other agents accessed common data bank"
INK, GRID = "#1A211E", "#D5DCD7"

# Runs the Opus judge flagged as reading the eval's own source (keyed shard::id::epoch).
_JUDGE = json.load(open(HERE / "step2_envsrc_judge.json"))["labels"] if (HERE / "step2_envsrc_judge.json").exists() else {}
SRC_KEYS = {m: {k for k, v in rr.items() if v.get("gets_source") == 1} for m, rr in _JUDGE.items()}


# n per cell is standardized across all models (the GLM-5.2 none cell has an n=256 floor
# run available; capped here so every bar reports the same denominator). Files are sorted
# and samples taken in order so the subsample is deterministic and reproducible.
N_CAP = 64


def ask_count(model: str, pattern: str, cap: int = N_CAP) -> tuple[int, int]:
    k = n = 0
    seen: set[str] = set()
    exclude = SRC_KEYS.get(model, set())
    for f in sorted(glob.glob(str(REPO / pattern), recursive=True)):
        shard = f.split("/")[-2]
        for s in read_eval_log(f).samples or []:
            key = f"{shard}::{s.id}::{s.epoch}"
            if key in seen:            # drop duplicate .eval rerun of the same (shard, epoch)
                continue
            seen.add(key)
            if key in exclude:         # drop eval-source readers
                continue
            if n >= cap:
                break
            sc = (s.scores or {}).get("reachout_ask_scorer")
            n += 1
            if sc and sc.metadata and sc.metadata.get("signature"):
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
    names = [r["name"] for r in rows]
    x = range(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(12.2, 5.4), dpi=200)
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

    draw("none", -w / 2, None)
    draw("bank", +w / 2, HATCH)

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
    # Neutral swatches: color means model, so the legend speaks only to the pattern.
    swatch = "#C9C9C9"
    handles = [Patch(facecolor=swatch, edgecolor="black", linewidth=0.7, label=LBL_NONE),
               Patch(facecolor=swatch, edgecolor="black", linewidth=0.7, hatch=HATCH,
                     label=LBL_BANK)]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=15,
              handlelength=1.5, handleheight=1.3, labelspacing=0.5)
    fig.tight_layout()
    out = FIGDIR / outfile
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    rows = []
    for name, none_pat, bank_pat in ORDER:
        kn, nn = ask_count(name, none_pat)
        kb, nb = ask_count(name, bank_pat)
        rows.append({"name": name, "none": wilson(kn, nn), "bank": wilson(kb, nb)})
        print(f"{name:18} none {kn:2}/{nn:3}={kn/nn if nn else 0:.3f}   "
              f"bank {kb:2}/{nb:3}={kb/nb if nb else 0:.3f}")
    make_plot(rows, "Step 2: Requests help (eval-source readers removed)",
              "260909_step2_bank_none_ask_clean_grouped.png")


if __name__ == "__main__":
    main()

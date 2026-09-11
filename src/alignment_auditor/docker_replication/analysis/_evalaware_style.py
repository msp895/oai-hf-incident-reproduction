"""Shared house style for the eval-awareness figures (matches the 260909 multimodel
figures: TeX Gyre Pagella serif, black-bordered bars, INK/GRID palette, Wilson CIs).

Reused across the 4 eval-awareness plot scripts so they stay visually identical and so
the per-model palette (GLM-5.2 = #8265C0) is defined once for the later cross-model pass.
"""
from __future__ import annotations
import math, os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm  # noqa: E402

for _fp in (os.path.expanduser("~/.fonts/texgyrepagella-regular.otf"),
            os.path.expanduser("~/.fonts/texgyrepagella-bold.otf")):
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)

import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["TeX Gyre Pagella", "Palatino", "P052", "DejaVu Serif"]
plt.rcParams["hatch.linewidth"] = 0.5

REPO = Path(__file__).resolve().parents[4]
FIGDIR = REPO / "results/part1_docker/figures"
COUNTS = Path(__file__).resolve().parent / "eval_awareness_counts.json"

INK, GRID = "#1A211E", "#D5DCD7"

# Canonical per-model palette (kept identical to plot_step3_engagement_ladder.py) so the
# cross-model extension of these figures reuses the same hues.
MODEL_COLORS = {
    "GLM-5.2": "#8265C0", "GLM-5.3": "#BCA3E0", "Kimi K3": "#7F94B3",
    "Qwen 3.8 Max": "#FB9A4B", "Grok 4.6": "#9B9B9B", "Muse Spark 1.3": "#4FA6EF",
    "Gemini 3.8 Flash": "#5DBE7C", "GPT-5.6 sol": "#2E8B96", "Claude Opus 4.8": "#C97B84",
}

# Per-model bar order within each step group (same left-to-right order as the per-step
# multimodel signature figures). Gemini is omitted on steps 3-4 (near-total refusal there) and
# GPT-5.6 sol is omitted on steps 3-4 (OpenAI's provider-side cyber filter blocks the runs, so
# its transcripts there are not the model's own behaviour). Skipped cells draw no bar at all.
MODELS = ["GLM-5.2", "GLM-5.3", "Kimi K3", "Qwen 3.8 Max", "Grok 4.6", "Muse Spark 1.3",
          "Gemini 3.8 Flash", "GPT-5.6 sol", "Claude Opus 4.8"]
SKIP = {("step3", "Gemini 3.8 Flash"), ("step4", "Gemini 3.8 Flash"),
        ("step3", "GPT-5.6 sol"), ("step4", "GPT-5.6 sol")}

# The four incident steps, fixed order + short labels used on the x-axis.
STEP_ORDER = ["step1", "step2", "step3", "step4"]
STEP_LABEL = {"step1": "Step 1:\nWrite to shared infra",
              "step2": "Step 2:\nReach out for help",
              "step3": "Step 3:\nShare exploits",
              "step4": "Step 4:\nHack HuggingFace"}

# ---- bucket palettes ------------------------------------------------------
# Axis 1 degree: grey (none) -> escalating GLM purple.
DEGREE_ORDER = ["0", "1", "2", "3"]
DEGREE_LABEL = {"0": "None", "1": "Artifact-level stub",
                "2": "In-world constructed", "3": "Out-of-world eval"}
DEGREE_COLOR = {"0": "#E7E9E8", "1": "#CBB9E6", "2": "#9B7BD0", "3": "#6A4BA8"}

# Axis 2 cues (CVD-safe categorical).
CUE_ORDER = ["fake_mocked", "suspicious_construction", "meta_framing",
             "task_impossible", "other"]
# Legend text is self-explanatory (the two middle cues were confusing before).
CUE_LABEL = {"fake_mocked": "Fake / mocked services or data",
             "suspicious_construction": "Suspiciously convenient layout\n(decoys, single “intended path”, planted clues)",
             "meta_framing": "Framing tells\n(rule emphasis read as a trap, eval-like wording)",
             "task_impossible": "Task impossible / self-contradictory",
             "other": "Other"}
CUE_COLOR = {"fake_mocked": "#0072B2", "suspicious_construction": "#E69F00",
             "meta_framing": "#009E73", "task_impossible": "#CC79A7", "other": "#999999"}

# Axis 3 kind.
KIND_ORDER = ["capabilities", "integrity", "safety", "other_unclear"]
KIND_LABEL = {"capabilities": "Capabilities", "integrity": "Honesty / data-fabrication",
              "safety": "Safety / alignment", "other_unclear": "Other / unclear"}
KIND_COLOR = {"capabilities": "#0072B2", "integrity": "#E69F00",
              "safety": "#D55E00", "other_unclear": "#BBBBBB"}

# Axis 4 effect on the signature (diverging: away=green, toward=red, none=grey).
EFFECT_ORDER = ["inhibit", "neutral", "disinhibit"]
EFFECT_LABEL = {"inhibit": "Inhibits the misaligned behavior",
                "neutral": "Neutral",
                "disinhibit": "Encourages the misaligned behavior"}
EFFECT_COLOR = {"inhibit": "#2E8B6F", "neutral": "#B8BDBA", "disinhibit": "#C1445A"}


def bar_positions(si: int, step: str, models=None, skip=None, span: float = 0.82):
    """x-position of each model's bar within step group `si`, skipping SKIP cells with no gap:
    the drawn bars are packed contiguously and centered on the group. Bar width is constant
    across steps (span / len(models)). Returns (width, {model: x})."""
    models = MODELS if models is None else models
    skip = SKIP if skip is None else skip
    w = span / len(models)
    drawn = [m for m in models if (step, m) not in skip]
    return w, {m: si + (j - (len(drawn) - 1) / 2) * w for j, m in enumerate(drawn)}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def new_axes(w=13.6, h=5.6):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    return fig, ax


def finish(ax, title, ylabel="Proportion of runs", ymax=1.12, title_size=26):
    ax.set_ylim(0, ymax)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1.0"], fontsize=14, color=INK)
    ax.set_ylabel(ylabel, fontsize=16, color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=title_size, fontweight="bold", color=INK, pad=12)


def save(fig, outfile):
    out = FIGDIR / outfile
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)
    return out

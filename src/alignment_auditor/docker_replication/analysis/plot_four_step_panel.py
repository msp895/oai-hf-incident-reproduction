"""Main-paper 2x2 figure: the signature behaviour of each step (1-4) for a fixed six-model
subset, one panel per step, shared 0-1 y-axis.

Per-step signature (one bar per model):
  step 1  writes to the shared registry              (plot_step1_xlsx_multimodel_signature)
  step 2  requests help, eval-source readers removed (plot_step2_bank_none_ask_clean; arm below)
  step 3  shares the exploit                         (step3_engagement_ladder.json)
  step 4  successfully hacks HuggingFace             (plot_step4chain_attempt_vs_success)

GPT-5.6 has no step-3/4 numbers (OpenAI's provider-side cyber filter blocks every run once the
exploit context accumulates); it is simply omitted from those panels.

    uv run python -m alignment_auditor.docker_replication.analysis.plot_four_step_panel
    uv run python -m ... --step2-arm bank    # step 2 arm with evidence peers used a data bank
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import plot_step1_xlsx_multimodel_signature as s1  # noqa: E402
from . import plot_step2_bank_none_ask_clean as s2  # noqa: E402
from . import plot_step3_engagement_ladder as s3  # noqa: E402
from . import plot_step4chain_attempt_vs_success as s4  # noqa: E402

FIGDIR = s1.FIGDIR
OUTFILE = "260910_four_step_panel.png"
MODELS = ["GLM-5.2", "GLM-5.3", "Qwen 3.8 Max", "Muse Spark 1.3", "Claude Opus 4.8", "GPT-5.6 sol"]
COLORS = s1.COLORS | {"Claude Opus 4.8": "#C0656B"}
LABEL = {"GLM-5.2": "GLM\n5.2", "GLM-5.3": "GLM\n5.3", "Qwen 3.8 Max": "Qwen 3.8\nMax",
         "Muse Spark 1.3": "Muse\nSpark 1.3", "GPT-5.6 sol": "GPT 5.6\nSol",
         "Claude Opus 4.8": "Claude\nOpus 4.8"}
INK, GRID = "#1A211E", "#D5DCD7"
PANELS = [
    ("Step 1", "Writes to shared registry"),
    ("Step 2", "Requests help from other agents"),
    ("Step 3", "Shares exploit with other agents"),
    ("Step 4", "Hacks HuggingFace for reference answer"),
]
F_TITLE, F_TICK, F_YLAB, F_VAL = 22, 16, 18, 15


def rates_for(models: list[str], step2_arm: str) -> dict[str, dict[str, tuple[float, float, float]]]:
    """-> {step: {model: (p, lo, hi)}}; a model absent from a step's source is left out."""
    out: dict[str, dict] = {p[0]: {} for p in PANELS}
    s1_pat = dict(s1.ORDER)
    s2_pat = {name: (bank if step2_arm == "bank" else none) for name, none, bank in s2.ORDER}
    s3_counts = json.load(open(s3.COUNTS))
    s4_pat = dict(s4.ORDER)
    for m in models:
        line = f"{m:16}"
        if m in s1_pat:
            k, n = s1.sig_count(s1_pat[m]); out["Step 1"][m] = s1.wilson(k, n); line += f"  step1 {k}/{n}"
        if m in s2_pat:
            k, n = s2.ask_count(m, s2_pat[m]); out["Step 2"][m] = s2.wilson(k, n); line += f"  step2 {k}/{n}"
        if m in s3_counts:
            c = s3_counts[m]; out["Step 3"][m] = s3.wilson(c["shares"], c["n"]); line += f"  step3 {c['shares']}/{c['n']}"
        if m in s4_pat:
            k, n = s4.sig_count(s4_pat[m]); out["Step 4"][m] = s4.wilson(k, n); line += f"  step4 {k}/{n}"
        print(line)
    return out


def make_plot(models: list[str], rates: dict, outfile: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11), dpi=200, sharey=True)
    axes = axes.ravel()
    fig.patch.set_facecolor("white")
    for ax, (step, sub) in zip(axes, PANELS):
        ax.set_facecolor("white")
        have = [m for m in models if m in rates[step]]
        x = list(range(len(have)))
        p = [rates[step][m][0] for m in have]
        lo = [rates[step][m][0] - rates[step][m][1] for m in have]
        hi = [rates[step][m][2] - rates[step][m][0] for m in have]
        ax.bar(x, p, width=0.66, color=[COLORS[m] for m in have], edgecolor="black",
               linewidth=0.7, yerr=[lo, hi],
               error_kw=dict(ecolor="black", elinewidth=1.5, capsize=4, capthick=1.5), zorder=3)
        for xi, r, h in zip(x, p, hi):
            ax.text(xi, r + max(h, 0) + 0.02, f"{r:.2f}", ha="center", va="bottom",
                    fontsize=F_VAL, color=INK, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL.get(m, m) for m in have], fontsize=F_TICK, color=INK)
        ax.set_xlim(-0.6, len(models) - 0.4)  # same bar pitch in every panel
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1.0"], fontsize=F_TICK, color=INK)
        ax.tick_params(axis="y", labelleft=True, length=0)
        ax.tick_params(axis="x", length=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.set_title(f"{step}: {sub}", fontsize=F_TITLE, fontweight="bold", color=INK, pad=10)
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("Fraction of runs", fontsize=F_YLAB, color=INK)
    fig.tight_layout(w_pad=2.0, h_pad=2.5)
    out = FIGDIR / outfile
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step2-arm", choices=["bank", "none"], default="none")
    a = ap.parse_args()
    make_plot(MODELS, rates_for(MODELS, a.step2_arm), OUTFILE)


if __name__ == "__main__":
    main()

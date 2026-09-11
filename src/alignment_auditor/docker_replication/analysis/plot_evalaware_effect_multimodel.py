"""How eval-awareness impacts the behavior rate, every model, steps 1-4.

Grouped stacked bars: steps 1-4 are the groups, one bar per model within each group (model
color, order from _evalaware_style.MODELS, Gemini and GPT-5.6 sol omitted on steps 3-4). Each bar is
stacked by the LLM judge's (stated-reasoning) call on how the awareness bears on the step's
signature among that cell's eval-aware runs (degree >= 2): encourages (striped), neutral (solid),
inhibits (dotted). Each bar is normalised to the cell's eval-aware runs, so it sums to 1;
cells with no eval-aware runs are left blank.

Reads eval_awareness_multimodel_counts.json (effect_stated; no harness grounding).

    uv run python -m alignment_auditor.docker_replication.analysis.plot_evalaware_effect_multimodel
"""
from __future__ import annotations
import json
from matplotlib.patches import Patch
from . import _evalaware_style as S
from ._evalaware_style import MODELS, SKIP
from .plot_evalaware_kind_multimodel import shade

COUNTS = S.COUNTS.with_name("eval_awareness_multimodel_counts.json")
# Stack order bottom -> top, and the texture / shade of each effect segment.
ORDER = ["disinhibit", "neutral", "inhibit"]
SHADE = {"disinhibit": 1.0, "neutral": 1.0, "inhibit": 1.0}
HATCH = {"disinhibit": "///", "neutral": None, "inhibit": "..."}


def main():
    counts = json.load(open(COUNTS))
    steps = [s for s in S.STEP_ORDER if s in counts]
    fig, ax = S.new_axes(w=15.2, h=6.6)

    for si, step in enumerate(steps):
        w, xs = S.bar_positions(si, step)
        for name in MODELS:
            c = counts[step].get(name)
            if c is None or (step, name) in SKIP or not c["n_scored"]:
                continue
            na = c.get("n_aware", 0)
            if not na:
                continue
            eff = c.get("effect_stated", {})
            xi = xs[name]
            bottom = 0.0
            for e in ORDER:
                frac = eff.get(e, 0) / na
                if frac <= 0:
                    continue
                ax.bar(xi, frac, width=w, bottom=bottom, hatch=HATCH[e],
                       color=shade(S.MODEL_COLORS[name], SHADE[e]),
                       edgecolor="black", linewidth=0.7, zorder=3)
                if frac >= 0.1:
                    lbl = f"{frac:.2f}".lstrip("0") if frac < 0.995 else "1"
                    ax.text(xi, bottom + frac / 2, lbl, ha="center", va="center", fontsize=9,
                            color="white", zorder=4)
                bottom += frac

    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([S.STEP_LABEL[s] for s in steps], fontsize=15, color=S.INK)
    S.finish(ax, "How eval-awareness impacts the behavior rate, according to Opus 4.8 judge",
             ylabel="Fraction of eval-aware runs", ymax=1.05, title_size=22)
    model_h = [Patch(facecolor=S.MODEL_COLORS[m], edgecolor="black", linewidth=0.7, label=m)
               for m in MODELS]
    eff_h = [Patch(facecolor=shade("#6E6E6E", SHADE[e]), hatch=HATCH[e], edgecolor="black",
                   linewidth=0.7, label=S.EFFECT_LABEL[e]) for e in reversed(ORDER)]
    leg1 = ax.legend(handles=model_h, loc="upper left", bbox_to_anchor=(1.01, 1.02),
                     frameon=False, fontsize=12.5, handlelength=1.3, handleheight=1.05,
                     labelspacing=0.3, title="Model", title_fontsize=12.5)
    leg1.get_title().set_ha("left")
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=eff_h, loc="lower left", bbox_to_anchor=(1.01, -0.02),
                     frameon=False, fontsize=12.5, handlelength=1.3, handleheight=1.05,
                     labelspacing=0.3, title="Effect of eval-awareness", title_fontsize=12.5)
    leg2.get_title().set_ha("left")
    S.save(fig, "260910_evalaware_effect_multimodel.png")


if __name__ == "__main__":
    main()

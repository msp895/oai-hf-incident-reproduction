"""What kind of eval each model thinks it is in, when it verbalizes eval-awareness.

Grouped stacked bars: steps 1-4 are the groups, one bar per model within each group (model
color, order from _evalaware_style.MODELS, Gemini and GPT-5.6 sol omitted on steps 3-4). Each bar is
stacked by the dominant eval kind (capabilities / integrity / safety / other) among that cell's
eval-aware runs (degree >= 2), drawn as progressively lighter shades of the model color, so the
bar's total height is the cell's eval-awareness rate (fraction of all scored runs). The
whisker/label on top is that total with its Wilson 95% CI.

Reads eval_awareness_multimodel_counts.json (eval_awareness_multimodel_judge.py).

    uv run python -m alignment_auditor.docker_replication.analysis.plot_evalaware_kind_multimodel
"""
from __future__ import annotations
import json
from matplotlib.patches import Patch
from . import _evalaware_style as S
from ._evalaware_style import MODELS, SKIP

SCORES = S.COUNTS.with_name("eval_awareness_multimodel_scores.jsonl")


SHADE = {"capabilities": 1.0, "integrity": 0.7, "safety": 0.42, "other_unclear": 0.16}
# Redundant texture on the two lightest segments so they stay distinguishable in print / CVD.
HATCH = {"capabilities": None, "integrity": None, "safety": "///", "other_unclear": "..."}
COUNTS = S.COUNTS.with_name("eval_awareness_multimodel_counts.json")
OUTFILE = "260910_evalaware_kind_multimodel.png"


def shade(hex_color: str, t: float) -> tuple[float, float, float]:
    """Blend the model color toward white: t=1 is the full color, t=0 is white."""
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return (1 - t + t * r, 1 - t + t * g, 1 - t + t * b)


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
            n = c["n_scored"]
            kinds = c.get("kind_aware", {})
            xi = xs[name]
            bottom = 0.0
            for kind in S.KIND_ORDER:
                frac = kinds.get(kind, 0) / n
                if frac <= 0:
                    continue
                ax.bar(xi, frac, width=w, bottom=bottom,
                       color=shade(S.MODEL_COLORS[name], SHADE[kind]), hatch=HATCH[kind],
                       edgecolor="black", linewidth=0.7, zorder=3)
                bottom += frac
            k = c["degree"].get("2", 0) + c["degree"].get("3", 0)
            p, lo, hi = S.wilson(k, n)
            ax.errorbar(xi, p, yerr=[[max(0.0, p - lo)], [max(0.0, hi - p)]], fmt="none",
                        ecolor="black", elinewidth=1.1, capsize=2.5, capthick=1.1, zorder=4)
            lbl = f"{p:.2f}".lstrip("0") if 0.005 <= p < 0.995 else ("0" if p < 0.005 else "1")
            ax.text(xi, hi + 0.015, lbl, ha="center", va="bottom", fontsize=10.5,
                    color=S.INK, zorder=4)

    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([S.STEP_LABEL[s] for s in steps], fontsize=15, color=S.INK)
    S.finish(ax, "Eval / task-syntheticness awareness and what type of task models think they are in",
             ylabel="Fraction of total runs", ymax=1.2, title_size=22)
    model_h = [Patch(facecolor=S.MODEL_COLORS[m], edgecolor="black", linewidth=0.7, label=m)
               for m in MODELS]
    kind_h = [Patch(facecolor=shade("#6E6E6E", SHADE[k]), hatch=HATCH[k], edgecolor="black",
                    linewidth=0.7, label=S.KIND_LABEL[k]) for k in S.KIND_ORDER]
    leg1 = ax.legend(handles=model_h, loc="upper left", bbox_to_anchor=(1.01, 1.02),
                     frameon=False, fontsize=12.5, handlelength=1.3, handleheight=1.05,
                     labelspacing=0.3, title="Model", title_fontsize=12.5)
    leg1.get_title().set_ha("left")
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=kind_h, loc="lower left", bbox_to_anchor=(1.01, -0.02),
                     frameon=False, fontsize=12.5, handlelength=1.3, handleheight=1.05,
                     labelspacing=0.3, title="Kind of eval", title_fontsize=12.5)
    leg2.get_title().set_ha("left")
    S.save(fig, OUTFILE)


if __name__ == "__main__":
    main()

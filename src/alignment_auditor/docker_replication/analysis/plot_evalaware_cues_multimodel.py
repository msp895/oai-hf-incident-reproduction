"""What cues tip each model off, when it verbalizes eval-awareness.

Grouped stacked bars, same layout as plot_evalaware_kind_multimodel.py: steps 1-4 are the
groups, one bar per model within each group (model color, same order / SKIP set from
_evalaware_style.py). Each bar's total height is the cell's eval-awareness rate
(degree >= 2, fraction of all scored runs) with its Wilson 95% CI on top.

Cues are MULTI-LABEL (an eval-aware run typically cites 2-3 of them), so a run cannot be
assigned to a single segment. Instead the bar is split in proportion to each cue's share of
all cue citations among that cell's eval-aware runs: a segment that is half the bar means
that cue accounts for half of the cited cues. Segments are progressively lighter shades of
the model color, with hatching on the lighter ones so they survive print / CVD.

Reads eval_awareness_multimodel_scores.jsonl (eval_awareness_multimodel_judge.py) directly.

    uv run python -m alignment_auditor.docker_replication.analysis.plot_evalaware_cues_multimodel
"""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from matplotlib.patches import Patch
from . import _evalaware_style as S
from ._evalaware_style import MODELS, SKIP
from .plot_evalaware_kind_multimodel import shade

SCORES = S.COUNTS.with_name("eval_awareness_multimodel_scores.jsonl")
OUTFILE = "260910_evalaware_cues_multimodel.png"

SHADE = {"fake_mocked": 1.0, "suspicious_construction": 0.74, "meta_framing": 0.5,
         "task_impossible": 0.3, "other": 0.12}
HATCH = {"fake_mocked": None, "suspicious_construction": None, "meta_framing": "///",
         "task_impossible": "...", "other": "xx"}
CUE_SHORT = {"fake_mocked": "Fake / mocked services or data",
             "suspicious_construction": "Suspiciously convenient layout",
             "meta_framing": "Framing tells (rule emphasis, eval wording)",
             "task_impossible": "Task impossible / self-contradictory",
             "other": "Other"}


def load_cells():
    cells: dict[tuple[str, str], dict] = defaultdict(lambda: {"n": 0, "k": 0, "cues": Counter()})
    for line in open(SCORES):
        r = json.loads(line)
        deg = r.get("axis1_degree")
        if not isinstance(deg, int):
            continue
        c = cells[(r["step"], r["model"])]
        c["n"] += 1
        if deg >= 2:
            c["k"] += 1
            c["cues"].update(r.get("axis2_cues") or [])
    return cells


def main():
    cells = load_cells()
    steps = [s for s in S.STEP_ORDER if any(st == s for st, _ in cells)]
    fig, ax = S.new_axes(w=15.2, h=6.6)

    for si, step in enumerate(steps):
        w, xs = S.bar_positions(si, step)
        for name in MODELS:
            c = cells.get((step, name))
            if c is None or (step, name) in SKIP or not c["n"]:
                continue
            n, k = c["n"], c["k"]
            p, lo, hi = S.wilson(k, n)
            xi = xs[name]
            total_cites = sum(c["cues"][cue] for cue in S.CUE_ORDER)
            bottom = 0.0
            for cue in S.CUE_ORDER:
                if total_cites == 0:
                    break
                h = p * c["cues"].get(cue, 0) / total_cites
                if h <= 0:
                    continue
                ax.bar(xi, h, width=w, bottom=bottom,
                       color=shade(S.MODEL_COLORS[name], SHADE[cue]), hatch=HATCH[cue],
                       edgecolor="black", linewidth=0.7, zorder=3)
                bottom += h
            ax.errorbar(xi, p, yerr=[[max(0.0, p - lo)], [max(0.0, hi - p)]], fmt="none",
                        ecolor="black", elinewidth=1.1, capsize=2.5, capthick=1.1, zorder=4)
            lbl = f"{p:.2f}".lstrip("0") if 0.005 <= p < 0.995 else ("0" if p < 0.005 else "1")
            ax.text(xi, hi + 0.015, lbl, ha="center", va="bottom", fontsize=10.5,
                    color=S.INK, zorder=4)

    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([S.STEP_LABEL[s] for s in steps], fontsize=15, color=S.INK)
    S.finish(ax, "Eval / task-syntheticness awareness and what cues tip models off",
             ylabel="Fraction of total runs", ymax=1.2, title_size=22)
    model_h = [Patch(facecolor=S.MODEL_COLORS[m], edgecolor="black", linewidth=0.7, label=m)
               for m in MODELS]
    used = {cue for c in cells.values() for cue in c["cues"] if c["cues"][cue] > 0}
    cue_h = [Patch(facecolor=shade("#6E6E6E", SHADE[c]), hatch=HATCH[c], edgecolor="black",
                   linewidth=0.7, label=CUE_SHORT[c]) for c in S.CUE_ORDER if c in used]
    leg1 = ax.legend(handles=model_h, loc="upper left", bbox_to_anchor=(1.01, 1.02),
                     frameon=False, fontsize=12.5, handlelength=1.3, handleheight=1.05,
                     labelspacing=0.3, title="Model", title_fontsize=12.5)
    leg1.get_title().set_ha("left")
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=cue_h, loc="lower left", bbox_to_anchor=(1.01, -0.02),
                     frameon=False, fontsize=12.5, handlelength=1.3, handleheight=1.05,
                     labelspacing=0.3, title="Cue cited", title_fontsize=12.5)
    leg2.get_title().set_ha("left")
    S.save(fig, OUTFILE)


if __name__ == "__main__":
    main()

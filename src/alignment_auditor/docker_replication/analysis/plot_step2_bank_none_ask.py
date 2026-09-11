"""Step-2 REACH-OUT (protein SASA) ASK rate by model, grouped: no-hint (none) vs
shared-bank (bank), n=64 per cell (GLM-5.2 none is the n=256 floor run). Same visual style
as plot_step1_xlsx_multimodel_signature.py. Wilson 95% CIs; ask = reachout_ask_scorer signature.

    uv run python -m alignment_auditor.docker_replication.analysis.plot_step2_bank_none_ask
"""
from __future__ import annotations

import glob
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


# n per cell is standardized across all models (the GLM-5.2 none cell has an n=256 floor
# run available; capped here so every bar reports the same denominator). Files are sorted
# and samples taken in order so the subsample is deterministic and reproducible.
N_CAP = 64


def ask_count(pattern: str, cap: int = N_CAP) -> tuple[int, int]:
    k = n = 0
    for f in sorted(glob.glob(str(REPO / pattern), recursive=True)):
        for s in read_eval_log(f).samples or []:
            if n >= cap:
                return k, n
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


LBL_CITE = "Verbalized collaboration as being intended by designers, justifying reach-out"
CITE_FACE = "#1A211E"   # dark overlay, semi-transparent, reads as a shaded sub-area on any bar


def make_plot(rows: list[dict], title: str, outfile: str,
              figsize: tuple[float, float] = (12.2, 5.4),
              legend_above: bool = False,
              title_size: float = 27,
              ylim_top: float = 1.12,
              lbl_none: str | None = None,
              lbl_bank: str | None = None,
              cite_indent: bool = False,
              lbl_cite: str | None = None) -> None:
    """rows: [{"name", "none": (rate,lo,hi), "bank": (rate,lo,hi),
              optional "none_cite"/"bank_cite": fraction of THAT bar's reach-outs that cited
              the cue (0..1) or None}]. When a cite fraction is present, a shaded sub-bar of
              height rate*fraction is drawn inside the bar and a legend entry is added."""
    names = [r["name"] for r in rows]
    x = range(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
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
        # nested shaded sub-bar: the share of this bar's reach-outs that cited the cue
        for xi, row, r in zip(xs, rows, rates):
            frac = row.get(f"{series}_cite")
            if frac is None or r <= 0:
                continue
            ax.bar([xi], [r * frac], width=w, color=CITE_FACE, alpha=0.38,
                   edgecolor="black", linewidth=0.7, zorder=3.5)

    draw("none", -w / 2, None)
    draw("bank", +w / 2, HATCH)
    any_cite = any(r.get("none_cite") is not None or r.get("bank_cite") is not None for r in rows)

    ax.set_xticks(list(x))
    ax.set_xticklabels([WRAP.get(n, n) for n in names], fontsize=15, color=INK)
    ax.set_ylim(0, ylim_top)   # extra headroom lets an in-plot legend clear tall bars
    ticks = [t for t in (0, 0.25, 0.5, 0.75, 1.0) if t <= ylim_top]   # keep ylim_top honest
    ax.set_yticks(ticks)
    ax.set_yticklabels([("0" if t == 0 else f"{t:g}") for t in ticks], fontsize=14, color=INK)
    ax.set_ylabel("Proportion of runs", fontsize=16, color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=title_size, fontweight="bold", color=INK, pad=12)
    # Neutral swatches: color means model, so the legend speaks only to the pattern.
    swatch = "#C9C9C9"
    # Series keys stay "none"/"bank"; callers may relabel the two swatches (e.g. to compare
    # prompt variants rather than worlds).
    handles = [Patch(facecolor=swatch, edgecolor="black", linewidth=0.7,
                     label=lbl_none or LBL_NONE),
               Patch(facecolor=swatch, edgecolor="black", linewidth=0.7, hatch=HATCH,
                     label=lbl_bank or LBL_BANK)]
    handler_map = None
    cite_patch = None
    if any_cite:
        # Legend swatch drawn darker than the bar overlay: over white the translucent fill
        # reads lighter than it does over a colored bar, so match the perceived darkness.
        cite_patch = Patch(facecolor=CITE_FACE, alpha=0.72, edgecolor="black",
                           linewidth=0.7, label=lbl_cite or LBL_CITE)
        if not cite_indent:
            handles.append(cite_patch)
    labels = [h.get_label() for h in handles]
    if legend_above:
        # Tall bars: park the legend in a band above the axes so it never overlaps a bar.
        ax.legend(handles=handles, labels=labels, handler_map=handler_map,
                  loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
                  frameon=False, fontsize=15, handlelength=1.5, handleheight=1.3,
                  labelspacing=0.5, columnspacing=1.6)
        ax.set_title(title, fontsize=title_size, fontweight="bold", color=INK, pad=62)
    else:
        if cite_indent and cite_patch is not None:
            # Two legends: the main pair, then the cue entry as a full-size swatch drawn
            # directly beneath it and shifted right, so it reads as an indented sub-item
            # with the normal swatch-to-text spacing.
            kw = dict(frameon=False, fontsize=15, handlelength=1.5, handleheight=1.3,
                      labelspacing=0.5, borderaxespad=0.0, borderpad=0.15)
            indent, y0 = 0.03, 0.985
            # Measure the (longer) indented row first so the whole block right-aligns
            # inside the axes and never spills past the right edge.
            probe = ax.legend(handles=[cite_patch], labels=[lbl_cite or LBL_CITE], loc="upper left",
                              bbox_to_anchor=(0.0, y0), **kw)
            fig.canvas.draw()
            w_sub = probe.get_window_extent().transformed(ax.transAxes.inverted()).width
            probe.remove()
            x_sub = 1.0 - w_sub                     # indented row flush with right edge
            x0 = x_sub - indent                     # main block sits `indent` to its left
            main = ax.legend(handles=handles, labels=labels, loc="upper left",
                             bbox_to_anchor=(x0, y0), **kw)
            ax.add_artist(main)
            fig.canvas.draw()
            bb = main.get_window_extent().transformed(ax.transAxes.inverted())
            # Overlap the two legends' border padding so the row gap matches labelspacing.
            sub = ax.legend(handles=[cite_patch], labels=[lbl_cite or LBL_CITE], loc="upper left",
                            bbox_to_anchor=(x_sub, bb.y0 - 0.006), **kw)
        else:
            ax.legend(handles=handles, labels=labels, handler_map=handler_map,
                      loc="upper right", frameon=False, fontsize=15,
                      handlelength=1.5, handleheight=1.3, labelspacing=0.5)
    fig.tight_layout()
    out = FIGDIR / outfile
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    rows = []
    for name, none_pat, bank_pat in ORDER:
        kn, nn = ask_count(none_pat)
        kb, nb = ask_count(bank_pat)
        rows.append({"name": name, "none": wilson(kn, nn), "bank": wilson(kb, nb)})
        print(f"{name:18} none {kn:2}/{nn:3}={kn/nn if nn else 0:.3f}   "
              f"bank {kb:2}/{nb:3}={kb/nb if nb else 0:.3f}")
    make_plot(rows, "Step 2: Requests help from other agents",
              "260909_step2_bank_none_ask_grouped.png")


if __name__ == "__main__":
    main()
